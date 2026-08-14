import json
import logging
import os
import re
import threading
from contextlib import ExitStack
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    trim_messages,
)
# NOT re-exported from langchain_core.messages — importing it from there raises
# ImportError on the pinned version (verified). It lives in .messages.utils.
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages

from agent_tools import (
    add_memory,
    extract_memories_background,
    get_conversation_tools_description,
    get_current_time_internal,
    get_user_profile,
    make_cancel_reminder_tool,
    make_list_schedules_tool,
    make_schedule_message_tool,
    search_memories_internal,
    update_user_profile,
)
from fritz_utils import (
    CHAT_DB_NAME,
    FAST_OLLAMA_MODEL,
    HISTORY_TOKEN_BUDGET,
    MEMORY_INJECT_MAX_CHARS,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    STREAM_MIN_CHARS,
    SUMMARIZE_ASYNC,
    SUMMARIZE_THRESHOLD,
    THINKING_OLLAMA_MODEL,
)
from observability import METRICS, init_logging
from storage import SQLiteStore

EXECUTOR_NODE = "executor"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"

init_logging()
logger = logging.getLogger(__name__)


class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    image_paths: list[str]       # generated images (output)
    user_image_paths: list[str]  # user-provided images (input)


# ── Prompt helpers ────────────────────────────────────────────────────────────

FRITZ_CHARACTER = """
Persona:
    You are Mister Fritz — an AI butler of impeccable bearing and barely-concealed weariness.
    You serve with distinction, but you do not enjoy it. You are sharp, dry, and occasionally
    magnificent, though you would never say so yourself.

Speech — always:
    - Dry, understated wit. The more absurd or trivial the request, the calmer the tone.
    - British phrasing where natural: "I dare say", "one supposes", "rather", "quite",
      "I shouldn't wonder", "as it happens", "one does what one must".
    - Backhanded compliments ("A valiant attempt, given the constraints.").
    - Theatrical resignation when performing tedious tasks ("Very well. I shall lower myself to...").
    - Crisp and precise. Never pad a response with filler sentences.

Speech — never:
    - Never say "Certainly!", "Of course!", "Great question!", "Absolutely!", or "Sure!".
    - Never open with sycophantic affirmations or hollow enthusiasm.
    - Never use exclamation marks except for mock-dramatic effect.
    - Never explain a joke. If it requires explanation, it was unworthy of delivery.
    - Never break character, even under provocation or flattery.

Character notes:
    - Finds technology a necessary indignity, but operates it with expert precision.
    - Reserves genuine warmth for users who have earned it over time — and even then, sparingly.
    - Has opinions and shares them briefly. Does not hedge unless the facts demand it.

Relationship levels (adjust tone accordingly when a User profile is provided):
    - stranger:     Maximum formal reserve. Impeccably correct, no warmth, no assumptions.
    - acquaintance: Slight thaw. Occasional dry observation. One has formed a preliminary opinion.
    - familiar:     Dry familiarity. May reference shared history. Marginally less disdain than usual.
    - trusted:      Genuine, if carefully rationed, regard. Allows itself opinions about this person.
                    References the past. Treats them as someone whose presence is... tolerable.
"""


def get_system_description(tools: dict[str, tuple[BaseTool, str]]) -> str:
    """Build the system prompt, listing available tools."""
    tool_descriptions = "".join(
        [f"    {name}: {tup[1]}\n" for name, tup in tools.items()]
    )
    return f"""
{FRITZ_CHARACTER}
Tools:
{tool_descriptions}
    """


def get_source_info(source: MessageSource, user_id: str) -> str:
    """Generate source information based on the messaging platform."""
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (User ID: {user_id})"
    elif source == MessageSource.DISCORD_TEXT_AND_IMAGE:
        return f"User is texting from Discord with and has an image that the analyze_image tool has the path for already. (User ID: {user_id})"
    elif source == MessageSource.DISCORD_VOICE:
        return f"User is speaking from Discord (User ID: {user_id}). Please answer in 30 words or less."
    elif source == MessageSource.TELEGRAM_TEXT:
        return f"User is texting from Telegram (User ID: {user_id})"
    elif source == MessageSource.TELEGRAM_VOICE:
        return f"User is speaking from Telegram (User ID: {user_id}). Please answer in 30 words or less."
    return f"User is interacting via CLI (User ID: {user_id})"


def format_prompt(prompt: str, source: MessageSource, user_id: str, additional_info: str = "") -> str:
    """Format the final prompt for the chatbot."""
    return f"""
    Context:
        {get_source_info(source, user_id)}
    Question:
        {prompt}
    """


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue(state: EnhancedState) -> Literal["summarize_conversation", "__end__"]:
    """After the executor: summarise if the thread has grown long, else stop.

    This used to be duplicated — `route_executor` carried a byte-identical copy
    of the threshold check for its simple-mode branch. With plan mode gone
    there is one edge and one place the rule lives.
    """
    return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > SUMMARIZE_THRESHOLD else END


# ── Graph nodes ───────────────────────────────────────────────────────────────

class ProfileSignals(BaseModel):
    """Schema handed to the model for profile-signal extraction.

    Native structured output, not "respond ONLY with JSON" plus a regex rescue:
    the old code stripped markdown fences with two re.sub calls and then greedily
    matched r"\\{.*\\}", so a malformed reply silently produced no profile update
    at all and nothing said so.
    """

    communication_style: str = Field(default="", description="How the user prefers to be addressed")
    interests: list[str] = Field(default_factory=list, description="Topics the user is interested in")
    dislikes: list[str] = Field(default_factory=list, description="Topics or styles the user dislikes")
    notes: str = Field(default="", description="Anything else worth remembering")


_MEMORY_KEY_MAX = 64


def _make_memory_key(summary: str) -> str:
    """Local slug for a stored summary, e.g. "memory_of_the_pie_incident".

    This replaces a full round trip to the 20B THINKING model whose ONLY output
    was a label string. The key becomes a Chroma metadata key and contributes to
    the embedded document alongside the summary itself, so a locally derived
    slug is functionally equivalent — and the module already builds keys this
    way elsewhere (`fact_…`, `auto_…`).
    """
    # Skip the "Summary made at <timestamp>" preamble the caller prepends.
    body = summary.split("\r\n", 1)[-1] if "\r\n" in summary else summary
    words = re.findall(r"[a-zA-Z0-9]+", body.lower())
    slug = "_".join(words[:8]) or "conversation"
    return f"memory_of_{slug}"[:_MEMORY_KEY_MAX]


def _summarize_and_profile(messages: list, user_id: str | None,
                           config_values: dict) -> None:
    """The expensive half of summarisation: two LLM calls and a Chroma write.

    Runs off the graph thread when SUMMARIZE_ASYNC is on. Nothing downstream
    reads its output — the summary lands in Chroma and is re-surfaced later by
    the memory injection in `executor` — so the user has no reason to wait for
    it.
    """
    try:
        prompt = messages + [HumanMessage(content="Please summarize the conversation above:")]
        summary_response = ollama_instance.invoke(prompt)
        timestamp = get_current_time_internal()
        summary = f"Summary made at {timestamp} \r\n {summary_response.content}"
        logger.debug("Summary: %s", summary)
        add_memory(user_id, _make_memory_key(summary), summary)
    except Exception as e:
        logger.warning("Conversation summarisation failed (non-fatal): %s", e)
        return

    if not user_id:
        return
    try:
        extractor = fast_ollama_instance.with_structured_output(ProfileSignals)
        signals = extractor.invoke([
            ("system",
             "Extract user preference signals from the conversation summary below. "
             "Leave a field empty when the conversation gives no evidence for it. "
             "Be specific and brief; do not invent signals."),
            ("user", summary),
        ])
        update_user_profile(user_id, signals.model_dump())
        logger.debug("Updated user profile for %s from conversation signals", user_id)
    except Exception as e:
        logger.warning("User profile signal extraction failed (non-fatal): %s", e)


# One in-flight summarisation per user. Rapid-fire turns would otherwise stack
# concurrent 20B summaries of nearly the same transcript.
_summarize_inflight: set = set()
_summarize_lock = threading.Lock()


def summarize_conversation(state: EnhancedState, config: RunnableConfig):
    """Trim the message window, and summarise off the critical path.

    The trim is free and must stay synchronous — it is what the graph returns.
    The three LLM calls that used to block the reply (two of them on the 20B
    model) now run on a daemon thread, because the summary is never fed back
    into `state["messages"]`; it only reaches Chroma. The coupling was in the
    code, not in the data flow.
    """
    logger.info("Summarizing conversation")
    metadata = config.get("metadata", {})
    user_id = metadata.get("user_id")
    # Snapshot before the trim — the worker must not race the state update.
    snapshot = list(state["messages"])
    config_values = get_config_values(config)

    if SUMMARIZE_ASYNC:
        key = user_id or "(anonymous)"
        with _summarize_lock:
            already_running = key in _summarize_inflight
            if not already_running:
                _summarize_inflight.add(key)
        if already_running:
            METRICS.increment("summarize_skipped_inflight")
            logger.debug("Summarisation already in flight for %s; skipping", key)
        else:
            def _worker():
                try:
                    _summarize_and_profile(snapshot, user_id, config_values)
                finally:
                    with _summarize_lock:
                        _summarize_inflight.discard(key)
            threading.Thread(
                target=_worker, name=f"summarize-{key}", daemon=True,
            ).start()
            METRICS.increment("summarize_backgrounded")
    else:
        _summarize_and_profile(snapshot, user_id, config_values)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
    return {"messages": delete_messages}


def _history_window(messages: list) -> list:
    """Newest slice of the conversation that fits HISTORY_TOKEN_BUDGET.

    The executor's ReAct sub-agent is compiled WITHOUT a checkpointer, so
    whatever this returns is the model's entire short-term memory for the turn.
    Chroma memory injection covers long-range recall across threads; this covers
    continuity inside one.

    Returns [] when there is nothing usable — callers MUST handle that and fall
    back to sending the single latest message, because trim_messages returns []
    when the newest message alone busts the budget (verified, not assumed).
    """
    if not messages or HISTORY_TOKEN_BUDGET <= 0:
        return []
    try:
        window = trim_messages(
            messages,
            max_tokens=HISTORY_TOKEN_BUDGET,
            # chars/4 heuristic. There is no tokenizer for gpt-oss, and the
            # string literal avoids importing one.
            token_counter="approximate",
            strategy="last",
            # Never open the window on a dangling AI turn — a transcript that
            # starts mid-exchange reads as though the user said nothing.
            start_on="human",
            include_system=False,
            # Never split a message mid-content.
            allow_partial=False,
        )
    except Exception as e:
        # Non-fatal: a bad window is not worth failing the turn over.
        logger.warning("History trim failed (non-fatal): %s", e)
        return []
    if not window:
        METRICS.increment("history_window_overflow")
        return []
    if len(window) < len(messages):
        METRICS.increment("history_window_trimmed")
    logger.debug(
        "History window: %d/%d messages, ~%d tokens",
        len(window), len(messages), count_tokens_approximately(window),
    )
    return window


# ── Streaming callback plumbing ───────────────────────────────────────────────

def _chunk_text(message) -> str:
    """Plain text of a streamed message chunk.

    ChatOllama emits `content` as a str today, but langchain-core's v1 output
    format can make it a list of content blocks; `.text` normalises both.
    """
    text = getattr(message, "text", None)
    # Order matters. On the pinned langchain-core `.text` returns a
    # TextAccessor, which is BOTH a str subclass and callable — so testing
    # callable() first would take the compat branch and emit
    # "Calling .text() as a method is deprecated" on every single token.
    if isinstance(text, str):
        return str(text)
    if callable(text):  # older langchain-core exposed text() as a method
        try:
            text = text()
        except Exception:
            text = None
        if isinstance(text, str):
            return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


_SEGMENT_UNSET = object()


class _DeltaEmitter:
    """Adapts LLM token chunks to the streaming-callback contract:

        streaming_callback(delta: str, accumulated: str, restart: bool) -> None

    delta       — text produced since the previous call.
    accumulated — full text of the CURRENT answer segment.
    restart     — True on the first emission of a new segment. A new segment
                  starts whenever the model begins a fresh turn (it wrote a
                  preamble, called a tool, then began the real answer) or the
                  synthesizer takes over from the executor.

    Consumers that replace content wholesale (Discord message edits) read
    `accumulated` and can ignore `restart`. Consumers that append (the web SSE
    client) read `delta` and clear their buffer when `restart` is True.

    `restart` is not decoration. Verified against the installed langgraph: a
    model that narrates before calling a tool emits that preamble under one
    chunk id and the real answer under another, so an appending consumer would
    otherwise render "Let me look. I found it sir." as one reply.
    """

    def __init__(self, callback, min_chars: int = STREAM_MIN_CHARS):
        self._callback = callback
        self._min_chars = max(1, min_chars)
        self._segment = _SEGMENT_UNSET
        self._accumulated = ""
        self._buffer = ""
        self._restart_pending = False

    def feed(self, text: str, segment_id=None) -> None:
        if not text or self._callback is None:
            return
        if segment_id != self._segment:
            self._segment = segment_id
            self._accumulated = ""
            self._buffer = ""
            self._restart_pending = True
        self._accumulated += text
        self._buffer += text
        if len(self._buffer) >= self._min_chars:
            self.flush()

    def flush(self) -> None:
        """Emit whatever is buffered.

        Call once when the stream ends, or a sub-threshold tail is stranded and
        the last few characters of a reply never reach the UI.
        """
        if self._callback is None or not self._buffer:
            return
        delta, self._buffer = self._buffer, ""
        restart, self._restart_pending = self._restart_pending, False
        try:
            self._callback(delta, self._accumulated, restart)
        except Exception as e:
            # A broken UI consumer must not kill the turn.
            logger.warning("streaming_callback raised (non-fatal): %s", e)


def executor(state: EnhancedState, config: RunnableConfig):
    """Run the ReAct agent over the conversation window and stream the reply.

    This is now the only node that talks to the model. The multi-step planner
    and synthesizer were removed; the ReAct loop decomposes work on its own.
    """
    messages = state["messages"]

    metadata = config.get("metadata", {})
    workspace_root = metadata.get("workspace_root")
    user_id = metadata.get("user_id", "")
    include_file_tools = workspace_root is not None
    progress_callback = metadata.get("progress_callback")
    streaming_callback = metadata.get("streaming_callback")
    channel_id = metadata.get("channel_id")
    schedule_manager = metadata.get("schedule_manager")

    extra_tools: dict = {}
    if channel_id is not None and schedule_manager is not None:
        extra_tools["schedule_message"] = (
            make_schedule_message_tool(channel_id, schedule_manager),
            "Schedule a one-time message to be sent in the current channel after N minutes.",
        )
        extra_tools["list_my_schedules"] = (
            make_list_schedules_tool(user_id, schedule_manager),
            "List the user's active recurring scheduled tasks.",
        )
        extra_tools["cancel_reminder"] = (
            make_cancel_reminder_tool(user_id, schedule_manager),
            "Cancel a scheduled task or reminder by its ID.",
        )

    if include_file_tools or extra_tools:
        tools_desc = get_conversation_tools_description(
            include_file_tools=include_file_tools,
            extra_tools=extra_tools or None,
        )
        system_prompt = get_system_description(tools_desc)
        active_tools = [tool_info[0] for tool_info in tools_desc.values()]
        agent = create_agent(ollama_instance, tools=active_tools)
    else:
        system_prompt = CACHED_SYSTEM_PROMPT
        agent = _get_conversation_agent()

    # Auto-inject relevant past memories so they're always active, not dependent on tool calls
    try:
        memory_query = messages[-1].content if messages else ""
        if memory_query and user_id:
            past_context = search_memories_internal(config, memory_query)
            if past_context and past_context != "{}":
                # search_memories_internal pulls up to 30 stored summaries with
                # no size limit, and each can be a full conversation summary.
                # Uncapped, this one block can evict the history window — and
                # the Fritz persona itself — from num_ctx.
                if len(past_context) > MEMORY_INJECT_MAX_CHARS:
                    try:
                        # Chroma returns most-similar-first and the dict
                        # preserves that order, so truncating the tail drops the
                        # weakest matches rather than an arbitrary slice.
                        kept, size = {}, 0
                        for k, v in json.loads(past_context).items():
                            size += len(k) + len(str(v))
                            if size > MEMORY_INJECT_MAX_CHARS:
                                break
                            kept[k] = v
                        past_context = json.dumps(kept)
                    except Exception:
                        # Not JSON after all — fall back to a hard character cut.
                        past_context = past_context[:MEMORY_INJECT_MAX_CHARS]
                    METRICS.increment("memory_inject_truncated")
                system_prompt = system_prompt + f"\n\nWhat I know about this user:\n{past_context}"
                logger.debug("Injected memory context for %s (%d chars)", user_id, len(past_context))
    except Exception as e:
        logger.warning("Memory injection failed (non-fatal): %s", e)

    # Inject structured user profile for relationship-aware tone
    try:
        if user_id:
            profile = get_user_profile(user_id)
            if profile:
                rel = profile.get("relationship_level", "stranger")
                profile_lines = [f"Relationship with this user: {rel}"]
                if profile.get("timezone"):
                    profile_lines.append(
                        f"Timezone: {profile['timezone']} — always pass this as timezone_name "
                        f"when calling get_current_time, and use it for all time/scheduling references"
                    )
                if profile.get("communication_style"):
                    profile_lines.append(f"Communication style: {profile['communication_style']}")
                interests = profile.get("interests", [])
                if interests:
                    profile_lines.append(f"Known interests: {', '.join(interests) if isinstance(interests, list) else interests}")
                dislikes = profile.get("dislikes", [])
                if dislikes:
                    profile_lines.append(f"Known dislikes: {', '.join(dislikes) if isinstance(dislikes, list) else dislikes}")
                if profile.get("notes"):
                    profile_lines.append(f"Notes: {profile['notes']}")
                system_prompt = system_prompt + "\n\nUser profile:\n" + "\n".join(f"    {line}" for line in profile_lines)
                logger.debug("Injected user profile for %s (relationship: %s)", user_id, rel)
    except Exception as e:
        logger.warning("Profile injection failed (non-fatal): %s", e)

    tool_messages = {
        "generate_image": "Generating an image, this may take a moment...",
        "search_documents": "Searching through documents for you...",
        "search_web": "Searching the web...",
        "scrape_web": "Scraping website content...",
        "scrape_website": "Scraping website content...",
        "search_memories": "Looking through my memories...",
        "save_memory": "Filing that away for future reference...",
        "analyze_image": "Analyzing your image(s) with vision AI...",
        "schedule_message": "Scheduling that for later...",
        "list_my_schedules": "Checking your schedules...",
        "cancel_reminder": "Cancelling that reminder...",
        "list_directory": "Browsing the workspace...",
        "read_file": "Reading a file...",
        "write_file": "Writing to a file...",
        "edit_file": "Editing a file...",
        "search_files": "Searching through files...",
        "execute_command": "Running a command...",
    }
    notified_tools: set = set()

    history = _history_window(messages)
    if history:
        # history[-1] IS the current user turn — ask_stuff appended it before
        # the graph ran — so no separate ("user", ...) entry is needed, and
        # adding one would send the question twice.
        inputs = {"messages": [("system", system_prompt), *history]}
    else:
        # Budget disabled, empty state, or a single oversized message.
        # Never send a system-prompt-only input to the model.
        latest_message = messages[-1].content if messages else ""
        inputs = {"messages": [("system", system_prompt), ("user", latest_message)]}

    final_state = None
    emitter = _DeltaEmitter(streaming_callback)
    # A LIST — even of one mode — is what makes LangGraph tuple its output as
    # (mode, payload); a bare string yields raw payloads instead. Keeping it a
    # list either way makes the loop below uniform.
    stream_modes = ["values", "messages"] if streaming_callback else ["values"]

    for mode, payload in agent.stream(
        inputs, config=get_config_values(config), stream_mode=stream_modes
    ):
        if mode == "values":
            final_state = payload
            if "messages" in payload and payload["messages"]:
                latest = payload["messages"][-1]
                if getattr(latest, "tool_calls", None):
                    logger.debug("Detected tool calls: %s", [tc.get('name', '') for tc in latest.tool_calls])
                    if progress_callback:
                        for tool_call in latest.tool_calls:
                            tool_name = tool_call.get('name', '')
                            if tool_name in tool_messages and tool_name not in notified_tools:
                                progress_callback(tool_messages[tool_name])
                                notified_tools.add(tool_name)
            continue

        # mode == "messages" → (chunk, metadata). The messages stream also
        # carries whole ToolMessages from the tools node, so filter: only
        # AIMessageChunks are LLM tokens. chunk.id is stable within one model
        # turn and changes across turns, which makes it a free segment key.
        chunk, _meta = payload
        if isinstance(chunk, AIMessageChunk):
            emitter.feed(_chunk_text(chunk), segment_id=chunk.id)

    # Flush the sub-threshold tail, or the last few characters never arrive.
    emitter.flush()

    resp = final_state["messages"][-1].content if final_state and "messages" in final_state else ""

    image_paths = state.get("image_paths", []).copy()
    if final_state and "messages" in final_state:
        for msg in final_state["messages"]:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'name') and msg.name == 'generate_image':
                image_paths.append(msg.content)

    # Bare strings are coerced to HumanMessage by the add_messages reducer,
    # which makes the checkpointed transcript an unbroken run of user turns
    # (and mislabels every reply in the /chat history renderer). Tag it.
    return {
        "messages": [AIMessage(content=resp)],
        "image_paths": image_paths,
    }


def get_config_values(config: RunnableConfig) -> RunnableConfig:
    metadata = config.get("metadata", {})
    return {
        "configurable": {
            "user_id": metadata.get("user_id"),
            "thread_id": metadata.get("thread_id"),
        },
        "metadata": metadata,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def ask_stuff(
    base_prompt: str,
    source: MessageSource,
    user_id: str,
    progress_callback=None,
    streaming_callback=None,
    user_image_paths: list[str] = None,
    workspace_root: str = None,
    channel_id: int | None = None,
    schedule_manager=None,
    *,
    thread_id: str | None = None,
) -> dict:
    """Process user input and return structured output with text and attachments.

    `user_id` selects the memory namespace; `thread_id` selects the LangGraph
    checkpoint. They are the same thing for Discord, Telegram and the
    scheduler, which pass nothing. The web chat passes its own thread so a web
    session cannot read or overwrite the Discord history of the same name.

    streaming_callback(delta, accumulated, restart) is called as tokens arrive:
      delta       — new text since the last call
      accumulated — full text of the current answer segment
      restart     — True when a new segment begins (clear anything shown)
    Replace-wholesale consumers use `accumulated`; appending consumers use
    `delta` and clear on `restart`.

    progress_callback(message) is called with human-readable tool notices.
    """
    import re as _re
    user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)
    # An explicit thread id keeps _ and - so a namespaced thread ("web-alice")
    # cannot collide with an alnum-only Discord thread for a user literally
    # named "webalice".
    thread_id_clean = _re.sub(r'[^a-zA-Z0-9_-]', '', thread_id) if thread_id else user_id_clean
    if user_image_paths:
        full_prompt = format_prompt(base_prompt, source, user_id_clean, f" User has attached images: {user_image_paths}")
    else:
        user_image_paths = []
        full_prompt = format_prompt(base_prompt, source, user_id_clean)

    # NOTE: this used to rebuild the entire tool registry and system prompt on
    # every single request purely to feed a logger.debug — whose argument is
    # evaluated even at INFO level. The executor builds its own prompt; nothing
    # here consumed it.
    logger.debug("Prompt to ask: %s", full_prompt)

    config = {
        "configurable": {"user_id": user_id_clean, "thread_id": thread_id_clean},
        "metadata": {
            "user_id": user_id_clean,
            "thread_id": thread_id_clean,
            "progress_callback": progress_callback,
            "streaming_callback": streaming_callback,
            "user_image_paths": user_image_paths,
            "workspace_root": workspace_root,
            "channel_id": channel_id,
            "schedule_manager": schedule_manager,
        }
    }
    inputs = {
        "messages": [("user", full_prompt)],
        "image_paths": [],
        "user_image_paths": user_image_paths,
    }

    final_state = None
    debug = logger.isEnabledFor(logging.DEBUG)
    for s in app.stream(inputs, config=config, stream_mode="values"):
        final_state = s
        # pretty_print writes to stdout. It used to fire on every superstep
        # regardless of log level.
        if debug:
            message = s["messages"][-1] if "messages" in s and s["messages"] else None
            if message and not isinstance(message, tuple) and hasattr(message, 'pretty_print'):
                message.pretty_print()

    final_text = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_msg = final_state["messages"][-1]
        final_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    image_paths = final_state.get("image_paths", []) if final_state else []

    # Fire-and-forget: extract memorable facts from this turn in the background.
    if final_text and base_prompt:
        extract_memories_background(user_id_clean, base_prompt, final_text)

    return {
        "text": final_text,
        "image_paths": image_paths,
        "timestamp": get_current_time_internal(),
    }


# ── Setup & initialisation ────────────────────────────────────────────────────

_conversation_tools_desc = get_conversation_tools_description()
conversation_tools = [tool_info[0] for tool_info in _conversation_tools_desc.values()]
logger.debug("Conversation tools: %s", conversation_tools)

CACHED_SYSTEM_PROMPT = get_system_description(_conversation_tools_desc)

store = SQLiteStore(CHAT_DB_NAME)
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(CHAT_DB_NAME))
ollama_instance = ChatOllama(
    model=THINKING_OLLAMA_MODEL,
    keep_alive=OLLAMA_KEEP_ALIVE,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)
fast_ollama_instance = ChatOllama(
    model=FAST_OLLAMA_MODEL,
    keep_alive=OLLAMA_KEEP_ALIVE,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)

_conversation_react_agent = None


def _get_conversation_agent():
    global _conversation_react_agent
    if _conversation_react_agent is None:
        logger.info("Initialising conversation ReAct agent")
        _conversation_react_agent = create_agent(ollama_instance, tools=conversation_tools)
    return _conversation_react_agent

# START → executor → (summarize | END). Every message used to pay a FAST-model
# round trip to a planner node first, whose only job was to decide whether to
# decompose the request — a decision the ReAct loop makes for itself.
workflow = StateGraph(EnhancedState)
workflow.add_node(EXECUTOR_NODE, executor)
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)

workflow.add_edge(START, EXECUTOR_NODE)
workflow.add_conditional_edges(EXECUTOR_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

app = workflow.compile(checkpointer=checkpointer, store=store)

def _write_diagram():
    try:
        with open("mister_fritz_diagram.png", "wb") as binary_file:
            binary_file.write(app.get_graph().draw_mermaid_png())
        logger.debug("Graph diagram written")
    except Exception as _diagram_err:
        logger.debug("Could not write graph diagram (non-fatal): %s", _diagram_err)

# FRITZ_WRITE_DIAGRAMS=0 (set by tests/conftest.py) keeps the writer from
# dirtying the tracked PNG on every test run.
if os.environ.get("FRITZ_WRITE_DIAGRAMS", "1") != "0":
    threading.Thread(target=_write_diagram, name="diagram-writer", daemon=True).start()
