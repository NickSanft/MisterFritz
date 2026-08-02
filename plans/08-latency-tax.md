# 8. Cut fixed per-message latency

[← back to index](README.md)

**Effort:** L (1-3 days)  
**Depends on:** nothing

## Goal
Today every single Discord/Telegram/web-chat message — including "hi" — pays a mandatory FAST-model round trip to a planning LLM before the executor ever starts, and roughly every 7th turn additionally blocks the user behind three more LLM calls (two of them on the 20B THINKING model) before `ask_stuff` returns. When this lands, a trivial message reaches the ReAct executor with zero preceding LLM calls (a pure-Python heuristic decides whether the LLM planner is worth invoking), conversation summarization + profile-signal extraction run on a daemon thread so the user never waits for them, the 20B call that existed only to invent a memory-key string like `memory_of_pie` is replaced by a local slug function, all three hand-rolled "respond ONLY with JSON" + regex-scrape sites use Ollama's native JSON-schema structured output (so a parse failure stops silently degrading planning to simple mode), and two pieces of per-request dead work (building a full system prompt only to `logger.debug` it; `pretty_print()`ing every superstep to stdout unconditionally) are gone.

## Definition of done

- [ ] `should_run_planner("hi", MessageSource.DISCORD_TEXT)` returns False and `route_start` sends that turn straight to `executor`; with `LOG_LEVEL=DEBUG` neither "Planner chose simple mode" nor "Planner created N-step plan" appears for a trivial message — only the new "Planner gate: skipped" line.
- [ ] `should_run_planner("research the tokio async runtime then write me a one-page report on it", MessageSource.DISCORD_TEXT)` returns True and the LLM planner still runs, still produces a >1-step plan, and plan mode still reaches `synthesizer`.
- [ ] `summarize_conversation` makes ZERO LLM calls on the graph thread: a unit test that patches `mister_fritz.ollama_instance` and `mister_fritz.fast_ollama_instance` and invokes the node asserts `invoke` was never called on either, and the return value contains only `RemoveMessage` objects.
- [ ] The 20B memory-key call at `mister_fritz.py:239` is deleted; `_make_memory_key("...")` returns a `memory_of_<slug>` string of <= 64 chars with no network access.
- [ ] `planner`, the profile-signal extraction, and `agent_tools._extract_and_store_memories` all pass a JSON schema to the model (`ChatOllama.with_structured_output(...)` / `ollama.chat(format=...)`); no `re.sub(r"^```...")`, no `re.search(r"\{.*\}")`, and no `re.search(r'\[.*?\]')` remains in either file.
- [ ] `mister_fritz.py` no longer imports `json` (both call sites removed) and `ruff check .` passes clean.
- [ ] `ask_stuff` no longer calls `get_conversation_tools_description()` per request, and `message.pretty_print()` only fires when `logger.isEnabledFor(logging.DEBUG)`.
- [ ] `PLANNER_MODE=always SUMMARIZE_ASYNC=false` restores the pre-change latency profile with no code revert.
- [ ] `pytest tests/ --cov=. --cov-fail-under=60` passes; `tests/test_mister_fritz.py` no longer contains the obsolete code-fence / surrounding-text / non-string-coercion tests, and contains new gate, router, memory-key and off-path-summarize tests.
- [ ] New knobs `PLANNER_MODE`, `PLANNER_MIN_CHARS`, `SUMMARIZE_ASYNC` are documented in `.env.example` and the change has a Phase-style entry under `### Performance` in `CHANGELOG.md`.

## Current state (verified against the working tree)
Verified by reading the files in this session; the audit's leads are confirmed with two small line-number corrections noted below.

(a) UNCONDITIONAL PLANNER. `mister_fritz.py:634-635` is exactly `workflow.add_edge(START, PLANNER_NODE)` / `workflow.add_edge(PLANNER_NODE, EXECUTOR_NODE)` — no conditional edge, so `planner()` (`:162-223`) runs on every turn. It builds a 2-message prompt (`:168-184`), calls `fast_ollama_instance.invoke(planner_prompt)` at `:187`, strips markdown fences with two `re.sub` calls (`:191-192`), greedily rescues a JSON object with `re.search(r"\{.*\}", raw, re.DOTALL)` at `:194`, `json.loads` at `:197`, and on ANY exception silently falls back to simple mode at `:201-204`. Plan mode only engages when `needs_planning and len(steps) > 1` (`:206`). Net effect: a parse failure is indistinguishable from "no plan needed", and the FAST-model round trip is paid unconditionally.

(b) SUMMARIZATION ON THE CRITICAL PATH. `should_continue` (`:136-138`) and `route_executor` (`:141-157`) both route to `SUMMARIZE_CONVERSATION_NODE` when `len(state["messages"]) > SUMMARIZE_THRESHOLD` (default 15, `fritz_utils.py:97`). `summarize_conversation` (`:226-272`) then blocks `ask_stuff` for three LLM calls: (1) `ollama_instance.invoke(messages)` at `:231` — full history through the 20B THINKING model; (2) a SECOND 20B call at `:239` (the prompt list is `:235-238`, not `:236-239` as the audit stated) whose sole output is a label string, per the system prompt at `:236` "...short sentence describing this memory starting with the word \"memory\". Example - memory_of_pie"; (3) `fast_ollama_instance.invoke(signal_prompt)` at `:259` for profile signals, again fence-stripped (`:260-262`) and regex-rescued (`:263`). Then `add_memory` at `:241` and the trim at `:271` (`RemoveMessage(id=m.id) for m in state["messages"][:-1]`). Because the trim leaves exactly one message, the threshold fires again ~7 turns later. Crucially: the generated summary is NEVER put back into `state["messages"]` — it only lands in Chroma and is re-surfaced later by the auto-injection at `:329-333`. So the trim and the summarization are already logically independent; only the code path couples them.

(c) MEMORY KEY IS JUST A LABEL. `agent_tools.add_memory` (`:87-91`) does `memory_dict = {memory_key: memory_to_store}` then `put((user_id,), str(uuid.uuid4()), memory_dict)`. In `ChromaStore.mset` (`storage.py:151-169`) the key becomes a metadata key and, since none of `page_content`/`text`/`content` are present, the embedded document text is `json.dumps(value)` — i.e. the key contributes to the embedding alongside the whole summary. A locally-derived slug is functionally equivalent. Existing convention for locally-built keys already exists: `save_memory` uses `f"fact_{fact[:50].replace(' ', '_').lower()}"` (`agent_tools.py:354`) and auto-extraction uses `f"auto_{...}"` (`:140`).

(d) THIRD JSON SITE. `agent_tools._extract_and_store_memories` (`:105-144`) calls raw `_ollama_client.chat(...)` at `:126-130` with `_EXTRACTION_PROMPT` (`:94-102`) saying "Return ONLY a JSON array", then `re.search(r'\[.*?\]', content, re.DOTALL)` at `:132`. That regex is non-greedy and would truncate at the first `]`, so a nested array silently loses facts — a latent bug this change removes for free.

(e) STRUCTURED OUTPUT IS AVAILABLE. Verified in the venv: `langchain_ollama/chat_models.py:658` declares `format: Literal["", "json"] | JsonSchemaValue | None`, and `with_structured_output(schema, *, method=...)` at `:1253-1257` defaults to `method="json_schema"`, which binds `format=<schema.model_json_schema()>` at `:1560-1562`. `ollama/_client.py:297` confirms `chat(..., format: Optional[Union[Literal['','json'], JsonSchemaValue]])`. Installed: langchain-ollama 1.0.1, ollama 0.6.1, pydantic 2.12.5.

(f) DEAD PER-REQUEST WORK. `ask_stuff:543-545` computes `include_file_tools`, then `system_prompt = get_system_description(get_conversation_tools_description(include_file_tools))`, then `logger.debug("Role description: %s", system_prompt)`. `system_prompt` and `include_file_tools` are used nowhere else in `ask_stuff` (verified by reading `:523-593`) — the whole tool registry is rebuilt per request purely to feed a debug log whose argument is evaluated even at INFO level. `:575-576` calls `message.pretty_print()` (stdout write) on every superstep unconditionally. `:642` repeats `logger.debug("Conversation tools description: %s", get_conversation_tools_description())` at module scope, duplicating `:600`.

(g) CONCURRENCY BASELINE. `ask_stuff` is called concurrently from at least three places: `bot_commands.py:395` (and other Discord handlers), `admin_panel.py:448` via `run_in_executor` and `admin_panel.py:522` on a plain daemon thread, and `scheduler.py:109-111` via `run_in_executor`. `SqliteSaver` guards every DB op with `self.lock = threading.Lock()` (`langgraph/checkpoint/sqlite/__init__.py:88`, and the docstring at `:63-64` explicitly says `check_same_thread=False` is safe because of it). `agent_tools.extract_memories_background` (`:147-154`) is the established fire-and-forget precedent — a daemon thread that writes Chroma while the main thread is free.

(h) PRE-EXISTING QUIRK FOUND WHILE VERIFYING. `executor` returns `{"messages": [resp]}` where `resp` is a bare `str` (`:452`). I confirmed empirically that `langgraph.graph.add_messages([...], ["text"])` coerces a bare string to a **HumanMessage**, not an AIMessage. So every Fritz reply is stored with the wrong role in the checkpoint, which affects both the summarization prompt and `admin_panel._load_chat_history` (`:344-368`). This is out of scope here — flagged for the history-window item.

## Change sites

### `fritz_utils.py:118-119 (insert immediately after MEMORY_EXTRACT_MIN_REPLY_CHARS)`

Add the three new tunables. PLANNER_MODE is a tri-state string rather than a bool so it doubles as the rollback lever (`always` = pre-change behaviour) and the kill switch (`off` = never plan). Unknown values fall back to `auto` so a typo can't disable planning silently.

# AFTER (new block, following the existing comment-then-constant style)

# Heuristic planner gate. Previously every message paid a FAST-model round trip
# asking "does this need a plan?". `auto` (default) runs that LLM call only when
# the raw user request is at least PLANNER_MIN_CHARS long AND matches a
# multi-step phrasing pattern. `always` restores the old unconditional
# behaviour (rollback lever); `off` disables multi-step planning entirely.
_PLANNER_MODES = ("auto", "always", "off")
PLANNER_MODE: str = os.environ.get("PLANNER_MODE", "auto").strip().lower()
if PLANNER_MODE not in _PLANNER_MODES:
    PLANNER_MODE = "auto"

# Minimum length (chars) of the raw user request before the LLM planner is even
# considered. Below this the request goes straight to the executor.
PLANNER_MIN_CHARS: int = int(os.environ.get("PLANNER_MIN_CHARS", "60"))

# Run conversation summarisation + profile-signal extraction on a daemon thread
# instead of blocking the reply. Set to false to restore synchronous ordering
# (useful for deterministic debugging).
SUMMARIZE_ASYNC: bool = os.environ.get(
    "SUMMARIZE_ASYNC", "true"
).strip().lower() in ("1", "true", "yes", "on")

### `mister_fritz.py:1-3`

Drop `import json` — after the planner and profile-signal rewrites, its only two uses (`:197`, `:265`) are gone and ruff's F401 gate in CI will fail otherwise. Keep `import re` (still used by the new `_make_memory_key`). Add pydantic.

# BEFORE
import json
import logging
import re
import threading

# AFTER
import logging
import re
import threading
...
from pydantic import BaseModel, Field

### `mister_fritz.py:29-37`

Import the new knobs from fritz_utils alongside the existing ones (alphabetical order is already the house style in this block).

from fritz_utils import (
    CHAT_DB_NAME,
    FAST_OLLAMA_MODEL,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    PLANNER_MIN_CHARS,
    PLANNER_MODE,
    SUMMARIZE_ASYNC,
    SUMMARIZE_THRESHOLD,
    THINKING_OLLAMA_MODEL,
)

### `mister_fritz.py:50-58`

Add one boolean to EnhancedState carrying the gate decision. Computing the predicate in `ask_stuff` (which already has the RAW `base_prompt` and the `MessageSource`) instead of in the router avoids having to parse the user's text back out of the `format_prompt`-wrapped message and avoids putting a MessageSource enum into `config["metadata"]`. Old checkpoints that lack the key just read as falsy via `.get()`.

class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    image_paths: list[str]       # generated images (output)
    user_image_paths: list[str]  # user-provided images (input)
    original_request: str        # raw user message, captured by planner
    plan: list[str]              # ordered steps; empty = simple mode
    current_step: int            # index into plan
    step_results: list[str]      # accumulated per-step results
    needs_planner: bool          # heuristic gate verdict, set by ask_stuff

### `mister_fritz.py:134-158 (add above should_continue)`

New pure-Python gate predicate + the START router. The predicate is deliberately biased toward False: a missed plan degrades to simple ReAct mode (which can still call several tools in one loop), whereas a spurious True costs exactly the round trip we are trying to delete — i.e. the status quo.

_VOICE_SOURCES = frozenset({MessageSource.DISCORD_VOICE, MessageSource.TELEGRAM_VOICE})

# Phrases that reliably indicate sequential dependencies. Tuned to under-trigger.
_MULTI_STEP_PATTERN = re.compile(
    r"\bthen\b"
    r"|\bafter (?:that|which|you)\b"
    r"|\bstep[- ]by[- ]step\b"
    r"|\bstep 1\b"
    r"|\bfirst\b[\s\S]{0,120}?\b(?:second|next|finally|lastly)\b"
    r"|\bmake (?:me )?a plan\b"
    r"|\bplan (?:this|it) out\b"
    r"|\bresearch\b[\s\S]{0,120}?\b(?:write|report|summar|draft)"
    r"|\bfor each\b",
    re.IGNORECASE,
)


def should_run_planner(request: str, source: MessageSource | None = None) -> bool:
    """Decide locally whether the LLM planner is worth a round trip.

    Operates on the RAW user text (not the format_prompt-wrapped message, which
    always carries ~90 chars of Context/Question boilerplate).
    """
    if PLANNER_MODE == "off":
        return False
    if PLANNER_MODE == "always":
        return True
    if source in _VOICE_SOURCES:
        return False  # voice replies are capped at 30 words; never worth planning
    text = (request or "").strip()
    if len(text) < PLANNER_MIN_CHARS:
        return False
    return bool(_MULTI_STEP_PATTERN.search(text))


def route_start(state: EnhancedState) -> Literal["planner", "executor"]:
    """Skip the planner node entirely when the heuristic gate said no."""
    if state.get("needs_planner"):
        METRICS.increment("planner_invoked")
        return PLANNER_NODE
    METRICS.increment("planner_skipped")
    logger.info("Planner gate: skipped (heuristic)")
    return EXECUTOR_NODE

### `mister_fritz.py:162-223`

Replace the fence-strip + greedy-regex + json.loads block (:186-204) with a Pydantic schema driven through Ollama's native structured output. `steps` is declared required (no default) so it lands in the JSON schema's `required` array — some Ollama/llama.cpp builds handle optional properties inconsistently. The try/except stays: a ValidationError or a transport error still means simple mode, but now it's a genuine failure rather than an unparseable-but-valid response. Build the structured runnable INSIDE the function from the module global so `patch.object(mister_fritz, "fast_ollama_instance")` still works in tests; `.with_structured_output()` is a local `.bind()` + parser construction, no network.

# module scope, near FRITZ_CHARACTER
class PlanDecision(BaseModel):
    """Planner verdict, enforced by Ollama's JSON-schema output mode."""
    needs_planning: bool = Field(
        description="True only when the request requires sequential actions."
    )
    steps: list[str] = Field(
        description="Ordered steps, 5 maximum. Empty list when needs_planning is false."
    )


_PLANNER_SYSTEM = (
    "You are a planning assistant. Analyze the user's request and decide "
    "whether it can be answered in a single step or requires multiple steps.\n"
    "Use multi-step only when the request clearly requires sequential actions "
    "(e.g., research then write a report, fetch data then analyse it). "
    "Simple questions, chat, or single-tool lookups do NOT need planning. "
    "Keep plans to 5 steps maximum; return an empty steps list otherwise."
)

# inside planner(), replacing :168-204
    planner_prompt = [("system", _PLANNER_SYSTEM), ("user", latest_message)]
    try:
        decision = fast_ollama_instance.with_structured_output(PlanDecision).invoke(
            planner_prompt
        )
        needs_planning = bool(decision.needs_planning)
        steps = [str(s) for s in (decision.steps or [])[:5]] if needs_planning else []
    except Exception as e:
        logger.warning("Planner structured output failed (%s); falling back to simple mode", e)
        needs_planning = False
        steps = []
# :206-223 (the `needs_planning and len(steps) > 1` gate and both return dicts) unchanged

### `mister_fritz.py:226-272`

The node keeps its name (SUMMARIZE_CONVERSATION_NODE wiring, diagram, docs all stay valid) but its body reduces to: snapshot the messages, hand them to a daemon thread, return the RemoveMessage trim. The trim stays SYNCHRONOUS and is NOT deferred to the next turn — it costs nothing (pure state reduction, no LLM), it is what keeps the next turn's context window small, and deferring it would also change what `admin_panel._load_chat_history` renders. The background worker never touches graph state or the checkpointer; it only reads the snapshot list it was handed and writes to the Chroma singleton — the exact pattern `agent_tools.extract_memories_background` already uses. A per-user in-flight guard stops a rapid-fire user from stacking N concurrent 20B summaries.

class ProfileSignals(BaseModel):
    """User preference signals extracted from a conversation summary."""
    communication_style: str = Field(description="Empty string if no evidence.")
    interests: list[str] = Field(description="Empty list if no evidence.")
    dislikes: list[str] = Field(description="Empty list if no evidence.")
    notes: str = Field(description="Empty string if no evidence.")


_PROFILE_SIGNAL_SYSTEM = (
    "Extract user preference signals from the conversation summary below. "
    "Use empty string / empty list for fields with no evidence. Be specific "
    "and brief. Do not invent signals not supported by the conversation."
)

_MEMORY_KEY_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "was", "were", "has", "have",
    "user", "assistant", "conversation", "summary", "about", "they", "them",
    "from", "into", "then", "than", "also", "just", "some", "what", "which",
})


def _make_memory_key(summary_text: str) -> str:
    """Derive the Chroma document label locally instead of spending a 20B call.

    agent_tools.add_memory stores {key: summary} under a uuid4 id, so the key is
    only a label inside the metadata dict (and part of the embedded text). A
    content-word slug is as useful as an LLM's and costs nothing. Mirrors the
    existing local key convention in save_memory / _extract_and_store_memories.
    """
    words = re.findall(r"[a-z0-9]{3,}", (summary_text or "").lower())
    keep = [w for w in words if w not in _MEMORY_KEY_STOPWORDS][:5]
    return f"memory_of_{'_'.join(keep)}"[:64] if keep else "memory_of_conversation"


_SUMMARY_INFLIGHT: set[str] = set()
_SUMMARY_INFLIGHT_LOCK = threading.Lock()


def _summarize_and_profile(user_id: str | None, messages_snapshot: list) -> None:
    """Off the critical path. Touches no graph state and no checkpointer —
    it reads only the list it was handed and writes to the Chroma singleton."""
    try:
        with METRICS.time_block("summarize_background"):
            convo = list(messages_snapshot) + [
                HumanMessage(content="Please summarize the conversation above:")
            ]
            summary_response = ollama_instance.invoke(convo)
            summary = (
                f"Summary made at {get_current_time_internal()} \r\n "
                f"{summary_response.content}"
            )
            logger.debug("Summary: %s", summary)
            add_memory(user_id, _make_memory_key(summary_response.content), summary)
            if user_id:
                try:
                    signals = fast_ollama_instance.with_structured_output(
                        ProfileSignals
                    ).invoke([("system", _PROFILE_SIGNAL_SYSTEM), ("user", summary)])
                    update_user_profile(user_id, signals.model_dump())
                    logger.debug("Updated user profile for %s from conversation signals", user_id)
                except Exception as e:
                    logger.warning("User profile signal extraction failed (non-fatal): %s", e)
    except Exception as e:
        logger.warning("Background summarisation failed (non-fatal): %s", e)


def _spawn_summary(user_id: str | None, messages_snapshot: list) -> bool:
    """Start the background summary unless one is already in flight for this user."""
    key = str(user_id)
    with _SUMMARY_INFLIGHT_LOCK:
        if key in _SUMMARY_INFLIGHT:
            METRICS.increment("summarize_skipped_inflight")
            logger.debug("Summary already in flight for %s; skipping", key)
            return False
        _SUMMARY_INFLIGHT.add(key)

    def _run() -> None:
        try:
            _summarize_and_profile(user_id, messages_snapshot)
        finally:
            with _SUMMARY_INFLIGHT_LOCK:
                _SUMMARY_INFLIGHT.discard(key)

    threading.Thread(target=_run, name=f"summarize-{key}", daemon=True).start()
    METRICS.increment("summarize_backgrounded")
    return True


def summarize_conversation(state: EnhancedState, config: RunnableConfig):
    """Trim the conversation window; hand the LLM work to a background thread.

    The trim stays on the critical path because it is free and it is what keeps
    the next turn's context small. Only the three LLM calls move off-path.
    """
    logger.info("Trimming conversation; summarising off-path")
    user_id = config.get("metadata", {}).get("user_id")
    snapshot = list(state["messages"])
    if SUMMARIZE_ASYNC:
        _spawn_summary(user_id, snapshot)
    else:
        _summarize_and_profile(user_id, snapshot)
    return {"messages": [RemoveMessage(id=m.id) for m in snapshot[:-1]]}

### `mister_fritz.py:543-545`

Delete all three lines. `system_prompt` and `include_file_tools` are dead after :545 (verified by reading the whole of ask_stuff, :523-593) — `workspace_root` itself is what flows into metadata at :556. This removes a full rebuild of the 9-tool registry (plus file-tool registry when a workspace is set) per request, done solely to feed a debug log whose argument is evaluated even at INFO.

# BEFORE
    include_file_tools = workspace_root is not None
    system_prompt = get_system_description(get_conversation_tools_description(include_file_tools))
    logger.debug("Role description: %s", system_prompt)
    logger.debug("Prompt to ask: %s", full_prompt)

# AFTER
    logger.debug("Prompt to ask: %s", full_prompt)

### `mister_fritz.py:561-576`

(1) Seed `original_request` with `full_prompt` instead of `""`. This is semantically a no-op when the planner runs (planner sets the identical value at :210/:218) but is REQUIRED once the planner can be skipped — otherwise `executor`'s memory-query at :329 and `synthesizer` at :461 would read a STALE `original_request` left in the checkpoint by a previous turn. (2) Seed `needs_planner` from the gate. (3) Hoist the debug check out of the stream loop so `pretty_print()` (a stdout write per superstep) only happens at DEBUG.

    inputs = {
        "messages": [("user", full_prompt)],
        "image_paths": [],
        "user_image_paths": user_image_paths,
        "original_request": full_prompt,   # was "" — must be seeded now the planner can be skipped
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "needs_planner": should_run_planner(base_prompt, source),
    }

    final_state = None
    _trace = logger.isEnabledFor(logging.DEBUG)
    for s in app.stream(inputs, config=config, stream_mode="values"):
        final_state = s
        if _trace:
            message = s["messages"][-1] if "messages" in s and s["messages"] else None
            if message and not isinstance(message, tuple) and hasattr(message, "pretty_print"):
                message.pretty_print()

### `mister_fritz.py:628-642`

Swap the unconditional START edge for a conditional one (targets inferred from route_start's Literal return, matching the existing style of should_continue/route_executor). Delete the redundant module-level debug at :642, which rebuilds the whole tool registry at import and duplicates :600.

# BEFORE
workflow.add_edge(START, PLANNER_NODE)
workflow.add_edge(PLANNER_NODE, EXECUTOR_NODE)
...
app = workflow.compile(checkpointer=checkpointer, store=store)

logger.debug("Conversation tools description: %s", get_conversation_tools_description())

# AFTER
workflow.add_conditional_edges(START, route_start)
workflow.add_edge(PLANNER_NODE, EXECUTOR_NODE)
...
app = workflow.compile(checkpointer=checkpointer, store=store)
# (redundant module-level debug removed; :600 already logs the tool list)

### `agent_tools.py:94-144`

Give the fire-and-forget memory extraction a real schema. Wrap the fact list in an object (`{"facts": [...]}`) — Ollama's structured-output examples are all object-rooted and a top-level array schema is the riskier shape. Drops the non-greedy `re.search(r'\[.*?\]', ...)` at :132, which silently truncated at the first `]`.

_MEMORY_FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts"],
}

_EXTRACTION_PROMPT = (
    "You are a memory extraction assistant. Given a short conversation snippet, "
    "extract any facts about the USER (not the assistant) that are worth remembering "
    "long-term: preferences, personal details, timezone, job, interests, dislikes, "
    "habits, names of people they mention, etc. "
    "Put short plain-English fact strings in the 'facts' field. "
    "Use an empty list if there is nothing notable. Do not explain.\n\n"
    "User said: {user_msg}\nAssistant replied: {assistant_msg}"
)

# inside _extract_and_store_memories, replacing :126-137
    try:
        response = _ollama_client.chat(
            model=FAST_OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=OLLAMA_KEEP_ALIVE,
            format=_MEMORY_FACTS_SCHEMA,
        )
        facts = json.loads(response.message.content).get("facts", [])
        if not isinstance(facts, list):
            return
        for fact in facts[:5]:  # cap at 5 per turn to avoid noise
            ...  # :138-142 unchanged
    except Exception as e:
        logger.debug("Memory extraction skipped (non-fatal): %s", e)

### `agent_tools.py:178-207`

Serialise `update_user_profile`. It is a read-modify-write (`get_user_profile` at :185 → `put` at :202-206) that increments `interaction_count` at :196. It was previously only ever called from the graph thread; now a daemon thread calls it, so two overlapping calls (e.g. two users' background summaries, or a background summary racing a future caller) could lose an increment and therefore stall the relationship_level progression. One module-level lock, zero behaviour change otherwise.

_PROFILE_LOCK = threading.Lock()   # add `import threading` at module top


def update_user_profile(user_id: str, updates: dict) -> None:
    """...existing docstring...

    Serialised by _PROFILE_LOCK: this is a read-modify-write and is now called
    from the background summarisation thread as well as the graph thread.
    """
    with _PROFILE_LOCK:
        profile = get_user_profile(user_id)
        ...  # :186-207 body unchanged, indented one level

### `.env.example:42-57 (insert alongside the existing SUMMARIZE_THRESHOLD / MEMORY_EXTRACT block)`

Document all three new knobs in the '----- Tunables -----' section, commented-out with defaults shown, matching the surrounding style exactly.

# Messages-in-conversation threshold before the agent summarises and trims.
# SUMMARIZE_THRESHOLD=15
# Run summarisation + profile extraction on a background thread so the user
# never waits for it. Set to false to restore synchronous (slower) ordering.
# SUMMARIZE_ASYNC=true
# When the LLM planner runs. "auto" (default) uses a local heuristic so trivial
# messages skip the planner round trip entirely; "always" restores the old
# unconditional planner; "off" disables multi-step planning.
# PLANNER_MODE=auto
# Minimum raw-request length (chars) before the heuristic will even consider
# invoking the LLM planner. Lower it if multi-step requests are being missed.
# PLANNER_MIN_CHARS=60

### `tests/test_mister_fritz.py:1-127 (substantial rewrite)`

See testPlan for the exact per-function disposition. The module docstring (:1-8) and the TestPlannerParsing docstring (:52, 'Cover the brittle JSON-extraction logic in planner()') both describe behaviour that no longer exists. The `_fake_response` helper (:45-48) becomes unused and should be deleted.

# New shared helper replacing _fake_response:
def _patch_planner(decision=None, exc=None):
    """Patch fast_ollama_instance so with_structured_output(...).invoke() returns
    a PlanDecision (or raises)."""
    m = MagicMock()
    runnable = MagicMock()
    if exc is not None:
        runnable.invoke.side_effect = exc
    else:
        runnable.invoke.return_value = decision
    m.with_structured_output.return_value = runnable
    return patch.object(mister_fritz, "fast_ollama_instance", m)

# Example rewritten test:
    def test_multi_step_plan_extracted(self):
        from mister_fritz import PlanDecision
        with _patch_planner(PlanDecision(needs_planning=True,
                                         steps=["fetch data", "summarise it"])):
            result = planner(_state_with_message("do a thing"), config={"metadata": {}})
        self.assertEqual(result["plan"], ["fetch data", "summarise it"])

### `tests/test_agent_tools.py:200-213`

`test_substantial_turn_still_calls_llm` sets `mock_resp.message.content = "[]"`. After the change `json.loads("[]")` yields a list, `.get("facts")` raises AttributeError, and the broad `except Exception` at :143 swallows it — so the test would keep PASSING while silently exercising the error path. Update the payload and assert the schema is actually passed.

    def test_substantial_turn_still_calls_llm(self):
        with patch("agent_tools._ollama_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.message.content = '{"facts": []}'
            mock_client.chat.return_value = mock_resp
            agent_tools._extract_and_store_memories(
                user_id="alice",
                user_message="I work as a backend engineer at a fintech startup in Berlin.",
                assistant_response=(
                    "Excellent. I shall make a note of your profession and location. "
                    "Berlin is a fine city for the trade."
                ),
            )
        mock_client.chat.assert_called_once()
        self.assertEqual(
            mock_client.chat.call_args.kwargs["format"],
            agent_tools._MEMORY_FACTS_SCHEMA,
        )

### `README.md:255`

This bullet describes exactly the tests being deleted ('planner JSON parsing (code fences, surrounding text, malformed input, exception fallback)'). Rewrite it to describe the new coverage.

# BEFORE
- `mister_fritz` — planner JSON parsing (code fences, surrounding text, malformed input, exception fallback)

# AFTER
- `mister_fritz` — heuristic planner gate (`should_run_planner`) and START routing, structured-output planner decisions, local memory-key derivation, off-critical-path summarisation

### `CHANGELOG.md:44-45 (append to the ### Performance list, after the Phase 14 block)`

Phase-style entry matching the existing 'Phase N — <topic>' convention with nested sub-bullets.

- **Phase 15 — per-message latency tax.**
  - The LangGraph `START` edge is now conditional. A local heuristic (`should_run_planner`) decides whether the FAST-model planner round trip is worth paying; "hi" now reaches the executor with zero preceding LLM calls. New `PLANNER_MODE` (`auto` / `always` / `off`) and `PLANNER_MIN_CHARS` knobs; `planner_invoked` / `planner_skipped` counters expose the hit rate in `/health`.
  - Conversation summarisation moved off the critical path. `summarize_conversation` now only trims the message window (free) and hands the three LLM calls to a daemon thread, guarded per-user so rapid-fire turns can't stack concurrent 20B summaries. New `SUMMARIZE_ASYNC` kill switch; `summarize_background` latency and `summarize_backgrounded` / `summarize_skipped_inflight` counters.
  - The second 20B call whose only job was inventing a label like `memory_of_pie` is gone — `_make_memory_key` derives the slug locally from the summary's content words.
  - All three "respond ONLY with JSON" + regex-scrape sites (planner, profile signals, `agent_tools` memory extraction) now use Ollama's native structured outputs. A malformed response is a real error instead of silently degrading planning to simple mode, and the non-greedy `[.*?]` regex that truncated fact lists at the first `]` is gone.
  - `ask_stuff` no longer rebuilds the entire tool registry per request just to feed a debug log, and `pretty_print()` per superstep is now DEBUG-gated.

## Steps

1. Commit 1 — config. Add `PLANNER_MODE`, `PLANNER_MIN_CHARS`, `SUMMARIZE_ASYNC` to `fritz_utils.py` after `MEMORY_EXTRACT_MIN_REPLY_CHARS` (line 119); document all three in `.env.example` in the Tunables block; add a `TestPlannerAndSummarizeKnobs` class to `tests/test_fritz_utils.py` next to the existing `TestConstantDefaults`.
2. Commit 2 — dead work (zero behavioural risk, land first for a clean baseline). Delete `mister_fritz.py:543-545`; hoist `logger.isEnabledFor(logging.DEBUG)` out of the `app.stream` loop and gate `pretty_print()` on it (`:571-576`); delete the redundant module-level debug at `:642`. Run `ruff check .` and the full suite — nothing should change.
3. Commit 3 — seed `original_request`. Change `mister_fritz.py:565` from `"original_request": "",` to `"original_request": full_prompt,`. Semantically a no-op today (planner writes the identical value) but it is the precondition for skipping the planner without leaking last turn's request into `executor`'s memory query at `:329`. Verify by running a two-turn conversation with `LOG_LEVEL=DEBUG` and checking the 'Injected memory context' line.
4. Commit 4 — planner gate. Add `needs_planner: bool` to `EnhancedState`; add `_VOICE_SOURCES`, `_MULTI_STEP_PATTERN`, `should_run_planner`, `route_start` above `should_continue`; seed `needs_planner` in `ask_stuff`'s `inputs`; swap `workflow.add_edge(START, PLANNER_NODE)` for `workflow.add_conditional_edges(START, route_start)`. Add `TestPlannerGate` and `TestRouteStart` to `tests/test_mister_fritz.py`. Sanity-check the generated `mister_fritz_diagram.png` shows the branch (it is written on a daemon thread at `:652`).
5. Commit 5 — structured planner. Add `PlanDecision` + `_PLANNER_SYSTEM` at module scope; replace `planner()`'s `:186-204` body with `with_structured_output`; delete `import json` from line 1 and confirm `ruff check .` is clean (this is the step that trips F401 if the import is left behind). Rewrite `TestPlannerParsing` per the test plan.
6. Commit 6 — local memory key. Add `_MEMORY_KEY_STOPWORDS` and `_make_memory_key`; use it at the `add_memory` call site, deleting the `response_key_inputs` prompt (`:235-238`) and the second `ollama_instance.invoke` (`:239`). Add `TestMemoryKey`.
7. Commit 7 — summarisation off-path. Add `ProfileSignals`, `_PROFILE_SIGNAL_SYSTEM`, `_SUMMARY_INFLIGHT` + lock, `_summarize_and_profile`, `_spawn_summary`; reduce `summarize_conversation` to snapshot + spawn + `RemoveMessage`. Add `TestSummarizeOffPath` and `TestSummaryInflightGuard`.
8. Commit 8 — profile write lock. Add `import threading` and `_PROFILE_LOCK` to `agent_tools.py`; wrap the body of `update_user_profile` (`:185-207`).
9. Commit 9 — structured memory extraction. Add `_MEMORY_FACTS_SCHEMA`, rewrite `_EXTRACTION_PROMPT` to name the `facts` field, pass `format=` to `_ollama_client.chat`, replace the `re.search(r'\[.*?\]')` parse with `json.loads(...).get("facts", [])`. Update `test_substantial_turn_still_calls_llm` and add `TestMemoryExtractionStructuredOutput`.
10. Commit 10 — docs. Update `README.md:255`; append the Phase 15 entry to the `### Performance` list in `CHANGELOG.md` (after the Phase 14 block ending at line 44).
11. Verification pass against a live Ollama (see testPlan.manualVerification) — this is the step that cannot be settled statically and must not be skipped.

## Config and env changes

- PLANNER_MODE (new, fritz_utils.py) — `auto` | `always` | `off`, default `auto`. `always` reproduces today's unconditional planner and is the no-code-change rollback lever; `off` disables multi-step planning entirely. Unknown values silently normalise to `auto`.
- PLANNER_MIN_CHARS (new, fritz_utils.py) — int, default 60. Minimum length of the RAW user request (not the `format_prompt`-wrapped message, which always adds ~90 chars of boilerplate) before the heuristic will consider invoking the LLM planner.
- SUMMARIZE_ASYNC (new, fritz_utils.py) — bool, default true. When false, `summarize_conversation` runs `_summarize_and_profile` inline instead of spawning a thread, restoring the old synchronous ordering.
- .env.example — all three added to the '----- Tunables (defaults shown; uncomment to override) -----' block alongside the existing SUMMARIZE_THRESHOLD entry, commented out with the surrounding comment style.
- No change to SUMMARIZE_THRESHOLD's default (15). Its meaning shifts though: it no longer trades latency against context, only context size. Raising it belongs to the history-window item.

## Tests
### New

- tests/test_mister_fritz.py::TestPlannerGate — truth table for `should_run_planner`: ("hi", DISCORD_TEXT) -> False; a 200-char chatty message with no multi-step phrasing -> False; "research the tokio async runtime then write me a one-page report on it" -> True; "first fetch the logs, then summarise the errors" -> True; a long multi-step request with source=DISCORD_VOICE -> False; same with TELEGRAM_VOICE -> False; None and "" -> False; with `patch.object(mister_fritz, "PLANNER_MODE", "always")` even "hi" -> True; with "off" even a multi-step request -> False. NOTE: `PLANNER_MODE` and `PLANNER_MIN_CHARS` must be read from the module scope at call time for `patch.object` to work — import them as module-level names in `mister_fritz` and reference them bare inside the function (same pattern `fritz_utils.is_admin` uses for ROOT_USER).
- tests/test_mister_fritz.py::TestRouteStart — `route_start({"needs_planner": True, "messages": []})` == "planner"; `False` -> "executor"; key absent entirely (simulating an old checkpoint) -> "executor".
- tests/test_mister_fritz.py::TestPlannerStructuredOutput (replaces TestPlannerParsing) — needs_planning=False -> plan []; two steps -> both preserved; ten steps -> capped at 5; single step -> collapses to simple mode; `with_structured_output(...).invoke` raising RuntimeError -> simple mode; `original_request` preserved. Each test must assert that `fast_ollama_instance.with_structured_output` was called with `PlanDecision` — otherwise a bare `MagicMock` makes several of these pass vacuously (see risks).
- tests/test_mister_fritz.py::TestMemoryKey — `_make_memory_key` returns a `memory_of_` prefix; length <= 64; deterministic for the same input; drops stopwords; returns "memory_of_conversation" for "" and for punctuation-only input.
- tests/test_mister_fritz.py::TestSummarizeOffPath — with `mister_fritz.ollama_instance` and `mister_fritz.fast_ollama_instance` patched and `mister_fritz._spawn_summary` patched, call `summarize_conversation(state, {"metadata": {"user_id": "alice"}})`; assert every returned message is a `RemoveMessage`, assert `len(returned) == len(state["messages"]) - 1`, assert `ollama_instance.invoke` and `fast_ollama_instance.invoke` were NEVER called, and assert `_spawn_summary` received a list equal to (but not identical to — it must be a snapshot copy) `state["messages"]`.
- tests/test_mister_fritz.py::TestSummarizeSyncFallback — with `patch.object(mister_fritz, "SUMMARIZE_ASYNC", False)` and `_summarize_and_profile` patched, assert the node calls it inline and does not spawn a thread.
- tests/test_mister_fritz.py::TestSummaryInflightGuard — patch `threading.Thread` inside `mister_fritz`; first `_spawn_summary("alice", [])` returns True, a second call while the id is still in `_SUMMARY_INFLIGHT` returns False and starts no thread; a call for a different user_id still returns True. Clear `_SUMMARY_INFLIGHT` in tearDown.
- tests/test_agent_tools.py::TestMemoryExtractionStructuredOutput — assert `_ollama_client.chat` receives `format=_MEMORY_FACTS_SCHEMA`; assert `{"facts": ["alice lives in Berlin", "alice is a backend engineer"]}` produces two `add_memory` calls (patch `agent_tools.add_memory`); assert a response with 8 facts is capped at 5; assert a fact shorter than 9 chars is dropped (existing `len(fact.strip()) > 8` rule at :139).
- tests/test_agent_tools.py::TestProfileLock — assert `update_user_profile` is serialised: patch `get_user_profile` and `_get_chroma_store`, run 20 calls across 4 threads, assert the final `interaction_count` written equals 20 (this test would fail against the pre-change unlocked version).
- tests/test_fritz_utils.py::TestPlannerAndSummarizeKnobs — `PLANNER_MODE` defaults to "auto" and is always one of the three legal values; `PLANNER_MIN_CHARS` is a positive int; `SUMMARIZE_ASYNC` is a bool. Follows the existing `TestConstantDefaults` style.

### Existing tests affected

- tests/test_mister_fritz.py::TestPlannerParsing::test_multi_step_plan_extracted — WILL FAIL. Expects `["fetch data", "summarise it"]`; with `fast_ollama_instance` a bare MagicMock, `with_structured_output(...).invoke()` returns a MagicMock, iterating it yields nothing, so plan == []. Rewrite against `PlanDecision`.
- tests/test_mister_fritz.py::TestPlannerParsing::test_plan_capped_at_five_steps — WILL FAIL for the same reason. Rewrite.
- tests/test_mister_fritz.py::TestPlannerParsing::test_handles_markdown_code_fences — WILL FAIL. DELETE: markdown fences are impossible under `format=<schema>`.
- tests/test_mister_fritz.py::TestPlannerParsing::test_handles_surrounding_text — WILL FAIL. DELETE: obsolete for the same reason.
- tests/test_mister_fritz.py::TestPlannerParsing::test_non_string_steps_coerced — WILL FAIL / obsolete. Pydantic v2 `list[str]` does NOT coerce `[1, 2, 3]`; it raises ValidationError, which the try/except turns into simple mode. DELETE and replace with a test asserting a ValidationError from the structured runnable falls back to simple mode.
- tests/test_mister_fritz.py::TestPlannerParsing::test_simple_mode_when_needs_planning_false — PASSES VACUOUSLY after the change (a MagicMock decision yields needs_planning truthy but zero steps, so the `len(steps) > 1` gate at :206 sends it to simple mode anyway). Rewrite to return a real `PlanDecision(needs_planning=False, steps=[])`.
- tests/test_mister_fritz.py::TestPlannerParsing::test_handles_bare_code_fences — PASSES VACUOUSLY. DELETE.
- tests/test_mister_fritz.py::TestPlannerParsing::test_malformed_json_falls_back_to_simple_mode — PASSES VACUOUSLY. DELETE (superseded by the ValidationError test).
- tests/test_mister_fritz.py::TestPlannerParsing::test_empty_response_falls_back_to_simple_mode — PASSES VACUOUSLY. DELETE.
- tests/test_mister_fritz.py::TestPlannerParsing::test_single_step_plan_collapses_to_simple_mode — passes but for the wrong reason. Rewrite against a real `PlanDecision(needs_planning=True, steps=["just one"])`.
- tests/test_mister_fritz.py::TestPlannerParsing::test_llm_exception_falls_back_to_simple_mode — KEEP, update the mock: set `m.with_structured_output.side_effect = RuntimeError("ollama exploded")` (or make the returned runnable's `invoke` raise).
- tests/test_mister_fritz.py::TestPlannerParsing::test_original_request_preserved — KEEP, update to the new mock shape. Behaviour is unchanged: `planner` still returns `original_request: latest_message`.
- tests/test_mister_fritz.py — the `_fake_response` helper (lines 45-48) becomes unused; delete it. The module docstring (lines 1-8) and the class docstring (line 52, "Cover the brittle JSON-extraction logic in planner()") describe the deleted behaviour and must be rewritten.
- tests/test_agent_tools.py::TestMemoryExtractionSkipGuard::test_substantial_turn_still_calls_llm — still PASSES but silently exercises the exception path (`json.loads("[]")` is a list, `.get` raises, the broad except at agent_tools.py:143 swallows it). Update `mock_resp.message.content` to `'{"facts": []}'` and add a `format=` assertion.
- tests/test_agent_tools.py::TestMemoryExtractionSkipGuard::test_short_user_message_skips_extraction and ::test_short_reply_skips_extraction — UNAFFECTED (they return before any chat call).
- tests/test_admin_panel.py::test_authed_send_invokes_ask_stuff_with_username and the other `/chat/send` + `/chat/stream` tests — UNAFFECTED: they replace the whole `mister_fritz` module via `patch.dict(sys.modules, {"mister_fritz": fake_module})`, and `ask_stuff`'s signature is unchanged.
- tests/test_observability.py — UNAFFECTED; its only `ask_stuff` reference (line 201) is a metric-name string, not the function.
- Coverage gate: `--cov-fail-under=60`. Net lines are removed from `mister_fritz.py` (a large deleted block in `summarize_conversation` and `planner`) and the new code is well covered, so the gate should move in the right direction — but re-run the full `pytest tests/ --cov=. --cov-fail-under=60` locally before pushing.

### Manual verification

- STRUCTURED OUTPUT SMOKE TEST (cannot be settled statically — this is the experiment that decides whether the whole structured-output half of this plan is viable). Against a live Ollama: `python -c "from langchain_ollama import ChatOllama; from mister_fritz import PlanDecision, _PLANNER_SYSTEM; s = ChatOllama(model='llama3.2', keep_alive='5m').with_structured_output(PlanDecision); [print(m, '->', s.invoke([('system', _PLANNER_SYSTEM), ('user', m)])) for m in ['hi', 'research the tokio async runtime then write me a one-page report on it']]"`. Expect a real `PlanDecision` both times, with `needs_planning=False, steps=[]` for "hi". If llama3.2 errors or hangs under `format=<schema>`, fall back to `method="json_mode"` (which sets `format="json"` and keeps the schema in the prompt) and keep a `json.loads` + Pydantic `model_validate` parse.
- STRUCTURED OUTPUT LATENCY. Time 20 iterations of `llm.invoke(prompt)` vs `llm.with_structured_output(PlanDecision).invoke(prompt)` on the same warm model and compare p50. Grammar-constrained decoding is usually neutral-to-faster here (fewer tokens emitted), but if it is materially slower, the structured-output change is a reliability win with a latency cost and should be reported as such — it does not block the gate or the off-path work.
- PLANNER SKIP. `LOG_LEVEL=DEBUG` the bot, DM it "hi". Expect exactly one "Planner gate: skipped (heuristic)" line and NEITHER "Planner chose simple mode" nor "Planner created N-step plan". Then DM "research the tokio async runtime then write me a one-page report on it" and expect "Planner created N-step plan" plus the per-step progress callbacks.
- END-TO-END LATENCY. Time a "hi" round trip before and after (wall clock from message send to first token). Report the delta honestly — the expected win is one FAST-model round trip on a warm model, typically a few hundred ms, not seconds. If `prewarm.py` has not run (cold model) the win is much larger.
- SUMMARISATION OFF-PATH. Drive a thread past `SUMMARIZE_THRESHOLD` (16+ messages — easiest with `SUMMARIZE_THRESHOLD=4`). The reply must return without the multi-second 20B stall. Confirm a `summarize-<user>` thread appears in `threading.enumerate()` / the logs, then confirm `/health` shows `summarize_background` latency and a `summarize_backgrounded` counter, and that a `memory_of_*` entry landed in Chroma (`/export` or `privacy.export_memories`).
- CONCURRENT TURN DURING SUMMARISATION. Trigger the summarise threshold, then immediately send a second message. The second message must be answered normally with no exception. Verify the checkpoint is intact afterwards via `GET /chat/history` — the trim already committed synchronously, so the window should show exactly the post-trim message set plus the new turn, not a partially-applied state.
- RAPID-FIRE GUARD. Trigger the threshold twice in quick succession and confirm `summarize_skipped_inflight` increments rather than two 20B summaries running concurrently.
- MEMORY KEY QUALITY. Inspect a few `memory_of_*` keys produced by `_make_memory_key` in the Chroma export and confirm they read as topical slugs, not stopword soup. Then confirm `search_memories` still surfaces those summaries for a relevant query — the key is part of the embedded `json.dumps(value)` text, so a degraded key marginally changes the embedding.
- ROLLBACK DRILL. Restart with `PLANNER_MODE=always SUMMARIZE_ASYNC=false` and confirm the pre-change behaviour returns (planner runs for "hi", summarisation blocks the reply).

## Risks

- STRUCTURED OUTPUT MAY NOT WORK ON THE CONFIGURED MODELS. This is the single largest unknown and cannot be verified by reading code. `format=<json schema>` is compiled to a GBNF grammar by llama.cpp; small models sometimes stall or emit degenerate output under tight grammars. MITIGATED by design: all three structured sites use FAST_OLLAMA_MODEL (llama3.2 by default), never THINKING_OLLAMA_MODEL (gpt-oss) — reasoning models are the ones most likely to emit a thinking preamble ahead of the JSON, and the only THINKING call (the summary itself, `:231`) stays free-text. DETECT: the "Planner structured output failed" warning firing repeatedly, or `summarize_background` latency spiking. SETTLE with the smoke test in manualVerification; fall back to `method="json_mode"` if needed.
- MAGICMOCK MAKES REWRITTEN TESTS PASS VACUOUSLY. With `fast_ollama_instance` patched as a bare MagicMock, `with_structured_output(...).invoke()` returns a MagicMock; `bool(mock.needs_planning)` is True but `[str(s) for s in mock.steps[:5]]` is `[]` (MagicMock's `__iter__` yields nothing), so the `len(steps) > 1` gate silently sends everything to simple mode and four existing tests keep passing while testing nothing. Every rewritten planner test MUST return a real `PlanDecision` and assert `with_structured_output` was called with the schema class.
- SILENT `ruff` FAILURE IN CI. Removing both `json.loads` sites makes `import json` (mister_fritz.py:1) unused; ruff's F401 is in the `select = ["E", "F", "W"]` set and CI runs `ruff check .` before the tests. Delete the import in the same commit. `import re` must stay (used by `_make_memory_key` and `_MULTI_STEP_PATTERN`) — if `_make_memory_key` is placed in another module instead, `re` also becomes unused.
- GATE FALSE NEGATIVE — a genuinely multi-step request phrased without any trigger word (or in a language other than English: "primero X, luego Y") skips the planner. CONSEQUENCE is quality, not failure: the request is handled by the simple-mode ReAct agent, which can still call several tools in one loop. This is the deliberate direction of the bias. DETECT: users reporting that complex requests got shallower answers; compare the `planner_invoked` / `planner_skipped` ratio in `/health` against the pre-change turn count. MITIGATE by lowering `PLANNER_MIN_CHARS` or adding phrases to `_MULTI_STEP_PATTERN`.
- GATE FALSE POSITIVE — `\bthen\b` alone is loose and will match filler ("back then", "I'll do it then") in any message over 60 chars. CONSEQUENCE is exactly the status quo: one wasted FAST round trip, after which the LLM planner almost certainly answers `needs_planning: false`. Acceptable, but it caps the achievable win on chatty long messages.
- STALE `original_request` IF STEP 3 IS SKIPPED. `original_request` lives in the checkpoint. Today the planner overwrites it every turn; once the planner can be skipped, an un-seeded `original_request` would leave LAST turn's request in place, and `executor:329` would run the memory-similarity search against the wrong text. Seeding it in `ask_stuff`'s `inputs` (step 3) is mandatory, not optional — land it before the conditional edge.
- BACKGROUND SUMMARY LOST AT SHUTDOWN. Daemon threads are killed at interpreter exit, so a summary in flight during a restart is lost. The trim already committed, so the conversation state is consistent — only that one summary never reaches Chroma. Identical exposure to the existing `extract_memories_background`. Accepted; a `_SUMMARY_INFLIGHT`-aware atexit join is deliberately NOT added (it would block shutdown behind a 20B call).
- ONE-TURN STALE MEMORY INJECTION. If the user sends another message while the background summary is still running, that turn's auto-injected memory context (`executor:330-332`) will not include the in-flight summary. One turn of staleness; the conversation window itself is already correct because the trim was synchronous.
- PROFILE INCREMENT RACE. `update_user_profile` is a read-modify-write that bumps `interaction_count` (`agent_tools.py:196`), which drives `relationship_level`. Moving its caller onto a daemon thread makes concurrent calls possible. `_PROFILE_LOCK` (step 8) closes it. Without the lock the failure is silent and slow: the relationship level simply stops progressing.
- MERGE CONFLICT WITH THE history-window ITEM. Both items edit `summarize_conversation`, `should_continue`/`route_executor`, and the meaning of `SUMMARIZE_THRESHOLD`. They are not dependent (neither must land first) but they WILL conflict textually. Land one, rebase the other, and re-verify that the history-window trimming logic still runs on the graph thread with no LLM call.
- CHROMA METADATA KEY CONSTRAINTS. `_make_memory_key`'s output becomes a metadata key in `ChromaStore.mset` (`storage.py:151-169`). The `memory_of_` prefix and `[a-z0-9_]` charset avoid colliding with the reserved `namespace` / `original_key` keys that store.mset injects, and the 64-char cap keeps it well inside Chroma's limits. Do not loosen the character class.
- LOST STDOUT TRACE. Gating `pretty_print()` behind DEBUG removes a stdout trace some operator may be relying on when running the bot in a terminal at INFO. Recover it with `LOG_LEVEL=DEBUG`; call it out in the CHANGELOG entry.

## Rollback
"Two layers. (1) RUNTIME, no deploy needed: set `PLANNER_MODE=always` and `SUMMARIZE_ASYNC=false` and restart. `should_run_planner` then returns True unconditionally (every turn routes START->planner exactly as before) and `summarize_conversation` calls `_summarize_and_profile` inline on the graph thread, restoring the old blocking behaviour. This is precisely why `PLANNER_MODE` is a tri-state string rather than a bool. The three behaviour-neutral changes — structured outputs, the local memory key, and the dead-work deletions — are NOT covered by this lever, by design: they change no observable behaviour except that a malformed model response is now a logged failure instead of a silent downgrade. (2) CODE: the eleven commits are independent and revert cleanly in reverse order. Commit 9 (agent_tools structured extraction) and commit 2 (dead-work deletions) can be reverted in isolation without touching the graph. If only the structured-output half misbehaves on your models, revert commits 5, 7 (the ProfileSignals portion) and 9 and keep the gate + off-path work, which are the actual latency wins. No schema migration and no checkpoint migration is involved: `needs_planner` is a new optional state key that old checkpoints simply lack, and `.get()` reads it as falsy (routing those turns to the executor — the fast path — which is the safe default)."

## Open questions for you to decide

- Is `PLANNER_MIN_CHARS=60` the right floor, and is bare `\bthen\b` too loose a trigger? These are pure judgement calls with no right answer until there is production data. Suggestion: ship with the `planner_invoked` / `planner_skipped` counters, watch the ratio in `/health` for a week, and tune. If `planner_invoked` stays above ~15% of turns, tighten the regex.
- Should plan mode survive at all? The planner+executor-loop+synthesizer path is ~150 lines serving what the gate now predicts is a small single-digit percentage of turns, and modern ReAct agents handle most sequential work in one loop. Deleting `PLANNER_NODE` and `SYNTHESIZER_NODE` outright would be a much bigger latency and complexity win. EXPLICITLY DEFERRED — this plan keeps plan mode fully intact and only stops paying for it on every message. Revisit once the counters show how often it actually fires.
- Should the generated summary be re-injected into `state["messages"]` as a `SystemMessage` after the trim, rather than only landing in Chroma and being re-surfaced by the similarity search at `executor:329-333`? Today the trim discards everything but the last message and the summary is only recoverable if the embedding search happens to retrieve it. That is arguably the real context-loss bug — but it belongs to the history-window item, not here, and moving summarisation off-path is a prerequisite either way (an off-path summary cannot be injected into the same turn's state).
- Now that summarisation is free, `SUMMARIZE_THRESHOLD=15` (which trims down to ONE message) is far more aggressive than it needs to be. Raising it is the obvious follow-up but it changes token cost per turn and therefore belongs to the history-window owner's judgement, not this item's.
- Keep `SUMMARIZE_ASYNC` permanently or delete it after a soak period? It exists purely as a rollback lever and a determinism aid for debugging; leaving it forever means maintaining two code paths in `summarize_conversation`. Recommend keeping it — it is three lines and makes the node deterministically testable.
- `search_memories_internal` (agent_tools.py:75-84) merges `ChromaStore` metadata dicts straight into its JSON payload, which means the injected "What I know about this user" block also contains the internal `namespace` and `original_key` keys for all 30 results. Noticed while verifying `add_memory`'s key semantics. Out of scope here, but it is wasted context tokens on every single turn and is worth a line in the history-window item.
- `executor` returns a bare `str` at `:452`, which `add_messages` coerces to a **HumanMessage** — verified empirically this session. Every Fritz reply is therefore stored with the wrong role, which corrupts both the summarisation prompt and `admin_panel._load_chat_history`. Deliberately NOT fixed here (it changes what the model sees on every turn, which is the history-window item's blast radius), but it should be fixed by someone soon.
