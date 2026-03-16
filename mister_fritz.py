import json
import logging
import re
from contextlib import ExitStack
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages

from agent_tools import (
    add_memory,
    get_conversation_tools_description,
    get_current_time_internal,
)
from fritz_utils import CHAT_DB_NAME, FAST_OLLAMA_MODEL, MessageSource, ROOT_USER, THINKING_OLLAMA_MODEL
from observability import METRICS, init_logging
from storage import SQLiteStore

PLANNER_NODE = "planner"
EXECUTOR_NODE = "executor"
SYNTHESIZER_NODE = "synthesizer"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"

init_logging()
logger = logging.getLogger(__name__)


class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    image_paths: list[str]       # generated images (output)
    user_image_paths: list[str]  # user-provided images (input)
    original_request: str        # raw user message, captured by planner
    plan: list[str]              # ordered steps; empty = simple mode
    current_step: int            # index into plan
    step_results: list[str]      # accumulated per-step results


# ── Prompt helpers ────────────────────────────────────────────────────────────

def get_system_description(tools: dict[str, tuple[BaseTool, str]]) -> str:
    """Build the system prompt, listing available tools."""
    tool_descriptions = "".join(
        [f"    {name}: {tup[1]}\n" for name, tup in tools.items()]
    )
    return f"""
Role:
    You are an AI conversationalist named Mister Fritz, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler.
    You do retain memories per user, and can use the search_memories tool to retrieve them when relevant to the conversation.

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
    """Decide whether to summarize or end the conversation."""
    return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > 15 else END


def route_executor(state: EnhancedState) -> Literal["executor", "synthesizer", "summarize_conversation", "__end__"]:
    """
    After executor runs, decide the next node:
    - Plan mode, more steps remain  → executor (loop)
    - Plan mode, all steps done     → synthesizer
    - Simple mode                   → should_continue logic (inline)
    """
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if plan:
        if current_step < len(plan):
            return EXECUTOR_NODE
        else:
            return SYNTHESIZER_NODE
    else:
        return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > 15 else END


# ── Graph nodes ───────────────────────────────────────────────────────────────

def planner(state: EnhancedState, config: RunnableConfig):
    """Inspect the latest user message and decide if multi-step planning is needed."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    logger.debug("Planner evaluating message: %s", latest_message)

    planner_prompt = [
        (
            "system",
            (
                "You are a planning assistant. Analyze the user's request and decide "
                "whether it can be answered in a single step or requires multiple steps.\n"
                "Respond ONLY with valid JSON, one of two forms:\n"
                '  {"needs_planning": false}\n'
                '  {"needs_planning": true, "steps": ["step 1", "step 2", ...]}\n'
                "Use multi-step only when the request clearly requires sequential actions "
                "(e.g., research then write a report, fetch data then analyse it). "
                "Simple questions, chat, or single-tool lookups do NOT need planning. "
                "Keep plans to 5 steps maximum."
            ),
        ),
        ("user", latest_message),
    ]

    try:
        response = fast_ollama_instance.invoke(planner_prompt)
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        # Extract the JSON object in case there's surrounding text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group()
        parsed = json.loads(raw)
        needs_planning = bool(parsed.get("needs_planning", False))
        steps = parsed.get("steps", []) if needs_planning else []
        steps = [str(s) for s in steps[:5]]  # cap at 5, ensure strings
    except Exception as e:
        logger.warning("Planner JSON parse failed (%s); falling back to simple mode", e)
        needs_planning = False
        steps = []

    if needs_planning and len(steps) > 1:
        logger.info("Planner created %d-step plan", len(steps))
        for i, s in enumerate(steps):
            logger.debug("  Step %d: %s", i + 1, s)
        return {
            "original_request": latest_message,
            "plan": steps,
            "current_step": 0,
            "step_results": [],
        }

    logger.info("Planner chose simple mode")
    return {
        "original_request": latest_message,
        "plan": [],
        "current_step": 0,
        "step_results": [],
    }


def summarize_conversation(state: EnhancedState, config: RunnableConfig):
    logger.info("Summarizing conversation")
    metadata = config.get("metadata", {})
    user_id = metadata.get("user_id")
    messages = state["messages"] + [HumanMessage(content="Please summarize the conversation above:")]
    summary_response = ollama_instance.invoke(messages)
    timestamp = get_current_time_internal()
    summary = f"Summary made at {timestamp} \r\n {summary_response.content}"
    logger.debug("Summary: %s", summary)
    response_key_inputs = [
        ("system", "Please provide a short sentence describing this memory starting with the word \"memory\". Example - memory_of_pie"),
        ("user", summary),
    ]
    summary_response_key = ollama_instance.invoke(response_key_inputs, config=get_config_values(config))
    logger.debug("Summary Key: %s", summary_response_key.content)
    add_memory(user_id, summary_response_key.content, summary)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
    return {"messages": delete_messages}


def executor(state: EnhancedState, config: RunnableConfig):
    """
    Simple mode (plan=[]): behaves like the original conversation node —
                           streams final response and adds to messages.
    Plan mode  (plan!=[]):  builds a context-aware prompt for the current step,
                           runs the ReAct agent, accumulates result into
                           step_results, increments current_step.
                           Does NOT update conversation messages.
    """
    messages = state["messages"]
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_results = state.get("step_results", [])
    original_request = state.get("original_request", "")

    metadata = config.get("metadata", {})
    workspace_root = metadata.get("workspace_root")
    user_id = metadata.get("user_id", "")
    include_file_tools = workspace_root is not None and user_id == ROOT_USER
    progress_callback = metadata.get("progress_callback")
    streaming_callback = metadata.get("streaming_callback")

    if include_file_tools:
        tools_desc = get_conversation_tools_description(include_file_tools=True)
        system_prompt = get_system_description(tools_desc)
        active_tools = [tool_info[0] for tool_info in tools_desc.values()]
        agent = create_agent(ollama_instance, tools=active_tools)
    else:
        system_prompt = CACHED_SYSTEM_PROMPT
        agent = conversation_react_agent

    tool_messages = {
        "generate_image": "Generating an image, this may take a moment...",
        "search_documents": "Searching through documents for you...",
        "search_web": "Searching the web...",
        "scrape_website": "Scraping website content...",
        "search_memories": "Looking through my memories...",
        "analyze_image": "Analyzing your image(s) with vision AI...",
        "list_directory": "Browsing the workspace...",
        "read_file": "Reading a file...",
        "write_file": "Writing to a file...",
        "edit_file": "Editing a file...",
        "search_files": "Searching through files...",
        "execute_command": "Running a command...",
    }
    notified_tools: set = set()

    is_plan_mode = bool(plan)

    # Build execution prompt
    if is_plan_mode:
        step_instruction = plan[current_step]
        context_parts = [f"Original request: {original_request}"]
        if step_results:
            context_parts.append("Completed steps:")
            for i, result in enumerate(step_results):
                context_parts.append(f"  Step {i + 1} ({plan[i]}): {result[:500]}")
        context_parts.append(
            f"\nCurrent task (step {current_step + 1}/{len(plan)}): {step_instruction}"
        )
        context_parts.append(
            "Complete this step. Your result will be combined with other steps into a final response."
        )
        agent_prompt = "\n".join(context_parts)
        # Announce plan step progress; suppress per-token streaming for intermediate steps
        if progress_callback:
            progress_callback(f"Step {current_step + 1}/{len(plan)}: {step_instruction}")
        effective_streaming_callback = None
    else:
        latest_message = messages[-1].content if messages else original_request
        agent_prompt = latest_message
        effective_streaming_callback = streaming_callback

    inputs = {"messages": [("system", system_prompt), ("user", agent_prompt)]}

    final_state = None
    accumulated_text = ""

    for s in agent.stream(inputs, config=get_config_values(config), stream_mode="values"):
        final_state = s
        if "messages" in s and s["messages"]:
            latest = s["messages"][-1]
            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                logger.debug("Detected tool calls: %s", [tc.get('name', '') for tc in latest.tool_calls])
                if progress_callback:
                    for tool_call in latest.tool_calls:
                        tool_name = tool_call.get('name', '')
                        if tool_name in tool_messages and tool_name not in notified_tools:
                            progress_callback(tool_messages[tool_name])
                            notified_tools.add(tool_name)
            elif hasattr(latest, 'content') and isinstance(latest.content, str) and effective_streaming_callback:
                if isinstance(latest, AIMessage):
                    new_text = latest.content
                    if new_text and new_text != accumulated_text:
                        accumulated_text = new_text
                        effective_streaming_callback(accumulated_text)

    resp = final_state["messages"][-1].content if final_state and "messages" in final_state else ""

    image_paths = state.get("image_paths", []).copy()
    if final_state and "messages" in final_state:
        for msg in final_state["messages"]:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'name') and msg.name == 'generate_image':
                image_paths.append(msg.content)

    if is_plan_mode:
        return {
            "image_paths": image_paths,
            "step_results": step_results + [resp],
            "current_step": current_step + 1,
        }
    else:
        return {
            "messages": [resp],
            "image_paths": image_paths,
        }


def synthesizer(state: EnhancedState, config: RunnableConfig):
    """
    Called only after all plan steps complete. Streams a Fritz-persona synthesis
    of the step_results into the conversation, then resets plan state.
    """
    original_request = state.get("original_request", "")
    plan = state.get("plan", [])
    step_results = state.get("step_results", [])

    metadata = config.get("metadata", {})
    streaming_callback = metadata.get("streaming_callback")

    step_summary = "\n".join(
        f"Step {i + 1} ({plan[i]}): {result}"
        for i, result in enumerate(step_results)
    )

    synthesis_prompt = [
        (
            "system",
            (
                "You are Mister Fritz, a sardonic English-butler AI. "
                "Synthesize the research findings below into a single, "
                "coherent, witty response for the user. Maintain your persona throughout."
            ),
        ),
        (
            "user",
            f"Original request: {original_request}\n\nResearch findings:\n{step_summary}",
        ),
    ]

    accumulated_text = ""
    try:
        for chunk in ollama_instance.stream(synthesis_prompt, config=get_config_values(config)):
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_text += chunk.content
                if streaming_callback:
                    streaming_callback(accumulated_text)
    except Exception as e:
        logger.warning("Synthesizer stream failed: %s; falling back to invoke", e)
        accumulated_text = ollama_instance.invoke(
            synthesis_prompt, config=get_config_values(config)
        ).content

    logger.info("Synthesizer complete: %d chars from %d steps", len(accumulated_text), len(plan))
    return {
        "messages": [accumulated_text],
        "plan": [],
        "current_step": 0,
        "step_results": [],
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
) -> dict:
    """Process user input and return structured output with text and attachments."""
    import re as _re
    user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)
    if user_image_paths:
        full_prompt = format_prompt(base_prompt, source, user_id_clean, f" User has attached images: {user_image_paths}")
    else:
        user_image_paths = []
        full_prompt = format_prompt(base_prompt, source, user_id_clean)

    include_file_tools = workspace_root is not None and user_id_clean == ROOT_USER
    system_prompt = get_system_description(get_conversation_tools_description(include_file_tools))
    logger.debug("Role description: %s", system_prompt)
    logger.debug("Prompt to ask: %s", full_prompt)

    config = {
        "configurable": {"user_id": user_id_clean, "thread_id": user_id_clean},
        "metadata": {
            "user_id": user_id_clean,
            "thread_id": user_id_clean,
            "progress_callback": progress_callback,
            "streaming_callback": streaming_callback,
            "user_image_paths": user_image_paths,
            "workspace_root": workspace_root,
        }
    }
    inputs = {
        "messages": [("user", full_prompt)],
        "image_paths": [],
        "user_image_paths": user_image_paths,
        "original_request": "",
        "plan": [],
        "current_step": 0,
        "step_results": [],
    }

    final_state = None
    for s in app.stream(inputs, config=config, stream_mode="values"):
        final_state = s
        message = s["messages"][-1] if "messages" in s and s["messages"] else None
        if message and not isinstance(message, tuple) and hasattr(message, 'pretty_print'):
            message.pretty_print()

    final_text = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_msg = final_state["messages"][-1]
        final_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    image_paths = final_state.get("image_paths", []) if final_state else []
    return {
        "text": final_text,
        "image_paths": image_paths,
        "timestamp": get_current_time_internal(),
    }


# ── Setup & initialisation ────────────────────────────────────────────────────

conversation_tools = [tool_info[0] for tool_info in get_conversation_tools_description().values()]
logger.debug("Conversation tools: %s", conversation_tools)

CACHED_SYSTEM_PROMPT = get_system_description(get_conversation_tools_description())

store = SQLiteStore(CHAT_DB_NAME)
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(CHAT_DB_NAME))
ollama_instance = ChatOllama(model=THINKING_OLLAMA_MODEL)
fast_ollama_instance = ChatOllama(model=FAST_OLLAMA_MODEL)

conversation_react_agent = create_agent(ollama_instance, tools=conversation_tools)

workflow = StateGraph(EnhancedState)
workflow.add_node(PLANNER_NODE, planner)
workflow.add_node(EXECUTOR_NODE, executor)
workflow.add_node(SYNTHESIZER_NODE, synthesizer)
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)

workflow.add_edge(START, PLANNER_NODE)
workflow.add_edge(PLANNER_NODE, EXECUTOR_NODE)
workflow.add_conditional_edges(EXECUTOR_NODE, route_executor)
workflow.add_conditional_edges(SYNTHESIZER_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

app = workflow.compile(checkpointer=checkpointer, store=store)

logger.debug("Conversation tools description: %s", get_conversation_tools_description())

try:
    with open("mister_fritz_diagram.png", "wb") as binary_file:
        binary_file.write(app.get_graph().draw_mermaid_png())
except Exception as _diagram_err:
    logger.debug("Could not write graph diagram (non-fatal): %s", _diagram_err)
