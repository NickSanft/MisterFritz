import base64
import importlib.util
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime

import httpx
import ollama as _ollama_client
import pytz
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool

import document_engine
import image_generator
from file_tools import get_file_tools_description
from fritz_utils import (
    DOC_STORAGE_DESCRIPTION,
    FAST_OLLAMA_MODEL,
    MEMORY_EXTRACT_MIN_REPLY_CHARS,
    MEMORY_EXTRACT_MIN_USER_CHARS,
    VISION_MODEL,
)
from observability import METRICS
from storage import ChromaStore

logger = logging.getLogger(__name__)

# Shared HTTP client for scrape_web. Reusing the client gives us connection
# pooling and HTTP keep-alive across calls. Timeout splits connect vs read so
# a slow-but-reachable host doesn't get the same budget as a dead one.
_HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0),
    follow_redirects=True,
    headers={"User-Agent": "MisterFritz/1.0 (+scrape_web)"},
)

_chroma_store = None


def _get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record_tool(name: str) -> None:
    METRICS.increment(f"tool.{name}")


def get_current_time_internal(timezone_name: str = "US/Central") -> str:
    """Return the current time in the given IANA timezone as an RFC3339 string."""
    try:
        tz = pytz.timezone(timezone_name)
    except pytz.UnknownTimeZoneError:
        logger.warning("Unknown timezone %r, falling back to US/Central", timezone_name)
        tz = pytz.timezone("US/Central")
    now = datetime.now(pytz.utc).astimezone(tz)
    timestamp = now.isoformat()
    logger.debug("Current time (%s): %s", timezone_name, timestamp)
    return timestamp


def search_memories_internal(config: RunnableConfig, query: str) -> str:
    user_id = config.get("metadata").get("user_id")
    search_result = _get_chroma_store().search(query, (str(user_id),), limit=30)
    summaries = {}
    for _, summary_dict in search_result:
        for key, summary in summary_dict.items():
            summaries[key] = summary
    json_summaries = json.dumps(summaries)
    logger.debug("Memory search returned %d items for %s", len(summaries), user_id)
    return json_summaries


def add_memory(user_id: str, memory_key: str, memory_to_store: str) -> str:
    """Store a memory for a user."""
    memory_dict = {memory_key: memory_to_store}
    _get_chroma_store().put((str(user_id),), str(uuid.uuid4()), memory_dict)
    return "Added memory for {}: {}".format(memory_key, memory_to_store)


_EXTRACTION_PROMPT = (
    "You are a memory extraction assistant. Given a short conversation snippet, "
    "extract any facts about the USER (not the assistant) that are worth remembering "
    "long-term: preferences, personal details, timezone, job, interests, dislikes, "
    "habits, names of people they mention, etc. "
    "Return ONLY a JSON array of short plain-English fact strings. "
    "Return [] if there is nothing notable. Do not explain.\n\n"
    "User said: {user_msg}\nAssistant replied: {assistant_msg}"
)


def _extract_and_store_memories(user_id: str, user_message: str, assistant_response: str) -> None:
    """Run a fast LLM pass to pull facts from the turn and persist them. Non-blocking caller."""
    if not user_message or not assistant_response:
        return
    # Trivial turns ("hi", "lol", "thanks") almost never carry memorable facts.
    # Gating them out skips a wasted LLM call + N embedding writes per turn.
    if (
        len(user_message.strip()) < MEMORY_EXTRACT_MIN_USER_CHARS
        or len(assistant_response.strip()) < MEMORY_EXTRACT_MIN_REPLY_CHARS
    ):
        METRICS.increment("memory_extract_skipped")
        logger.debug(
            "Skipping memory extraction for %s (user=%d chars, reply=%d chars)",
            user_id, len(user_message), len(assistant_response),
        )
        return
    prompt = _EXTRACTION_PROMPT.format(
        user_msg=user_message[:600],
        assistant_msg=assistant_response[:600],
    )
    try:
        response = _ollama_client.chat(
            model=FAST_OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.message.content.strip()
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not match:
            return
        facts = json.loads(match.group())
        if not isinstance(facts, list):
            return
        for fact in facts[:5]:  # cap at 5 per turn to avoid noise
            if isinstance(fact, str) and len(fact.strip()) > 8:
                key = f"auto_{fact[:50].replace(' ', '_').lower()}"
                add_memory(user_id, key, fact.strip())
                logger.debug("Auto-extracted memory for %s: %s", user_id, fact)
    except Exception as e:
        logger.debug("Memory extraction skipped (non-fatal): %s", e)


def extract_memories_background(user_id: str, user_message: str, assistant_response: str) -> None:
    """Spawn a daemon thread to extract memories without blocking the response."""
    import threading
    threading.Thread(
        target=_extract_and_store_memories,
        args=(user_id, user_message, assistant_response),
        daemon=True,
    ).start()


_RELATIONSHIP_LEVELS = [
    (25, "trusted"),
    (10, "familiar"),
    (3,  "acquaintance"),
    (0,  "stranger"),
]


def get_user_profile(user_id: str) -> dict:
    """Fetch the structured profile for a user, returning an empty dict if none exists."""
    result = _get_chroma_store().get(f"profile_{user_id}")
    if result:
        raw = result.get("profile_data", "")
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
    return {}


def update_user_profile(user_id: str, updates: dict) -> None:
    """Merge signal updates into the user profile and persist it.

    Automatically increments the interaction count and derives the
    relationship_level from it — those two fields are not settable via updates.
    """
    profile = get_user_profile(user_id)

    for key, value in updates.items():
        if key in ("interaction_count", "relationship_level"):
            continue  # managed internally
        if isinstance(value, list):
            existing = profile.get(key, [])
            merged = list({v for v in (existing if isinstance(existing, list) else []) + value if v})
            profile[key] = merged
        elif value:  # skip empty strings
            profile[key] = value

    profile["interaction_count"] = profile.get("interaction_count", 0) + 1
    count = profile["interaction_count"]
    profile["relationship_level"] = next(
        level for threshold, level in _RELATIONSHIP_LEVELS if count >= threshold
    )

    _get_chroma_store().put(
        (str(user_id),),
        f"profile_{user_id}",
        {"profile_data": json.dumps(profile)},
    )
    logger.debug("Updated profile for %s: count=%d level=%s", user_id, count, profile["relationship_level"])


# ── LangChain tools ───────────────────────────────────────────────────────────

@tool(parse_docstring=True)
def get_current_time(timezone_name: str = "US/Central") -> str:
    """
    Returns the current time as a string in RFC3339 (YYYY-MM-DDTHH:MM:SS) format.

    Always pass the user's timezone if you know it. Check the user profile for a
    'timezone' field before calling. Example - 2025-01-13T23:11:56.337644-06:00

    Args:
        timezone_name: IANA timezone name e.g. "America/Chicago", "America/New_York",
                       "Europe/London", "Asia/Tokyo". Defaults to US/Central.
    """
    _record_tool("get_current_time")
    return get_current_time_internal(timezone_name)


@tool(parse_docstring=True)
def scrape_web(url: str):
    """
    Takes in the string of the URL and returns results all readable text from the website

    Args:
    url: The URL to pull from the internet.

    Returns:
    string: The readable text from the website.
    """
    try:
        _record_tool("scrape_web")
        response = _HTTP_CLIENT.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            element.extract()
        text = soup.get_text(' ')
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return clean_text
    except Exception as e:
        METRICS.record_error("scrape_web", e)
        logger.warning("Scrape web error: %s", e)
        return f"Error: {e}"


@tool(parse_docstring=True)
def search_web(text_to_search: str):
    """
    Takes in a string and returns results from the internet.

    Args:
    text_to_search: The text to search the internet for information.

    Returns:
    list: A list of dictionaries, each containing string keys and string values representing the search results.
    """
    _record_tool("search_web")
    results = DDGS().text(text_to_search, max_results=5)
    logger.debug("Search web results count: %s", len(results) if results else 0)
    return results


@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int, config: RunnableConfig):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice: The number of dice to roll.
    num_sides: The number of sides on each die.
    config: The RunnableConfig.

    Returns:
    list: A list containing the result of each die roll.
    """
    _record_tool("roll_dice")
    user_id = config.get("metadata").get("user_id")
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")
    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return f"Here are the results: {user_id}. {rolls}"


@tool(parse_docstring=True)
def generate_image(prompt: str):
    """
    Use this tool to generate an image if the user asks you to

    Args:
    prompt: The prompt to give to Stable Diffusion to generate the image.

    Returns:
    string: The path of the image.
    """
    _record_tool("generate_image")
    return image_generator.generate_image(prompt)


@tool(parse_docstring=True)
def search_documents(query: str):
    """
    Use this tool to get information when the user asks about the description provided in the prompt.

    Args:
    query: The question or search term to look for in the documents that match the search_documents description.

    Returns:
    string: The answer derived from the documents.
    """
    _record_tool("search_documents")
    return document_engine.query_documents(query)


@tool(parse_docstring=True)
def search_memories(config: RunnableConfig, query: str):
    """ This function returns memories in JSON format based on a search term.

    Args:
        config: The RunnableConfig.
        query: The keywords do to a semantic search for.
    """
    _record_tool("search_memories")
    return search_memories_internal(config, query)


@tool(parse_docstring=True)
def save_memory(config: RunnableConfig, fact: str):
    """Save an important fact, preference, or detail about the user for future conversations.

    Use this proactively during conversation when the user reveals something worth remembering:
    personal details, stated preferences, communication style, interests, things they dislike,
    or anything that would help personalise future responses. Do not wait until the end —
    save it the moment it is mentioned.

    Args:
        config: The RunnableConfig.
        fact: The fact or preference to remember, written as a plain sentence.

    Returns:
        string: Confirmation that the memory was saved.
    """
    _record_tool("save_memory")
    user_id = config.get("metadata", {}).get("user_id", "unknown")
    key = f"fact_{fact[:50].replace(' ', '_').lower()}"
    return add_memory(user_id, key, fact)


@tool(parse_docstring=True)
def analyze_image(config: RunnableConfig, question: str = "What is in this image?"):
    """
    Analyzes images using a vision model which has already been downloaded.

    Args:
        config: The RunnableConfig containing user_image_paths in metadata.
        question: The specific question to ask about the image(s). Default: "What is in this image?"

    Returns:
        string: A description of what's in the image(s).
    """
    metadata = config.get("metadata", {})
    user_images = metadata.get("user_image_paths", [])
    _record_tool("analyze_image")
    logger.debug("User images: %s", user_images)

    if not user_images:
        return "No images were attached to analyze. Please ask the user to attach an image."

    try:
        encoded_images = []
        for img_path in user_images:
            try:
                with open(img_path, "rb") as image_file:
                    encoded_images.append(base64.b64encode(image_file.read()).decode("utf-8"))
            except Exception as e:
                METRICS.record_error("analyze_image.read", e)
                logger.warning("Error reading image %s: %s", img_path, e)

        if not encoded_images:
            return "Could not read the attached images. Please try again."

        logger.info("Analyzing %d image(s) with %s", len(encoded_images), VISION_MODEL)
        response = _ollama_client.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': question,
                'images': encoded_images,
            }]
        )
        analysis = response['message']['content']
        logger.info("Vision analysis complete: %d chars", len(analysis))
        return analysis

    except Exception as e:
        METRICS.record_error("analyze_image", e)
        error_msg = f"Error analyzing image: {str(e)}"
        logger.warning(error_msg)
        return error_msg


# ── Skill auto-loader ─────────────────────────────────────────────────────────

_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
_skills_cache: dict | None = None


def _load_skills() -> dict:
    """Auto-discover and load tools from the skills/ directory.

    Any .py file in skills/ that exposes a register() -> dict function is
    imported and its tools added to the registry. Results are cached so
    discovery only runs once per process.
    """
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    _skills_cache = {}
    if not os.path.isdir(_SKILLS_DIR):
        return _skills_cache

    for filename in sorted(os.listdir(_SKILLS_DIR)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        filepath = os.path.join(_SKILLS_DIR, filename)
        try:
            spec = importlib.util.spec_from_file_location(f"skills.{module_name}", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if callable(getattr(module, "register", None)):
                registered = module.register()
                if isinstance(registered, dict):
                    _skills_cache.update(registered)
                    logger.info("Loaded skill '%s': %d tool(s)", module_name, len(registered))
        except Exception as e:
            logger.warning("Failed to load skill '%s': %s", module_name, e)

    return _skills_cache


# ── Tool registry ─────────────────────────────────────────────────────────────

def make_list_schedules_tool(user_id: str, schedule_manager) -> BaseTool:
    """Return a tool that lists the user's active recurring schedules."""

    @tool
    def list_my_schedules() -> str:
        """List all of your active recurring scheduled tasks.
        Use this when the user asks what they have scheduled or what reminders are active."""
        try:
            schedules = schedule_manager.list_schedules(user_id)
        except Exception as e:
            return f"Failed to retrieve schedules: {e}"
        if not schedules:
            return "You have no active recurring scheduled tasks."
        lines = []
        for s in schedules:
            label = f" — {s['description']}" if s["description"] else ""
            lines.append(f"ID `{s['id']}`{label}: every `{s['schedule']}` → {s['prompt']}")
        return "\n".join(lines)

    return list_my_schedules


def make_cancel_reminder_tool(user_id: str, schedule_manager) -> BaseTool:
    """Return a tool that cancels a scheduled task by ID."""

    @tool
    def cancel_reminder(schedule_id: str) -> str:
        """Cancel a scheduled task or reminder by its ID.
        Use list_my_schedules first if you need to find the ID.

        Args:
            schedule_id: The 8-character schedule ID to cancel.
        """
        try:
            removed = schedule_manager.remove_schedule(schedule_id, user_id)
            return f"Schedule `{schedule_id}` cancelled." if removed else f"No schedule found with ID `{schedule_id}`."
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"Failed to cancel: {e}"

    return cancel_reminder


def make_schedule_message_tool(channel_id: int, schedule_manager) -> BaseTool:
    """Return a tool that lets Fritz schedule a one-time message in the current channel."""

    @tool
    def schedule_message(delay_minutes: int, message: str) -> str:
        """Schedule a one-time message to be sent in the current Discord channel after a delay.
        Use this when the user asks you to remind them of something or say something in N minutes.
        delay_minutes must be a positive integer. message is what Fritz should say when the time arrives."""
        try:
            sid = schedule_manager.schedule_once(channel_id, "scheduled", delay_minutes, message)
            return f"Scheduled (ID: {sid}). Will deliver in {delay_minutes} minute(s)."
        except Exception as e:
            logger.warning("schedule_message tool failed: %s", e)
            return f"Failed to schedule: {e}"

    return schedule_message


def get_conversation_tools_description(
    include_file_tools: bool = False,
    extra_tools: dict[str, tuple[BaseTool, str]] | None = None,
) -> dict[str, tuple[BaseTool, str]]:
    """Return a dict of {name: (tool, description)} for all conversation tools.

    extra_tools: optional caller-supplied tools to merge in (e.g. schedule_message,
    list_my_schedules, cancel_reminder which need runtime context like channel_id).
    """
    tools = {
        "get_current_time": (get_current_time, "Fetch the current time. Pass the user's timezone_name if known."),
        "scrape_web": (scrape_web, "If provided a URL by the user, this can be used to scrape a website's HTML."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something."),
        "roll_dice": (roll_dice, "Roll different types of dice."),
        "search_memories": (search_memories, "Returns a JSON payload of stored memories you have had with a user based on a search term."),
        "save_memory": (save_memory, "Proactively save a fact, preference, or personal detail about the user for future conversations."),
        "search_documents": (search_documents, f"Search local documents. Use this for questions about: {DOC_STORAGE_DESCRIPTION}"),
        "generate_image": (generate_image, "Generates an image based on a given prompt."),
        "analyze_image": (analyze_image, "Analyzes an image. If the user asks about an image, assume that the tool knows its location."),
    }
    if extra_tools:
        tools.update(extra_tools)
    tools.update(_load_skills())
    if include_file_tools:
        tools.update(get_file_tools_description())
    return tools
