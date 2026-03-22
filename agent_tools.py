import base64
import importlib.util
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime
from typing import Optional

import ollama as _ollama_client
import pytz
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool

import document_engine
import image_generator
from file_tools import get_file_tools_description
from fritz_utils import DOC_STORAGE_DESCRIPTION, ROOT_USER, VISION_MODEL
from observability import METRICS
from storage import ChromaStore

logger = logging.getLogger(__name__)

_chroma_store = None


def _get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record_tool(name: str) -> None:
    METRICS.increment(f"tool.{name}")


def get_current_time_internal() -> str:
    """Return the current CST time as an RFC3339 string."""
    cst = pytz.timezone('US/Central')
    cst_now = datetime.now(pytz.utc).astimezone(cst)
    timestamp = cst_now.isoformat()
    logger.debug("Current time: %s", timestamp)
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
def get_current_time():
    """
    Returns the current time as a string in RFC3339 (YYYY-MM-DDTHH:MM:SS) format.

    Example - 2025-01-13T23:11:56.337644-06:00
    """
    _record_tool("get_current_time")
    return get_current_time_internal()


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
        response = requests.get(url, timeout=10)
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

def get_conversation_tools_description(include_file_tools: bool = False) -> dict[str, tuple[BaseTool, str]]:
    """Return a dict of {name: (tool, description)} for all conversation tools."""
    tools = {
        "get_current_time": (get_current_time, "Fetch the current time (US / Central Standard Time)."),
        "scrape_web": (scrape_web, "If provided a URL by the user, this can be used to scrape a website's HTML."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something."),
        "roll_dice": (roll_dice, "Roll different types of dice."),
        "search_memories": (search_memories, "Returns a JSON payload of stored memories you have had with a user based on a search term."),
        "save_memory": (save_memory, "Proactively save a fact, preference, or personal detail about the user for future conversations."),
        "search_documents": (search_documents, f"Search local documents. Use this for questions about: {DOC_STORAGE_DESCRIPTION}"),
        "generate_image": (generate_image, "Generates an image based on a given prompt."),
        "analyze_image": (analyze_image, "Analyzes an image. If the user asks about an image, assume that the tool knows its location."),
    }
    tools.update(_load_skills())
    if include_file_tools:
        tools.update(get_file_tools_description())
    return tools
