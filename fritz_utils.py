import json
import os
import shutil
from enum import Enum

from dotenv import load_dotenv

# Load .env file if present (silently ignored when absent)
load_dotenv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_key_from_json_config_file(key_name: str) -> str | None:
    """Read a key from config.json (legacy fallback — prefer env vars)."""
    file_path = "config.json"
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get(key_name)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, Exception):
        return None


# Keep the public name so existing callers (e.g. main_discord.py) don't break
# during the transition period.
def get_key_from_json_config_file(key_name: str) -> str | None:
    return _get_key_from_json_config_file(key_name)


def _env_or_json(env_key: str, json_key: str, default: str | None = None) -> str | None:
    """Return env var if set, else fall back to config.json, else default."""
    return os.environ.get(env_key) or _get_key_from_json_config_file(json_key) or default


def _find_binary(name: str, fallback: str) -> str:
    """Return the system binary path if found, otherwise the bundled fallback."""
    system = shutil.which(name)
    return system if system else fallback


# ---------------------------------------------------------------------------
# Paths & storage
# ---------------------------------------------------------------------------

DOC_FOLDER = os.environ.get("DOC_FOLDER", "./input")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_store")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "word_docs_rag")
CHAT_DB_NAME = os.environ.get("CHAT_DB_NAME", "chat_history.db")
SCHEDULE_DB = os.environ.get("SCHEDULE_DB", "schedules.db")
INDEXED_FILES_PATH = os.path.join(CHROMA_DB_PATH, "indexed_files.txt")

# ---------------------------------------------------------------------------
# Ollama / model config
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
THINKING_OLLAMA_MODEL = os.environ.get("THINKING_OLLAMA_MODEL", "gpt-oss")
FAST_OLLAMA_MODEL = os.environ.get("FAST_OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

DISCORD_KEY = "discord_bot_token"  # legacy config.json key (kept for compat)
DISCORD_BOT_TOKEN: str | None = _env_or_json("DISCORD_BOT_TOKEN", DISCORD_KEY)

# ---------------------------------------------------------------------------
# Application config
# ---------------------------------------------------------------------------

DOC_STORAGE_DESCRIPTION: str = _env_or_json(
    "DOC_STORAGE_DESCRIPTION",
    "doc_storage_description",
    "anything you don't know about.",
)

ROOT_USER: str | None = _env_or_json("ROOT_USER", "root_user")

# ---------------------------------------------------------------------------
# FFmpeg — prefer system install; fall back to bundled Windows executables
# ---------------------------------------------------------------------------

FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH") or _find_binary("ffmpeg", "./ffmpeg.exe")
FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH") or _find_binary("ffprobe", "./ffprobe.exe")

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
# Options: tiny, base, small, medium, large  (larger = slower but more accurate)
# "base" is a good default — ~140 MB, runs on CPU in reasonable time.
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
# Set to "cuda" if a GPU is available.
WHISPER_LANGUAGE: str | None = os.environ.get("WHISPER_LANGUAGE") or None
# None = auto-detect language. Set e.g. "en" to force English and speed up inference.

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN: str | None = _env_or_json("TELEGRAM_BOT_TOKEN", "telegram_bot_token")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Raise RuntimeError with a clear message if required config is missing."""
    missing = []
    if not DISCORD_BOT_TOKEN:
        missing.append("DISCORD_BOT_TOKEN  (or 'discord_bot_token' in config.json)")
    if not ROOT_USER:
        missing.append("ROOT_USER  (or 'root_user' in config.json)")
    if missing:
        raise RuntimeError(
            "Missing required configuration:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nSet these in a .env file or as environment variables."
            + " See .env.example for reference."
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_TEXT_AND_IMAGE = 1,
    DISCORD_VOICE = 2,
    LOCAL = 3,
    TELEGRAM_TEXT = 4,
    TELEGRAM_VOICE = 5,
