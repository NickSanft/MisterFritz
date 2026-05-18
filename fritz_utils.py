import json
import os
import shutil
from enum import Enum

from dotenv import load_dotenv

# Load .env file if present (silently ignored when absent)
load_dotenv()

__version__ = "0.1.0"


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
# Parent directory for per-user sandboxed workspaces created via /workspace enable.
# Each enabled user gets their own subdirectory WORKSPACES_ROOT/<user_id>/.
WORKSPACES_ROOT = os.environ.get("WORKSPACES_ROOT", "./workspaces")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "word_docs_rag")
DB_NAME      = os.environ.get("DB_NAME", "fritz.db")
CHAT_DB_NAME = os.environ.get("CHAT_DB_NAME", DB_NAME)
SCHEDULE_DB  = os.environ.get("SCHEDULE_DB",  DB_NAME)
INDEXED_FILES_PATH = os.path.join(CHROMA_DB_PATH, "indexed_files.txt")

# ---------------------------------------------------------------------------
# Ollama / model config
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
THINKING_OLLAMA_MODEL = os.environ.get("THINKING_OLLAMA_MODEL", "gpt-oss")
FAST_OLLAMA_MODEL = os.environ.get("FAST_OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")
# Hard cap on a single Ollama request. Prevents a hung model from wedging the bot.
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Tunables (formerly magic numbers in module bodies)
# ---------------------------------------------------------------------------

# Number of conversation messages before the agent triggers a summarisation pass.
SUMMARIZE_THRESHOLD: int = int(os.environ.get("SUMMARIZE_THRESHOLD", "15"))

# Max number of lines returned by the file_tools.read_file tool.
MAX_READ_LINES: int = int(os.environ.get("MAX_READ_LINES", "500"))

# Max file size (bytes) the file_tools will read or search through. 1 MiB default.
MAX_FILE_SIZE_BYTES: int = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(1_048_576)))

# Truncation length for execute_command stdout/stderr returned to the LLM.
EXEC_OUTPUT_TRUNCATE: int = int(os.environ.get("EXEC_OUTPUT_TRUNCATE", "10000"))

# Minimum delay (minutes) for one-shot scheduled tasks. Prevents accidental
# "schedule in 0 minutes" foot-guns.
SCHEDULE_MIN_DELAY_MIN: int = int(os.environ.get("SCHEDULE_MIN_DELAY_MIN", "1"))

# ---------------------------------------------------------------------------
# File-tool sandbox
# ---------------------------------------------------------------------------

# Allowlist for the `execute_command` file tool. Only argv[0] values listed here
# are permitted. Override with EXEC_ALLOWED_COMMANDS as a comma-separated list.
# Keep it tight by default — this is ROOT_USER-gated, but defence-in-depth.
_DEFAULT_EXEC_ALLOWED = (
    "ls,dir,pwd,cd,echo,cat,type,head,tail,wc,"
    "python,python3,pip,pytest,"
    "node,npm,npx,"
    "git,"
    "go,cargo,"
    "make,"
    "grep,find,where"
)
EXEC_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    cmd.strip().lower()
    for cmd in os.environ.get("EXEC_ALLOWED_COMMANDS", _DEFAULT_EXEC_ALLOWED).split(",")
    if cmd.strip()
)

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

# Additional admin Discord usernames, beyond ROOT_USER. Comma-separated env var.
# Lets you grant a friend admin powers without rotating ROOT_USER.
ADMIN_USERS: frozenset[str] = frozenset(
    u.strip()
    for u in os.environ.get("ADMIN_USERS", "").split(",")
    if u.strip()
)


def is_admin(user_id: str | None) -> bool:
    """Return True if user_id is the project owner (ROOT_USER) or in ADMIN_USERS.

    Reads from the module's own scope at call time so tests can patch
    ROOT_USER / ADMIN_USERS without re-importing callers.
    """
    if not user_id:
        return False
    if ROOT_USER and user_id == ROOT_USER:
        return True
    return user_id in ADMIN_USERS

# ---------------------------------------------------------------------------
# FFmpeg — prefer system install; fall back to bundled Windows executables
# ---------------------------------------------------------------------------

FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH") or _find_binary("ffmpeg", "./ffmpeg.exe")
FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH") or _find_binary("ffprobe", "./ffprobe.exe")

# ---------------------------------------------------------------------------
# Whisper STT (faster-whisper)
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "small")
# Options: tiny, base, small, medium, large-v2, large-v3
# "small" is the recommended default — good accuracy, reasonable CPU speed.
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
# Set to "cuda" if a GPU is available.
WHISPER_COMPUTE_TYPE: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
# int8 halves memory and roughly doubles CPU speed with minimal accuracy loss.
# Use "float16" on CUDA, "float32" for maximum accuracy on CPU.
WHISPER_LANGUAGE: str | None = os.environ.get("WHISPER_LANGUAGE") or None
# None = auto-detect language. Set e.g. "en" to force English and speed up inference.
WHISPER_BEAM_SIZE: int = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
# Higher = more accurate but slower. 5 is the Whisper default.
WHISPER_INITIAL_PROMPT: str = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    "Mister Fritz is an AI assistant. The user is speaking a conversational message.",
)
# Seeds the transcription with context — helps with proper nouns and domain vocabulary.

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
