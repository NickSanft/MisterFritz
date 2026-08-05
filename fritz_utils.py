import functools
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

@functools.lru_cache(maxsize=1)
def _load_config_json() -> dict:
    """Parse config.json once and cache the result.

    Legacy fallback path — modern config comes from env vars. The cache is
    safe because the bot doesn't reload config.json mid-run; values are
    consumed at import time only.
    """
    try:
        with open("config.json", "r") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, Exception):
        return {}


def _get_key_from_json_config_file(key_name: str) -> str | None:
    """Read a key from config.json (legacy fallback — prefer env vars)."""
    return _load_config_json().get(key_name)


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

# How long Ollama keeps a model resident in memory after the last request.
# Conservative default matches Ollama's own ("5m"). Set to "-1" to pin models
# forever — fastest, but VRAM stays consumed. Accepted formats are anything
# Ollama parses (e.g. "30s", "5m", "1h", "-1").
OLLAMA_KEEP_ALIVE: str = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")

# ---------------------------------------------------------------------------
# Tunables (formerly magic numbers in module bodies)
# ---------------------------------------------------------------------------

# Number of conversation messages before the agent triggers a summarisation pass.
SUMMARIZE_THRESHOLD: int = int(os.environ.get("SUMMARIZE_THRESHOLD", "15"))

# Token budget for the slice of conversation history handed to the executor's
# ReAct sub-agent each turn. The sub-agent is compiled WITHOUT a checkpointer,
# so this window is the only short-term memory the model gets; Chroma memory
# injection covers long-range recall. Sized against num_ctx=32768 (see
# modelfiles/): ~900 system prompt + ~1900 tool schemas + ~1000 injected
# memories + 4096 history still leaves ~24k for tool output and the reply.
# Set to 0 to disable the window entirely and restore pre-window behaviour.
HISTORY_TOKEN_BUDGET: int = int(os.environ.get("HISTORY_TOKEN_BUDGET", "4096"))

# Hard cap (characters) on the Chroma memory blob auto-injected into the
# system prompt. search_memories_internal pulls up to 30 stored summaries with
# no size limit; uncapped, that block alone can evict the history window — and
# the system prompt itself — from the model's context.
MEMORY_INJECT_MAX_CHARS: int = int(os.environ.get("MEMORY_INJECT_MAX_CHARS", "4000"))

# Characters buffered before the streaming pipeline fires streaming_callback.
# 1 emits every token; raising it coalesces deltas (fewer SSE frames / Discord
# hops) at the cost of choppier streaming.
STREAM_MIN_CHARS: int = max(1, int(os.environ.get("STREAM_MIN_CHARS", "1")))

# Minimum seconds between Discord message edits and between worker-thread →
# event-loop streaming hops. Keeps a ~40 token/s model from queueing 40
# coroutines/s onto the gateway loop.
DISCORD_STREAM_MIN_INTERVAL: float = float(os.environ.get("DISCORD_STREAM_MIN_INTERVAL", "1.5"))

# When true, conversation summarisation runs on a background thread instead of
# blocking the reply that crossed SUMMARIZE_THRESHOLD. Set false to restore
# the old synchronous ordering (rollback / deterministic debugging).
SUMMARIZE_ASYNC: bool = os.environ.get("SUMMARIZE_ASYNC", "true").lower() in ("1", "true", "yes")

# Max number of lines returned by the file_tools.read_file tool.
MAX_READ_LINES: int = int(os.environ.get("MAX_READ_LINES", "500"))

# Max file size (bytes) the file_tools will read or search through. 1 MiB default.
MAX_FILE_SIZE_BYTES: int = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(1_048_576)))

# Truncation length for execute_command stdout/stderr returned to the LLM.
EXEC_OUTPUT_TRUNCATE: int = int(os.environ.get("EXEC_OUTPUT_TRUNCATE", "10000"))

# Minimum delay (minutes) for one-shot scheduled tasks. Prevents accidental
# "schedule in 0 minutes" foot-guns.
SCHEDULE_MIN_DELAY_MIN: int = int(os.environ.get("SCHEDULE_MIN_DELAY_MIN", "1"))

# Max recurring schedules any single user may have at once. Abuse cap.
MAX_SCHEDULES_PER_USER: int = int(os.environ.get("MAX_SCHEDULES_PER_USER", "10"))

# Memory-extraction skip thresholds. Conversation turns shorter than these
# (e.g. "hi", "lol") trigger a wasted LLM call + N embedding writes for
# zero useful signal. Lower these if you're losing valid short facts.
MEMORY_EXTRACT_MIN_USER_CHARS: int = int(os.environ.get("MEMORY_EXTRACT_MIN_USER_CHARS", "20"))
MEMORY_EXTRACT_MIN_REPLY_CHARS: int = int(os.environ.get("MEMORY_EXTRACT_MIN_REPLY_CHARS", "40"))

# Worker count for the shared bounded thread pool (bot_adapters.run_blocking)
# that keeps ask_stuff / STT / TTS work off the Discord event loop.
BLOCKING_POOL_SIZE: int = int(os.environ.get("BLOCKING_POOL_SIZE", "8"))

# Concurrent SDXL renders permitted by the /gen semaphore. Must be >= 1;
# 0 would deadlock the command. The pipeline is GPU-bound — leave at 1 unless
# you have VRAM to burn.
IMAGE_GEN_MAX_CONCURRENCY: int = int(os.environ.get("IMAGE_GEN_MAX_CONCURRENCY", "1"))

# Concurrent XTTS syntheses permitted by the /voice semaphore. Must be >= 1.
TTS_MAX_CONCURRENCY: int = int(os.environ.get("TTS_MAX_CONCURRENCY", "1"))

# Admin panel: shared password gate + local-only port. If ADMIN_PANEL_PASSWORD
# is unset the panel won't start at all.
ADMIN_PANEL_PASSWORD: str | None = os.environ.get("ADMIN_PANEL_PASSWORD") or None
ADMIN_PANEL_PORT: int = int(os.environ.get("ADMIN_PANEL_PORT", "8001"))


# Secret used to HMAC-sign the chat identity cookie. If unset, we auto-generate
# one and persist it to .chat_cookie_secret on first boot so cookies survive
# restarts. The file is gitignored.
def _load_or_make_chat_cookie_secret() -> str:
    env_value = os.environ.get("CHAT_COOKIE_SECRET")
    if env_value:
        return env_value
    secret_path = ".chat_cookie_secret"
    try:
        with open(secret_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
            if existing:
                return existing
    except FileNotFoundError:
        pass
    import secrets as _secrets
    new_secret = _secrets.token_hex(32)
    try:
        with open(secret_path, "w", encoding="utf-8") as f:
            f.write(new_secret)
    except OSError:
        # Read-only filesystem — fall back to an in-process ephemeral secret.
        pass
    return new_secret


CHAT_COOKIE_SECRET: str = _load_or_make_chat_cookie_secret()

# Hard caps on chat-uploaded files (Phase web-chat-4). Images are 10 MB each
# by default — enough for high-res photos, low enough that someone can't
# accidentally fill the disk dragging a video in. Documents share the same
# default; admins can bump for large PDFs.
CHAT_IMAGE_UPLOAD_MAX_BYTES: int = int(
    os.environ.get("CHAT_IMAGE_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024))
)
CHAT_DOC_UPLOAD_MAX_BYTES: int = int(
    os.environ.get("CHAT_DOC_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024))
)
# Allowed MIME types for image uploads. Strict whitelist — no SVG (XSS risk),
# no animated formats beyond GIF.
CHAT_ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif",
})

# Syntax-highlight fenced code blocks in web-chat replies (Pygments via the
# markdown codehilite extension). Escape hatch: set false if highlighting
# misbehaves — code still renders as plain <pre> blocks.
CHAT_CODE_HIGHLIGHT: bool = os.environ.get("CHAT_CODE_HIGHLIGHT", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# File-tool sandbox
# ---------------------------------------------------------------------------

# Allowlist for the `execute_command` file tool. Only argv[0] values listed here
# are permitted. Override with EXEC_ALLOWED_COMMANDS as a comma-separated list.
#
# NOT ROOT_USER-gated since Phase 7b: any user who runs `/workspace enable`
# reaches this tool. Every entry here that can evaluate caller-supplied code —
# python/node (-c/-e), pip/npm (install scripts), make (user Makefile),
# git (-c, ! aliases), find (-exec) — is arbitrary code execution for that
# user. That capability is deliberately kept and contained by EXEC_REQUIRE_ADMIN
# below plus the environment scrub in file_tools._build_exec_env.
_DEFAULT_EXEC_ALLOWED = (
    "ls,dir,pwd,echo,cat,type,head,tail,wc,"
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

# execute_command only. The other five file tools stay open to any workspace
# holder (Phase 7b) — they are confined by _resolve_safe_path. Running programs
# is not confined by anything the bot controls, so it needs an identity check.
# Falsy values: 0/false/no/off/empty.
EXEC_REQUIRE_ADMIN: bool = os.environ.get(
    "EXEC_REQUIRE_ADMIN", "true"
).strip().lower() not in ("0", "false", "no", "off", "")

# Environment variables handed to execute_command children. Everything else is
# dropped, so DISCORD_BOT_TOKEN / ADMIN_PANEL_PASSWORD / CHAT_COOKIE_SECRET /
# OLLAMA_HOST never reach a subprocess. Names are matched case-insensitively
# (os.environ upper-cases keys on Windows).
_DEFAULT_EXEC_ENV_PASSTHROUGH_NT = (
    "PATH,PATHEXT,SYSTEMROOT,WINDIR,COMSPEC,SYSTEMDRIVE,"
    "TEMP,TMP,USERPROFILE,HOMEDRIVE,HOMEPATH,APPDATA,LOCALAPPDATA,"
    "NUMBER_OF_PROCESSORS,PROCESSOR_ARCHITECTURE,OS"
)
_DEFAULT_EXEC_ENV_PASSTHROUGH_POSIX = "PATH,HOME,TMPDIR,LANG,LC_ALL,TZ,TERM,USER,LOGNAME"
EXEC_ENV_PASSTHROUGH: frozenset[str] = frozenset(
    name.strip().upper()
    for name in os.environ.get(
        "EXEC_ENV_PASSTHROUGH",
        _DEFAULT_EXEC_ENV_PASSTHROUGH_NT if os.name == "nt"
        else _DEFAULT_EXEC_ENV_PASSTHROUGH_POSIX,
    ).split(",")
    if name.strip()
)

# Hard ceiling on the execute_command timeout (seconds). Was a bare constant in
# file_tools; env-configurable now to match every other tunable.
EXEC_TIMEOUT_MAX: int = int(os.environ.get("EXEC_TIMEOUT_MAX", "30"))

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

DISCORD_KEY = "discord_bot_token"  # legacy config.json key (kept for compat)
DISCORD_BOT_TOKEN: str | None = _env_or_json("DISCORD_BOT_TOKEN", DISCORD_KEY)

# When true, user-facing error messages append the exception type and text.
# Off by default: end users get butler copy plus a log ref, and the traceback
# stays in the log. Turn on for local debugging of a private instance.
DISCORD_ERROR_DETAIL: bool = os.environ.get("DISCORD_ERROR_DETAIL", "").lower() in ("1", "true", "yes")

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
