import functools
import json
import logging
import os
import re
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
#
# Raised from 15 now that summarisation runs off the critical path
# (SUMMARIZE_ASYNC): 15 was aggressive because each pass blocked the reply
# behind three LLM calls, so the cost of summarising often had to be paid
# sooner than the conversation warranted. In the background the trade flips —
# a higher threshold buys longer in-thread memory, and the window the executor
# replays is bounded by this value plus one. Each pass does get slower, since
# it feeds the whole message list to the 20B model, but nobody waits for it.
SUMMARIZE_THRESHOLD: int = int(os.environ.get("SUMMARIZE_THRESHOLD", "30"))

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

# Hard cap (characters) on what scrape_web returns to the agent. Same reasoning
# as the memory blob above: one long article, returned whole, lands in the
# executor's input and can push the conversation window out of num_ctx. The
# deleted browser_tools.py had this cap; the live scraper never did.
SCRAPE_MAX_CHARS: int = int(os.environ.get("SCRAPE_MAX_CHARS", "8000"))

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

# Concurrency knobs that must never be < 1. asyncio.Semaphore(0) is perfectly
# legal and simply never releases, so IMAGE_GEN_MAX_CONCURRENCY=0 did not fail
# — it made /gen hang on `async with` until Discord expired the interaction 15
# minutes later, every time, with nothing in the log to explain it.
#
# Clamped at import so the bot cannot deadlock, and remembered so
# validate_config() can say so out loud once logging is actually configured
# (a warning emitted at import time may have nowhere to go yet).
_CLAMPED_KNOBS: list[tuple[str, str]] = []


def _at_least_one(name: str, default: str) -> int:
    """Read an int env var, clamping to a minimum of 1 and recording if we did.

    The raw string is what gets recorded, so the warning can quote back exactly
    what was set — including a non-numeric value, which is a different mistake
    from setting 0 and deserves to read that way.
    """
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        _CLAMPED_KNOBS.append((name, repr(raw)))
        return 1
    if value < 1:
        _CLAMPED_KNOBS.append((name, str(value)))
        return 1
    return value


# Worker count for the shared bounded thread pool (bot_adapters.run_blocking)
# that keeps ask_stuff / STT / TTS work off the Discord event loop.
BLOCKING_POOL_SIZE: int = _at_least_one("BLOCKING_POOL_SIZE", "8")

# Concurrent SDXL renders permitted by the /gen semaphore. The pipeline is
# GPU-bound — leave at 1 unless you have VRAM to burn.
IMAGE_GEN_MAX_CONCURRENCY: int = _at_least_one("IMAGE_GEN_MAX_CONCURRENCY", "1")

# Concurrent XTTS syntheses permitted by the /voice semaphore.
TTS_MAX_CONCURRENCY: int = _at_least_one("TTS_MAX_CONCURRENCY", "1")

# Admin panel: shared password gate + local-only port. If ADMIN_PANEL_PASSWORD
# is unset the panel won't start at all.
ADMIN_PANEL_PASSWORD: str | None = os.environ.get("ADMIN_PANEL_PASSWORD") or None
ADMIN_PANEL_PORT: int = int(os.environ.get("ADMIN_PANEL_PORT", "8001"))

# Shared secret required to obtain a /chat identity cookie.
#
# Deliberately does NOT fall back to ADMIN_PANEL_PASSWORD: handing someone chat
# access must never hand them the admin panel. If this is unset the chat
# surface refuses every login rather than minting free identities — that
# failure is intentional and loud, and start_admin_panel logs why.
CHAT_PASSWORD: str | None = os.environ.get("CHAT_PASSWORD") or None

# Optional allowlist of usernames that may be claimed at /chat/login. Empty
# (default) = any sanitised name, once the password checks out. The password is
# the perimeter; the username is only namespacing, so without this allowlist
# anyone holding the password can claim anyone else's name and read their
# conversation.
CHAT_ALLOWED_USERS: frozenset[str] = frozenset(
    u.strip() for u in os.environ.get("CHAT_ALLOWED_USERS", "").split(",") if u.strip()
)

# Mark the chat cookie Secure. Off by default: the panel is normally reached
# over plain http through an SSH tunnel, where Secure would silently break login.
CHAT_COOKIE_SECURE: bool = os.environ.get("CHAT_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# CHAT_SHARE_DISCORD_THREAD and CHAT_THREAD_PREFIX are RETIRED. The `web-`
# namespace now lives inside the identity itself (canonical_user_id), so the
# prefix has nothing left to do, and IDENTITY_LINKS replaces the sharing flag —
# it links the whole identity rather than only the thread, so memories,
# schedules and workspace follow the conversation instead of being left behind.
# See the [Unreleased] CHANGELOG entry for the upgrade note.


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
# Image formats accepted on chat upload, keyed by the format name Pillow
# reports after actually decoding the header. The value is the canonical
# (extension, mime) we store under and serve back — the Content-Type the
# *client* declares is advisory only, since anything can claim image/png over
# an HTML or SVG body. No SVG entry: Pillow won't decode one, which is exactly
# the property we want.
CHAT_ALLOWED_IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
    "GIF": ("gif", "image/gif"),
}
# Derived so the two lists cannot drift apart. Still used as the cheap
# first-pass check on the declared Content-Type before anything is read.
CHAT_ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset(
    mime for _ext, mime in CHAT_ALLOWED_IMAGE_FORMATS.values()
)

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


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
# ONE transformation, in one place. Everything downstream — Chroma namespace,
# LangGraph thread_id, schedules.user_id, workspaces.user_id, the admin gate —
# consumes the output of canonical_user_id() verbatim.
#
# Before this, the same display string was transformed four different ways at
# four layers, so write paths and delete paths disagreed. That IS the /forget
# bug for punctuated usernames.

# A DASH, not a colon (DECISIONS #5). A colon is illegal in Windows filenames
# and silently creates an NTFS alternate data stream on write — and identities
# reach filenames in several places. The dash costs a closed platform allowlist
# in split_user_id (below) so that "web-alice-bob" still parses correctly.
IDENTITY_SEPARATOR = "-"
KNOWN_PLATFORMS: frozenset[str] = frozenset({"discord", "telegram", "web", "local"})
# Same charset admin_panel.chat_login already used for usernames — dashes
# included. Keeping them is safe because split_user_id partitions on the FIRST
# dash and no platform name contains one, so "web-alice-bob" resolves to
# platform "web", id "alice-bob" exactly as DECISIONS #5 requires.
_IDENT_STRIP_RE = re.compile(r"[^a-zA-Z0-9_-]")
_TOKEN_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def canonical_user_id(platform: str, raw_id) -> str:
    """Return the stable, namespaced identity for a user: '<platform>-<id>'.

    platform: 'discord' | 'telegram' | 'web' | 'local'
    raw_id:   the platform's IMMUTABLE id where one exists (a Discord or
              Telegram numeric snowflake), otherwise the self-asserted name
              (web). Using the snowflake is the point: a Discord rename no
              longer orphans memories, workspace, schedules and admin rights.
    """
    plat = (platform or "local").strip().lower()
    ident = _IDENT_STRIP_RE.sub("", str(raw_id or "").strip())[:64]
    if not ident:
        raise ValueError(
            f"cannot build a canonical id from platform={platform!r} raw_id={raw_id!r}")
    return f"{plat}{IDENTITY_SEPARATOR}{ident}"


def split_user_id(user_id: str | None) -> tuple[str | None, str]:
    """('discord', '123') for canonical ids; (None, <as-is>) for legacy ones.

    The platform allowlist is what makes a dash separator safe: a legacy
    display name containing a dash ("jean-luc") has no known platform prefix,
    so it is correctly reported as legacy rather than split at the first dash.
    """
    if not user_id or IDENTITY_SEPARATOR not in user_id:
        return None, (user_id or "")
    plat, _, ident = user_id.partition(IDENTITY_SEPARATOR)
    plat = plat.lower()
    return (plat, ident) if plat in KNOWN_PLATFORMS and ident else (None, user_id)


def is_canonical_user_id(user_id: str | None) -> bool:
    return split_user_id(user_id)[0] is not None


def safe_user_token(user_id: str | None) -> str:
    """Filesystem- and URL-safe rendering of an identity.

    Canonical ids are already safe by construction (dash separator, alnum and
    underscore elsewhere). This exists for legacy ids that may still carry
    punctuation, and as a guard for anything building a path from an identity.
    """
    return _TOKEN_SAFE_RE.sub("_", user_id or "") or "anonymous"


# Per-channel conversation threads. Turning this on BRANCHES every existing
# conversation: the identity-only thread stays in the DB untouched, but new
# messages start a fresh per-channel thread. Off by default so an upgrade
# preserves continuity.
THREADS_PER_CHANNEL: bool = os.environ.get("THREADS_PER_CHANNEL", "false").lower() in ("1", "true", "yes")

# Transitional: match ROOT_USER / ADMIN_USERS against the human display name as
# well as the canonical id.
#
# Defaults to FALSE (DECISIONS #6) — secure from day one, with no impersonation
# window. The cost is that you must put your numeric Discord ID in ADMIN_USERS
# BEFORE deploying, or you lose admin commands. While this is true, anyone who
# takes the owner's freed-up username inherits admin.
ADMIN_LEGACY_NAME_MATCH: bool = os.environ.get("ADMIN_LEGACY_NAME_MATCH", "false").lower() in ("1", "true", "yes")


def _parse_identity_links() -> dict[str, str]:
    """IDENTITY_LINKS=web-alice=discord-123456789,web-bob=discord-987654321"""
    out: dict[str, str] = {}
    for pair in os.environ.get("IDENTITY_LINKS", "").split(","):
        alias, sep, primary = pair.strip().partition("=")
        if sep and alias.strip() and primary.strip():
            out[alias.strip()] = primary.strip()
    return out


IDENTITY_LINKS: dict[str, str] = _parse_identity_links()


def resolve_identity(user_id: str) -> str:
    """Follow IDENTITY_LINKS one hop. Deliberately not transitive — a chain
    would make the effective identity depend on config ordering."""
    return IDENTITY_LINKS.get(user_id, user_id)


def thread_id_for(user_id: str, channel_key: str | None = None) -> str:
    """LangGraph thread id for an identity, optionally per channel.

    With THREADS_PER_CHANNEL off (default) this is the identity alone, which
    preserves today's one-thread-per-user behaviour across an upgrade.
    """
    if not THREADS_PER_CHANNEL or not channel_key:
        return user_id
    return f"{user_id}#{safe_user_token(str(channel_key))}"


def is_admin(user_id: str | None, display_name: str | None = None) -> bool:
    """Return True if the caller is the project owner (ROOT_USER) or in ADMIN_USERS.

    `user_id` should be a canonical id. `display_name` is consulted ONLY when
    ADMIN_LEGACY_NAME_MATCH is on, so a pre-migration config keeps working for
    one release — at the cost of matching admin rights on a mutable username.

    Reads from the module's own scope at call time so tests can patch
    ROOT_USER / ADMIN_USERS without re-importing callers.
    """
    candidates = [user_id]
    if ADMIN_LEGACY_NAME_MATCH and display_name:
        candidates.append(display_name)
    for candidate in candidates:
        if not candidate:
            continue
        if ROOT_USER and candidate == ROOT_USER:
            return True
        if candidate in ADMIN_USERS:
            return True
    return False

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

    for knob, requested in _CLAMPED_KNOBS:
        logging.getLogger(__name__).warning(
            "%s was set to %s, which is not a usable worker count; using 1. "
            "Below 1 a semaphore never releases (the guarded command hangs "
            "until Discord expires the interaction) and a thread pool refuses "
            "to start.", knob, requested,
        )

    legacy = [u for u in ([ROOT_USER] if ROOT_USER else []) + sorted(ADMIN_USERS)
              if not is_canonical_user_id(u)]
    if legacy:
        logging.getLogger(__name__).warning(
            "ROOT_USER/ADMIN_USERS still use display names (%s). Admin rights are "
            "matched on a mutable username, and ADMIN_LEGACY_NAME_MATCH is %s. "
            "Run `python migrate_identity.py --dry-run` to get the canonical ids, "
            "then set e.g. ROOT_USER=discord-123456789.",
            ", ".join(legacy),
            "ON — anyone taking that username inherits admin"
            if ADMIN_LEGACY_NAME_MATCH else
            "OFF — admin commands will NOT work until you set a canonical id",
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
