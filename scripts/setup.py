"""
Interactive setup wizard for Mister Fritz.

Run from the repo root:

    python scripts/setup.py

Stdlib-only — safe to run before `pip install -r requirements.txt`.
Idempotent: re-running skips work that's already done.

What it does:
  1. Verifies Python >= 3.12.
  2. Checks Ollama is reachable; prints install instructions if not.
  3. Creates / pulls the four required models (skips ones already present).
  4. Prompts for a Discord bot token, validates it against the Discord API.
  5. Prompts for the bot owner's Discord username.
  6. Writes a complete .env file (backing up any existing one).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
ENV_PATH = REPO_ROOT / ".env"
MODELFILES_DIR = REPO_ROOT / "modelfiles"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
MIN_PYTHON = (3, 12)

# (local model name, modelfile path or None for direct pull)
REQUIRED_MODELS = [
    ("gpt-oss", "gpt-oss-20b-modelfile.txt"),
    ("llama3.2", "llama3.2-modelfile.txt"),
    ("llava", "llava-modelfile.txt"),
    ("mxbai-embed-large", None),  # direct pull, no modelfile customisation
]

# ── ANSI colour helpers (no-op on Windows pre-VT or NO_COLOR) ────────────────

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"


def info(msg: str) -> None:
    print(f"{_c('36', '•')} {msg}")


def success(msg: str) -> None:
    print(f"{_c('32', '✓')} {msg}")


def warn(msg: str) -> None:
    print(f"{_c('33', '!')} {msg}")


def err(msg: str) -> None:
    print(f"{_c('31', '✗')} {msg}", file=sys.stderr)


def section(title: str) -> None:
    print()
    print(_c("1;35", f"── {title} " + "─" * max(0, 60 - len(title) - 4)))


def prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{_c('34', '?')} {question}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            err("Aborted.")
            sys.exit(1)
        if raw:
            return raw
        if default is not None:
            return default
        err("A value is required.")


def confirm(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"{_c('34', '?')} {question} {suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            err("Aborted.")
            sys.exit(1)
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


# ── Step 1: Python version ───────────────────────────────────────────────────

def check_python() -> None:
    section("Checking Python version")
    actual = sys.version_info[:2]
    if actual < MIN_PYTHON:
        err(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
            f"you have {actual[0]}.{actual[1]}. "
            "Install a newer Python from https://www.python.org/downloads/"
        )
        sys.exit(1)
    success(f"Python {actual[0]}.{actual[1]} OK")


# ── Step 2: Ollama health ────────────────────────────────────────────────────

OLLAMA_INSTALL_HINTS = {
    "Windows": "Download from https://ollama.com/download/windows",
    "Darwin":  "brew install ollama  (or https://ollama.com/download/mac)",
    "Linux":   "curl -fsSL https://ollama.com/install.sh | sh",
}


def check_ollama(host: str) -> None:
    section("Checking Ollama")
    info(f"Pinging {host}/api/tags …")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            if resp.status == 200:
                success("Ollama is reachable")
                return
            err(f"Ollama responded with HTTP {resp.status}")
    except (urllib.error.URLError, TimeoutError) as e:
        err(f"Could not reach Ollama at {host}: {e}")

    hint = OLLAMA_INSTALL_HINTS.get(platform.system(), "https://ollama.com/download")
    print()
    print(f"  Install Ollama: {_c('1', hint)}")
    print("  Then start the service (usually auto-starts on macOS/Windows;")
    print("  on Linux run `ollama serve`) and re-run this script.")
    sys.exit(1)


def list_local_models(host: str) -> set[str]:
    """Return the set of locally-available Ollama model names."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        return set()
    names: set[str] = set()
    for m in data.get("models", []):
        name = m.get("name", "")
        names.add(name)
        # Ollama tags look like "llama3.2:latest" — also store the bare name.
        if ":" in name:
            names.add(name.split(":", 1)[0])
    return names


# ── Step 3: model setup ──────────────────────────────────────────────────────

def _ollama_cli_available() -> bool:
    return shutil.which("ollama") is not None


def ensure_models(host: str) -> None:
    section("Setting up models")
    if not _ollama_cli_available():
        err(
            "The `ollama` CLI is not on PATH. The wizard needs it to create/pull "
            "models. Add it to PATH and re-run."
        )
        sys.exit(1)

    present = list_local_models(host)
    info(f"Found {len(present)} model(s) already installed.")

    for model_name, modelfile in REQUIRED_MODELS:
        if model_name in present:
            success(f"{model_name} is already installed")
            continue

        if modelfile is not None:
            mf_path = MODELFILES_DIR / modelfile
            if not mf_path.exists():
                warn(f"Modelfile missing: {mf_path}. Skipping {model_name}.")
                continue
            info(f"Building {model_name} from {modelfile} (this takes a while) …")
            cmd = ["ollama", "create", "-f", str(mf_path), model_name]
        else:
            info(f"Pulling {model_name} (this takes a while) …")
            cmd = ["ollama", "pull", model_name]

        rc = subprocess.call(cmd)
        if rc != 0:
            err(f"`{' '.join(cmd)}` exited with code {rc}.")
            if not confirm("Continue with the rest of the setup anyway?", default=False):
                sys.exit(1)
        else:
            success(f"{model_name} ready")


# ── Step 4: Discord token validation ─────────────────────────────────────────

DISCORD_ME_URL = "https://discord.com/api/v10/users/@me"


def validate_discord_token(token: str) -> dict | None:
    """Hit Discord's /users/@me. Returns the user dict on success, None on failure."""
    req = urllib.request.Request(
        DISCORD_ME_URL,
        headers={"Authorization": f"Bot {token}", "User-Agent": "MisterFritz-Setup/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
            return None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None
        warn(f"Discord API returned HTTP {e.code} — try again.")
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        warn(f"Could not reach Discord API: {e}")
        return None


def gather_discord_token(existing: str | None) -> str:
    section("Discord bot token")
    print("  Create a bot at https://discord.com/developers/applications")
    print("  Bot → Reset Token → copy. Make sure 'Message Content Intent' is enabled.")
    print()

    if existing:
        if confirm("Existing DISCORD_BOT_TOKEN found in .env. Keep it?", default=True):
            user = validate_discord_token(existing)
            if user:
                success(f"Existing token is valid (bot: {user.get('username', '?')})")
                return existing
            warn("Existing token failed validation. Let's get a new one.")

    while True:
        token = prompt("Paste your Discord bot token")
        info("Validating against Discord …")
        user = validate_discord_token(token)
        if user:
            success(f"Token valid — bot: {user.get('username', '?')}#{user.get('discriminator', '0')}")
            return token
        err("Token rejected (HTTP 401). Double-check you copied it correctly.")


# ── Step 5: ROOT_USER ────────────────────────────────────────────────────────

def gather_root_user(existing: str | None) -> str:
    section("Bot owner (ROOT_USER)")
    print("  This is your numeric Discord user ID, not your username.")
    print("  Enable Developer Mode in Discord, then right-click your name")
    print("  and choose 'Copy User ID'.")
    print("  Used for admin-only commands like /workspace and /schedule add.")
    print()
    print("  A username works too, but only while ADMIN_LEGACY_NAME_MATCH=true —")
    print("  and then anyone who takes your username takes your admin rights.")
    if existing:
        if confirm(f"Keep existing ROOT_USER='{existing}'?", default=True):
            return existing
    raw = prompt("Your Discord user ID")
    # Offer the canonical form rather than silently rewriting what they typed:
    # a username may be exactly what they meant on a pre-migration install.
    if raw.isdigit():
        return f"discord-{raw}"
    return raw


# ── Step 6: write .env ───────────────────────────────────────────────────────

def parse_env(path: Path) -> dict[str, str]:
    """Minimal .env parser — KEY=VALUE per line, ignores comments and blanks."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def write_env(values: dict[str, str]) -> None:
    section("Writing .env")
    if not ENV_EXAMPLE.exists():
        err(f"Cannot find {ENV_EXAMPLE}. Are you running from the repo root?")
        sys.exit(1)

    if ENV_PATH.exists():
        backup = ENV_PATH.with_name(f".env.bak.{int(time.time())}")
        info(f"Backing up existing .env to {backup.name}")
        shutil.copy2(ENV_PATH, backup)

    # Start from .env.example, then overwrite the keys we collected.
    template = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    written_keys: set[str] = set()
    out_lines: list[str] = []
    for line in template:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in values:
                out_lines.append(f"{key}={values[key]}")
                written_keys.add(key)
                continue
        out_lines.append(line)

    # Append any values that weren't in the template (defensive — shouldn't happen).
    extras = [(k, v) for k, v in values.items() if k not in written_keys]
    if extras:
        out_lines.append("")
        out_lines.append("# ----- Added by setup.py -----")
        for k, v in extras:
            out_lines.append(f"{k}={v}")

    ENV_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    success(f"Wrote {ENV_PATH}")


# ── Step 7: CUDA hint (informational only) ───────────────────────────────────

def cuda_hint() -> None:
    section("GPU detection")
    try:
        import torch  # type: ignore
    except ImportError:
        warn("PyTorch is not installed yet (run `pip install -r requirements.txt`).")
        info("Image generation and TTS will be very slow without a CUDA GPU.")
        return
    if torch.cuda.is_available():
        success(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
    else:
        warn("CUDA not available. Image generation and TTS will fall back to CPU (slow).")


# ── Main flow ────────────────────────────────────────────────────────────────

def main() -> None:
    print(_c("1", "Mister Fritz — setup wizard"))
    print("Press Ctrl+C at any time to abort.")

    existing = parse_env(ENV_PATH)
    host = existing.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST

    check_python()
    check_ollama(host)
    ensure_models(host)

    token = gather_discord_token(existing.get("DISCORD_BOT_TOKEN") or None)
    root_user = gather_root_user(existing.get("ROOT_USER") or None)

    cuda_hint()

    write_env({
        **existing,
        "DISCORD_BOT_TOKEN": token,
        "ROOT_USER": root_user,
        "OLLAMA_HOST": host,
    })

    section("All set")
    success("Setup complete.")
    print()
    print("  Next steps:")
    print("    1. pip install -r requirements.txt   (if you haven't already)")
    print("    2. python main_discord.py")
    print()
    print("  Run /help in Discord once the bot is online to see what it can do.")


if __name__ == "__main__":
    main()
