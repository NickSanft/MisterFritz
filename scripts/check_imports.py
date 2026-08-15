"""Import every core module and report which ones fail.

The point is to catch a dependency that was dropped from the core set but is
still needed at import time. `pytest` alone will not catch it: the suite stubs
the heavy optional modules in tests/conftest.py, so a module can import fine
under test and still explode on a real boot.

Modules that legitimately require an optional extra are listed in OPTIONAL and
reported separately — a failure there is information, not an error.

    python scripts/check_imports.py

Exit codes:
    0  — every core module imported
    1  — at least one core module failed
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

# Import-time config validation and network/model probes are not the point of
# this check, so give the modules the minimum they need to load.
os.environ.setdefault("DISCORD_BOT_TOKEN", "check-imports-placeholder")
os.environ.setdefault("ROOT_USER", "discord-0")
os.environ.setdefault("CHAT_COOKIE_SECRET", "check-imports-placeholder")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORE = [
    "admin_panel",
    "agent_tools",
    "bot_adapters",
    "bot_commands",
    "cards",
    "chat_auth",
    "document_engine",
    "file_tools",
    "fritz_utils",
    "identity_store",
    "main_discord",
    "migrate_db",
    "migrate_identity",
    "mister_fritz",
    "observability",
    "prewarm",
    "privacy",
    "scheduler",
    "storage",
    "workspace_store",
]

# module -> the extra that provides it
OPTIONAL = {
    "image_generator": "image",
    "tts": "voice",
    "stt": "voice",
    "main_telegram": "telegram",
}


def _try(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=3).strip().splitlines()[-1]


def main() -> int:
    failures = []
    print(f"Core modules ({len(CORE)}):")
    for name in CORE:
        ok, err = _try(name)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {err}"))
        if not ok:
            failures.append((name, err))

    print(f"\nOptional modules ({len(OPTIONAL)}) - a failure here is expected in a core-only install:")
    for name, extra in sorted(OPTIONAL.items()):
        ok, err = _try(name)
        status = "ok  " if ok else f"n/a  (needs [{extra}])"
        print(f"  {status}  {name}" + ("" if ok else f"  -- {err}"))

    if failures:
        print(f"\n{len(failures)} core module(s) failed to import.")
        return 1
    print(f"\nAll {len(CORE)} core modules imported cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
