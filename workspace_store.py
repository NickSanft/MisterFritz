"""SQLite-backed store for per-user file-tool workspaces.

Each user who runs `/workspace enable` gets a row here pointing at the
directory the file tools will sandbox them to. Persists across bot restarts.

Schema lives in the existing fritz.db SQLite (re-uses SCHEDULE_DB which is
just the unified DB path).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone

from fritz_utils import SCHEDULE_DB, WORKSPACES_ROOT

logger = logging.getLogger(__name__)

# Guards table creation. Connection-level locking is fine for everything else
# since we open a fresh connection per operation.
_INIT_LOCK = threading.Lock()
_INITIALISED = False


def _init_db() -> None:
    global _INITIALISED
    with _INIT_LOCK:
        if _INITIALISED:
            return
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    user_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    enabled_at TEXT NOT NULL
                )
            """)
            conn.commit()
        _INITIALISED = True


def _safe_user_dir(user_id: str) -> str:
    """Sanitise a user_id for filesystem use. Discord usernames can include
    dots and other characters that we don't want bleeding into directory names.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", user_id) or "anonymous"


def get(user_id: str) -> str | None:
    """Return the workspace path for a user, or None if they have none."""
    if not user_id:
        return None
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        row = conn.execute(
            "SELECT path FROM workspaces WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row[0] if row else None


def set_path(user_id: str, path: str) -> None:
    """Register (or update) a workspace for a user. Path must already exist."""
    if not user_id:
        raise ValueError("user_id is required")
    if not os.path.isdir(path):
        raise ValueError(f"Workspace path does not exist: {path}")
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        conn.execute(
            "INSERT INTO workspaces (user_id, path, enabled_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET path = excluded.path, "
            "enabled_at = excluded.enabled_at",
            (user_id, path, now),
        )
        conn.commit()
    logger.info("Registered workspace for %s at %s", user_id, path)


def enable_sandboxed(user_id: str) -> str:
    """Create (if needed) WORKSPACES_ROOT/<user_id>/ and register it. Returns the path."""
    if not user_id:
        raise ValueError("user_id is required")
    safe_name = _safe_user_dir(user_id)
    target = os.path.abspath(os.path.join(WORKSPACES_ROOT, safe_name))
    os.makedirs(target, exist_ok=True)
    set_path(user_id, target)
    return target


def remove(user_id: str) -> bool:
    """Drop a user's workspace registration. Does NOT delete files on disk.

    Returns True if a row was removed, False if none existed.
    """
    if not user_id:
        return False
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        cur = conn.execute("DELETE FROM workspaces WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0


def list_all() -> list[dict]:
    """Return every registered workspace. Admin use (the upcoming web panel)."""
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        rows = conn.execute(
            "SELECT user_id, path, enabled_at FROM workspaces ORDER BY enabled_at"
        ).fetchall()
    return [
        {"user_id": uid, "path": path, "enabled_at": enabled}
        for uid, path, enabled in rows
    ]
