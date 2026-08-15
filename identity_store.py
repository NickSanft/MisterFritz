"""Display-name map for canonical user identities.

Once `user_id` is `discord-123456789`, nothing else knows the human's name.
This table is the alias side-channel: written on every inbound message, read by
the prompt builder (so Fritz still addresses people by name), the admin panel's
user list, and the scheduler — which has no live user object when a cron job
fires hours later.

Lives in the same fritz.db as schedules and workspaces (SCHEDULE_DB), and
deliberately mirrors workspace_store.py's shape: module-level functions, a lazy
_init_db under a lock, best-effort writes.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone

from fritz_utils import SCHEDULE_DB, split_user_id

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALISED = False
# Skip the write when the name has not changed — one conversation turn should
# not cost a gratuitous SQLite round trip.
_NAME_CACHE: dict[str, str] = {}
_NAME_CACHE_LOCK = threading.Lock()


def _init_db() -> None:
    global _INITIALISED
    with _INIT_LOCK:
        if _INITIALISED:
            return
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_aliases (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    platform TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        _INITIALISED = True


def record(user_id: str, name: str | None, platform: str | None = None) -> None:
    """Upsert the human-readable name for an identity.

    Best-effort by design: a failure here must never break a conversation turn.
    """
    if not user_id or not name:
        return
    with _NAME_CACHE_LOCK:
        if _NAME_CACHE.get(user_id) == name:
            return
    try:
        _init_db()
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute(
                "INSERT INTO user_aliases (user_id, display_name, platform, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "display_name = excluded.display_name, "
                "platform = excluded.platform, "
                "updated_at = excluded.updated_at",
                (user_id, name, platform, now),
            )
            conn.commit()
        with _NAME_CACHE_LOCK:
            _NAME_CACHE[user_id] = name
    except Exception as e:
        logger.debug("identity_store.record failed for %s (non-fatal): %s", user_id, e)


def display_name(user_id: str, default: str | None = None) -> str:
    """Human name for an identity.

    Falls back to the caller's default, then to the id's bare part — so a
    prompt says "alice" or at worst "123456789", never "discord-123456789".
    """
    if not user_id:
        return default or ""
    with _NAME_CACHE_LOCK:
        cached = _NAME_CACHE.get(user_id)
    if cached:
        return cached
    try:
        _init_db()
        with sqlite3.connect(SCHEDULE_DB) as conn:
            row = conn.execute(
                "SELECT display_name FROM user_aliases WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row and row[0]:
            with _NAME_CACHE_LOCK:
                _NAME_CACHE[user_id] = row[0]
            return row[0]
    except Exception as e:
        logger.debug("identity_store.display_name failed for %s (non-fatal): %s", user_id, e)
    if default:
        return default
    _platform, bare = split_user_id(user_id)
    return bare or user_id


def forget(user_id: str) -> bool:
    """Drop the alias row for an identity. Called by /forget all — the display
    name is personal data like any other."""
    if not user_id:
        return False
    with _NAME_CACHE_LOCK:
        _NAME_CACHE.pop(user_id, None)
    try:
        _init_db()
        with sqlite3.connect(SCHEDULE_DB) as conn:
            cur = conn.execute("DELETE FROM user_aliases WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.debug("identity_store.forget failed for %s (non-fatal): %s", user_id, e)
        return False


def list_all() -> list[dict]:
    """Every known identity and display name. Used by the admin panel."""
    try:
        _init_db()
        with sqlite3.connect(SCHEDULE_DB) as conn:
            rows = conn.execute(
                "SELECT user_id, display_name, platform, updated_at "
                "FROM user_aliases ORDER BY display_name COLLATE NOCASE"
            ).fetchall()
    except Exception as e:
        logger.debug("identity_store.list_all failed (non-fatal): %s", e)
        return []
    return [
        {"user_id": r[0], "display_name": r[1], "platform": r[2], "updated_at": r[3]}
        for r in rows
    ]
