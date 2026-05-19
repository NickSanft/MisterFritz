"""User-data deletion and export helpers for the /forget and /export commands.

Centralises every place per-user state lives so the slash-command handlers
can stay thin and the future web admin panel can call the same operations.

Each delete function is best-effort: a failure in one store still tries the
others and returns whatever it managed. Counts are reported back so the user
sees how much was actually removed.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, Optional

from fritz_utils import CHAT_DB_NAME

logger = logging.getLogger(__name__)


def _sanitise_thread_id(user_id: str) -> str:
    """Mirror mister_fritz.ask_stuff's transformation of the username into a
    LangGraph thread_id (alphanumeric only)."""
    return re.sub(r"[^a-zA-Z0-9]", "", user_id or "")


# ── Memories + profile (Chroma KV store) ─────────────────────────────────────

def forget_memories(user_id: str) -> int:
    """Drop every memory and profile entry for user_id from the Chroma store.

    Returns the number of entries removed.
    """
    if not user_id:
        return 0
    # Lazy import: ChromaStore boots embeddings on first construction, and
    # we go through the singleton so the cost is paid once per process.
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().delete_namespace((str(user_id),))
    except Exception as e:
        logger.warning("forget_memories failed for %s: %s", user_id, e)
        return 0


def export_memories(user_id: str) -> list[dict]:
    """Return every memory + profile entry for user_id, ready to JSON-serialise."""
    if not user_id:
        return []
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().export_namespace((str(user_id),))
    except Exception as e:
        logger.warning("export_memories failed for %s: %s", user_id, e)
        return []


# ── Conversation checkpoints (LangGraph SqliteSaver) ─────────────────────────

def forget_conversation(user_id: str) -> int:
    """Drop the LangGraph SqliteSaver state for this user's thread.

    Returns the total number of rows removed across the checkpoints + writes
    tables. Next message starts a fresh conversation.
    """
    if not user_id:
        return 0
    thread_id = _sanitise_thread_id(user_id)
    if not thread_id:
        return 0
    try:
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            cur1 = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,),
            )
            cur2 = conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,),
            )
            conn.commit()
            return cur1.rowcount + cur2.rowcount
    except sqlite3.OperationalError as e:
        # Tables may not exist yet on a fresh deployment — that's fine.
        logger.debug("forget_conversation: no checkpoint tables yet (%s)", e)
        return 0
    except Exception as e:
        logger.warning("forget_conversation failed for %s: %s", user_id, e)
        return 0


def count_conversation_checkpoints(user_id: str) -> int:
    """Return the number of checkpoint rows for user_id. Used by /export."""
    if not user_id:
        return 0
    thread_id = _sanitise_thread_id(user_id)
    if not thread_id:
        return 0
    try:
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,),
            ).fetchone()
            return count
    except Exception:
        return 0


# ── Schedules (APScheduler-backed) ───────────────────────────────────────────

def forget_schedules(user_id: str, schedule_manager: Any) -> int:
    """Bulk-delete every schedule belonging to user_id."""
    if not user_id or schedule_manager is None:
        return 0
    try:
        return schedule_manager.remove_all_for_user(user_id)
    except Exception as e:
        logger.warning("forget_schedules failed for %s: %s", user_id, e)
        return 0


def export_schedules(user_id: str, schedule_manager: Any) -> list[dict]:
    """Return user's schedules in a JSON-friendly shape."""
    if not user_id or schedule_manager is None:
        return []
    try:
        return schedule_manager.list_schedules(user_id)
    except Exception as e:
        logger.warning("export_schedules failed for %s: %s", user_id, e)
        return []


# ── Workspace registration ──────────────────────────────────────────────────

def forget_workspace(user_id: str) -> bool:
    """Drop the user's workspace registration. Files on disk are kept."""
    if not user_id:
        return False
    import workspace_store
    try:
        return workspace_store.remove(user_id)
    except Exception as e:
        logger.warning("forget_workspace failed for %s: %s", user_id, e)
        return False


def get_workspace_for_export(user_id: str) -> Optional[str]:
    if not user_id:
        return None
    import workspace_store
    try:
        return workspace_store.get(user_id)
    except Exception:
        return None


# ── Aggregates ───────────────────────────────────────────────────────────────

def forget_all(user_id: str, schedule_manager: Any = None) -> dict:
    """Run every forget_* op and report counts back. Best-effort — partial
    failure in one store does not abort the others."""
    return {
        "memories": forget_memories(user_id),
        "conversation_rows": forget_conversation(user_id),
        "schedules": forget_schedules(user_id, schedule_manager),
        "workspace_dropped": forget_workspace(user_id),
    }


def export_user_data(user_id: str, schedule_manager: Any = None) -> dict:
    """Return a JSON-serialisable snapshot of everything we have on this user."""
    return {
        "user_id": user_id,
        "memories": export_memories(user_id),
        "schedules": export_schedules(user_id, schedule_manager),
        "conversation_checkpoint_count": count_conversation_checkpoints(user_id),
        "workspace_path": get_workspace_for_export(user_id),
    }
