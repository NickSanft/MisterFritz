"""User-data deletion and export helpers for the /forget and /export commands.

Centralises every place per-user state lives so the slash-command handlers
can stay thin and the future web admin panel can call the same operations.

Each delete function is best-effort: a failure in one store still tries the
others and returns whatever it managed. Counts are reported back so the user
sees how much was actually removed.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

from fritz_utils import CHAT_DB_NAME, resolve_identity, thread_id_for

logger = logging.getLogger(__name__)


def _sanitise_thread_id(user_id: str) -> str:
    """Thread id for an identity.

    Was the second of four divergent copies of the id-stripping regex; it now
    delegates to the one in fritz_utils. ask_stuff no longer transforms the id
    at all, so neither may this — matching its old alnum-only stripping would
    now MISS the thread it is trying to delete.
    """
    return thread_id_for(user_id) if user_id else ""


# ── Memories + profile (Chroma KV store) ─────────────────────────────────────

def forget_memories(user_id: str) -> int:
    """Drop every memory and profile entry for user_id from the Chroma store.

    THE /forget BUG. This deleted the namespace for the RAW id while
    agent_tools.add_memory wrote to the STRIPPED one (ask_stuff used to strip
    the id before putting it in metadata). For any username containing
    punctuation the two never matched, so /forget memories reported success and
    deleted nothing. The fix was to stop transforming the id at all — both
    sides now use the identity verbatim.
    """
    if not user_id:
        return 0
    # Lazy import: ChromaStore boots embeddings on first construction, and
    # we go through the singleton so the cost is paid once per process.
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().delete_namespace((str(resolve_identity(user_id)),))
    except Exception as e:
        logger.warning("forget_memories failed for %s: %s", user_id, e)
        return 0


def export_memories(user_id: str) -> list[dict]:
    """Return every memory + profile entry for user_id, ready to JSON-serialise.

    Had the same raw/stripped mismatch as forget_memories, which is why the
    admin panel's per-user memory counts read 0 for punctuated names.
    """
    if not user_id:
        return []
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().export_namespace((str(resolve_identity(user_id)),))
    except Exception as e:
        logger.warning("export_memories failed for %s: %s", user_id, e)
        return []


# ── Conversation checkpoints (LangGraph SqliteSaver) ─────────────────────────

def _like_escape(value: str) -> str:
    """Escape LIKE wildcards. Identities legitimately contain `_`, which LIKE
    reads as "any single character" — without this, forgetting `discord-1_a`
    would also wipe `discord-1Xa`."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def forget_conversation(user_id: str, thread_id: str | None = None) -> int:
    """Drop the LangGraph SqliteSaver state for a thread.

    Defaults to the thread derived from user_id. With THREADS_PER_CHANNEL on,
    one identity owns several threads — `<identity>` plus `<identity>#<channel>`
    for every channel they have spoken in — so the identity form clears the
    whole set. Pass thread_id explicitly to target exactly one, which is what
    /chat's "New conversation" does so it cannot clear the Discord history
    instead of the one the user is looking at.

    The `#` in the suffix is what makes the sweep safe: `discord-1#%` cannot
    match `discord-10#general`, because the character after `discord-1` must
    be `#`.

    Returns the total number of rows removed across the checkpoints + writes
    tables. Next message starts a fresh conversation.
    """
    if not user_id and not thread_id:
        return 0
    sweep_channels = False
    if thread_id:
        # Used verbatim. It used to be re-stripped here, which was a fourth
        # place an id could be quietly altered — and a strip that removes one
        # character turns "delete this thread" into "delete nothing", or worse
        # "delete a different thread". The SQL is parameterised, so passing it
        # through is safe; a malformed thread id now matches zero rows, which
        # is visible, rather than silently retargeting.
        pass
    else:
        thread_id = _sanitise_thread_id(user_id)
        sweep_channels = True
    if not thread_id:
        return 0
    if sweep_channels:
        where = "thread_id = ? OR thread_id LIKE ? ESCAPE '\\'"
        params = (thread_id, _like_escape(thread_id) + "#%")
    else:
        where = "thread_id = ?"
        params = (thread_id,)
    try:
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            cur1 = conn.execute(
                f"DELETE FROM checkpoints WHERE {where}", params,
            )
            cur2 = conn.execute(
                f"DELETE FROM writes WHERE {where}", params,
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
            # Mirrors forget_conversation's sweep, so /export's count matches
            # what /forget conversation would actually remove.
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM checkpoints "
                "WHERE thread_id = ? OR thread_id LIKE ? ESCAPE '\\'",
                (thread_id, _like_escape(thread_id) + "#%"),
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
        # The display-name alias is personal data too — leaving it behind would
        # mean "forget me" still knows what you are called.
        "alias_dropped": forget_alias(user_id),
    }


def forget_alias(user_id: str) -> bool:
    """Drop the identity → display-name mapping."""
    if not user_id:
        return False
    import identity_store
    try:
        return identity_store.forget(user_id)
    except Exception as e:
        logger.warning("forget_alias failed for %s: %s", user_id, e)
        return False


def export_user_data(user_id: str, schedule_manager: Any = None) -> dict:
    """Return a JSON-serialisable snapshot of everything we have on this user."""
    import identity_store
    return {
        "user_id": user_id,
        "display_name": identity_store.display_name(user_id, default=""),
        "memories": export_memories(user_id),
        "schedules": export_schedules(user_id, schedule_manager),
        "conversation_checkpoint_count": count_conversation_checkpoints(user_id),
        "workspace_path": get_workspace_for_export(user_id),
    }
