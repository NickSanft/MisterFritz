"""User-data deletion and export helpers for the /forget and /export commands.

Centralises every place per-user state lives so the slash-command handlers
can stay thin and the future web admin panel can call the same operations.

Every operation here RAISES PrivacyOperationFailed when a store refuses, and
returns a count only when it actually ran. It used to catch and return 0, [] or
False - which is exactly what "nothing stored" looks like, so a total failure
rendered as a clean success: "Removed 0 memory entry(ies)." for someone whose
memories were all still there.

The aggregates (forget_all, export_user_data) stay best-effort: one store
refusing does not abort the others, and they name what refused in `failed` so
the caller can say so instead of reporting zeros.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import ExitStack
from typing import Any, Optional

from fritz_utils import CHAT_DB_NAME, resolve_identity, thread_id_for

logger = logging.getLogger(__name__)


class PrivacyOperationFailed(RuntimeError):
    """A store refused to delete or export.

    Raised rather than swallowed because the neutral values these functions
    used to return — 0, [] and False — are indistinguishable from the ordinary
    "there was nothing there", and /forget is the one place in the app where
    that difference is the whole point.
    """

    def __init__(self, operation: str, cause: BaseException):
        super().__init__(f"{operation} failed: {cause}")
        self.operation = operation
        self.cause = cause


def _failed(operation: str, user_id: str, cause: BaseException) -> PrivacyOperationFailed:
    """Log in the wording these failures have always had, and build the raise."""
    logger.warning("%s failed for %s: %s", operation, user_id, cause)
    return PrivacyOperationFailed(operation, cause)


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
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return 0
    # Lazy import: ChromaStore boots embeddings on first construction, and
    # we go through the singleton so the cost is paid once per process.
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().delete_namespace((str(user_id),))
    except Exception as e:
        raise _failed("forget_memories", user_id, e) from e


def export_memories(user_id: str) -> list[dict]:
    """Return every memory + profile entry for user_id, ready to JSON-serialise.

    Had the same raw/stripped mismatch as forget_memories, which is why the
    admin panel's per-user memory counts read 0 for punctuated names.
    """
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return []
    from storage import get_default_chroma_store
    try:
        return get_default_chroma_store().export_namespace((str(user_id),))
    except Exception as e:
        raise _failed("export_memories", user_id, e) from e


# ── Conversation checkpoints (LangGraph SqliteSaver) ─────────────────────────

def _sweep_where(thread_id: str) -> tuple[str, tuple]:
    """SQL matching a thread and every per-channel thread under it.

    A literal prefix comparison, not LIKE. LIKE needed its `_` and `%`
    escaped, since identities contain `_`, and even escaped it was
    case-insensitive: SQLite's LIKE folds ASCII case. Web identities keep
    theirs, so `web-alice` and `web-Alice` are different people, and with
    THREADS_PER_CHANNEL on, forgetting one swept the other's conversation too.
    `substr(...) = ?` compares exactly, under SQLite's default BINARY collation.
    """
    prefix = thread_id + "#"
    return ("thread_id = ? OR substr(thread_id, 1, ?) = ?",
            (thread_id, len(prefix), prefix))


def forget_conversation(user_id: str, thread_id: str | None = None) -> int:
    """Drop the LangGraph SqliteSaver state for a thread.

    Defaults to the thread derived from user_id. With THREADS_PER_CHANNEL on,
    one identity owns several threads — `<identity>` plus `<identity>#<channel>`
    for every channel they have spoken in — so the identity form clears the
    whole set. Pass thread_id explicitly to target exactly one, which is what
    /chat's "New conversation" does so it cannot clear the Discord history
    instead of the one the user is looking at.

    The `#` in the suffix is what makes the sweep safe: the prefix `discord-1#`
    cannot match `discord-10#general`, because the character after `discord-1`
    must be `#`. See _sweep_where for why the match is exact and
    case-sensitive.

    Returns the total number of rows removed across the checkpoints + writes
    tables. Next message starts a fresh conversation.

    Waits for any turn in flight on those threads first. A turn loads the
    whole conversation when it starts and commits all of it back when it
    ends, so a delete that landed in between was undone a moment later: the
    forget reported success and the conversation returned. Every caller gets
    this, because every caller comes through here: Discord's /forget
    conversation and /forget all, the admin panel's forget-user, and the web
    chat's "Burn the correspondence". A turn someone cancelled ends within a
    token. One nobody cancelled, such as a Discord reply still being written,
    is waited for, and the forget answers when it has finished.
    """
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
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
        where, params = _sweep_where(thread_id)
    else:
        where = "thread_id = ?"
        params = (thread_id,)
    try:
        with _no_turn_in_flight(_threads_to_forget(thread_id, sweep_channels, where, params)):
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
        raise _failed("forget_conversation", user_id, e) from e


def _threads_to_forget(thread_id: str, sweep_channels: bool, where: str, params: tuple) -> list[str]:
    """Every thread the delete will touch, plus any with a turn running on it.

    The second half matters for a thread with no checkpoint yet, whose turn
    began before the forget: it would commit the conversation's first
    exchange after the delete. The match is the same as the SQL's: the thread
    itself, and with a sweep anything under "<thread>#".
    """
    threads = {thread_id}
    if sweep_channels:
        try:
            with sqlite3.connect(CHAT_DB_NAME) as conn:
                threads.update(row[0] for row in conn.execute(
                    f"SELECT DISTINCT thread_id FROM checkpoints WHERE {where}", params))
        except sqlite3.OperationalError:
            pass        # no checkpoint table yet, so no rows to protect
        threads.update(t for t in _brain_threads_in_flight() if t.startswith(thread_id + "#"))
    return sorted(threads)


def _brain_threads_in_flight() -> list[str]:
    import mister_fritz        # imported at call time: it is the whole brain
    return mister_fritz.threads_in_flight()


def _no_turn_in_flight(threads: list[str]) -> ExitStack:
    """Hold the turn gate of every one of `threads` until the delete is done.

    Taken in sorted order, so two forgets over overlapping threads cannot each
    hold a gate the other is waiting for. A turn holds only its own gate, so
    it cannot be part of a cycle either.
    """
    import mister_fritz
    stack = ExitStack()
    try:
        for t in threads:
            stack.enter_context(mister_fritz.one_turn_at_a_time(t, may_wait=True))
    except BaseException:
        stack.close()
        raise
    return stack


def count_conversation_checkpoints(user_id: str) -> int:
    """Return the number of checkpoint rows for user_id. Used by /export."""
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return 0
    thread_id = _sanitise_thread_id(user_id)
    if not thread_id:
        return 0
    try:
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            # The same sweep as forget_conversation, so /export's count
            # matches what /forget conversation would actually remove.
            where, params = _sweep_where(thread_id)
            (count,) = conn.execute(
                f"SELECT COUNT(*) FROM checkpoints WHERE {where}", params,
            ).fetchone()
            return count
    except sqlite3.OperationalError as e:
        # No checkpoint tables yet on a fresh deployment, the same case
        # forget_conversation treats as a legitimate zero.
        logger.debug("count_conversation_checkpoints: no tables yet (%s)", e)
        return 0
    except Exception as e:
        raise _failed("count_conversation_checkpoints", user_id, e) from e


# ── Schedules (APScheduler-backed) ───────────────────────────────────────────

def forget_schedules(user_id: str, schedule_manager: Any) -> int:
    """Bulk-delete every schedule belonging to user_id."""
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id or schedule_manager is None:
        return 0
    try:
        return schedule_manager.remove_all_for_user(user_id)
    except Exception as e:
        raise _failed("forget_schedules", user_id, e) from e


def export_schedules(user_id: str, schedule_manager: Any) -> list[dict]:
    """Return user's schedules in a JSON-friendly shape."""
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id or schedule_manager is None:
        return []
    try:
        return schedule_manager.list_schedules(user_id)
    except Exception as e:
        raise _failed("export_schedules", user_id, e) from e


# ── Workspace registration ──────────────────────────────────────────────────

def forget_workspace(user_id: str) -> bool:
    """Drop the user's workspace registration. Files on disk are kept."""
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return False
    import workspace_store
    try:
        return workspace_store.remove(user_id)
    except Exception as e:
        raise _failed("forget_workspace", user_id, e) from e


def get_workspace_for_export(user_id: str) -> Optional[str]:
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return None
    import workspace_store
    try:
        return workspace_store.get(user_id)
    except Exception as e:
        raise _failed("get_workspace_for_export", user_id, e) from e


# ── Aggregates ───────────────────────────────────────────────────────────────

def forget_relay(user_id: str) -> int:
    """Forget every relayed message this person sent or received.

    Messages still inside the rate window are scrubbed rather than deleted,
    so forgetting cannot reset a cap; see relay_store.forget_relay. Blocks are
    left alone in both directions (DECISIONS #22): those placed
    against this person are other people's, and those they placed are dropped
    only by /relay forget-blocks, so /forget all never re-arms a harasser.
    """
    # NOT resolved here, unlike its siblings: relay_store resolves exactly
    # once, as it did when the rows were written, and also matches the
    # account as given. Resolving here as well followed a second hop under
    # chained IDENTITY_LINKS (A->B, B->C) and acted on C's relays instead of
    # B's — A's own messages untouched and someone else's gone.
    if not user_id:
        return 0
    import relay_store
    try:
        return relay_store.forget_relay(user_id)
    except Exception as e:
        raise _failed("forget_relay", user_id, e) from e


def export_relay(user_id: str) -> dict:
    """This person's relayed messages, both ways, and the blocks they placed."""
    # NOT resolved here: relay_store resolves once; see forget_relay.
    empty = {"sent": [], "received": [], "blocks": []}
    if not user_id:
        return empty
    import relay_store
    try:
        return relay_store.export_relay(user_id)
    except Exception as e:
        raise _failed("export_relay", user_id, e) from e
        return empty


def forget_all(user_id: str, schedule_manager: Any = None) -> dict:
    """Run every forget_* op and report counts back.

    Still best-effort — one store refusing does not abort the others — but a
    refusal is now named in `failed` instead of being counted as zero. The
    neutral value is kept beside it so the report stays renderable; what stops
    that zero being read as "there was nothing there" is `failed`.
    """
    # NOT resolved here. Every operation below resolves at its own entry
    # point, so resolving here as well was a second hop through
    # IDENTITY_LINKS: under chained links (A->B, B->C), /forget all from A
    # deleted C's memories, conversation, schedules and workspace.
    operations = (
        ("memories", 0, lambda: forget_memories(user_id)),
        ("conversation_rows", 0, lambda: forget_conversation(user_id)),
        ("schedules", 0, lambda: forget_schedules(user_id, schedule_manager)),
        ("workspace_dropped", False, lambda: forget_workspace(user_id)),
        # The display-name alias is personal data too — leaving it behind would
        # mean "forget me" still knows what you are called.
        ("alias_dropped", False, lambda: forget_alias(user_id)),
        # Messages this person sent or received through the relay. NOT their
        # relay blocks: see forget_relay.
        ("relays", 0, lambda: forget_relay(user_id)),
    )
    result: dict = {}
    failed: list[str] = []
    for key, neutral, operation in operations:
        try:
            result[key] = operation()
        except PrivacyOperationFailed as e:
            result[key] = neutral
            failed.append(e.operation)
    result["failed"] = failed
    return result


def forget_alias(user_id: str) -> bool:
    """Drop the identity → display-name mapping."""
    # Resolved HERE, at the entry point. It used to happen inside two of
    # these functions and nowhere else, so an IDENTITY_LINKS alias was
    # honoured when forgetting memories but ignored when forgetting the
    # conversation, schedules or workspace — a /forget that reported
    # success while leaving most of the person behind.
    user_id = resolve_identity(user_id) if user_id else user_id
    if not user_id:
        return False
    import identity_store
    try:
        return identity_store.forget(user_id)
    except Exception as e:
        raise _failed("forget_alias", user_id, e) from e


def export_user_data(user_id: str, schedule_manager: Any = None) -> dict:
    """Return a JSON-serialisable snapshot of everything we have on this user.

    `failed` lists any section that could not be read. It is empty on a
    complete export, and a caller that ignores it is telling somebody that a
    section they hold is absent.
    """
    # NOT resolved here. Every operation below resolves at its own entry
    # point, so resolving here as well was a second hop through
    # IDENTITY_LINKS: under chained links (A->B, B->C), /forget all from A
    # deleted C's memories, conversation, schedules and workspace.
    resolved = resolve_identity(user_id) if user_id else user_id
    import identity_store
    sections = (
        ("memories", [], lambda: export_memories(user_id)),
        ("schedules", [], lambda: export_schedules(user_id, schedule_manager)),
        ("conversation_checkpoint_count", 0,
         lambda: count_conversation_checkpoints(user_id)),
        ("workspace_path", None, lambda: get_workspace_for_export(user_id)),
        ("relays", {}, lambda: export_relay(user_id)),
    )
    data: dict = {
        "user_id": resolved,
        "display_name": identity_store.display_name(resolved, default=""),
    }
    failed: list[str] = []
    for key, neutral, section in sections:
        try:
            data[key] = section()
        except PrivacyOperationFailed as e:
            # An export that quietly omits a section is worse than a partial
            # one that says which section is missing: somebody checking what
            # is held about them would conclude it is not held.
            data[key] = neutral
            failed.append(e.operation)
    data["failed"] = failed
    return data
