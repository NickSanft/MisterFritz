"""Persistence and the admission gate for the direct-message relay.

`/tell` carries an attributed message from one guild member to another by DM,
and the recipient's reply routes back through the same path. This module owns
both halves of the state that makes that safe: the delivery log that a reply
is routed by, and the opt-outs that decide whether a message may be sent at
all. See plans/12-direct-message-relay.md.

Nothing here imports discord. The gate is a pure function of the database and
the config, so it can be tested exhaustively without a gateway connection —
which matters, because the caps are the only thing standing between this
feature and a DM-spam primitive.

Schema lives in the existing fritz.db SQLite (re-uses SCHEDULE_DB, which is
just the unified DB path), following workspace_store.py.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import fritz_utils
from fritz_utils import SCHEDULE_DB, resolve_identity

logger = logging.getLogger(__name__)

# Guards table creation. Connection-level locking is fine for everything else
# since we open a fresh connection per operation.
_INIT_LOCK = threading.Lock()
_INITIALISED = False

# Stored in relay_optouts.blocked_id to mean "nobody may reach me".
#
# A sentinel rather than a nullable column on purpose. SQLite treats NULLs as
# distinct in a unique index, so a nullable blocked_id would let the same
# person insert the global opt-out repeatedly and would make the primary key a
# guard that guards nothing. The sentinel is provably collision-free: a real id
# can never be "*", because canonical_user_id('discord', '*') raises ValueError
# — _IDENT_STRIP_RE reduces "*" to the empty string.
BLOCK_EVERYONE = "*"

# How long a reservation may sit un-marked before counting ignores it and the
# reconciliation sweep (PR 6) may close it. Covers the window between the row
# going in and the DM round trip coming back; anything longer is a crash, and a
# crash must not permanently consume the sender's quota.
RESERVATION_GRACE_SEC = 60

# Statuses. The column default is 'pending' (the schema is pinned by the plan),
# but no code path relies on it: every INSERT names its status explicitly.
STATUS_RESERVED = "reserved"
STATUS_DELIVERED = "delivered"
STATUS_FAILED = "failed"

# The one string every recipient-side refusal returns, verbatim.
#
# A blanket block, a block on this specific sender, and Discord refusing the DM
# outright (Forbidden / 50007) must be indistinguishable. If they differ by so
# much as a word, /tell becomes a reliable detector for "has X blocked me",
# which is precisely what an opt-out exists to avoid revealing. Every clause
# below is therefore true in all three cases. PR 3 pins this with a test
# asserting the two reply strings are equal, rather than two separate literals.
REFUSED = (
    "I couldn't deliver that message. They may not be accepting relayed "
    "messages at the moment."
)


class RelayStoreError(RuntimeError):
    """A relay write did not happen.

    Raised rather than logged-and-swallowed (identity_store.py:76-77 does the
    latter, and it is a trap here): a swallowed write means telling the sender
    "delivered" when there is no row for the reply to route back through.
    """


@dataclass(frozen=True)
class Reservation:
    """Quota has been taken and a row exists. Send the DM, then mark it."""

    ok = True
    id: str
    sender_id: str
    recipient_id: str
    created_at: str
    expires_at: str


@dataclass(frozen=True)
class Denial:
    """Nothing was written and nothing should be sent.

    `reason` is for logs and tests; `message` is what the sender may be shown.
    """

    ok = False
    reason: str
    message: str
    retry_at: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    """ISO-8601 UTC, the repo's stored-timestamp format.

    Always produced by this one helper so the strings stay lexicographically
    comparable — which is what lets the windows below be plain SQL `>=`.
    """
    return when.isoformat()


def _init_db() -> None:
    global _INITIALISED
    with _INIT_LOCK:
        if _INITIALISED:
            return
        with sqlite3.connect(SCHEDULE_DB) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relay_messages (
                    id             TEXT PRIMARY KEY,
                    sender_id      TEXT NOT NULL,
                    recipient_id   TEXT NOT NULL,
                    origin_id      TEXT,
                    body           TEXT NOT NULL,
                    composed       INTEGER NOT NULL DEFAULT 0,
                    guild_id       INTEGER,
                    dm_channel_id  INTEGER,
                    dm_message_id  INTEGER,
                    status         TEXT NOT NULL DEFAULT 'pending',
                    error          TEXT,
                    created_at     TEXT NOT NULL,
                    delivered_at   TEXT,
                    expires_at     TEXT NOT NULL,
                    closed_at      TEXT
                )
            """)
            # Partial, and load-bearing. dm_message_id is NULL while a row is
            # in flight (many sit there at once) and unique once delivered,
            # because a duplicate would route a reply to the wrong person.
            # SQLite counts NULLs as distinct in a unique index, which is
            # exactly the behaviour needed.
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_relay_dm_message
                    ON relay_messages(dm_message_id) WHERE dm_message_id IS NOT NULL
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relay_sender "
                "ON relay_messages(sender_id, created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relay_recipient "
                "ON relay_messages(recipient_id, created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relay_open "
                "ON relay_messages(status, expires_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relay_optouts (
                    user_id     TEXT NOT NULL,
                    blocked_id  TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    PRIMARY KEY (user_id, blocked_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relay_optouts_blocked "
                "ON relay_optouts(blocked_id)")
            conn.commit()
        _INITIALISED = True


# ---------------------------------------------------------------------------
# Opt-outs
# ---------------------------------------------------------------------------

def block(user_id: str, blocked_id: str = BLOCK_EVERYONE) -> bool:
    """Refuse relayed messages from `blocked_id` (default: from everyone).

    Returns True if this added a block, False if it was already in place.
    Idempotent, because the button that calls it can be pressed twice.
    """
    if not user_id:
        raise ValueError("user_id is required")
    # Both ends resolved, for the same reason reserve_send resolves both: a
    # block placed as discord-123 must bind when the sender arrives as
    # web-alice, and vice versa.
    user_id = resolve_identity(user_id)
    blocked_id = (blocked_id if blocked_id == BLOCK_EVERYONE
                  else resolve_identity(blocked_id))
    if not blocked_id:
        raise ValueError("blocked_id is required")
    if user_id == blocked_id:
        raise ValueError("cannot block yourself")
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO relay_optouts (user_id, blocked_id, created_at) "
            "VALUES (?, ?, ?)",
            (user_id, blocked_id, _iso(_now())),
        )
        conn.commit()
    return cur.rowcount > 0


def unblock(user_id: str, blocked_id: str = BLOCK_EVERYONE) -> bool:
    """Drop a block. Returns True if one was actually removed."""
    if not user_id:
        raise ValueError("user_id is required")
    user_id = resolve_identity(user_id)
    blocked_id = (blocked_id if blocked_id == BLOCK_EVERYONE
                  else resolve_identity(blocked_id))
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        cur = conn.execute(
            "DELETE FROM relay_optouts WHERE user_id = ? AND blocked_id = ?",
            (user_id, blocked_id),
        )
        conn.commit()
    return cur.rowcount > 0


def list_blocks(user_id: str) -> list[str]:
    """Every id this user refuses, oldest first. BLOCK_EVERYONE may be among them."""
    if not user_id:
        return []
    user_id = resolve_identity(user_id)
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        rows = conn.execute(
            "SELECT blocked_id FROM relay_optouts WHERE user_id = ? "
            "ORDER BY created_at, blocked_id",
            (user_id,),
        ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# The admission gate
# ---------------------------------------------------------------------------

def _count_and_oldest(conn, column: str, user_id: str,
                      window_start: str, stale_before: str) -> tuple[int, str | None]:
    """Messages charged to `user_id` in the window, and the oldest one's time.

    Charged means delivered, or reserved recently enough to still be in
    flight. Deliberately excludes:

    - `failed` rows: nothing reached the recipient, so charging the sender for
      Discord being down would be a penalty for someone else's outage;
    - `reserved` rows older than the grace window: those are crashes, and a
      crash that permanently consumed quota would be unrecoverable without
      manual DB surgery.

    The oldest charged row is what the retry time is computed from — it is the
    first one that will fall out of the window.
    """
    row = conn.execute(
        f"""SELECT COUNT(*), MIN(created_at) FROM relay_messages
            WHERE {column} = ? AND created_at >= ?
              AND (status = ? OR (status = ? AND created_at >= ?))""",
        (user_id, window_start, STATUS_DELIVERED, STATUS_RESERVED, stale_before),
    ).fetchone()
    return (row[0] or 0), row[1]


def _rate_denial(reason: str, oldest: str | None, who: str) -> Denial:
    """Rate-limit refusals say when to come back.

    Unlike the recipient-side refusals these are explicit, because they are a
    fact about the sender's own behaviour. There is nothing to leak.
    """
    retry_at = None
    if oldest:
        try:
            retry_at = _iso(datetime.fromisoformat(oldest) + timedelta(hours=1))
        except ValueError:  # pragma: no cover - a hand-edited row
            retry_at = None
    when = f" Try again after {retry_at[11:16]} UTC." if retry_at else ""
    return Denial(reason=reason, message=f"{who}{when}", retry_at=retry_at)


def _gate(conn, sender_id: str, recipient_id: str, body: str,
          window_start: str, stale_before: str) -> Denial | None:
    """The checks that need the database, in order. None means "allowed".

    Runs inside the caller's BEGIN IMMEDIATE, which is why it takes a
    connection rather than opening its own: counting in one transaction and
    inserting in another is what makes the caps advisory.
    """
    blocks = {r[0] for r in conn.execute(
        "SELECT blocked_id FROM relay_optouts WHERE user_id = ? "
        "AND blocked_id IN (?, ?)",
        (recipient_id, sender_id, BLOCK_EVERYONE),
    ).fetchall()}
    if sender_id in blocks:
        return Denial("blocked_sender", REFUSED)
    if BLOCK_EVERYONE in blocks:
        return Denial("blocked_everyone", REFUSED)

    if len(body) > fritz_utils.RELAY_MAX_BODY_CHARS:
        return Denial(
            "body_too_long",
            f"That message is {len(body)} characters; the limit is "
            f"{fritz_utils.RELAY_MAX_BODY_CHARS}.",
        )

    sent, oldest = _count_and_oldest(
        conn, "sender_id", sender_id, window_start, stale_before)
    if sent >= fritz_utils.RELAY_MAX_PER_SENDER_PER_HOUR:
        return _rate_denial(
            "sender_rate_limited", oldest,
            f"You have relayed {sent} messages in the past hour, which is "
            f"the limit.")

    # The only cap that constrains a brigade: many senders, one recipient.
    # Last because its denial names no one — it is the least informative
    # refusal, and the one it is safest to be explicit about.
    received, oldest = _count_and_oldest(
        conn, "recipient_id", recipient_id, window_start, stale_before)
    if received >= fritz_utils.RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR:
        return _rate_denial(
            "recipient_rate_limited", oldest,
            "They have received as many relayed messages this hour as I will "
            "pass on.")

    return None


def reserve_send(
    sender_id: str,
    recipient_id: str,
    body: str,
    *,
    guild_id: int | None = None,
    origin_id: str | None = None,
    composed: bool = False,
    recipient_is_bot: bool = False,
    recipient_is_fritz: bool = False,
) -> Reservation | Denial:
    """Decide whether this message may be sent, and if so take its quota.

    Every count and the reservation INSERT happen inside one BEGIN IMMEDIATE.
    That is the whole point of this function: counting in one transaction and
    inserting in another makes the caps advisory, because two concurrent
    `/tell`s both read "9 sent" and both write the tenth and eleventh.

    On success the caller sends the DM and then calls `mark_sent` (or
    `mark_failed`). Until one of those lands the row holds quota, which is
    released automatically after RESERVATION_GRACE_SEC.

    `recipient_is_bot` and `recipient_is_fritz` come from the caller, because
    only the Discord layer knows them. Setting both for Fritz is correct and
    expected; see the ordering note below.
    """
    if not sender_id or not recipient_id:
        raise ValueError("sender_id and recipient_id are required")
    if body is None:
        raise ValueError("body is required")

    # Resolved at BOTH ends, here, inside the store.
    #
    # With IDENTITY_LINKS=web-alice=discord-123, a block stored against
    # discord-123 must be honoured when the sender arrives as web-alice.
    # Resolving only the sender leaves a live bypass — the same write-path /
    # delete-path divergence the comment in privacy.py:44-49 documents.
    sender_id = resolve_identity(sender_id)
    recipient_id = resolve_identity(recipient_id)

    if not fritz_utils.RELAY_ENABLED:
        return Denial("disabled", "The message relay is switched off.")

    # Order is most recipient-protective first, and the recipient-side
    # refusals all return the same string. See REFUSED.
    #
    # Fritz is checked before the generic bot branch, which inverts the two as
    # the plan writes them. Deliberate, and please don't put it back: Fritz IS
    # a bot, so the generic branch would shadow his own reply every time and
    # leave it permanently dead. Ordering them this way also means a caller
    # that sets both flags — the honest thing to do — still gets the better
    # copy. The order matters for nothing else here: all three are refusals
    # with no side effects, so the only thing at stake is which sentence the
    # sender reads.
    if recipient_is_fritz:
        return Denial(
            "recipient_is_fritz",
            "I am already here, and reading. Just say it to me directly.",
        )
    if recipient_is_bot:
        return Denial("recipient_is_bot", "Bots don't read their messages.")
    if recipient_id == sender_id:
        return Denial("recipient_is_sender", "You are already talking to yourself.")

    now = _now()
    window_start = _iso(now - timedelta(hours=1))
    stale_before = _iso(now - timedelta(seconds=RESERVATION_GRACE_SEC))

    _init_db()
    # isolation_level=None turns off the driver's implicit transaction
    # handling so BEGIN IMMEDIATE is ours to place. IMMEDIATE (not DEFERRED)
    # takes the write lock up front: a deferred transaction would upgrade only
    # at the INSERT, which is after the counts have already been read, and is
    # exactly the race this transaction exists to close.
    rid = str(uuid.uuid4())[:8]
    created_at = _iso(now)
    expires_at = _iso(now + timedelta(minutes=fritz_utils.RELAY_REPLY_WINDOW_MIN))

    conn = sqlite3.connect(SCHEDULE_DB, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            denial = _gate(conn, sender_id, recipient_id, body,
                           window_start, stale_before)
            if denial is not None:
                conn.execute("ROLLBACK")
                return denial
            conn.execute(
                """INSERT INTO relay_messages
                   (id, sender_id, recipient_id, origin_id, body, composed,
                    guild_id, status, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rid, sender_id, recipient_id, origin_id, body, int(composed),
                 guild_id, STATUS_RESERVED, created_at, expires_at),
            )
            conn.execute("COMMIT")
        except BaseException:
            # Explicit, rather than relying on close() to roll back: an open
            # IMMEDIATE transaction holds the write lock, and every other
            # relay caller is blocked behind it until it goes.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - already unwinding
                pass
            raise
    except sqlite3.Error as e:
        raise RelayStoreError(f"could not reserve a relay message: {e}") from e
    finally:
        conn.close()

    return Reservation(id=rid, sender_id=sender_id, recipient_id=recipient_id,
                       created_at=created_at, expires_at=expires_at)


def mark_sent(relay_id: str, dm_message_id: int, dm_channel_id: int | None = None) -> None:
    """Record that the DM went out, and bind the id a reply will arrive on.

    `dm_message_id` is the reply-routing key, so a collision here would route
    someone's reply to the wrong person. The partial unique index turns that
    into an IntegrityError, which is raised, not swallowed.
    """
    if not relay_id:
        raise ValueError("relay_id is required")
    if not dm_message_id:
        raise ValueError("dm_message_id is required")
    _init_db()
    try:
        with sqlite3.connect(SCHEDULE_DB) as conn:
            cur = conn.execute(
                "UPDATE relay_messages SET status = ?, dm_message_id = ?, "
                "dm_channel_id = ?, delivered_at = ? WHERE id = ?",
                (STATUS_DELIVERED, dm_message_id, dm_channel_id,
                 _iso(_now()), relay_id),
            )
            conn.commit()
    except sqlite3.IntegrityError as e:
        raise RelayStoreError(
            f"dm_message_id {dm_message_id} is already bound to another relay "
            f"message; refusing to make it ambiguous: {e}") from e
    except sqlite3.Error as e:
        raise RelayStoreError(f"could not mark relay {relay_id} sent: {e}") from e
    if cur.rowcount == 0:
        raise RelayStoreError(f"no relay message with id {relay_id}")


def mark_failed(relay_id: str, reason: str) -> None:
    """Record that the DM did not go out. Releases the quota it held."""
    if not relay_id:
        raise ValueError("relay_id is required")
    _init_db()
    try:
        with sqlite3.connect(SCHEDULE_DB) as conn:
            cur = conn.execute(
                "UPDATE relay_messages SET status = ?, error = ?, closed_at = ? "
                "WHERE id = ?",
                (STATUS_FAILED, str(reason)[:500], _iso(_now()), relay_id),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise RelayStoreError(f"could not mark relay {relay_id} failed: {e}") from e
    if cur.rowcount == 0:
        raise RelayStoreError(f"no relay message with id {relay_id}")


def get(relay_id: str) -> dict | None:
    """One relay row as a dict, or None. Read-only; used by tests and PR 4."""
    if not relay_id:
        return None
    _init_db()
    with sqlite3.connect(SCHEDULE_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM relay_messages WHERE id = ?", (relay_id,)).fetchone()
    return dict(row) if row else None
