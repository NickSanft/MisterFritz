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
# Our side could not deliver: a Discord outage, a network error, a bug. Not the
# sender's doing, so it costs them nothing.
STATUS_FAILED = "failed"
# The recipient's side would not take it: they blocked the sender, blocked
# everyone, or have their Discord DMs closed. One status for all three, written
# identically by both paths — see _REFUSED_ERROR.
STATUS_REFUSED = "refused"
# Discord refused the CONTENT (its harmful-link filter, a malformed body). The
# sender's doing, so the sender's charge — otherwise a body Discord reliably
# blocks is an unlimited, free supply of DM-channel opens on the bot's token.
# Never charged to the recipient: nothing reached them.
STATUS_REJECTED = "rejected"

# The `error` stored on every refused row, whichever path refused it. The
# Discord error code for a closed-DM refusal goes to the log line, never here:
# PR 6's /export shows a sender their own rows, and a row that said "50007"
# where another said nothing would tell them which of their refusals was a
# block.
_REFUSED_ERROR = "recipient_refused"

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
    """Nothing should be sent.

    `reason` is for logs and tests; `message` is what the sender may be shown.

    `relay_id` is set on a recipient-side refusal (a block), and ONLY there:
    that refusal leaves a `reserved` row open, exactly as a send that Discord
    is about to refuse with a 403 does, and the caller must settle it with
    mark_refused at the moment it would have settled a 403. See reserve_send.
    Every other denial writes nothing.
    """

    ok = False
    reason: str
    message: str
    retry_at: str | None = None
    relay_id: str | None = None


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

def _count_and_oldest(conn, column: str, user_id: str, window_start: str,
                      stale_before: str, *, sender_side: bool) -> tuple[int, str | None]:
    """Messages charged to `user_id` in the window, and the oldest one's time.

    Charged means delivered, or reserved recently enough to still be in
    flight — plus, on the SENDER's side only, refused and rejected.
    Deliberately excludes:

    - `failed` rows: nothing reached the recipient for reasons on our side, so
      charging the sender would be a penalty for someone else's outage;
    - `reserved` rows older than the grace window: those are crashes, and a
      crash that permanently consumed quota would be unrecoverable without
      manual DB surgery;
    - `refused` rows on the RECIPIENT's side: nothing reached them, and
      counting it would let a blocked harasser exhaust their inbound cap and
      lock out everyone else trying to reach them.

    Why refused rows charge the sender at all: a refusal from a block costs
    nothing to discover, but a refusal from closed DMs costs a real Discord
    call that ends in a 403 — the pattern Discord's anti-spam reads as abuse,
    with enforcement landing on the bot's token. And the two must be charged
    identically, or the rate limit itself becomes the block detector: ten
    free attempts at one person and ten charged attempts at another would
    tell the sender which of them had blocked them.

    The oldest charged row is what the retry time is computed from — it is the
    first one that will fall out of the window.
    """
    settled = [STATUS_DELIVERED] + ([STATUS_REFUSED, STATUS_REJECTED] if sender_side else [])
    marks = ", ".join("?" * len(settled))
    row = conn.execute(
        f"""SELECT COUNT(*), MIN(created_at) FROM relay_messages
            WHERE {column} = ? AND created_at >= ?
              AND (status IN ({marks}) OR (status = ? AND created_at >= ?))""",
        (user_id, window_start, *settled, STATUS_RESERVED, stale_before),
    ).fetchone()
    return (row[0] or 0), row[1]


def _rate_denial(reason: str, oldest: str | None, who: str, *,
                 reveal_time: bool = True) -> Denial:
    """Rate-limit refusals say when to come back — when that is theirs to know.

    For the sender's own cap the oldest charged row is the sender's own, so
    its time leaks nothing. For the recipient's cap it is SOMEONE ELSE's
    message to that person: "try again after 15:05" says a third party
    relayed to them at 14:05, and a sender who counts their own messages
    learns how many others did. That denial says only "later".
    """
    if not reveal_time:
        return Denial(reason=reason, message=f"{who} Do try again later.")
    retry_at = None
    if oldest:
        try:
            retry_at = _iso(datetime.fromisoformat(oldest) + timedelta(hours=1))
        except ValueError:  # pragma: no cover - a hand-edited row
            retry_at = None
    when = f" Try again after {retry_at[11:16]} UTC." if retry_at else ""
    return Denial(reason=reason, message=f"{who}{when}", retry_at=retry_at)


_RECIPIENT_REFUSALS = frozenset({"blocked_sender", "blocked_everyone"})


def _gate(conn, sender_id: str, recipient_id: str, body: str,
          window_start: str, stale_before: str) -> Denial | None:
    """The checks that need the database, in order. None means "allowed".

    Runs inside the caller's BEGIN IMMEDIATE, which is why it takes a
    connection rather than opening its own: counting in one transaction and
    inserting in another is what makes the caps advisory.

    THE ORDER IS THE PRIVACY PROPERTY. Every check that does not depend on
    the recipient's choices comes first; the blocks come LAST, immediately
    before the send. Then REFUSED always means "everything else was fine and
    the recipient's side declined" — exactly what Discord's own 403 for
    closed DMs means, since that can only be discovered at send time.

    The plan specified the opposite ("most recipient-protective first") and
    PR 1 shipped it. It was a free block detector: a 5,000-character message
    came back REFUSED if the recipient had blocked you and "too long" if not
    — definitive, because closed DMs are never reached for an oversized body,
    and costing nothing, because a denial writes no row and spends no quota.
    Being over your own hourly cap did the same. Do not move the blocks up.
    """
    if len(body) > fritz_utils.RELAY_MAX_BODY_CHARS:
        return Denial(
            "body_too_long",
            f"That message is {len(body)} characters; the limit is "
            f"{fritz_utils.RELAY_MAX_BODY_CHARS}.",
        )

    sent, oldest = _count_and_oldest(
        conn, "sender_id", sender_id, window_start, stale_before,
        sender_side=True)
    if sent >= fritz_utils.RELAY_MAX_PER_SENDER_PER_HOUR:
        return _rate_denial(
            "sender_rate_limited", oldest,
            f"You have tried to relay {sent} messages in the past hour, which "
            f"is the limit.")

    # The only cap that constrains a brigade: many senders, one recipient.
    # Before the blocks, for the same reason as everything else here: after
    # them, "they have had enough messages" would only ever reach senders
    # who were NOT blocked. It does tell a blocked sender roughly how busy
    # their target's inbox is — a smaller leak than the block itself, and one
    # any unblocked sender can already see.
    received, oldest = _count_and_oldest(
        conn, "recipient_id", recipient_id, window_start, stale_before,
        sender_side=False)
    if received >= fritz_utils.RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR:
        return _rate_denial(
            "recipient_rate_limited", oldest,
            "They have received as many relayed messages this hour as I will "
            "pass on.", reveal_time=False)

    blocks = {r[0] for r in conn.execute(
        "SELECT blocked_id FROM relay_optouts WHERE user_id = ? "
        "AND blocked_id IN (?, ?)",
        (recipient_id, sender_id, BLOCK_EVERYONE),
    ).fetchall()}
    if sender_id in blocks:
        return Denial("blocked_sender", REFUSED)
    if BLOCK_EVERYONE in blocks:
        return Denial("blocked_everyone", REFUSED)

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
    `mark_refused` / `mark_rejected` / `mark_failed`). Until one of those
    lands the row holds quota, which is released automatically after
    RESERVATION_GRACE_SEC.

    A recipient-side refusal ALSO returns with a reservation open, carried on
    `Denial.relay_id`. The caller owes it the same treatment a 403 gets: hold
    it until the same deadline, then mark_refused, then answer. Anything
    quicker makes the block observable again (see the comment at the INSERT).

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

    # These three are facts about the recipient that anyone can see — it is a
    # bot, it is you — so they leak nothing and can go first. Everything that
    # depends on the recipient's CHOICES goes last; see _gate.
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
            if denial is not None and denial.reason in _RECIPIENT_REFUSALS:
                # Written as an ordinary RESERVATION, exactly as a send that
                # passes the gate and then meets Discord's 403 is written, and
                # left open for the caller to settle with mark_refused on the
                # same schedule. Identical strings and identical final rows
                # were not enough: a 403-bound send sits `reserved` for its
                # whole Discord round trip, counting toward the recipient's
                # inbound cap, while a block that went straight to `refused`
                # never counted at all. A second account racing the first saw
                # "their inbox is full" only when there was no block — which
                # answered the question for free. Same lifecycle, same state,
                # at every moment another sender can observe.
                conn.execute(
                    """INSERT INTO relay_messages
                       (id, sender_id, recipient_id, origin_id, body, composed,
                        guild_id, status, created_at, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rid, sender_id, recipient_id, origin_id, body, int(composed),
                     guild_id, STATUS_RESERVED, created_at, expires_at),
                )
                conn.execute("COMMIT")
                return Denial(denial.reason, denial.message, relay_id=rid)
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


def mark_refused(relay_id: str) -> None:
    """Record that the recipient's side refused the DM (Discord 403).

    Leaves the row byte-for-byte as reserve_send leaves one refused by a
    block: status, error and closed_at == created_at. Takes no reason on
    purpose — the Discord code belongs in the caller's log line, and anything
    written here would distinguish the two refusals on a later /export.
    Still charged to the sender, like a block; see _count_and_oldest.
    """
    if not relay_id:
        raise ValueError("relay_id is required")
    _init_db()
    try:
        with sqlite3.connect(SCHEDULE_DB) as conn:
            cur = conn.execute(
                "UPDATE relay_messages SET status = ?, error = ?, "
                "closed_at = created_at WHERE id = ?",
                (STATUS_REFUSED, _REFUSED_ERROR, relay_id),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise RelayStoreError(f"could not mark relay {relay_id} refused: {e}") from e
    if cur.rowcount == 0:
        raise RelayStoreError(f"no relay message with id {relay_id}")


def mark_rejected(relay_id: str, reason: str) -> None:
    """Record that Discord refused the message's content (a 400).

    Charged to the sender, never to the recipient; see STATUS_REJECTED.
    """
    if not relay_id:
        raise ValueError("relay_id is required")
    _init_db()
    try:
        with sqlite3.connect(SCHEDULE_DB) as conn:
            cur = conn.execute(
                "UPDATE relay_messages SET status = ?, error = ?, closed_at = ? "
                "WHERE id = ?",
                (STATUS_REJECTED, str(reason)[:500], _iso(_now()), relay_id),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise RelayStoreError(f"could not mark relay {relay_id} rejected: {e}") from e
    if cur.rowcount == 0:
        raise RelayStoreError(f"no relay message with id {relay_id}")


def mark_failed(relay_id: str, reason: str) -> None:
    """Record that the DM did not go out for reasons on OUR side.

    Releases the quota it held. For a recipient-side refusal use mark_refused
    instead — the difference is who is charged, and what /export can reveal.
    """
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
