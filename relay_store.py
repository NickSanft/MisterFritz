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

import contextlib
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

# The window every cap counts over (see _gate). forget_relay and purge_expired
# both use it: a row inside it is still holding someone's quota.
RATE_WINDOW = timedelta(hours=1)

# What forget_relay left of a row: the value of relay_messages.forgotten.
FORGOTTEN_SCRUBBED = 1   # only what the caps count; purged once out of RATE_WINDOW
FORGOTTEN_ANCHOR = 2     # that, plus what its recipient needs to block the sender

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


@contextlib.contextmanager
def _db():
    """One connection per operation, as elsewhere in the repo — but closed.

    `with sqlite3.connect(...) as conn` only commits or rolls back; it never
    closes, and the connection lingers until garbage collection. Harmless on
    CPython, noisy under warnings, and a real leak anywhere refcounting is
    not immediate.
    """
    conn = sqlite3.connect(SCHEDULE_DB)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


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
        with _db() as conn:
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
            # Display columns, added after the tables first shipped. CREATE
            # TABLE IF NOT EXISTS never alters an existing table, so they are
            # added in place. Neither is an identity: both hold only what a
            # person was shown or typed, which is the whole point of them.
            #   relay_messages.shown_as — the author line the recipient saw.
            #   relay_optouts.label     — how a block is named back to its owner.
            #   relay_messages.sender_account — the account that actually sent
            #     it, BEFORE IDENTITY_LINKS. sender_id holds the resolved
            #     identity, which is what the caps count; this is what a
            #     recipient blocks when they block "the sender of that message".
            #   relay_messages.recipient_account — the account the sender
            #     addressed, before IDENTITY_LINKS. /export shows a sender this,
            #     not recipient_id: the resolved id would tell them which main
            #     account the alt they wrote to belongs to.
            #   relay_messages.forgotten — 0, or FORGOTTEN_SCRUBBED /
            #     FORGOTTEN_ANCHOR on rows forget_relay may not delete yet.
            for table, column, decl in (("relay_messages", "shown_as", "TEXT"),
                                        ("relay_optouts", "label", "TEXT"),
                                        ("relay_messages", "sender_account", "TEXT"),
                                        ("relay_messages", "recipient_account", "TEXT"),
                                        ("relay_messages", "forgotten", "INTEGER NOT NULL DEFAULT 0")):
                existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            conn.commit()
        _INITIALISED = True


# ---------------------------------------------------------------------------
# Opt-outs
# ---------------------------------------------------------------------------

def block(user_id: str, blocked_id: str = BLOCK_EVERYONE, *,
          label: str | None = None) -> bool:
    """Refuse relayed messages from `blocked_id` (default: from everyone).

    Returns True if this added a block, False if it was already in place.
    Idempotent, because the button that calls it can be pressed twice.

    `blocked_id` is stored AS SUPPLIED, not resolved through IDENTITY_LINKS;
    _gate resolves both sides when it checks. Resolving here stored the
    linked account, and everything that later showed a block back to its
    owner — /relay status, the "already blocked" reply — then disclosed which
    Discord account a web or Telegram identity, or an alt, belongs to. The
    blocker's own id is still resolved: it is theirs.

    `label` is how the block is named back to its owner: the author line
    they were shown on the relay they blocked from, or the ID they typed.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not blocked_id:
        raise ValueError("blocked_id is required")
    user_id = resolve_identity(user_id)
    if blocked_id != BLOCK_EVERYONE and resolve_identity(blocked_id) == user_id:
        raise ValueError("cannot block yourself")
    _init_db()
    with _db() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO relay_optouts (user_id, blocked_id, created_at, label) "
            "VALUES (?, ?, ?, ?)",
            (user_id, blocked_id, _iso(_now()), label),
        )
        conn.commit()
    return cur.rowcount > 0


def unblock(user_id: str, blocked_id: str = BLOCK_EVERYONE) -> bool:
    """Drop a block, exactly as it was placed. True if one was removed.

    Matched as supplied, like block(): resolving here would let "You had no
    block on X" answer whether X is linked to someone you did block.
    """
    if not user_id:
        raise ValueError("user_id is required")
    user_id = resolve_identity(user_id)
    _init_db()
    with _db() as conn:
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
    with _db() as conn:
        rows = conn.execute(
            "SELECT blocked_id FROM relay_optouts WHERE user_id = ? "
            "ORDER BY created_at, blocked_id",
            (user_id,),
        ).fetchall()
    return [r[0] for r in rows]


def list_block_entries(user_id: str) -> list[dict]:
    """This user's blocks as {blocked_id, label}, oldest first.

    No row key. SQLite rowids are one global sequence, so handing them out as
    tokens told a user, by the gap between two of their own, that someone
    else had placed a block in between — and a rowid is reused after a
    delete, so a stale token could remove a different, newer block. The
    command layer names each entry by a keyed hash of (owner, blocked_id).
    """
    if not user_id:
        return []
    user_id = resolve_identity(user_id)
    _init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT blocked_id, label FROM relay_optouts WHERE user_id = ? "
            "ORDER BY created_at, blocked_id",
            (user_id,),
        ).fetchall()
    return [{"blocked_id": r[0], "label": r[1]} for r in rows]


def forget_blocks(user_id: str) -> int:
    """Drop every block this user has placed. Returns how many went.

    The explicit escape hatch DECISIONS #22 requires: /forget all deliberately
    leaves your blocks in place, because a privacy command that silently
    re-arms a harasser is a safety regression. This is the separately invoked
    way to drop them. Blocks OTHER people placed against you are theirs, and
    nothing here can touch them.
    """
    if not user_id:
        return 0
    user_id = resolve_identity(user_id)
    _init_db()
    with _db() as conn:
        cur = conn.execute("DELETE FROM relay_optouts WHERE user_id = ?", (user_id,))
        conn.commit()
    return cur.rowcount


def recent_senders(recipient_id: str, limit: int = 25) -> list[dict]:
    """Each ACCOUNT that has relayed to this recipient, most recent first, as
    {relay_id, account, shown_as} for the latest relay it delivered.

    Feeds /relay block's autocomplete, which is how a recipient standing in
    a DM with the bot names someone to block: a slash command's user picker
    cannot reach them from there. The autocomplete shows `shown_as` — the
    author line exactly as this recipient saw it — and hands the client only
    `relay_id`.

    Grouped by the sending account, not the resolved identity: two accounts
    linked in IDENTITY_LINKS are two entries, as the recipient saw them.
    Merged into one, the list itself said they were the same person.

    A message whose SENDER has since forgotten it is still offered (see
    forget_relay): being blockable is not theirs to withdraw. One the
    recipient forgot is not.

    DELIVERED messages only. A refused attempt never reached the recipient,
    and listing it would tell them that someone they blocked keeps trying —
    a fact about the sender's behaviour, which the recipient never received.

    One grouped read. It runs on the event loop at every autocomplete
    keystroke, where the correlated subquery it replaced was quadratic in the
    recipient's relays. SQLite takes the bare columns from the row holding
    the MAX; the unary + keeps the planner on the recipient index rather than
    scanning every delivered relay through idx_relay_open.
    """
    if not recipient_id:
        return []
    recipient_id = resolve_identity(recipient_id)
    _init_db()
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, COALESCE(sender_account, sender_id) AS account, shown_as,
                      MAX(delivered_at) AS last
               FROM relay_messages
               WHERE recipient_id = ? AND +status = ? AND sender_id != ? AND forgotten != ?
               GROUP BY account ORDER BY last DESC LIMIT ?""",
            (recipient_id, STATUS_DELIVERED, recipient_id, FORGOTTEN_SCRUBBED,
             max(1, int(limit))),
        ).fetchall()
    return [{"relay_id": r[0], "account": r[1], "shown_as": r[2]} for r in rows]


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

    # Blocks are stored as their owner supplied them (see block()), so each
    # is resolved here, at the moment it has to bind. Both ends, still: the
    # sender arrived resolved, and a block typed as an alias must catch them.
    blocks = [r[0] for r in conn.execute(
        "SELECT blocked_id FROM relay_optouts WHERE user_id = ?", (recipient_id,),
    ).fetchall()]
    if any(b != BLOCK_EVERYONE and resolve_identity(b) == sender_id for b in blocks):
        return Denial("blocked_sender", REFUSED)
    if BLOCK_EVERYONE in blocks:
        return Denial("blocked_everyone", REFUSED)

    return None


_INSERT_RESERVATION = """INSERT INTO relay_messages
    (id, sender_id, sender_account, recipient_id, recipient_account, origin_id, body, composed,
     guild_id, status, created_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


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
    # The account itself, kept before resolution: see sender_account in
    # _init_db. Everything the gate counts and checks uses the resolved id.
    sender_account = sender_id
    recipient_account = recipient_id
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

    # One statement for both paths — the block's open reservation and the
    # ordinary one — so the two rows cannot drift apart.
    reservation = (rid, sender_id, sender_account, recipient_id, recipient_account,
                   origin_id, body,
                   int(composed), guild_id, STATUS_RESERVED, created_at, expires_at)

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
                conn.execute(_INSERT_RESERVATION, reservation)
                conn.execute("COMMIT")
                return Denial(denial.reason, denial.message, relay_id=rid)
            if denial is not None:
                conn.execute("ROLLBACK")
                return denial
            conn.execute(_INSERT_RESERVATION, reservation)
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


def mark_sent(relay_id: str, dm_message_id: int, dm_channel_id: int | None = None,
              shown_as: str | None = None) -> None:
    """Record that the DM went out, and bind the id a reply will arrive on.

    `shown_as` is the author line exactly as the recipient saw it. Whatever
    later names this sender back to them uses it, never a name looked up
    afresh — that could be one they were never shown.

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
        with _db() as conn:
            # A row forgotten while its DM was in flight stays forgotten:
            # status and time still land, so the send is charged, but the
            # routing key and the author line are not written back onto a
            # scrubbed row. An anchor takes both — they are what it is for.
            cur = conn.execute(
                "UPDATE relay_messages SET status = ?, "
                "dm_message_id = CASE WHEN forgotten = ? THEN NULL ELSE ? END, "
                "dm_channel_id = CASE WHEN forgotten = 0 THEN ? ELSE NULL END, "
                "delivered_at = ?, "
                "shown_as = CASE WHEN forgotten = ? THEN NULL ELSE ? END WHERE id = ?",
                (STATUS_DELIVERED, FORGOTTEN_SCRUBBED, dm_message_id, dm_channel_id,
                 _iso(_now()), FORGOTTEN_SCRUBBED, shown_as, relay_id),
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
        with _db() as conn:
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
        with _db() as conn:
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
        with _db() as conn:
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


# ---------------------------------------------------------------------------
# Privacy, reconciliation and retention (PR 6)
# ---------------------------------------------------------------------------

def _mine(me: str, account: str) -> tuple[str, str, dict]:
    """SQL for "this person sent it" and "this person received it", and params.

    Each side matches the resolved identity OR the literal account. The
    account columns hold what was typed and shown, so a row still belongs to
    the account that wrote it after an IDENTITY_LINKS entry is added or
    removed; matching the resolved column alone left such rows neither
    forgotten nor exported. `IS` rather than `=`: rows older than the account
    columns hold NULL there, and NOT (x = NULL) is NULL, not true.
    """
    return ("(sender_id = :me OR sender_account IS :account)",
            "(recipient_id = :me OR recipient_account IS :account)",
            {"me": me, "account": account})


def forget_relay(user_id: str) -> int:
    """Forget every relayed message this person sent OR received.

    Both directions, because a shared exchange cannot be half-deleted: keeping
    one half so the other party can still export it would mean "forget me"
    did not.

    Rows still inside the rate window are SCRUBBED, not deleted. The caps are
    counted off these rows, so deleting them handed back the whole hour's
    quota: send until refused, /forget all, send again — an unbounded loop
    against the one control standing between this feature and a DM-spam
    primitive, and the only one that stops a brigade. What survives is what
    the gate needs and nothing else: the two ids, the status and the time.
    The body, the names and the routing key go at once, and the row itself
    goes when the hourly purge finds it outside the window.

    A message this person SENT that reached someone else keeps a little more,
    and keeps it until normal retention: the sending account, the author line
    its recipient was shown, and the DM it arrived as. That is what "Block
    this sender" and /relay block's picker need, and it belongs to the
    recipient as much as the sender — it is what they were shown, and the DM
    is still in their inbox. Scrubbing it let anyone disarm the block on
    themselves: relay, /forget all, relay again. It is the same rule as the
    blocks below — a privacy command must not be a safety regression for
    someone else — and nothing of it is exported, counted or routed. The body
    goes regardless.

    Blocks are deliberately untouched, in BOTH directions (DECISIONS #22).
    Blocks placed AGAINST this person are other people's data, and deleting
    them would let anyone launder away every block on themselves. Blocks this
    person PLACED survive too: a privacy command must not quietly re-arm a
    harasser, so dropping those is /relay forget-blocks, separately.

    Returns how many messages this person could have seen: everything they
    sent, and what was DELIVERED to them. Counting the rest would tell them,
    in a number, how many attempts at them were refused — what /export
    deliberately withholds.

    Takes the id as the person presented it, and resolves it once, here — as
    reserve_send did when the rows were written.
    """
    if not user_id:
        return 0
    sent, received, params = _mine(resolve_identity(user_id), user_id)
    now = _now()
    params.update(now=_iso(now), window=_iso(now - RATE_WINDOW),
                  delivered=STATUS_DELIVERED, reserved=STATUS_RESERVED,
                  scrubbed=FORGOTTEN_SCRUBBED, anchor=FORGOTTEN_ANCHOR)
    # Everything of this person's that is not a block anchor for someone else.
    theirs = f"({received} OR ({sent} AND forgotten != :anchor))"
    _init_db()
    with _db() as conn:
        visible = conn.execute(
            f"""SELECT COUNT(*) FROM relay_messages WHERE forgotten = 0
                  AND ({sent} OR ({received} AND status = :delivered))""",
            params).fetchone()[0]
        # Reserved too: one still in flight is delivered a moment from now,
        # and mark_sent then binds its DM to this anchor like any other.
        conn.execute(
            f"""UPDATE relay_messages
               SET body = '', recipient_account = NULL, origin_id = NULL,
                   guild_id = NULL, dm_channel_id = NULL, error = NULL,
                   closed_at = COALESCE(closed_at, :now), forgotten = :anchor
               WHERE {sent} AND NOT {received} AND forgotten = 0
                 AND status IN (:delivered, :reserved)""", params)
        conn.execute(
            f"DELETE FROM relay_messages WHERE {theirs} AND created_at < :window",
            params)
        conn.execute(
            f"""UPDATE relay_messages
               SET body = '', shown_as = NULL, sender_account = NULL,
                   recipient_account = NULL, origin_id = NULL, guild_id = NULL,
                   dm_channel_id = NULL, dm_message_id = NULL, error = NULL,
                   closed_at = COALESCE(closed_at, :now), forgotten = :scrubbed
               WHERE {theirs} AND created_at >= :window""", params)
    return visible


def export_relay(user_id: str) -> dict:
    """What this person may see of the relay: their messages both ways, and
    the blocks they placed. Never anyone else's blocks, least of all the ones
    placed against them.

    Everything named here is what they typed or were shown. A message they
    sent names the account they addressed (recipient_account) — and NOTHING
    for a row written before that column existed, rather than falling back to
    recipient_id, which is resolved and would name the main account behind an
    alt. A message they received names its sender by the author line they
    saw (shown_as).

    Received means DELIVERED. A refused attempt never reached them, and
    listing it would tell them someone they blocked keeps trying. Their own
    refused attempts appear under "sent", and read identically whether a
    block or closed DMs refused them. Forgotten rows appear nowhere.
    """
    if not user_id:
        return {"sent": [], "received": [], "blocks": []}
    sent_by, received_by, params = _mine(resolve_identity(user_id), user_id)
    user_id = params["me"]
    params["delivered"] = STATUS_DELIVERED
    _init_db()
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        sent = conn.execute(
            f"""SELECT id, recipient_account AS "to", body, status, created_at,
                      delivered_at, origin_id
               FROM relay_messages WHERE {sent_by} AND forgotten = 0
               ORDER BY created_at""",
            params).fetchall()
        received = conn.execute(
            f"""SELECT id, shown_as AS "from", body, delivered_at, origin_id
               FROM relay_messages
               WHERE {received_by} AND status = :delivered AND forgotten = 0
               ORDER BY delivered_at""",
            params).fetchall()
        blocks = conn.execute(
            """SELECT blocked_id, label, created_at FROM relay_optouts
               WHERE user_id = ? ORDER BY created_at""",
            (user_id,)).fetchall()
    return {
        "sent": [dict(r) for r in sent],
        "received": [dict(r) for r in received],
        "blocks": [{"everyone": r["blocked_id"] == BLOCK_EVERYONE,
                    "blocked": None if r["blocked_id"] == BLOCK_EVERYONE else r["blocked_id"],
                    "label": r["label"], "created_at": r["created_at"]} for r in blocks],
    }


# When this process started. Every reservation created before it belongs to a
# process that no longer exists; none created since may be touched by
# reconcile_pending, because on_ready fires again on every reconnect — and a
# reconcile that closed LIVE reservations turned delivered relays permanently
# "lapsed".
_PROCESS_STARTED = _iso(_now())


def reconcile_pending(before: str | None = None) -> int:
    """Close every reservation a previous process left open.

    A row goes in `reserved`, the DM is sent, and the row is settled. A crash
    between those steps leaves a row that will never route a reply and never
    tell the sender anything. Any reservation older than this process is such
    a row, however young: a grace period measured from "now" missed a restart
    inside a minute, and closed live rows on a reconnect.

    They become `failed`, which charges no one. Some may in fact have been
    delivered — there is no way to tell from here — and a reply to one finds
    no row; the router then recognises Fritz's own relay embed and says the
    exchange has lapsed, rather than handing the reply to the agent.
    """
    cutoff = before or _PROCESS_STARTED
    _init_db()
    with _db() as conn:
        cur = conn.execute(
            "UPDATE relay_messages SET status = ?, error = ?, closed_at = ? "
            "WHERE status = ? AND created_at < ?",
            (STATUS_FAILED, "reconciled: no outcome was recorded", _iso(_now()),
             STATUS_RESERVED, cutoff))
    if cur.rowcount:
        logger.warning("relay: closed %d reservation(s) left open by a crash", cur.rowcount)
    return cur.rowcount


def purge_expired(retention_days: int | None = None) -> int:
    """Delete what the relay no longer needs. Run hourly, and at boot.

    - Rows past RELAY_RETENTION_DAYS — unless still answerable. Bodies sit in
      plaintext in fritz.db, so retention is a privacy decision; but deleting
      a relay whose reply window is still open turned the recipient's answer
      into a conversation with Fritz. If the reply window is configured longer
      than retention, the window wins, and validate_config says so.
    - Scrubbed rows once they leave the rate window: they were kept only so
      the caps could still count them. A block anchor stays for normal
      retention if its message was delivered, and goes with the scrubbed rows
      if it never was.

    Blocks are never purged: a block is a standing instruction, not a record,
    and letting it expire would quietly re-arm whoever it was placed against.
    """
    days = fritz_utils.RELAY_RETENTION_DAYS if retention_days is None else retention_days
    now = _now()
    retention_cutoff = _iso(now - timedelta(days=max(1, int(days))))
    _init_db()
    with _db() as conn:
        cur = conn.execute(
            """DELETE FROM relay_messages
               WHERE (created_at < :retention
                      AND NOT (status = :delivered AND closed_at IS NULL
                               AND expires_at >= :now))
                  OR (created_at < :window
                      AND (forgotten = :scrubbed
                           OR (forgotten = :anchor AND status != :delivered)))""",
            {"retention": retention_cutoff, "delivered": STATUS_DELIVERED,
             "now": _iso(now), "window": _iso(now - RATE_WINDOW),
             "scrubbed": FORGOTTEN_SCRUBBED, "anchor": FORGOTTEN_ANCHOR})
    if cur.rowcount:
        logger.info("relay: purged %d message(s)", cur.rowcount)
    return cur.rowcount


def get_by_dm_message(dm_message_id: int) -> dict | None:
    """The relay a delivered DM belongs to, or None if it is not one of ours.

    The lookup every message-anchored action goes through: the "Block this
    sender" context menu now, and reply routing in PR 4. Anchored on the
    message rather than on "whoever last relayed to you", because a recipient
    holding relays from two people must never have one mistaken for the other.
    """
    if not dm_message_id:
        return None
    _init_db()
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        # Not a scrubbed row, even if one somehow held the key: whoever asks
        # would be handed a row with no account on it, and a caller falling
        # back to the resolved sender_id names the main behind an alt.
        row = conn.execute(
            "SELECT * FROM relay_messages WHERE dm_message_id = ? AND forgotten != ?",
            (int(dm_message_id), FORGOTTEN_SCRUBBED)).fetchone()
    return dict(row) if row else None


def get(relay_id: str) -> dict | None:
    """One relay row as a dict, or None. Read-only; used by tests and PR 4."""
    if not relay_id:
        return None
    _init_db()
    with _db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM relay_messages WHERE id = ?", (relay_id,)).fetchone()
    return dict(row) if row else None
