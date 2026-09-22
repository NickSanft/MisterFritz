"""Everything a person can do from a relayed DM.

When Fritz carries a message for someone (see relay_store and /tell), the
recipient can answer it two ways, both anchored on that one message: reply to
the DM in Discord, or press its [Reply] button and write in the form it opens.
This module decides whether an incoming DM or form is such an answer, and
carries it back. The DM's other two buttons live here too: [Not now], and
[Block sender], which answers exactly as /relay block does.

It is direction-agnostic: a reply is just a relay with the ends swapped, so it
goes through the same admission gate, takes the same caps, honours the same
blocks and is refused on the same schedule. There is no "reply handler"
separate from sending.

**This module must never import mister_fritz, directly or through anything
else** — which is why the presentation it shares with /tell lives in
relay_format rather than bot_commands. A relayed message is attacker-
controlled text, and the agent binds file tools with read/write/exec whenever
the user has a workspace. Code that cannot reach ask_stuff cannot contaminate
the LangGraph checkpoint or the Chroma memories injected into that person's
next prompt. tests/test_relay_router.py asserts the import graph, because a
guard that says "do not import this" is only as good as the thing that checks.

What does NOT route, and why it matters:

- Bare text. A recipient who types "sure, 8 works" into the DM gets a normal
  Fritz turn and the sender hears nothing. That is the correct failure: a
  session, a TTL window or first-reply-wins would silently capture whatever
  she typed next — "remind me to take my meds at 9" delivered to a near
  stranger — and no disclosure copy repairs that.
- A forward. Forwarding the relay DM carries reference.message_id for the very
  row we would look up, with empty content, so without the MessageType.reply
  gate a forward would relay nothing to the original sender and burn the
  exchange.
Nothing here reads bare text, and nothing keeps session state: a button's
custom_id carries the relay id, and every press is checked against the row
that id names and the message it was pressed on.

- A reply to anything that is not a relay: it falls through to the agent
  exactly as before. A reply to a relay that is no longer live — expired,
  closed, or with no row left at all because someone forgot it or the purge
  ran — does NOT fall through: it is told the exchange has lapsed. Fritz's own
  relay embed is how a relay with no row is still recognised
  (relay_format.relay_kind).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import partial

import discord

import fritz_utils
import relay_format
import relay_store
from bot_adapters import run_blocking
from fritz_utils import canonical_user_id, resolve_identity, split_user_id
from observability import METRICS, audit_log

logger = logging.getLogger(__name__)

# Said to the person who just replied, never to the other party.
LAPSED = ("That exchange has lapsed, so I have not carried this anywhere. "
          "Anything you send now stays between us.")
NO_WORDS = ("There were no words in that to carry, so I have carried nothing. "
            "I relay words, not parcels.")
UNREACHABLE = ("I have no way to reach whoever sent that. "
               + relay_format.NOTHING_SENT)
OWN_COPY = ("That is your own copy of what you sent, so a reply to it reaches no "
            "one. Their answer, if they send one, arrives here as a new message, "
            "and you can reply to that.")
NOT_YOURS = "That is not a message I carried to you, so I cannot act on it."
NOT_NOW = ("As you wish. They will not be told. Should you think better of it, reply "
           "to the message itself while it is still open.")
# With the relay off a reply would reach no one, so it is not suggested.
NOT_NOW_OFF = "As you wish. They will not be told."
IN_FLIGHT = "That has only just arrived. Press it again in a moment."
RELAY_OFF = "The message relay is switched off, so I can carry nothing. " + relay_format.NOTHING_SENT
# Not "reply to the message instead": the likeliest cause is the store, and
# a native reply that cannot reach it goes to the agent rather than to them.
REPLY_FAILED = ("Something went wrong on my side. " + relay_format.NOTHING_SENT
                + " Do try again shortly.")
SUBMIT_FAILED = ("Something went wrong on my side, so I cannot confirm what became of "
                 "that. Do not send it again until you have checked.")
BLOCK_FAILED = ("I could not block them just now. Apps \u2192 Block this sender, on "
                "this message, or /relay block will do it.")
UNHANDLED = ("That button no longer works. Reply to the message itself to answer, or "
             "use /relay block to stop them.")
BLOCK_GONE = ("I carried that message, but I no longer hold who sent it, so I cannot "
              "block them from it. /relay block takes their user ID or a mention, and "
              "/relay block-everyone refuses everyone.")
SELF_BLOCK = "You cannot block yourself, however tempting."


async def try_route_reply(client, ctx) -> bool:
    """Was this DM an answer to a relayed message, and has it been dealt with?

    True means on_message must stop: this module has answered the person in
    front of it, one way or another. False means the message is nothing to do
    with the relay and belongs to the agent, exactly as before.
    """
    if not fritz_utils.RELAY_ENABLED:
        return False

    # Message.type, not MessageReference.type. In discord.py 2.6.4
    # MessageReferenceType.reply is a literal alias of `default` (both 0), so
    # testing it would accept pins, crossposts, thread starters and poll
    # results alike. MessageType.reply (19) is the real discriminator, and it
    # exists in older versions too, so this does not depend on which
    # interpreter a contributor happens to run.
    if ctx.type is not discord.MessageType.reply:
        return False
    reference = ctx.reference
    if reference is None or not reference.message_id:
        return False

    try:
        row = await run_blocking(relay_store.get_by_dm_message, reference.message_id)
    except Exception as e:
        # The agent still gets the message: a store outage must not swallow a
        # DM, and this person is not owed a relay they may not have intended.
        logger.error("relay lookup failed for message %s: %s", reference.message_id, e)
        return False
    if row is None:
        # No row — but it may still have been a relay: forgotten by either
        # party, purged, or closed by reconcile. Fritz's own relay embed says
        # so, and a reply to it must not become a conversation turn containing
        # an answer meant for someone else. Anything else is the agent's.
        kind = relay_format.relay_kind(getattr(reference, "resolved", None),
                                       getattr(client, "user", None))
        if kind == "relay":
            await _say(ctx, LAPSED)
            return True
        if kind == "receipt":
            # A sender answering their own copy. It used to reach the agent,
            # carrying words meant for the other person.
            await _say(ctx, OWN_COPY)
            return True
        return False

    replier = canonical_user_id("discord", ctx.author.id)
    if not delivered_to(row, replier):
        # Near-tautological in a 1:1 DM, and three lines: it is the guard that
        # stops a forged or mis-indexed anchor delivering someone's words to
        # the wrong person.
        logger.warning("relay %s anchored by %s, who is not its recipient",
                       row["id"], replier)
        return False

    await _answer_relay(client, row, ctx.author, (ctx.content or "").strip(),
                        partial(_say, ctx), attachments=ctx.attachments)
    return True


async def _answer_relay(client, row: dict, replier, body: str, say, *,
                        attachments=()) -> None:
    """Carry `body` back along `row`, or say why not. `replier` has already
    been checked as the row's recipient.

    Shared by a native reply and the [Reply] form, so the two can never
    disagree about what is still open, what counts as words, or where an
    answer goes. `say` answers the person in front of us and never raises.
    """
    if not _is_open(row):
        # Deliberately NOT falling through. The reply would otherwise become a
        # Fritz conversation turn containing a message meant for someone else.
        await say(LAPSED)
        return
    if not body:
        await say(NO_WORDS)
        return
    target = _reply_target(row)
    if target is None:
        logger.warning("relay %s has no Discord account to reply to (%r)",
                       row["id"], row.get("sender_account") or row["sender_id"])
        await say(UNREACHABLE)
        return
    await _carry_back(client, replier, row, target, body, say, attachments=attachments)


def delivered_to(row: dict, me: str) -> bool:
    """Was this relay delivered to `me`, the canonical account in front of us?

    The recipient stored is already resolved, so it is compared with `me`
    resolved once, never resolved again: a second hop through chained
    IDENTITY_LINKS locked linked recipients out of their own relays. The
    account the DM was actually sent to counts too, so a link added or
    removed after delivery cannot lock the real recipient out — the way
    relay_store._mine matches rows for forget and export. It admits nobody
    new: that account is the one whose DMs the message is in.
    """
    return (row["recipient_id"] == resolve_identity(me)
            or (row.get("recipient_account") is not None and row["recipient_account"] == me))


def _is_open(row: dict) -> bool:
    """A relay can be answered while it is delivered, unclosed and unexpired."""
    if row["status"] != relay_store.STATUS_DELIVERED or row["closed_at"]:
        return False
    try:
        expires = datetime.fromisoformat(row["expires_at"])
        if expires.tzinfo is None:
            # Everything this store writes is aware UTC; a naive value means a
            # hand-edited row or some future writer. Reading it as UTC beats
            # raising: an exception here escapes into on_message, which has no
            # handler around the hook, and the DM would vanish entirely.
            expires = expires.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) <= expires
    except (TypeError, ValueError):     # pragma: no cover - a hand-edited row
        return False


def _reply_target(row: dict) -> str | None:
    """Whose account the reply goes to: the one that actually sent the relay.

    sender_account is what the recipient was shown and what they would block;
    sender_id is that identity after IDENTITY_LINKS, which is what the caps
    count. Reply to the account. Older rows have no sender_account.
    """
    for candidate in (row.get("sender_account"), row["sender_id"]):
        if candidate and split_user_id(candidate)[0] == "discord":
            return candidate
    return None


async def _say(ctx, text: str) -> None:
    """Answer the person in front of us, and never let that failure escape.

    on_message has no error handling around this hook; an exception here would
    surface as a traceback in the log and nothing at all to the user.
    """
    try:
        await ctx.channel.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception as e:
        logger.warning("could not answer a relay reply: %s", type(e).__name__)


async def _carry_back(client, author, row: dict, target: str, body: str, say, *,
                      attachments=()) -> None:
    """Take the reply through the gate and deliver it, or say why not.

    `author` is whoever wrote it — ctx.author for a native reply, the
    presser for the form — and is who the carried embed names.
    """
    replier = canonical_user_id("discord", author.id)
    guild_id = row["guild_id"]
    # Drawn before anything is decided, so a block and a Discord 403 are
    # measured from the same start. See relay_format.REFUSAL_WINDOW_SEC.
    deadline = relay_format.refusal_deadline()

    try:
        outcome = await run_blocking(
            relay_store.reserve_send, replier, target, body,
            guild_id=guild_id, origin_id=row["id"])
    except Exception as e:
        logger.error("relay reply could not be reserved: %s", e)
        await say("I could not arrange that. " + relay_format.NOTHING_SENT)
        return

    if not outcome.ok:
        METRICS.increment(f"relay.denied.{outcome.reason}")
        if outcome.relay_id is not None:
            # A block: the same lifecycle, schedule and audit line a 403 gets.
            audit_log("relay_refused", **relay_format.refusal_audit(
                replier, target, outcome.relay_id, guild_id, body))
            await _refuse(say, outcome.relay_id, deadline)
            return
        audit_log("relay_denied", sender=replier, recipient=target, guild_id=guild_id,
                  reason=outcome.reason, chars=len(body),
                  body_digest=relay_format.audit_digest(body))
        await say(outcome.message)
        return

    shown_as = relay_format.author_line(author)
    carried = "" if not attachments else (
        f" They attached {len(attachments)} file"
        f"{'s' if len(attachments) != 1 else ''}, which I do not carry.")
    embed = relay_format.relay_embed(
        body, shown_as=shown_as, icon_url=author.display_avatar.url,
        footer=(relay_format.FOOTER_REPLY + " for you." + carried
                + "\nReply to this to answer; /relay block to stop."))
    audit = relay_format.refusal_audit(replier, target, outcome.id, guild_id, body)

    try:
        # create_dm with a bare Object: no member cache, no fetch_user, one
        # POST at worst. Both round trips - opening the channel and sending -
        # raise here, so one try covers them.
        channel = await client.create_dm(discord.Object(id=int(split_user_id(target)[1])))
        # The reply is a relay in its own right, so it carries the same
        # buttons, bound to ITS row: the original sender is that row's
        # recipient, and may answer, decline or block in turn.
        sent = await channel.send(embed=embed, view=relay_view(outcome.id),
                                  allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden as e:
        logger.info("relay reply %s refused by Discord: status=%s code=%s",
                    outcome.id, e.status, e.code)
        audit_log("relay_refused", **audit)
        await _refuse(say, outcome.id, deadline)
        return
    except discord.NotFound as e:
        await _failed(say, outcome.id, audit, e,
                      "Discord could not find that account. " + relay_format.NOTHING_SENT)
        return
    except discord.RateLimited as e:
        await _failed(say, outcome.id, audit, e,
                      f"Discord has asked me to wait {e.retry_after:.0f} seconds before "
                      f"sending more. {relay_format.NOTHING_SENT} Do try again shortly.")
        return
    except discord.DiscordServerError as e:
        await _failed(say, outcome.id, audit, e,
                      f"Discord is having difficulties of its own. "
                      f"{relay_format.NOTHING_SENT} Do try again shortly.")
        return
    except discord.HTTPException as e:
        if e.status == 400:
            # The content itself. The sender's doing, so the sender's charge.
            await _failed(say, outcome.id, audit, e,
                          f"Discord declined that message as written. {relay_format.NOTHING_SENT}",
                          mark=relay_store.mark_rejected)
            return
        note = (f"Discord is rate-limiting me. {relay_format.NOTHING_SENT} Try again in a minute."
                if e.status == 429 else
                f"Discord would not take that message. {relay_format.NOTHING_SENT}")
        await _failed(say, outcome.id, audit, e, note)
        return
    except Exception as e:
        await _failed(say, outcome.id, audit, e,
                      f"Something went wrong on my side. {relay_format.NOTHING_SENT}")
        return

    recorded = True
    try:
        await run_blocking(relay_store.mark_sent, outcome.id, sent.id,
                           sent.channel.id, shown_as)
    except Exception as e:
        # Delivered, and it cannot be un-delivered. Say so rather than
        # reporting a failure that did not happen.
        recorded = False
        logger.error("relay reply %s delivered but not recorded: %s", outcome.id, e)
    METRICS.increment("relay.delivered")
    audit_log("relay_sent", **audit)

    confirmation = "Carried back, in your words."
    if attachments:
        confirmation += " Your attachment stayed here; I carry words, not parcels."
    if not recorded:
        confirmation += (" I failed to note it down, however, so a further reply "
                         "may not find its way.")
    await say(confirmation)


async def _refuse(say, relay_id: str, deadline: float) -> None:
    """Settle and answer a recipient-side refusal, on the shared schedule.

    A block and Discord's 403 are answered at the same randomly drawn moment
    and leave the same row, so neither the copy nor the timing says which one
    it was. See relay_format.REFUSAL_WINDOW_SEC.
    """
    await relay_format.sleep_until(deadline)
    await relay_format.settle(relay_store.mark_refused, relay_id)
    METRICS.increment("relay.refused")
    await say(relay_store.REFUSED)


async def _failed(say, relay_id: str, audit: dict, exc: BaseException, note: str,
                  *, mark=None) -> None:
    """The send failed. The person in front of us is told, and told plainly.

    They are the only one in the room: the original sender cannot see any of
    this, and telling them instead would be telling nobody.
    """
    # Resolved here, not in the signature: a default argument binds at
    # definition time, so patching relay_store.mark_failed in a test had no
    # effect at all and the bookkeeping assertions passed vacuously.
    mark = mark or relay_store.mark_failed
    status = getattr(exc, "status", None)
    reason = type(exc).__name__ + (f" {status}" if status else "")
    logger.warning("relay reply %s failed: %s", relay_id, reason)
    await relay_format.settle(mark, relay_id, reason)
    METRICS.increment("relay.failed")
    audit_log("relay_failed", error=reason, **audit)
    await say(note)


# ---------------------------------------------------------------------------
# Blocking a relay's sender
# ---------------------------------------------------------------------------

def relay_sender(row: dict) -> str:
    """What to block when blocking "the sender of this relay": the account
    that actually sent it, not the identity it resolves to.

    Blocks are resolved once, at the gate. Blocking the already-resolved id
    resolved it twice — under chained links that caught a different identity
    and let the real sender through — and showed the recipient a linked
    account they were never shown. Rows written before sender_account existed
    fall back to sender_id.
    """
    return row.get("sender_account") or row["sender_id"]


async def place_block(me: str, target: str, label: str) -> str:
    """Place one block and return what to say about it.

    Shared by /relay block, Apps -> Block this sender, and the relayed DM's
    [Block sender] button, so all three answer in the same words. The same
    answer whether or not the block was new: "you had already blocked X"
    would say, for two ids that resolve to one person, that they are linked.
    A store error propagates; each caller answers it in its own way.
    """
    try:
        await run_blocking(relay_store.block, me, target, label=label)
    except ValueError:
        return SELF_BLOCK
    # Who blocked, never whom: the audit log outlives /relay forget-blocks,
    # and a permanent record of the blocked party would outlive the block.
    audit_log("relay_block", user_id=me, kind="person")
    return (f"Done. I shall carry nothing further to you from "
            f"{relay_format.block_label(label)}. They will not be told.")


# ---------------------------------------------------------------------------
# The relayed DM's buttons
# ---------------------------------------------------------------------------
#
# DynamicItems, so a button keeps working on a DM sent before the last
# restart: the relay id rides in the custom_id, and the class that handles it
# is registered once, at import, by main_discord. Four rules keep that true,
# each learned from discord.py 2.6.4's source:
#
# - A relay view holds ONLY these items, and is never stopped or given a
#   timeout. Stopping a view that also holds an ordinary custom_id button
#   unregisters every DynamicItem class in it, for the whole process.
# - To change the buttons, send view=None or a fresh relay_view. Never re-send
#   the View a callback is handed: it is rebuilt from the message, its other
#   buttons become ordinary items, and one press then runs two callbacks.
# - Every press is answered by us. The library answers nothing and logs
#   nothing when a check fails; an exception is logged and the person sees
#   "This interaction failed".
# - The templates are disjoint, so a custom_id matches exactly one class.
#   Two matching patterns would both run.

CUSTOM_ID_PREFIX = "fritz:relay:"
_RID = r"(?P<rid>[0-9a-f]{8})"          # relay_store ids are str(uuid4())[:8]
_RID_RE = re.compile(r"[0-9a-f]{8}")


def _is_theirs(row: dict | None, relay_id: str, me: str) -> bool:
    """Is `row` the relay this button was sent with, delivered to `me`?

    The row is found by the message the button was pressed on — Discord's to
    supply, not the presser's — and must also carry the id in the custom_id.
    """
    return row is not None and row["id"] == relay_id and delivered_to(row, me)


def _row_for_press(message_id: int, relay_id: str) -> dict | None:
    """The row a press on this message is about. Blocking: callers choose the
    event loop or the pool.

    Found by the message, which is bound when mark_sent lands — and the DM
    can be on screen, and pressed, before that, because mark_sent waits in
    the worker pool behind whatever is there. Meanwhile the button's own id
    names the row: Discord took it off this very message, so it is not the
    presser's to choose. Only a row still in flight is taken that way; one
    that is delivered and bound elsewhere is some other message's.
    """
    row = relay_store.get_by_dm_message(message_id)
    if row is None:
        pending = relay_store.get(relay_id)
        if pending is not None and pending["status"] == relay_store.STATUS_RESERVED:
            return pending
    return row


def _shown_on(message) -> str | None:
    """The author line on one of Fritz's relayed DMs: what its recipient saw,
    for a row that has not recorded it yet."""
    try:
        return message.embeds[0].author.name or None
    except (AttributeError, IndexError, TypeError):
        return None


def _ours(interaction) -> bool:
    """Is the message pressed on one of Fritz's relays, row or no row?"""
    return relay_format.relay_kind(interaction.message,
                                   getattr(interaction.client, "user", None)) == "relay"


async def _tell(interaction, text: str) -> None:
    """Answer a press or a form ephemerally, whichever way is still open, and
    never let the failure to do so escape."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True,
                                            allowed_mentions=discord.AllowedMentions.none())
        else:
            await interaction.response.send_message(
                text, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
    except Exception as e:
        logger.warning("could not answer a relay button: %s", type(e).__name__)


async def _withdraw(interaction, text: str) -> None:
    """Take the buttons off the message, then say `text` to the presser.

    Editing is the press's one response, so it is also the acknowledgement:
    the buttons vanish at once, and a second press cannot follow.
    """
    try:
        await interaction.response.edit_message(view=None)
    except Exception as e:
        logger.info("could not withdraw relay buttons: %s", type(e).__name__)
    await _tell(interaction, text)


class _RelayButton:
    """What the three buttons share. Each subclass names its KIND, label,
    style and the copy for a failure nobody foresaw."""

    KIND = ""
    LABEL = ""
    STYLE = discord.ButtonStyle.secondary
    FAILED = REPLY_FAILED

    def __init__(self, relay_id: str) -> None:
        # fullmatch here as well: the library checks a custom_id against its
        # template with re.match when an item is BUILT, but with re.fullmatch
        # when it is PRESSED, so a malformed id would build a button that
        # silently never works.
        if not isinstance(relay_id, str) or not _RID_RE.fullmatch(relay_id):
            raise ValueError(f"not a relay id: {relay_id!r}")
        self.relay_id = relay_id
        super().__init__(discord.ui.Button(
            label=self.LABEL, style=self.STYLE,
            custom_id=f"{CUSTOM_ID_PREFIX}{self.KIND}:{relay_id}"), row=0)

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        # Nothing else here: this runs inside the press's three seconds, and
        # an exception is only logged, never answered.
        return cls(match["rid"])

    async def callback(self, interaction) -> None:
        METRICS.increment(f"relay.button.{self.KIND}")
        try:
            await self.pressed(interaction)
        except Exception as e:
            logger.error("relay %s: %s press failed: %s", self.relay_id, self.KIND, e,
                         exc_info=True)
            await _tell(interaction, self.FAILED)

    async def pressed(self, interaction) -> None:   # pragma: no cover - abstract
        raise NotImplementedError


class ReplyButton(_RelayButton, discord.ui.DynamicItem[discord.ui.Button],
                  template=r"fritz:relay:reply:" + _RID):
    """[Reply]: opens the form whose submission is the reply."""

    KIND = "reply"
    LABEL = "Reply"
    STYLE = discord.ButtonStyle.primary

    async def pressed(self, interaction) -> None:
        if not fritz_utils.RELAY_ENABLED:
            await _tell(interaction, RELAY_OFF)
            return
        me = canonical_user_id("discord", interaction.user.id)
        # On the event loop, not through run_blocking, and deliberately: the
        # form must be this press's FIRST response, so it cannot defer, and
        # the worker pool can be full of model turns for minutes. One indexed
        # read, as /relay block's autocomplete does. Everything is checked
        # again when the form comes back, since that can be much later.
        try:
            row = _row_for_press(interaction.message.id, self.relay_id)
        except Exception as e:
            logger.error("relay %s: reply lookup failed: %s", self.relay_id, e)
            await _tell(interaction, REPLY_FAILED)
            return
        if row is None and _ours(interaction):
            await _withdraw(interaction, LAPSED)     # forgotten, or purged
            return
        if not _is_theirs(row, self.relay_id, me):
            if row is not None:
                logger.warning("relay %s: reply pressed by %s on relay %s's message",
                               self.relay_id, me, row["id"])
            await _tell(interaction, NOT_YOURS)
            return
        # A row still in flight cannot be open yet; the form is checked
        # again when it comes back, by which time it will be.
        if row["status"] != relay_store.STATUS_RESERVED and not _is_open(row):
            await _withdraw(interaction, LAPSED)
            return
        if _reply_target(row) is None:
            await _tell(interaction, UNREACHABLE)
            return
        await interaction.response.send_modal(reply_form(row["id"]))


class NotNowButton(_RelayButton, discord.ui.DynamicItem[discord.ui.Button],
                   template=r"fritz:relay:notnow:" + _RID):
    """[Not now]: the buttons go, the sender is told nothing, and the message
    can still be answered by replying to it until it lapses.

    It reads and writes nothing. Only the two people in a DM can press its
    buttons — the recipient and Fritz — so there is no one else to check for,
    and nothing it does needs the row.
    """

    KIND = "notnow"
    LABEL = "Not now"

    async def pressed(self, interaction) -> None:
        await _withdraw(interaction, NOT_NOW if fritz_utils.RELAY_ENABLED else NOT_NOW_OFF)


class BlockButton(_RelayButton, discord.ui.DynamicItem[discord.ui.Button],
                  template=r"fritz:relay:block:" + _RID):
    """[Block sender]: the one-press opt-out DECISIONS #23 promises.

    Blocks what Apps -> Block this sender blocks, in the same words, and works
    whether or not the relay is switched on: people may refuse in advance.
    Reversible with /relay unblock, which is why it asks nothing first.
    """

    KIND = "block"
    LABEL = "Block sender"
    STYLE = discord.ButtonStyle.danger
    FAILED = BLOCK_FAILED

    async def pressed(self, interaction) -> None:
        # The buttons go first: that is this press's acknowledgement, and it
        # means a double press cannot follow. Blocking twice would be harmless
        # anyway; it is one idempotent write.
        try:
            await interaction.response.edit_message(view=None)
        except Exception as e:
            logger.info("could not withdraw relay buttons: %s", type(e).__name__)
            # The press still needs answering within three seconds, and what
            # follows waits on the worker pool.
            try:
                await interaction.response.defer(ephemeral=True, thinking=True)
            except Exception as e:
                logger.info("could not acknowledge a block press: %s", type(e).__name__)
        me = canonical_user_id("discord", interaction.user.id)
        row = await run_blocking(_row_for_press, interaction.message.id, self.relay_id)
        if row is None and _ours(interaction):
            await _tell(interaction, BLOCK_GONE)
            return
        if not _is_theirs(row, self.relay_id, me):
            await _tell(interaction, NOT_YOURS)
            return
        await _tell(interaction, await place_block(
            me, relay_sender(row),
            row["shown_as"] or _shown_on(interaction.message) or "the sender of that message"))


RELAY_BUTTONS = (ReplyButton, NotNowButton, BlockButton)


def relay_view(relay_id: str) -> discord.ui.View:
    """The buttons for one relayed DM. Send it; never keep it.

    Stopped before it is sent, on purpose. discord.py stores every unfinished
    view it sends, and storing this one would register its classes as a side
    effect — so a missed add_dynamic_items would work until the first
    restart and then fail on every DM ever sent. Stopped, it is serialised
    exactly the same, and only the registration in main_discord makes the
    buttons work. That registration is what the tests check.
    """
    view = discord.ui.View(timeout=None)
    for cls in RELAY_BUTTONS:
        view.add_item(cls(relay_id))
    view.stop()
    return view


def install(client) -> None:
    """Register the relayed DM's buttons and the [Reply] form's listener on a
    commands.Bot. Idempotent: registration is keyed by template, and the
    listener is added once."""
    client.add_dynamic_items(*RELAY_BUTTONS)
    if on_interaction not in client.extra_events.get("on_interaction", []):
        # A listener, not @client.event, so it adds to on_interaction rather
        # than claiming it.
        client.add_listener(on_interaction, "on_interaction")


def ensure_installed(client) -> list[type]:
    """on_ready's check: install anything missing, loudly. Returns what was.

    Should find nothing — main_discord installs at import — and is the only
    line anywhere that would say so if it ever did.
    """
    missing = missing_registrations(client)
    if missing:
        logger.error("relay buttons were not registered (%s); registering them now",
                     ", ".join(cls.__name__ for cls in missing))
        install(client)
    return missing


def missing_registrations(client) -> list[type]:
    """The button classes `client` would not dispatch to. [] when all are
    registered — and when discord.py's internals have moved, since then
    there is nothing to check against; relay_view's test covers that."""
    store = getattr(getattr(client, "_connection", None), "_view_store", None)
    registered = getattr(store, "_dynamic_items", None)
    if registered is None:
        return []
    return [cls for cls in RELAY_BUTTONS if registered.get(
        cls.__discord_ui_compiled_template__) is not cls]


# ---------------------------------------------------------------------------
# The [Reply] form
# ---------------------------------------------------------------------------
#
# Submissions are handled in on_interaction, from the payload, and NOT by the
# Modal's on_submit. discord.py keeps an open form only in memory, so one
# submitted after a restart used to be dropped with a DEBUG line and nothing
# said to the person who wrote it. Read from the payload, every submission is
# handled the same way, restart or no restart, and the stored Modal is inert.

_WORDS_ID = "words"
_FORM_RE = re.compile(r"fritz:relay:form:" + _RID)
# Only how long discord.py remembers a form someone closed without sending.
# Submissions do not depend on it (see above).
FORM_TIMEOUT_SEC = 15 * 60
# Discord's own ceiling for a text input.
_TEXT_INPUT_MAX = 4000


def reply_form(relay_id: str) -> discord.ui.Modal:
    """The form [Reply] opens. Titles and labels stay within Discord's 45
    characters and the length within its 4000, neither of which the library
    checks: an oversized one fails only when Discord rejects it."""
    form = discord.ui.Modal(title="Your reply", custom_id=f"fritz:relay:form:{relay_id}",
                            timeout=FORM_TIMEOUT_SEC)
    form.add_item(discord.ui.TextInput(
        label="Carried in exactly these words",
        custom_id=_WORDS_ID,
        style=discord.TextStyle.paragraph,
        placeholder="They will see exactly what you write here, and nothing else.",
        required=True,
        max_length=min(relay_format.max_body_chars(fritz_utils.RELAY_MAX_BODY_CHARS),
                       _TEXT_INPUT_MAX)))
    return form


def _submitted_words(data: dict) -> str:
    """The text a form came back with, whichever component shape carries it:
    an action row (the classic shape) or a Label (discord.py 2.6's)."""
    stack = list(data.get("components") or [])
    while stack:
        component = stack.pop(0)
        if not isinstance(component, dict):
            continue
        if component.get("custom_id") == _WORDS_ID:
            return str(component.get("value") or "")
        stack.extend(component.get("components") or [])
        if isinstance(component.get("component"), dict):
            stack.append(component["component"])
    return ""


async def _submit_reply(interaction, relay_id: str, words: str) -> None:
    """A [Reply] form came back: carry it, through the same gate as a native
    reply, after checking everything again — the relay may have lapsed, been
    forgotten or been purged while the form was open."""
    # First, before anything that can wait: the refusal schedule alone holds
    # an answer for up to two seconds, and a form has three.
    await interaction.response.defer(ephemeral=True, thinking=True)
    say = partial(_tell, interaction)
    METRICS.increment("relay.button.reply_sent")
    if not fritz_utils.RELAY_ENABLED:
        await say(RELAY_OFF)
        return
    me = canonical_user_id("discord", interaction.user.id)
    try:
        row = await run_blocking(relay_store.get, relay_id)
    except Exception as e:
        logger.error("relay %s: form lookup failed: %s", relay_id, e)
        await say("I could not look that up. " + relay_format.NOTHING_SENT)
        return
    # The form's custom_id comes back from the client, so it is the
    # presser's to choose: it names nothing until the row it names is shown
    # to have been delivered to them. Someone else's relay is answered
    # exactly as no relay at all, so a guessed id says nothing about whether
    # it exists — as a guessed /relay block token already does not.
    if row is not None and not delivered_to(row, me):
        logger.warning("relay %s: form submitted by %s, who is not its recipient",
                       relay_id, me)
        row = None
    if row is None:
        await say(LAPSED)
        return
    METRICS.increment("relay.replies_routed")
    await _answer_relay(interaction.client, row, interaction.user, words.strip(), say)


async def on_interaction(interaction) -> None:
    """Registered by main_discord for every interaction.

    Handles [Reply] form submissions (see above), and is the backstop for a
    relay button no class is registered for — which would otherwise fail on
    every DM ever sent, with nothing in the log to say why.
    """
    try:
        data = interaction.data or {}
        custom_id = data.get("custom_id") or ""
        if not custom_id.startswith(CUSTOM_ID_PREFIX):
            return
        if interaction.type is discord.InteractionType.modal_submit:
            match = _FORM_RE.fullmatch(custom_id)
            if match:
                await _submit_reply(interaction, match["rid"], _submitted_words(data))
            return
        if interaction.type is discord.InteractionType.component:
            store = getattr(getattr(interaction.client, "_connection", None),
                            "_view_store", None)
            patterns = getattr(store, "_dynamic_items", None)
            if patterns is not None and not any(p.fullmatch(custom_id) for p in patterns):
                logger.error("relay button %r pressed, but no handler is registered for it",
                             custom_id)
                await _tell(interaction, UNHANDLED)
    except Exception as e:
        logger.error("relay interaction failed: %s", e, exc_info=True)
        await _tell(interaction, SUBMIT_FAILED)
