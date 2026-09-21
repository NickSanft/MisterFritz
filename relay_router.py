"""Routing for replies to relayed DMs.

When Fritz carries a message for someone (see relay_store and /tell), the
recipient answers it by replying to that DM in Discord. This module decides
whether an incoming DM is such an answer, and carries it back.

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
- A reply to anything that is not a relay: it falls through to the agent
  exactly as before. A reply to a relay that is no longer live — expired,
  closed, or with no row left at all because someone forgot it or the purge
  ran — does NOT fall through: it is told the exchange has lapsed. Fritz's own
  relay embed is how a relay with no row is still recognised
  (relay_format.relay_kind).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

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
    if row["recipient_id"] != resolve_identity(replier):
        # Near-tautological in a 1:1 DM, and three lines: it is the guard that
        # stops a forged or mis-indexed anchor delivering someone's words to
        # the wrong person.
        logger.warning("relay %s anchored by %s, who is not its recipient",
                       row["id"], replier)
        return False

    if not _is_open(row):
        # Deliberately NOT falling through. The reply would otherwise become a
        # Fritz conversation turn containing a message meant for someone else.
        await _say(ctx, LAPSED)
        return True

    body = (ctx.content or "").strip()
    if not body:
        await _say(ctx, NO_WORDS)
        return True

    target = _reply_target(row)
    if target is None:
        logger.warning("relay %s has no Discord account to reply to (%r)",
                       row["id"], row.get("sender_account") or row["sender_id"])
        await _say(ctx, UNREACHABLE)
        return True

    await _carry_back(client, ctx, row, target, body)
    return True


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


async def _carry_back(client, ctx, row: dict, target: str, body: str) -> None:
    """Take the reply through the gate and deliver it, or say why not."""
    replier = canonical_user_id("discord", ctx.author.id)
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
        await _say(ctx, "I could not arrange that. " + relay_format.NOTHING_SENT)
        return

    if not outcome.ok:
        METRICS.increment(f"relay.denied.{outcome.reason}")
        if outcome.relay_id is not None:
            # A block: the same lifecycle, schedule and audit line a 403 gets.
            audit_log("relay_refused", **relay_format.refusal_audit(
                replier, target, outcome.relay_id, guild_id, body))
            await _refuse(ctx, outcome.relay_id, deadline)
            return
        audit_log("relay_denied", sender=replier, recipient=target, guild_id=guild_id,
                  reason=outcome.reason, chars=len(body),
                  body_digest=relay_format.audit_digest(body))
        await _say(ctx, outcome.message)
        return

    shown_as = relay_format.author_line(ctx.author)
    carried = "" if not ctx.attachments else (
        f" They attached {len(ctx.attachments)} file"
        f"{'s' if len(ctx.attachments) != 1 else ''}, which I do not carry.")
    embed = relay_format.relay_embed(
        body, shown_as=shown_as, icon_url=ctx.author.display_avatar.url,
        footer=(relay_format.FOOTER_REPLY + " for you." + carried
                + "\nReply to this to answer; /relay block to stop."))
    audit = relay_format.refusal_audit(replier, target, outcome.id, guild_id, body)

    try:
        # create_dm with a bare Object: no member cache, no fetch_user, one
        # POST at worst. Both round trips - opening the channel and sending -
        # raise here, so one try covers them.
        channel = await client.create_dm(discord.Object(id=int(split_user_id(target)[1])))
        sent = await channel.send(embed=embed,
                                  allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden as e:
        logger.info("relay reply %s refused by Discord: status=%s code=%s",
                    outcome.id, e.status, e.code)
        audit_log("relay_refused", **audit)
        await _refuse(ctx, outcome.id, deadline)
        return
    except discord.NotFound as e:
        await _failed(ctx, outcome.id, audit, e,
                      "Discord could not find that account. " + relay_format.NOTHING_SENT)
        return
    except discord.RateLimited as e:
        await _failed(ctx, outcome.id, audit, e,
                      f"Discord has asked me to wait {e.retry_after:.0f} seconds before "
                      f"sending more. {relay_format.NOTHING_SENT} Do try again shortly.")
        return
    except discord.DiscordServerError as e:
        await _failed(ctx, outcome.id, audit, e,
                      f"Discord is having difficulties of its own. "
                      f"{relay_format.NOTHING_SENT} Do try again shortly.")
        return
    except discord.HTTPException as e:
        if e.status == 400:
            # The content itself. The sender's doing, so the sender's charge.
            await _failed(ctx, outcome.id, audit, e,
                          f"Discord declined that message as written. {relay_format.NOTHING_SENT}",
                          mark=relay_store.mark_rejected)
            return
        note = (f"Discord is rate-limiting me. {relay_format.NOTHING_SENT} Try again in a minute."
                if e.status == 429 else
                f"Discord would not take that message. {relay_format.NOTHING_SENT}")
        await _failed(ctx, outcome.id, audit, e, note)
        return
    except Exception as e:
        await _failed(ctx, outcome.id, audit, e,
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
    if ctx.attachments:
        confirmation += " Your attachment stayed here; I carry words, not parcels."
    if not recorded:
        confirmation += (" I failed to note it down, however, so a further reply "
                         "may not find its way.")
    await _say(ctx, confirmation)


async def _refuse(ctx, relay_id: str, deadline: float) -> None:
    """Settle and answer a recipient-side refusal, on the shared schedule.

    A block and Discord's 403 are answered at the same randomly drawn moment
    and leave the same row, so neither the copy nor the timing says which one
    it was. See relay_format.REFUSAL_WINDOW_SEC.
    """
    await relay_format.sleep_until(deadline)
    await relay_format.settle(relay_store.mark_refused, relay_id)
    METRICS.increment("relay.refused")
    await _say(ctx, relay_store.REFUSED)


async def _failed(ctx, relay_id: str, audit: dict, exc: BaseException, note: str,
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
    await _say(ctx, note)
