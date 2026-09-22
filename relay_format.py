"""Presentation and timing shared by both ends of the direct-message relay.

/tell (bot_commands) and the reply router (relay_router) have to agree, to the
character and to the second, about how a relayed message looks, how a
recipient-side refusal is timed, and what an audit line may say. This module
is where that agreement lives.

It exists as its own module for one structural reason: **relay_router must
never import mister_fritz**, and bot_commands does. A relayed message is
attacker-controlled text, and the agent binds file tools with read/write/exec
whenever the user has a workspace. Code that cannot reach ask_stuff cannot
contaminate the LangGraph checkpoint or the Chroma memories injected into that
person's next prompt. Nothing here may import bot_commands, mister_fritz or
anything that reaches them; tests/test_relay_router.py asserts it.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import random
import re
import unicodedata

import discord

import fritz_utils
from bot_adapters import run_blocking

logger = logging.getLogger(__name__)

# Matches bot_commands.FRITZ_COLOUR, and --accent in
# admin_templates/_theme_admin.html, so every surface shares one brand colour.
# A test pins the two together rather than trusting them to stay equal.
RELAY_COLOUR = discord.Colour(0x5B3F30)

# Discord's cap on an embed description, which is where a relayed body goes.
# The library does not check it — an over-length embed is a 400 at send time.
EMBED_DESCRIPTION_MAX = 4096

# Every failure copy on a send path says this, in words. "Never
# optimistic-ack": a sender hears a message went only once Discord has
# returned it, and hears plainly when it did not.
NOTHING_SENT = "Nothing was sent."

# Recipient-side refusals — a block, and Discord's 403 for closed DMs — are
# answered, settled and released on ONE schedule: a deadline drawn from this
# window at the start of the attempt. A block is decided in a millisecond and
# a 403 after a Discord round trip; answering each as soon as it was known was
# a timing oracle for the question the shared refusal copy exists to leave
# unanswered, and padding only the block path merely moved that oracle. Tests
# set this to (0, 0).
REFUSAL_WINDOW_SEC = (1.0, 2.0)


# How each kind of relay message's footer begins. Every sender builds its
# footer from these, and relay_kind reads them back off Fritz's own message to
# recognise a relay whose row is gone — forgotten, purged or reconciled — so a
# reply to it is told the exchange has lapsed instead of becoming a
# conversation turn with Fritz. They are therefore a FORMAT: change one and
# every DM already sent with the old wording stops being recognised.
FOOTER_RELAYED = "Sent with /tell from"
FOOTER_REPLY = "A reply to the message I carried"
FOOTER_RECEIPT = "Your /tell from"


def relay_kind(message, bot_user) -> str | None:
    """"relay" for a relay DM Fritz sent, "receipt" for a sender's own copy of
    one, None for anything else — including a message that was deleted, or
    one Discord did not include with the reply.

    Only Fritz's own messages count: the footer text alone is forgeable by
    anyone who can post an embed, but a DM with the bot contains nobody's
    messages except the bot's and the person's own.
    """
    if not isinstance(message, discord.Message) or bot_user is None:
        return None
    if getattr(message.author, "id", None) != getattr(bot_user, "id", object()):
        return None
    footer = message.embeds[0].footer.text if message.embeds else None
    if not footer:
        return None
    if footer.startswith(FOOTER_RECEIPT):
        return "receipt"
    if footer.startswith((FOOTER_RELAYED, FOOTER_REPLY)):
        return "relay"
    return None


# [0-9], never \d: \d also matches non-ASCII digits ("١٢٣…"), which then
# turned into a different id, or an unhandled error, further down.
MENTION_RE = re.compile(r"<@!?([0-9]{15,25})>")


def mention_label(canonical: str) -> str:
    """The label for a block typed as an id: a mention of exactly that id,
    which the viewer's own client renders from what it already knows."""
    return f"<@{fritz_utils.split_user_id(canonical)[1]}>"


def block_label(label: str | None, blocked_id: str | None = None) -> str:
    """A block's label, fit for message content.

    Here rather than in bot_commands because the relayed DM's own Block button
    answers in the same words as /relay block and the context menu, and the
    router that handles it must never import bot_commands.
    """
    if label and MENTION_RE.fullmatch(label):
        return label
    if label:
        return discord.utils.escape_markdown(label)
    # A block placed without a label (only possible from code, never from a
    # command) falls back to a mention of what is stored.
    if blocked_id and fritz_utils.split_user_id(blocked_id)[0] == "discord":
        return mention_label(blocked_id)
    return "someone"


def max_body_chars(configured: int) -> int:
    """The longest body a relay will carry, whatever the operator configured."""
    return max(1, min(configured, EMBED_DESCRIPTION_MAX))


def refusal_deadline() -> float:
    lo, hi = REFUSAL_WINDOW_SEC
    return asyncio.get_running_loop().time() + random.uniform(lo, hi)


async def sleep_until(deadline: float) -> None:
    delay = deadline - asyncio.get_running_loop().time()
    if delay > 0:
        await asyncio.sleep(delay)


def plain(text: str) -> str:
    """Drop control and format characters: bidi overrides, zero-width joiners
    and the like, which can reorder or hide text in a recipient's client."""
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf"))


def author_line(user) -> str:
    """Who a relayed message says it is from: "@username · Display".

    The @username leads because it is the one part nobody can borrow: unique
    across Discord, from a character set with no room for tricks. The display
    name is sender-controlled twice over — a guild nickname can be set to
    anyone's name, and it can carry a "(@alice)" of its own, which made
    "Display (@username)" read as "Alice (@alice) (@mallory)". So it comes
    second, stripped of handle syntax ('@' and parentheses) and of anything
    that can reorder or hide text.
    """
    handle = "@" + plain(user.name)
    display = plain(getattr(user, "display_name", None) or "")
    display = " ".join(display.replace("@", "").replace("(", "").replace(")", "").split())[:64]
    if not display or display == user.name:
        return handle[:256]
    return f"{handle} · {display}"[:256]


def relay_embed(body: str, *, shown_as: str, icon_url: str | None, footer: str) -> discord.Embed:
    """The one shape a relayed message takes, in either direction.

    The author field sits structurally outside the sender-controlled
    description, which is what stops a body forging "— from @admin". The
    description is verbatim: no escaping, because inside an embed the sender
    cannot forge the author line whatever markdown they write, and escaping
    would make "verbatim" untrue.
    """
    embed = discord.Embed(description=body, colour=RELAY_COLOUR,
                          timestamp=discord.utils.utcnow())
    embed.set_author(name=shown_as[:256], icon_url=icon_url)
    embed.set_footer(text=footer[:2048])
    return embed


def audit_digest(body: str) -> str:
    """A keyed fingerprint of a relay body for audit.log — never the body.

    The plan specified sha256(body)[:16]. Unkeyed, that IS the message for
    anything short or guessable: "ok", "yes", "running late" fall to a
    dictionary in microseconds. Keyed with the host's persisted secret, the
    same body still yields the same token — one message sprayed at thirty
    people is still visibly one message — but the log alone cannot be reversed.
    The subkey is derived under its own label so this never shares key
    material with the chat cookie it is borrowed from.
    """
    key = hmac.new(fritz_utils.CHAT_COOKIE_SECRET.encode("utf-8"),
                   b"relay-audit-digest-v1", hashlib.sha256).digest()
    return hmac.new(key, body.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def refusal_audit(sender: str, recipient: str, relay_id: str,
                  guild_id: int | None, message: str) -> dict:
    """The one audit record for a recipient-side refusal, block or 403 alike.

    Built in one place so the two cannot drift apart: the audit log outlives
    /relay forget-blocks, and a field that differed between them would be a
    permanent record of who blocked whom.
    """
    return dict(sender=sender, recipient=recipient, relay_id=relay_id,
                guild_id=guild_id, chars=len(message), body_digest=audit_digest(message))


async def settle(fn, relay_id: str, *args) -> None:
    """Record how a relay ended without letting a store error eat the reply.

    On the failure paths the sender must still be told nothing was sent; a
    bookkeeping error there is logged, and the reservation it leaves behind
    stops holding quota by itself after RESERVATION_GRACE_SEC.
    """
    try:
        await run_blocking(fn, relay_id, *args)
    except Exception as e:
        # getattr: this handler must not be able to raise. A partial or a
        # wrapped callable has no __name__, and an AttributeError here would
        # take down exactly the reply this function exists to protect.
        logger.error("relay %s: %s failed: %s", relay_id,
                     getattr(fn, "__name__", repr(fn)), e)
