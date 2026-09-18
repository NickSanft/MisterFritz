import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import random
import time
import unicodedata
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from bot_adapters import fritz_error, run_blocking, split_into_chunks
from cards import draw_cards, get_remaining_card_number, reload_deck
from document_engine import query_documents
import fritz_utils
from fritz_utils import (
    FAST_OLLAMA_MODEL,
    FFMPEG_PATH,
    IMAGE_GEN_MAX_CONCURRENCY,
    MessageSource,
    THINKING_OLLAMA_MODEL,
    TTS_MAX_CONCURRENCY,
    __version__,
    canonical_user_id,
)
import identity_store
# NOTE: `image_generator` and `tts` are deliberately NOT imported here. Both
# pull torch/diffusers/xformers at module level, and bot_commands sits on the
# import path of main_discord — so importing them here makes the entire
# multi-GB GPU stack a hard requirement just to START the bot, even for someone
# who never runs /gen or /voice. They are deferred to the point of use; see
# gen_slash below and main_discord's TTS bootstrap. This mirrors what
# agent_tools.generate_image already does.
from mister_fritz import ask_stuff
from observability import METRICS, audit_log, get_health_snapshot
import privacy
import relay_store
import workspace_store

logger = logging.getLogger(__name__)

# Matches --accent in admin_templates/_theme_admin.html so the Discord and web
# surfaces share one brand colour.
FRITZ_COLOUR = discord.Colour(0x5B3F30)


def _identity(interaction: discord.Interaction) -> str:
    """Canonical id for the invoking user, recording their display name on the
    way past.

    Every slash command used `interaction.user.name` as the storage key, which
    meant a Discord rename orphaned that person's memories, schedules and
    workspace — and a Telegram user with the same handle shared them. This is
    the single conversion point for the whole cog.
    """
    user_id = canonical_user_id("discord", interaction.user.id)
    identity_store.record(
        user_id,
        getattr(interaction.user, "display_name", None) or interaction.user.name,
        "discord",
    )
    return user_id


def _display(interaction: discord.Interaction) -> str:
    """Human-readable name, for prose and card decks — never a storage key."""
    return getattr(interaction.user, "display_name", None) or interaction.user.name


async def _reply_error(interaction: discord.Interaction, operation: str,
                       exc: BaseException | None = None, *,
                       note: str | None = None, record: bool = True) -> None:
    """Send butler-voiced failure copy ephemerally, whichever response stage
    the interaction is in.

    A command that has already deferred must answer via followup; one that has
    not must answer via response. Getting it wrong raises, which is how a
    failure handler ends up compounding the failure.
    """
    text = fritz_error(operation, exc, note=note, record=record)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True)
        else:
            await interaction.response.send_message(text, ephemeral=True)
    except Exception as e:
        # Exception, not just HTTPException: a transport error here (aiohttp,
        # a timeout) escaped to the cog error handler, which then answered a
        # SECOND time. Not BaseException — cancellation must still propagate.
        logger.warning("Could not deliver error reply for %s: %s",
                       operation, type(e).__name__)


async def handle_app_command_error(interaction: discord.Interaction,
                                   error: app_commands.AppCommandError) -> None:
    """Single entry point for cog_app_command_error and tree.on_error.

    Without this, an out-of-range argument or any uncaught exception left the
    user staring at a spinner until Discord timed the interaction out.
    """
    original = getattr(error, "original", error)
    name = interaction.command.qualified_name if interaction.command else "unknown"
    if isinstance(error, app_commands.CheckFailure):
        await _reply_error(interaction, f"app_command.{name}", None,
                           note="That command is not available to you. One does have standards.")
        return
    if (isinstance(error, app_commands.TransformerError)
            and name.split(" ")[0] == "tell"
            and error.type is discord.AppCommandOptionType.user):
        # The Member annotation on /tell's recipient is what enforces "anyone in
        # a shared server": Discord only resolves a Member for someone who is
        # in this guild, and the library refuses anything else. The generic
        # "permitted range" copy below is meaningless for that.
        await _reply_error(interaction, f"app_command.{name}", None,
                           note="I can only carry word to members of this server, "
                                "and that person is not one. Nothing was sent.")
        return
    if isinstance(error, app_commands.TransformerError):
        await _reply_error(interaction, f"app_command.{name}", None,
                           note="That value is outside the permitted range. Do try one I can work with.")
        return
    if isinstance(original, discord.HTTPException):
        await _reply_error(interaction, f"app_command.{name}", original,
                           note="Discord declined to carry that message. It was, I suspect, too long.")
        return
    await _reply_error(interaction, f"app_command.{name}", original)


async def handle_tree_error(interaction: discord.Interaction,
                            error: app_commands.AppCommandError) -> None:
    """tree.on_error backstop: only for failures no command handler owns.

    discord.py runs a command's own error handlers and THEN tree.on_error,
    unconditionally (app_commands/tree.py, `_call`: `_invoke_error_handlers`
    followed by `self.on_error`). The library's default on_error copes by
    returning early when the command has handlers of its own. Installing
    handle_app_command_error directly as tree.on_error dropped that check, so
    every failure in this cog was handled twice: two error replies to the
    user, two refs in the log, and every error counted twice in METRICS.

    Same test the default uses. `_has_any_error_handlers` is private, but it
    is precisely the predicate the library's own on_error relies on, and a
    public stand-in (`command.binding is not None`) would silently swallow
    errors from any future cog that forgets to define cog_app_command_error.
    """
    command = interaction.command
    if command is not None and command._has_any_error_handlers():
        return
    await handle_app_command_error(interaction, error)


def _format_uptime(seconds: int) -> str:
    """Render a seconds count as e.g. '2d 3h 17m' or '4m 12s'."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def _render_image(prompt: str) -> str:
    """Import image_generator and render, both inside the worker thread.

    Exists so the import cannot be run on the event loop by accident. It has to
    stay a function-level import: tests/test_packaging.py asserts the core
    install is torch-free, and a module-level `from image_generator import ...`
    here would make the [image] extra mandatory just to start the bot.
    """
    from image_generator import generate_image
    return generate_image(prompt)


class _ForgetConfirmView(discord.ui.View):
    """30-second confirmation view for /forget all.

    Only the user who triggered the command can press Confirm — other users
    clicking the buttons get an ephemeral rejection. After the timeout, the
    buttons disable.
    """

    def __init__(self, requester: str, schedule_manager):
        super().__init__(timeout=30.0)
        self.requester = requester
        self.schedule_manager = schedule_manager

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # Compare canonical ids, not names: a rename between opening the
        # confirmation and pressing Confirm used to lock the requester out of
        # their own dialog.
        if canonical_user_id("discord", interaction.user.id) != self.requester:
            await interaction.response.send_message(
                "This confirmation isn't for you.", ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        result = await run_blocking(privacy.forget_all, self.requester, self.schedule_manager)
        audit_log("forget", user_id=self.requester, scope="all", result=result)
        await interaction.response.edit_message(
            content=(
                "✅ All data removed:\n"
                f"• memories: {result['memories']}\n"
                f"• conversation rows: {result['conversation_rows']}\n"
                f"• schedules: {result['schedules']}\n"
                f"• workspace dropped: {result['workspace_dropped']}"
            ),
            view=None,
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="Aborted. Nothing was deleted.", view=None)


async def _require_admin(interaction: discord.Interaction) -> bool:
    """Reject the interaction with an ephemeral message if the caller isn't an admin.

    An admin is ROOT_USER or anyone listed in ADMIN_USERS. Returns True if the
    caller is authorised, False otherwise. The caller should short-circuit on
    False.
    """
    if fritz_utils.is_admin(
        canonical_user_id("discord", interaction.user.id),
        display_name=interaction.user.name,
    ):
        return True
    await interaction.response.send_message(
        "You do not have permission to use this command.", ephemeral=True
    )
    return False


# ── Direct-message relay helpers ──────────────────────────────────────────────

# Discord's cap on an embed description, which is where a relayed body goes.
# The library does not check it (an over-length embed is a 400 at send time),
# so the slash option itself is capped and the client will not let a longer
# message be submitted, whatever RELAY_MAX_BODY_CHARS says.
_EMBED_DESCRIPTION_MAX = 4096


def _tell_max_chars(configured: int) -> int:
    return max(1, min(configured, _EMBED_DESCRIPTION_MAX))


TELL_MAX_CHARS = _tell_max_chars(fritz_utils.RELAY_MAX_BODY_CHARS)

# Every failure copy on the send path says this, in words. "Never
# optimistic-ack": the sender hears a message went only once Discord has
# returned it, and hears plainly when it did not.
_NOTHING_SENT = "Nothing was sent."

# Recipient-side refusals — a block, and Discord's 403 for closed DMs — are
# answered, settled and released on ONE schedule: a deadline drawn from this
# window at the start of the command. A block is decided in a millisecond and
# a 403 after a Discord round trip; answering each as soon as it is known was
# a timing oracle for the question REFUSED exists to leave unanswered, and a
# pad on the block path alone only moved that oracle. Both now wait for the
# same randomly drawn moment, which comfortably outlasts a normal round trip.
# A 403 that arrives after its deadline is answered late and is
# distinguishable; that is the residual, and it needs a slow Discord. Tests
# set this to (0, 0).
_REFUSAL_WINDOW_SEC = (1.0, 2.0)


def _refusal_deadline() -> float:
    lo, hi = _REFUSAL_WINDOW_SEC
    return asyncio.get_running_loop().time() + random.uniform(lo, hi)


async def _sleep_until(deadline: float) -> None:
    delay = deadline - asyncio.get_running_loop().time()
    if delay > 0:
        await asyncio.sleep(delay)


def _plain(text: str) -> str:
    """Drop control and format characters: bidi overrides, zero-width joiners
    and the like, which can reorder or hide text in the recipient's client."""
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf"))


def _relay_author_line(user) -> str:
    """Who a relayed message says it is from: "@username · Display".

    The @username leads because it is the one part nobody can borrow: unique
    across Discord, from a character set with no room for tricks. The display
    name is sender-controlled twice over — a guild nickname can be set to
    anyone's name, and it can carry a "(@alice)" of its own, which made
    "Display (@username)" read as "Alice (@alice) (@mallory)". So it comes
    second, stripped of handle syntax ('@' and parentheses) and of anything
    that can reorder or hide text.
    """
    handle = "@" + _plain(user.name)
    display = _plain(getattr(user, "display_name", None) or "")
    display = " ".join(display.replace("@", "").replace("(", "").replace(")", "").split())[:64]
    if not display or display == user.name:
        return handle[:256]
    return f"{handle} \u00b7 {display}"[:256]


def _audit_digest(body: str) -> str:
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


async def _settle(fn, relay_id: str, *args) -> None:
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


async def _answer(interaction: discord.Interaction, text: str, **kwargs) -> None:
    """Send a /tell reply, and never let a failure to send it escape.

    Once the relay's outcome is decided, a failure to REPORT it must not reach
    the cog error handler. Its copy ("Discord declined to carry that message")
    would tell a sender whose message WAS delivered that it was not, and a
    sender who believes that sends it again. Exception, not BaseException:
    cancellation still propagates.
    """
    try:
        await interaction.followup.send(text, ephemeral=True, **kwargs)
    except Exception as e:
        logger.warning("could not deliver a /tell reply: %s", type(e).__name__)


class FritzCommands(commands.Cog):
    """All MisterFritz slash commands."""

    # `sayer` is a tts.TTSEngine, left untyped so the annotation does not drag
    # torch back into the import graph. None when the [voice] extra is absent.
    def __init__(self, bot: commands.Bot, sayer, schedule_manager=None):
        self.bot = bot
        self.sayer = sayer
        self.schedule_manager = schedule_manager
        # Admission control for the GPU-bound commands. Waiters park on the
        # event loop, NOT on a pool thread, so a queue of /gen requests can
        # never starve the shared blocking pool of workers.
        #
        # Instance-level, not module-level, on purpose: asyncio primitives bind
        # to the first loop that awaits them. A module-level semaphore would
        # attach to whichever test's loop ran first, and in production would
        # wedge /gen permanently after a gateway reconnect built a new loop.
        self._image_semaphore = asyncio.Semaphore(IMAGE_GEN_MAX_CONCURRENCY)
        self._tts_semaphore = asyncio.Semaphore(TTS_MAX_CONCURRENCY)

    async def cog_app_command_error(self, interaction: discord.Interaction,
                                    error: app_commands.AppCommandError) -> None:
        """discord.py dispatches every failure from a command in this cog here."""
        await handle_app_command_error(interaction, error)

    # ── Scheduled tasks ───────────────────────────────────────────────────────

    schedule = app_commands.Group(name="schedule", description="Manage scheduled Fritz tasks")

    @schedule.command(name="add", description="Schedule a recurring Fritz prompt in this channel")
    @app_commands.describe(
        every="When to run: interval ('30m', '2h', '1d') or cron ('0 9 * * *')",
        prompt="What to ask Fritz each time",
        description="Optional label to help identify this schedule",
    )
    async def schedule_add(
        self,
        interaction: discord.Interaction,
        every: str,
        prompt: str,
        description: Optional[str] = None,
    ):
        METRICS.increment("discord_commands.schedule_add")
        if self.schedule_manager is None:
            await interaction.response.send_message(
                "Scheduler is not available.", ephemeral=True
            )
            return
        try:
            schedule_id = self.schedule_manager.add_schedule(
                user_id=_identity(interaction),
                channel_id=interaction.channel_id,
                guild_id=interaction.guild_id,
                prompt=prompt,
                schedule_expr=every,
                description=description or "",
            )
            await interaction.response.send_message(
                f"✅ Schedule `{schedule_id}` created.\n"
                f"**Every:** `{every}`\n"
                f"**Prompt:** {prompt}",
                ephemeral=True,
            )
        except ValueError as e:
            await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        except Exception as e:
            await _reply_error(interaction, "discord_commands.schedule_add", e)

    @schedule.command(name="list", description="List your active scheduled tasks")
    async def schedule_list(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.schedule_list")
        if self.schedule_manager is None:
            await interaction.response.send_message("Scheduler is not available.", ephemeral=True)
            return
        schedules = self.schedule_manager.list_schedules(_identity(interaction))
        if not schedules:
            await interaction.response.send_message(
                "You have no scheduled tasks. Use `/schedule add` to create one.", ephemeral=True
            )
            return
        embed = discord.Embed(title="Your scheduled tasks", colour=FRITZ_COLOUR)
        for s in schedules[:25]:          # Discord caps an embed at 25 fields
            label = f" — {s['description']}" if s["description"] else ""
            embed.add_field(
                name=f"`{s['id']}` · every `{s['schedule']}`{label}",
                value=s["prompt"][:1024],  # per-field value cap
                inline=False,
            )
        if len(schedules) > 25:
            embed.set_footer(text=f"Showing 25 of {len(schedules)}.")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @schedule.command(name="remove", description="Remove a scheduled task by its ID")
    @app_commands.describe(schedule_id="The schedule ID shown in /schedule list")
    async def schedule_remove(self, interaction: discord.Interaction, schedule_id: str):
        METRICS.increment("discord_commands.schedule_remove")
        if self.schedule_manager is None:
            await interaction.response.send_message("Scheduler is not available.", ephemeral=True)
            return
        try:
            removed = self.schedule_manager.remove_schedule(schedule_id, _identity(interaction))
            if removed:
                await interaction.response.send_message(
                    f"✅ Schedule `{schedule_id}` removed.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"❌ No schedule found with ID `{schedule_id}`.", ephemeral=True
                )
        except PermissionError as e:
            await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        except Exception as e:
            await _reply_error(interaction, "discord_commands.schedule_remove", e)

    @schedule.command(name="list_all", description="(Admin) List scheduled tasks across all users")
    async def schedule_list_all(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.schedule_list_all")
        if not await _require_admin(interaction):
            return
        if self.schedule_manager is None:
            await interaction.response.send_message("Scheduler is not available.", ephemeral=True)
            return
        schedules = self.schedule_manager.list_all_schedules()
        if not schedules:
            await interaction.response.send_message("No schedules registered.", ephemeral=True)
            return
        lines = [f"**All scheduled tasks ({len(schedules)}):**"]
        for s in schedules:
            label = f" — {s['description']}" if s["description"] else ""
            lines.append(
                f"`{s['id']}` (@{s['user_id']}) every `{s['schedule']}`{label}\n  _{s['prompt']}_"
            )
        body = "\n".join(lines)
        # Long lists chunk into multiple ephemeral replies to respect the 2000-char limit.
        chunks = split_into_chunks(body)
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

    # ── Privacy: /forget and /export ─────────────────────────────────────────

    forget = app_commands.Group(name="forget", description="Delete data Fritz has stored about you")

    @forget.command(name="memories", description="Delete every memory and profile entry Fritz has saved about you")
    async def forget_memories_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.forget.memories")
        user_id = _identity(interaction)
        count = await run_blocking(privacy.forget_memories, user_id)
        audit_log("forget", user_id=user_id, scope="memories", removed=count)
        await interaction.response.send_message(
            f"✅ Removed {count} memory entry(ies).", ephemeral=True,
        )

    @forget.command(name="conversation", description="Reset your conversation thread — next message starts fresh")
    async def forget_conversation_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.forget.conversation")
        user_id = _identity(interaction)
        count = await run_blocking(privacy.forget_conversation, user_id)
        audit_log("forget", user_id=user_id, scope="conversation", removed=count)
        await interaction.response.send_message(
            f"✅ Cleared {count} checkpoint row(s). Your next message starts a fresh thread.",
            ephemeral=True,
        )

    @forget.command(name="schedules", description="Cancel every scheduled task you have")
    async def forget_schedules_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.forget.schedules")
        user_id = _identity(interaction)
        count = await run_blocking(privacy.forget_schedules, user_id, self.schedule_manager)
        audit_log("forget", user_id=user_id, scope="schedules", removed=count)
        await interaction.response.send_message(
            f"✅ Cancelled {count} schedule(s).", ephemeral=True,
        )

    @forget.command(
        name="all",
        description="Delete EVERYTHING Fritz has stored about you (memories, conversation, schedules, workspace)",
    )
    async def forget_all_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.forget.all")
        user_id = _identity(interaction)
        # Two-step confirmation: present a confirm/cancel view to the user.
        view = _ForgetConfirmView(user_id, self.schedule_manager)
        await interaction.response.send_message(
            "⚠️ This will permanently delete:\n"
            "• all stored memories and your profile\n"
            "• your conversation history checkpoint\n"
            "• every scheduled task you have\n"
            "• your workspace registration (files on disk are kept)\n"
            "\nClick **Confirm** within 30 seconds to proceed.",
            view=view, ephemeral=True,
        )

    @app_commands.command(
        name="export",
        description="Download a JSON snapshot of all data Fritz has stored about you",
    )
    async def export_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.export")
        user_id = _identity(interaction)
        await interaction.response.defer(ephemeral=True, thinking=True)
        data = await run_blocking(privacy.export_user_data, user_id, self.schedule_manager)
        payload = json.dumps(data, indent=2, default=str).encode("utf-8")
        # Discord's free-tier per-message attachment cap is 25 MB; we cap at 8 MB
        # as a comfortable safety margin since exports should be tiny.
        if len(payload) > 8 * 1024 * 1024:
            await interaction.followup.send(
                f"❌ Export is too large ({len(payload) / 1024 / 1024:.1f} MB). "
                "Please run `/forget memories` first to trim, then try again.",
                ephemeral=True,
            )
            return
        audit_log("export", user_id=user_id, bytes=len(payload))
        attachment = discord.File(
            io.BytesIO(payload), filename=f"misterfritz_export_{user_id}.json",
        )
        await interaction.followup.send(
            "Here's your data. Stored locally on this server — nothing was sent to a third party.",
            file=attachment, ephemeral=True,
        )

    # ── Card game ─────────────────────────────────────────────────────────────

    @app_commands.command(name="draw", description="Draw cards from a deck")
    @app_commands.describe(num_cards="How many cards to draw (1-40)")
    async def draw_slash(self, interaction: discord.Interaction,
                         num_cards: app_commands.Range[int, 1, 40]):
        # Unbounded before: ~48 cards crosses Discord's 2000-char cap and
        # followup.send raised an uncaught HTTPException, so the user saw only
        # a spinner. Range rejects it up front; the chunking below is belt and
        # braces, and keeps the trailing ``` summary block intact when it does
        # need to split.
        METRICS.increment("discord_commands.draw")
        await interaction.response.defer(thinking=True)
        drawn = draw_cards(num_cards, _identity(interaction), _display(interaction))
        # split_into_chunks returns [] for empty input, which would make the
        # loop below a no-op — and a deferred interaction that never receives a
        # followup leaves the user staring at "thinking…" until it expires.
        # Something must always be sent.
        for chunk in split_into_chunks(drawn) or [
            "The deck yielded nothing, which I did not think possible."
        ]:
            await interaction.followup.send(content=chunk)

    @app_commands.command(name="cards_remaining", description="Check cards remaining in the deck")
    async def cards_remaining_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        await interaction.followup.send(
            content=get_remaining_card_number(_identity(interaction), _display(interaction))
        )

    @app_commands.command(name="reload_deck", description="Reloads the deck (use if you goof up)")
    async def reload_deck_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        await interaction.followup.send(
            content=reload_deck(_identity(interaction), _display(interaction))
        )

    # ── General ───────────────────────────────────────────────────────────────

    @app_commands.command(name="hello", description="Say hello to the bot")
    async def hello_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.hello")
        await interaction.response.send_message(f"Hello, {_display(interaction)}!")

    @app_commands.command(name="health", description="Check the system health metrics")
    async def health_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.health")
        snap = get_health_snapshot()
        total_errors = sum(snap["errors"].values()) if snap["errors"] else 0
        embed = discord.Embed(title="Mister Fritz — status", colour=FRITZ_COLOUR)
        embed.add_field(name="Uptime",
                        value=_format_uptime(int(snap["uptime_sec"])), inline=True)
        embed.add_field(name="Messages",
                        value=str(snap["counters"].get("discord_messages", 0)), inline=True)
        embed.add_field(name="Errors", value=str(total_errors), inline=True)
        if snap["latencies"]:
            lat = "\n".join(
                f"{n}: {s['avg_sec']:.2f}s (n={s['count']})"
                for n, s in snap["latencies"].items()
            )
            embed.add_field(name="Latency", value=lat[:1024], inline=False)
        if snap["last_error"]:
            # Name only — the message could carry a path or a token.
            name, _ts, _msg = snap["last_error"]
            embed.set_footer(text=f"Last error: {name}")
        # Ephemeral now: this was the one interaction reply in the file that
        # posted metrics into channel history for everyone to read.
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="help", description="Show what Mister Fritz can do")
    async def help_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.help")
        embed = discord.Embed(
            title="Mister Fritz — at your service",
            description="DM me, or `@mention` me in a channel, and I shall run the full agent.",
            colour=FRITZ_COLOUR,
        )
        embed.add_field(name="How to talk to me", value=(
            "• DM or `@mention` — the full agent\n"
            "• Attach an image and I shall analyse it\n"
            "• Send a voice message and I shall transcribe it"
        ), inline=False)
        embed.add_field(name="Tools at my disposal", value=(
            "• Web search and page scraping\n"
            "• Local document RAG (drop files in `input/`)\n"
            "• Per-user memory of past conversations\n"
            "• Image generation and analysis\n"
            "• Dice, the time, and scheduled reminders"
        ), inline=False)
        embed.add_field(name="Slash commands", value=(
            "• `/lore <query>` — search local documents\n"
            "• `/gen <prompt>` — generate an image\n"
            "• `/voice <message>` — synthesise speech\n"
            "• `/join` · `/leave` — voice channel\n"
            "• `/draw <n>` · `/cards_remaining` · `/reload_deck`\n"
            "• `/schedule add|list|remove`\n"
            "• `/health` · `/about`\n"
            "• `/workspace <path>` — admin only"
        ), inline=False)
        embed.set_footer(text="Run /about for version and storage details.")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="about", description="Show version, models, and data storage info")
    async def about_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.about")
        snap = get_health_snapshot()
        # BUG FIX: the key is "uptime_sec" (observability.py), not
        # "uptime_seconds". The .get default silently swallowed the miss, so
        # /about reported "0s" on every single invocation.
        uptime = _format_uptime(int(snap.get("uptime_sec", 0)))
        embed = discord.Embed(
            title=f"Mister Fritz v{__version__}",
            description="_An AI butler of impeccable bearing and barely-concealed weariness._",
            colour=FRITZ_COLOUR,
            url="https://github.com/NickSanft/MisterFritz",
        )
        embed.add_field(name="Thinking model", value=f"`{THINKING_OLLAMA_MODEL}`", inline=True)
        embed.add_field(name="Fast model", value=f"`{FAST_OLLAMA_MODEL}`", inline=True)
        embed.add_field(name="Uptime", value=uptime, inline=True)
        embed.add_field(name="Data storage", value=(
            "Conversation summaries, memories, and schedules live on this host "
            "and nowhere else. Documents dropped in `input/` are indexed into a "
            "local ChromaDB."
        ), inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ── AI / content ──────────────────────────────────────────────────────────

    @app_commands.command(name="voice", description="Generate audio from text")
    @app_commands.describe(message="The text you want the bot to say")
    async def voice_slash(self, interaction: discord.Interaction, message: str):
        await interaction.response.defer(thinking=True)
        if self.sayer is None:
            # Core-only install: the [voice] extra is not present, so there is
            # no TTS engine. Say so plainly rather than raising AttributeError
            # into the generic error handler.
            await interaction.followup.send(
                "I regret that I have no voice on this instance. "
                "Install the optional speech dependencies with "
                "`pip install \".[voice]\"` and restart me.",
                ephemeral=True,
            )
            return
        try:
            with METRICS.time_block("discord_commands.voice"):
                response_data = await run_blocking(
                    ask_stuff, message, MessageSource.DISCORD_VOICE, _identity(interaction),
                    display_name=_display(interaction),
                    channel_key=str(interaction.channel_id) if interaction.channel_id else None,
                )
                original_response = response_data["text"]
            # Outside the agent timer, and the synthesis is timed separately:
            # a /voice that waits on another user's synthesis should not report
            # that wait as its own work. See the note in gen_slash.
            queued_at = time.perf_counter()
            async with self._tts_semaphore:
                METRICS.record_latency("discord_commands.voice.tts.queue",
                                       time.perf_counter() - queued_at)
                with METRICS.time_block("discord_commands.voice.tts"):
                    output_file = await run_blocking(
                        self.sayer.generate_speech, original_response,
                    )
        except Exception as e:
            # Previously absent: a failure in either call raised out of the
            # handler, leaving the deferred interaction hanging with a spinner
            # until Discord timed it out.
            # record=False — see the note in gen_slash.
            await _reply_error(interaction, "discord_commands.voice", e, record=False)
            return
        try:
            if interaction.guild and interaction.guild.voice_client:
                interaction.guild.voice_client.play(
                    discord.FFmpegPCMAudio(executable=FFMPEG_PATH, source=output_file)
                )
                await interaction.followup.send(f"Playing voice for: '{message}'")
            else:
                await interaction.followup.send(
                    "You are not connected to a voice channel, uploading as a file...",
                    files=[discord.File(output_file)],
                )
        except AttributeError as e:
            # No record= here: this block is OUTSIDE the time_block above, so
            # nothing has recorded this failure yet.
            await _reply_error(interaction, "discord_commands.voice", e)

    @app_commands.command(name="gen", description="Generate an image based on a prompt")
    @app_commands.describe(prompt="The image description")
    async def gen_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(thinking=True)
        logger.info("Image generation request: %s", prompt)
        try:
            if self._image_semaphore.locked():
                await interaction.followup.send(
                    "\U0001f5bc️ Queued — another image is rendering."
                )
            # The semaphore is acquired OUTSIDE the timer. Enclosing it meant
            # a queued /gen reported another user's render as its own latency:
            # with IMAGE_GEN_MAX_CONCURRENCY=1 the recorded time was mostly
            # queue wait, which made the histogram unusable for answering "is
            # generation slow?". Queue time is still measured — separately,
            # where it answers a different and equally useful question.
            queued_at = time.perf_counter()
            async with self._image_semaphore:
                METRICS.record_latency("discord_commands.gen.queue",
                                       time.perf_counter() - queued_at)
                with METRICS.time_block("discord_commands.gen"):
                    # The import goes INSIDE the offloaded callable, not beside
                    # it. Deferring it off module scope keeps [image] optional,
                    # but executing it here would still run image_generator's
                    # module body — diffusers, torch, xformers and a
                    # torch.cuda.is_available() probe, ~10s measured — on the
                    # event loop, which is the freeze this offload exists to
                    # remove. Only the first /gen per process pays it; after
                    # that it is a sys.modules hit inside the worker thread.
                    output_file = await run_blocking(_render_image, prompt)
            await interaction.followup.send(content="Here is your file!", file=discord.File(output_file))
        except Exception as e:
            # record=False: time_block already recorded this failure and
            # re-raised it, so recording again double-counts the error.
            await _reply_error(interaction, "discord_commands.gen", e, record=False)

    @app_commands.command(name="lore", description="Query the document engine for lore")
    @app_commands.describe(query="The question about the lore")
    async def lore_slash(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer(thinking=True)
        logger.info("Lore request: %s", query)
        # The first call also triggers document_engine.initialize_vectorstore(),
        # which walks DOC_FOLDER and ingests the whole corpus — unbounded
        # first-call latency, previously on the event loop.
        with METRICS.time_block("discord_commands.lore"):
            original_response = await run_blocking(query_documents, query)
        # Continuations go through followup, not channel.send: the latter posts
        # detached messages that can interleave with other traffic. The old
        # "The answer was over 2000 …" header announced an implementation
        # detail nobody asked about.
        # Empty is a real outcome here — the corpus may hold nothing on the
        # subject — and split_into_chunks returns [] for it, so without a
        # fallback the deferred interaction never gets a followup and the user
        # watches "thinking…" until it expires.
        for chunk in split_into_chunks(original_response) or [
            "The library holds nothing on that, sir."
        ]:
            await interaction.followup.send(chunk)

    # ── Voice channel ─────────────────────────────────────────────────────────

    @app_commands.command(name="join", description="Join the voice channel you are currently in")
    async def join_slash(self, interaction: discord.Interaction):
        try:
            METRICS.increment("discord_commands.join")
            if interaction.user.voice and interaction.user.voice.channel:
                channel = interaction.user.voice.channel
                await channel.connect()
                await interaction.response.send_message(f"Joined {channel.name}!")
            else:
                await interaction.response.send_message(
                    "You are not in a voice channel, sir. I cannot join you "
                    "somewhere you are not.", ephemeral=True
                )
        except Exception as e:
            await _reply_error(interaction, "discord_commands.join", e,
                               note="The channel declined my company. It happens.")

    @app_commands.command(name="leave", description="Leave the current voice channel")
    async def leave_slash(self, interaction: discord.Interaction):
        try:
            METRICS.increment("discord_commands.leave")
            if interaction.guild.voice_client:
                await interaction.guild.voice_client.disconnect()
                await interaction.response.send_message("Disconnected.")
            else:
                await interaction.response.send_message(
                    "I am not in a voice channel. One cannot leave a room one "
                    "was never in.", ephemeral=True
                )
        except Exception as e:
            await _reply_error(interaction, "discord_commands.leave", e,
                               note="I appear to be stuck. How undignified.")

    # ── File workspace ────────────────────────────────────────────────────────

    workspace = app_commands.Group(name="workspace", description="Manage the file-tools workspace")

    @workspace.command(name="status", description="Show your current workspace, if any")
    async def workspace_status(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.workspace.status")
        author = _identity(interaction)
        current = workspace_store.get(author)
        if current:
            await interaction.response.send_message(
                f"Your workspace: `{current}`", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "You have no workspace. Run `/workspace enable` to create a sandboxed one.",
                ephemeral=True,
            )

    @workspace.command(
        name="enable",
        description="Create a sandboxed workspace for yourself and enable file tools",
    )
    async def workspace_enable(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.workspace.enable")
        author = _identity(interaction)
        try:
            path = workspace_store.enable_sandboxed(author)
        except Exception as e:
            await _reply_error(interaction, "discord_commands.workspace_enable", e)
            return
        await interaction.response.send_message(
            f"✅ Workspace ready at `{path}`.\n"
            "File tools (read, write, edit, search, list, run) are now active in "
            "your conversations. Drop files in that directory to give Fritz access.",
            ephemeral=True,
        )

    @workspace.command(name="disable", description="Forget your workspace (files on disk are kept)")
    async def workspace_disable(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.workspace.disable")
        author = _identity(interaction)
        removed = workspace_store.remove(author)
        if removed:
            await interaction.response.send_message(
                "✅ Workspace disabled. Your files were not deleted — re-enable any time.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "You don't have a workspace to disable.", ephemeral=True
            )

    @workspace.command(
        name="set",
        description="(Admin) Register an existing directory as your workspace",
    )
    @app_commands.describe(path="Absolute or ~ path to a directory on the bot host")
    async def workspace_set(self, interaction: discord.Interaction, path: str):
        METRICS.increment("discord_commands.workspace.set")
        if not await _require_admin(interaction):
            return
        author = _identity(interaction)
        expanded = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(expanded):
            await interaction.response.send_message(
                f"Directory does not exist: `{expanded}`", ephemeral=True
            )
            return
        workspace_store.set_path(author, expanded)
        await interaction.response.send_message(
            f"✅ Workspace set to `{expanded}`. File tools active in conversations.",
            ephemeral=True,
        )

    # ── Direct-message relay ──────────────────────────────────────────────────
    # plans/12-direct-message-relay.md. The admission rules — caps, blocks, and
    # the order that keeps a block unprobeable — live in relay_store, not here,
    # so the phase-2 agent tool cannot route around them.

    # guild_only alone serialises only the deprecated dm_permission field;
    # allowed_contexts sends the current one. allowed_installs pins HOW Fritz
    # is present: without it, a copy of Fritz user-installed into someone's
    # account can run /tell from a server Fritz was never invited to, and DM
    # its members. All three on the Group, because on a subcommand they are
    # silently dropped; and all three are enforced by Discord, not by the
    # library, so tell_message checks both conditions itself as well.
    tell = app_commands.Group(
        name="tell",
        description="Carry a message to someone in this server, by DM",
        guild_only=True,
        allowed_contexts=app_commands.AppCommandContext(guild=True),
        allowed_installs=app_commands.AppInstallationType(guild=True),
    )

    @tell.command(name="message",
                  description="Deliver your exact words to someone in this server, by DM")
    @app_commands.describe(
        recipient="Who to tell. They must be a member of this server.",
        message="Your words. They are delivered exactly as written, under your name.",
    )
    async def tell_message(self, interaction: discord.Interaction,
                           recipient: discord.Member,
                           message: app_commands.Range[str, 1, TELL_MAX_CHARS]):
        # `recipient: discord.Member`, not User: Discord resolves a Member only
        # for someone actually in this guild, with no member cache and no HTTP
        # call, and the library rejects anything else before this runs. That IS
        # the "anyone in a shared server" rule. User.mutual_guilds would be the
        # obvious alternative and is a trap: it scans the member cache, and the
        # members intent is off, so it would silently say no.
        METRICS.increment("discord_commands.tell.message")
        # First, always. Opening the DM and sending is two round trips against
        # a three-second acknowledgement deadline, and either can sleep on a 429.
        await interaction.response.defer(ephemeral=True, thinking=True)
        # Drawn now, before anything is decided, so that a block and a 403 are
        # measured from the same start. See _REFUSAL_WINDOW_SEC.
        deadline = _refusal_deadline()

        if interaction.guild_id is None or not interaction.is_guild_integration():
            await _answer(interaction,
                          "I carry messages between members of a server I have been "
                          "invited to. Do ask me from inside one. " + _NOTHING_SENT)
            return
        if not message.strip():
            # An embed with a blank description does not error; it simply
            # arrives empty. Discord's min_length does not stop whitespace.
            await _answer(interaction, "There is nothing there to carry. " + _NOTHING_SENT)
            return

        sender_id = _identity(interaction)
        recipient_id = canonical_user_id("discord", recipient.id)
        fritz = getattr(self.bot, "user", None)
        try:
            outcome = await run_blocking(
                relay_store.reserve_send, sender_id, recipient_id, message,
                guild_id=interaction.guild_id,
                recipient_is_bot=bool(recipient.bot),
                recipient_is_fritz=fritz is not None and recipient.id == fritz.id,
            )
        except Exception as e:
            await _reply_error(interaction, "relay.reserve", e,
                               note="I could not arrange that. " + _NOTHING_SENT)
            return

        if not outcome.ok:
            METRICS.increment(f"relay.denied.{outcome.reason}")
            audit_log("relay_denied", sender=sender_id, recipient=recipient_id,
                      guild_id=interaction.guild_id, reason=outcome.reason,
                      chars=len(message), body_digest=_audit_digest(message))
            if outcome.relay_id is not None:
                # A block. It left a reservation open, exactly as a send
                # heading for a 403 does, and is settled the same way.
                await self._refuse(interaction, outcome.relay_id, deadline)
                return
            await _answer(interaction, outcome.message)
            return

        await self._deliver_relay(interaction, outcome, recipient, message, deadline)

    async def _refuse(self, interaction: discord.Interaction, relay_id: str,
                      deadline: float) -> None:
        """Settle and answer a recipient-side refusal, on the shared schedule.

        Both kinds come through here — a block, and Discord's 403 — so that
        they match in what the sender reads (REFUSED), in the row left behind
        (refused, charged to the sender), in when the answer arrives, and in
        when the reservation stops counting toward the recipient's inbox.
        Review found each of those, in turn, telling them apart.
        """
        await _sleep_until(deadline)
        await _settle(relay_store.mark_refused, relay_id)
        METRICS.increment("relay.refused")
        await _answer(interaction, relay_store.REFUSED)

    async def _deliver_relay(self, interaction: discord.Interaction,
                             reservation: "relay_store.Reservation",
                             recipient: discord.Member, message: str,
                             deadline: float) -> None:
        guild = interaction.guild
        where = _plain(guild.name) if guild is not None and guild.name else ""
        where = where or "a server you share"
        embed = discord.Embed(description=message, colour=FRITZ_COLOUR,
                              timestamp=discord.utils.utcnow())
        embed.set_author(name=_relay_author_line(interaction.user),
                         icon_url=interaction.user.display_avatar.url)
        embed.set_footer(text=f"Sent with /tell from {where}. "
                              "The words are theirs; I merely carry them.")

        # One try covers both round trips: Member.send opens the DM channel
        # itself, and either request can be the one that fails. Forbidden and
        # NotFound are HTTPException subclasses, so they come first; RateLimited
        # is NOT one, so it must precede the HTTPException clause or it would
        # fall through to the generic handler.
        audit = dict(sender=reservation.sender_id, recipient=reservation.recipient_id,
                     relay_id=reservation.id, guild_id=interaction.guild_id,
                     chars=len(message), body_digest=_audit_digest(message))
        try:
            sent = await recipient.send(embed=embed,
                                        allowed_mentions=discord.AllowedMentions.none())
        except discord.Forbidden as e:
            # The recipient's side refused: DMs closed, or Fritz blocked.
            # Terminal — never retried, because repeated 403s to a closed inbox
            # are what Discord's anti-spam reads as abuse, and enforcement
            # lands on the bot's token.
            logger.info("relay %s refused by Discord: status=%s code=%s",
                        reservation.id, e.status, e.code)
            audit_log("relay_refused", discord_code=e.code, **audit)
            await self._refuse(interaction, reservation.id, deadline)
            return
        except discord.NotFound as e:
            await self._relay_failed(interaction, reservation, audit, e,
                                     "Discord could not find that account. " + _NOTHING_SENT)
            return
        except discord.RateLimited as e:
            await self._relay_failed(
                interaction, reservation, audit, e,
                f"Discord has asked me to wait {e.retry_after:.0f} seconds before "
                f"sending more. {_NOTHING_SENT} Do try again shortly.")
            return
        except discord.DiscordServerError as e:
            await self._relay_failed(
                interaction, reservation, audit, e,
                f"Discord is having difficulties of its own. {_NOTHING_SENT} "
                "Do try again shortly.")
            return
        except discord.HTTPException as e:
            if e.status == 400:
                # The content itself: Discord's harmful-link filter, or a body
                # it will not accept. The sender's doing, so the sender's
                # charge — see relay_store.STATUS_REJECTED.
                await self._relay_failed(
                    interaction, reservation, audit, e,
                    f"Discord declined the message as written. {_NOTHING_SENT}",
                    mark=relay_store.mark_rejected)
                return
            if e.status == 429:
                note = f"Discord is rate-limiting me. {_NOTHING_SENT} Try again in a minute."
            else:
                note = f"Discord would not take that message. {_NOTHING_SENT}"
            await self._relay_failed(interaction, reservation, audit, e, note)
            return
        except Exception as e:
            await self._relay_failed(interaction, reservation, audit, e,
                                     f"Something went wrong on my side. {_NOTHING_SENT}")
            return

        recorded = True
        try:
            await run_blocking(relay_store.mark_sent, reservation.id,
                               sent.id, sent.channel.id)
        except Exception as e:
            # Delivered, and it cannot be un-delivered. Say so, rather than
            # reporting a failure that did not happen.
            recorded = False
            logger.error("relay %s delivered but not recorded: %s", reservation.id, e)
        METRICS.increment("relay.delivered")
        audit_log("relay_sent", **audit)

        receipt = await self._relay_receipt(interaction, recipient, message, where)
        confirm = f"Delivered to {recipient.mention}."
        if not receipt:
            confirm += (" I could not leave you a copy in your own DMs; if you "
                        "have them closed, you will have no record of what you sent.")
        if not recorded:
            confirm += " I failed to note it down, however, so I cannot vouch for what happens next."
        await _answer(interaction, confirm,
                      allowed_mentions=discord.AllowedMentions.none())

    async def _relay_failed(self, interaction: discord.Interaction,
                            reservation: "relay_store.Reservation", audit: dict,
                            exc: BaseException, note: str, *,
                            mark=relay_store.mark_failed) -> None:
        """The send failed. Always answered, with a ref for the log.

        mark_failed (the default) is free to the sender: an outage, a 429,
        a bug of ours. mark_rejected is charged: Discord refused the content.
        """
        status = getattr(exc, "status", None)
        reason = type(exc).__name__ + (f" {status}" if status else "")
        await _settle(mark, reservation.id, reason)
        METRICS.increment("relay.failed")
        audit_log("relay_failed", error=reason, **audit)
        await _reply_error(interaction, "relay.send", exc, note=note)

    async def _relay_receipt(self, interaction: discord.Interaction,
                             recipient: discord.Member, message: str,
                             where: str) -> bool:
        """Leave the sender a copy of what went, in their own DMs.

        Only ever called after a delivery. Best-effort: the relay already
        stands. Mentions are suppressed on this leg too.
        """
        embed = discord.Embed(description=message, colour=FRITZ_COLOUR,
                              timestamp=discord.utils.utcnow())
        embed.set_author(name=f"To {_relay_author_line(recipient)}"[:256],
                         icon_url=recipient.display_avatar.url)
        embed.set_footer(text=f"Your /tell from {where}, as delivered.")
        try:
            await interaction.user.send(embed=embed,
                                        allowed_mentions=discord.AllowedMentions.none())
        except Exception as e:
            logger.info("relay receipt not delivered to sender: %s", type(e).__name__)
            return False
        return True
