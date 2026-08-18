import asyncio
import io
import json
import logging
import os
import time
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
    except discord.HTTPException:
        logger.warning("Could not deliver error reply for %s", operation)


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
    if isinstance(error, app_commands.TransformerError):
        await _reply_error(interaction, f"app_command.{name}", None,
                           note="That value is outside the permitted range. Do try one I can work with.")
        return
    if isinstance(original, discord.HTTPException):
        await _reply_error(interaction, f"app_command.{name}", original,
                           note="Discord declined to carry that message. It was, I suspect, too long.")
        return
    await _reply_error(interaction, f"app_command.{name}", original)


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
