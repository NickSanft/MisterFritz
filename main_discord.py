import asyncio
import logging
import os
import time
import uuid

import discord
from discord.ext import commands

from admin_panel import start_admin_panel
from bot_adapters import fritz_error, run_blocking, split_into_chunks  # noqa: F401 — split_into_chunks re-exported for tests
from bot_commands import FritzCommands, handle_app_command_error
from fritz_utils import (
    DISCORD_BOT_TOKEN,
    DISCORD_STREAM_MIN_INTERVAL,
    EMBEDDING_MODEL,
    FAST_OLLAMA_MODEL,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    THINKING_OLLAMA_MODEL,
    VISION_MODEL,
    canonical_user_id,
    safe_user_token,
    validate_config,
)
import identity_store
from mister_fritz import ask_stuff
from observability import METRICS, init_logging, start_metrics_server
from prewarm import prewarm_models
from scheduler import ScheduleManager
# stt is safe at module level: it imports pydub (core) and defers the Whisper
# model itself. tts is NOT — see the deferred import in on_ready.
from stt import transcribe as _whisper_transcribe
import workspace_store

init_logging()
logger = logging.getLogger(__name__)


class StreamingMessageHandler:
    """Manages incremental Discord message updates with rate limiting."""

    def __init__(self, message: discord.Message, loop: asyncio.AbstractEventLoop,
                 min_update_interval: float = DISCORD_STREAM_MIN_INTERVAL):
        self.message = message
        self.loop = loop
        self.min_update_interval = min_update_interval
        self.current_text = ""
        # What was last actually sent to Discord, i.e. status + body. Tracked
        # separately from current_text because the status can change while the
        # body does not, and the dedup guard below must notice that.
        self.current_rendered = ""
        self.last_update_time = 0
        self.update_task = None
        self.is_updating = False
        self.pending_text = None
        # Tool/plan progress, rendered inside this same placeholder message
        # rather than as separate permanent sends.
        self.status_text: str | None = None

    def _compose(self, body: str) -> str:
        """Render the status line plus the streamed body, tail-windowed to fit.

        The old `[:2000]` head-truncation froze the visible text the moment a
        reply passed the cap — the user watched a stationary prefix while
        tokens kept arriving. A tail window keeps the newest text on screen;
        the complete reply is delivered by final_update's chunking.
        """
        head = f"{self.status_text}\n\n" if self.status_text else ""
        room = 2000 - len(head)
        if len(body) > room:
            body = "…" + body[-(room - 1):]
        return head + body

    async def set_status(self, status: str | None):
        """Show or clear the progress line inside the placeholder message."""
        self.status_text = status
        await self.update_text(self.pending_text or self.current_text)

    async def update_text(self, new_text: str):
        """Update the message with new text, respecting rate limits."""
        self.pending_text = new_text
        if not self.is_updating:
            self.is_updating = True
            await self._perform_update()

    async def _perform_update(self):
        """Perform the actual message edit with rate limiting."""
        while self.is_updating:
            # Compare the COMPOSED output, not just the body: a new status line
            # over unchanged text is still a change the user needs to see, and
            # comparing bodies alone would silently swallow it.
            #
            # `is not None`, NOT truthiness. A status arriving before the first
            # token — the common case for a tool-using turn, where the notice
            # is the whole point — has pending_text == "", which is falsy. The
            # old guard skipped the edit entirely, so the placeholder sat on
            # "Mister Fritz is thinking..." and the user never saw "Searching
            # the web..." until tokens started arriving, by which point it was
            # stale. A status alone is reason enough to render.
            body = self.pending_text if self.pending_text is not None else self.current_text
            rendered = self._compose(body) if (self.pending_text is not None or self.status_text) else None
            if rendered is not None and rendered != self.current_rendered:
                time_since_last = time.time() - self.last_update_time
                if time_since_last < self.min_update_interval:
                    await asyncio.sleep(self.min_update_interval - time_since_last)
                try:
                    await self.message.edit(content=rendered)
                    self.current_text = body
                    self.current_rendered = rendered
                    self.last_update_time = time.time()
                    self.pending_text = None
                except discord.errors.HTTPException as e:
                    logger.warning("Error editing message: %s", e)
                    await asyncio.sleep(2)
            else:
                self.is_updating = False

    async def final_update(self, final_text: str, files: list = None):
        """Deliver the complete reply: the placeholder gets chunk 1, the rest follow.

        Chunking lives here rather than in on_message, which used to re-chunk
        the same text a second time with its own copy of the logic.
        """
        while self.is_updating:
            await asyncio.sleep(0.1)
        # The progress line disappears along with the reply it described.
        self.status_text = None
        chunks = split_into_chunks(final_text) or [final_text]
        try:
            await self.message.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await self.message.channel.send(chunk)
            if files:
                await self.message.channel.send(files=files)
        except discord.errors.HTTPException as e:
            logger.warning("Error in final update: %s", e)


def _cleanup_temp_files(paths: list[str], request_id: str) -> None:
    """Remove temporary files created during message processing."""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Cleaned up %s for %s", path, request_id)
        except Exception as e:
            METRICS.record_error("cleanup_error", e)
            logger.warning("Error cleaning up %s: %s", path, e)


command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)

sayer = None
schedule_manager = None


@client.event
async def on_ready():
    global sayer, schedule_manager
    # No `loop` binding needed here any more — run_blocking gets its own.
    # on_message still binds one, for run_coroutine_threadsafe in the
    # streaming/progress callbacks; do not remove that one.
    if sayer is None:
        # Both the import AND the construction go into the worker thread.
        #
        # Deferring the import off module scope keeps the [voice] extra
        # optional, but running it here would still execute tts's module body —
        # torch plus TTS.api, ~17s measured — on the event loop. That is long
        # enough to trip discord.py's "heartbeat blocked for more than 10
        # seconds" warning on every boot with [voice] installed. Absent the
        # extra, /voice reports itself unavailable and everything else works.
        def _load_tts():
            from tts import TTSEngine
            return TTSEngine()

        logger.info("Loading TTS engine...")
        try:
            sayer = await run_blocking(_load_tts)
        except ImportError as e:
            logger.warning(
                "TTS unavailable (%s). /voice is disabled; install the [voice] "
                "extra to enable it.", e,
            )
        else:
            logger.info("TTS engine ready")
    schedule_manager = ScheduleManager(client)
    schedule_manager.start()
    # Start the read-only admin panel (no-op if ADMIN_PANEL_PASSWORD is unset).
    start_admin_panel(schedule_manager=schedule_manager)
    # Pre-warm Ollama in a daemon thread so the first DM doesn't pay model-load lag.
    prewarm_models(
        chat_models=(THINKING_OLLAMA_MODEL, FAST_OLLAMA_MODEL, VISION_MODEL),
        embedding_models=(EMBEDDING_MODEL,),
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    await client.add_cog(FritzCommands(client, sayer, schedule_manager))
    # Backstop for app-command failures raised OUTSIDE the cog (e.g. a stale
    # sync producing CommandNotFound), which the cog hook never sees.
    client.tree.on_error = handle_app_command_error
    logger.info("Logged in as %s", client.user)
    try:
        synced = await client.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


def make_streaming_callback(handler, loop, min_interval=None):
    """Build the streaming callback ask_stuff invokes from a worker thread.

    A module-level factory rather than a closure inside on_message purely so it
    can be tested: the arity, the cross-thread rate limit, and the fact that a
    restart bypasses that limit are all easy to break and were covered by
    nothing, because reaching them meant standing up a live gateway.

    A Discord edit replaces the entire message body, so this sends
    `accumulated` (which resets by itself when `restart` fires), not the delta.

    Called once per TOKEN. At ~40 tokens/s that would queue 40 coroutines/s
    onto the gateway loop, so the cross-thread hop itself is rate-limited — the
    handler's own throttle only paces the edits, not the scheduling.
    final_update() always writes the complete text, so a dropped hop loses
    nothing. A restart is never dropped: it marks a NEW answer segment, and
    skipping it would leave the previous segment's text on screen.
    """
    if min_interval is None:
        min_interval = DISCORD_STREAM_MIN_INTERVAL
    state = {"last": 0.0}

    def streaming_callback(delta: str, accumulated: str, restart: bool = False):
        now = time.monotonic()
        if not restart and now - state["last"] < min_interval:
            return
        state["last"] = now
        asyncio.run_coroutine_threadsafe(handler.update_text(accumulated), loop)

    return streaming_callback


@client.event
async def on_message(ctx):
    # THE IDENTITY BOUNDARY. Everything downstream keys off `user_id`, which is
    # the immutable snowflake — not the display name, which Discord lets people
    # change at will and which collides across platforms. `author` survives only
    # for the prompt and for filenames.
    user_id = canonical_user_id("discord", ctx.author.id)
    author = ctx.author.display_name or ctx.author.name
    channel = ctx.channel
    message_clean = ctx.clean_content
    if ctx.author == client.user:
        return
    elif ctx.content.startswith(command_prefix):
        await client.process_commands(ctx)
        return
    elif not isinstance(channel, discord.DMChannel) and not client.user.mentioned_in(ctx):
        return

    # Recorded AFTER the early returns, not before. Above the self-check this
    # ran for every message Fritz himself sent, so the bot account had an alias
    # row in user_aliases — it was recorded as one of its own users. It also
    # fired for ambient guild chatter the bot was never addressed in, quietly
    # building a roster of people who had not interacted with it at all.
    identity_store.record(user_id, author, "discord")

    METRICS.increment("discord_messages")
    request_id = str(uuid.uuid4())[:8]
    logger.info("Incoming message %s from %s", request_id, author)

    loop = asyncio.get_running_loop()

    user_image_paths = []
    temp_audio_paths = []          # tracked so they can be cleaned up
    source = MessageSource.DISCORD_TEXT
    if ctx.attachments:
        logger.info("Processing %d attachment(s) for %s", len(ctx.attachments), request_id)
        for attachment in ctx.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                source = MessageSource.DISCORD_TEXT_AND_IMAGE
                try:
                    os.makedirs("temp_images", exist_ok=True)
                    file_path = os.path.join("temp_images", f"{safe_user_token(user_id)}_{attachment.id}_{attachment.filename}")
                    await attachment.save(file_path)
                    user_image_paths.append(file_path)
                    logger.info("Saved image %s for %s", file_path, request_id)
                except Exception as e:
                    METRICS.record_error("attachment_save", e)
                    logger.warning("Error saving image attachment: %s", e)
            elif attachment.content_type and attachment.content_type.startswith('audio/'):
                # The image branch above already had this try/except; without
                # it a save or transcription failure escaped on_message
                # entirely and the user got no reply at all.
                try:
                    os.makedirs("temp_audio", exist_ok=True)
                    file_path = os.path.join("temp_audio", f"{safe_user_token(user_id)}_{attachment.id}_{attachment.filename}")
                    await attachment.save(file_path)
                    temp_audio_paths.append(file_path)
                    logger.info("Saved audio %s for %s", file_path, request_id)
                    voice_text = await speech_to_text(file_path)
                    if not message_clean:
                        message_clean = voice_text
                    logger.info("Voice text: %s", voice_text)
                except Exception as e:
                    METRICS.record_error("attachment_audio", e)
                    logger.warning("Error handling audio attachment: %s", e)

    if user_image_paths:
        status_msg = await ctx.channel.send(f"✍️ *Analyzing {len(user_image_paths)} image(s)...*")
    else:
        status_msg = await ctx.channel.send("✍️ *Mister Fritz is thinking...*")

    streaming_handler = StreamingMessageHandler(status_msg, loop)
    streaming_callback = make_streaming_callback(streaming_handler, loop)

    def progress_callback(message: str):
        # Was ctx.channel.send: a permanent, un-deleteable message per tool
        # notice — and because these are scheduled onto the loop from a worker
        # thread, they could land BELOW the placeholder they were describing.
        # Now they render inside it and vanish with the reply.
        asyncio.run_coroutine_threadsafe(streaming_handler.set_status(message), loop)

    try:
        start_time = time.time()
        response_data = await run_blocking(
            ask_stuff,
            message_clean, source, user_id,
            progress_callback, streaming_callback,
            user_image_paths,
            workspace_store.get(user_id),
            ctx.channel.id,
            schedule_manager,
            display_name=author,
            channel_key=str(ctx.channel.id),
        )
        METRICS.record_latency("ask_stuff", time.time() - start_time)
    except Exception as e:
        await status_msg.edit(content=fritz_error("ask_stuff", e))
        _cleanup_temp_files(user_image_paths + temp_audio_paths, request_id)
        return

    logger.debug("Response data for %s: %s", request_id, response_data)

    if not response_data or not response_data.get("text"):
        original_response = (
            "I find I have nothing whatsoever to say on the matter. "
            "A rare occurrence, and not one I intend to dwell upon. Do try again."
        )
        image_paths = []
    else:
        original_response = response_data["text"]
        image_paths = response_data.get("image_paths", [])

    files = []
    for image_path in image_paths:
        try:
            files.append(discord.File(image_path))
        except Exception as e:
            METRICS.record_error("image_load", e)
            logger.warning("Error loading image file %s: %s", image_path, e)

    # final_update owns chunking now; this used to duplicate that logic.
    await streaming_handler.final_update(original_response, files)

    _cleanup_temp_files(user_image_paths + temp_audio_paths, request_id)


async def speech_to_text(file_path: str) -> str | None:
    """Thin async wrapper around the Whisper STT module."""
    return await run_blocking(_whisper_transcribe, file_path)


if __name__ == '__main__':
    validate_config()
    start_metrics_server()
    client.run(DISCORD_BOT_TOKEN)
