import asyncio
import logging
import os
import time
import uuid

import discord
from discord.ext import commands

from bot_adapters import split_into_chunks  # noqa: F401 — re-exported for tests
from bot_commands import FritzCommands
from fritz_utils import DISCORD_BOT_TOKEN, ROOT_USER, MessageSource, validate_config
from mister_fritz import ask_stuff
from observability import METRICS, init_logging, start_metrics_server
from scheduler import ScheduleManager
from stt import transcribe as _whisper_transcribe
from tts import TTSEngine

init_logging()
logger = logging.getLogger(__name__)


class StreamingMessageHandler:
    """Manages incremental Discord message updates with rate limiting."""

    def __init__(self, message: discord.Message, loop: asyncio.AbstractEventLoop, min_update_interval: float = 1.5):
        self.message = message
        self.loop = loop
        self.min_update_interval = min_update_interval
        self.current_text = ""
        self.last_update_time = 0
        self.update_task = None
        self.is_updating = False
        self.pending_text = None

    async def update_text(self, new_text: str):
        """Update the message with new text, respecting rate limits."""
        self.pending_text = new_text
        if not self.is_updating:
            self.is_updating = True
            await self._perform_update()

    async def _perform_update(self):
        """Perform the actual message edit with rate limiting."""
        while self.is_updating:
            if self.pending_text and self.pending_text != self.current_text:
                time_since_last = time.time() - self.last_update_time
                if time_since_last < self.min_update_interval:
                    await asyncio.sleep(self.min_update_interval - time_since_last)
                try:
                    await self.message.edit(content=self.pending_text[:2000])
                    self.current_text = self.pending_text
                    self.last_update_time = time.time()
                    self.pending_text = None
                except discord.errors.HTTPException as e:
                    logger.warning("Error editing message: %s", e)
                    await asyncio.sleep(2)
            else:
                self.is_updating = False

    async def final_update(self, final_text: str, files: list = None):
        """Perform the final update with complete text and optional file attachments."""
        while self.is_updating:
            await asyncio.sleep(0.1)
        try:
            await self.message.edit(content=final_text[:2000])
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
user_workspaces: dict[str, str] = {}
schedule_manager = None


@client.event
async def on_ready():
    global sayer, schedule_manager
    loop = asyncio.get_running_loop()
    if sayer is None:
        logger.info("Loading TTS engine...")
        sayer = await loop.run_in_executor(None, TTSEngine)
        logger.info("TTS engine ready")
    schedule_manager = ScheduleManager(client)
    schedule_manager.start()
    await client.add_cog(FritzCommands(client, sayer, user_workspaces, schedule_manager))
    logger.info("Logged in as %s", client.user)
    try:
        synced = await client.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


@client.event
async def on_message(ctx):
    author = ctx.author.name
    channel = ctx.channel
    message_clean = ctx.clean_content
    if ctx.author == client.user:
        return
    elif ctx.content.startswith(command_prefix):
        await client.process_commands(ctx)
        return
    elif not isinstance(channel, discord.DMChannel) and not client.user.mentioned_in(ctx):
        return

    METRICS.increment("discord_messages")
    request_id = str(uuid.uuid4())[:8]
    logger.info("Incoming message %s from %s", request_id, author)

    loop = asyncio.get_running_loop()

    user_image_paths = []
    source = MessageSource.DISCORD_TEXT
    if ctx.attachments:
        logger.info("Processing %d attachment(s) for %s", len(ctx.attachments), request_id)
        for attachment in ctx.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                source = MessageSource.DISCORD_TEXT_AND_IMAGE
                try:
                    os.makedirs("temp_images", exist_ok=True)
                    file_path = os.path.join("temp_images", f"{author}_{attachment.id}_{attachment.filename}")
                    await attachment.save(file_path)
                    user_image_paths.append(file_path)
                    logger.info("Saved image %s for %s", file_path, request_id)
                except Exception as e:
                    METRICS.record_error("attachment_save", e)
                    logger.warning("Error saving image attachment: %s", e)
            elif attachment.content_type and attachment.content_type.startswith('audio/'):
                os.makedirs("temp_audio", exist_ok=True)
                file_path = os.path.join("temp_audio", f"{author}_{attachment.id}_{attachment.filename}")
                await attachment.save(file_path)
                logger.info("Saved audio %s for %s", file_path, request_id)
                voice_text = await speech_to_text(file_path)
                if not message_clean:
                    message_clean = voice_text
                logger.info("Voice text: %s", voice_text)

    if user_image_paths:
        status_msg = await ctx.channel.send(f"✍️ *Analyzing {len(user_image_paths)} image(s)...*")
    else:
        status_msg = await ctx.channel.send("✍️ *Mister Fritz is thinking...*")

    streaming_handler = StreamingMessageHandler(status_msg, loop)

    def streaming_callback(partial_text: str):
        asyncio.run_coroutine_threadsafe(streaming_handler.update_text(partial_text), loop)

    def progress_callback(message: str):
        asyncio.run_coroutine_threadsafe(ctx.channel.send(message), loop)

    try:
        start_time = time.time()
        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(
                message_clean, source, author,
                progress_callback, streaming_callback,
                user_image_paths,
                user_workspaces.get(author) if author == ROOT_USER else None,
                ctx.channel.id,
                schedule_manager,
            )
        )
        METRICS.record_latency("ask_stuff", time.time() - start_time)
    except Exception as e:
        METRICS.record_error("ask_stuff", e)
        logger.exception("Error during ask_stuff for %s", request_id)
        await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        _cleanup_temp_files(user_image_paths, request_id)
        return

    logger.debug("Response data for %s: %s", request_id, response_data)

    if not response_data or not response_data.get("text"):
        original_response = "The bot got sad and doesn't want to talk to you at the moment :("
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

    if len(original_response) > 2000:
        chunks = split_into_chunks(original_response)
        for i, chunk in enumerate(chunks):
            chunk_files = files if i == 0 else []
            if i == 0:
                await streaming_handler.final_update(chunk, chunk_files)
            else:
                await ctx.channel.send(chunk, files=chunk_files)
    else:
        await streaming_handler.final_update(original_response, files)

    _cleanup_temp_files(user_image_paths, request_id)


async def speech_to_text(file_path: str) -> str | None:
    """Thin async wrapper around the Whisper STT module."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _whisper_transcribe, file_path)


if __name__ == '__main__':
    validate_config()
    start_metrics_server()
    client.run(DISCORD_BOT_TOKEN)
