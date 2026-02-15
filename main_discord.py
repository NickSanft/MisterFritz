import asyncio
import logging
import os
import time
import uuid
import speech_recognition as sr

import discord
from discord import app_commands
from discord.ext import commands
from pydub import AudioSegment

from deck_of_cards_integration import draw_cards, get_remaining_card_number, reload_deck
from document_engine import query_documents
from fritz_utils import get_key_from_json_config_file, MessageSource, DISCORD_KEY, FFMPEG_PATH, FFPROBE_PATH
from image_generator import generate_image
from mister_fritz import ask_stuff
from tts import TTSEngine
from observability import init_logging, METRICS, get_health_snapshot, format_health_text

r = sr.Recognizer()

init_logging()
logger = logging.getLogger(__name__)

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

if not os.path.exists(FFMPEG_PATH):
    logger.error("ffmpeg.exe not found at %s", FFMPEG_PATH)
if not os.path.exists(FFPROBE_PATH):
    logger.error("ffprobe.exe not found at %s - pydub needs this to read OGG files", FFPROBE_PATH)



class StreamingMessageHandler:
    """Manages incremental Discord message updates with rate limiting."""

    def __init__(self, message: discord.Message, loop: asyncio.AbstractEventLoop, min_update_interval: float = 1.5):
        """
        Args:
            message: The Discord message to edit with streaming content
            loop: The asyncio event loop for scheduling edits
            min_update_interval: Minimum seconds between edits (default 1.5 for rate limiting)
        """
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

        # If we're not currently updating, schedule an update
        if not self.is_updating:
            self.is_updating = True
            await self._perform_update()

    async def _perform_update(self):
        """Perform the actual message edit with rate limiting."""
        while self.is_updating:
            # Check if we have pending text to update
            if self.pending_text and self.pending_text != self.current_text:
                # Check if enough time has passed since last update
                time_since_last = time.time() - self.last_update_time
                if time_since_last < self.min_update_interval:
                    # Wait for the remaining time
                    await asyncio.sleep(self.min_update_interval - time_since_last)

                # Perform the edit
                try:
                    text_to_send = self.pending_text[:2000]  # Discord limit
                    await self.message.edit(content=text_to_send)
                    self.current_text = self.pending_text
                    self.last_update_time = time.time()
                    self.pending_text = None
                except discord.errors.HTTPException as e:
                    logger.warning("Error editing message: %s", e)
                    await asyncio.sleep(2)  # Back off on error
            else:
                # No pending updates, stop the update loop
                self.is_updating = False

    async def final_update(self, final_text: str, files: list = None):
        """Perform the final update with complete text and optional file attachments."""
        # Wait for any pending updates to complete
        while self.is_updating:
            await asyncio.sleep(0.1)

        try:
            # If text is too long, we can't attach files to edits, so send files separately
            if len(final_text) > 2000:
                await self.message.edit(content=final_text[:2000])
                if files:
                    await self.message.channel.send(files=files)
            else:
                await self.message.edit(content=final_text)
                # Send files separately since we can't attach to edits
                if files:
                    await self.message.channel.send(files=files)
        except discord.errors.HTTPException as e:
            logger.warning("Error in final update: %s", e)

command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)

connection = None
sayer = TTSEngine()


@client.event
async def on_ready():
    logger.info("Logged in as %s", client.user)
    try:
        # This registers the slash commands with Discord
        synced = await client.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@client.tree.command(name="draw", description="Draw cards from a deck!")
@app_commands.describe(num_cards="How many cards to draw")
async def draw_slash(interaction: discord.Interaction, num_cards: int):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(content=draw_cards(num_cards, interaction.user.name))

@client.tree.command(name="cards_remaining", description="Check cards remaining in the deck")
async def cards_remaining_slash(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(content=get_remaining_card_number(interaction.user.name))

@client.tree.command(name="reload_deck", description="Reloads the deck (use if you goof up)")
async def reload_deck_slash(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    await interaction.followup.send(content=reload_deck(interaction.user.name))


# --- CONVERTED COMMANDS ---

@client.tree.command(name="hello", description="Say hello to the bot")
async def hello_slash(interaction: discord.Interaction):
    author = interaction.user.name
    METRICS.increment("discord_commands.hello")
    await interaction.response.send_message(f"Hello, {author}!")


@client.tree.command(name="voice", description="Generate audio from text")
@app_commands.describe(message="The text you want the bot to say")
async def voice_slash(interaction: discord.Interaction, message: str):
    # Defer because TTS generation might take > 3 seconds
    await interaction.response.defer(thinking=True)

    author = interaction.user.name
    METRICS.increment("discord_commands.voice")

    # Generate the text response first
    original_response = ask_stuff(message, MessageSource.DISCORD_VOICE, author)["text"]
    # Generate the audio file
    output_file = await sayer.generate_speech(original_response)

    try:
        if interaction.guild and interaction.guild.voice_client:
            # Play in voice channel
            interaction.guild.voice_client.play(discord.FFmpegPCMAudio(executable="./ffmpeg.exe", source=output_file))
            await interaction.followup.send(f"Playing voice for: '{message}'")
        else:
            # Upload as file
            files = [discord.File(output_file)]
            await interaction.followup.send("You are not connected to a voice channel, uploading as a file...",
                                            files=files)
    except AttributeError as e:
        METRICS.record_error("discord_commands.voice", e)
        await interaction.followup.send("Something crazy happened!")


@client.tree.command(name="gen", description="Generate an image based on a prompt")
@app_commands.describe(prompt="The image description")
async def gen_slash(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    METRICS.increment("discord_commands.gen")
    logger.info("Image generation request: %s", prompt)

    try:
        output_file = generate_image(prompt)
        file = discord.File(output_file)
        await interaction.followup.send(content="Here is your file!", file=file)
    except Exception as e:
        METRICS.record_error("discord_commands.gen", e)
        await interaction.followup.send(f"Failed to generate image: {e}")


@client.tree.command(name="join", description="Join the voice channel you are currently in")
async def join_slash(interaction: discord.Interaction):
    try:
        METRICS.increment("discord_commands.join")
        if interaction.user.voice and interaction.user.voice.channel:
            channel = interaction.user.voice.channel
            await channel.connect()
            await interaction.response.send_message(f"Joined {channel.name}!")
        else:
            await interaction.response.send_message("You are not connected to a voice channel, buddy!", ephemeral=True)
    except Exception as e:
        METRICS.record_error("discord_commands.join", e)
        # Using followup in case we deferred (though we didn't here, safe to use response if not deferred)
        if not interaction.response.is_done():
            await interaction.response.send_message("I couldn't join the channel.")


@client.tree.command(name="lore", description="Query the document engine for lore")
@app_commands.describe(query="The question about the lore")
async def lore_slash(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    METRICS.increment("discord_commands.lore")
    logger.info("Lore request: %s", query)

    # We deferred, so we use followup to send the result
    original_response = query_documents(query)
    resp_len = len(original_response)
    author = interaction.user.name

    if resp_len > 2000:
        response = "The answer was over 2000 ({}), so you're getting multiple messages {} \r\n".format(resp_len,
                                                                                                       author) + original_response
        responses = split_into_chunks(response)
        for i, chunk in enumerate(responses):
            if i == 0:
                await interaction.followup.send(chunk)
            else:
                await interaction.channel.send(chunk)
    else:
        await interaction.followup.send(original_response)


@client.tree.command(name="leave", description="Leave the current voice channel")
async def leave_slash(interaction: discord.Interaction):
    try:
        METRICS.increment("discord_commands.leave")
        if interaction.guild.voice_client:
            await interaction.guild.voice_client.disconnect()
            await interaction.response.send_message("Disconnected.")
        else:
            await interaction.response.send_message("I am not connected to a voice channel, buddy!", ephemeral=True)
    except Exception as e:
        METRICS.record_error("discord_commands.leave", e)
        await interaction.response.send_message("Error attempting to leave.")


@client.tree.command(name="health", description="Check the system health metrics")
async def health_slash(interaction: discord.Interaction):
    METRICS.increment("discord_commands.health")
    snapshot = get_health_snapshot()
    await interaction.response.send_message(format_health_text(snapshot))


@client.event
async def on_message(ctx):
    author = ctx.author.name
    channel_type = ctx.channel
    message_clean = ctx.clean_content
    if ctx.author == client.user:
        return
    # We still check process_commands just in case any old prefix commands exist or are added later
    elif ctx.content.startswith(command_prefix):
        await client.process_commands(ctx)
        return
    elif not isinstance(channel_type, discord.DMChannel) and not client.user.mentioned_in(ctx):
        return

    METRICS.increment("discord_messages")
    request_id = str(uuid.uuid4())[:8]
    logger.info("Incoming message %s from %s", request_id, author)

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Download and save any image attachments
    user_image_paths = []
    source = MessageSource.DISCORD_TEXT
    if ctx.attachments:
        logger.info("Processing %d attachment(s) for %s", len(ctx.attachments), request_id)
        for attachment in ctx.attachments:
            # Check if it's an image
            if attachment.content_type and attachment.content_type.startswith('image/'):
                source = MessageSource.DISCORD_TEXT_AND_IMAGE
                try:
                    # Create temp directory if it doesn't exist
                    import os
                    temp_dir = "temp_images"
                    os.makedirs(temp_dir, exist_ok=True)

                    # Save the image
                    file_path = os.path.join(temp_dir, f"{author}_{attachment.id}_{attachment.filename}")
                    await attachment.save(file_path)
                    user_image_paths.append(file_path)
                    logger.info("Saved image %s for %s", file_path, request_id)
                except Exception as e:
                    METRICS.record_error("attachment_save", e)
                    logger.warning("Error saving image attachment: %s", e)
            elif attachment.content_type and attachment.content_type.startswith('audio/'):
                # Create temp directory if it doesn't exist
                import os
                temp_dir = "temp_audio"
                os.makedirs(temp_dir, exist_ok=True)

                # Save the image
                file_path = os.path.join(temp_dir, f"{author}_{attachment.id}_{attachment.filename}")
                await attachment.save(file_path)
                logger.info("Saved audio %s for %s", file_path, request_id)
                voice_text = await speech_to_text(file_path)
                if not message_clean:
                    message_clean = voice_text
                else:
                    logger.debug("Voice text type: %s", type(voice_text))
                logger.info("Voice text: %s", voice_text)

    # Send initial status message immediately
    if user_image_paths:
        status_msg = await ctx.channel.send(f"✍️ *Analyzing {len(user_image_paths)} image(s)...*")
    else:
        status_msg = await ctx.channel.send("✍️ *Mister Fritz is thinking...*")

    # Create streaming handler for updating the message
    streaming_handler = StreamingMessageHandler(status_msg, loop)

    # Create a streaming callback that updates Discord message from the worker thread
    def streaming_callback(partial_text: str):
        """Update Discord message with partial text from the worker thread."""
        # Schedule the coroutine in the event loop from the worker thread
        asyncio.run_coroutine_threadsafe(
            streaming_handler.update_text(partial_text),
            loop
        )

    # Create a progress callback that sends separate messages for tool status
    def progress_callback(message: str):
        """Send progress updates to Discord from the worker thread."""
        # Schedule the coroutine in the event loop from the worker thread
        asyncio.run_coroutine_threadsafe(
            ctx.channel.send(message),
            loop
        )

    # Run the blocking ask_stuff function in a thread with streaming enabled
    try:
        start_time = time.time()
        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(message_clean, source, author, None, streaming_callback, user_image_paths)
        )
        METRICS.record_latency("ask_stuff", time.time() - start_time)
    except Exception as e:
        METRICS.record_error("ask_stuff", e)
        logger.exception("Error during ask_stuff for %s", request_id)
        await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        # Clean up temporary images
        for img_path in user_image_paths:
            try:
                import os
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info("Cleaned up %s after error", img_path)
            except Exception as cleanup_error:
                METRICS.record_error("cleanup_error", cleanup_error)
                logger.warning("Error cleaning up %s: %s", img_path, cleanup_error)
        return

    logger.debug("Response data for %s: %s", request_id, response_data)

    if not response_data or not response_data.get("text"):
        original_response = "The bot got sad and doesn't want to talk to you at the moment :("
        image_paths = []
    else:
        original_response = response_data["text"]
        image_paths = response_data.get("image_paths", [])

    # Prepare any image files for upload
    files = []
    if image_paths:
        for image_path in image_paths:
            try:
                files.append(discord.File(image_path))
            except Exception as e:
                METRICS.record_error("image_load", e)
                logger.warning("Error loading image file %s: %s", image_path, e)

    # --- THE EDIT LOGIC ---
    resp_len = len(original_response)

    if resp_len > 2000:
        # For long responses, split into chunks
        responses = split_into_chunks(original_response)
        for i, chunk in enumerate(responses):
            # Attach files to the first message only
            chunk_files = files if i == 0 else []
            if i == 0:
                # Edit the streaming message with the first chunk
                await streaming_handler.final_update(chunk, chunk_files)
            else:
                # Send additional chunks as new messages
                await ctx.channel.send(chunk, files=chunk_files)
    else:
        # Use the streaming handler's final update for the complete response
        await streaming_handler.final_update(original_response, files)

    # Clean up temporary user images after processing
    for img_path in user_image_paths:
        try:
            import os
            if os.path.exists(img_path):
                os.remove(img_path)
                logger.info("Cleaned up %s for %s", img_path, request_id)
        except Exception as cleanup_error:
            METRICS.record_error("cleanup_error", cleanup_error)
            logger.warning("Error cleaning up %s: %s", img_path, cleanup_error)


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


async def speech_to_text(file_path: str):
    wav_file = f"{file_path}.wav"
    logger.info("Saved audio: %s", file_path)
    AudioSegment.from_ogg(file_path).export(wav_file, format="wav")
    logger.info("Converted %s to %s", file_path, wav_file)
    voice_message = sr.AudioFile(wav_file)
    with voice_message as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        METRICS.record_error("speech_to_text", e)
        logger.warning("Speech to text error: %s", e)


if __name__ == '__main__':
    discord_secret = get_key_from_json_config_file(DISCORD_KEY)
    client.run(discord_secret)