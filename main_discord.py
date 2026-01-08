import asyncio
import time

import discord
from discord.ext import commands

from document_engine import query_documents
from fritz_utils import get_key_from_json_config_file, MessageSource, DISCORD_KEY
from image_generator import generate_image
from mister_fritz import ask_stuff
from tts import StuffSayer


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
                    print(f"Error editing message: {e}")
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
            print(f"Error in final update: {e}")

command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)

connection = None
sayer = StuffSayer()


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.command()
async def hello(ctx):
    author = ctx.author.name
    await ctx.send(f"Hello, {author}!")

@client.command()
async def voice(ctx, *, message):
    author = ctx.author.name
    original_response = ask_stuff(message, MessageSource.DISCORD_VOICE, author)["text"]
    output_file = sayer.say_stuff_advanced(original_response)

    try:
        if ctx.voice_client:
            await ctx.voice_client.play(discord.FFmpegPCMAudio(executable="./ffmpeg.exe",source=output_file))
        else:
            files = [discord.File(output_file)]
            await ctx.send("You are not connected to a voice channel, uploading as a file...", files=files)
    except AttributeError as e:
        await ctx.send("Something crazy happened!")


@client.command()
async def gen(ctx, *, message):
    print(message)
    output_file = generate_image(message)
    file = discord.File(output_file)
    # Send the file
    await ctx.send(file=file, content="Here is your file!")


@client.command()
async def join(ctx):
    try:
        channel = ctx.author.voice.channel
        await channel.connect()
    except AttributeError as e:
        await ctx.send("You are not connected to a voice channel, buddy!")

@client.command()
async def lore(ctx, *, message):
    print(message)
    original_message = await ctx.send("This may take a few seconds, please wait. This message will be updated with the result!")
    original_response = query_documents(message)
    resp_len = len(original_response)
    author = ctx.author.name

    if resp_len > 2000:
        response = "The answer was over 2000 ({}), so you're getting multiple messages {} \r\n".format(resp_len,
                                                                                                       author) + original_response
        responses = split_into_chunks(response)
        for i, response in enumerate(responses):
            await ctx.send(response)
    else:
        await original_message.edit(content=original_response)

@client.command()
async def leave(ctx):
    try:
        await ctx.voice_client.disconnect()
    except AttributeError as e:
        await ctx.send("I am not connected to a voice channel, buddy!")


@client.event
async def on_message(ctx):
    author = ctx.author.name
    channel_type = ctx.channel
    if ctx.author == client.user:
        return
    elif ctx.content.startswith(command_prefix):
        await client.process_commands(ctx)
        return
    elif not isinstance(channel_type, discord.DMChannel) and not client.user.mentioned_in(ctx):
        return

    print(f"Incoming message from: {author}")

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Send initial status message immediately
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
        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(ctx.clean_content, MessageSource.DISCORD_TEXT, author, progress_callback, streaming_callback)
        )
    except Exception as e:
        print(f"Error during ask_stuff: {e}")
        await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        return

    print(response_data)

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
                print(f"Error loading image file {image_path}: {e}")

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


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


if __name__ == '__main__':
    discord_secret = get_key_from_json_config_file(DISCORD_KEY)
    client.run(discord_secret)
