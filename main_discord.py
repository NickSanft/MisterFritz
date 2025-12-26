import asyncio

import discord
from discord.ext import commands

from document_engine import query_documents
from fritz_utils import get_key_from_json_config_file, MessageSource
from image_generator import generate_image
from mister_fritz import ask_stuff

discord_key = "discord_bot_token"
command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)

connection = None


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.command()
async def hello(ctx):
    author = ctx.author.name
    await ctx.send(f"Hello, {author}!")

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

    # This variable will hold the message object if the 5-second timer finishes
    patience_msg = None

    async def send_patience_warning():
        nonlocal patience_msg
        await asyncio.sleep(5)
        print("Time's up!")
        patience_msg = await ctx.channel.send("Please hold, this requires more thought. This message will be edited once complete.")

    # Start the timer task
    patience_task = asyncio.create_task(send_patience_warning())

    # Run the blocking ask_stuff function in a thread
    loop = asyncio.get_running_loop()
    try:
        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(ctx.clean_content, MessageSource.DISCORD_TEXT, author)
        )
    finally:
        # Stop the timer if ask_stuff finishes (even if it already sent the message)
        patience_task.cancel()

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
        responses = split_into_chunks(original_response)
        for i, chunk in enumerate(responses):
            # Attach files to the first message only
            chunk_files = files if i == 0 else []
            if patience_msg:
                await patience_msg.edit(content=chunk)
                patience_msg = None
                # Send files separately if we edited the patience message
                if chunk_files:
                    await ctx.channel.send(files=chunk_files)
            else:
                await ctx.channel.send(chunk, files=chunk_files)
    else:
        # If we have a patience message, EDIT it. Otherwise, SEND a new one.
        if patience_msg:
            await patience_msg.edit(content=original_response)
            # Send files separately since we can't attach to edit
            if files:
                await ctx.channel.send(files=files)
        else:
            await ctx.channel.send(original_response, files=files)


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


if __name__ == '__main__':
    discord_secret = get_key_from_json_config_file(discord_key)
    client.run(discord_secret)
