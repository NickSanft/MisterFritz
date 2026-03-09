import logging
import os

import discord
from discord import app_commands
from discord.ext import commands

from cards import draw_cards, get_remaining_card_number, reload_deck
from document_engine import query_documents
from fritz_utils import FFMPEG_PATH, ROOT_USER, MessageSource
from image_generator import generate_image
from mister_fritz import ask_stuff
from observability import METRICS, format_health_text, get_health_snapshot
from tts import TTSEngine

logger = logging.getLogger(__name__)


def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


class FritzCommands(commands.Cog):
    """All MisterFritz slash commands."""

    def __init__(self, bot: commands.Bot, sayer: TTSEngine, user_workspaces: dict[str, str]):
        self.bot = bot
        self.sayer = sayer
        self.user_workspaces = user_workspaces

    # ── Card game ─────────────────────────────────────────────────────────────

    @app_commands.command(name="draw", description="Draw cards from a deck!")
    @app_commands.describe(num_cards="How many cards to draw")
    async def draw_slash(self, interaction: discord.Interaction, num_cards: int):
        await interaction.response.defer(thinking=True)
        await interaction.followup.send(content=draw_cards(num_cards, interaction.user.name))

    @app_commands.command(name="cards_remaining", description="Check cards remaining in the deck")
    async def cards_remaining_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        await interaction.followup.send(content=get_remaining_card_number(interaction.user.name))

    @app_commands.command(name="reload_deck", description="Reloads the deck (use if you goof up)")
    async def reload_deck_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        await interaction.followup.send(content=reload_deck(interaction.user.name))

    # ── General ───────────────────────────────────────────────────────────────

    @app_commands.command(name="hello", description="Say hello to the bot")
    async def hello_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.hello")
        await interaction.response.send_message(f"Hello, {interaction.user.name}!")

    @app_commands.command(name="health", description="Check the system health metrics")
    async def health_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.health")
        await interaction.response.send_message(format_health_text(get_health_snapshot()))

    # ── AI / content ──────────────────────────────────────────────────────────

    @app_commands.command(name="voice", description="Generate audio from text")
    @app_commands.describe(message="The text you want the bot to say")
    async def voice_slash(self, interaction: discord.Interaction, message: str):
        await interaction.response.defer(thinking=True)
        METRICS.increment("discord_commands.voice")
        original_response = ask_stuff(message, MessageSource.DISCORD_VOICE, interaction.user.name)["text"]
        output_file = await self.sayer.generate_speech(original_response)
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
            METRICS.record_error("discord_commands.voice", e)
            await interaction.followup.send("Something crazy happened!")

    @app_commands.command(name="gen", description="Generate an image based on a prompt")
    @app_commands.describe(prompt="The image description")
    async def gen_slash(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(thinking=True)
        METRICS.increment("discord_commands.gen")
        logger.info("Image generation request: %s", prompt)
        try:
            output_file = generate_image(prompt)
            await interaction.followup.send(content="Here is your file!", file=discord.File(output_file))
        except Exception as e:
            METRICS.record_error("discord_commands.gen", e)
            await interaction.followup.send(f"Failed to generate image: {e}")

    @app_commands.command(name="lore", description="Query the document engine for lore")
    @app_commands.describe(query="The question about the lore")
    async def lore_slash(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer(thinking=True)
        METRICS.increment("discord_commands.lore")
        logger.info("Lore request: %s", query)
        original_response = query_documents(query)
        author = interaction.user.name
        if len(original_response) > 2000:
            header = f"The answer was over 2000 ({len(original_response)}), so you're getting multiple messages {author} \r\n"
            chunks = split_into_chunks(header + original_response)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.followup.send(chunk)
                else:
                    await interaction.channel.send(chunk)
        else:
            await interaction.followup.send(original_response)

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
                    "You are not connected to a voice channel, buddy!", ephemeral=True
                )
        except Exception as e:
            METRICS.record_error("discord_commands.join", e)
            if not interaction.response.is_done():
                await interaction.response.send_message("I couldn't join the channel.")

    @app_commands.command(name="leave", description="Leave the current voice channel")
    async def leave_slash(self, interaction: discord.Interaction):
        try:
            METRICS.increment("discord_commands.leave")
            if interaction.guild.voice_client:
                await interaction.guild.voice_client.disconnect()
                await interaction.response.send_message("Disconnected.")
            else:
                await interaction.response.send_message(
                    "I am not connected to a voice channel, buddy!", ephemeral=True
                )
        except Exception as e:
            METRICS.record_error("discord_commands.leave", e)
            await interaction.response.send_message("Error attempting to leave.")

    # ── File workspace ────────────────────────────────────────────────────────

    @app_commands.command(name="workspace", description="Set or view the workspace directory for file operations")
    @app_commands.describe(path="The directory path to use as workspace (leave empty to see current)")
    async def workspace_slash(self, interaction: discord.Interaction, path: str = None):
        METRICS.increment("discord_commands.workspace")
        author = interaction.user.name

        if author != ROOT_USER:
            await interaction.response.send_message(
                "You do not have permission to use file operations.", ephemeral=True
            )
            return

        if path is None:
            current = self.user_workspaces.get(author)
            if current:
                await interaction.response.send_message(f"Current workspace: `{current}`", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "No workspace set. Use `/workspace <path>` to set a directory for file operations.",
                    ephemeral=True,
                )
            return

        expanded = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(expanded):
            await interaction.response.send_message(
                f"Directory does not exist: `{expanded}`", ephemeral=True
            )
            return

        self.user_workspaces[author] = expanded
        await interaction.response.send_message(
            f"Workspace set to: `{expanded}`\nFile tools (read, write, edit, search, list) are now active in conversations.",
            ephemeral=True,
        )
