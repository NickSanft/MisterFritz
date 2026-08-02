# 10. Discord surface polish

[← back to index](README.md)

**Effort:** L (1-3 days)  
**Depends on:** token-streaming

## Goal
Today the Discord surface leaks raw Python exception strings to end users, breaks the Mister Fritz persona in five separate places, spams the channel with permanent un-deleteable progress messages that can land below the streaming placeholder, splits long replies mid-word and mid-code-fence, silently truncates mid-stream text at 2000 chars, leaks `/health` metrics publicly, has zero `discord.Embed` usage, and has no app-command error handler at all — so an out-of-range `/draw 500` raises an uncaught `HTTPException` and the user sees nothing but a spinning "thinking" state. When this is done: `split_into_chunks` never breaks a word or an unbalanced ```` ``` ```` fence; every user-facing failure reads in Fritz's voice with a short log-correlation ref while the real exception goes only to the log and `METRICS`; tool/plan progress renders inside the streaming placeholder instead of as separate permanent sends; `/help`, `/about`, `/health`, and `/schedule list` are `discord.Embed`s in the project accent colour `#5b3f30`; `/health` is ephemeral like the other 37 interaction responses; a `cog_app_command_error` handler plus a `tree.on_error` backstop catch everything a slash command can raise; `/draw` is bounded by `app_commands.Range[int, 1, 40]` and its output is chunked; tool notices are butler-voiced; and `temp_audio/` files are cleaned up like `temp_images/` already are.

## Definition of done

- [ ] `split_into_chunks("```py\ncode…", 2000)` on input that straddles a fence boundary returns chunks where every chunk has a balanced number of ``` fence lines, and no chunk exceeds `chunk_size`.
- [ ] `split_into_chunks` no longer splits mid-word when a whitespace/newline boundary exists in the second half of the budget window; `"".join(chunks) == original` still holds for input containing no unbalanced fences.
- [ ] All six existing tests in `tests/test_discord_commands.py::TestSplitIntoChunks` still pass unmodified (verified by running pytest, not by inspection).
- [ ] `grep -n 'str(e)\|{e}' bot_commands.py main_discord.py` returns no hit inside a string that is passed to `interaction.response.send_message`, `interaction.followup.send`, `ctx.channel.send`, or `status_msg.edit` — except the two intentional `ValueError`/`PermissionError` validation paths (`bot_commands.py:149`, `:190`) which carry author-written messages, not tracebacks.
- [ ] The strings "The bot got sad and doesn't want to talk to you at the moment :(", "Something crazy happened!", and "buddy!" no longer appear anywhere in `bot_commands.py` or `main_discord.py`.
- [ ] Progress notices during a DM appear inside the placeholder message created at `main_discord.py:179-181`; `ctx.channel.send` is no longer called from `progress_callback`, so no orphaned progress messages remain in the channel after a reply lands.
- [ ] `/help`, `/about`, `/health`, and `/schedule list` all send a `discord.Embed` with `colour=discord.Colour(0x5B3F30)`.
- [ ] `/health` sends with `ephemeral=True`.
- [ ] `/about` reports a non-zero uptime after the bot has been up for >1s (fixes the `uptime_seconds` vs `uptime_sec` key mismatch at `bot_commands.py:367`).
- [ ] `FritzCommands.cog_app_command_error` exists and, for a synthetic `app_commands.CommandInvokeError` wrapping a `RuntimeError`, replies ephemerally with butler copy containing no substring of the original exception message.
- [ ] `/draw` is annotated `app_commands.Range[int, 1, 40]`, and `draw_slash` sends its result through `split_into_chunks` so the trailing ``` fenced summary block survives a split.
- [ ] `/lore` continuation chunks use `interaction.followup.send`, not `interaction.channel.send`, and the "The answer was over 2000 …" header at `bot_commands.py:434` is gone.
- [ ] After a voice-message DM, the saved file under `temp_audio/` is removed (the directory does not grow — currently it holds 9 leaked files).
- [ ] `ruff check .` is clean and `pytest tests/ --cov-fail-under=60` passes.

## Current state (verified against the working tree)
VERIFIED against the repo this session. Corrections to the audit findings are marked [CORRECTION].

**Chunker.** `bot_adapters.py:8-16` — `split_into_chunks` is pure index slicing: `[s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]`. No word, line, or fence awareness. It has three consumers: `main_discord.py:230`, `bot_commands.py:215` (`/schedule list_all`), `bot_commands.py:435` (`/lore`). `main_telegram.py` does NOT use it — it truncates with `reply[:4096]` at `:32` and `:57`, so Telegram is out of scope here.

**Mid-stream truncation.** `StreamingMessageHandler._perform_update` at `main_discord.py:63` does `await self.message.edit(content=self.pending_text[:2000])`. Past 2000 chars the user watches a frozen prefix. `final_update` at `:78` truncates identically, and `main_discord.py:229-238` then separately re-chunks the same text — duplicated logic across two functions.

**Progress spam.** `main_discord.py:188-189`:
```python
def progress_callback(message: str):
    asyncio.run_coroutine_threadsafe(ctx.channel.send(message), loop)
```
Each tool notice and each plan step becomes a permanent `ctx.channel.send`, never deleted, and — because these are scheduled onto the loop from the `run_in_executor` worker thread — they can be ordered after the placeholder created at `:179-181`. Producers are `mister_fritz.py:426` (`progress_callback(tool_messages[tool_name])`) and `mister_fritz.py:404` (`progress_callback(f"Step {current_step + 1}/{len(plan)}: {step_instruction}")`).

**Raw exceptions reaching users.** All confirmed at the audited lines: `main_discord.py:208` (`f"❌ An error occurred: {str(e)}"`), `bot_commands.py:152` (`f"❌ Failed to create schedule: {e}"`), `:193` (`f"❌ Failed to remove schedule: {e}"`), `:423` (`f"Failed to generate image: {e}"`), `:509` (`f"❌ Could not create workspace: {e}"`). `bot_commands.py:149` and `:190` also interpolate `{e}`, but those are a `ValueError` from `ScheduleManager.add_schedule` and a `PermissionError` from `remove_schedule` — author-written validation text, correctly surfaced. Leave those two alone.

**Persona breaks.** `main_discord.py:215` — `"The bot got sad and doesn't want to talk to you at the moment :("`. `bot_commands.py:410` — `"Something crazy happened!"`. `bot_commands.py:456` and `:472` — `"…voice channel, buddy!"`. `bot_commands.py:461` / `:476` — `"I couldn't join the channel."` / `"Error attempting to leave."` (flat, not sardonic). The persona these violate is stated at `bot_commands.py:371` (`_An AI butler of impeccable bearing and barely-concealed weariness._`) and defined in full at `mister_fritz.py:62-94` (`FRITZ_CHARACTER`) — note it explicitly forbids exclamation marks except for mock-dramatic effect.

**Zero embeds.** A repo-wide grep for `discord.Embed|Embed\(` over `*.py` returns no production hit. `/help` is a markdown wall built at `bot_commands.py:335-360` and sent at `:361`; `/about` at `:369-385`, sent `:386`; `/health` at `:330` sends `format_health_text(get_health_snapshot())` as plain text; `/schedule list` builds `lines` at `:166-169` and sends at `:170`.

**[CORRECTION / NEW BUG not in the audit] `/about` uptime is always 0.** `bot_commands.py:367` reads `snap.get("uptime_seconds", 0)`, but `observability.get_health_snapshot()` returns the key `"uptime_sec"` (`observability.py:304`). The `.get` default silently swallows it, so `_format_uptime(0)` renders `"0s"` on every invocation. Fix while rewriting `/about`.

**`/health` not ephemeral.** `bot_commands.py:330` — `await interaction.response.send_message(format_health_text(get_health_snapshot()))`, no `ephemeral=True`. Every other interaction reply in the file passes it.

**No error handler.** Grep for `cog_app_command_error|tree.on_error|on_error` over `*.py` returns zero production hits. Confirmed the API exists in the pinned lib: `discord.py==2.6.4` (`requirements.txt:44`), and `.venv/Lib/site-packages/discord/ext/commands/cog.py:649` defines `async def cog_app_command_error(self, interaction, error)`. `app_commands.Range` is exported (`transformers.py:65`) and `CommandInvokeError` / `TransformerError` / `CheckFailure` exist in `app_commands/errors.py`.

**`/draw` unbounded.** `bot_commands.py:304-308`: `async def draw_slash(self, interaction, num_cards: int)` → `interaction.followup.send(content=draw_cards(num_cards, ...))`. `cards.draw_cards` (`cards.py:43-86`) emits one ~41-char line per card plus a ```` ``` ````-fenced summary at `:75`/`:84`; ~48 cards crosses Discord's 2000-char cap and `followup.send` raises `HTTPException` with nothing caught. Measured budget: ~40 cards ≈ 1755 chars, safe; 50 ≈ 2150, unsafe.

**`/lore` uses the wrong send.** `bot_commands.py:433-440`: builds a clunky public header at `:434` (`f"The answer was over 2000 ({len(original_response)}), so you're getting multiple messages {author} \r\n"`) and sends continuation chunks with `await interaction.channel.send(chunk)` at `:440` instead of `interaction.followup.send`.

**Generic tool notices.** `mister_fritz.py:364-382` — a 17-entry `tool_messages` dict of plain assistant boilerplate ("Searching the web...", "Filing that away for future reference..."). These strings are also consumed by the web chat: `admin_panel.py:513-517` forwards them as an SSE `progress` event.

**`temp_audio/` leaks.** `main_discord.py:168-172` saves the attachment to `temp_audio/{author}_{id}_{name}` and never records the path. `_cleanup_temp_files` is called at `:209` and `:240` with `user_image_paths` only. `stt.transcribe` (`stt.py:110-115`) removes only its own intermediate `.wav`, not the source. Confirmed empirically: `temp_audio/` currently holds 9 leaked files. The audio branch at `:168-176` also has no `try/except`, unlike the image branch at `:159-167`, so an attachment-save or transcription failure escapes `on_message` entirely.

**Test baseline.** `pytest tests/test_discord_commands.py tests/test_bot_commands.py -q` → 26 passed (7.65s) on the current tree.

## Change sites

### `bot_adapters.py:1-16 (whole file)`

Replace the pure-slicing `split_into_chunks` with a boundary- and fence-aware version, and add the shared `fritz_error()` copy/logging helper. Keep the module dependency-light: only stdlib + `observability.METRICS` (no discord import, so `main_telegram.py` can still import it).

# BEFORE (bot_adapters.py:8-16)
def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]

# AFTER
import logging
import uuid

from observability import METRICS

logger = logging.getLogger(__name__)

FENCE = "```"


def _find_break(text: str, limit: int) -> int:
    """Index to cut `text` at (<= limit), preferring paragraph > line > word.

    Falls back to a hard cut at `limit` when no boundary sits in the second
    half of the window — a boundary at index 3 of a 2000-char budget would
    waste the whole message.
    """
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    for sep in ("\n\n", "\n", " "):
        idx = window.rfind(sep)
        if idx > limit // 2:
            return idx + len(sep)
    return limit


def _open_fence_lang(text: str) -> str | None:
    """Info string of a ``` fence left open at the end of `text`, else None."""
    lang = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            lang = stripped[len(FENCE):].strip() if lang is None else None
    return lang


def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    """Split `s` into chunks of at most chunk_size characters.

    Breaks on paragraph/line/word boundaries where one exists, and never
    leaves a ``` code fence unbalanced: a fence that spans a cut is closed
    at the end of one chunk and reopened (with its language tag) at the top
    of the next.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not s:
        return []
    if len(s) <= chunk_size:
        return [s]

    chunks: list[str] = []
    remaining = s
    carry_lang: str | None = None

    while remaining:
        prefix = f"{FENCE}{carry_lang}\n" if carry_lang is not None else ""
        budget = max(1, chunk_size - len(prefix))
        cut = _find_break(remaining, budget)
        piece = prefix + remaining[:cut]
        open_lang = _open_fence_lang(piece)
        if open_lang is not None and cut < len(remaining):
            # Ending inside a fence — re-cut to reserve room for the closer.
            budget = max(1, budget - (len(FENCE) + 1))
            cut = _find_break(remaining, budget)
            piece = prefix + remaining[:cut]
            open_lang = _open_fence_lang(piece)
            if open_lang is not None:
                piece = piece.rstrip("\n") + "\n" + FENCE
        remaining = remaining[cut:]
        carry_lang = open_lang if remaining else None
        chunks.append(piece)
    return chunks


_ERROR_COPY = (
    "I'm afraid that did not go to plan. The particulars have been noted in "
    "the log, where such things belong."
)


def fritz_error(operation: str, exc: BaseException | None = None, *, note: str | None = None) -> str:
    """Record `exc` and return user-facing copy in Mister Fritz's voice.

    The real exception goes to METRICS and the log only. The returned string
    carries an 8-char ref so an admin can grep the log for the exact failure.
    Set DISCORD_ERROR_DETAIL=1 to append the exception text for local debugging.
    """
    ref = uuid.uuid4().hex[:8]
    if exc is not None:
        METRICS.record_error(operation, exc)
        logger.exception("[%s] %s failed", ref, operation, exc_info=exc)
    else:
        logger.error("[%s] %s failed", ref, operation)
    parts = [_ERROR_COPY if note is None else note]
    if DISCORD_ERROR_DETAIL and exc is not None:
        parts.append(f"`{type(exc).__name__}: {exc}`")
    parts.append(f"(ref `{ref}`)")
    return " ".join(parts)

### `fritz_utils.py:193-198 (Discord section)`

Add the single new config knob, `DISCORD_ERROR_DETAIL`, next to `DISCORD_BOT_TOKEN` in the existing `# Discord` block, following the established `os.environ.get` + typed-constant convention.

# fritz_utils.py, after DISCORD_BOT_TOKEN (currently line 198)
DISCORD_BOT_TOKEN: str | None = _env_or_json("DISCORD_BOT_TOKEN", DISCORD_KEY)

# When true, user-facing error messages append the exception type and text.
# Off by default: end users get butler copy plus a log ref, and the traceback
# stays in the log. Turn on for local debugging of a private instance.
DISCORD_ERROR_DETAIL: bool = os.environ.get("DISCORD_ERROR_DETAIL", "").lower() in ("1", "true", "yes")

### `main_discord.py:35-83 (StreamingMessageHandler)`

Add a status line that renders inside the placeholder message (so progress no longer needs its own sends), replace the mid-stream head-truncation with a tail window so long streams keep moving, and move the >2000-char chunking into `final_update` so `on_message` stops duplicating it.

class StreamingMessageHandler:
    def __init__(self, message, loop, min_update_interval: float = 1.5):
        ...
        self.pending_text = None
        self.status_text: str | None = None   # NEW: butler-voiced progress line

    def _compose(self, body: str) -> str:
        """Render status line + streamed body, tail-windowed to fit 2000 chars.

        Head-truncation (the old `[:2000]`) freezes the visible text once the
        reply passes the cap; a tail window keeps the newest tokens on screen.
        The complete text is delivered by final_update's chunking.
        """
        head = f"{self.status_text}\n\n" if self.status_text else ""
        room = 2000 - len(head)
        if len(body) > room:
            body = "…" + body[-(room - 1):]
        return head + body

    async def set_status(self, status: str | None):
        """Show/clear the progress line inside the placeholder message."""
        self.status_text = status
        await self.update_text(self.pending_text or self.current_text)

    async def _perform_update(self):
        while self.is_updating:
            if self.pending_text is not None and self.pending_text != self.current_text:
                ...
                try:
                    await self.message.edit(content=self._compose(self.pending_text))   # was self.pending_text[:2000]
                    ...

    async def final_update(self, final_text: str, files: list = None):
        """Deliver the complete reply: edit the placeholder with chunk 1, send the rest."""
        while self.is_updating:
            await asyncio.sleep(0.1)
        self.status_text = None            # progress line disappears with the reply
        chunks = split_into_chunks(final_text) or [final_text]
        try:
            await self.message.edit(content=chunks[0])   # was final_text[:2000]
            for chunk in chunks[1:]:
                await self.message.channel.send(chunk)
            if files:
                await self.message.channel.send(files=files)
        except discord.errors.HTTPException as e:
            logger.warning("Error in final update: %s", e)

### `main_discord.py:152-176 (attachment loop)`

Track saved audio paths for cleanup and give the audio branch the same try/except the image branch already has.

    user_image_paths = []
    temp_audio_paths = []          # NEW — currently these files are never removed
    source = MessageSource.DISCORD_TEXT
    ...
            elif attachment.content_type and attachment.content_type.startswith('audio/'):
                try:                                        # NEW — image branch has this, audio didn't
                    os.makedirs("temp_audio", exist_ok=True)
                    file_path = os.path.join("temp_audio", f"{author}_{attachment.id}_{attachment.filename}")
                    await attachment.save(file_path)
                    temp_audio_paths.append(file_path)      # NEW
                    logger.info("Saved audio %s for %s", file_path, request_id)
                    voice_text = await speech_to_text(file_path)
                    if not message_clean:
                        message_clean = voice_text
                    logger.info("Voice text: %s", voice_text)
                except Exception as e:
                    METRICS.record_error("attachment_audio", e)
                    logger.warning("Error handling audio attachment: %s", e)

### `main_discord.py:185-189, 205-210, 214-216, 229-240`

Route progress into the placeholder instead of `ctx.channel.send`; replace the raw-exception edit and the sad-bot fallback with butler copy; drop the now-duplicated chunking loop; clean up audio temp files at both exit paths.

    def progress_callback(message: str):
        # BEFORE: asyncio.run_coroutine_threadsafe(ctx.channel.send(message), loop)
        #   -> a permanent, un-deleteable message per tool notice, orderable
        #      below the placeholder.
        asyncio.run_coroutine_threadsafe(streaming_handler.set_status(message), loop)

    try:
        ...
    except Exception as e:
        # BEFORE: await status_msg.edit(content=f"❌ An error occurred: {str(e)}")
        await status_msg.edit(content=fritz_error("ask_stuff", e))
        _cleanup_temp_files(user_image_paths + temp_audio_paths, request_id)   # was user_image_paths
        return

    if not response_data or not response_data.get("text"):
        # BEFORE: "The bot got sad and doesn't want to talk to you at the moment :("
        original_response = (
            "I find I have nothing whatsoever to say on the matter. "
            "A rare occurrence, and not one I intend to dwell upon. Do try again."
        )
        image_paths = []
    ...
    # BEFORE: an if len(original_response) > 2000 block re-chunking by hand
    #         (main_discord.py:229-238). final_update now owns chunking.
    await streaming_handler.final_update(original_response, files)

    _cleanup_temp_files(user_image_paths + temp_audio_paths, request_id)   # was user_image_paths


# main_discord.py:11 — split_into_chunks stays imported (final_update uses it and
# tests/test_discord_commands.py imports it from here), but the `# noqa: F401`
# comment can go since it is now genuinely used.
from bot_adapters import fritz_error, split_into_chunks

### `main_discord.py:106-131 (on_ready)`

Install a `tree.on_error` backstop so app-command failures outside the cog (e.g. `CommandNotFound` after a stale sync) still produce butler copy rather than a silent console traceback.

from bot_commands import FritzCommands, handle_app_command_error


@client.event
async def on_ready():
    ...
    await client.add_cog(FritzCommands(client, sayer, schedule_manager))
    client.tree.on_error = handle_app_command_error   # NEW — backstop outside the cog
    logger.info("Logged in as %s", client.user)

### `bot_commands.py:1-30 (imports) + new module constants`

Import `fritz_error`, define the shared embed colour matching the web accent, and add the ephemeral error-reply helper plus the module-level `handle_app_command_error` that both the cog hook and `tree.on_error` delegate to.

from bot_adapters import fritz_error, split_into_chunks

# Matches --accent in admin_templates/base.html:12 so the Discord and web
# surfaces share one brand colour.
FRITZ_COLOUR = discord.Colour(0x5B3F30)


async def _reply_error(interaction: discord.Interaction, operation: str,
                       exc: BaseException | None = None, *, note: str | None = None) -> None:
    """Send butler-voiced failure copy ephemerally, whichever response stage we're in."""
    text = fritz_error(operation, exc, note=note)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True)
        else:
            await interaction.response.send_message(text, ephemeral=True)
    except discord.HTTPException:
        logger.warning("Could not deliver error reply for %s", operation)


async def handle_app_command_error(interaction: discord.Interaction,
                                   error: app_commands.AppCommandError) -> None:
    """Single entry point for cog_app_command_error and tree.on_error."""
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

### `bot_commands.py:102-109 (FritzCommands class body)`

Add the cog-level app-command error hook. discord.py 2.6.4 dispatches to it automatically for every command defined in this cog (verified at .venv/Lib/site-packages/discord/ext/commands/cog.py:375-376, :649).

class FritzCommands(commands.Cog):
    """All MisterFritz slash commands."""

    def __init__(self, bot, sayer, schedule_manager=None):
        ...

    async def cog_app_command_error(self, interaction: discord.Interaction,
                                    error: app_commands.AppCommandError) -> None:
        await handle_app_command_error(interaction, error)

### `bot_commands.py:154-170 (schedule_list), 148-152, 189-193`

Turn `/schedule list` into an embed; swap the two raw-`{e}` handlers for `_reply_error`. Leave the ValueError (:149) and PermissionError (:190) branches alone — those carry author-written validation text, not tracebacks.

    @schedule.command(name="list", description="List your active scheduled tasks")
    async def schedule_list(self, interaction: discord.Interaction):
        ...
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

    # schedule_add, line 150-152:
        except Exception as e:
            # BEFORE: send_message(f"❌ Failed to create schedule: {e}", ephemeral=True)
            await _reply_error(interaction, "discord_commands.schedule_add", e)

    # schedule_remove, line 191-193:
        except Exception as e:
            # BEFORE: send_message(f"❌ Failed to remove schedule: {e}", ephemeral=True)
            await _reply_error(interaction, "discord_commands.schedule_remove", e)

### `bot_commands.py:304-308 (draw_slash)`

Bound `num_cards` with `app_commands.Range` and chunk the output — `cards.draw_cards` closes with a ``` fenced summary block (cards.py:75, :84), which is exactly the fence case the new chunker handles.

    @app_commands.command(name="draw", description="Draw cards from a deck!")
    @app_commands.describe(num_cards="How many cards to draw (1-40)")
    async def draw_slash(self, interaction: discord.Interaction,
                         num_cards: app_commands.Range[int, 1, 40]):
        METRICS.increment("discord_commands.draw")
        await interaction.response.defer(thinking=True)
        # 40 cards ≈ 1755 chars; chunk anyway so the trailing ``` block survives.
        for chunk in split_into_chunks(draw_cards(num_cards, interaction.user.name)):
            await interaction.followup.send(content=chunk)

### `bot_commands.py:327-330 (health_slash)`

Make /health an ephemeral embed. Reuse the existing `get_health_snapshot()` dict directly rather than the flat `format_health_text()` string so each stat gets its own field.

    @app_commands.command(name="health", description="Check the system health metrics")
    async def health_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.health")
        snap = get_health_snapshot()
        total_errors = sum(snap["errors"].values()) if snap["errors"] else 0
        embed = discord.Embed(title="Mister Fritz — status", colour=FRITZ_COLOUR)
        embed.add_field(name="Uptime", value=_format_uptime(int(snap["uptime_sec"])), inline=True)
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
            name, ts, _msg = snap["last_error"]   # message deliberately omitted
            embed.set_footer(text=f"Last error: {name}")
        # BEFORE: send_message(format_health_text(get_health_snapshot()))  — public!
        await interaction.response.send_message(embed=embed, ephemeral=True)

# NOTE: format_health_text is then unused in bot_commands.py — drop it from the
# import at line 24 or ruff F401 will flag it. observability.format_health_text
# itself stays (tests/test_observability.py exercises it directly).

### `bot_commands.py:332-386 (help_slash and about_slash)`

Convert both markdown walls to embeds and fix the /about uptime key bug at :367.

    @app_commands.command(name="help", description="Show what Mister Fritz can do")
    async def help_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.help")
        embed = discord.Embed(
            title="Mister Fritz — at your service",
            description="DM me, or `@mention` me in a channel, and I shall run the full agent.",
            colour=FRITZ_COLOUR,
        )
        embed.add_field(name="How to talk to me", value=(
            "• DM or `@mention` — full agent\n"
            "• Attach an image and I shall analyse it\n"
            "• Send a voice message and I shall transcribe it"
        ), inline=False)
        embed.add_field(name="Tools at my disposal", value=( ... ), inline=False)
        embed.add_field(name="Slash commands", value=( ... ), inline=False)
        embed.set_footer(text="Run /about for version and storage details.")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="about", description="Show version, models, and data storage info")
    async def about_slash(self, interaction: discord.Interaction):
        METRICS.increment("discord_commands.about")
        snap = get_health_snapshot()
        # BUG FIX: the key is "uptime_sec" (observability.py:304), not
        # "uptime_seconds" — the .get default made /about always report "0s".
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

### `bot_commands.py:408-410, 421-423, 425-442, 446-476, 506-510`

Butler-voice the remaining copy and stop leaking exceptions: /voice AttributeError, /gen failure, /lore (drop the header + use followup for continuations), /join and /leave 'buddy!' lines, /workspace enable failure.

    # voice_slash :408-410
        except AttributeError as e:
            # BEFORE: await interaction.followup.send("Something crazy happened!")
            #         (plus a bare METRICS.record_error — fritz_error does that now)
            await _reply_error(interaction, "discord_commands.voice", e,
                               note="The voice apparatus has declined to cooperate.")

    # gen_slash :421-423
        except Exception as e:
            # BEFORE: await interaction.followup.send(f"Failed to generate image: {e}")
            await _reply_error(interaction, "discord_commands.gen", e,
                               note="The image would not come. Some subjects resist depiction.")

    # lore_slash :433-442 — BEFORE built a header at :434 and used
    # interaction.channel.send at :440 for continuations.
        for chunk in split_into_chunks(original_response) or [original_response]:
            await interaction.followup.send(chunk)

    # join_slash :455-457 / leave_slash :471-473
        # BEFORE: "You are not connected to a voice channel, buddy!"
        "You are not in a voice channel. I can hardly join you in one you do not occupy."
        # BEFORE: "I am not connected to a voice channel, buddy!"
        "I am not in a voice channel. One cannot leave where one has never been."
    # join_slash :458-461 / leave_slash :474-476 exception arms -> _reply_error(...)

    # workspace_enable :506-510
        except Exception as e:
            # BEFORE: send_message(f"❌ Could not create workspace: {e}", ephemeral=True)
            await _reply_error(interaction, "discord_commands.workspace.enable", e,
                               note="The workspace could not be prepared.")
            return

### `mister_fritz.py:364-382, 403-404`

Rewrite the 17 tool-progress strings and the plan-step notice in Fritz's voice (per FRITZ_CHARACTER at mister_fritz.py:62-94: dry, understated, no exclamation marks). These strings also feed the web chat's SSE `progress` event via admin_panel.py:513-517, so both surfaces improve at once.

    tool_messages = {
        # BEFORE: "Generating an image, this may take a moment..."
        "generate_image":   "🎨 Conjuring an image. These things will not be rushed.",
        "search_documents": "📚 Consulting the library.",
        "search_web":       "🔍 Consulting the wider world. One does what one must.",
        "scrape_web":       "📄 Reading the page so that you need not.",
        "scrape_website":   "📄 Reading the page so that you need not.",
        "search_memories":  "🧠 Rifling through my recollections.",
        "save_memory":      "🗄️ Noting that down for posterity.",
        "analyze_image":    "👁️ Examining your image.",
        "schedule_message": "📅 Entering it in the diary.",
        "list_my_schedules":"📅 Reviewing your diary.",
        "cancel_reminder":  "🗑️ Striking that from the diary.",
        "list_directory":   "📁 Surveying the workspace.",
        "read_file":        "📄 Perusing a file.",
        "write_file":       "✍️ Committing it to disk.",
        "edit_file":        "✍️ Amending a file.",
        "search_files":     "🔎 Searching the files.",
        "execute_command":  "⚙️ Running a command. Under protest.",
    }

    # :403-404 — BEFORE: f"Step {current_step + 1}/{len(plan)}: {step_instruction}"
        if progress_callback:
            progress_callback(f"📋 Step {current_step + 1} of {len(plan)} — {step_instruction}")

### `.env.example:append to the Tunables block (after AUDIT_LOG_PATH, ~line 63)`

Document the one new knob, per the project convention that every knob appears here.

# When set, user-facing Discord error messages append the exception type and
# text. Off by default — users get butler copy plus a log ref, and the
# traceback stays in the log. Development only.
# DISCORD_ERROR_DETAIL=0

### `CHANGELOG.md:insert under [Unreleased] alongside the other Phase entries (~line 9-40)`

Add a phase-style entry matching the existing prose style (bolded phase heading, nested bullets, named env vars in backticks).

### Changed
- **Phase 15 — Discord surface polish.**
  - `split_into_chunks` is now boundary- and fence-aware: breaks on paragraph > line > word, and closes/reopens ``` code fences (with the language tag) across a cut instead of shipping half a fence.
  - New `bot_adapters.fritz_error(operation, exc, note=…)`: records the exception to `METRICS` and the log with an 8-char ref, and returns user-facing copy in Fritz's voice. Every `f"...{e}"` that reached a Discord user is gone. New `DISCORD_ERROR_DETAIL` env var re-enables the raw text for local debugging.
  - Tool and plan-step progress now renders inside the streaming placeholder rather than as separate permanent `channel.send`s, so it can no longer land below the reply or litter the channel.
  - `/help`, `/about`, `/health`, and `/schedule list` are `discord.Embed`s in `#5b3f30` — the same accent the admin panel and web chat use. `/health` is now ephemeral.
  - New `FritzCommands.cog_app_command_error` plus a `tree.on_error` backstop. Slash-command failures produce butler copy instead of a silent console traceback.
  - `/draw` is bounded by `app_commands.Range[int, 1, 40]` and its output is chunked — a large draw used to raise an uncaught `HTTPException` and show the user nothing.
  - `/lore` continuations use `followup.send` instead of `channel.send`, and the "The answer was over 2000…" header is gone.

### Fixed
- `/about` reported `0s` uptime on every invocation — it read `uptime_seconds` from the health snapshot, whose key is `uptime_sec`.
- Voice-message attachments saved under `temp_audio/` were never deleted. They are now cleaned up on both the success and failure paths, like `temp_images/` already was.

### `README.md:35, 38, 253`

Update the slash-command table for the new /draw bound and the now-ephemeral /health, and the test-coverage bullet that names split_into_chunks.

| `/draw [n]` | Draw cards from a deck (1-40) |
| `/health` | Show system health and metrics (ephemeral) |
...
- `bot_adapters` — fence-aware `split_into_chunks`, `fritz_error` copy/logging
- `discord_commands` — `StreamingMessageHandler` rate limiting and status line

## Steps

1. Step 1 — chunker. Rewrite `bot_adapters.split_into_chunks` with `_find_break` and `_open_fence_lang` as sketched. Do NOT touch `tests/test_discord_commands.py` yet. Run `pytest tests/test_discord_commands.py -q`. All six `TestSplitIntoChunks` tests are expected to still pass: the design only narrows the budget (and thus changes chunk lengths) on the re-cut branch, which fires only when a chunk ends inside an open fence — none of the six fixtures contain a fence. If any of them fails, the design has drifted; stop and reconcile before continuing. Commit.
2. Step 2 — chunker tests. Add a `TestFenceAwareChunking` class to `tests/test_discord_commands.py` beneath `TestSplitIntoChunks` (same file: it is already the home for chunker tests and it already imports the symbol via `from main_discord import split_into_chunks`). Cover: fence reopened with its language tag; every chunk has an even count of ``` fence lines; no chunk exceeds `chunk_size`; a word boundary is preferred over a mid-word cut; a single long unbroken token still hard-splits. Commit.
3. Step 3 — error helper. Add `DISCORD_ERROR_DETAIL` to `fritz_utils.py` (Discord section, after `DISCORD_BOT_TOKEN`) and to `.env.example`. Add `fritz_error()` to `bot_adapters.py`, importing `METRICS` from `observability` (no circular import: `observability` imports only stdlib plus optional `prometheus_client`). Add unit tests: the returned string contains neither `str(exc)` nor the exception class name when the flag is off; it does when the flag is on; `METRICS.record_error` is called with the operation name. Commit.
4. Step 4 — Discord error plumbing. In `bot_commands.py`: add `FRITZ_COLOUR`, `_reply_error`, and the module-level `handle_app_command_error`; add `FritzCommands.cog_app_command_error` delegating to it. In `main_discord.py` `on_ready`, set `client.tree.on_error = handle_app_command_error` after `add_cog`. Commit.
5. Step 5 — replace every leaking/persona-breaking string. Work through `bot_commands.py` in line order: `:152`, `:193`, `:410`, `:423`, `:456`, `:461`, `:472`, `:476`, `:509`. Leave `:149` and `:190` untouched (author-written `ValueError`/`PermissionError` text, and `tests/test_bot_commands.py::TestScheduleAddOpenToAll::test_value_error_surfaces_as_user_message` asserts on it). Then `main_discord.py:208` and `:215`. Run `pytest tests/test_bot_commands.py -q` — expect `test_permission_error_surfaces_to_caller` to still pass (that path is unchanged) but re-verify. Commit.
6. Step 6 — embeds. Convert `/schedule list` (`:154-170`), `/health` (`:327-330`, add `ephemeral=True`), `/help` (`:332-361`), and `/about` (`:363-386`, fixing the `uptime_seconds` → `uptime_sec` key bug at `:367`). Drop `format_health_text` from the `observability` import at `bot_commands.py:24` once `/health` no longer calls it, or ruff F401 will fail CI. Run `ruff check .`. Commit.
7. Step 7 — streaming handler. Add `status_text`, `_compose`, and `set_status` to `StreamingMessageHandler`; replace the `[:2000]` head-truncation at `:63` with `_compose`; move chunking into `final_update` and delete the duplicated `if len(original_response) > 2000` block at `main_discord.py:229-238`. Rewire `progress_callback` at `:188-189` to call `streaming_handler.set_status`. Note: `streaming_handler` is defined at `:183`, before the callbacks at `:185-189`, so the closure is already in scope — no reordering needed. Update `tests/test_discord_commands.py::TestStreamingMessageHandler` as described in the test plan. Commit.
8. Step 8 — /draw and /lore. Apply `app_commands.Range[int, 1, 40]` and chunked followups to `draw_slash` (`:304-308`); rewrite `lore_slash`'s `:433-442` block to drop the header and use `interaction.followup.send` for every chunk. Add a `METRICS.increment("discord_commands.draw")` while there — the other commands all have one and `/draw` does not. Commit.
9. Step 9 — temp_audio cleanup. Add `temp_audio_paths` to `on_message`, wrap the audio branch (`:168-176`) in try/except like the image branch, and pass `user_image_paths + temp_audio_paths` to `_cleanup_temp_files` at both `:209` and `:240`. Manually delete the 9 already-leaked files under `temp_audio/`. Commit.
10. Step 10 — butler-voiced tool notices. Rewrite `mister_fritz.py:364-382` `tool_messages` and the plan-step string at `:404`. Run the full suite: `pytest tests/ -q`. Confirm `tests/test_admin_panel.py::TestChatStreamProgressEvents::test_progress_callback_yields_progress_events` still passes — it uses its own literal strings inside `fake_ask_stuff`, so it is decoupled from this dict, but verify rather than assume. Commit.
11. Step 11 — docs. CHANGELOG.md Phase 15 entry under `[Unreleased]`, README table rows 35/38 and the test bullet at 253, `.env.example` already done in step 3. Run `ruff check .` and `pytest tests/ --cov=. --cov-fail-under=60` one final time. Commit.

## Config and env changes

- `DISCORD_ERROR_DETAIL` (new, default false) — when truthy (`1`/`true`/`yes`), `bot_adapters.fritz_error` appends `type(exc).__name__: exc` to the user-facing message. Off by default so end users get butler copy plus a log ref only. Defined in `fritz_utils.py` in the `# Discord` block after `DISCORD_BOT_TOKEN`; documented commented-out in `.env.example` under the Tunables block.
- No other new knobs. The embed colour `0x5B3F30` is a hardcoded module constant (`bot_commands.FRITZ_COLOUR`) mirroring `--accent` in `admin_templates/base.html:12` — a per-deployment brand colour env var is not worth a knob for a single-persona bot.
- No changes to `requirements.txt`. `discord.py==2.6.4` (line 44) already provides `discord.Embed`, `app_commands.Range`, `Cog.cog_app_command_error`, and `CommandTree.on_error`.

## Tests
### New

- `tests/test_discord_commands.py::TestFenceAwareChunking::test_open_fence_is_closed_and_reopened` — build a string with a ```python fence whose body exceeds `chunk_size`; assert chunk 0 ends with a bare ``` line and chunk 1 starts with ```python.
- `tests/test_discord_commands.py::TestFenceAwareChunking::test_every_chunk_has_balanced_fences` — for each chunk, `sum(1 for line in chunk.split("\n") if line.lstrip().startswith("```")) % 2 == 0`.
- `tests/test_discord_commands.py::TestFenceAwareChunking::test_no_chunk_exceeds_limit` — assert `max(len(c) for c in chunks) <= chunk_size` on fenced input (guards the reserve-room-for-the-closer arithmetic).
- `tests/test_discord_commands.py::TestFenceAwareChunking::test_prefers_word_boundary` — prose input; assert no chunk starts or ends mid-word (`chunk[0]` is not a lowercase letter following a letter in the previous chunk's tail).
- `tests/test_discord_commands.py::TestFenceAwareChunking::test_unbreakable_token_hard_splits` — `"a" * 5000` still yields chunks of exactly 2000/2000/1000 (documents that the fence reserve does not fire on fence-free input).
- `tests/test_bot_adapters.py` (new file — `bot_adapters` has no heavy imports, so this needs none of the `sys.modules` mocking dance) — `TestFritzError::test_exception_text_not_in_message` (assert `str(exc)` and `type(exc).__name__` absent from the return value with the flag off), `::test_detail_flag_includes_exception` (patch `bot_adapters.DISCORD_ERROR_DETAIL` True), `::test_metrics_record_error_called` (patch `bot_adapters.METRICS`), `::test_ref_is_stable_within_one_call` (the ref in the returned string also appears in the caplog record).
- `tests/test_bot_commands.py::TestAppCommandErrorHandler` — new class. `test_command_invoke_error_replies_ephemerally_without_traceback`: construct `app_commands.CommandInvokeError(MagicMock(), RuntimeError("boom: /secret/path"))`, call `await cog.cog_app_command_error(interaction, err)`, assert `interaction.response.send_message` called with `ephemeral=True` and that `"boom"` and `"/secret/path"` are absent from the sent text. `test_check_failure_gets_permission_copy`. `test_uses_followup_when_response_already_done`: set `interaction.response.is_done.return_value = True` and assert `interaction.followup.send` was used instead.
- `tests/test_bot_commands.py::TestHealthIsEphemeralEmbed` — call `await cog.health_slash.callback(cog, interaction)` and assert `send_message` kwargs contain `ephemeral=True` and an `embed` whose `colour == discord.Colour(0x5B3F30)`.
- `tests/test_bot_commands.py::TestAboutUptime::test_about_reports_real_uptime` — regression for the `uptime_sec` key bug: patch `bot_commands.get_health_snapshot` to return `{"uptime_sec": 3661, "counters": {}, "errors": {}, "latencies": {}, "last_error": None}` and assert the embed's uptime field is `"1h 1m"`, not `"0s"`.
- `tests/test_bot_commands.py::TestErrorCopyHasNoRawExceptions` — parametrise over `schedule_add`, `schedule_remove`, `gen_slash`, `workspace_enable` with a side-effect `Exception("SENTINEL_LEAK_TOKEN")`; assert `"SENTINEL_LEAK_TOKEN"` never appears in any `send_message`/`followup.send` argument.
- `tests/test_discord_commands.py::TestStreamingStatusLine` — `test_set_status_renders_above_body` (call `set_status`, then `update_text`, assert the edited content starts with the status text); `test_final_update_clears_status` (assert the final edit contains no status text); `test_long_stream_shows_tail_not_head` (feed 3000 chars, assert the edited content ends with the last characters of the body, i.e. head truncation is gone).

### Existing tests affected

- `tests/test_discord_commands.py::TestSplitIntoChunks` (all six: `test_short_string_not_split`, `test_exact_boundary_not_split`, `test_long_string_split_correctly`, `test_chunks_reassemble_to_original`, `test_empty_string_returns_empty_list`, `test_custom_chunk_size`) — I traced all six against the new algorithm and expect ZERO edits. Specifically: `test_long_string_split_correctly` ("a"*5000 → 2000/2000/1000) survives because the 4-char fence reserve is applied only on the re-cut branch, which never fires on fence-free input; `test_chunks_reassemble_to_original` survives because `_find_break` returns `idx + len(sep)`, keeping the separator at the tail of the preceding chunk. This is the highest-risk assumption in the plan — run these six FIRST after step 1 and reconcile before proceeding if any fails.
- `tests/test_discord_commands.py::TestStreamingMessageHandler::test_final_update_long_text_truncated_in_edit` — MUST BE RENAMED AND STRENGTHENED. The assertion (`len(content) <= 2000`) still passes once `final_update` chunks, since chunk 0 is ≤ 2000. But the name now lies: the remainder is no longer discarded. Rename to `test_final_update_long_text_chunks_into_followups` and add `msg.channel.send.assert_called()` plus an assertion that the concatenation of the edit content and the sent chunks covers the whole 2500-char input.
- `tests/test_discord_commands.py::TestStreamingMessageHandler::test_final_update_short_text` — asserts `msg.edit.assert_called_with(content="Short response")`. Still passes: `split_into_chunks("Short response")` short-circuits to `[s]`. Verify, do not pre-emptively edit.
- `tests/test_discord_commands.py::TestStreamingMessageHandler::test_final_update_with_files_sends_separately` and `::test_update_text_calls_edit`, `::test_pending_text_tracks_latest`, `::test_rate_limiting_respected` — all still pass; `_compose` is a pass-through when `status_text` is None and the body is under 2000. Verify.
- `tests/test_bot_commands.py::TestScheduleAddOpenToAll::test_value_error_surfaces_as_user_message` — asserts `"max 10"` appears in the reply. This is why `bot_commands.py:149` must NOT be routed through `fritz_error`. If step 5 touches that line, this test breaks. Treat it as the guard rail it is.
- `tests/test_bot_commands.py::TestScheduleRemoveOpenToAll::test_permission_error_surfaces_to_caller` — same reasoning for `:190`. Unchanged.
- `tests/test_bot_commands.py::TestScheduleListOpenToAll::test_non_admin_can_list_their_own_schedules` — asserts only `manager.list_schedules.assert_called_once_with("regular_user")` with a `[]` return, which hits the early-return branch at `bot_commands.py:161-165`. The embed conversion is below that branch, so this test is unaffected. Verify.
- `tests/test_bot_commands.py::TestScheduleListAllAdminOnly::test_admin_can_view_all_schedules` — `/schedule list_all` still uses `split_into_chunks` at `:215` and asserts `send_message` called once. Two schedules produce far less than 2000 chars → one chunk → one call. Unaffected. `/schedule list_all` is deliberately NOT converted to an embed (see deferred refactors).
- `tests/test_admin_panel.py::TestChatStreamProgressEvents::test_progress_callback_yields_progress_events` (line 621-642) — asserts on `["Searching the web...", "Reading results..."]`, but those literals are emitted by the test's own `fake_ask_stuff`, not by `mister_fritz.tool_messages`. Changing the tool copy does NOT break it. Named here because it is the one test that looks like it would.
- `tests/test_observability.py::` (lines 173-207) — exercises `format_health_text` directly. `/health` no longer calls it, but `observability.format_health_text` stays in place, so these are unaffected. Do not delete the function.

### Manual verification

- Run the bot against a real Discord guild. DM it something that triggers a multi-tool plan (e.g. "search the web for X, then save what you learn"). Confirm: the progress line appears INSIDE the placeholder message, updates in place as tools fire, and vanishes when the reply lands — with zero leftover progress messages in the channel.
- Ask for a long code answer (>2000 chars, containing a ```python block that straddles the 2000 boundary). Confirm both chunks render as proper code blocks in the Discord client, with no half-fence and no mid-word break.
- Run `/draw 40` — confirm the reply arrives complete with a rendered ``` summary block. Then attempt `/draw 500`: the Discord client should refuse it in the picker (Range is enforced client-side). To exercise the server-side path, temporarily narrow the Range to `[1, 2]` without re-syncing the command tree and invoke `/draw 5` — the `TransformerError` arm of `handle_app_command_error` should fire with butler copy.
- Run `/health` in a shared channel — confirm only you see it. Run `/help`, `/about`, `/schedule list` — confirm all four render as embeds with the brown `#5b3f30` sidebar.
- Stop Ollama, then DM the bot. Confirm the placeholder becomes butler copy with a `(ref …)` and NO traceback, and that grepping the log for that ref surfaces the full exception.
- Send a voice message. Confirm the transcription works and that `ls temp_audio/` is empty afterwards.
- Cannot be settled statically: whether `set_status` scheduled via `asyncio.run_coroutine_threadsafe` from the `run_in_executor` worker interleaves cleanly with the token stream under the 1.5s `min_update_interval`. The experiment: trigger a plan with 3+ steps and watch for a flickering or stuck status line. If it flickers, serialise `set_status` and `update_text` behind a single `asyncio.Lock` on the handler — this is exactly the surface the token-streaming item reworks, which is why that item lands first.

## Risks

- The single largest assumption is that all six existing `TestSplitIntoChunks` tests survive the chunker rewrite unmodified. I traced the algorithm by hand against each fixture but did not execute the new code. Detection: run `pytest tests/test_discord_commands.py -k SplitIntoChunks` immediately after step 1, before anything else is touched. If `test_long_string_split_correctly` fails with 1996/1996/1008, the fence reserve is being applied unconditionally — move it into the re-cut branch as sketched.
- Moving chunking into `final_update` means the handler now issues N sends where it previously issued one edit. Under Discord's per-channel rate limit a 10-chunk reply could be throttled mid-delivery, leaving the user with a partial answer. Detection: watch for `discord.errors.HTTPException` / 429 in the log during the long-code-answer manual test. Mitigation if it bites: `await asyncio.sleep(0.5)` between continuation sends, or cap continuations at 5 chunks with a "the remainder is available on request" tail.
- Routing progress into the placeholder means a slow tool sequence overwrites the streamed body's edit budget (`min_update_interval=1.5s`). If a plan fires 6 tools in 3 seconds, some notices will be coalesced away and the user sees fewer of them than before. This is arguably desirable, but it IS a behaviour change. Detection: the multi-tool manual test. This interacts directly with the token-streaming item — resolve the edit-scheduling policy there, not here.
- `fritz_error` in `bot_adapters` imports `observability.METRICS` at module scope. `main_telegram.py:11-14` imports `bot_adapters` transitively via nothing today (it does not import it at all) but `bot_commands.py:11` and `main_discord.py:11` do. Detection: `python -m compileall -q .` in CI plus the existing import-order dance in `tests/test_discord_commands.py`. No cycle exists (`observability` imports only stdlib), but confirm with a bare `python -c "import bot_adapters"`.
- `cog_app_command_error` only catches errors from commands defined inside `FritzCommands`. Everything currently is, but a future command registered directly on `client.tree` would bypass it — hence the `tree.on_error` backstop in `on_ready`. Detection: none automatic; the backstop is the mitigation.
- Embed field values are capped at 1024 chars and an embed at 6000 total. `/schedule list` with 10 long prompts, or `/health` with many latency entries, can silently drop content if the caps are not respected. Mitigation is in the sketches (`[:1024]`, `[:25]` fields); detection is the manual `/schedule list` check with a maxed-out schedule set.
- Emoji in the butler-voiced tool notices (🎨, 🗄️, 👁️) are multi-codepoint in a couple of cases and will render inconsistently in the web chat's plain-text `progress` SSE line vs Discord. Low severity; detection is the web-chat manual pass at `:8001/chat`.

## Rollback
"No feature flag is warranted — this is presentation-layer only, with no schema, no persisted state, and no external contract. The work is split into 11 individually-committable steps precisely so any one can be reverted alone. The two steps with the widest blast radius are step 1 (chunker, affects `/lore`, `/schedule list_all`, and every long DM reply) and step 7 (StreamingMessageHandler, affects every DM); `git revert` of either is clean because neither adds imports the other steps depend on — step 7 uses `split_into_chunks`, which exists in both the old and new form with an identical signature. `DISCORD_ERROR_DETAIL=1` is the operational escape hatch if the new error copy turns out to hide something an admin needs mid-incident; it requires only an env change and a restart, no code revert. The `temp_audio` cleanup (step 9) is the only step that deletes data — it removes files the bot itself created and already treats as disposable, matching the existing `temp_images` behaviour, and `stt.transcribe` has already consumed the file by the time cleanup runs."

## Open questions for you to decide

- Is `DISCORD_ERROR_DETAIL` worth adding at all? It partially undoes the fix it ships alongside, and the `(ref …)` token already lets an admin grep the log for the exact exception. Drop it and the plan loses its only new config knob. My weak recommendation is to keep it — this is a homelab bot whose operator is usually the person hitting the error — but it is the first thing to cut if you want the change smaller.
- Should `/health` be admin-gated via `_require_admin` in addition to being ephemeral? It exposes error counts, per-tool latency, and the last error's operation name. Ephemeral stops it leaking to a channel; it does not stop a curious guild member from reading it. The audit only asked for ephemeral, so that is what the plan does.
- Is `Range[int, 1, 40]` the right bound for `/draw`? I measured 40 cards at roughly 1755 chars including a deck-reload line, so it fits one message; 50 is roughly 2150 and does not. Since step 8 also chunks the output, a higher bound is technically safe — the question is whether drawing more than 40 cards at once is a real use case or just a footgun.
- The plan leaves `/schedule list_all` (`bot_commands.py:195-218`) as chunked plain text rather than an embed, because it is the one command whose output is genuinely unbounded and embed field caps would force a second, different truncation strategy. Confirm that asymmetry with `/schedule list` is acceptable, or fold it in as a follow-up.
- Where should the fence-aware chunker's tests live? The plan adds them to `tests/test_discord_commands.py` because that file already owns `TestSplitIntoChunks`, but the symbol actually lives in `bot_adapters.py`, which has no heavy imports and could be tested without the `sys.modules` mocking preamble. A cleaner long-term home is a `tests/test_bot_adapters.py` that owns both the chunker and `fritz_error` tests, with the `main_discord` re-export tests staying put. The plan splits the difference — `fritz_error` tests go in the new file, chunker tests stay next to their siblings. Reasonable people could move all of them.
- DEFERRED, deliberately: (a) collapsing the near-identical `join_slash`/`leave_slash` bodies and the eight `METRICS.increment("discord_commands.X")` calls behind a decorator; (b) extracting an `EmbedBuilder` for the four informational commands — with only four call sites the abstraction costs more than it saves; (c) unifying `main_discord.StreamingMessageHandler` with the web chat's SSE token path in `admin_panel.py:509-552`, which are solving the same problem twice. All three are tempting while in this code and all three belong in their own item.
- Cannot be settled statically: whether Ollama's streaming cadence through `mister_fritz.py:416-433` produces progress and token callbacks close enough together to make the in-placeholder status line flicker. The experiment is the multi-tool manual test in the test plan; the fix, if needed, is an `asyncio.Lock` on the handler, and it belongs to the token-streaming item this one depends on.
