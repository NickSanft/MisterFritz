# 2. Get blocking work off the Discord event loop

[← back to index](README.md)

**Effort:** M (half day)  
**Depends on:** nothing

## Goal
Today three slash commands (`/voice`, `/gen`, `/lore`) run multi-second-to-multi-minute CPU/GPU work directly on the Discord event loop, so a single `/gen` freezes heartbeats, streaming edits, `on_message`, and every other user's interaction for the duration. After this change every blocking call in `bot_commands.py` runs on a shared, bounded thread pool exposed as `bot_adapters.run_blocking(...)`; `TTSEngine.generate_speech` becomes a real synchronous function (it is currently `async def` with zero awaits, which is a lie that also silently breaks the `tts.py` CLI); SDXL and XTTS are serialised by an `asyncio.Semaphore` on the cog plus a process-wide `threading.Lock` in `image_generator` so the newly-possible concurrency cannot thrash or OOM the GPU; and `main_discord.on_message` is moved onto the same bounded pool so there is exactly one place to reason about worker capacity. The bot stays responsive to `/hello` and DMs while an image renders.

## Definition of done

- [ ] `grep -n 'run_blocking\|run_in_executor' bot_commands.py` shows every one of the three blocking call sites wrapped: `ask_stuff` (was :395), `self.sayer.generate_speech` (was :396), `generate_image` (was :419), `query_documents` (was :431).
- [ ] `tts.py` line 64 reads `def generate_speech(` — no `async` — and `python tts.py "testing one two"` writes a real .wav and prints its path (today it prints `<coroutine object ...>` and emits a never-awaited RuntimeWarning).
- [ ] `bot_adapters.py` exports `run_blocking` backed by a single named `ThreadPoolExecutor` sized from `fritz_utils.BLOCKING_POOL_SIZE`; `main_discord.py` uses it for `ask_stuff`, `speech_to_text`, and the startup `TTSEngine` load instead of `run_in_executor(None, ...)`.
- [ ] Two concurrent `/gen` invocations never run `image_generator.generate_image` at the same time (asserted by a test that records peak concurrency == 1), and waiting callers park on the asyncio semaphore, not on a pool thread.
- [ ] `image_generator.get_pipeline()` cannot double-load the 7 GB SDXL pipeline when called from two threads — the lazy init is under a module-level lock.
- [ ] New env knobs `BLOCKING_POOL_SIZE`, `IMAGE_GEN_MAX_CONCURRENCY`, `TTS_MAX_CONCURRENCY` are declared in `fritz_utils.py` and documented in `.env.example`.
- [ ] `ruff check .` passes and `pytest tests/ --cov=. --cov-fail-under=60` passes (the CI gate in `.github/workflows/ci.yml:38`).
- [ ] CHANGELOG.md has a `**Phase 15 — Discord event loop.**` bullet under the existing `### Performance` section.

## Current state (verified against the working tree)
CONFIRMED — every audit claim checked by reading the files this session, with two corrections noted below.

`bot_commands.py` has zero `run_in_executor` / `to_thread` / `asyncio` usage anywhere (its imports are `io, json, logging, os, typing` at :1-5). The cog is live: `main_discord.py:124` does `await client.add_cog(FritzCommands(client, sayer, schedule_manager))`.

Blocking calls on the loop, all confirmed at the stated lines:
- `bot_commands.py:395` — `ask_stuff(message, MessageSource.DISCORD_VOICE, interaction.user.name)["text"]` inside `async def voice_slash` (:392). This is the same full LangGraph agent that `main_discord.py:193` is careful to offload.
- `bot_commands.py:396` — `output_file = await self.sayer.generate_speech(original_response)`. `tts.py:64` declares `async def generate_speech(...)` and its body (:83-120) contains no `await` at all; it calls `self.tts.tts_to_file(...)` synchronously at :98-104 and :107-113. Awaiting it therefore runs Coqui XTTS inference inline on the loop.
- `bot_commands.py:419` — `output_file = generate_image(prompt)`, imported from `image_generator` at :22. `image_generator.generate_image` (:171-211) calls `get_pipeline()` (:20-54), which on first call lazily downloads/loads `stabilityai/stable-diffusion-xl-base-1.0` in fp16 and moves it to CUDA — the ~7 GB first-call cost — then runs 25 denoising steps.
- `bot_commands.py:431` — `original_response = query_documents(query)`, imported from `document_engine` at :13. `document_engine.query_documents` is at :730 (audit said 739-742 for the lazy ingest; that is the guard block inside it — confirmed exactly: `with VECTORSTORE_LOCK: needs_init = GLOBAL_VECTORSTORE is None` then `if needs_init: initialize_vectorstore()` at :739-742). `initialize_vectorstore` (:379) walks `DOC_FOLDER`, builds `OllamaEmbeddings`, and ingests the whole corpus — unbounded first-call latency.

The correct pattern already exists in the same repo: `main_discord.py:193-203` (`await loop.run_in_executor(None, lambda: ask_stuff(...))`), `main_discord.py:185-189` (callback marshaling via `asyncio.run_coroutine_threadsafe(..., loop)`), `main_discord.py:243-246` (`speech_to_text`), `main_telegram.py:26,45,52`, `scheduler.py:109`, `admin_panel.py:388,457,630,642`.

CORRECTION 1 to the audit: it says making `generate_speech` sync would break "its CLI caller at tts.py:143". It would not — that caller is **already broken**. `tts.py:143` is `output = engine.generate_speech(message=args.text, ...)` with no `await` inside a plain `def main()`, so today it binds a coroutine object and `tts.py:149` prints `File generated at: <coroutine object TTSEngine.generate_speech at 0x...>` while never synthesising anything. Making the method sync **fixes** the CLI. Grep across the repo confirms the only two callers of `generate_speech` are `bot_commands.py:396` and `tts.py:143`; `admin_panel.py` does not use TTS.

CORRECTION 2 / additional finding the audit missed: the GPU concurrency hazard is not purely hypothetical-after-this-change. `agent_tools.py:295-306` exposes `generate_image` as an agent tool (`agent_tools.py:533`), and that path already runs in a worker thread via `main_discord.py:193`. So two concurrent DMs can *already today* race `image_generator.get_pipeline()`'s unguarded `if _pipeline is None: ... _pipeline = ...` (image_generator.py:22-30) and double-load the pipeline. Offloading `/gen` widens the window, so the lock belongs in `image_generator`, not only in the cog.

Marshaling question — answered, no work needed: `/voice` calls `ask_stuff` positionally with only `(message, source, user_id)` (bot_commands.py:395), leaving `progress_callback`/`streaming_callback` at their `None` defaults (`mister_fritz.py:523-533`). `image_generator.generate_image` (:171) has no callback parameter. `document_engine.query_documents` (:730) takes only `user_input` and `include_sources`. So none of the three commands passes a callback into worker-thread code, and no `run_coroutine_threadsafe` marshaling is required. The only new cross-thread → loop communication introduced here is the optional "queued" notice, which is sent from the event loop *before* entering the executor.

Executor sizing today: `main_discord.py` uses `run_in_executor(None, ...)` (the loop's default `ThreadPoolExecutor`, `min(32, cpu_count+4)` workers on 3.12) at :112, :193, :246; `scheduler.py:109` and `admin_panel.py` share that same default pool. Nothing bounds concurrent Ollama/GPU work.

Test state: no existing test in `tests/` exercises `voice_slash`, `gen_slash`, or `lore_slash`, and nothing imports `tts` for real (`tests/test_bot_commands.py:27` and `tests/test_discord_commands.py:27-28` install `sys.modules` stubs). So nothing currently breaks — but `tests/test_bot_commands.py:37-46 _fake_interaction()` only stubs `interaction.response.send_message`; it has no `defer` or `followup.send`, which is exactly why these three commands have never been tested and is the first thing that must change.

## Change sites

### `bot_adapters.py:1-16 (append after split_into_chunks)`

Add the shared bounded thread pool and the `run_blocking` helper. This is the only new shared machinery; the module's docstring rule ("anything platform-specific belongs in main_*.py") is respected because run_blocking is platform-neutral.

# --- add to the import block at the top of bot_adapters.py ---
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from fritz_utils import BLOCKING_POOL_SIZE

# ... existing split_into_chunks() unchanged ...

# One shared, *bounded* pool for every blocking call the bots make. Bounded on
# purpose: Ollama and the GPU serialise anyway, so an unbounded pool only
# queues the contention one layer deeper where it is invisible.
_BLOCKING_POOL = ThreadPoolExecutor(
    max_workers=BLOCKING_POOL_SIZE,
    thread_name_prefix="fritz-blocking",
)


async def run_blocking(func, *args, **kwargs):
    """Run a blocking callable on the shared pool and await its result.

    Use this for anything that touches Ollama, the GPU, or the disk from inside
    an async Discord handler. Exceptions propagate to the awaiting coroutine.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _BLOCKING_POOL, functools.partial(func, *args, **kwargs)
    )

### `fritz_utils.py:insert after 119 (MEMORY_EXTRACT_MIN_REPLY_CHARS), before the admin-panel block at 121`

Declare the three new env knobs in the existing '# Tunables (formerly magic numbers in module bodies)' section, matching the surrounding comment style.

# Size of the shared thread pool that the bot adapters use for blocking work
# (ask_stuff, SDXL, XTTS, Whisper, RAG). Bounded deliberately: the underlying
# Ollama / GPU resources serialise anyway. Raise it if /health shows ask_stuff
# latency climbing while Ollama itself is idle.
BLOCKING_POOL_SIZE: int = int(os.environ.get("BLOCKING_POOL_SIZE", "8"))

# Max concurrent Stable Diffusion XL generations. 1 by default — a second
# concurrent generation on the same CUDA device thrashes VRAM (or OOMs)
# rather than finishing sooner.
IMAGE_GEN_MAX_CONCURRENCY: int = int(os.environ.get("IMAGE_GEN_MAX_CONCURRENCY", "1"))

# Max concurrent Coqui XTTS syntheses. Same reasoning as above.
TTS_MAX_CONCURRENCY: int = int(os.environ.get("TTS_MAX_CONCURRENCY", "1"))

### `tts.py:18-48 (__init__), 64-69 (signature), 94-120 (body)`

Drop `async` from `generate_speech` — it has no awaits and is a synchronous XTTS call. Add an instance lock so the shared `self.tts` object cannot be driven by two threads at once, which becomes possible the moment the cog offloads it.

# top of file
import threading

class TTSEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        ...
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Coqui's TTS object is not safe for concurrent inference. Callers now
        # reach generate_speech from a worker-thread pool, so serialise here.
        self._synth_lock = threading.Lock()
        logger.info(f"Initializing TTS on device: {self.device}")
        ...

-   async def generate_speech(self,
+   def generate_speech(self,
                        message: str,
                        speaker: str = "Baldur Sanjin",
                        ...) -> str:
        ...
        logger.info(f"Generating audio for: '{message[:30]}...'")
        try:
-           if reference_wav:
-               ...
-               self.tts.tts_to_file(...)
-           else:
-               ...
-               self.tts.tts_to_file(...)
+           with self._synth_lock:
+               if reference_wav:
+                   logger.info(f"Cloning voice from: {reference_wav}")
+                   self.tts.tts_to_file(
+                       text=message, file_path=str(output_path),
+                       speaker_wav=reference_wav, language=language,
+                       split_sentences=True,
+                   )
+               else:
+                   logger.info(f"Using speaker: {speaker}")
+                   self.tts.tts_to_file(
+                       text=message, file_path=str(output_path),
+                       speaker=speaker, language=language,
+                       split_sentences=True,
+                   )
            logger.info(f"Generation successful: {output_path}")
            return str(output_path)

# tts.py:143 in main() needs NO edit — `output = engine.generate_speech(...)`
# becomes correct for the first time once the method is sync.

### `image_generator.py:18 (_pipeline), 20-54 (get_pipeline), 171-211 (generate_image)`

Guard the lazy pipeline init and the generation itself with a module-level lock so the slash command, the agent tool (agent_tools.py:306), and any future caller cannot double-load 7 GB or run two CUDA graphs concurrently.

import threading

# Global pipeline instance - loaded once and reused
_pipeline = None
# Guards both the lazy load and generation. The lazy load is the sharper edge:
# two threads hitting `if _pipeline is None` together would each pull ~7 GB.
_PIPELINE_LOCK = threading.Lock()


def get_pipeline():
    """Lazy load and return the global pipeline instance."""
    global _pipeline
    with _PIPELINE_LOCK:
        if _pipeline is None:
            print("Loading SDXL pipeline (first time only)...")
            _pipeline = AutoPipelineForText2Image.from_pretrained(...)
            ...
        return _pipeline


def generate_image(prompt, negative_prompt="", num_inference_steps=25, guidance_scale=12):
    ...
    pipeline = get_pipeline()          # takes and releases the lock
    with _PIPELINE_LOCK:               # re-take for the actual inference
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, \
            negative_pooled_prompt_embeds = encode_prompt_xl(
                pipeline, prompt, negative_prompt
            )
        image = pipeline(
            prompt_embeds=prompt_embeds,
            ...
        ).images[0]
    print("Image generation completed!")
    ...  # os.makedirs / save unchanged, outside the lock

# NOTE: threading.Lock is re-entrant-unsafe, hence get_pipeline() releasing
# before generate_image re-takes. Do NOT nest them.

### `bot_commands.py:1-29 (imports), 105-108 (__init__), 390-410 (voice_slash), 412-423 (gen_slash), 425-442 (lore_slash)`

Import run_blocking and the two concurrency knobs; create per-cog asyncio semaphores in __init__; wrap all four blocking calls; add the missing failure path in voice_slash so a deferred interaction is always answered.

-from bot_adapters import split_into_chunks
+from bot_adapters import run_blocking, split_into_chunks
 from fritz_utils import (
     FAST_OLLAMA_MODEL,
     FFMPEG_PATH,
+    IMAGE_GEN_MAX_CONCURRENCY,
     MessageSource,
+    TTS_MAX_CONCURRENCY,
     THINKING_OLLAMA_MODEL,
     __version__,
 )
+import asyncio


 class FritzCommands(commands.Cog):
     def __init__(self, bot, sayer: TTSEngine, schedule_manager=None):
         self.bot = bot
         self.sayer = sayer
         self.schedule_manager = schedule_manager
+        # Admission control for the GPU-bound commands. Waiters park on the
+        # event loop, NOT on a pool thread, so a queue of /gen requests can
+        # never starve the shared blocking pool.
+        # Instance-level (not module-level) on purpose: asyncio primitives bind
+        # to the first loop that awaits them, and IsolatedAsyncioTestCase gives
+        # every test its own loop.
+        self._image_semaphore = asyncio.Semaphore(IMAGE_GEN_MAX_CONCURRENCY)
+        self._tts_semaphore = asyncio.Semaphore(TTS_MAX_CONCURRENCY)

     async def voice_slash(self, interaction: discord.Interaction, message: str):
         await interaction.response.defer(thinking=True)
-        METRICS.increment("discord_commands.voice")
-        original_response = ask_stuff(message, MessageSource.DISCORD_VOICE, interaction.user.name)["text"]
-        output_file = await self.sayer.generate_speech(original_response)
+        try:
+            with METRICS.time_block("discord_commands.voice"):
+                response_data = await run_blocking(
+                    ask_stuff, message, MessageSource.DISCORD_VOICE, interaction.user.name,
+                )
+                original_response = response_data["text"]
+                async with self._tts_semaphore:
+                    output_file = await run_blocking(
+                        self.sayer.generate_speech, original_response,
+                    )
+        except Exception as e:
+            logger.exception("Voice synthesis failed for %s", interaction.user.name)
+            await interaction.followup.send(f"Failed to generate speech: {e}")
+            return
         try:
             if interaction.guild and interaction.guild.voice_client:
                 ...unchanged...

     async def gen_slash(self, interaction: discord.Interaction, prompt: str):
         await interaction.response.defer(thinking=True)
-        METRICS.increment("discord_commands.gen")
         logger.info("Image generation request: %s", prompt)
         try:
-            output_file = generate_image(prompt)
+            if self._image_semaphore.locked():
+                await interaction.followup.send(
+                    "\U0001f5bc️ Queued — another image is rendering."
+                )
+            with METRICS.time_block("discord_commands.gen"):
+                async with self._image_semaphore:
+                    output_file = await run_blocking(generate_image, prompt)
             await interaction.followup.send(content="Here is your file!", file=discord.File(output_file))
         except Exception as e:
             METRICS.record_error("discord_commands.gen", e)
             await interaction.followup.send(f"Failed to generate image: {e}")

     async def lore_slash(self, interaction: discord.Interaction, query: str):
         await interaction.response.defer(thinking=True)
-        METRICS.increment("discord_commands.lore")
         logger.info("Lore request: %s", query)
-        original_response = query_documents(query)
+        # First call also triggers document_engine.initialize_vectorstore()
+        # (document_engine.py:739-742) — a full corpus ingest. Off-loop now.
+        with METRICS.time_block("discord_commands.lore"):
+            original_response = await run_blocking(query_documents, query)
         author = interaction.user.name
         ...chunking unchanged...

# METRICS.time_block(name) increments the same counter name the old
# METRICS.increment(name) did (observability.py:142), so /health keeps its
# existing counters and gains an average-latency line for each command.

### `main_discord.py:11 (import), 112 (TTSEngine load), 193-203 (ask_stuff), 243-246 (speech_to_text)`

Route on_message and startup work through the same bounded pool so there is one capacity knob instead of an unbounded default executor silently shared with discord.py internals.

-from bot_adapters import split_into_chunks  # noqa: F401 — re-exported for tests
+from bot_adapters import run_blocking, split_into_chunks  # noqa: F401 — re-exported for tests

     if sayer is None:
         logger.info("Loading TTS engine...")
-        sayer = await loop.run_in_executor(None, TTSEngine)
+        sayer = await run_blocking(TTSEngine)

         start_time = time.time()
-        response_data = await loop.run_in_executor(
-            None,
-            lambda: ask_stuff(
-                message_clean, source, author,
-                progress_callback, streaming_callback,
-                user_image_paths,
-                workspace_store.get(author),
-                ctx.channel.id,
-                schedule_manager,
-            )
-        )
+        response_data = await run_blocking(
+            ask_stuff,
+            message_clean, source, author,
+            progress_callback, streaming_callback,
+            user_image_paths,
+            workspace_store.get(author),
+            ctx.channel.id,
+            schedule_manager,
+        )

 async def speech_to_text(file_path: str) -> str | None:
     """Thin async wrapper around the Whisper STT module."""
-    loop = asyncio.get_running_loop()
-    return await loop.run_in_executor(None, _whisper_transcribe, file_path)
+    return await run_blocking(_whisper_transcribe, file_path)

# `loop` is still needed at main_discord.py:150 for run_coroutine_threadsafe in
# streaming_callback/progress_callback (:185-189) — do not delete that binding.

### `tests/test_bot_commands.py:37-46 (_fake_interaction), plus new test classes at end of file`

_fake_interaction must gain `response.defer`, `followup.send`, and `channel.send` AsyncMocks — the three commands under test all call `await interaction.response.defer(...)`, which today would raise TypeError against the bare MagicMock. Then add the new offload/serialisation tests.

def _fake_interaction(username: str) -> MagicMock:
     interaction = MagicMock()
     interaction.user = MagicMock()
     interaction.user.name = username
     interaction.channel_id = 12345
     interaction.guild_id = 67890
     interaction.response = MagicMock()
     interaction.response.send_message = AsyncMock()
+    # /voice, /gen, /lore defer first and answer via followup.
+    interaction.response.defer = AsyncMock()
+    interaction.followup = MagicMock()
+    interaction.followup.send = AsyncMock()
+    interaction.channel = MagicMock()
+    interaction.channel.send = AsyncMock()
+    # Force the "not in a voice channel" branch of voice_slash.
+    interaction.guild = None
     return interaction

### `.env.example:insert after 57 (MEMORY_EXTRACT_MIN_REPLY_CHARS), inside the '----- Tunables -----' block`

Document all three new knobs, commented-out with defaults shown, matching the file's existing style.

# Size of the shared thread pool the bot uses for blocking work (agent runs,
# image generation, TTS, Whisper, RAG). Bounded on purpose — Ollama and the
# GPU serialise anyway. Raise if /health shows ask_stuff latency climbing
# while Ollama itself is idle.
# BLOCKING_POOL_SIZE=8
# Max concurrent Stable Diffusion XL generations. Keep at 1 on a single GPU;
# a second concurrent render thrashes VRAM instead of finishing sooner.
# IMAGE_GEN_MAX_CONCURRENCY=1
# Max concurrent Coqui XTTS syntheses. Same reasoning.
# TTS_MAX_CONCURRENCY=1

### `CHANGELOG.md:insert at 45 (after the Phase 14 block, before '### Added' at 46)`

Add a Phase 15 bullet under the existing ### Performance section, matching the Phase 10-14 nested-bullet style.

- **Phase 15 — Discord event loop.**
  - `/voice`, `/gen`, and `/lore` no longer run their blocking work on the Discord event loop. `ask_stuff`, `TTSEngine.generate_speech`, `image_generator.generate_image`, and `document_engine.query_documents` are all dispatched through a new `bot_adapters.run_blocking()` helper. Previously a single `/gen` froze heartbeats, streaming edits, and every other user's command for the length of an SDXL render.
  - New shared bounded `ThreadPoolExecutor` (`BLOCKING_POOL_SIZE`, default 8) replaces the loop's unbounded default executor for `on_message`, `speech_to_text`, and the startup TTS load, so worker capacity is now a single explicit knob.
  - `TTSEngine.generate_speech` is no longer `async def` — it never awaited anything. As a side effect the `python tts.py "..."` CLI works for the first time; it previously printed a coroutine object and synthesised nothing.
  - New `IMAGE_GEN_MAX_CONCURRENCY` (default 1) and `TTS_MAX_CONCURRENCY` (default 1) gate the GPU-bound commands with an `asyncio.Semaphore`, and `image_generator` grew a module-level lock so the lazy 7 GB SDXL pipeline load can no longer be raced by the slash command and the agent tool at once.
  - `/voice`, `/gen`, and `/lore` now record latency via `METRICS.time_block`, so `/health` shows per-command averages.

## Steps

1. Step 1 — config knobs. Add `BLOCKING_POOL_SIZE`, `IMAGE_GEN_MAX_CONCURRENCY`, `TTS_MAX_CONCURRENCY` to `fritz_utils.py` after line 119, and document all three in `.env.example` after line 57. Commit alone: no behaviour change, nothing reads them yet.
2. Step 2 — the helper. Add `_BLOCKING_POOL` and `async def run_blocking(func, *args, **kwargs)` to `bot_adapters.py` (imports `asyncio`, `functools`, `ThreadPoolExecutor`, and `BLOCKING_POOL_SIZE` from `fritz_utils`). Add `tests/test_bot_adapters.py` covering it. Commit — still nothing calls it.
3. Step 3 — make TTS honest. Drop `async` from `tts.py:64`, add `self._synth_lock = threading.Lock()` in `__init__` and hold it around both `tts_to_file` calls (:98-113). Leave `tts.py:143` untouched. This step alone breaks `bot_commands.py:396` (`await` on a non-awaitable), so it must be committed together with step 4 or the branch is briefly red — prefer squashing 3 and 4.
4. Step 4 — offload the cog. In `bot_commands.py`: add `import asyncio`, switch to `from bot_adapters import run_blocking, split_into_chunks`, import `IMAGE_GEN_MAX_CONCURRENCY` and `TTS_MAX_CONCURRENCY` from `fritz_utils`, create `self._image_semaphore` / `self._tts_semaphore` in `__init__` (:105-108), and rewrite the four call sites at :395, :396, :419, :431 per the sketch. Include the new try/except in `voice_slash` so a deferred interaction is always answered.
5. Step 5 — protect the GPU pipeline. Add `_PIPELINE_LOCK = threading.Lock()` to `image_generator.py` next to `_pipeline` (:18); take it inside `get_pipeline()` (:20-54) and re-take it around `encode_prompt_xl` + the `pipeline(...)` call in `generate_image` (:188-200). Keep `os.makedirs` / `image.save` outside the lock. Do not nest the two acquisitions — `threading.Lock` is not re-entrant.
6. Step 6 — one pool for everything. In `main_discord.py`, change the import at :11 and replace `run_in_executor(None, ...)` at :112, :193-203, and :246 with `run_blocking(...)`. Keep the `loop = asyncio.get_running_loop()` binding at :150 — `streaming_callback`/`progress_callback` at :185-189 still need it for `run_coroutine_threadsafe`.
7. Step 7 — tests. Extend `_fake_interaction` in `tests/test_bot_commands.py:37-46` with `defer`, `followup.send`, `channel.send`, and `guild = None`, then add the four new test classes listed in the test plan. Add `TestRunBlocking` to `tests/test_bot_adapters.py`.
8. Step 8 — observability (small, keeps the fix measurable). Replace `METRICS.increment("discord_commands.voice"|".gen"|".lore")` with `with METRICS.time_block(...)` blocks. `time_block` increments the identical counter name (`observability.py:142`) so nothing in `/health` regresses; it adds an average-latency line per command.
9. Step 9 — docs. Add the Phase 15 bullet to `CHANGELOG.md` at line 45. Run `ruff check .` and the full `pytest tests/ --cov=. --cov-fail-under=60`.

## Config and env changes

- fritz_utils.BLOCKING_POOL_SIZE — env `BLOCKING_POOL_SIZE`, default "8". Worker count for the shared blocking pool used by bot_adapters.run_blocking.
- fritz_utils.IMAGE_GEN_MAX_CONCURRENCY — env `IMAGE_GEN_MAX_CONCURRENCY`, default "1". Concurrent SDXL renders permitted by the /gen semaphore. Must be >= 1; 0 would deadlock the command.
- fritz_utils.TTS_MAX_CONCURRENCY — env `TTS_MAX_CONCURRENCY`, default "1". Concurrent XTTS syntheses permitted by the /voice semaphore. Must be >= 1.
- All three added to .env.example under the existing '----- Tunables (defaults shown; uncomment to override) -----' section, commented out.
- No Dockerfile / docker-compose.yml / infra/k8s configmap change is required — the defaults are correct for the containerised deployment. Optionally add BLOCKING_POOL_SIZE to infra/k8s/configmap.yaml later if the deployment is scaled.

## Tests
### New

- tests/test_bot_adapters.py (new file) — `TestRunBlocking.test_runs_callable_off_the_event_loop`: call `await run_blocking(lambda: threading.get_ident())` and assert the returned ident differs from `threading.get_ident()` in the test body.
- tests/test_bot_adapters.py — `TestRunBlocking.test_passes_positional_and_keyword_args`: `await run_blocking(lambda a, b=0: a + b, 1, b=2)` == 3 (proves the `functools.partial` wiring, which plain `run_in_executor` would not give for kwargs).
- tests/test_bot_adapters.py — `TestRunBlocking.test_exception_propagates_to_awaiter`: a callable raising `ValueError("boom")` must surface as `ValueError` at the await site, not be swallowed.
- tests/test_bot_adapters.py — `TestBlockingPool.test_pool_is_bounded_by_config`: assert `bot_adapters._BLOCKING_POOL._max_workers == fritz_utils.BLOCKING_POOL_SIZE`. (Touches a private attr; acceptable as a guard that the knob is actually wired.)
- tests/test_bot_commands.py — `TestVoiceSlashOffload.test_ask_stuff_and_tts_run_off_the_event_loop`: patch `bot_commands.ask_stuff` and `cog.sayer.generate_speech` with fakes that record `threading.get_ident()`; assert both idents differ from the test thread's and that `interaction.followup.send` was awaited with a `files=` kwarg (guild=None branch). Patch `bot_commands.discord.File`.
- tests/test_bot_commands.py — `TestVoiceSlashOffload.test_failure_answers_the_deferred_interaction`: make `bot_commands.ask_stuff` raise `RuntimeError("ollama down")`; assert `interaction.followup.send` was called once with text containing "Failed to generate speech". Guards the previously-missing error path that left users staring at "The application did not respond".
- tests/test_bot_commands.py — `TestGenSlashOffload.test_generate_image_runs_in_worker_thread`: patch `bot_commands.generate_image` (already a MagicMock attribute thanks to the `_ensure_mock("image_generator")` stub at test_bot_commands.py:25) and `bot_commands.discord.File`; assert off-thread execution and that a followup was sent.
- tests/test_bot_commands.py — `TestGenSlashOffload.test_two_concurrent_gens_are_serialised`: fake `generate_image` increments/decrements a counter under a `threading.Lock` with a 50 ms `time.sleep`; drive two `cog.gen_slash.callback(...)` coroutines through `asyncio.gather` and assert peak concurrency == 1. This is the test that proves the GPU-thrash guard.
- tests/test_bot_commands.py — `TestGenSlashOffload.test_failure_reports_to_user_and_records_metric`: `generate_image` raises; assert `interaction.followup.send` text contains "Failed to generate image".
- tests/test_bot_commands.py — `TestLoreSlashOffload.test_query_documents_runs_in_worker_thread`: patch `bot_commands.query_documents` (module-level name bound by the `from document_engine import query_documents` at bot_commands.py:13) and assert off-thread execution.
- tests/test_bot_commands.py — `TestLoreSlashOffload.test_long_answer_still_chunks_across_messages`: `query_documents` returns 5000 chars; assert `interaction.followup.send` called once and `interaction.channel.send` called for the remaining chunks — regression guard on the branch at bot_commands.py:433-440 that the rewrite touches.
- tests/test_bot_commands.py — `TestCogSemaphoreWiring.test_semaphores_sized_from_config`: patch `fritz_utils.IMAGE_GEN_MAX_CONCURRENCY`/`TTS_MAX_CONCURRENCY` is NOT sufficient (bot_commands imports the values by name at import time), so instead assert `cog._image_semaphore._value == bot_commands.IMAGE_GEN_MAX_CONCURRENCY`. Note this asymmetry in the test comment.

### Existing tests affected

- tests/test_bot_commands.py::_fake_interaction (lines 37-46) — MUST be extended with `response.defer`, `followup.send`, `channel.send` AsyncMocks and `guild = None`. Without this every new test raises `TypeError: object MagicMock can't be used in 'await' expression`. The 11 existing tests only assert on `interaction.response.send_message`, so the extension does not affect them.
- tests/test_bot_commands.py::_make_cog (lines 49-54) — no signature change needed, but note that `FritzCommands.__init__` now constructs two `asyncio.Semaphore` objects. Every existing `_make_cog()` call site (lines 97, 112, 130, 140, 152, 164, 179, 191, 204, 212, 225) is inside an `IsolatedAsyncioTestCase` async test, so each gets a fresh cog on its own loop. Confirmed safe. Do NOT hoist the semaphores to module level — that would bind them to whichever test's loop ran first and raise `RuntimeError: ... is bound to a different event loop` in every subsequent test.
- tests/test_bot_commands.py — the following 11 tests must still pass unchanged: TestRequireAdmin::{test_root_user_allowed, test_admin_user_from_list_allowed, test_non_admin_user_rejected_with_ephemeral}, TestScheduleAddOpenToAll::{test_non_admin_can_schedule_add, test_value_error_surfaces_as_user_message}, TestScheduleRemoveOpenToAll::{test_non_admin_can_remove_own_schedule, test_permission_error_surfaces_to_caller}, TestScheduleListOpenToAll::test_non_admin_can_list_their_own_schedules, TestScheduleListAllAdminOnly::{test_non_admin_blocked_from_list_all, test_admin_can_view_all_schedules}, TestWorkspaceEnableOpenToAll::*, TestWorkspaceSetAdminOnly::test_non_admin_blocked_from_workspace_set. None of them touch the edited commands.
- tests/test_discord_commands.py:34 — `from main_discord import split_into_chunks, StreamingMessageHandler`. This import survives because main_discord.py:11 keeps re-exporting `split_into_chunks` (add `run_blocking` to the same line, keep the `# noqa: F401`). If you drop the re-export, all 6 TestSplitIntoChunks tests and all 6 TestStreamingMessageHandler tests fail at import.
- tests/test_discord_commands.py::TestStreamingMessageHandler (lines 78-128) — unaffected: `StreamingMessageHandler` is not touched, and the `loop` binding at main_discord.py:150 that feeds it is retained.
- tests/test_document_engine.py:57-58 — evicts a MagicMock `document_engine` from sys.modules before importing the real module. Unaffected, because `bot_commands.py:13` keeps the `from document_engine import query_documents` form (the name is bound once at import). Do NOT switch bot_commands to `document_engine.query_documents(...)` attribute access — it would change what `patch("bot_commands.query_documents")` targets and add coupling to test collection order.
- tests/test_agent_tools.py, tests/test_admin_panel.py, tests/test_observability.py — unaffected. No test in the suite imports the real `tts` module (both test_bot_commands.py:27 and test_discord_commands.py:27-28 install sys.modules stubs), so the `async def` → `def` change on generate_speech breaks nothing that exists today.

### Manual verification

- Responsiveness, the actual point: start the bot, fire `/gen a cathedral made of glass` from one account, then immediately `/hello` and `/health` from another. Both must reply within a second while SDXL renders. Before this change they hang for the full render.
- Cold-corpus `/lore`: delete `chroma_store/`, restart, run `/lore <anything>` (this triggers `initialize_vectorstore()` at document_engine.py:742), and DM the bot at the same time. The DM must be answered while ingestion runs.
- GPU serialisation: fire `/gen` twice within a second from two accounts. Expect the second to get the '🖼️ Queued' followup, and `nvidia-smi` to show one SDXL process's VRAM footprint, not two. Both images must eventually arrive.
- TTS CLI regression proof: `python tts.py "testing one two three"` must print `File generated at: output/speech_<hash>.wav` and that file must exist and be audible. On master it prints a coroutine repr, writes nothing, and emits `RuntimeWarning: coroutine 'TTSEngine.generate_speech' was never awaited`.
- `/voice hello there` in a voice channel and out of one — verify both the `voice_client.play(...)` branch and the file-upload branch at bot_commands.py:398-407 still work after the restructure.
- `/health` must now show `Latency discord_commands.gen`, `Latency discord_commands.voice`, `Latency discord_commands.lore` lines (from `METRICS.time_block`) alongside the existing counters, and `Discord messages` must be unchanged.
- Pool sizing sanity: set `BLOCKING_POOL_SIZE=2`, fire three DMs at once, confirm the third is served (queued, not dropped) and that raising the knob back to 8 removes the queueing delay.

## Risks

- asyncio.Semaphore bound to the wrong loop. `asyncio.Semaphore` binds to the first loop that awaits it. Storing them on the cog instance (created once in `main_discord.py:124` inside `on_ready`, and freshly per-test by `_make_cog()`) is correct. If anyone hoists them to `bot_commands` module scope, every test after the first fails with `RuntimeError: ... is bound to a different event loop` and, in production, a bot reconnect that rebuilds the loop would wedge /gen permanently. Detect: the new gather-based concurrency test fails immediately. Mitigation: the `__init__` placement plus the explanatory comment in the sketch.
- Bounded pool is a behaviour change for on_message. Master uses the loop's default executor (~20 workers on an 8-core box); this drops to 8. Under a burst of 9+ simultaneous DMs the 9th queues. This is intentional (Ollama serialises anyway) but it is a real regression vector for a busy server. Detect: `/health` shows `Latency ask_stuff` rising while Ollama's own logs show idle gaps. Mitigation: raise `BLOCKING_POOL_SIZE`; the knob exists precisely for this.
- threading.Lock in image_generator pins pool threads. A thread blocked on `_PIPELINE_LOCK` occupies a worker. The asyncio semaphore keeps /gen out of the pool while queued, but the agent-tool path (`agent_tools.py:306`, reached via ask_stuff in a worker) bypasses the semaphore, so up to `BLOCKING_POOL_SIZE` threads could stack on the lock during a long first-time 7 GB load. Bounded and self-clearing, but it means DMs stall behind a cold SDXL load. Detect: several minutes of `Latency ask_stuff` spike right after the first-ever image request. Deferred mitigation: prewarm SDXL in `prewarm.py` alongside the Ollama models.
- threading.Lock is not re-entrant — if someone later calls `get_pipeline()` from inside the `with _PIPELINE_LOCK:` block in `generate_image`, the process deadlocks with no traceback. Detect: /gen hangs forever with no log line after 'Loading SDXL pipeline'. Mitigation: the sketch deliberately acquires-releases in `get_pipeline()` then re-acquires; add a comment. Consider `threading.RLock` if this proves fragile — `document_engine.py:55` already uses `RLock` for exactly this reason.
- Discord interaction expiry under queueing. `defer(thinking=True)` buys 15 minutes. With `IMAGE_GEN_MAX_CONCURRENCY=1` and ~60 s renders, the 15th queued /gen's followup fails with `404 Unknown Webhook`. Detect: `discord.errors.NotFound` in the logs from `gen_slash`'s followup. Mitigation for now: the '🖼️ Queued' notice sets expectations. A hard queue-depth reject is deferred (see openQuestions).
- Making generate_speech sync is a public signature change on TTSEngine. Verified there are exactly two callers repo-wide (`bot_commands.py:396`, `tts.py:143`) and no test imports the real module — but any out-of-tree script doing `await engine.generate_speech(...)` breaks with `TypeError: object str can't be used in 'await' expression`. Low risk, and the alternative (leaving a lying `async def`) is worse.
- ThreadPoolExecutor and interpreter shutdown: `concurrent.futures` installs an atexit hook that joins non-daemon workers, so Ctrl-C during a 60 s render waits for it. Identical to master's default-executor behaviour, so not a regression, but worth knowing when the bot seems slow to die.
- Cannot be settled statically: whether this specific GPU actually degrades on two concurrent SDXL renders, or whether Coqui XTTS `tts_to_file` is re-entrant on one TTS object. See openQuestions for the experiments that settle both.

## Rollback
"Straight `git revert` of the squashed commit. The change is additive and self-contained across `bot_adapters.py`, `fritz_utils.py`, `tts.py`, `image_generator.py`, `bot_commands.py`, `main_discord.py`, `.env.example`, `CHANGELOG.md`, and two test files — no schema, no persisted state, no wire format, nothing to migrate. A feature flag is not warranted: the whole point is that the old path is a bug.\n\nPartial escape hatches without a revert, if the change is directionally right but mis-tuned:\n  - Pool feels too small (DMs queueing): raise `BLOCKING_POOL_SIZE` to 32 — that approximates master's unbounded default-executor behaviour without touching code.\n  - GPU is actually fine with parallel renders: raise `IMAGE_GEN_MAX_CONCURRENCY` / `TTS_MAX_CONCURRENCY`. Never set either to 0 — the semaphore would block forever and the command would hang until the interaction expires.\n\nIf only the TTS signature change proves problematic (an out-of-tree caller awaits it), the narrow fix is to keep `def generate_speech` and add a thin `async def generate_speech_async(self, *a, **kw): return await run_blocking(self.generate_speech, *a, **kw)` shim rather than reverting the whole commit."

## Open questions for you to decide

- Does this GPU actually degrade on two concurrent SDXL renders? Not answerable by reading code. Experiment: set `IMAGE_GEN_MAX_CONCURRENCY=2`, fire two /gen simultaneously, watch `nvidia-smi -l 1` for VRAM and the per-image wall time in the logs. If total throughput improves and VRAM headroom holds, 2 is a better default for this host. The `image_generator._PIPELINE_LOCK` still serialises the CUDA graph regardless, so raising the semaphore above 1 only helps if you ALSO narrow that lock to cover just `get_pipeline()` — decide together, not separately.
- Is Coqui XTTS `tts_to_file` re-entrant on a single TTS object? Experiment: call `TTSEngine.generate_speech` from two threads on the same instance with `force_regenerate=True` and distinct text, and diff the two WAVs against sequentially-generated references. If it is re-entrant, `self._synth_lock` can be dropped. The lock costs nothing today (`TTS_MAX_CONCURRENCY=1` already serialises the only caller), so keeping it is the safe default.
- Is 8 the right `BLOCKING_POOL_SIZE` default? It is a judgement call: Ollama's own `OLLAMA_NUM_PARALLEL` (not currently set anywhere in this repo) is the real ceiling on useful concurrency. If the deployment sets it, the pool should probably match it plus a couple of slots for Whisper/RAG. Worth a look at the target host before shipping.
- Should a queued /gen be rejected rather than queued past some depth? With `IMAGE_GEN_MAX_CONCURRENCY=1` the 15-minute interaction window caps out around 14 queued requests. A `if self._image_semaphore._value == 0 and depth > N: reject` guard is ~6 lines but needs a depth counter. Deferred as out of scope for 'get work off the loop' — flag it if this bot serves more than a handful of users.
- Should `scheduler.py:109`, `main_telegram.py:26/45/52`, and `admin_panel.py:388/457/630/642` also move onto the shared pool? They already offload correctly, so they are not broken — but they still use the unbounded default executor, which means the bounded-pool guarantee is only partial: a scheduled task and a web-chat message can still both hit Ollama outside the budget. DELIBERATELY DEFERRED here to keep this change reviewable. Worth a follow-up item; note `admin_panel.py:554-558` uses a plain daemon thread on purpose (TestClient loop teardown) and must NOT be converted.
- Unrelated bug spotted while verifying: `bot_commands.py:367` reads `snap.get("uptime_seconds", 0)` but `observability.get_health_snapshot()` returns the key as `uptime_sec` (observability.py:304), so `/about` always reports an uptime of 0s. Not touched here — belongs to the `discord-polish` item.
