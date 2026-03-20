# Plan: Whisper STT + Multi-Platform Support

## Background & Current State

`main_discord.py` has a `speech_to_text()` function that uses `SpeechRecognition` + Google's cloud API. It is Discord-specific and tightly coupled into `on_message`. The core agent (`mister_fritz.py`) is already platform-agnostic — it accepts a `MessageSource` enum and a plain string. The gap is in the **ingestion layer**: audio→text conversion, and how platform adapters call the agent.

Goals:
1. Replace Google STT with local **OpenAI Whisper** (no cloud dependency, better accuracy, works offline).
2. Extract STT into a standalone, platform-agnostic module (`stt.py`).
3. Refactor the Discord adapter to consume that module.
4. Add a thin **Telegram adapter** (`main_telegram.py`) to prove the multi-platform model works.
5. Extend `fritz_utils.py` with Whisper-related config.
6. Full test coverage for the new STT module.

---

## Architecture After This Plan

```
                    ┌─────────────────────┐
                    │   Platform Adapters  │
                    │  main_discord.py     │  ← updated, uses stt.py
                    │  main_telegram.py    │  ← new
                    └─────────┬───────────┘
                              │ plain str
                    ┌─────────▼───────────┐
                    │      stt.py          │  ← new: Whisper-backed
                    └─────────────────────┘
                              │ plain str
                    ┌─────────▼───────────┐
                    │   mister_fritz.py    │  (unchanged)
                    └─────────────────────┘
```

---

## Step 1 — Add Whisper config to `fritz_utils.py`

**File:** `fritz_utils.py`

Add the following env-var-backed constants below the FFmpeg block:

```python
# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")
# Options: tiny, base, small, medium, large  (larger = slower but more accurate)
# "base" is a good default — ~140 MB, runs on CPU in reasonable time.
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
# Set to "cuda" if a GPU is available.
WHISPER_LANGUAGE: str | None = os.environ.get("WHISPER_LANGUAGE")
# None = auto-detect language. Set e.g. "en" to force English and speed up inference.
```

Also add `TELEGRAM_BOT_TOKEN` alongside `DISCORD_BOT_TOKEN`:

```python
TELEGRAM_BOT_TOKEN: str | None = _env_or_json("TELEGRAM_BOT_TOKEN", "telegram_bot_token")
```

---

## Step 2 — Create `stt.py` (standalone Whisper STT module)

**File:** `stt.py` (new, in project root)

This module owns all audio-to-text logic. It must:
- Be importable with no network call (model load is lazy / cached).
- Accept any audio file that `pydub` can read (ogg, mp3, wav, m4a, …).
- Return `str` on success, `None` on failure (same contract as the current function).
- Log errors via `logging`, record metrics via `METRICS`.

```python
"""stt.py — Whisper-backed speech-to-text for MisterFritz."""

import logging
import os
import tempfile

from pydub import AudioSegment

from fritz_utils import FFMPEG_PATH, FFPROBE_PATH, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL
from observability import METRICS

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

logger = logging.getLogger(__name__)

_whisper_model = None  # lazy-loaded singleton


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper  # imported here so the module loads even if whisper is absent
        logger.info("Loading Whisper model '%s' on device '%s'", WHISPER_MODEL, WHISPER_DEVICE)
        _whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
        logger.info("Whisper model loaded")
    return _whisper_model


def transcribe(audio_file_path: str) -> str | None:
    """
    Convert an audio file to text using local Whisper.

    Accepts any format readable by pydub (ogg, mp3, m4a, wav, …).
    Returns the transcribed string, or None on failure.
    """
    wav_path = None
    try:
        # Convert to 16 kHz mono WAV — what Whisper expects
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        audio.export(wav_path, format="wav")
        logger.info("Converted '%s' → '%s' for Whisper", audio_file_path, wav_path)

        model = _get_model()
        kwargs = {}
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE
        result = model.transcribe(wav_path, **kwargs)
        text = result.get("text", "").strip()
        logger.info("Whisper transcription: %r", text)
        return text or None

    except Exception as exc:
        METRICS.record_error("whisper_stt", exc)
        logger.warning("Whisper STT failed for '%s': %s", audio_file_path, exc)
        return None

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
```

Key design decisions:
- Model is a **lazy singleton** — loaded once on first use, not at import time.
- Intermediate WAV is written to a `tempfile` and deleted in `finally`.
- `from_file()` replaces `from_ogg()` — handles any format pydub supports.

---

## Step 3 — Update `main_discord.py`

**Changes:**

1. Remove `import speech_recognition as sr` and `_recognizer = sr.Recognizer()`.
2. Replace the `speech_to_text()` function with a one-liner that delegates to `stt.transcribe()`:

```python
from stt import transcribe as _whisper_transcribe

async def speech_to_text(file_path: str) -> str | None:
    """Thin async wrapper around the Whisper STT module."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _whisper_transcribe, file_path)
```

3. The `AudioSegment` setup block in `main_discord.py` (lines 24–31) can be removed — `stt.py` now owns it.

No other changes needed — call sites (`on_message`) are unchanged.

---

## Step 4 — Add `MessageSource` values for Telegram

**File:** `fritz_utils.py` — extend the enum:

```python
class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_TEXT_AND_IMAGE = 1,
    DISCORD_VOICE = 2,
    LOCAL = 3,
    TELEGRAM_TEXT = 4,
    TELEGRAM_VOICE = 5,
```

**File:** `mister_fritz.py` — extend `get_source_info()`:

```python
elif source == MessageSource.TELEGRAM_TEXT:
    return f"User is texting from Telegram (User ID: {user_id})"
elif source == MessageSource.TELEGRAM_VOICE:
    return f"User is speaking from Telegram (User ID: {user_id}). Please answer in 30 words or less."
```

---

## Step 5 — Create `main_telegram.py` (Telegram adapter)

**File:** `main_telegram.py` (new, in project root)

Uses `python-telegram-bot>=20` (async, application-builder pattern).

```python
"""main_telegram.py — Telegram adapter for MisterFritz."""

import asyncio
import logging
import os
import tempfile

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from fritz_utils import TELEGRAM_BOT_TOKEN, MessageSource, validate_config
from mister_fritz import ask_stuff
from observability import init_logging, start_metrics_server
from stt import transcribe

init_logging()
logger = logging.getLogger(__name__)


async def handle_text(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text or ""
    await update.message.reply_text("✍️ *Mister Fritz is thinking...*", parse_mode="Markdown")

    loop = asyncio.get_running_loop()
    response_data = await loop.run_in_executor(
        None,
        lambda: ask_stuff(text, MessageSource.TELEGRAM_TEXT, user_id),
    )
    reply = response_data.get("text") or "I appear to have misplaced my thoughts."
    await update.message.reply_text(reply[:4096])


async def handle_voice(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        ogg_path = tmp.name
    try:
        await file.download_to_drive(ogg_path)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, transcribe, ogg_path)
        if not text:
            await update.message.reply_text("I'm afraid I couldn't make heads or tails of that audio.")
            return

        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(text, MessageSource.TELEGRAM_VOICE, user_id),
        )
        reply = response_data.get("text") or "Most peculiar — I have no response."
        await update.message.reply_text(reply[:4096])
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)


def main():
    validate_config()
    start_metrics_server()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Telegram bot starting")
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

## Step 6 — Update `requirements.txt`

Add:

```
openai-whisper>=20231117
python-telegram-bot>=20.0
```

Remove (if not used elsewhere):
```
SpeechRecognition
```

Note: `openai-whisper` pulls in `torch` — first install will be large (~2 GB with CPU torch). Use `--index-url https://download.pytorch.org/whl/cpu` for a CPU-only torch to keep it smaller.

---

## Step 7 — Update `Dockerfile`

The existing Dockerfile installs `ffmpeg` via apt. Add:

```dockerfile
# Whisper needs torch; install CPU-only variant to keep image lean
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install openai-whisper python-telegram-bot
```

Or, simpler — just let `requirements.txt` drive it and the existing `pip install -r requirements.txt` picks it up. Add a `.dockerignore` exclusion for the Whisper model cache if pre-downloading:

```
~/.cache/whisper/
```

Set the Whisper model env var in `docker-compose.yml`:

```yaml
environment:
  - WHISPER_MODEL=base
  - WHISPER_DEVICE=cpu
```

---

## Step 8 — Update `.env.example`

```env
# Whisper STT
WHISPER_MODEL=base          # tiny|base|small|medium|large
WHISPER_DEVICE=cpu          # cpu or cuda
WHISPER_LANGUAGE=           # blank = auto-detect

# Telegram (optional)
TELEGRAM_BOT_TOKEN=
```

---

## Step 9 — Tests

**File:** `tests/test_stt.py` (new)

Test strategy: mock both `whisper` and `pydub` — no real audio or model needed.

```python
"""Tests for stt.py (Whisper speech-to-text)."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub heavy deps before importing stt
# ---------------------------------------------------------------------------

def _make_whisper_stub(transcription="hello world"):
    stub = types.ModuleType("whisper")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": transcription}
    stub.load_model = MagicMock(return_value=mock_model)
    return stub, mock_model


@pytest.fixture(autouse=True)
def reset_whisper_singleton():
    """Ensure the lazy singleton is cleared between tests."""
    import stt
    stt._whisper_model = None
    yield
    stt._whisper_model = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscribe:

    def test_returns_transcribed_text(self, tmp_path):
        """Happy path: audio file → text string."""
        whisper_stub, mock_model = _make_whisper_stub("good evening")
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio), \
             patch("stt.AudioSegment.from_file"):
            fake_audio.export = MagicMock()
            import stt
            result = stt.transcribe(str(tmp_path / "audio.ogg"))

        assert result == "good evening"

    def test_returns_none_on_whisper_failure(self, tmp_path):
        """Whisper exception → returns None, does not raise."""
        whisper_stub = types.ModuleType("whisper")
        whisper_stub.load_model = MagicMock(side_effect=RuntimeError("GPU OOM"))

        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio):
            import stt
            result = stt.transcribe(str(tmp_path / "audio.ogg"))

        assert result is None

    def test_returns_none_on_empty_transcription(self, tmp_path):
        """Whisper returning empty string → returns None."""
        whisper_stub, _ = _make_whisper_stub("")
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio):
            import stt
            result = stt.transcribe(str(tmp_path / "audio.ogg"))

        assert result is None

    def test_model_is_loaded_once(self, tmp_path):
        """Whisper model must be a lazy singleton — loaded only once."""
        whisper_stub, mock_model = _make_whisper_stub("hi")
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio):
            import stt
            stt.transcribe(str(tmp_path / "a.ogg"))
            stt.transcribe(str(tmp_path / "b.ogg"))

        whisper_stub.load_model.assert_called_once()

    def test_temp_wav_cleaned_up_on_success(self, tmp_path):
        """Temporary WAV file must be deleted after successful transcription."""
        whisper_stub, _ = _make_whisper_stub("cleanup test")
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        created_paths = []

        def fake_export(path, format):
            created_paths.append(path)
            open(path, "w").close()  # create so os.path.exists → True

        fake_audio.export = fake_export

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio):
            import stt
            stt.transcribe(str(tmp_path / "audio.ogg"))

        for p in created_paths:
            assert not os.path.exists(p), f"Temp file not cleaned up: {p}"

    def test_temp_wav_cleaned_up_on_failure(self, tmp_path):
        """Temporary WAV must be deleted even when Whisper throws."""
        whisper_stub = types.ModuleType("whisper")
        whisper_stub.load_model = MagicMock(side_effect=Exception("fail"))
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        created_paths = []

        def fake_export(path, format):
            created_paths.append(path)
            open(path, "w").close()

        fake_audio.export = fake_export

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio):
            import stt
            stt.transcribe(str(tmp_path / "audio.ogg"))

        for p in created_paths:
            assert not os.path.exists(p)

    def test_language_passed_to_whisper(self, tmp_path):
        """WHISPER_LANGUAGE env var must be forwarded to model.transcribe()."""
        whisper_stub, mock_model = _make_whisper_stub("bonjour")
        fake_audio = MagicMock()
        fake_audio.set_frame_rate.return_value = fake_audio
        fake_audio.set_channels.return_value = fake_audio

        with patch.dict(sys.modules, {"whisper": whisper_stub}), \
             patch("stt.AudioSegment.from_file", return_value=fake_audio), \
             patch("stt.WHISPER_LANGUAGE", "fr"):
            import stt
            stt.transcribe(str(tmp_path / "audio.ogg"))

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "fr"
```

Also update `tests/test_discord_commands.py`:
- Remove any `speech_recognition` mock.
- Add a patch for `stt.transcribe` where the existing `speech_to_text` mock was.

---

## Step 10 — CI / GitHub Actions

Update `.github/workflows/` (or whichever CI file exists) to:

1. Install `openai-whisper` in the test environment (or mock it — since tests mock `whisper`, no real install needed for unit tests).
2. Add `WHISPER_MODEL=base` and `WHISPER_DEVICE=cpu` to env vars alongside existing ones.
3. Keep the 60% coverage gate.

---

## Rollout Order

| Step | File(s) touched | Risk |
|------|-----------------|------|
| 1 | `fritz_utils.py` | None — additive |
| 2 | `stt.py` (new) | None |
| 3 | `main_discord.py` | Low — same function contract |
| 4 | `fritz_utils.py`, `mister_fritz.py` | Low — additive enum values |
| 5 | `main_telegram.py` (new) | None — new file, opt-in |
| 6 | `requirements.txt` | Medium — new heavy dep (torch) |
| 7 | `Dockerfile` | Medium |
| 8 | `.env.example` | None |
| 9 | `tests/test_stt.py` (new) | None |
| 10 | CI config | Low |

Start with Steps 1→9, validate tests pass, then do 6→7→10.

---

## Open Questions / Decisions Needed

1. **Whisper model size**: `base` (~140 MB, ~5 RTF on CPU) vs `small` (~460 MB, better accuracy). For voice messages on Discord/Telegram, `base` is usually sufficient.
2. **GPU support**: If the host machine has a CUDA GPU, set `WHISPER_DEVICE=cuda` — transcription goes from ~5s to ~0.3s per clip. Worth documenting in `.env.example`.
3. **`python-telegram-bot` version**: v20+ is fully async (matches the Discord adapter pattern). v13 is sync-only. Plan assumes v20+.
4. **Streaming TTS for Telegram**: The current `TTSEngine` is Discord-specific (sends voice channel audio). Telegram uses `.ogg` voice messages. Out of scope for this plan — add as a follow-up.
5. **Whisper model pre-download in Docker**: First run downloads the model (~140 MB). For prod containers, consider `RUN python -c "import whisper; whisper.load_model('base')"` in the Dockerfile to bake it into the image.
