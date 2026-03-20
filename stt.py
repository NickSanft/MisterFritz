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
