"""stt.py — faster-whisper speech-to-text for MisterFritz."""

import logging
import os
import tempfile

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from fritz_utils import (
    FFMPEG_PATH,
    FFPROBE_PATH,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)
from observability import METRICS

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

logger = logging.getLogger(__name__)

_whisper_model = None  # lazy-loaded singleton

_TARGET_DBFS = -20.0      # normalise clips to this loudness level
_SILENCE_THRESH_OFFSET = -16  # dB below clip dBFS → silence threshold
_MIN_SILENCE_MS = 100     # runs of silence shorter than this are kept


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # deferred — heavy import
        logger.info(
            "Loading faster-whisper model '%s' on device '%s' (compute_type='%s')",
            WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("faster-whisper model loaded")
    return _whisper_model


def _preprocess(audio: AudioSegment) -> AudioSegment:
    """Normalise volume and trim leading/trailing silence."""
    # Normalise to a consistent loudness so quiet recordings aren't missed
    if audio.dBFS != float("-inf"):
        audio = audio.apply_gain(_TARGET_DBFS - audio.dBFS)

    # Trim leading/trailing silence so Whisper's VAD isn't confused by padding
    silence_thresh = audio.dBFS + _SILENCE_THRESH_OFFSET
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=_MIN_SILENCE_MS,
        silence_thresh=silence_thresh,
    )
    if nonsilent:
        audio = audio[nonsilent[0][0]:nonsilent[-1][1]]

    return audio


def transcribe(audio_file_path: str) -> str | None:
    """
    Convert an audio file to text using local faster-whisper.

    Accepts any format readable by pydub (ogg, mp3, m4a, wav, …).
    Returns the transcribed string, or None on failure.
    """
    wav_path = None
    try:
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio = _preprocess(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        audio.export(wav_path, format="wav")
        logger.info("Preprocessed '%s' → '%s' for Whisper", audio_file_path, wav_path)

        model = _get_model()
        kwargs = dict(
            beam_size=WHISPER_BEAM_SIZE,
            initial_prompt=WHISPER_INITIAL_PROMPT,
        )
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE

        segments, info = model.transcribe(wav_path, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        logger.info(
            "Whisper transcription (lang=%s, prob=%.2f): %r",
            info.language, info.language_probability, text,
        )
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
