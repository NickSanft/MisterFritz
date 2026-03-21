"""
Tests for stt.py (faster-whisper speech-to-text).

All heavy dependencies (faster_whisper, pydub) are mocked so no model
download or audio file is required to run these tests.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_faster_whisper_stub(transcription: str = "hello world"):
    """
    Return a (stub_module, mock_model, MockWhisperModel) triple.

    faster-whisper API:
        model = WhisperModel(name, device=..., compute_type=...)
        segments, info = model.transcribe(path, ...)
        text = " ".join(seg.text for seg in segments)
    """
    stub = types.ModuleType("faster_whisper")

    mock_segment = MagicMock()
    mock_segment.text = transcription

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    MockWhisperModel = MagicMock(return_value=mock_model)
    stub.WhisperModel = MockWhisperModel

    return stub, mock_model, MockWhisperModel


def _make_audio_mock(dbfs: float = -20.0):
    """Return a pydub AudioSegment mock that supports the full processing chain."""
    audio = MagicMock()
    audio.dBFS = dbfs
    audio.set_frame_rate.return_value = audio
    audio.set_channels.return_value = audio
    audio.apply_gain.return_value = audio
    audio.__getitem__ = MagicMock(return_value=audio)  # audio[start:end]
    return audio


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestTranscribe(unittest.TestCase):

    def setUp(self):
        import stt
        stt._whisper_model = None

    def tearDown(self):
        import stt
        stt._whisper_model = None

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_returns_transcribed_text(self):
        """Happy path: audio file → transcribed string."""
        stub, _, _ = _make_faster_whisper_stub("good evening")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertEqual(result, "good evening")

    def test_joins_multiple_segments(self):
        """Multiple segments are joined with spaces into a single string."""
        stub = types.ModuleType("faster_whisper")
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = "Hello"
        seg2.text = "world"
        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], info)
        stub.WhisperModel = MagicMock(return_value=mock_model)
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertEqual(result, "Hello world")

    def test_strips_whitespace_from_result(self):
        """Leading/trailing whitespace in Whisper output is stripped."""
        stub, _, _ = _make_faster_whisper_stub("  hello there  ")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertEqual(result, "hello there")

    # ------------------------------------------------------------------
    # Failure / edge cases
    # ------------------------------------------------------------------

    def test_returns_none_on_model_load_failure(self):
        """WhisperModel() raising → returns None, does not propagate."""
        stub = types.ModuleType("faster_whisper")
        stub.WhisperModel = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertIsNone(result)

    def test_returns_none_on_transcribe_failure(self):
        """model.transcribe() raising → returns None, does not propagate."""
        stub = types.ModuleType("faster_whisper")
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("inference error")
        stub.WhisperModel = MagicMock(return_value=mock_model)
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertIsNone(result)

    def test_returns_none_on_empty_transcription(self):
        """Empty transcription result → returns None."""
        stub, _, _ = _make_faster_whisper_stub("")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertIsNone(result)

    def test_returns_none_on_whitespace_only_transcription(self):
        """Whitespace-only transcription → returns None."""
        stub, _, _ = _make_faster_whisper_stub("   ")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            result = stt.transcribe("audio.ogg")

        self.assertIsNone(result)

    def test_returns_none_on_audio_load_failure(self):
        """AudioSegment.from_file() raising → returns None, does not propagate."""
        stub, _, _ = _make_faster_whisper_stub("ignored")

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", side_effect=Exception("bad file")):
            import stt
            result = stt.transcribe("not_a_real_file.ogg")

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def test_model_loaded_only_once(self):
        """WhisperModel must be constructed exactly once across multiple calls."""
        stub, _, MockWhisperModel = _make_faster_whisper_stub("hi")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            stt.transcribe("a.ogg")
            stt.transcribe("b.ogg")
            stt.transcribe("c.ogg")

        MockWhisperModel.assert_called_once()

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    def test_audio_resampled_to_16khz_mono(self):
        """Audio is converted to 16 kHz mono before preprocessing."""
        stub, _, _ = _make_faster_whisper_stub("test")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            stt.transcribe("audio.ogg")

        audio.set_frame_rate.assert_called_once_with(16000)
        audio.set_channels.assert_called_once_with(1)

    def test_normalisation_applied_to_non_silent_audio(self):
        """apply_gain must be called when audio is not completely silent."""
        stub, _, _ = _make_faster_whisper_stub("test")
        audio = _make_audio_mock(dbfs=-35.0)  # quiet audio

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            stt.transcribe("audio.ogg")

        audio.apply_gain.assert_called_once()

    def test_normalisation_skipped_for_silent_audio(self):
        """apply_gain must NOT be called when dBFS is -inf (completely silent)."""
        stub, _, _ = _make_faster_whisper_stub("")
        audio = _make_audio_mock(dbfs=float("-inf"))

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[]):
            import stt
            stt.transcribe("audio.ogg")

        audio.apply_gain.assert_not_called()

    def test_silence_trim_applied_when_nonsilent_regions_found(self):
        """Audio is sliced to the nonsilent region when detect_nonsilent returns data."""
        stub, _, _ = _make_faster_whisper_stub("trimmed")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[200, 800]]):
            import stt
            stt.transcribe("audio.ogg")

        audio.__getitem__.assert_called_once()

    def test_silence_trim_skipped_when_all_silent(self):
        """Audio is not sliced when detect_nonsilent returns empty list."""
        stub, _, _ = _make_faster_whisper_stub("")
        audio = _make_audio_mock(dbfs=float("-inf"))

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[]):
            import stt
            stt.transcribe("audio.ogg")

        audio.__getitem__.assert_not_called()

    # ------------------------------------------------------------------
    # Temp file cleanup
    # ------------------------------------------------------------------

    def test_temp_wav_deleted_on_success(self):
        """Temporary WAV file is removed after a successful transcription."""
        stub, _, _ = _make_faster_whisper_stub("cleanup test")
        audio = _make_audio_mock()
        created = []

        def fake_export(path, format):
            open(path, "w").close()
            created.append(path)

        audio.export = fake_export

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            stt.transcribe("audio.ogg")

        for p in created:
            self.assertFalse(os.path.exists(p), f"Temp file not cleaned up: {p}")

    def test_temp_wav_deleted_on_failure(self):
        """Temporary WAV is removed even when Whisper raises."""
        stub = types.ModuleType("faster_whisper")
        stub.WhisperModel = MagicMock(side_effect=Exception("fail"))
        audio = _make_audio_mock()
        created = []

        def fake_export(path, format):
            open(path, "w").close()
            created.append(path)

        audio.export = fake_export

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]):
            import stt
            stt.transcribe("audio.ogg")

        for p in created:
            self.assertFalse(os.path.exists(p), f"Temp file not cleaned up: {p}")

    # ------------------------------------------------------------------
    # Whisper kwargs
    # ------------------------------------------------------------------

    def test_language_forwarded_when_set(self):
        """WHISPER_LANGUAGE is forwarded to model.transcribe() when set."""
        stub, mock_model, _ = _make_faster_whisper_stub("bonjour")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]), \
             patch("stt.WHISPER_LANGUAGE", "fr"):
            import stt
            stt.transcribe("audio.ogg")

        _, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("language"), "fr")

    def test_language_not_forwarded_when_none(self):
        """language kwarg is absent when WHISPER_LANGUAGE is None."""
        stub, mock_model, _ = _make_faster_whisper_stub("hello")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]), \
             patch("stt.WHISPER_LANGUAGE", None):
            import stt
            stt.transcribe("audio.ogg")

        _, kwargs = mock_model.transcribe.call_args
        self.assertNotIn("language", kwargs)

    def test_beam_size_forwarded(self):
        """WHISPER_BEAM_SIZE is forwarded to model.transcribe()."""
        stub, mock_model, _ = _make_faster_whisper_stub("test")
        audio = _make_audio_mock()

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]), \
             patch("stt.WHISPER_BEAM_SIZE", 8):
            import stt
            stt.transcribe("audio.ogg")

        _, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("beam_size"), 8)

    def test_initial_prompt_forwarded(self):
        """WHISPER_INITIAL_PROMPT is forwarded to model.transcribe()."""
        stub, mock_model, _ = _make_faster_whisper_stub("test")
        audio = _make_audio_mock()
        prompt = "Custom context prompt."

        with patch.dict(sys.modules, {"faster_whisper": stub}), \
             patch("stt.AudioSegment.from_file", return_value=audio), \
             patch("stt.detect_nonsilent", return_value=[[0, 1000]]), \
             patch("stt.WHISPER_INITIAL_PROMPT", prompt):
            import stt
            stt.transcribe("audio.ogg")

        _, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("initial_prompt"), prompt)


if __name__ == "__main__":
    unittest.main()
