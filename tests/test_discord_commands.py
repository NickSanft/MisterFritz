"""
Tests for Discord-specific utilities in main_discord.py.

main_discord.py has module-level side-effects (TTSEngine init, bot creation).
We mock the heavy modules before importing so no Discord connection or TTS
model load is required.
"""
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# tts (prevents a TTS model download), image_generator and document_engine are
# stubbed in tests/conftest.py before any test module is collected.

from main_discord import split_into_chunks, StreamingMessageHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Tests for split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks(unittest.TestCase):
    def test_short_string_not_split(self):
        result = split_into_chunks("hello", 2000)
        self.assertEqual(result, ["hello"])

    def test_exact_boundary_not_split(self):
        s = "x" * 2000
        result = split_into_chunks(s, 2000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], s)

    def test_long_string_split_correctly(self):
        s = "a" * 5000
        result = split_into_chunks(s, 2000)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2000)
        self.assertEqual(len(result[1]), 2000)
        self.assertEqual(len(result[2]), 1000)

    def test_chunks_reassemble_to_original(self):
        original = "hello world " * 300  # 3600 chars
        chunks = split_into_chunks(original, 2000)
        self.assertEqual("".join(chunks), original)

    def test_empty_string_returns_empty_list(self):
        result = split_into_chunks("", 2000)
        self.assertEqual(result, [])

    def test_custom_chunk_size(self):
        result = split_into_chunks("abcde", 2)
        self.assertEqual(result, ["ab", "cd", "e"])


# ---------------------------------------------------------------------------
# Tests for StreamingMessageHandler
# ---------------------------------------------------------------------------

class TestStreamingMessageHandler(unittest.IsolatedAsyncioTestCase):
    def _make_handler(self, min_interval: float = 0.0):
        mock_message = MagicMock()
        mock_message.edit = AsyncMock()
        mock_message.channel = MagicMock()
        mock_message.channel.send = AsyncMock()
        loop = asyncio.get_event_loop()
        return StreamingMessageHandler(mock_message, loop, min_update_interval=min_interval), mock_message

    async def test_update_text_calls_edit(self):
        handler, msg = self._make_handler()
        await handler.update_text("Hello!")
        msg.edit.assert_called()

    async def test_final_update_short_text(self):
        handler, msg = self._make_handler()
        await handler.final_update("Short response")
        msg.edit.assert_called_with(content="Short response")

    async def test_final_update_long_text_truncated_in_edit(self):
        handler, msg = self._make_handler()
        long_text = "x" * 2500
        await handler.final_update(long_text)
        call_args = msg.edit.call_args
        content = call_args[1].get("content") or (call_args[0][0] if call_args[0] else "")
        self.assertLessEqual(len(content), 2000)

    async def test_final_update_with_files_sends_separately(self):
        handler, msg = self._make_handler()
        fake_file = MagicMock()
        await handler.final_update("Short", files=[fake_file])
        # Files should be sent as a separate channel.send when text is short
        msg.channel.send.assert_called()

    async def test_pending_text_tracks_latest(self):
        handler, msg = self._make_handler(min_interval=0.0)
        # Multiple quick updates — final state should reflect last update
        handler.pending_text = "first"
        handler.pending_text = "second"
        handler.pending_text = "third"
        await handler.update_text("third")
        self.assertEqual(handler.current_text, "third")

    async def test_rate_limiting_respected(self):
        handler, msg = self._make_handler(min_interval=0.05)
        handler.last_update_time = time.time()  # simulate a recent edit
        start = time.time()
        await handler.update_text("Rate limited text")
        elapsed = time.time() - start
        # Should have waited ~0.05s
        self.assertGreaterEqual(elapsed, 0.04)


if __name__ == "__main__":
    unittest.main()
