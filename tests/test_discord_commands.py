"""
Tests for Discord-specific utilities in main_discord.py.

main_discord.py has module-level side-effects (TTSEngine init, bot creation).
We mock the heavy modules before importing so no Discord connection or TTS
model load is required.
"""
import asyncio
import pathlib
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


class TestStatusLine(unittest.IsolatedAsyncioTestCase):
    """Progress renders inside the placeholder message. It used to be a
    separate permanent ctx.channel.send per tool notice, which could land
    below the very placeholder it was describing."""

    def _make_handler(self):
        msg = MagicMock()
        msg.edit = AsyncMock()
        msg.channel = MagicMock()
        msg.channel.send = AsyncMock()
        return StreamingMessageHandler(msg, asyncio.get_event_loop(),
                                       min_update_interval=0.0), msg

    async def test_status_before_the_first_token_is_shown(self):
        """The common case for a tool-using turn, and the one that was broken.

        A progress notice arriving before any token had pending_text == "",
        which is falsy — the old guard skipped the edit entirely, so the
        placeholder sat on "Mister Fritz is thinking..." and the notice only
        appeared once tokens started arriving, by which point it was stale.
        The existing test below hides this because it calls update_text first.
        """
        handler, msg = self._make_handler()
        await handler.set_status("🌐 Making enquiries further afield.")
        msg.edit.assert_awaited()
        self.assertIn("Making enquiries further afield.",
                      msg.edit.call_args.kwargs["content"])

    async def test_status_then_tokens_keeps_both(self):
        handler, msg = self._make_handler()
        await handler.set_status("📚 Consulting the library.")
        await handler.update_text("Here is what I found")
        content = msg.edit.call_args.kwargs["content"]
        self.assertTrue(content.startswith("📚 Consulting the library."))
        self.assertIn("Here is what I found", content)

    async def test_status_appears_above_the_body(self):
        handler, msg = self._make_handler()
        await handler.update_text("the reply so far")
        await handler.set_status("🔍 Searching the web…")
        content = msg.edit.call_args.kwargs["content"]
        self.assertTrue(content.startswith("🔍 Searching the web…"))
        self.assertIn("the reply so far", content)

    async def test_clearing_the_status_removes_it(self):
        handler, msg = self._make_handler()
        await handler.update_text("body")
        await handler.set_status("working…")
        await handler.set_status(None)
        self.assertEqual(msg.edit.call_args.kwargs["content"], "body")

    async def test_composed_output_never_exceeds_the_cap(self):
        handler, _msg = self._make_handler()
        handler.status_text = "a status line of some length"
        composed = handler._compose("x" * 5000)
        self.assertLessEqual(len(composed), 2000)

    async def test_long_body_keeps_the_TAIL_not_the_head(self):
        # The old [:2000] froze the visible text once a reply passed the cap —
        # the user watched a stationary prefix while tokens kept arriving.
        handler, _msg = self._make_handler()
        body = "START" + ("x" * 3000) + "NEWEST"
        composed = handler._compose(body)
        self.assertTrue(composed.endswith("NEWEST"))
        self.assertNotIn("START", composed)
        self.assertTrue(composed.startswith("…"))

    async def test_final_update_clears_the_status_line(self):
        handler, msg = self._make_handler()
        handler.status_text = "still working…"
        await handler.final_update("the finished reply")
        self.assertIsNone(handler.status_text)
        self.assertEqual(msg.edit.call_args.kwargs["content"], "the finished reply")


class TestFinalUpdateChunking(unittest.IsolatedAsyncioTestCase):
    """final_update owns chunking; on_message used to duplicate the logic."""

    def _make_handler(self):
        msg = MagicMock()
        msg.edit = AsyncMock()
        msg.channel = MagicMock()
        msg.channel.send = AsyncMock()
        return StreamingMessageHandler(msg, asyncio.get_event_loop(),
                                       min_update_interval=0.0), msg

    async def test_long_reply_is_chunked_across_messages(self):
        handler, msg = self._make_handler()
        await handler.final_update("word " * 900)      # ~4500 chars
        self.assertGreaterEqual(msg.channel.send.await_count, 1)
        first = msg.edit.call_args.kwargs["content"]
        self.assertLessEqual(len(first), 2000)

    async def test_every_chunk_fits_the_cap(self):
        handler, msg = self._make_handler()
        await handler.final_update("word " * 1500)
        sent = [msg.edit.call_args.kwargs["content"]]
        sent += [c.args[0] for c in msg.channel.send.await_args_list if c.args]
        for chunk in sent:
            with self.subTest(n=len(chunk)):
                self.assertLessEqual(len(chunk), 2000)

    async def test_no_text_is_lost_across_the_chunks(self):
        handler, msg = self._make_handler()
        marker = "UNIQUE_TAIL_MARKER"
        await handler.final_update(("word " * 900) + marker)
        sent = [msg.edit.call_args.kwargs["content"]]
        sent += [c.args[0] for c in msg.channel.send.await_args_list if c.args]
        self.assertIn(marker, "".join(sent))

    async def test_files_still_follow_the_text(self):
        handler, msg = self._make_handler()
        await handler.final_update("word " * 900, files=[MagicMock()])
        self.assertTrue(any(
            "files" in c.kwargs for c in msg.channel.send.await_args_list))


class TestIdentityRecordedOnlyForRealTurns(unittest.TestCase):
    """The bot used to record ITSELF as one of its own users.

    identity_store.record ran before the `ctx.author == client.user` guard, so
    every message Fritz sent upserted an alias row for the bot account — and it
    also fired for ambient guild chatter he was never addressed in, quietly
    building a roster of people who had not interacted with him at all.

    Source-level because on_message is a client event handler whose collaborators
    (a live gateway, a Message, a Channel) make a behavioural test far more
    scaffolding than the assertion is worth — and the bug is purely one of
    statement order, which is exactly what this checks.
    """

    def test_record_comes_after_the_early_returns(self):
        src = (pathlib.Path(__file__).resolve().parent.parent
               / "main_discord.py").read_text(encoding="utf-8")
        body = src.split("async def on_message(", 1)[1]
        self_guard = body.index("ctx.author == client.user")
        mention_guard = body.index("client.user.mentioned_in(ctx)")
        record = body.index("identity_store.record(")
        self.assertGreater(record, self_guard,
                           "identity_store.record runs before the self-check — "
                           "the bot records itself as a user")
        self.assertGreater(record, mention_guard,
                           "identity_store.record runs before the mention "
                           "check — ambient chatter creates alias rows")


if __name__ == "__main__":
    unittest.main()
