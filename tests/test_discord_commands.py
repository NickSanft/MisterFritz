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

import main_discord  # noqa: E402
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




class TestStreamingCallbackFactory(unittest.IsolatedAsyncioTestCase):
    """The callback ask_stuff invokes from a worker thread.

    It was a closure inside on_message, so reaching it meant standing up a live
    gateway — and nothing covered its arity, its cross-thread rate limit, or
    the fact that a restart must bypass that limit. Extracting the factory is
    what made these three assertions possible at all.
    """

    def _handler(self):
        h = MagicMock()
        h.update_text = AsyncMock()
        return h

    async def test_rate_limit_drops_intermediate_hops(self):
        """At ~40 tokens/s an unthrottled callback queues 40 coroutines a
        second onto the gateway loop."""
        h = self._handler()
        cb = main_discord.make_streaming_callback(h, asyncio.get_event_loop(),
                                                  min_interval=60.0)
        cb("a", "a", True)      # first call always goes
        cb("b", "ab", False)    # dropped
        cb("c", "abc", False)   # dropped
        await asyncio.sleep(0.05)   # let run_coroutine_threadsafe drain
        self.assertEqual(h.update_text.await_count, 1)

    async def test_restart_is_never_dropped(self):
        """A restart marks a NEW answer segment. Skipping it would leave the
        previous segment's text on screen."""
        h = self._handler()
        cb = main_discord.make_streaming_callback(h, asyncio.get_event_loop(),
                                                  min_interval=60.0)
        cb("a", "a", False)
        cb("x", "x", True)      # must bypass the limit
        await asyncio.sleep(0.05)   # let run_coroutine_threadsafe drain
        self.assertEqual(h.update_text.await_count, 2)
        self.assertEqual(h.update_text.await_args.args[0], "x")

    async def test_it_sends_accumulated_not_the_delta(self):
        """A Discord edit replaces the whole body, so the delta alone would
        show only the newest token."""
        h = self._handler()
        cb = main_discord.make_streaming_callback(h, asyncio.get_event_loop(),
                                                  min_interval=0.0)
        cb("well", "Very well", False)
        await asyncio.sleep(0.05)   # let run_coroutine_threadsafe drain
        self.assertEqual(h.update_text.await_args.args[0], "Very well")

    def test_it_accepts_the_three_argument_contract(self):
        import inspect
        cb = main_discord.make_streaming_callback(self._handler(), MagicMock())
        params = list(inspect.signature(cb).parameters)
        self.assertEqual(params, ["delta", "accumulated", "restart"])

class OnReadyCase(unittest.IsolatedAsyncioTestCase):
    """Drives the real `on_ready` body with everything heavy patched out.

    Nothing executed `on_ready` before this class: it was read as source by
    test_packaging and test_relay_buttons, never run. But its whole job is to
    assemble the process - scheduler, relay housekeeping, admin panel,
    pre-warm, the command cog, the tree error hook - so the only question
    worth asking is whether it reaches the end, and only the real body can
    answer that.
    """

    def setUp(self):
        self.client = MagicMock()
        self.client.add_cog = AsyncMock()
        self.client.tree = MagicMock()
        self.client.tree.sync = AsyncMock(return_value=[])
        self.scheduler = MagicMock()
        self.cog = MagicMock()
        replacements = {
            "client": self.client,
            "sayer": None,                      # so the TTS branch is entered
            "schedule_manager": None,
            "ScheduleManager": MagicMock(return_value=self.scheduler),
            "relay_housekeeping": AsyncMock(),
            "start_admin_panel": MagicMock(),
            "prewarm_models": MagicMock(),
            "FritzCommands": MagicMock(return_value=self.cog),
        }
        for name, value in replacements.items():
            patcher = patch.object(main_discord, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        installed = patch.object(main_discord.relay_router, "ensure_installed", MagicMock())
        installed.start()
        self.addCleanup(installed.stop)

    def _loading_tts(self, outcome):
        """Patch the one run_blocking call on_ready makes - the TTS load."""
        return patch.object(main_discord, "run_blocking", AsyncMock(**outcome))

    def assert_the_bot_came_up(self):
        """The assertions that matter: a boot that stops early is silent."""
        self.assertTrue(main_discord.ScheduleManager.called, "no scheduler")
        self.scheduler.start.assert_called_once()
        main_discord.relay_housekeeping.assert_awaited_once()
        main_discord.start_admin_panel.assert_called_once()
        main_discord.prewarm_models.assert_called_once()
        self.client.add_cog.assert_awaited_once_with(self.cog)
        self.assertIs(self.client.tree.on_error, main_discord.handle_tree_error)
        self.client.tree.sync.assert_awaited_once()


class TestABrokenVoiceExtraDoesNotTakeTheBootWithIt(OnReadyCase):
    """TTSEngine logs and re-raises whatever the model load threw (tts.py:50),
    and none of those is an ImportError: a failed download of the ~2GB XTTS
    weights, a CUDA OOM, a corrupt cache, an unsupported device string.

    Before this, `except ImportError` let every one of them escape on_ready,
    which skipped the scheduler, the relay housekeeping, the admin panel, the
    pre-warm, `add_cog` and `tree.sync` - so the bot logged in with no slash
    commands at all and nothing in the log tying that to TTS.
    """

    async def test_the_commands_still_register(self):
        with self._loading_tts({"side_effect": RuntimeError("CUDA out of memory")}):
            with self.assertLogs("main_discord", level="ERROR"):
                await main_discord.on_ready()
        self.assert_the_bot_came_up()

    async def test_voice_is_the_only_thing_lost(self):
        """sayer stays None, which is what makes /voice report itself
        unavailable rather than half-work."""
        with self._loading_tts({"side_effect": RuntimeError("no CUDA device")}):
            with self.assertLogs("main_discord", level="ERROR"):
                await main_discord.on_ready()
        self.assertIsNone(main_discord.sayer)
        self.assertIsNone(main_discord.FritzCommands.call_args.args[1])

    async def test_an_absent_extra_keeps_its_own_wording(self):
        """README.md:157 promises an absent extra reports itself unavailable
        rather than crashing. That path is a warning about installing the
        extra; a present-but-broken one is an error. Keep them distinct."""
        with self._loading_tts({"side_effect": ImportError("No module named 'TTS'")}):
            with self.assertLogs("main_discord", level="WARNING") as caught:
                await main_discord.on_ready()
        # The level is the distinction, not the wording: both messages
        # mention the extra, so only "WARNING and nothing worse" says this
        # path is still the ordinary absent-extra one.
        self.assertEqual([r.levelname for r in caught.records], ["WARNING"])
        self.assertIn("install the [voice]", "; ".join(caught.output))
        self.assert_the_bot_came_up()

    async def test_a_working_engine_is_handed_to_the_cog(self):
        engine = MagicMock(name="TTSEngine")
        with self._loading_tts({"return_value": engine}):
            await main_discord.on_ready()
        self.assertIs(main_discord.sayer, engine)
        self.assertIs(main_discord.FritzCommands.call_args.args[1], engine)
        self.assert_the_bot_came_up()


if __name__ == "__main__":
    unittest.main()
