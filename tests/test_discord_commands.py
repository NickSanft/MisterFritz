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

import admin_panel  # noqa: E402
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
        self.client.tree = MagicMock()
        self.client.tree.sync = AsyncMock(return_value=[])
        self.scheduler = MagicMock()
        self.cog = MagicMock()
        self.cog.sayer = MagicMock(name="already-held-engine")

        # A real registry rather than a bare mock: on_ready asks get_cog
        # whether it has already registered, so a client that always answers
        # "yes" (any MagicMock attribute is truthy) or always "no" would make
        # the idempotency tests below meaningless in opposite directions.
        self.cogs = {}

        async def _add_cog(cog, **kwargs):
            self.cogs["FritzCommands"] = cog

        self.client.add_cog = AsyncMock(side_effect=_add_cog)
        self.client.get_cog = MagicMock(side_effect=lambda name: self.cogs.get(name))

        # Module-level once-only state, reset so test order cannot decide
        # whether a step is skipped.
        for module, name, value in (
            (main_discord, "_models_prewarmed", False),
            (admin_panel, "_PANEL_THREAD", None),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

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


class TestOnReadyIsIdempotent(OnReadyCase):
    """discord.py dispatches `ready` on every READY, not only the first, so a
    gateway reconnect — routine on a home connection — re-ran all of on_ready.

    It built a second ScheduleManager with its own AsyncIOScheduler and
    replayed every persisted row onto it, while the first was never stopped
    (ScheduleManager.stop had no caller anywhere). Every /schedule row then
    fired twice per period, each firing a channel post and a full ask_stuff
    turn, compounding to N+1 after N reconnects and recovering only on a
    process restart.
    """

    async def reconnect(self, times=2):
        for _ in range(times):
            await main_discord.on_ready()

    async def test_one_scheduler_however_many_reconnects(self):
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect(times=4)
        self.assertEqual(main_discord.ScheduleManager.call_count, 1)
        self.scheduler.start.assert_called_once()

    async def test_the_surviving_scheduler_is_the_one_everything_else_holds(self):
        """The subtler half of the bug: add_cog raised on the duplicate name,
        which aborted the rest of on_ready and left the live cog holding
        manager #1 while on_message read manager #2 out of the global."""
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect()
        built = main_discord.ScheduleManager.return_value
        self.assertIs(main_discord.schedule_manager, built)
        self.assertIs(main_discord.FritzCommands.call_args.args[2], built)
        for call in main_discord.relay_housekeeping.await_args_list:
            self.assertIs(call.args[0], built)

    async def test_the_cog_is_registered_once(self):
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect()
        self.client.add_cog.assert_awaited_once()
        self.assertEqual(main_discord.FritzCommands.call_count, 1)

    async def test_the_models_are_pre_warmed_once(self):
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect(times=3)
        main_discord.prewarm_models.assert_called_once()

    async def test_the_admin_panel_is_started_once(self):
        """The call is made every time; the guard that matters is inside
        start_admin_panel, because the port belongs to the process. This
        asserts the call still happens, so the guard is reached."""
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect()
        self.assertEqual(main_discord.start_admin_panel.call_count, 2)

    async def test_housekeeping_still_runs_on_every_reconnect(self):
        """Deliberately not guarded: it closes reservations a previous process
        left open and purges expired rows, and its own docstring says it must
        tolerate re-running."""
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect(times=3)
        self.assertEqual(main_discord.relay_housekeeping.await_count, 3)

    async def test_the_commands_are_still_synced_on_a_reconnect(self):
        """The regression this bug hid: add_cog raised before the sync, so a
        reconnect never reached it. Left unguarded so a failed first sync can
        still be repaired."""
        with self._loading_tts({"return_value": MagicMock()}):
            await self.reconnect(times=3)
        self.assertEqual(self.client.tree.sync.await_count, 3)
        self.assertIs(self.client.tree.on_error, main_discord.handle_tree_error)

    async def test_a_tts_engine_that_arrives_late_reaches_the_live_cog(self):
        """TTS is retried while it has not succeeded, so the engine can turn up
        after the cog was registered holding None. Without attaching it, the
        retry is inert and /voice stays unavailable for the whole process."""
        engine = MagicMock(name="TTSEngine")
        self.cog.sayer = None
        with self._loading_tts({"side_effect": RuntimeError("model download failed")}):
            with self.assertLogs("main_discord", level="ERROR"):
                await main_discord.on_ready()
        self.assertIsNone(self.cogs["FritzCommands"].sayer)

        with self._loading_tts({"return_value": engine}):
            await main_discord.on_ready()
        self.assertIs(self.cogs["FritzCommands"].sayer, engine)

    async def test_a_cog_that_already_has_an_engine_is_left_alone(self):
        engine = MagicMock(name="TTSEngine")
        with self._loading_tts({"return_value": engine}):
            await self.reconnect()
        held = self.cogs["FritzCommands"].sayer
        self.assertIsNot(held, engine, "the first cog's own engine was replaced")


class TestTheAdminPanelBindsItsPortOnce(unittest.TestCase):
    """on_ready calls start_admin_panel on every reconnect, and two uvicorn
    servers cannot share a port: the second bind loses and leaves a dead
    thread, while the app it built holds a different schedule_manager than the
    live one."""

    def setUp(self):
        for name, value in (("_PANEL_THREAD", None),
                            ("ADMIN_PANEL_PASSWORD", "pw"),
                            ("CHAT_PASSWORD", "pw")):
            patcher = patch.object(admin_panel, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _start_twice(self):
        made = []

        class FakeThread:
            def __init__(self, **kwargs):
                made.append(self)
                self.alive = False

            def start(self):
                self.alive = True

            def is_alive(self):
                return self.alive

        with patch.object(admin_panel.threading, "Thread", FakeThread), \
                patch.object(admin_panel, "create_app", MagicMock()), \
                patch.object(admin_panel, "uvicorn", MagicMock()):
            first = admin_panel.start_admin_panel()
            second = admin_panel.start_admin_panel()
        return first, second, made

    def test_only_one_server_thread_is_created(self):
        first, second, made = self._start_twice()
        self.assertEqual(len(made), 1)

    def test_the_second_call_reports_the_port_that_is_serving(self):
        first, second, _made = self._start_twice()
        self.assertEqual(second, first)
        self.assertEqual(second, admin_panel.ADMIN_PANEL_PORT)

    def test_a_dead_thread_does_not_block_a_restart(self):
        """The guard asks whether the server is alive, not whether one was
        ever made, so a crashed panel can still be brought back."""
        _first, _second, made = self._start_twice()
        made[0].alive = False
        with patch.object(admin_panel.threading, "Thread", MagicMock()), \
                patch.object(admin_panel, "create_app", MagicMock()), \
                patch.object(admin_panel, "uvicorn", MagicMock()):
            self.assertEqual(admin_panel.start_admin_panel(),
                             admin_panel.ADMIN_PANEL_PORT)
            self.assertTrue(admin_panel.threading.Thread.called)

    def test_a_disabled_panel_never_marks_itself_started(self):
        with patch.object(admin_panel, "ADMIN_PANEL_PASSWORD", None):
            self.assertIsNone(admin_panel.start_admin_panel())
        self.assertIsNone(admin_panel._PANEL_THREAD)


if __name__ == "__main__":
    unittest.main()
