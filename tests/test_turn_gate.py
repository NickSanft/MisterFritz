"""One turn at a time per LangGraph thread.

Nothing serialised a turn before this. `thread_id_for` returns the bare
identity unless THREADS_PER_CHANNEL is on, so one person's DMs, every channel
they mention Fritz in, and the web chat all share a single thread; the blocking
pool allows BLOCKING_POOL_SIZE turns at once and `on_message` offloads every
message onto it with no per-user gate. Two overlapping turns both streamed
against the one SqliteSaver, whose internal lock makes each individual database
operation safe but does not make a turn's read-modify-write atomic. The later
writer won, and one question with its answer vanished from the history with no
error, no metric and no log line.

The first class below is the canary for that: it performs the interleave
directly against a real SqliteSaver and asserts an exchange is lost. If it ever
starts failing, the checkpointer has begun serialising turns itself and the
gate has become belt without braces - which is worth knowing rather than
leaving a comment asserting something stale.
"""
import os
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mister_fritz as mf  # noqa: E402
from fritz_utils import MessageSource  # noqa: E402

TIMEOUT = 10        # generous: these are handshakes, not races


class _Transcript:
    """A read-modify-write turn against a real checkpointer, with the read and
    the write separable so the interleave can be forced rather than raced."""

    def __init__(self, saver, thread_id="discord-1"):
        self.saver = saver
        self.config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    def read(self) -> list:
        from langgraph.checkpoint.base import empty_checkpoint  # noqa: F401
        found = self.saver.get_tuple(self.config)
        if not found:
            return []
        return list(found.checkpoint["channel_values"].get("messages", []))

    def write(self, messages) -> None:
        from langgraph.checkpoint.base import empty_checkpoint
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"]["messages"] = list(messages)
        self.saver.put(self.config, checkpoint, {"step": len(messages)}, {})


class CheckpointCase(unittest.TestCase):
    def setUp(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.path = os.path.join(tempfile.mkdtemp(), "turns.db")
        self._saver_cm = SqliteSaver.from_conn_string(self.path)
        self.saver = self._saver_cm.__enter__()
        self.addCleanup(lambda: self._saver_cm.__exit__(None, None, None))
        self.transcript = _Transcript(self.saver)

    def interleave(self, gated: bool) -> list:
        """Run two turns whose read and write are forced to interleave.

        Turn A reads, then waits for turn B to finish entirely before writing.
        Unguarded that is the exact shape of the bug - A writes a transcript it
        read before B existed, so B's exchange is gone. Guarded, B cannot read
        until A has written, and the handshake below simply never completes in
        that order, which is the point.
        """
        a_has_read = threading.Event()
        b_is_done = threading.Event()

        def turn(word, *, waits_for_b):
            def body():
                messages = self.transcript.read()
                if waits_for_b:
                    a_has_read.set()
                    b_is_done.wait(timeout=1)       # short: it must not hang when gated
                messages.append(word)
                self.transcript.write(messages)
                if not waits_for_b:
                    b_is_done.set()

            if gated:
                with mf.one_turn_at_a_time("discord-1"):
                    body()
            else:
                body()

        first = threading.Thread(target=turn, args=("first",), kwargs={"waits_for_b": True})
        second = threading.Thread(target=turn, args=("second",), kwargs={"waits_for_b": False})
        first.start()
        a_has_read.wait(timeout=TIMEOUT)
        second.start()
        first.join(timeout=TIMEOUT)
        second.join(timeout=TIMEOUT)
        self.assertFalse(first.is_alive() or second.is_alive(), "a turn never finished")
        return self.transcript.read()


class TestTheHazardIsReal(CheckpointCase):
    def test_an_unserialised_turn_loses_the_other_exchange(self):
        """The canary. SqliteSaver's own lock does not make this safe, and
        plan 08 never claimed it did."""
        surviving = self.interleave(gated=False)
        self.assertEqual(surviving, ["first"],
                         "the checkpointer now appears to serialise turns itself; "
                         "re-read whether the gate is still load-bearing")


class TestTheGateKeepsBothExchanges(CheckpointCase):
    def test_both_turns_survive_when_serialised(self):
        surviving = self.interleave(gated=True)
        self.assertEqual(sorted(surviving), ["first", "second"])


class GateCase(unittest.TestCase):
    def setUp(self):
        mf._TURN_GATES.clear()
        self.addCleanup(mf._TURN_GATES.clear)
        self.metrics = unittest.mock.MagicMock()
        patcher = unittest.mock.patch.object(mf, "METRICS", self.metrics)
        patcher.start()
        self.addCleanup(patcher.stop)

    def counted(self) -> list:
        return [call.args[0] for call in self.metrics.increment.call_args_list]


class TestOneTurnAtATime(GateCase):
    def test_a_second_turn_waits_for_the_first(self):
        inside = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()

        def first():
            with mf.one_turn_at_a_time("thread-a"):
                inside.set()
                release.wait(timeout=TIMEOUT)

        def second():
            with mf.one_turn_at_a_time("thread-a"):
                second_entered.set()

        a = threading.Thread(target=first)
        a.start()
        self.assertTrue(inside.wait(timeout=TIMEOUT))
        b = threading.Thread(target=second)
        b.start()
        self.assertFalse(second_entered.wait(timeout=0.3),
                         "the second turn ran while the first held the thread")
        release.set()
        self.assertTrue(second_entered.wait(timeout=TIMEOUT))
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)

    def test_waiting_is_counted(self):
        inside = threading.Event()
        release = threading.Event()

        def first():
            with mf.one_turn_at_a_time("thread-a"):
                inside.set()
                release.wait(timeout=TIMEOUT)

        a = threading.Thread(target=first)
        a.start()
        inside.wait(timeout=TIMEOUT)
        b = threading.Thread(target=lambda: self._enter_and_leave("thread-a"))
        b.start()
        # The counter is incremented before the blocking acquire, so it is
        # observable while the waiter is still parked.
        for _ in range(int(TIMEOUT * 100)):
            if "turn_waited_for_thread" in self.counted():
                break
            threading.Event().wait(0.01)
        self.assertIn("turn_waited_for_thread", self.counted())
        release.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)

    @staticmethod
    def _enter_and_leave(thread_id):
        with mf.one_turn_at_a_time(thread_id):
            pass

    def test_a_turn_that_has_to_wait_does_not_count_as_waiting_twice(self):
        self._enter_and_leave("thread-a")
        self.assertNotIn("turn_waited_for_thread", self.counted())

    def test_two_different_threads_never_wait_for_each_other(self):
        """One person's turn must not delay another's. Two people are two
        thread ids, so the gate must be per thread and not global."""
        both_inside = threading.Barrier(2, timeout=TIMEOUT)
        failures = []

        def turn(thread_id):
            try:
                with mf.one_turn_at_a_time(thread_id):
                    both_inside.wait()
            except Exception as e:                      # pragma: no cover
                failures.append(e)

        threads = [threading.Thread(target=turn, args=(f"thread-{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=TIMEOUT)
            self.assertFalse(t.is_alive())
        self.assertEqual(failures, [], "the two turns did not overlap")

    def test_a_third_concurrent_turn_is_refused_immediately(self):
        """A waiter holds a pool thread for as long as the turn ahead of it, so
        an unbounded queue would let one person stall the bot for everyone.
        Two is enough to answer a follow-up in order, which is the case that
        actually happens."""
        inside = threading.Event()
        release = threading.Event()

        def first():
            with mf.one_turn_at_a_time("thread-a"):
                inside.set()
                release.wait(timeout=TIMEOUT)

        a = threading.Thread(target=first)
        a.start()
        inside.wait(timeout=TIMEOUT)
        b = threading.Thread(target=lambda: self._enter_and_leave("thread-a"))
        b.start()
        for _ in range(int(TIMEOUT * 100)):             # let b become the waiter
            if mf._TURN_GATES["thread-a"].participants == 2:
                break
            threading.Event().wait(0.01)

        with self.assertRaises(mf.ThreadBusy):
            self._enter_and_leave("thread-a")

        release.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)

    def test_a_refused_turn_leaves_no_trace(self):
        inside = threading.Event()
        release = threading.Event()

        def first():
            with mf.one_turn_at_a_time("thread-a"):
                inside.set()
                release.wait(timeout=TIMEOUT)

        a = threading.Thread(target=first)
        a.start()
        inside.wait(timeout=TIMEOUT)
        b = threading.Thread(target=lambda: self._enter_and_leave("thread-a"))
        b.start()
        for _ in range(int(TIMEOUT * 100)):
            if mf._TURN_GATES["thread-a"].participants == 2:
                break
            threading.Event().wait(0.01)
        with self.assertRaises(mf.ThreadBusy):
            self._enter_and_leave("thread-a")
        self.assertEqual(mf._TURN_GATES["thread-a"].participants, 2,
                         "the refused turn was counted as a participant")
        release.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)
        self.assertEqual(mf._TURN_GATES, {})

    def test_the_gate_is_forgotten_once_the_last_turn_leaves(self):
        """Otherwise this accumulates an entry per person per channel for the
        life of the process."""
        self._enter_and_leave("thread-a")
        self._enter_and_leave("thread-b")
        self.assertEqual(mf._TURN_GATES, {})

    def test_a_turn_that_raises_still_releases_the_thread(self):
        with self.assertRaises(ValueError):
            with mf.one_turn_at_a_time("thread-a"):
                raise ValueError("the model fell over")
        self.assertEqual(mf._TURN_GATES, {})
        self._enter_and_leave("thread-a")          # must not deadlock


class TestAskStuffUsesTheGate(GateCase):
    """The integration point. app.stream is replaced, because the question is
    whether two calls can be inside it at once - not what the model says."""

    def setUp(self):
        super().setUp()
        self.overlapped = []
        self.inside = []
        self.app = unittest.mock.MagicMock()
        self.app.stream.side_effect = self._stream
        for name, value in (("app", self.app),
                            ("extract_memories_background", unittest.mock.MagicMock())):
            patcher = unittest.mock.patch.object(mf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.hold = threading.Event()
        self.entered = threading.Event()

    def _stream(self, inputs, config=None, stream_mode=None):
        self.inside.append(1)
        if len(self.inside) > 1:
            self.overlapped.append(True)
        self.entered.set()
        self.hold.wait(timeout=TIMEOUT)
        self.inside.pop()
        return iter(())

    def _ask(self, user_id="discord-1", prompt="hello"):
        return mf.ask_stuff(prompt, MessageSource.DISCORD_TEXT, user_id)

    def test_two_turns_from_one_person_do_not_overlap(self):
        a = threading.Thread(target=self._ask)
        a.start()
        self.assertTrue(self.entered.wait(timeout=TIMEOUT))
        b = threading.Thread(target=self._ask)
        b.start()
        threading.Event().wait(0.3)          # long enough for an overlap to show
        self.hold.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)
        self.assertEqual(self.app.stream.call_count, 2)
        self.assertEqual(self.overlapped, [], "two turns were inside app.stream at once")

    def test_two_people_are_not_held_up_by_each_other(self):
        a = threading.Thread(target=lambda: self._ask(user_id="discord-1"))
        a.start()
        self.assertTrue(self.entered.wait(timeout=TIMEOUT))
        b = threading.Thread(target=lambda: self._ask(user_id="discord-2"))
        b.start()
        for _ in range(int(TIMEOUT * 100)):
            if self.overlapped:
                break
            threading.Event().wait(0.01)
        self.hold.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)
        self.assertEqual(self.overlapped, [True],
                         "two different people were serialised against each other")

    def test_a_third_concurrent_message_is_told_to_wait(self):
        a = threading.Thread(target=self._ask)
        a.start()
        self.assertTrue(self.entered.wait(timeout=TIMEOUT))
        b = threading.Thread(target=self._ask)
        b.start()
        for _ in range(int(TIMEOUT * 100)):
            if mf._TURN_GATES.get("discord-1") and \
                    mf._TURN_GATES["discord-1"].participants == 2:
                break
            threading.Event().wait(0.01)

        answer = self._ask(prompt="and another thing")

        self.assertEqual(answer["image_paths"], [])
        self.assertIn("timestamp", answer)
        self.assertIn("still attending", answer["text"])
        self.assertIn("turn_refused_thread_busy", self.counted())
        self.assertEqual(self.app.stream.call_count, 1,
                         "the refused turn reached the model anyway")
        self.hold.set()
        a.join(timeout=TIMEOUT)
        b.join(timeout=TIMEOUT)

    def test_an_explicit_thread_id_is_what_gates(self):
        """The web chat passes thread_id so a web session cannot overwrite the
        Discord history of the same identity - so it must gate on that, not on
        the identity."""
        a = threading.Thread(
            target=lambda: mf.ask_stuff("hello", MessageSource.DISCORD_TEXT, "discord-1",
                                        thread_id="web-discord-1"))
        a.start()
        self.assertTrue(self.entered.wait(timeout=TIMEOUT))
        self.assertIn("web-discord-1", mf._TURN_GATES)
        self.assertNotIn("discord-1", mf._TURN_GATES)
        self.hold.set()
        a.join(timeout=TIMEOUT)


class TestAScheduledTurnWaitsRatherThanBeingRefused(GateCase):
    """A reminder posts whatever text comes back straight into the channel, so
    being refused would deliver "One moment, sir" AS the reminder and lose the
    real one. It is also not the thing the cap defends against: one firing per
    schedule per period, on the default executor rather than the bounded
    blocking pool."""

    def test_a_patient_caller_is_not_refused_at_the_cap(self):
        inside = threading.Event()
        release = threading.Event()

        def holder():
            with mf.one_turn_at_a_time("thread-a"):
                inside.set()
                release.wait(timeout=TIMEOUT)

        a = threading.Thread(target=holder)
        a.start()
        inside.wait(timeout=TIMEOUT)
        b = threading.Thread(target=lambda: self._enter("thread-a"))
        b.start()
        for _ in range(int(TIMEOUT * 100)):
            if mf._TURN_GATES["thread-a"].participants == 2:
                break
            threading.Event().wait(0.01)

        # An interactive caller is refused here; a patient one queues.
        with self.assertRaises(mf.ThreadBusy):
            self._enter("thread-a")

        entered = threading.Event()

        def patient():
            with mf.one_turn_at_a_time("thread-a", may_wait=True):
                entered.set()

        c = threading.Thread(target=patient)
        c.start()
        self.assertFalse(entered.wait(timeout=0.3), "the patient turn did not wait")
        release.set()
        self.assertTrue(entered.wait(timeout=TIMEOUT),
                        "the patient turn never got its turn")
        for t in (a, b, c):
            t.join(timeout=TIMEOUT)
        self.assertEqual(mf._TURN_GATES, {})

    @staticmethod
    def _enter(thread_id, **kwargs):
        with mf.one_turn_at_a_time(thread_id, **kwargs):
            pass

    def test_ask_stuff_passes_patience_through(self):
        seen = []
        import contextlib

        @contextlib.contextmanager
        def recording(thread_id, *, may_wait=False):
            seen.append(may_wait)
            yield

        app = unittest.mock.MagicMock()
        app.stream.return_value = iter(())
        with unittest.mock.patch.object(mf, "one_turn_at_a_time", recording), \
                unittest.mock.patch.object(mf, "app", app), \
                unittest.mock.patch.object(mf, "extract_memories_background",
                                           unittest.mock.MagicMock()):
            mf.ask_stuff("hello", MessageSource.DISCORD_TEXT, "discord-1")
            app.stream.return_value = iter(())
            mf.ask_stuff("remind", MessageSource.LOCAL, "discord-1", may_wait=True)
        self.assertEqual(seen, [False, True])


class TestTheSchedulerIsThePatientCaller(unittest.IsolatedAsyncioTestCase):
    """_run_task had no test at all, which is part of why the identity bug in
    A4 survived. This covers the one property A6 depends on: the turn it books
    queues rather than being refused."""

    async def test_run_task_asks_to_wait(self):
        import scheduler

        channel = unittest.mock.MagicMock()
        channel.send = unittest.mock.AsyncMock(
            return_value=unittest.mock.MagicMock(
                edit=unittest.mock.AsyncMock()))
        bot = unittest.mock.MagicMock()
        bot.get_channel.return_value = channel

        manager = scheduler.ScheduleManager.__new__(scheduler.ScheduleManager)
        manager.bot = bot

        asked = {}

        def fake_ask(prompt, source, user_id, **kwargs):
            asked.update(kwargs)
            asked["prompt"] = prompt
            return {"text": "the reminder itself", "image_paths": []}

        with unittest.mock.patch.dict(
                sys.modules, {"mister_fritz": unittest.mock.MagicMock(ask_stuff=fake_ask)}):
            with unittest.mock.patch.object(scheduler.identity_store, "display_name",
                                            return_value="Nick"):
                await manager._run_task("sid1", "discord-1", 4242, "stand up")

        self.assertTrue(asked.get("may_wait"),
                        "a scheduled turn would be refused and post the refusal")
        self.assertEqual(asked["prompt"], "stand up")


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
