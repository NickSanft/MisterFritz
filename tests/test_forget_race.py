"""
A forget that races a turn in flight.

A turn loads the whole conversation when it starts and commits all of it back
when it ends. A delete that landed in between was undone a moment later: the
forget reported success, and the conversation came back. Reproduced in review
of plan 14 P3 for the web chat's "Burn the correspondence", and then on every
other path: Discord's /forget conversation and /forget all, and the admin
panel's forget-user. privacy.forget_conversation now waits for every turn on
the threads it is about to delete.

Each race test runs the real graph and the real checkpointer, with a slow fake
model holding a turn in flight. Without the wait the delete lands first and the
turn's commit restores the conversation, so each would fail on the old code.
Every background turn reports how it ended, because "no history" is also what
a turn that never committed would leave behind.
"""
import asyncio
import itertools
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage

import admin_panel
import mister_fritz
import privacy
from fritz_utils import CHAT_DB_NAME, MessageSource
from test_cancel import _bounded, _Harness, _Record, _SlowModel

_USERS = itertools.count(1)


class _Turn:
    """A turn running on a background thread, which reports how it ended."""

    def __init__(self, test, target):
        self.test, self.outcome = test, {}

        def run():
            try:
                self.outcome["result"] = target()
            except BaseException as e:
                self.outcome["error"] = e

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finished(self, interrupted=False):
        self.thread.join(15)
        self.test.assertFalse(self.thread.is_alive(), "the turn never finished")
        self.test.assertNotIn("error", self.outcome, self.outcome.get("error"))
        result = self.outcome["result"]
        self.test.assertIs(result["interrupted"], interrupted)
        self.test.assertTrue(result["text"], "the turn produced nothing")


class _RaceHarness(_Harness):
    def setUp(self):
        super().setUp()
        self.user = f"discord-forgetrace{next(_USERS)}"
        self.agent = None
        p = patch.object(mister_fritz, "_get_conversation_agent", side_effect=lambda: self.agent)
        p.start()
        self.addCleanup(p.stop)
        self.threads_used = set()
        self.addCleanup(self._delete_threads)

    def _delete_threads(self):
        for thread in self.threads_used:
            mister_fritz.app.checkpointer.delete_thread(thread)

    def history(self, thread):
        return mister_fritz.app.get_state(
            {"configurable": {"thread_id": thread}}).values.get("messages", [])

    def rows(self, thread):
        with sqlite3.connect(CHAT_DB_NAME) as conn:
            return conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                                (thread,)).fetchone()[0]

    def seed(self, thread, user=None):
        """One finished exchange on `thread`: the conversation to forget."""
        self.threads_used.add(thread)
        self.agent = self.real_agent(_SlowModel(record=_Record(), total=2))
        _bounded(self, lambda: mister_fritz.ask_stuff(
            "a secret", MessageSource.LOCAL, user or self.user, thread_id=thread,
            streaming_callback=lambda *a: None))
        self.assertTrue(self.history(thread), "nothing to forget")

    def in_flight(self, thread, user=None, cancel_event=None, tokens=10):
        """A slow turn on `thread` (0.1 s a token), returned once it is streaming."""
        self.threads_used.add(thread)
        streaming = threading.Event()
        self.agent = self.real_agent(_SlowModel(record=_Record(), delay=0.1, total=tokens))
        turn = _Turn(self, lambda: mister_fritz.ask_stuff(
            "in flight", MessageSource.LOCAL, user or self.user, thread_id=thread,
            streaming_callback=lambda *a: streaming.set(), cancel_event=cancel_event))
        self.assertTrue(streaming.wait(10), "the turn never started")
        self.addCleanup(turn.thread.join, 15)
        return turn

    def wait_until_queued(self, thread, participants=2):
        """Until something besides the running turn waits on `thread`'s gate."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            gate = mister_fritz._TURN_GATES.get(thread)
            if gate is not None and gate.participants >= participants:
                return
            time.sleep(0.01)
        self.fail(f"nothing queued on {thread}")


class TestEveryForgetWaitsForTheTurnInFlight(_RaceHarness):
    def test_discord_forget_conversation(self):
        """The identity sweep, as /forget conversation runs it."""
        self.seed(self.user)
        turn = self.in_flight(self.user)
        removed = _bounded(self, lambda: privacy.forget_conversation(self.user))
        turn.finished()
        self.assertGreater(removed, 0)
        self.assertEqual(self.history(self.user), [], "the forgotten conversation came back")

    def test_a_per_channel_thread_in_the_sweep(self):
        thread = f"{self.user}#general"
        self.seed(thread)
        turn = self.in_flight(thread)
        _bounded(self, lambda: privacy.forget_conversation(self.user))
        turn.finished()
        self.assertEqual(self.history(thread), [])

    def test_an_idle_thread_whose_turn_starts_while_the_forget_waits(self):
        """The half of the gate list read from the checkpoints table. The
        forget lists the identity's threads, then waits on a busy one; a turn
        that starts meanwhile on an idle one would load the old conversation
        and commit it back after the delete."""
        idle = f"{self.user}#idle"
        self.seed(idle)
        busy = self.in_flight(self.user)
        forget = _Turn(self, lambda: {"text": "done", "interrupted": False,
                                      "removed": privacy.forget_conversation(self.user)})
        self.wait_until_queued(self.user)
        late = self.in_flight(idle, tokens=15)
        busy.finished()
        late.finished()
        forget.finished()
        self.assertEqual(self.history(idle), [], "the idle thread's conversation came back")

    def test_a_turn_that_has_not_written_its_first_checkpoint(self):
        """The half of the gate list read from the turns in flight. A turn
        holds its thread's gate a moment before LangGraph writes anything, so
        the checkpoints table cannot show it yet."""
        thread = f"{self.user}#first-words"
        self.threads_used.add(thread)
        holding, wrote = threading.Event(), threading.Event()

        def turn_before_its_first_write():
            with mister_fritz.one_turn_at_a_time(thread):
                holding.set()
                time.sleep(0.5)
                mister_fritz.app.update_state({"configurable": {"thread_id": thread}},
                                              {"messages": [HumanMessage(content="late")]})
                wrote.set()

        t = threading.Thread(target=turn_before_its_first_write, daemon=True)
        t.start()
        self.assertTrue(holding.wait(5))
        self.assertEqual(self.rows(thread), 0, "the setup wrote too early")
        _bounded(self, lambda: privacy.forget_conversation(self.user))
        t.join(10)
        self.assertTrue(wrote.is_set(), "the stand-in turn never wrote")
        self.assertEqual(self.history(thread), [], "the turn's first commit landed after the forget")

    def test_forget_all(self):
        """Discord's /forget all and the admin panel's forget-user both run it."""
        self.seed(self.user)
        turn = self.in_flight(self.user)
        result = _bounded(self, lambda: privacy.forget_all(self.user))
        turn.finished()
        self.assertNotIn("forget_conversation", result["failed"])
        self.assertGreater(result["conversation_rows"], 0)
        self.assertEqual(self.history(self.user), [])

    def test_the_web_chats_burn_through_its_endpoint(self):
        """The burn handler aborts the reply and posts /chat/forget at once."""
        from starlette.testclient import TestClient
        identity = "web-burnrace"
        thread = admin_panel._chat_thread_id(identity)
        self.seed(thread, user=identity)
        stop = threading.Event()
        turn = self.in_flight(thread, user=identity, cancel_event=stop)
        stop.set()
        client = TestClient(admin_panel.create_app("pw", chat_password="chatpw"))
        with patch.object(admin_panel, "_chat_identity", lambda request: (identity, "burnrace")):
            r = _bounded(self, lambda: client.post("/chat/forget", follow_redirects=False))
        turn.finished(interrupted=True)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.history(thread), [], "the burned conversation came back")


class TestThePanelKeepsServingWhileAForgetWaits(_RaceHarness):
    """Both forget endpoints run the forget off the event loop. On it, a forget
    waiting for a turn would stall every other request on the panel, web chat
    streams included, for as long as the turn ran."""

    def setUp(self):
        super().setUp()
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(
            admin_panel.create_app("pw", chat_password="chatpw"), host="127.0.0.1", port=0,
            log_level="warning", access_log=False, ws="none"))
        serving = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
        serving.start()

        def stop():
            server.should_exit = True
            serving.join(10)

        self.addCleanup(stop)
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        base = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        # Built now, so their setup (an SSL context each) is not what the
        # timing below measures.
        self.poster = httpx.Client(base_url=base, auth=("admin", "pw"), timeout=20)
        self.getter = httpx.Client(base_url=base, timeout=20)
        self.addCleanup(self.poster.close)
        self.addCleanup(self.getter.close)
        self.getter.get("/chat")         # and the connection is already open

    def _meanwhile(self, thread, path):
        outcome = {}
        forgetting = threading.Thread(daemon=True, target=lambda: outcome.update(
            status=self.poster.post(path).status_code))
        forgetting.start()
        self.wait_until_queued(thread)
        t0 = time.monotonic()
        meanwhile = self.getter.get("/chat")
        elapsed = time.monotonic() - t0
        forgetting.join(20)
        self.assertEqual(meanwhile.status_code, 200)
        self.assertLess(elapsed, 0.4, "the panel stopped answering while the forget waited")
        self.assertEqual(outcome.get("status"), 303)

    def test_the_admin_forget_user(self):
        self.seed(self.user)
        turn = self.in_flight(self.user)
        self._meanwhile(self.user, f"/users/{self.user}/forget")
        turn.finished()
        self.assertEqual(self.history(self.user), [])

    def test_the_web_chats_forget(self):
        """Another tab's reply is not cancelled by this one's burn, so the
        forget waits for all of it."""
        identity = "web-meanwhile"
        thread = admin_panel._chat_thread_id(identity)
        self.seed(thread, user=identity)
        turn = self.in_flight(thread, user=identity)
        with patch.object(admin_panel, "_chat_identity", lambda request: (identity, "meanwhile")):
            self._meanwhile(thread, "/chat/forget")
        turn.finished()
        self.assertEqual(self.history(thread), [])


class TestForgetWithNothingToWaitFor(_RaceHarness):
    def test_it_does_not_wait_and_leaves_no_gate_behind(self):
        self.seed(self.user)
        t0 = time.monotonic()
        _bounded(self, lambda: privacy.forget_conversation(self.user))
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual([t for t in mister_fritz.threads_in_flight() if t.startswith(self.user)], [])

    def test_an_explicit_thread_waits_only_for_its_own_and_touches_nothing_else(self):
        """/chat/forget names one thread; a turn on another must not hold it
        up, and no other thread of the identity may go with it."""
        busy, idle = f"{self.user}#elsewhere", f"{self.user}#idle"
        self.seed(self.user)
        self.seed(idle)
        turn = self.in_flight(busy)
        t0 = time.monotonic()
        _bounded(self, lambda: privacy.forget_conversation(self.user, thread_id=self.user))
        self.assertLess(time.monotonic() - t0, 0.5, "it waited for a turn on a different thread")
        self.assertGreater(self.rows(busy), 0, "a thread it was not asked about was deleted")
        turn.finished()
        self.assertEqual(self.history(self.user), [])
        self.assertTrue(self.history(idle), "a thread it was not asked about was deleted")

    def test_gates_are_taken_in_sorted_order(self):
        """Two forgets whose lists differ would otherwise take shared gates in
        different orders and could each hold one the other is waiting for."""
        for thread in (f"{self.user}#b", f"{self.user}#a", self.user):
            self.seed(thread)
        threads = privacy._threads_to_forget(self.user, True, *privacy._sweep_where(self.user))
        self.assertEqual(threads, sorted(threads))
        self.assertEqual(len(threads), 3)

    def test_two_forgets_at_once_do_not_deadlock(self):
        threads = [self.user, f"{self.user}#a"]
        for thread in threads:
            self.seed(thread)
        turns = [self.in_flight(thread) for thread in threads]
        forgets = [_Turn(self, lambda: {"text": "done", "interrupted": False,
                                        "removed": privacy.forget_conversation(self.user)})
                   for _ in range(2)]
        for f in forgets:
            f.finished()
        for t in turns:
            t.finished()
        for thread in threads:
            self.assertEqual(self.history(thread), [])

    def test_a_gate_that_cannot_be_taken_releases_those_already_held(self):
        """Otherwise every turn on the threads before it would wait forever."""
        threads = [self.user, f"{self.user}#a"]
        real_gate = mister_fritz.one_turn_at_a_time

        def gate(thread_id, **kwargs):
            if thread_id == threads[1]:
                raise RuntimeError("cannot take this gate")
            return real_gate(thread_id, **kwargs)

        held = None
        with patch.object(mister_fritz, "one_turn_at_a_time", gate):
            try:
                privacy._no_turn_in_flight(threads)
            except RuntimeError:
                # Checked here, while the traceback still holds the frame: once
                # it is cleared, CPython would release a leaked gate anyway.
                held = threads[0] in mister_fritz.threads_in_flight()
        self.assertIs(held, False, "a gate was left held")


if __name__ == "__main__":
    unittest.main()
