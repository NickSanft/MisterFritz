"""
Cooperative cancel: stopping a turn really stops the model (P3 of
plans/14-desktop-companion-build.md).

Most of this runs the REAL agent and the real graph against a slow fake chat
model, because the mechanism is what the plan got wrong. Its suggested test
drives a fake agent.stream, and that test passes while the model carries on:
LangGraph's close() waits for the model node's worker thread, which a fake
generator does not have. TestTheFakeAgentPath keeps that test for the part it
can see; everything that claims the model stopped is measured on the model,
and TestARealChatOllama measures the HTTP request itself.

Likewise the web half uses a real uvicorn and a real socket hanging up, because
TestClient has no way to disconnect.

Every call that could block runs through _bounded, so a regression fails an
assertion instead of hanging the suite: CI has no per-test timeout.
"""
import asyncio
import gc
import itertools
import json
import logging
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

import admin_panel
import mister_fritz
from fritz_utils import MessageSource
from mister_fritz import INTERRUPTED_MARK

_THREADS = itertools.count(1)


def _bounded(test: unittest.TestCase, fn, timeout: float = 15.0):
    """fn() on a daemon thread; fail, rather than hang, if it does not return."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:      # surfaced on the test's own thread below
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        test.fail(f"did not return within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class _Record:
    def __init__(self):
        self.calls = 0          # how many times the model was asked
        self.produced = 0       # how many tokens it generated
        self.closed = False     # whether its stream was torn down


class _SlowModel(BaseChatModel):
    """Generates tokens slowly and records how far it got before being stopped.

    `cancel_at` sets `cancel_event` as token N is produced, so a test can cancel
    mid-stream without racing a clock.
    """

    record: Any
    total: int = 400
    delay: float = 0.005
    cancel_at: int | None = None
    cancel_event: Any = None

    @property
    def _llm_type(self) -> str:
        return "slow-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # As ChatOllama does: a non-streaming call still streams underneath and
        # reports every token, which is why a cancel can stop it too.
        text = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            text += chunk.text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.record.calls += 1
        try:
            for i in range(self.total):
                self.record.produced += 1
                if self.cancel_at is not None and self.record.produced == self.cancel_at:
                    self.cancel_event.set()
                time.sleep(self.delay)
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"w{i} "))
        finally:
            self.record.closed = True


class _NarratesThenCallsATool(_SlowModel):
    """Says something, asks for a tool, and would answer on its second call."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.record.calls += 1
        if self.record.calls > 1:
            yield ChatGenerationChunk(message=AIMessageChunk(content="the real answer"))
            return
        yield ChatGenerationChunk(message=AIMessageChunk(content="Let me look."))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
            {"name": "slow_lookup", "args": "{}", "id": "call_1", "index": 0, "type": "tool_call_chunk"},
        ]))


class _Harness(unittest.TestCase):
    def setUp(self):
        self.record = _Record()
        self.thread = f"cancel-test-{next(_THREADS)}"
        self.extract = MagicMock()
        for name, value in (("search_memories_internal", MagicMock(return_value="{}")),
                            ("get_user_profile", MagicMock(return_value={})),
                            ("extract_memories_background", self.extract),
                            # One delta per token, whatever a developer's .env says.
                            ("STREAM_MIN_CHARS", 1)):
            p = patch.object(mister_fritz, name, value)
            p.start()
            self.addCleanup(p.stop)

    def real_agent(self, model=None, tools=()):
        model = model or _SlowModel(record=self.record)
        return create_agent(model, tools=list(tools), middleware=mister_fritz._agent_middleware())

    def ask(self, agent, **kwargs):
        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent):
            return _bounded(self, lambda: mister_fritz.ask_stuff(
                "stop me", MessageSource.LOCAL, "local-canceltest",
                thread_id=self.thread, **kwargs))

    def cancel_after(self, event, deltas):
        """A streaming_callback that sets `event` on the Nth delta, and a list of them all."""
        seen = []

        def callback(delta, accumulated, restart):
            seen.append(delta)
            if len(seen) == deltas:
                event.set()

        return callback, seen

    def stored_messages(self):
        return mister_fritz.app.get_state(
            {"configurable": {"thread_id": self.thread}}).values.get("messages", [])


class TestCancellingStopsTheModel(_Harness):
    def test_the_model_stops_generating(self):
        """The point of the feature, and what breaking out of agent.stream()
        alone does not do: close() waits for all 400 tokens."""
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 5)
        result = self.ask(self.real_agent(), streaming_callback=callback, cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertLess(self.record.produced, 100, "the model ran on after the cancel")
        self.assertTrue(self.record.closed, "the model's stream was never torn down")

    def test_nothing_more_reaches_the_person_after_the_cancel(self):
        event = threading.Event()
        callback, seen = self.cancel_after(event, 5)
        self.ask(self.real_agent(), streaming_callback=callback, cancel_event=event)
        self.assertEqual(len(seen), 5)

    def test_with_nobody_streaming_the_reply_so_far_is_still_kept(self):
        """The stop itself would happen anyway (ChatOllama reports tokens on a
        non-streaming call too); "messages" mode is what keeps the words."""
        event = threading.Event()
        model = _SlowModel(record=self.record, cancel_at=10, cancel_event=event)
        result = self.ask(self.real_agent(model), cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertLess(self.record.produced, 100)
        # The token in flight when the event lands may or may not be kept.
        self.assertTrue(result["text"].startswith("w0 w1 w2 w3 w4"), result["text"])

    def test_a_turn_cancelled_before_the_model_starts_never_calls_it(self):
        event = threading.Event()
        event.set()
        result = self.ask(self.real_agent(), cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertEqual(self.record.calls, 0)
        self.assertEqual(result["text"], INTERRUPTED_MARK.strip())

    def test_a_cancel_just_before_the_model_starts_never_reaches_it(self):
        """The narrow window the consumer's own check cannot see: it has read
        the last frame, and the model has not been called yet. Without the
        on_chat_model_start check the request goes to Ollama anyway, and a
        cold model holds the turn until its first token."""
        event = threading.Event()
        real = self.real_agent()

        class CancelsBetweenFrames:
            def stream(self, *args, **kwargs):
                frames = real.stream(*args, **kwargs)
                yield next(frames)      # the input echo; the consumer's check passes
                event.set()
                yield from frames

        result = self.ask(CancelsBetweenFrames(), streaming_callback=lambda *a: None,
                          cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertEqual(self.record.calls, 0, "the model was asked after the cancel")

    def test_a_cancel_during_a_tool_stops_the_next_model_call(self):
        """A running tool cannot be stopped, but nothing should follow it."""
        event = threading.Event()

        @tool
        def slow_lookup() -> str:
            """Look something up."""
            event.set()
            return "found it"

        agent = self.real_agent(_NarratesThenCallsATool(record=self.record), tools=[slow_lookup])
        result = self.ask(agent, streaming_callback=lambda *a: None, cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertEqual(self.record.calls, 1, "the model was called again after the cancel")
        self.assertIn("Let me look.", result["text"])
        self.assertTrue(result["text"].endswith(INTERRUPTED_MARK))

    def test_a_deliberate_stop_is_not_logged_as_an_error(self):
        """langchain_core logs every exception a callback raises at WARNING."""
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 5)
        with self.assertNoLogs("langchain_core.callbacks.manager", level="WARNING"):
            self.ask(self.real_agent(), streaming_callback=callback, cancel_event=event)

    def test_other_callback_errors_are_still_logged(self):
        record = logging.LogRecord("langchain_core.callbacks.manager", logging.WARNING, __file__, 1,
                                   "Error in %s.%s callback: %s",
                                   ("SomeTracer", "on_llm_end", "ValueError()"), None)
        self.assertTrue(mister_fritz._QuietDeliberateStops().filter(record))


class TestWhatAnInterruptedTurnLeavesBehind(_Harness):
    def _interrupted(self):
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 5)
        return self.ask(self.real_agent(), streaming_callback=callback, cancel_event=event)

    def test_the_partial_reply_is_checkpointed_with_the_mark(self):
        result = self._interrupted()
        last = self.stored_messages()[-1]
        self.assertIsInstance(last, AIMessage)
        self.assertTrue(last.content.startswith("w0 w1 w2 w3 w4"), last.content)
        self.assertTrue(last.content.endswith(INTERRUPTED_MARK))
        self.assertNotIn("  [interrupted]", last.content)
        self.assertEqual(result["text"], last.content)

    def test_what_the_person_said_is_kept(self):
        self._interrupted()
        self.assertEqual(self.stored_messages()[-2].content, "stop me")

    def test_half_a_reply_teaches_fritz_nothing(self):
        self._interrupted()
        self.extract.assert_not_called()

    def test_the_thread_is_usable_straight_afterwards(self):
        """A half-finished turn leaves next=("executor",) for the following
        turn to trip over; this one has to commit normally."""
        self._interrupted()
        self.assertEqual(mister_fritz.app.get_state(
            {"configurable": {"thread_id": self.thread}}).next, ())
        result = self.ask(self.real_agent(_SlowModel(record=_Record(), total=3)),
                          streaming_callback=lambda *a: None)
        self.assertFalse(result["interrupted"])
        self.assertEqual(result["text"], "w0 w1 w2 ")


class TestATurnCancelledWhileQueued(_Harness):
    """It holds one of the thread's two places while it waits. Left there, the
    person's next message would be refused as a third, after they had stopped
    both turns ahead of it."""

    def test_it_gives_its_place_up_and_never_runs(self):
        hold, holding = threading.Event(), threading.Event()

        def turn_ahead():
            with mister_fritz.one_turn_at_a_time(self.thread):
                holding.set()
                hold.wait(10)

        ahead = threading.Thread(target=turn_ahead, daemon=True)
        ahead.start()
        self.addCleanup(hold.set)
        self.assertTrue(holding.wait(5))

        event = threading.Event()
        threading.Timer(0.3, event.set).start()
        result = self.ask(self.real_agent(), streaming_callback=lambda *a: None, cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["text"], "")
        self.assertEqual(self.record.calls, 0)
        self.assertEqual(mister_fritz._TURN_GATES[self.thread].participants, 1,
                         "the cancelled turn kept its place in the queue")

        hold.set()
        ahead.join(5)
        self.assertNotIn(self.thread, mister_fritz._TURN_GATES)
        self.assertEqual(self.stored_messages(), [], "a turn that never ran left a record")


class TestWithoutACancel(_Harness):
    def test_no_event_changes_nothing(self):
        agent = self.real_agent(_SlowModel(record=self.record, total=3))
        result = self.ask(agent, streaming_callback=lambda *a: None)
        self.assertFalse(result["interrupted"])
        self.assertEqual(result["text"], "w0 w1 w2 ")
        self.extract.assert_called_once()

    def test_an_event_nobody_sets_changes_nothing(self):
        agent = self.real_agent(_SlowModel(record=self.record, total=3))
        result = self.ask(agent, cancel_event=threading.Event())
        self.assertFalse(result["interrupted"])
        self.assertEqual(result["text"], "w0 w1 w2 ")
        self.assertEqual(self.record.produced, 3)

    def test_a_refused_turn_says_it_was_not_interrupted(self):
        """The ThreadBusy path returns the same keys as the normal one."""
        with patch.object(mister_fritz, "one_turn_at_a_time",
                          side_effect=mister_fritz.ThreadBusy(self.thread)):
            result = self.ask(MagicMock())
        self.assertIs(result["interrupted"], False)


class _FakeAgent:
    """The plan's fake: yields scripted chunks and records being closed."""

    def __init__(self, ids=("seg",), per_segment=25):
        self.ids, self.per_segment = ids, per_segment
        self.consumed = 0
        self.closed = False

    def stream(self, inputs, config=None, stream_mode=None):
        try:
            for seg in self.ids:
                for _ in range(self.per_segment):
                    self.consumed += 1
                    time.sleep(0.01)
                    yield "messages", (AIMessageChunk(content=seg[0], id=seg), {})
        except GeneratorExit:
            self.closed = True
            raise


class TestTheFakeAgentPath(_Harness):
    """What a consumer-side break does: stop forwarding, and close the stream."""

    def test_a_cancel_closes_the_stream_before_the_reply_is_built(self):
        """Sampled while executor is still running. Read after ask_stuff has
        returned, `closed` would be True anyway: CPython finalises the
        generator when executor's frame goes, close() or no close()."""
        agent = _FakeAgent()
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 5)
        closed_when_counted = []

        def increment(name, value=1):
            if name == "turn_cancelled":
                closed_when_counted.append(agent.closed)

        with patch.object(mister_fritz.METRICS, "increment", side_effect=increment):
            result = self.ask(agent, streaming_callback=callback, cancel_event=event)
        self.assertLess(agent.consumed, 25)
        self.assertEqual(closed_when_counted, [True], "the stream was still open when the reply was built")
        self.assertTrue(result["interrupted"])
        self.assertTrue(result["text"].endswith(INTERRUPTED_MARK))

    def test_every_segment_is_kept_with_a_break_between_them(self):
        """_DeltaEmitter's `accumulated` covers only the current segment, so
        this cannot be read off it."""
        agent = _FakeAgent(ids=("aseg", "bseg"), per_segment=2)
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 3)
        result = self.ask(agent, streaming_callback=callback, cancel_event=event)
        self.assertEqual(result["text"], "aa\n\nb" + INTERRUPTED_MARK)


class _SlowNDJSONOllama(BaseHTTPRequestHandler):
    """/api/chat that streams `TOTAL` chunks slowly and records a hang-up."""

    protocol_version = "HTTP/1.1"
    TOTAL = 200
    state: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        st = type(self).state
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            b = (json.dumps(obj) + "\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

        try:
            for i in range(self.TOTAL):
                send({"model": "m", "created_at": "2026-01-01T00:00:00Z",
                      "message": {"role": "assistant", "content": f"w{i} "}, "done": False})
                st["written"] += 1
                time.sleep(0.02)
            send({"model": "m", "created_at": "2026-01-01T00:00:00Z",
                  "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            st["finished"] = True
        except OSError:
            st["hung_up_at"] = st["written"]
            st["hung_up"].set()
            self.close_connection = True


class TestARealChatOllama(_Harness):
    """The claim the fake model cannot make: that the HTTP request to Ollama is
    closed, so a dependency upgrade that kept it open would show here."""

    def test_its_request_is_closed_part_way(self):
        state = {"written": 0, "finished": False, "hung_up_at": None, "hung_up": threading.Event()}
        handler = type("Handler", (_SlowNDJSONOllama,), {"state": state})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        model = ChatOllama(model="m", base_url=f"http://127.0.0.1:{server.server_address[1]}")
        event = threading.Event()
        callback, _seen = self.cancel_after(event, 5)
        result = self.ask(self.real_agent(model), streaming_callback=callback, cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertTrue(state["hung_up"].wait(5), "Ollama never saw the request close")
        self.assertFalse(state["finished"])
        self.assertLess(state["hung_up_at"], _SlowNDJSONOllama.TOTAL // 2)


class TestAWebClientHangingUp(unittest.TestCase):
    """Through the real middleware stack, on a real socket. Measured while
    building this: request.is_disconnected() never returned True here, and a
    finally inside the generator could wait for a garbage collection."""

    def setUp(self):
        self.audited = threading.Event()
        self.audit = MagicMock(side_effect=lambda *a, **k: self.audited.set())
        for target, name, value in (
                (admin_panel, "_chat_identity", lambda request: ("web-canceltest", "canceltest")),
                (admin_panel, "audit_log", self.audit)):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def serve(self, fake_ask_stuff):
        import uvicorn
        p = patch.object(mister_fritz, "ask_stuff", fake_ask_stuff)
        p.start()
        self.addCleanup(p.stop)
        app = admin_panel.create_app("pw", chat_password="chatpw")
        # ws="none": no WebSocket here, and it keeps websockets.legacy's
        # deprecation out of this test.
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", access_log=False, ws="none"))
        thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
        thread.start()

        def stop():
            server.should_exit = True
            thread.join(10)

        self.addCleanup(stop)
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(server.started)
        return server.servers[0].sockets[0].getsockname()[1]

    def post(self, port):
        body = b"message=hello"
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.addCleanup(s.close)
        s.sendall(b"POST /chat/stream HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/x-www-form-urlencoded\r\n"
                  b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        return s

    def read_until(self, sock, marker, timeout=10):
        received = b""
        deadline = time.monotonic() + timeout
        while marker not in received and time.monotonic() < deadline:
            received += sock.recv(4096)
        return received

    def test_hanging_up_cancels_the_turn_and_nothing_before_it_does(self):
        events = []

        def fake_ask_stuff(message, source, user, *, streaming_callback=None, cancel_event=None, **_):
            events.append(cancel_event)
            streaming_callback("hello", "hello", True)
            interrupted = cancel_event.wait(10)
            return {"text": "hello [interrupted]", "image_paths": [], "interrupted": interrupted}

        sock = self.post(self.serve(fake_ask_stuff))
        self.assertIn(b"event: token", self.read_until(sock, b"event: token"))
        # Connected, a frame delivered, several empty polls of the queue: none
        # of that is a reason to stop.
        self.assertFalse(events[0].wait(0.3), "the turn was cancelled while the client was still there")
        sock.close()
        self.assertTrue(events[0].wait(5), "hanging up did not set cancel_event")
        self.assertTrue(self.audited.wait(15), "the turn never finished")
        self.assertEqual(self.audit.call_args.kwargs["result"], "interrupted")

    def test_a_hang_up_under_a_flood_of_tokens_is_not_left_to_the_collector(self):
        """When the hang-up lands in `await send`, a finally inside the
        generator runs only once a cyclic GC finalises it. With the collector
        off, a cancel that depends on it never comes."""
        events, flooding = [], threading.Event()

        def fake_ask_stuff(message, source, user, *, streaming_callback=None, cancel_event=None, **_):
            events.append(cancel_event)
            for i in range(200_000):
                if cancel_event.is_set():
                    break
                streaming_callback("x", "", i == 0)
                if i == 50:
                    flooding.set()
                time.sleep(0.0005)
            return {"text": "x [interrupted]", "image_paths": [], "interrupted": cancel_event.is_set()}

        sock = self.post(self.serve(fake_ask_stuff))
        # If the assertion below fails, stop the flood rather than let it run
        # on into later tests.
        self.addCleanup(lambda: events and events[0].set())
        self.assertTrue(flooding.wait(10))
        self.read_until(sock, b"event: token")
        gc.disable()
        self.addCleanup(gc.enable)
        sock.close()
        self.assertTrue(events[0].wait(3), "the cancel waited for a garbage collection")

    def test_a_turn_that_finishes_is_not_cancelled_or_called_interrupted(self):
        outcome = {}

        def fake_ask_stuff(message, source, user, *, streaming_callback=None, cancel_event=None, **_):
            streaming_callback("hello", "hello", True)
            # Long enough for the client to read the frame and for the SSE loop
            # to poll an empty queue a few times.
            time.sleep(0.3)
            outcome["set"] = cancel_event.is_set()
            return {"text": "hello", "image_paths": [], "interrupted": False}

        sock = self.post(self.serve(fake_ask_stuff))
        self.assertIn(b"event: done", self.read_until(sock, b"event: done"))
        self.assertTrue(self.audited.wait(15))
        self.assertIs(outcome["set"], False)
        self.assertEqual(self.audit.call_args.kwargs["result"], "ok")


if __name__ == "__main__":
    unittest.main()
