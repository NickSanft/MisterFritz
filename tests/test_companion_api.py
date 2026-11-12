"""
The desktop companion's /companion WebSocket (plan 14, PR 1): companion_api.

Most tests drive the route through Starlette's TestClient, with ask_stuff
replaced by FakeAskStuff. TestOverARealSocket serves the same app from
uvicorn, configured as the admin panel configures it, and talks to it with the
websockets client, because the TestClient never goes through uvicorn's
WebSocket protocol at all.

Also here, from P5's scaffold:

- FakeAskStuff is held to ask_stuff's real signature, so a test that passes
  against the fake cannot be calling ask_stuff in a way the real one refuses.
- The premise the design rests on: the admin panel's middleware stops HTTP
  requests only, so a WebSocket on the same app is reached with no Basic auth
  and no same-origin header. That is why COMPANION_TOKEN is the only gate.
"""
import base64
import contextlib
import inspect
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import ANY, MagicMock, patch

from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

import admin_panel
import companion_api
import fritz_utils
import identity_store
import mister_fritz
from fritz_utils import MessageSource

PANEL_PASSWORD = "panel-pw"
TOKEN = "t0k-" + "x" * 40


class FakeAskStuff:
    """Stands in for mister_fritz.ask_stuff in the companion's tests.

    Streams `deltas` through streaming_callback (restart on the first, as
    _DeltaEmitter does at a new segment), reports each `progress` line, and
    stops between deltas once cancel_event is set, returning what the real one
    returns: the reply so far with INTERRUPTED_MARK, and interrupted=True.
    `gate`, if given, is waited on after `pause_after` deltas, so a test can
    act while a turn is in flight. `fail`, if given, is raised instead.
    """

    def __init__(self, deltas=("Good evening", ", sir."), progress=(), images=(),
                 gate: threading.Event | None = None, pause_after: int = 1,
                 fail: BaseException | None = None):
        self.deltas, self.progress, self.images = list(deltas), list(progress), list(images)
        self.gate, self.pause_after, self.fail = gate, pause_after, fail
        self.calls: list[dict] = []
        self.in_flight = threading.Event()

    def __call__(self, base_prompt, source, user_id, progress_callback=None, streaming_callback=None,
                 user_image_paths=None, workspace_root=None, channel_id=None, schedule_manager=None, *,
                 thread_id=None, display_name=None, channel_key=None, may_wait=False, cancel_event=None):
        self.calls.append({k: v for k, v in locals().items() if k != "self"})
        if self.fail is not None:
            raise self.fail
        for line in self.progress:
            if progress_callback:
                progress_callback(line)
        said = ""
        for i, delta in enumerate(self.deltas):
            if cancel_event is not None and cancel_event.is_set():
                return self._result(said, interrupted=True)
            said += delta
            if streaming_callback:
                streaming_callback(delta, said, i == 0)
            if i + 1 == self.pause_after:
                self.in_flight.set()
                if self.gate is not None:
                    self.gate.wait(10)
        return self._result(said, interrupted=False)

    def _result(self, said, interrupted):
        text = f"{said.rstrip()}{mister_fritz.INTERRUPTED_MARK}".strip() if interrupted else said
        return {"text": text, "image_paths": list(self.images), "timestamp": "now",
                "interrupted": interrupted}


def companion_app(**knobs):
    """The admin panel's app with the companion knobs set as these tests need.

    Read at call time by companion_api and create_app, like is_admin reads its
    knobs, so patching fritz_utils is enough: no reload. A context manager
    yielding the app, so the patches end with the test.
    """
    values = {"COMPANION_ENABLED": True, "COMPANION_TOKEN": TOKEN, "COMPANION_MAX_CLIENTS": 2}
    values.update(knobs)
    stack = patch.multiple(fritz_utils, **values)

    class _Ctx:
        def __enter__(self):
            stack.start()
            return admin_panel.create_app(PANEL_PASSWORD, schedule_manager=None)

        def __exit__(self, *exc):
            stack.stop()

    return _Ctx()


def hello(**changes) -> dict:
    frame = {"type": "hello", "protocol": 1, "token": TOKEN,
             "client": {"id": "nick-pc", "name": "Nick's PC", "caps": ["text", "vrm"]}}
    frame.update(changes)
    return frame


def receive(ws, timeout: float = 10.0) -> dict:
    """ws.receive_json(), but a failure after `timeout` rather than a hang.

    A close from the server raises WebSocketDisconnect, as receive_json does.
    """
    box: dict = {}

    def run():
        try:
            box["frame"] = ws.receive_json()
        except BaseException as e:   # noqa: BLE001 - handed back to the test
            box["error"] = e

    reader = threading.Thread(target=run, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        raise AssertionError(f"nothing from the server within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["frame"]


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


class CompanionTest(unittest.TestCase):
    """Builds the app, connects desks, and cleans up after itself."""

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.audit = MagicMock()
        self.stack.enter_context(patch.object(companion_api, "audit_log", self.audit))
        # A store of names per test: identity_store caches names process-wide,
        # so otherwise a name an earlier test recorded could pass this one.
        names = self.stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        for name, value in (("SCHEDULE_DB", os.path.join(names, "names.db")),
                            ("_INITIALISED", False), ("_NAME_CACHE", {})):
            self.stack.enter_context(patch.object(identity_store, name, value))
        self.use(FakeAskStuff())

    def use(self, fake: FakeAskStuff) -> FakeAskStuff:
        self.fake = fake
        self.stack.enter_context(patch.object(mister_fritz, "ask_stuff", fake))
        if fake.gate is not None:
            # A gate left shut would hold a pool thread for its 10 s timeout.
            self.addCleanup(fake.gate.set)
        return fake

    def client(self, **knobs) -> TestClient:
        return TestClient(self.stack.enter_context(companion_app(**knobs)))

    def open(self, client: TestClient, **headers):
        return self.stack.enter_context(client.websocket_connect("/companion", headers=dict(headers)))

    def desk(self, via: TestClient | None = None, **hello_changes):
        """A connected desk past its welcome: (ws, welcome)."""
        ws = self.open(via or self.client())
        ws.send_json(hello(**hello_changes))
        welcome = receive(ws)
        self.assertEqual(welcome["type"], "welcome", welcome)
        return ws, welcome

    def closed_with(self, ws, timeout: float = 10.0) -> int:
        """The close code the server sends next; fails on any other frame."""
        with self.assertRaises(WebSocketDisconnect) as caught:
            frame = receive(ws, timeout)
            self.fail(f"expected a close, got {frame}")
        return caught.exception.code

    def turn(self, ws, turn_id="t1", text="good evening") -> list[dict]:
        """Says `text` and returns every frame up to its `state idle`."""
        ws.send_json({"type": "say", "turn_id": turn_id, "text": text})
        return self.until_idle(ws, turn_id)

    def until_idle(self, ws, turn_id) -> list[dict]:
        frames = []
        while not frames or frames[-1] != {"type": "state", "turn_id": turn_id, "state": "idle"}:
            frames.append(receive(ws))
        return frames


class TestTheFakeKeepsFaithWithAskStuff(unittest.TestCase):
    def test_it_takes_exactly_what_ask_stuff_takes(self):
        """Names, order, kinds and defaults. If ask_stuff changes, a fake that
        did not follow would let these tests call it in ways the real one
        would refuse, or miss a new keyword the route should pass."""
        def shape(fn):
            return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

        self.assertEqual(shape(FakeAskStuff.__call__)[1:], shape(mister_fritz.ask_stuff))

    def test_it_streams_deltas_with_restart_on_the_first(self):
        seen, notices = [], []
        fake = FakeAskStuff(deltas=("a", "b", "c"), progress=("Making enquiries",))
        result = fake("hi", MessageSource.DESKTOP_TEXT, "desktop-pc",
                      streaming_callback=lambda d, acc, r: seen.append((d, acc, r)),
                      progress_callback=notices.append)
        self.assertEqual(seen, [("a", "a", True), ("b", "ab", False), ("c", "abc", False)])
        self.assertEqual(notices, ["Making enquiries"])
        self.assertEqual(result, {"text": "abc", "image_paths": [], "timestamp": "now",
                                  "interrupted": False})

    def test_it_stops_between_deltas_when_cancelled(self):
        cancel, gate = threading.Event(), threading.Event()
        fake = FakeAskStuff(deltas=("one ", "two ", "three"), gate=gate, pause_after=1)
        box = {}
        t = threading.Thread(target=lambda: box.update(result=fake(
            "hi", MessageSource.DESKTOP_TEXT, "desktop-pc", cancel_event=cancel)))
        t.start()
        self.assertTrue(fake.in_flight.wait(5))
        cancel.set()
        gate.set()
        t.join(5)
        self.assertEqual(box["result"]["text"], "one" + mister_fritz.INTERRUPTED_MARK)
        self.assertIs(box["result"]["interrupted"], True)


class TestTheTokenWillBeTheOnlyGate(unittest.TestCase):
    """Basic auth, the CSP header and the same-origin check are all
    BaseHTTPMiddleware, which passes a non-HTTP scope straight through. So a
    WebSocket on the panel's app answers a client with no password and from
    any origin, and COMPANION_TOKEN is all that stands in front of
    /companion. If a middleware change ever stopped WebSockets here, the
    design would need revisiting, and this is where it would show."""

    def _probe(self, headers):
        async def echo(ws: WebSocket):
            await ws.accept()
            await ws.send_text("through")
            await ws.close()

        with companion_app() as app:
            app.router.routes.append(WebSocketRoute("/probe", echo))
            with TestClient(app).websocket_connect("/probe", headers=headers) as ws:
                return ws.receive_text()

    def test_no_password_and_no_origin_header(self):
        self.assertEqual(self._probe({}), "through")

    def test_a_cross_site_origin(self):
        """The HTTP side refuses this for a POST; a WebSocket never sees it."""
        self.assertEqual(self._probe({"sec-fetch-site": "cross-site", "origin": "https://evil.example"}),
                         "through")

    def test_while_the_same_app_still_guards_http(self):
        with companion_app() as app:
            client = TestClient(app)
            self.assertEqual(client.get("/").status_code, 401)
            self.assertEqual(client.get("/", auth=("anyone", PANEL_PASSWORD)).status_code, 200)


class TestTheRouteIsServedOnlyWhenSwitchedOn(CompanionTest):
    def test_off_or_without_a_token_there_is_no_route(self):
        for knobs in ({"COMPANION_ENABLED": False}, {"COMPANION_TOKEN": None}, {"COMPANION_TOKEN": ""}):
            with self.subTest(**knobs), companion_app(**knobs) as app:
                with self.assertRaises(WebSocketDisconnect):
                    with TestClient(app).websocket_connect("/companion"):
                        pass

    def test_on_with_a_token_it_is(self):
        ws, welcome = self.desk()
        self.assertEqual(welcome["protocol"], 1)


class TestHandshake(CompanionTest):
    def test_the_welcome_names_the_desk_and_its_own_thread(self):
        _ws, welcome = self.desk()
        self.assertEqual(welcome, {
            "type": "welcome", "protocol": 1, "identity": "desktop-nick-pc",
            "display_name": "Nick's PC", "server_caps": ["text"],
            "thread_id": "desktop-nick-pc#desktop",
        })
        self.audit.assert_any_call("companion_connect", result="ok", user_id="desktop-nick-pc",
                                   client="desktop-nick-pc", peer=ANY)

    def test_a_linked_desk_is_the_linked_identity_and_keeps_its_name(self):
        """IDENTITY_LINKS is parsed at import, so the test patches the dict.
        The machine's name goes on the desk's own row, never on the person's."""
        identity_store.record("discord-1", "Nick", "discord")
        with patch.dict(fritz_utils.IDENTITY_LINKS, {"desktop-nick-pc": "discord-1"}):
            ws, welcome = self.desk()
            self.assertEqual((welcome["identity"], welcome["display_name"], welcome["thread_id"]),
                             ("discord-1", "Nick", "discord-1#desktop"))
            self.turn(ws)
        self.assertEqual(identity_store.display_name("discord-1"), "Nick")
        self.assertEqual(identity_store.display_name("desktop-nick-pc"), "Nick's PC")
        call = self.fake.calls[0]
        # ask_stuff gets the unresolved id and follows the link itself, as it
        # does for every caller; the thread is the linked identity's desk.
        self.assertEqual((call["user_id"], call["thread_id"], call["display_name"]),
                         ("desktop-nick-pc", "discord-1#desktop", None))

    def test_a_wrong_token_is_closed_4401(self):
        for token in ("wrong", TOKEN[:-1], TOKEN + "x", "", "tökén-ünïcode"):
            with self.subTest(token=token):
                ws = self.open(self.client())
                ws.send_json(hello(token=token))
                self.assertEqual(self.closed_with(ws), 4401)
        self.audit.assert_any_call("companion_connect", result="bad_token", client="nick-pc", peer=ANY)

    def test_no_hello_in_time_is_closed_4408(self):
        with patch.object(companion_api, "HELLO_TIMEOUT_S", 0.2):
            ws = self.open(self.client())
            started = time.monotonic()
            self.assertEqual(self.closed_with(ws), 4408)
            self.assertLess(time.monotonic() - started, 5)

    def test_anything_but_a_hello_first_is_closed_4400(self):
        firsts = ["not json", json.dumps({"type": "say", "turn_id": "t1", "text": "hi"}),
                  json.dumps({"type": "nope"}), json.dumps({k: v for k, v in hello().items() if k != "client"}),
                  json.dumps(hello(client={"id": "Nick PC!", "name": "x", "caps": []})), b"\x00\x01"]
        for first in firsts:
            with self.subTest(first=first):
                ws = self.open(self.client())
                ws.send_bytes(first) if isinstance(first, bytes) else ws.send_text(first)
                self.assertEqual(self.closed_with(ws), 4400)

    def test_a_protocol_this_server_does_not_speak_is_closed_4426(self):
        ws = self.open(self.client())
        ws.send_json(hello(protocol=2))
        self.assertEqual(self.closed_with(ws), 4426)

    def test_the_token_is_checked_before_the_protocol(self):
        """A stranger learns nothing about this server, not even its version."""
        ws = self.open(self.client())
        ws.send_json(hello(protocol=2, token="wrong"))
        self.assertEqual(self.closed_with(ws), 4401)

    def test_one_more_than_the_maximum_is_closed_4409(self):
        """At values other than the default too, so a hard-coded 2 fails."""
        for most in (1, 2, 3):
            with self.subTest(most=most), contextlib.ExitStack() as desks:
                client = self.client(COMPANION_MAX_CLIENTS=most)
                admitted = []
                for n in range(most):
                    ws = desks.enter_context(client.websocket_connect("/companion"))
                    ws.send_json(hello(client={"id": f"desk-{n}", "name": "Desk", "caps": ["text"]}))
                    self.assertEqual(receive(ws)["type"], "welcome")
                    admitted.append(ws)
                one_more = desks.enter_context(client.websocket_connect("/companion"))
                one_more.send_json(hello())
                self.assertEqual(self.closed_with(one_more), 4409)
                # A desk that leaves gives its place back.
                admitted[0].close(1000)
                self.assertTrue(wait_for(lambda: len(companion_api._SESSIONS) == most - 1))
                again = desks.enter_context(client.websocket_connect("/companion"))
                again.send_json(hello())
                self.assertEqual(receive(again)["type"], "welcome")
            self.assertTrue(wait_for(lambda: not companion_api._SESSIONS))

    def test_a_newer_hello_this_version_cannot_read_is_still_4426(self):
        """A later protocol may add a cap v1's models refuse; its client
        should hear "not this version", not "bad frame"."""
        newer = hello(protocol=2, client={"id": "nick-pc", "name": "PC", "caps": ["text", "screen"]})
        ws = self.open(self.client())
        ws.send_json(newer)
        self.assertEqual(self.closed_with(ws), 4426)
        # Only with the right token, and only for another version.
        for frame in ({**newer, "token": "wrong"}, {**newer, "protocol": 1}, {**newer, "protocol": "2"}):
            with self.subTest(frame=frame):
                ws = self.open(self.client())
                ws.send_json(frame)
                self.assertEqual(self.closed_with(ws), 4400)

    def test_an_oversized_first_frame_is_refused_unread(self):
        """Not parsed as a frame, and not as JSON to look for a newer
        protocol either: the cap is there because parsing runs on the loop."""
        parse = MagicMock(side_effect=companion_api.parse_client_frame)
        skew = MagicMock(side_effect=companion_api._newer_hello)
        with patch.object(companion_api, "parse_client_frame", parse), \
             patch.object(companion_api, "_newer_hello", skew):
            for frame in (hello(), hello(protocol=2)):
                with self.subTest(protocol=frame["protocol"]):
                    ws = self.open(self.client())
                    ws.send_text(json.dumps(frame) + " " * companion_api.MAX_TEXT_FRAME_CHARS)
                    self.assertEqual(self.closed_with(ws), 4400)
        parse.assert_not_called()
        skew.assert_not_called()

    def test_a_bad_first_frame_cannot_write_lines_into_the_log(self):
        """pydantic quotes the client's `type` in its message."""
        forged = "x'\n2026-11-11 09:00:00,000 WARNING companion_api companion: discord-1 connected"
        with self.assertLogs("companion_api", level="INFO") as logs:
            ws = self.open(self.client())
            ws.send_json({"type": forged})
            self.assertEqual(self.closed_with(ws), 4400)
        self.assertTrue(logs.records)
        for record in logs.records:
            self.assertNotIn("\n", record.getMessage())

    def test_no_password_and_no_origin_needed(self):
        """The middleware premise, on /companion itself."""
        client = self.client()
        for headers in ({}, {"sec-fetch-site": "cross-site", "origin": "https://evil.example"}):
            with self.subTest(headers=headers):
                ws = self.open(client, **headers)
                ws.send_json(hello())
                self.assertEqual(receive(ws)["type"], "welcome")


class TestTurns(CompanionTest):
    def test_a_streamed_turn(self):
        self.use(FakeAskStuff(deltas=("Good evening", ", sir."), progress=("Making enquiries",)))
        ws, _ = self.desk()
        self.assertEqual(self.turn(ws, "7f3", "good evening"), [
            {"type": "state", "turn_id": "7f3", "state": "thinking"},
            {"type": "progress", "turn_id": "7f3", "text": "Making enquiries"},
            {"type": "token", "turn_id": "7f3", "delta": "Good evening", "restart": True},
            {"type": "token", "turn_id": "7f3", "delta": ", sir.", "restart": False},
            {"type": "done", "turn_id": "7f3", "text": "Good evening, sir.", "images": [],
             "interrupted": False},
            {"type": "state", "turn_id": "7f3", "state": "idle"},
        ])
        call = self.fake.calls[0]
        self.assertEqual((call["base_prompt"], call["source"], call["user_id"], call["thread_id"]),
                         ("good evening", MessageSource.DESKTOP_TEXT, "desktop-nick-pc",
                          "desktop-nick-pc#desktop"))
        self.assertIsNone(call["display_name"])
        self.assertIsInstance(call["cancel_event"], threading.Event)
        self.audit.assert_any_call("companion_turn", user_id="desktop-nick-pc", client="desktop-nick-pc",
                                   chars=12, result="ok", reply_chars=18, images=0, seconds=ANY)

    def test_what_is_said_is_stripped(self):
        ws, _ = self.desk()
        self.turn(ws, text="  good evening \n")
        self.assertEqual(self.fake.calls[0]["base_prompt"], "good evening")

    def test_one_turn_after_another(self):
        ws, _ = self.desk()
        for turn_id in ("a", "b", "c"):
            self.assertEqual(self.turn(ws, turn_id)[-2]["type"], "done")
        self.assertEqual(len(self.fake.calls), 3)

    def test_a_say_while_a_turn_runs_is_busy(self):
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        ws.send_json({"type": "say", "turn_id": "first", "text": "one"})
        self.assertEqual(receive(ws)["state"], "thinking")
        self.assertEqual(receive(ws)["delta"], "Good evening")
        ws.send_json({"type": "say", "turn_id": "second", "text": "two"})
        busy = receive(ws)
        self.assertEqual((busy["type"], busy["turn_id"], busy["code"]), ("error", "second", "busy"))
        gate.set()
        rest = self.until_idle(ws, "first")
        self.assertEqual([f["type"] for f in rest], ["token", "done", "state"])
        self.assertEqual(len(self.fake.calls), 1)

    def test_cancel_mid_stream(self):
        gate = threading.Event()
        self.use(FakeAskStuff(deltas=("one ", "two ", "three"), gate=gate))
        ws, _ = self.desk()
        ws.send_json({"type": "say", "turn_id": "t1", "text": "count"})
        self.assertEqual(receive(ws)["state"], "thinking")
        self.assertEqual(receive(ws)["delta"], "one ")
        ws.send_json({"type": "cancel", "turn_id": "t1"})
        # Frames are handled in order: the pong means the cancel was too.
        ws.send_json({"type": "ping", "t": 1})
        self.assertEqual(receive(ws), {"type": "pong", "t": 1})
        self.assertTrue(self.fake.calls[0]["cancel_event"].is_set())
        gate.set()
        done, idle = self.until_idle(ws, "t1")
        self.assertEqual(done, {"type": "done", "turn_id": "t1", "text": "one" + mister_fritz.INTERRUPTED_MARK,
                                "images": [], "interrupted": True})
        self.assertEqual(idle["state"], "idle")
        self.audit.assert_any_call("companion_turn", user_id=ANY, client=ANY, chars=5,
                                   result="interrupted", reply_chars=ANY, images=0, seconds=ANY)

    def test_the_turn_after_a_cancelled_one_runs_whole(self):
        """Each turn has its own cancel; one shared per session would make
        every turn after the first barge-in come back interrupted."""
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        ws.send_json({"type": "say", "turn_id": "t1", "text": "one"})
        receive(ws), receive(ws)
        ws.send_json({"type": "cancel", "turn_id": "t1"})
        ws.send_json({"type": "ping", "t": 4})
        self.assertEqual(receive(ws)["type"], "pong")
        gate.set()
        self.assertTrue(self.until_idle(ws, "t1")[-2]["interrupted"])
        self.assertFalse(self.turn(ws, "t2")[-2]["interrupted"])
        first, second = (call["cancel_event"] for call in self.fake.calls)
        self.assertIsNot(first, second)
        self.assertFalse(second.is_set())

    def barge_in(self, ws, *frames):
        """Says t1, waits for its first token, then sends `frames` in one go."""
        ws.send_json({"type": "say", "turn_id": "t1", "text": "one"})
        self.assertEqual(receive(ws)["state"], "thinking")
        self.assertEqual(receive(ws)["delta"], "Good evening")
        for frame in frames:
            ws.send_json(frame)
        ws.send_json({"type": "ping", "t": 5})
        frames_before_pong = []
        while (frame := receive(ws))["type"] != "pong":
            frames_before_pong.append(frame)
        return frames_before_pong

    def test_barge_in_cancel_and_say_in_one_go(self):
        """§ 2.2: barge-in is cancel plus a new turn, sent together. The new
        say waits for the cancelled turn to end, then runs."""
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        self.assertEqual(self.barge_in(ws, {"type": "cancel", "turn_id": "t1"},
                                       {"type": "say", "turn_id": "t2", "text": "two"}), [])
        gate.set()
        frames = self.until_idle(ws, "t1") + self.until_idle(ws, "t2")
        self.assertEqual([(f["type"], f["turn_id"]) for f in frames], [
            ("done", "t1"), ("state", "t1"), ("state", "t2"), ("token", "t2"), ("token", "t2"),
            ("done", "t2"), ("state", "t2")])
        self.assertTrue(frames[0]["interrupted"])
        self.assertEqual(frames[2]["state"], "thinking")
        self.assertEqual((frames[5]["text"], frames[5]["interrupted"]), ("Good evening, sir.", False))
        self.assertEqual([c["base_prompt"] for c in self.fake.calls], ["one", "two"])

    def test_only_one_say_waits_behind_a_cancel(self):
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        early = self.barge_in(ws, {"type": "cancel", "turn_id": "t1"},
                              {"type": "say", "turn_id": "t2", "text": "two"},
                              {"type": "say", "turn_id": "t3", "text": "three"})
        self.assertEqual([(f["type"], f["turn_id"], f["code"]) for f in early], [("error", "t3", "busy")])
        gate.set()
        self.until_idle(ws, "t1"), self.until_idle(ws, "t2")
        self.assertEqual([c["base_prompt"] for c in self.fake.calls], ["one", "two"])

    def test_a_say_waiting_behind_a_cancel_can_be_cancelled(self):
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        early = self.barge_in(ws, {"type": "cancel", "turn_id": "t1"},
                              {"type": "say", "turn_id": "t2", "text": "two"},
                              {"type": "cancel", "turn_id": "t2"})
        self.assertEqual(early, [{"type": "done", "turn_id": "t2", "text": "", "images": [], "interrupted": True},
                                 {"type": "state", "turn_id": "t2", "state": "idle"}])
        gate.set()
        self.assertTrue(self.until_idle(ws, "t1")[-2]["interrupted"])
        ws.send_json({"type": "ping", "t": 6})
        self.assertEqual(receive(ws)["type"], "pong")
        self.assertEqual([c["base_prompt"] for c in self.fake.calls], ["one"])

    def test_a_say_without_a_cancel_before_it_is_still_busy(self):
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        early = self.barge_in(ws, {"type": "say", "turn_id": "t2", "text": "two"},
                              {"type": "cancel", "turn_id": "t1"})
        self.assertEqual([(f["type"], f["turn_id"], f["code"]) for f in early], [("error", "t2", "busy")])
        gate.set()
        self.until_idle(ws, "t1")
        ws.send_json({"type": "ping", "t": 7})
        self.assertEqual(receive(ws)["type"], "pong")
        self.assertEqual(len(self.fake.calls), 1)

    def test_a_desk_that_goes_drops_the_say_waiting_behind_a_cancel(self):
        gate = threading.Event()
        fake = self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        self.barge_in(ws, {"type": "cancel", "turn_id": "t1"}, {"type": "say", "turn_id": "t2", "text": "two"})
        ws.close(1000)
        self.assertTrue(wait_for(lambda: not companion_api._SESSIONS))
        gate.set()
        self.assertFalse(wait_for(lambda: len(fake.calls) > 1, timeout=1.0), "the waiting say ran for nobody")
        self.assertEqual([c["base_prompt"] for c in fake.calls], ["one"])

    def test_a_turn_cancelled_before_it_reaches_the_model_never_does(self):
        """Its pool thread came free only after the cancel, or after its desk
        left: it is not handed to ask_stuff, so nothing is recorded."""
        turn = companion_api._Turn("t1")
        turn.cancel.set()
        session = type("S", (), {"identity": "desktop-pc", "canonical_id": "desktop-pc",
                                 "thread_id": "desktop-pc#desktop", "send_from_thread": print})()
        self.assertEqual(companion_api._ask(session, turn, "hello", MessageSource.DESKTOP_TEXT), ("", [], True))
        self.assertEqual(self.fake.calls, [])
        self.audit.assert_called_once_with("companion_turn", user_id="desktop-pc", client="desktop-pc",
                                           chars=5, result="cancelled_before_start")

    def test_a_cancel_for_another_turn_is_ignored(self):
        gate = threading.Event()
        self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        ws.send_json({"type": "say", "turn_id": "t1", "text": "hello"})
        receive(ws), receive(ws)
        for other in ("t0", "t2"):
            ws.send_json({"type": "cancel", "turn_id": other})
        ws.send_json({"type": "ping", "t": 2})
        self.assertEqual(receive(ws)["type"], "pong")
        self.assertFalse(self.fake.calls[0]["cancel_event"].is_set())
        gate.set()
        self.assertFalse(self.until_idle(ws, "t1")[-2]["interrupted"])

    def test_a_cancel_after_the_turn_is_over_is_ignored(self):
        ws, _ = self.desk()
        self.turn(ws, "t1")
        ws.send_json({"type": "cancel", "turn_id": "t1"})
        ws.send_json({"type": "ping", "t": 3})
        self.assertEqual(receive(ws), {"type": "pong", "t": 3})

    def test_a_failed_turn_is_an_error_in_fritz_s_words(self):
        self.use(FakeAskStuff(fail=RuntimeError(r"secret detail at C:\private\path")))
        ws, _ = self.desk()
        with self.assertLogs("bot_adapters", level="ERROR"):
            frames = self.turn(ws, "t1")
        self.assertEqual([f["type"] for f in frames], ["state", "error", "state"])
        error = frames[1]
        self.assertEqual((error["turn_id"], error["code"]), ("t1", "turn_failed"))
        self.assertIn("ref", error["text"])
        self.assertNotIn("secret", error["text"])
        self.audit.assert_any_call("companion_turn", user_id=ANY, client=ANY, chars=12, result="error",
                                   error=r"secret detail at C:\private\path")
        # The session goes on.
        self.use(FakeAskStuff())
        self.assertEqual(self.turn(ws, "t2")[-2]["type"], "done")

    def test_a_desk_that_goes_stops_its_turn(self):
        gate = threading.Event()
        fake = self.use(FakeAskStuff(gate=gate))
        ws, _ = self.desk()
        ws.send_json({"type": "say", "turn_id": "t1", "text": "hello"})
        self.assertTrue(fake.in_flight.wait(5))
        ws.close(1000)
        self.assertTrue(wait_for(lambda: fake.calls[0]["cancel_event"].is_set()),
                        "the model was left generating for nobody")
        self.assertTrue(wait_for(lambda: not companion_api._SESSIONS))

    def test_a_turn_s_images_arrive_as_data_urls(self):
        with tempfile.TemporaryDirectory() as parent, tempfile.TemporaryDirectory() as elsewhere:
            png = b"\x89PNG\r\n\x1a\n" + b"pixels"
            root = os.path.join(parent, "output")
            shown = os.path.join(root, "gen", "cat.png")
            # A sibling whose name starts with the root's: inside only by
            # string prefix, which the check must not mistake for inside.
            sibling = os.path.join(parent, "output_old", "cat.png")
            for path, data in ((shown, png), (os.path.join(root, "notes.txt"), b"text"),
                               (os.path.join(elsewhere, "secret.png"), png), (sibling, png)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
            self.use(FakeAskStuff(images=(shown, os.path.join(root, "notes.txt"),
                                          os.path.join(elsewhere, "secret.png"), sibling,
                                          os.path.join(root, "gone.png"))))
            with patch.object(admin_panel, "_chat_asset_roots", lambda: [os.path.abspath(root)]), \
                 self.assertLogs("companion_api", level="WARNING") as logs:
                ws, _ = self.desk()
                done = self.turn(ws)[-2]
        self.assertEqual(done["images"], ["data:image/png;base64," + base64.b64encode(png).decode()])
        self.assertEqual(len(logs.output), 4, logs.output)

    def test_an_image_too_large_is_left_out(self):
        with tempfile.TemporaryDirectory() as root:
            big = os.path.join(root, "big.png")
            with open(big, "wb") as f:
                f.write(b"x" * 101)
            self.use(FakeAskStuff(images=(big,)))
            with patch.object(admin_panel, "_chat_asset_roots", lambda: [os.path.abspath(root)]), \
                 patch.object(companion_api, "IMAGE_MAX_BYTES", 100), \
                 self.assertLogs("companion_api", level="WARNING"):
                ws, _ = self.desk()
                self.assertEqual(self.turn(ws)[-2]["images"], [])


class TestFrames(CompanionTest):
    def test_a_ping_is_answered_with_its_own_t(self):
        ws, _ = self.desk()
        ws.send_json({"type": "ping", "t": 1731370000123})
        self.assertEqual(receive(ws), {"type": "pong", "t": 1731370000123})

    def test_two_unanswered_pings_close_the_session_4000(self):
        with patch.object(companion_api, "PING_EVERY_S", 0.05):
            ws, _ = self.desk()
            self.assertEqual(receive(ws)["type"], "ping")
            self.assertEqual(receive(ws)["type"], "ping")
            self.assertEqual(self.closed_with(ws), 4000)

    def test_answered_pings_keep_it_open(self):
        with patch.object(companion_api, "PING_EVERY_S", 0.2):
            ws, _ = self.desk()
            for _ in range(4):
                ping = receive(ws)
                self.assertEqual(ping["type"], "ping")
                self.assertIsInstance(ping["t"], int)
                ws.send_json({"type": "pong", "t": ping["t"]})
            self.assertEqual(receive(ws)["type"], "ping")
            self.assertEqual(receive(ws)["type"], "ping")
            self.assertEqual(self.closed_with(ws), 4000)

    def test_a_bad_frame_is_answered_and_the_session_goes_on(self):
        ws, _ = self.desk()
        with self.assertLogs("companion_api", level="WARNING"):
            for raw in ("not json", json.dumps({"type": "nope"}), json.dumps([1, 2])):
                ws.send_text(raw)
                error = receive(ws)
                self.assertEqual((error["type"], error["code"], error["turn_id"]), ("error", "bad_frame", None))
                self.assertNotIn("nope", error["text"])
            # A say that names its turn is answered under that turn, which
            # the client is waiting on, though the frame itself is refused.
            for bad in ({"text": ""}, {"text": "   "}, {"text": "x" * 8001}, {}):
                ws.send_json({"type": "say", "turn_id": "t9", **bad})
                error = receive(ws)
                self.assertEqual((error["code"], error["turn_id"]), ("bad_frame", "t9"))
            ws.send_json({"type": "say", "turn_id": "not a turn id!", "text": "hi"})
            self.assertIsNone(receive(ws)["turn_id"])
        self.assertEqual(self.turn(ws)[-2]["type"], "done")
        self.assertEqual(len(self.fake.calls), 1)

    def test_an_oversized_frame_is_refused_unread_and_the_session_goes_on(self):
        ws, _ = self.desk()
        parse = MagicMock(side_effect=companion_api.parse_client_frame)
        with patch.object(companion_api, "parse_client_frame", parse), \
             self.assertLogs("companion_api", level="WARNING"):
            ws.send_text(json.dumps({"type": "say", "turn_id": "t1", "text": "x"})
                         + " " * companion_api.MAX_TEXT_FRAME_CHARS)
            error = receive(ws)
        self.assertEqual((error["code"], error["turn_id"]), ("bad_frame", "t1"))
        parse.assert_not_called()
        # The turn it names is found near either end, never by reading it all.
        pad = " " * companion_api.MAX_TEXT_FRAME_CHARS
        with self.assertLogs("companion_api", level="WARNING"):
            for raw, named in (('{"text":"x"' + pad + ',"turn_id":"t2","type":"say"}', "t2"),
                               ('{"type":"say","text":"' + "x" * 2000 + '","turn_id":"t3"' + pad + "}", None),
                               ('{"type":"say","turn_id":"bad id!"' + pad + "}", None)):
                ws.send_text(raw)
                self.assertEqual(receive(ws)["turn_id"], named)
        # The longest say there is still fits, however its client escapes it.
        longest = json.dumps({"type": "say", "turn_id": "t" * 64, "text": "\U0001F600" * 8000}, ensure_ascii=True)
        self.assertLess(len(longest), companion_api.MAX_TEXT_FRAME_CHARS)
        self.assertEqual(self.turn(ws)[-2]["type"], "done")

    def test_a_bad_frame_mid_session_cannot_write_lines_into_the_log(self):
        ws, _ = self.desk()
        forged = "x'\n2026-11-11 09:00:00,000 WARNING companion_api companion: discord-1 connected"
        with self.assertLogs("companion_api", level="WARNING") as logs:
            ws.send_json({"type": forged})
            self.assertEqual(receive(ws)["code"], "bad_frame")
        for record in logs.records:
            self.assertNotIn("\n", record.getMessage())

    def test_a_second_hello_is_a_bad_frame(self):
        ws, _ = self.desk()
        with self.assertLogs("companion_api", level="WARNING"):
            ws.send_json(hello())
            self.assertEqual(receive(ws)["code"], "bad_frame")

    def test_audio_and_widgets_are_not_served_yet(self):
        ws, welcome = self.desk()
        self.assertEqual(welcome["server_caps"], ["text"])
        ws.send_bytes(b"\x00\x01" * 1000)      # dropped unread: no answer
        ws.send_json({"type": "audio_end", "turn_id": "a1"})
        error = receive(ws)
        self.assertEqual((error["type"], error["code"], error["turn_id"]), ("error", "bad_frame", "a1"))
        ws.send_json({"type": "subscribe", "widgets": ["clock"]})
        error = receive(ws)
        self.assertEqual((error["code"], error["turn_id"]), ("bad_frame", None))
        self.assertEqual(self.fake.calls, [])


class TestOverARealSocket(CompanionTest):
    """uvicorn, configured as start_admin_panel configures it, and a real
    WebSocket client. The TestClient never touches uvicorn's WebSocket
    protocol, so without this a broken one would pass every other test."""

    def serve(self, **knobs) -> str:
        import asyncio

        import uvicorn
        app = self.stack.enter_context(companion_app(**knobs))
        server = uvicorn.Server(admin_panel.server_config(app, 0))
        thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
        thread.start()

        def stop():
            server.should_exit = True
            thread.join(10)

        # On the stack, so the clients connected after this close first.
        self.stack.callback(stop)
        self.assertTrue(wait_for(lambda: server.started, 10))
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}/companion"

    def connect(self, url):
        from websockets.sync.client import connect
        return self.stack.enter_context(connect(url, open_timeout=10, close_timeout=5))

    def test_a_whole_turn(self):
        socket = self.connect(self.serve())
        socket.send(json.dumps(hello()))
        self.assertEqual(json.loads(socket.recv(timeout=10))["identity"], "desktop-nick-pc")
        socket.send(json.dumps({"type": "say", "turn_id": "t1", "text": "good evening"}))
        frames = []
        while not frames or frames[-1].get("state") != "idle":
            frames.append(json.loads(socket.recv(timeout=10)))
        self.assertEqual([f["type"] for f in frames], ["state", "token", "token", "done", "state"])
        self.assertEqual(frames[3]["text"], "Good evening, sir.")

    def test_the_close_codes_reach_the_client(self):
        from websockets.exceptions import ConnectionClosedError
        url = self.serve()
        socket = self.connect(url)
        socket.send(json.dumps(hello(token="wrong")))
        with self.assertRaises(ConnectionClosedError) as caught:
            socket.recv(timeout=10)
        self.assertEqual(caught.exception.rcvd.code, 4401)

    def test_a_frame_over_the_limit_is_closed_1009_and_nothing_is_compressed(self):
        """uvicorn's own default is 16 MiB, read whole before the token is
        seen; with deflate on, 16 KiB on the wire became 16 MiB to parse."""
        from websockets.exceptions import ConnectionClosedError
        url = self.serve()
        socket = self.connect(url)
        self.assertEqual(socket.protocol.extensions, [], "permessage-deflate was negotiated")
        socket.send(b"\x00" * (admin_panel.WS_MAX_FRAME_BYTES + 1))
        with self.assertRaises(ConnectionClosedError) as caught:
            socket.recv(timeout=10)
        self.assertEqual(caught.exception.rcvd.code, 1009)
        # A 30-second clip of 16 kHz mono PCM16 fits in one frame (PR 9).
        self.assertGreaterEqual(admin_panel.WS_MAX_FRAME_BYTES, 30 * 16000 * 2)

    def test_a_client_that_never_answers_a_ping_is_closed(self):
        """The sans-I/O protocol sends no keepalive of its own, so this
        heartbeat is the only thing that notices a desk that has gone quiet."""
        from websockets.exceptions import ConnectionClosedError
        with patch.object(companion_api, "PING_EVERY_S", 0.1):
            socket = self.connect(self.serve())
            socket.send(json.dumps(hello()))
            socket.recv(timeout=10)
            with self.assertRaises(ConnectionClosedError) as caught:
                while True:
                    self.assertEqual(json.loads(socket.recv(timeout=10))["type"], "ping")
        self.assertEqual(caught.exception.rcvd.code, 4000)
        self.assertTrue(wait_for(lambda: not companion_api._SESSIONS))


if __name__ == "__main__":
    unittest.main()
