"""
The desktop companion's /companion WebSocket: the scaffold, ahead of the route
(P5 of plans/14-desktop-companion-build.md).

PR 1 builds companion_api.py and its tests land here. Until then this file
holds what PR 1's tests stand on, each checked now:

- FakeAskStuff, the stand-in for mister_fritz.ask_stuff. It replays scripted
  deltas through streaming_callback and honours cancel_event as the real one
  does since P3. Its signature is held to the real one, so a test that passes
  against the fake cannot be calling ask_stuff in a way the real one refuses.
- The premise the design rests on: the admin panel's middleware stops HTTP
  requests only, so a WebSocket on the same app is reached with no Basic auth
  and no same-origin header. That is why COMPANION_TOKEN is the only gate.
- That /companion is not served yet, which is what the reserved-knob warning
  in fritz_utils claims. PR 1 replaces this test with the real handshake.

The plan's acceptance line for P5 ("closed with 4401 on a bad token") needs the
route, so it moves to PR 1, as its own "the first real test lands in PR 1" says.
"""
import inspect
import threading
import unittest
from unittest.mock import patch

from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

import admin_panel
import fritz_utils
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
    act while a turn is in flight.
    """

    def __init__(self, deltas=("Good evening", ", sir."), progress=(), images=(),
                 gate: threading.Event | None = None, pause_after: int = 1):
        self.deltas, self.progress, self.images = list(deltas), list(progress), list(images)
        self.gate, self.pause_after = gate, pause_after
        self.calls: list[dict] = []
        self.in_flight = threading.Event()

    def __call__(self, base_prompt, source, user_id, progress_callback=None, streaming_callback=None,
                 user_image_paths=None, workspace_root=None, channel_id=None, schedule_manager=None, *,
                 thread_id=None, display_name=None, channel_key=None, may_wait=False, cancel_event=None):
        self.calls.append({k: v for k, v in locals().items() if k != "self"})
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
    """The admin panel's app with the companion knobs set as PR 1's tests need.

    Read at call time by companion_api, like is_admin reads its knobs, so
    patching fritz_utils is enough: no reload. A context manager yielding the
    app, so the patches end with the test.
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


class TestTheFakeKeepsFaithWithAskStuff(unittest.TestCase):
    def test_it_takes_exactly_what_ask_stuff_takes(self):
        """Names, order, kinds and defaults. If ask_stuff changes, a fake that
        did not follow would let PR 1's tests call it in ways the real one
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

    def test_it_records_every_call_by_name(self):
        fake = FakeAskStuff()
        stop = threading.Event()
        fake("hi", MessageSource.DESKTOP_TEXT, "desktop-pc", thread_id="discord-1#desktop",
             display_name="Nick", cancel_event=stop)
        call = fake.calls[0]
        self.assertEqual((call["source"], call["user_id"], call["thread_id"], call["display_name"]),
                         (MessageSource.DESKTOP_TEXT, "desktop-pc", "discord-1#desktop", "Nick"))
        self.assertIs(call["cancel_event"], stop)


class TestTheTokenWillBeTheOnlyGate(unittest.TestCase):
    """Basic auth, the CSP header and the same-origin check are all
    BaseHTTPMiddleware, which passes a non-HTTP scope straight through. So a
    WebSocket on the panel's app answers a client with no password and from
    any origin, and COMPANION_TOKEN is all that will stand in front of
    /companion. If a middleware change ever stopped WebSockets here, PR 1's
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


class TestCompanionIsNotServedYet(unittest.TestCase):
    """What fritz_utils' reserved-knob warning tells an operator. PR 1 builds
    the route and replaces this with the real handshake."""

    def test_enabled_or_not_there_is_no_route(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), companion_app(COMPANION_ENABLED=enabled) as app:
                with self.assertRaises(WebSocketDisconnect):
                    with TestClient(app).websocket_connect("/companion") as ws:
                        ws.receive_text()


if __name__ == "__main__":
    unittest.main()
