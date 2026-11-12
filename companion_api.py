"""/companion: the WebSocket the desktop app talks to (plan 14, PR 1).

One connection is one session: a `hello` naming the client and carrying
COMPANION_TOKEN, then text turns streamed back as the model writes them. The
frames are companion_protocol's. The rules are
plans/14-desktop-companion-build.md § 2.2:

- The first frame is a hello, within HELLO_TIMEOUT_S, or the socket is
  closed: 4408 if nothing came, 4400 if something else did, 4401 for a wrong
  token, 4426 for a protocol version this server does not speak (even when
  its hello has fields this version cannot read), and 4409 when
  COMPANION_MAX_CLIENTS sessions are already open.
- One turn at a time per session. A `say` while one runs is answered `busy`
  under the new turn_id. Barge-in is a `cancel` and a new `say` in one go: a
  say that arrives once the running turn is cancelled waits for it to end,
  then starts (done and idle for the old turn, then thinking for the new).
  One say waits so; another is busy. A cancel naming the waiting say drops it.
- Every turn the server accepts ends in exactly one `done` or one `error`
  with its turn_id, after every token of it, and then `state idle`.
- The server pings every PING_EVERY_S and closes with 4000 once two pings go
  unanswered. A client's ping is answered with a pong.
- A text frame longer than MAX_TEXT_FRAME_CHARS is refused unread: 4400
  before the hello, bad_frame after it, under the turn it names if that is in
  its first or last KiB. uvicorn closes with 1009 on any frame over its
  ws_max_size (admin_panel.server_config).

The token is the only gate. The admin panel's Basic auth, CSP and same-origin
checks are HTTP middleware, which a WebSocket passes straight through
(tests/test_companion_api.py holds that premise), so nothing else stands in
front of this route.

Identity: the client's id becomes `desktop-<id>`, which IDENTITY_LINKS may link
one hop to another identity, the owner's Discord one say, whose memories and
rights then follow. Linked to ROOT_USER, the token carries the owner's rights.
The desk keeps a thread of its own, `<identity>#desktop`, whatever
THREADS_PER_CHANNEL says (plan 14, Q7).

Threads: a turn runs on bot_adapters' bounded pool, as Discord's do. The
agent's callbacks fire on that pool thread and hand frames to the event loop,
and each session's one sender task writes every frame in the order it was
queued, so nothing is sent from two places at once.

Audio (PR 9) and widgets (Phase 3) are not served: the welcome's server_caps
say "text" only, a binary frame is dropped unread, and an `audio_end` or a
`subscribe` is answered `bad_frame`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

import fritz_utils
import identity_store
from bot_adapters import fritz_error, run_blocking
from companion_protocol import (PROTOCOL_VERSION, AudioEnd, BadFrame, Cancel, Done, Error, Hello, Ping,
                                Pong, Progress, Say, State, Subscribe, Token, Welcome, parse_client_frame)
from fritz_utils import MessageSource, canonical_user_id, resolve_identity
from observability import audit_log

logger = logging.getLogger(__name__)

# Read at call time, so tests can shorten them.
HELLO_TIMEOUT_S = 5.0
PING_EVERY_S = 20.0
UNANSWERED_PINGS_ALLOWED = 2

CLOSE_GONE_QUIET = 4000
CLOSE_BAD_FRAME = 4400
CLOSE_BAD_TOKEN = 4401
CLOSE_NO_HELLO = 4408
CLOSE_FULL = 4409
CLOSE_UNSUPPORTED = 4426

# What this server does today. PR 9 adds the audio caps.
SERVER_CAPS = ["text"]

# Refused before it is parsed. A hello is well under 1 KiB, and the longest
# say (8000 characters, every one an astral emoji written as 😀)
# is about 96,000; parsing is synchronous on the event loop, so a frame
# nobody legitimate sends must not get that far.
MAX_TEXT_FRAME_CHARS = 128 * 1024

# An image larger than this is left out of `done` and logged; base64 makes it
# a third larger again on the wire.
IMAGE_MAX_BYTES = 8 * 1024 * 1024

_COPY_BAD_FRAME = ("I'm afraid I could not make sense of that, sir. The particulars "
                   "have been noted in the log.")
_COPY_BUSY = ("One moment, sir: I am still attending to your last request. Do stop "
              "me first if you would rather I turned to this one.")
_COPY_NO_EARS = "Forgive me, sir, but I cannot hear you at this desk as yet. Do type instead."
_COPY_NO_WIDGETS = "I keep no widgets at this desk as yet, sir."


@dataclass(frozen=True)
class _Close:
    """Queued like a frame, so the close goes out after everything before it."""
    code: int


@dataclass
class _Turn:
    turn_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class _Waiting:
    """A say that came after its running turn was cancelled: barge-in."""
    turn_id: str
    text: str
    source: MessageSource


class _Session:
    def __init__(self, ws: WebSocket, hello: Hello):
        self.ws = ws
        self.canonical_id = canonical_user_id("desktop", hello.client.id)
        self.identity = resolve_identity(self.canonical_id)
        self.thread_id = f"{self.identity}#desktop"
        self.display_name = ""
        self.caps = set(hello.client.caps)
        self.out: asyncio.Queue = asyncio.Queue()
        self.active: _Turn | None = None
        self.waiting: _Waiting | None = None
        self.unanswered_pings = 0
        self.loop = asyncio.get_running_loop()

    def send(self, frame) -> None:
        """Queue a frame for the sender task. On the event loop only."""
        self.out.put_nowait(frame)

    def send_from_thread(self, frame) -> None:
        """Queue a frame from the turn's pool thread."""
        try:
            self.loop.call_soon_threadsafe(self.out.put_nowait, frame)
        except RuntimeError:
            # The loop has closed: the server is stopping and nobody is there
            # to read it. Raising here would fail the turn inside the agent.
            pass

    def close(self, code: int) -> None:
        self.out.put_nowait(_Close(code))


_SESSIONS: set[_Session] = set()
# A plain lock, never held across an await. The TestClient runs each
# connection on its own event loop in its own thread, so the set is shared
# between threads there, if not under uvicorn.
_SESSIONS_LOCK = threading.Lock()
# Turn tasks, held so they are not collected mid-turn; each removes itself.
_TURN_TASKS: set[asyncio.Task] = set()


def routes() -> list[WebSocketRoute]:
    return [WebSocketRoute("/companion", companion_endpoint, name="companion")]


async def companion_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = await _handshake(ws)
    if session is None:
        return
    tasks = [asyncio.create_task(_drain(session)), asyncio.create_task(_heartbeat(session))]
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            text = msg.get("text")
            if text is not None:
                _dispatch(session, text)
            # A binary frame is audio, which PR 9 serves. Dropped unread, never
            # buffered: the welcome did not offer audio-in.
    finally:
        _end_session(session)
        for task in tasks:
            task.cancel()


async def _handshake(ws: WebSocket) -> _Session | None:
    peer = ws.client.host if ws.client else None
    try:
        msg = await asyncio.wait_for(ws.receive(), HELLO_TIMEOUT_S)
    except TimeoutError:
        logger.info("companion: no hello from %s within %ss", peer, HELLO_TIMEOUT_S)
        await _close_now(ws, CLOSE_NO_HELLO)
        return None
    if msg["type"] == "websocket.disconnect":
        return None
    raw = msg.get("text")
    try:
        if raw is None:
            raise BadFrame("a binary frame before hello")
        if len(raw) > MAX_TEXT_FRAME_CHARS:
            raise BadFrame(f"a first frame of {len(raw)} characters")
        hello = parse_client_frame(raw)
        if not isinstance(hello, Hello):
            raise BadFrame(f"{hello.type} before hello")
    except BadFrame as e:
        # Never for an oversized frame: that would parse what the cap refused.
        skew = _newer_hello(raw) if raw is not None and len(raw) <= MAX_TEXT_FRAME_CHARS else None
        if skew is not None:
            logger.warning("companion: a hello for protocol %d, this server %d", skew, PROTOCOL_VERSION)
            audit_log("companion_connect", result="unsupported_protocol", protocol=skew, peer=peer)
            await _close_now(ws, CLOSE_UNSUPPORTED)
            return None
        # %r: the reason can quote the client's own `type`, newlines and all.
        logger.info("companion: closing %s, a bad first frame: %r", peer, str(e))
        await _close_now(ws, CLOSE_BAD_FRAME)
        return None

    if not _token_ok(hello.token):
        logger.warning("companion: wrong token from %s (client %r)", peer, hello.client.id)
        audit_log("companion_connect", result="bad_token", client=hello.client.id, peer=peer)
        await _close_now(ws, CLOSE_BAD_TOKEN)
        return None
    if hello.protocol != PROTOCOL_VERSION:
        logger.warning("companion: %r speaks protocol %d, this server %d",
                       hello.client.id, hello.protocol, PROTOCOL_VERSION)
        audit_log("companion_connect", result="unsupported_protocol", client=hello.client.id,
                  protocol=hello.protocol)
        await _close_now(ws, CLOSE_UNSUPPORTED)
        return None

    session = _Session(ws, hello)
    with _SESSIONS_LOCK:
        full = len(_SESSIONS) >= fritz_utils.COMPANION_MAX_CLIENTS
        if not full:
            _SESSIONS.add(session)
    if full:
        logger.warning("companion: refused %s, %d clients already connected",
                       session.canonical_id, fritz_utils.COMPANION_MAX_CLIENTS)
        audit_log("companion_connect", result="full", user_id=session.identity,
                  client=session.canonical_id)
        await _close_now(ws, CLOSE_FULL)
        return None

    # The client's name is the machine's ("Nick's PC"), so it is recorded
    # against the desk's own id, as the web chat records a web name, and never
    # against a linked identity, whose name is a person's. Not passed to
    # ask_stuff either, which would record it against the linked one.
    identity_store.record(session.canonical_id, hello.client.name, "desktop")
    session.display_name = identity_store.display_name(session.identity)
    session.send(Welcome(protocol=PROTOCOL_VERSION, identity=session.identity,
                         display_name=session.display_name, server_caps=SERVER_CAPS,
                         thread_id=session.thread_id))
    logger.info("companion: %s connected as %s", session.canonical_id, session.identity)
    audit_log("companion_connect", result="ok", user_id=session.identity,
              client=session.canonical_id, peer=peer)
    return session


def _token_ok(token: str) -> bool:
    expected = fritz_utils.COMPANION_TOKEN
    # As bytes: compare_digest refuses a str with anything outside ASCII.
    return bool(expected) and secrets.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


def _newer_hello(raw: str | None) -> int | None:
    """The version a hello names when this server cannot read it, or None.

    A later protocol may add a cap or a field version 1's models refuse, and
    its client should still hear 4426, not "bad frame", so it can fall back.
    Only with the right token, so a stranger still learns nothing.
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or data.get("type") != "hello":
        return None
    token, protocol = data.get("token"), data.get("protocol")
    if not isinstance(token, str) or type(protocol) is not int or protocol == PROTOCOL_VERSION:
        return None
    return protocol if _token_ok(token) else None


async def _close_now(ws: WebSocket, code: int) -> None:
    """Close before the sender task exists. The client may already be gone."""
    try:
        await ws.close(code)
    except Exception as e:
        logger.debug("companion: close %d not sent: %s", code, e)


def _end_session(session: _Session) -> None:
    """Synchronous on purpose: it must run whole even when the connection's
    task is being cancelled."""
    with _SESSIONS_LOCK:
        _SESSIONS.discard(session)
    # Nothing is dispatched after this, so a say waiting behind a cancel is
    # dropped for good: the turn ahead of it finds none to start.
    session.waiting = None
    if session.active is not None:
        # The client has gone; stop the model rather than finish for nobody.
        session.active.cancel.set()
    logger.info("companion: %s disconnected", session.canonical_id)


async def _drain(session: _Session) -> None:
    """The session's only writer: frames go out in the order they were queued."""
    try:
        while True:
            item = await session.out.get()
            if isinstance(item, _Close):
                await session.ws.close(item.code)
                return
            await session.ws.send_text(item.model_dump_json())
    except Exception as e:
        # The client went mid-send. The receive loop sees the disconnect and
        # ends the session; this task has nothing left to do.
        logger.debug("companion: sender for %s stopped: %s", session.canonical_id, e)


async def _heartbeat(session: _Session) -> None:
    while True:
        await asyncio.sleep(PING_EVERY_S)
        if session.unanswered_pings >= UNANSWERED_PINGS_ALLOWED:
            logger.info("companion: %s answered no ping in %d; closing",
                        session.canonical_id, session.unanswered_pings)
            session.close(CLOSE_GONE_QUIET)
            return
        session.unanswered_pings += 1
        session.send(Ping(t=int(time.time() * 1000)))


def _dispatch(session: _Session, raw: str) -> None:
    try:
        if len(raw) > MAX_TEXT_FRAME_CHARS:
            raise BadFrame(f"a frame of {len(raw)} characters", _turn_id_near_an_end(raw))
        frame = parse_client_frame(raw)
    except BadFrame as e:
        # %r: the reason can quote the client's own `type`, newlines and all.
        logger.warning("companion: bad frame from %s: %r", session.canonical_id, str(e))
        session.send(Error(turn_id=e.turn_id, code="bad_frame", text=_COPY_BAD_FRAME))
        return

    if isinstance(frame, Say):
        _start_turn(session, frame.turn_id, frame.text, MessageSource.DESKTOP_TEXT)
    elif isinstance(frame, Cancel):
        waiting = session.waiting
        if session.active is not None and session.active.turn_id == frame.turn_id:
            session.active.cancel.set()
        elif waiting is not None and waiting.turn_id == frame.turn_id:
            # Never started, so nothing ran: it ends as ask_stuff ends a turn
            # cancelled while queued, empty and interrupted.
            session.waiting = None
            session.send(Done(turn_id=waiting.turn_id, text="", interrupted=True))
            session.send(State(turn_id=waiting.turn_id, state="idle"))
        # Otherwise the turn is unknown or already over: nothing to stop.
    elif isinstance(frame, Ping):
        session.send(Pong(t=frame.t))
    elif isinstance(frame, Pong):
        session.unanswered_pings = 0
    elif isinstance(frame, AudioEnd):
        session.send(Error(turn_id=frame.turn_id, code="bad_frame", text=_COPY_NO_EARS))
    elif isinstance(frame, Subscribe):
        session.send(Error(code="bad_frame", text=_COPY_NO_WIDGETS))
    else:   # a second hello
        logger.warning("companion: %s sent %s mid-session", session.canonical_id, frame.type)
        session.send(Error(code="bad_frame", text=_COPY_BAD_FRAME))


_TURN_ID_FIELD = re.compile(r'"turn_id"\s*:\s*"([A-Za-z0-9_-]{1,64})"')
_TURN_ID_SCAN_CHARS = 1024


def _turn_id_near_an_end(raw: str) -> str | None:
    """The turn an oversized frame names, found without parsing it.

    Only its first and last KiB are searched, where a client's serialiser
    puts `turn_id` (before `text` or after it), so the turn the client waits
    on can still be ended with bad_frame, and nothing reads the whole frame.
    """
    for part in (raw[:_TURN_ID_SCAN_CHARS], raw[-_TURN_ID_SCAN_CHARS:]):
        found = _TURN_ID_FIELD.search(part)
        if found:
            return found.group(1)
    return None


def _start_turn(session: _Session, turn_id: str, text: str, source: MessageSource) -> None:
    active = session.active
    if active is not None:
        if active.cancel.is_set() and session.waiting is None:
            # Barge-in. The cancelled turn ends at the model's next token, or
            # when the tool in flight returns; this one starts then.
            session.waiting = _Waiting(turn_id, text, source)
            return
        session.send(Error(turn_id=turn_id, code="busy", text=_COPY_BUSY))
        return
    turn = _Turn(turn_id)
    session.active = turn
    session.send(State(turn_id=turn_id, state="thinking"))
    task = asyncio.create_task(_run_turn(session, turn, text, source))
    _TURN_TASKS.add(task)
    task.add_done_callback(_TURN_TASKS.discard)


async def _run_turn(session: _Session, turn: _Turn, text: str, source: MessageSource) -> None:
    try:
        reply, images, interrupted = await run_blocking(_ask, session, turn, text, source)
    except Exception as e:
        session.send(Error(turn_id=turn.turn_id, code="turn_failed",
                           text=fritz_error("companion_turn", e)))
    else:
        # After every token: the pool thread queued those with
        # call_soon_threadsafe before it returned, so they are ahead of this.
        session.send(Done(turn_id=turn.turn_id, text=reply, images=images, interrupted=interrupted))
    finally:
        # Cleared in the same step as the frames are queued, so the client's
        # next say, which can only come after it has read them, finds no turn.
        if session.active is turn:
            session.active = None
        session.send(State(turn_id=turn.turn_id, state="idle"))
        waiting, session.waiting = session.waiting, None
        # Not while this task is itself being cancelled: that is the server
        # stopping, and a task started now would be left behind.
        task = asyncio.current_task()
        if waiting is not None and not (task and task.cancelling()):
            _start_turn(session, waiting.turn_id, waiting.text, waiting.source)


def _ask(session: _Session, turn: _Turn, text: str, source: MessageSource) -> tuple[str, list[str], bool]:
    """The turn itself, on a pool thread."""
    if turn.cancel.is_set():
        # Cancelled, or its desk gone, while it waited for a pool thread. It
        # never reaches the model or the transcript, as ask_stuff treats a turn
        # cancelled while queued for its conversation.
        audit_log("companion_turn", user_id=session.identity, client=session.canonical_id,
                  chars=len(text), result="cancelled_before_start")
        return "", [], True

    from mister_fritz import ask_stuff   # heavy; and imported late, as the web chat does

    def streaming_callback(delta: str, accumulated: str, restart: bool = False) -> None:
        session.send_from_thread(Token(turn_id=turn.turn_id, delta=delta, restart=restart))

    def progress_callback(notice: str) -> None:
        session.send_from_thread(Progress(turn_id=turn.turn_id, text=notice))

    try:
        # The desk's own id, unresolved: ask_stuff follows IDENTITY_LINKS itself.
        result = ask_stuff(text, source, session.canonical_id,
                           progress_callback=progress_callback,
                           streaming_callback=streaming_callback,
                           thread_id=session.thread_id,
                           cancel_event=turn.cancel)
    except Exception as e:
        audit_log("companion_turn", user_id=session.identity, client=session.canonical_id,
                  chars=len(text), result="error", error=str(e))
        raise
    reply = result.get("text") or ""
    if not isinstance(reply, str):
        reply = str(reply)
    interrupted = bool(result.get("interrupted", False))
    images = _data_urls(result.get("image_paths") or [])
    audit_log("companion_turn", user_id=session.identity, client=session.canonical_id,
              chars=len(text), result="interrupted" if interrupted else "ok",
              reply_chars=len(reply), images=len(images),
              seconds=round(time.monotonic() - turn.started, 2))
    return reply, images, interrupted


def _data_urls(paths: list[str]) -> list[str]:
    """The turn's images as data: URLs, from the web chat's asset roots only.

    The roots and types /chat/assets serves, so nothing outside them reaches
    the desk; an image anywhere else is left out and logged.
    """
    from admin_panel import _CHAT_ASSET_CONTENT_TYPES, _chat_asset_roots

    roots = _chat_asset_roots()
    urls = []
    for path in paths:
        full = os.path.abspath(path)
        kind = _CHAT_ASSET_CONTENT_TYPES.get(os.path.splitext(full)[1].lower())
        inside = any(full.startswith(root + os.sep) for root in roots)
        if kind is None or not inside:
            logger.warning("companion: not sending image %s (outside the asset roots, or not an image)", path)
            continue
        try:
            if os.path.getsize(full) > IMAGE_MAX_BYTES:
                logger.warning("companion: not sending image %s, larger than %d bytes", path, IMAGE_MAX_BYTES)
                continue
            with open(full, "rb") as f:
                data = f.read()
        except OSError as e:
            logger.warning("companion: could not read image %s: %s", path, e)
            continue
        urls.append(f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}")
    return urls
