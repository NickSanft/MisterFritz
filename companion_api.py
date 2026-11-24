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
one hop to another identity, the owner's Discord one say, whose memories,
profile and name then follow. No turn here is checked for an admin's rights
today, but the id is the client's own claim, so whoever holds the token can be
any desk, and a desk linked to the owner is the owner.
The desk keeps a thread of its own, `<identity>#desktop`, whatever
THREADS_PER_CHANNEL says (plan 14, Q7).

Threads: a turn runs on bot_adapters' bounded pool, as Discord's do. The
agent's callbacks fire on that pool thread and hand frames to the event loop,
and each session's one sender task writes every frame in the order it was
queued, so nothing is sent from two places at once.

Audio (PR 9): the welcome offers `audio-in` once faster-whisper is
installed (the [voice] extra); without it a clip could only ever come back
unheard, so the desk is told to type instead. A push-to-talk clip is binary
frames of 16 kHz mono PCM16, AUDIO_MAX_BYTES at most, then an `audio_end`
naming its turn. That turn is a turn as a say's is (busy, barge-in and cancel
alike): `state transcribing`, the clip written to a WAV and transcribed on
the pool (stt.transcribe), then `stt` with what was heard and the turn itself
as MessageSource.DESKTOP_VOICE. Nothing heard, or the transcriber failing,
ends it with `stt_failed`; a clip that is no clip (empty, half a sample, not
16 kHz mono, too long) with `bad_frame`. What was said is never logged at INFO
or above, only how long it was (stt.transcribe keeps the words to DEBUG).

Speech (PR 9): a spoken turn's reply is spoken back, a typed one's is not (the
owner's call). With COMPANION_TTS=local the welcome offers `speak-local`, and
a desk whose hello says `speak-local` (it voices text itself) is sent the
reply a sentence at a time as it streams: `state speaking`, then `speak`
frames in order, the last marked final, all before `done`
(companion_tts.SentenceChunker cuts them, and speakable leaves out Markdown's
marks, fences and item numbers, and a link's address). A cancel stops them at the next sentence, and an
interrupted reply ends without its tail; a sentence said before a tool call
can't be unsaid when the client clears it. COMPANION_TTS=xtts (the brain
speaking) is not built yet, and `off` sends no speech.

Widgets (Phase 3) are not served: a `subscribe` is answered `bad_frame`.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field

from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

import companion_tts
import fritz_utils
import identity_store
from bot_adapters import fritz_error, run_blocking
from companion_protocol import (CLOSE_BAD_FRAME, CLOSE_BAD_TOKEN, CLOSE_FULL, CLOSE_GONE_QUIET,
                                CLOSE_NO_HELLO, CLOSE_UNSUPPORTED, PROTOCOL_VERSION, AudioEnd, BadFrame,
                                Cancel, Done, Error, Hello, Ping, Pong, Progress, Say, Speak, State,
                                Stt, Subscribe, Token, Welcome, parse_client_frame)
from fritz_utils import MessageSource, canonical_user_id, resolve_identity
from observability import audit_log

logger = logging.getLogger(__name__)

# Read at call time, so tests can shorten them.
HELLO_TIMEOUT_S = 5.0
PING_EVERY_S = 20.0
UNANSWERED_PINGS_ALLOWED = 2

# The close codes (CLOSE_*) are companion_protocol's, imported above.

# A push-to-talk clip: 16 kHz mono PCM16, 30 seconds at most, as the desktop
# sends it (desktop/src/shared/companion.ts, AUDIO_FORMAT and AUDIO_MAX_BYTES).
AUDIO_RATE = 16000
AUDIO_MAX_BYTES = 30 * AUDIO_RATE * 2


def server_caps() -> list[str]:
    """What this server does: text; audio-in once it can transcribe; and
    speak-local with COMPANION_TTS=local, a spoken turn's reply sent for the
    desk to voice itself."""
    caps = ["text"]
    if _can_transcribe():
        caps.append("audio-in")
    if fritz_utils.COMPANION_TTS == "local":
        caps.append("speak-local")
    return caps


def _can_transcribe() -> bool:
    """Whether faster-whisper is installed. Its model loads (or downloads) at
    the first clip, as stt.transcribe loads it for Discord."""
    return importlib.util.find_spec("faster_whisper") is not None

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
# The first after Discord's words for a voice note it could not make out
# (main_discord.py); the other two are the desk's own.
_COPY_NOT_HEARD = "I could not make out a word of that, sir. Do say it again, or type it."
_COPY_SILENCE = "I heard nothing at all, sir: hold the key while you speak."
_COPY_TOO_LONG = "That ran longer than I can listen to at once, sir: half a minute at most."
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
    """A say, or a clip, that came after its running turn was cancelled:
    barge-in. A clip is transcribed once it starts."""
    turn_id: str
    text: str
    source: MessageSource
    pcm: bytes | None = None


def _connected_as(canonical_id: str, identity: str) -> str:
    """The connect line: which desk, and who it is to the brain.

    This is where an operator looks to see whether a link took, and the
    likeliest mistake there is cannot be seen at startup: IDENTITY_LINKS names
    a desk (an example's, or one misspelt) that is not the one connecting. So
    a desk that connects as itself is told which other desks are linked, and
    to it, which are linked to it. What IDENTITY_LINKS gave is said as
    fritz_utils says it at startup; the desk's own id is held to the
    protocol's pattern before it gets here.
    """
    said = fritz_utils.said_identity
    if identity != canonical_id:
        return f"{canonical_id} connected as {said(identity)} (linked by IDENTITY_LINKS)"
    links = fritz_utils.IDENTITY_LINKS
    desks = sorted(alias for alias in links
                   if alias != canonical_id and fritz_utils.split_user_id(alias)[0] == "desktop")
    here = [alias for alias in desks if links[alias] == canonical_id]
    elsewhere = [alias for alias in desks if links[alias] != canonical_id]
    line = f"{canonical_id} connected as itself"
    if here:
        line += f" ({', '.join(map(said, here))} {'is' if len(here) == 1 else 'are'} linked to it)"
    if elsewhere:
        line += f": IDENTITY_LINKS links {', '.join(map(said, elsewhere))}, not this desk"
    return line


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
        # The clip the next audio_end names, as its binary frames come, and
        # whether it outgrew AUDIO_MAX_BYTES (then it is dropped whole). Kept
        # only when the welcome offered audio-in.
        self.hears = False
        self.audio = bytearray()
        self.audio_over = False
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
            elif msg.get("bytes") is not None:
                _hear(session, msg["bytes"])
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
    caps = server_caps()
    session.hears = "audio-in" in caps
    session.send(Welcome(protocol=PROTOCOL_VERSION, identity=session.identity,
                         display_name=session.display_name, server_caps=caps,
                         thread_id=session.thread_id))
    logger.info("companion: %s", _connected_as(session.canonical_id, session.identity))
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
    session.audio.clear()
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
        _end_clip(session, frame)
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


def _hear(session: _Session, data: bytes) -> None:
    """A binary frame: part of the clip its audio_end will name. A clip that
    outgrows AUDIO_MAX_BYTES is dropped whole, and its audio_end says so;
    without audio-in, every one is dropped unread."""
    if not session.hears or session.audio_over:
        return
    if len(session.audio) + len(data) > AUDIO_MAX_BYTES:
        logger.warning("companion: %s sent a clip of more than %d bytes; dropped",
                       session.canonical_id, AUDIO_MAX_BYTES)
        session.audio.clear()
        session.audio_over = True
        return
    session.audio.extend(data)


def _end_clip(session: _Session, frame: AudioEnd) -> None:
    """An audio_end: the clip so far is its turn's, and the next starts empty."""
    pcm, over = bytes(session.audio), session.audio_over
    session.audio.clear()
    session.audio_over = False
    if not session.hears:
        session.send(Error(turn_id=frame.turn_id, code="bad_frame", text=_COPY_NO_EARS))
        return
    if over:
        session.send(Error(turn_id=frame.turn_id, code="bad_frame", text=_COPY_TOO_LONG))
        return
    if frame.sample_rate != AUDIO_RATE or frame.channels != 1 or not pcm or len(pcm) % 2:
        logger.warning("companion: %s sent a clip that can't be one: %d bytes at %d Hz, %d channel(s)",
                       session.canonical_id, len(pcm), frame.sample_rate, frame.channels)
        session.send(Error(turn_id=frame.turn_id, code="bad_frame", text=_COPY_BAD_FRAME))
        return
    _start_turn(session, frame.turn_id, "", MessageSource.DESKTOP_VOICE, pcm=pcm)


def _start_turn(session: _Session, turn_id: str, text: str, source: MessageSource,
                pcm: bytes | None = None) -> None:
    """Start a turn: of `text`, or of what clip `pcm` says once transcribed."""
    active = session.active
    if active is not None:
        if active.cancel.is_set() and session.waiting is None:
            # Barge-in. The cancelled turn ends at the model's next token, or
            # when the tool in flight returns; this one starts then.
            session.waiting = _Waiting(turn_id, text, source, pcm)
            return
        session.send(Error(turn_id=turn_id, code="busy", text=_COPY_BUSY))
        return
    turn = _Turn(turn_id)
    session.active = turn
    session.send(State(turn_id=turn_id, state="transcribing" if pcm is not None else "thinking"))
    task = asyncio.create_task(_run_turn(session, turn, text, source, pcm))
    _TURN_TASKS.add(task)
    task.add_done_callback(_TURN_TASKS.discard)


async def _run_turn(session: _Session, turn: _Turn, text: str, source: MessageSource,
                    pcm: bytes | None = None) -> None:
    try:
        if pcm is not None:
            heard = await _transcribe(session, turn, pcm)
            if heard is None:
                return      # ended already, with done or stt_failed
            text = heard
            session.send(State(turn_id=turn.turn_id, state="thinking"))
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
            _start_turn(session, waiting.turn_id, waiting.text, waiting.source, waiting.pcm)


async def _transcribe(session: _Session, turn: _Turn, pcm: bytes) -> str | None:
    """What clip `pcm` says, sent as `stt`; or None once the turn has ended:
    cancelled (done, interrupted), nothing heard or the transcriber failing
    (stt_failed). Only how long it was is audited, never what was said, and
    stt.transcribe logs the words at DEBUG alone."""
    started = time.monotonic()
    heard = await run_blocking(_transcribe_clip, pcm)
    said = (heard or "").strip()
    if turn.cancel.is_set():
        # Cancelled, or its desk gone, while it was heard: it ends as a turn
        # cancelled before it reached the model, empty and interrupted.
        result = "cancelled"
        session.send(Done(turn_id=turn.turn_id, text="", interrupted=True))
    elif heard is None:
        # stt.transcribe has logged and counted the failure ("whisper_stt").
        result = "failed"
        session.send(Error(turn_id=turn.turn_id, code="stt_failed",
                           text=fritz_error("companion_stt", note=_COPY_NOT_HEARD)))
    elif not said:
        result = "silent"
        session.send(Error(turn_id=turn.turn_id, code="stt_failed", text=_COPY_SILENCE))
    else:
        result = "ok"
        session.send(Stt(turn_id=turn.turn_id, text=said))
    audit_log("companion_audio", user_id=session.identity, client=session.canonical_id,
              audio_seconds=round(len(pcm) / (2 * AUDIO_RATE), 2), chars=len(said), result=result,
              seconds=round(time.monotonic() - started, 2))
    return said if result == "ok" else None


def _speaks(session: _Session, source: MessageSource) -> bool:
    """Whether a turn's reply is spoken: a spoken turn's (a typed one is
    answered in the bubble alone), with COMPANION_TTS=local, to a desk that
    voices text itself (its hello's speak-local)."""
    return (source == MessageSource.DESKTOP_VOICE and fritz_utils.COMPANION_TTS == "local"
            and "speak-local" in session.caps)


class _Speaker:
    """A spoken turn's reply as speak frames, a sentence at a time, from the
    turn's pool thread: `state speaking` before the first, and the last
    marked final, an empty one if the reply's end left nothing more to say.
    A cancel stops them at the next sentence; an interrupted reply ends with
    none of its tail and no final (its `done` says it was interrupted)."""

    def __init__(self, session: _Session, turn: _Turn):
        self.session, self.turn = session, turn
        self.chunker = companion_tts.SentenceChunker()
        self.seq = 0

    def heard(self, delta: str, restart: bool) -> None:
        if self.turn.cancel.is_set():
            return
        for sentence in self.chunker.feed(delta, restart):
            self._say(sentence, final=False)

    def ended(self, interrupted: bool) -> None:
        if interrupted or self.turn.cancel.is_set():
            return
        rest = self.chunker.flush()
        for i, sentence in enumerate(rest):
            self._say(sentence, final=i == len(rest) - 1)
        if not rest and self.seq:
            self._say("", final=True)

    def _say(self, text: str, final: bool) -> None:
        if self.seq == 0:
            self.session.send_from_thread(State(turn_id=self.turn.turn_id, state="speaking"))
        self.session.send_from_thread(Speak(turn_id=self.turn.turn_id, seq=self.seq, text=text, final=final))
        self.seq += 1


def _transcribe_clip(pcm: bytes) -> str | None:
    """On a pool thread: the clip written to a WAV of its own, transcribed,
    and the WAV gone. stt.transcribe answers "" when it heard no speech and
    None when it failed."""
    import stt   # pydub at import, and faster-whisper only at the first clip

    fd, path = tempfile.mkstemp(prefix="companion_", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f, wave.open(f, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AUDIO_RATE)
            w.writeframes(pcm)
        return stt.transcribe(path)
    finally:
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("companion: could not delete the clip's WAV %s: %s", path, e)


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

    speaker = _Speaker(session, turn) if _speaks(session, source) else None

    def streaming_callback(delta: str, accumulated: str, restart: bool = False) -> None:
        session.send_from_thread(Token(turn_id=turn.turn_id, delta=delta, restart=restart))
        if speaker is not None:
            speaker.heard(delta, restart)

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
    if speaker is not None:
        # Before done: these are queued from this thread, as the tokens are.
        speaker.ended(interrupted)
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
