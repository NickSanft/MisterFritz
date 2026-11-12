"""The desktop companion's wire protocol: every frame /companion carries.

plans/14-desktop-companion-build.md § 2 specifies it, and this module is its
source of truth. companion_api speaks it, and the desktop app's TypeScript is
to be generated from it (scripts/gen_protocol_ts.py), so the two ends cannot
drift apart unnoticed.

Text frames are UTF-8 JSON objects with a `type`. Binary frames carry audio
only (Phase 2) and are always followed by an `audio_end`. The version is
negotiated in hello/welcome.

Where this differs from the plan's sketch, and why:

- ping and pong go both ways. The server pings and the client mirrors it
  (§ 2.2), so each side must accept both; the sketch had ping only from the
  client and pong only from the server.
- A turn_id is 1 to 64 of [A-Za-z0-9_-], room for a UUID. Every frame of a
  turn echoes it back, so it is bounded here rather than echoed at any size.
- Say.text is stripped before its length is checked: whitespace is not a
  question.
- Done.images are data: URLs (see Done).

Most models are declared ahead of the server code that sends or reads them,
so the generated types are whole from the start. Which frames the server
handles today is companion_api's business, not this module's.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter, ValidationError

PROTOCOL_VERSION = 1

# The client's id becomes the canonical identity "desktop-<id>", so it is
# held to a charset canonical_user_id keeps intact.
ClientId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")]
TurnId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
Cap = Literal["text", "audio-in", "audio-out", "speak-local", "vrm", "widgets"]
TurnState = Literal["listening", "transcribing", "thinking", "speaking", "idle"]
ErrorCode = Literal["busy", "bad_frame", "turn_failed", "stt_failed", "not_allowed", "internal"]


# ── client → server ──────────────────────────────────────────────────────────

class ClientInfo(BaseModel):
    id: ClientId
    name: str = Field(max_length=64)      # the machine's name, e.g. "Nick's PC"
    caps: list[Cap]


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    protocol: int = PROTOCOL_VERSION
    token: str
    client: ClientInfo


class Say(BaseModel):
    type: Literal["say"] = "say"
    turn_id: TurnId
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class AudioEnd(BaseModel):
    """Follows one or more binary frames of audio (Phase 2)."""
    type: Literal["audio_end"] = "audio_end"
    turn_id: TurnId
    sample_rate: int = 16000
    channels: int = 1
    format: Literal["pcm16"] = "pcm16"


class Cancel(BaseModel):
    type: Literal["cancel"] = "cancel"
    turn_id: TurnId


class Subscribe(BaseModel):
    """Phase 3: the widgets the client wants fed."""
    type: Literal["subscribe"] = "subscribe"
    widgets: list[str]


class Ping(BaseModel):
    """Either side; the other answers with a Pong carrying the same t."""
    type: Literal["ping"] = "ping"
    t: int


class Pong(BaseModel):
    type: Literal["pong"] = "pong"
    t: int


# ── server → client ──────────────────────────────────────────────────────────

class Welcome(BaseModel):
    type: Literal["welcome"] = "welcome"
    protocol: int
    identity: str
    display_name: str
    server_caps: list[Cap]
    thread_id: str


class State(BaseModel):
    type: Literal["state"] = "state"
    turn_id: str
    state: TurnState


class Stt(BaseModel):
    """Phase 2: what was heard."""
    type: Literal["stt"] = "stt"
    turn_id: str
    text: str


class Token(BaseModel):
    type: Literal["token"] = "token"
    turn_id: str
    delta: str
    # A new answer segment begins (the model narrated, called a tool and began
    # again): the client clears what it has shown of this turn.
    restart: bool = False


class Progress(BaseModel):
    type: Literal["progress"] = "progress"
    turn_id: str
    text: str


class Speech(BaseModel):
    """Phase 2, brain-side TTS: one sentence of audio."""
    type: Literal["speech"] = "speech"
    turn_id: str
    seq: int
    mime: Literal["audio/wav"] = "audio/wav"
    b64: str
    final: bool


class Speak(BaseModel):
    """Phase 2, body-side TTS: one sentence for the client to synthesise."""
    type: Literal["speak"] = "speak"
    turn_id: str
    seq: int
    text: str
    final: bool


class Done(BaseModel):
    """A turn's end. `text` is the whole reply, which supersedes the tokens.

    `images` are data: URLs (data:image/png;base64,...), so the client needs
    nothing but this socket to show them, on this machine or another.
    """
    type: Literal["done"] = "done"
    turn_id: str
    text: str
    images: list[str] = Field(default_factory=list)
    interrupted: bool = False


class Error(BaseModel):
    """`text` is Fritz's own words, fit to show; never a raw exception."""
    type: Literal["error"] = "error"
    turn_id: str | None = None
    code: ErrorCode
    text: str


class Widget(BaseModel):
    """Phase 3."""
    type: Literal["widget"] = "widget"
    name: str
    payload: dict
    ttl_s: int


class Reminder(BaseModel):
    """Phase 3."""
    type: Literal["reminder"] = "reminder"
    schedule_id: int
    text: str


ClientFrame = Annotated[Union[Hello, Say, AudioEnd, Cancel, Subscribe, Ping, Pong],
                        Field(discriminator="type")]
ServerFrame = Annotated[Union[Welcome, State, Stt, Token, Progress, Speech, Speak, Done, Error,
                              Widget, Reminder, Ping, Pong],
                        Field(discriminator="type")]

_CLIENT_FRAME: TypeAdapter = TypeAdapter(ClientFrame)
_SERVER_FRAME: TypeAdapter = TypeAdapter(ServerFrame)
_TURN_ID: TypeAdapter = TypeAdapter(TurnId)


class BadFrame(ValueError):
    """A text frame that is not a client frame.

    The message says briefly what was wrong, for the server's log. `turn_id`
    is the frame's own, when it named a well-formed one, so a broken `say` can
    still be answered under the turn the client is waiting on.
    """

    def __init__(self, reason: str, turn_id: str | None = None):
        super().__init__(reason)
        self.turn_id = turn_id


def parse_client_frame(raw: str | bytes) -> Hello | Say | AudioEnd | Cancel | Subscribe | Ping | Pong:
    """One client frame from its JSON, or BadFrame."""
    try:
        return _CLIENT_FRAME.validate_json(raw)
    except ValidationError as e:
        raise BadFrame(_reason(e), _turn_id_of(raw)) from None


def parse_server_frame(raw: str | bytes):
    """One server frame from its JSON. For tests and tools: the server builds
    its frames from the models and never parses them."""
    return _SERVER_FRAME.validate_json(raw)


def _reason(error: ValidationError) -> str:
    first = error.errors(include_input=False, include_url=False)[0]
    where = ".".join(str(part) for part in first.get("loc", ())) or "frame"
    more = f" (and {error.error_count() - 1} more)" if error.error_count() > 1 else ""
    return f"{where}: {first.get('msg', 'invalid')}{more}"[:300]


def _turn_id_of(raw: str | bytes) -> str | None:
    try:
        candidate = json.loads(raw).get("turn_id")
        return _TURN_ID.validate_python(candidate, strict=True)
    except (ValueError, TypeError, AttributeError, RecursionError, ValidationError):
        return None
