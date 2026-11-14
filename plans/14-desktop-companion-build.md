# 14 (build). Desktop companion — the build plan for Claude Code

[← back to index](README.md) · [plan 14 — why and what](14-desktop-companion.md) · [decisions](DECISIONS.md)

_Written November 2026 as the execution companion to plan 14. Plan 14 holds the reasoning, the interview decisions, the risks and the open questions; this document holds what to type. Signatures and behaviours below were read from the working trees at MisterFritz `aecfe90`, LorraineAtelier `d62b7ec` and DivoraVoiceMod `c9f066b`. Line numbers are given as a courtesy and **will drift — find the symbol, don't trust the line.**_

---

## 0. How to work this plan

**Order.** Sections 3 and 4 are PR-sized units in dependency order: P0 → P5 (brain prerequisites), then Spike 0, then PR 1 … PR 17. A PR is done when its *Tests* pass and its *Acceptance* line is true. Do not start a body PR before its brain dependency is merged; the dependency is stated on each.

**Gates.** Three things stop the line and go back to the owner: Spike 0 failing its table (§ 4.1); any test in § 3 that would require weakening an existing test; and any decision in § 7 whose default you cannot apply (for example Q1, hardware you cannot see).

**Repo conventions (brain).** Python ≥ 3.12, `ruff check .` (E/F/W, line length 160, E501/E701 ignored), `pytest tests/ -v`, `python scripts/check_imports.py`. New tunables go in the `# Tunables` block of `fritz_utils.py` (below `_at_least_one`, as `TTS_MAX_CONCURRENCY` does), in `.env.example`, *and* in the root README's **Configuration Reference** (`CONTRIBUTING.md:85-88`), batched in one commit per `plans/README.md`'s "one batched config-knob PR". Butler-voiced errors go through `bot_adapters.fritz_error`. Nothing user-visible leaks a raw exception.

**Repo conventions (body, `desktop/`).** pnpm ≥ 10, Node ≥ 20, Rust stable with **MSRV 1.89** (DivoraVoice's `rust-version`; the lifted `mark_non_rude` uses `std::ptr::without_provenance_mut`, Rust ≥ 1.84 — Lorraine's 1.77 is too low), Tauri 2.11.x pinned to Lorraine's lockfile versions (tauri 2.11.3, tauri-build 2.6.3, tauri-plugin-global-shortcut 2.3.2, windows 0.61, webview2-com 0.38). `pnpm build` runs `check-no-input.mjs`, `check-no-css-animation.mjs` and the protocol check. **Type-check Rust with `cargo check --manifest-path src-tauri/Cargo.toml` (no feature needed; it doesn't clobber), and never run a bare `cargo test`/`cargo build` in `src-tauri/` without `--features custom-protocol`** — without it the debug exe is rebuilt as a dev build that loads `localhost:1431` and every window shows "refused to connect" (Lorraine's `CLAUDE.md`). PowerShell mangles `git commit -m` when the message has embedded double quotes or a bare `/`: use single-quoted ASCII, or pipe a heredoc to `git commit -F -` from a Bash shell.

**Source trees you need on disk, read-only:** `LorraineAtelier/` (private — copy plumbing, never gift content or history; the lift list is plan 14 Appendix B) and `DivoraVoiceMod/` (MIT — lift the files named in § 4 with their licence header intact).

**When the owner is not there**, apply the defaults in § 7 and say which you applied at the top of the PR description.

---

## 1. Target layout

```
MisterFritz/
  companion_protocol.py          # pydantic models: the wire, versioned (§ 2)
  companion_api.py               # WebSocketRoute /companion, sessions, turn runner, widget feed (Ph3)
  companion_tts.py               # SentenceChunker + TTS worker (Ph2, PR 9)
  google_mcp.py                  # MultiServerMCPClient → owner-only tools (Ph4, PR 15)
  main_local.py                  # the brain without Discord (P1)
  scripts/gen_protocol_ts.py     # companion_protocol.py → desktop/src/shared/protocol.ts
  tests/test_companion_protocol.py
  tests/test_companion_api.py
  tests/test_companion_tts.py
  tests/test_google_mcp.py
  desktop/
    package.json  pnpm-lock.yaml  vite.config.ts  tsconfig.json  vitest.config.ts
    tailwind.config.js  postcss.config.js
    index.html        # avatar window page
    prompt.html       # prompt window page
    control.html      # control (settings) window page
    wallpaper.html    # Ph3
    src/
      avatar/   main.tsx  VrmActor.ts  PacedLoop.ts  LipSync.ts  Bubble.tsx  state.ts  hitRect.ts
      prompt/   main.tsx  Prompt.tsx
      control/  main.tsx  Control.tsx  sections/{Brain,Avatar,Hotkeys,Appearance,About}.tsx
      wallpaper/ main.tsx  StudyCircle.ts  widgets/{Clock,Weather,Desk,Agenda,Inbox}.tsx   # Ph3
      shared/   companion.ts  protocol.ts(generated)  settings.ts  motion.ts  inOrder.ts(lifted)  tauri.ts
      divora/   tokens.css(synced)  theme.ts  fonts/*.woff2  sigils/{index.ts,*.svg}  components/*.tsx(ported)
    src-tauri/
      Cargo.toml  tauri.conf.json  build.rs  capabilities/default.json  icons/
      src/
        main.rs  lib.rs
        overlay_machine.rs   # lifted from DivoraVoice, charms/recorder removed
        overlay_windows.rs   # lifted from DivoraVoice, trimmed (§ 4.2)
        non_rude.rs          # mark_non_rude / unmark_non_rude (from divora-capture native/window.rs)
        windows_ext.rs       # park/unpark, WS_EX_TOOLWINDOW, set_focusable, monitor helpers
        cursor.rs            # 15 Hz cursor poll → `cursor` event
        hook.rs              # from Lorraine (generic WH_MOUSE_LL + rect gate)
        hotkeys.rs           # global-shortcut registration, PTT press/release edges
        tray.rs
        settings.rs          # from Lorraine's store (atomic save), new schema
        audio_in.rs          # Ph2: mic capture → PCM16 16 kHz
        kokoro_tts.rs        # Ph2, only if Q2 = (c): lifted divora-core tts
        power.rs             # from Lorraine
        workerw.rs  host.rs  # Ph3, from Lorraine
        autostart.rs         # Ph5, from Lorraine
    assets/avatar/placeholder.vrm      # CC0, tiny; the real model lives in %APPDATA%
    scripts/check-no-input.mjs  check-no-css-animation.mjs  sync-divora-tokens.mjs
    tests/visual/                      # Ph3
    README.md
```

Identifier `com.divora.fritz`; app data `%APPDATA%\com.divora.fritz\` holding `settings.json`, `companion.token`, `avatar\fritz.vrm`, `crash.log`, `startup.log`.

---

## 2. The companion protocol (normative)

Transport: WebSocket at `ws://{COMPANION_BIND}:{ADMIN_PANEL_PORT}/companion`. Text frames are UTF-8 JSON objects with a `type` field. Binary frames carry audio only and are always followed by an `audio_end` text frame. `protocol: 1` is negotiated in `hello`/`welcome`; a server that does not support the client's version closes with **4426**.

### 2.1 Models (`companion_protocol.py`)

```python
from typing import Literal, Annotated
from pydantic import BaseModel, Field, StringConstraints

PROTOCOL_VERSION = 1
ClientId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")]
Cap = Literal["text", "audio-in", "audio-out", "speak-local", "vrm", "widgets"]
TurnState = Literal["listening", "transcribing", "thinking", "speaking", "idle"]

# ── client → server ──────────────────────────────────────────
class ClientInfo(BaseModel):
    id: ClientId                     # becomes canonical "desktop-<id>"
    name: str = Field(max_length=64) # display name, e.g. "Nick's PC"
    caps: list[Cap]

class Hello(BaseModel):
    type: Literal["hello"]
    protocol: int = PROTOCOL_VERSION
    token: str
    client: ClientInfo

class Say(BaseModel):
    type: Literal["say"]; turn_id: str; text: str = Field(min_length=1, max_length=8000)

class AudioEnd(BaseModel):          # follows one or more binary frames
    type: Literal["audio_end"]; turn_id: str
    sample_rate: int = 16000; channels: int = 1; format: Literal["pcm16"] = "pcm16"

class Cancel(BaseModel):
    type: Literal["cancel"]; turn_id: str

class Subscribe(BaseModel):         # Ph3
    type: Literal["subscribe"]; widgets: list[str]

class Ping(BaseModel):
    type: Literal["ping"]; t: int

ClientFrame = Hello | Say | AudioEnd | Cancel | Subscribe | Ping

# ── server → client ──────────────────────────────────────────
class Welcome(BaseModel):
    type: Literal["welcome"]; protocol: int; identity: str; display_name: str
    server_caps: list[Cap]; thread_id: str

class State(BaseModel):
    type: Literal["state"]; turn_id: str; state: TurnState

class Stt(BaseModel):
    type: Literal["stt"]; turn_id: str; text: str

class Token(BaseModel):
    type: Literal["token"]; turn_id: str; delta: str
    restart: bool = False           # _DeltaEmitter's segment restart; client clears its buffer

class Progress(BaseModel):
    type: Literal["progress"]; turn_id: str; text: str

class Speech(BaseModel):            # brain-side TTS: one sentence of audio
    type: Literal["speech"]; turn_id: str; seq: int; mime: Literal["audio/wav"]; b64: str; final: bool

class Speak(BaseModel):             # body-side TTS (Q2 = c): text to synthesise locally
    type: Literal["speak"]; turn_id: str; seq: int; text: str; final: bool

class Done(BaseModel):
    type: Literal["done"]; turn_id: str; text: str; images: list[str] = []; interrupted: bool = False

ErrorCode = Literal["busy", "bad_frame", "turn_failed", "stt_failed", "not_allowed", "internal"]
class Error(BaseModel):
    type: Literal["error"]; turn_id: str | None = None; code: ErrorCode; text: str  # text is butler-voiced

class Widget(BaseModel):            # Ph3
    type: Literal["widget"]; name: str; payload: dict; ttl_s: int

class Reminder(BaseModel):          # Ph3/5
    type: Literal["reminder"]; schedule_id: int; text: str

class Pong(BaseModel):
    type: Literal["pong"]; t: int

ServerFrame = Welcome | State | Stt | Token | Progress | Speech | Speak | Done | Error | Widget | Reminder | Pong
```

`parse_client_frame(raw: str) -> ClientFrame` uses a discriminated union on `type`; a parse failure yields `Error(code="bad_frame")` and, for a frame before `hello`, close **4400**.

### 2.2 Session rules

- First frame must be `hello` within 5 s or close **4408**. Bad token → close **4401**. More than `COMPANION_MAX_CLIENTS` live sessions → close **4409**. Unsupported `protocol` → close **4426**.
- One active turn per session. A `say`/`audio_end` while a turn is active → `Error(code="busy")` referencing the *new* `turn_id`; the client must `cancel` first. (Barge-in is `cancel` + new turn; the body does this in one go.)
- `cancel` for an unknown or finished `turn_id` is ignored.
- Server sends `ping` every 20 s and closes **4000** after two missed `pong`s; client mirrors.
- Every turn emits exactly one `done` or one `error` with that `turn_id`, after all `token`/`speech`/`speak` frames for it.
- Identity: `identity = resolve_identity(canonical_user_id("desktop", client.id))`; `thread_id = f"{identity}#desktop"` (explicit, independent of `THREADS_PER_CHANNEL`, per plan 14 § Identity).

### 2.3 Example turn

```
→ {"type":"hello","protocol":1,"token":"…","client":{"id":"nick-pc","name":"Nick's PC","caps":["text","audio-in","audio-out","vrm"]}}
← {"type":"welcome","protocol":1,"identity":"discord-1234","display_name":"Nick","server_caps":["text","audio-in","audio-out"],"thread_id":"discord-1234#desktop"}
→ {"type":"say","turn_id":"7f3…","text":"good evening"}
← {"type":"state","turn_id":"7f3…","state":"thinking"}
← {"type":"token","turn_id":"7f3…","delta":"Good evening, sir.","restart":true}
← {"type":"token","turn_id":"7f3…","delta":" The hour is late"}
← {"type":"state","turn_id":"7f3…","state":"speaking"}          # Ph2 only
← {"type":"speech","turn_id":"7f3…","seq":0,"mime":"audio/wav","b64":"UklG…","final":false}
← {"type":"done","turn_id":"7f3…","text":"Good evening, sir. The hour is late…","images":[],"interrupted":false}
← {"type":"state","turn_id":"7f3…","state":"idle"}
```

### 2.4 TypeScript generation

`scripts/gen_protocol_ts.py` imports `companion_protocol`, walks the unions and writes `desktop/src/shared/protocol.ts` with one `interface` per model, a `ClientFrame`/`ServerFrame` union, `PROTOCOL_VERSION`, and `isServerFrame(x): x is ServerFrame`. `--check` exits 1 if the file on disk differs. Run in both CI jobs.

---

## 3. Brain-side work

### P0 — declare `websockets` (XS)

`pyproject.toml` core `dependencies`: add `"websockets>=13"` next to `uvicorn`, and change the uvicorn comment to say the WebSocket route needs it. `requirements.txt` already pins `websockets==15.0.1` (transitive) — leave the pin, it now has a direct consumer. **Test:** `python -c "import websockets"` in CI's core job (it is already imported transitively, so this is a declaration fix; the test is `scripts/check_imports.py` not regressing).

### P1 — the brain without Discord (S)

*Change sites.* `admin_panel.py::start_admin_panel(schedule_manager=None) -> Optional[int]` already owns the port guard and the thread; nothing to extract. **Add** `main_local.py`:

```python
"""Run the web surfaces (admin panel, web chat, companion) without Discord or Telegram."""
import signal, threading, logging
from dotenv import load_dotenv

def main() -> int:
    load_dotenv()
    from fritz_utils import ADMIN_PANEL_PASSWORD, ADMIN_PANEL_PORT
    import identity_store
    from admin_panel import start_admin_panel
    logging.basicConfig(level=logging.INFO)
    if not ADMIN_PANEL_PASSWORD:
        print("ADMIN_PANEL_PASSWORD is not set; nothing to serve."); return 2
    # identity_store initialises itself lazily on first record()/display_name(); nothing to call here.
    port = start_admin_panel(schedule_manager=None)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    print(f"Serving on 127.0.0.1:{port}. Ctrl+C to stop.")
    stop.wait()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Verify what `main_discord.py` initialises before `start_admin_panel` (identity store, Chroma warm-up, `prewarm_models`) and mirror the ones the web surfaces need. Add `main_local` and `companion_api`/`companion_protocol` to `pyproject.toml`'s explicit `py-modules` list (`pyproject.toml:172`). **Tests:** `tests/test_main_local.py` — importing `main_local` has no side effects; `main()` returns 2 with no password (monkeypatch). **Acceptance:** `python main_local.py` serves `/chat` with Discord unconfigured.

### P2 — `MessageSource.DESKTOP_*` (XS)

`fritz_utils.py` `class MessageSource(Enum)`: append `DESKTOP_TEXT = 6,` and `DESKTOP_VOICE = 7,` (keep the file's trailing-comma style — the existing members are tuples and the code compares by identity). `KNOWN_PLATFORMS`: add `"desktop"`; update the docstring of `canonical_user_id` that lists platforms.

`mister_fritz.py::get_source_info` — add before the CLI fallthrough:

```python
    elif source == MessageSource.DESKTOP_TEXT:
        return f"User is typing to you at their desk, where you are present as a figure on their screen ({who})"
    elif source == MessageSource.DESKTOP_VOICE:
        return f"User is speaking to you at their desk ({who}). Please answer in 30 words or less."
```

**Tests:** extend `tests/test_agent_tools.py::test_get_source_info_*` (`:60-66`) with both new members; `canonical_user_id("desktop", "nick-pc") == "desktop-nick-pc"`.

### P3 — cooperative cancel (M) · *also fixes the web chat's "Enough"*

*Change sites.*

1. `mister_fritz.py::ask_stuff` — add keyword-only `cancel_event: threading.Event | None = None`; build `turn_flags = {"interrupted": False}` and put both `"cancel_event": cancel_event` and `"turn_flags": turn_flags` into `config["metadata"]` (the dict built at `:984-986`). LangGraph shallow-copies `metadata` before a node runs, so the executor must mutate the **inner** `turn_flags` dict, never add keys to `metadata` itself. Add `"interrupted": turn_flags["interrupted"]` to the returned dict — on the normal path **and** the `ThreadBusy` path (`:1025`), which returns the same keys.
2. `mister_fritz.py::executor` — read `cancel_event = metadata.get("cancel_event")` beside the other metadata reads; in the inner loop:

   BEFORE
   ```python
   for mode, payload in _until_the_bound(
       agent.stream(inputs, config=get_config_values(config), stream_mode=stream_modes),
       lambda: bound_hit.append(True),
   ):
       if mode == "values":
   ```
   AFTER
   ```python
   interrupted = False
   for mode, payload in _until_the_bound(
       agent.stream(inputs, config=get_config_values(config), stream_mode=stream_modes),
       lambda: bound_hit.append(True),
   ):
       if cancel_event is not None and cancel_event.is_set():
           interrupted = True
           break                      # closes the generator → closes the Ollama stream
       if mode == "values":
   ```
   After `emitter.flush()`: when `interrupted`, the reply text is the text streamed so far plus `INTERRUPTED_MARK = " [interrupted]"` — keep a running concatenation across segments in the executor for this, because `_DeltaEmitter`'s `accumulated` is per *segment*, not per reply — and set `metadata["turn_flags"]["interrupted"] = True` (mutating the inner dict, see step 1).
3. `admin_panel.py::chat_stream` — create `cancel_event = threading.Event()` per request, pass it to `ask_stuff`, and set it when the client disconnects (`await request.is_disconnected()` checked in `_event_generator`'s poll loop). Update the "Enough" notice copy in `admin_templates/chat.html` to say the reply has been stopped.

**Tests** (`tests/test_mister_fritz.py` or a new `tests/test_cancel.py`): a fake `agent.stream` that yields 50 `("messages", (AIMessageChunk("x"), {}))` with a 10 ms sleep each and records `GeneratorExit` on close; set the event after 5 → the stored reply ends with `[interrupted]`, fewer than 50 chunks were consumed, `GeneratorExit` was seen (proves the upstream stream is closed), and `ask_stuff(...)["interrupted"] is True`. A second test with no event → unchanged behaviour. **Acceptance:** web chat "Enough" measurably stops Ollama (`ollama ps` load drops / GPU utilisation falls).

### P4 — knobs (XS, one commit)

`fritz_utils.py` `# Tunables` block, below `_at_least_one`, with the file's `# ---` banner style:

```python
# ---------------------------------------------------------------------------
# Desktop companion (plan 14)
# ---------------------------------------------------------------------------
COMPANION_ENABLED: bool = os.environ.get("COMPANION_ENABLED", "false").lower() in ("1", "true", "yes")
COMPANION_TOKEN: str | None = os.environ.get("COMPANION_TOKEN") or None
COMPANION_BIND: str = os.environ.get("COMPANION_BIND", "127.0.0.1")   # Ph5 may widen to a Tailscale address; never 0.0.0.0
COMPANION_MAX_CLIENTS: int = _at_least_one("COMPANION_MAX_CLIENTS", "2")
COMPANION_TTS: str = os.environ.get("COMPANION_TTS", "local")         # "local" (body speaks: `speak` frames) | "xtts" | "off"
GOOGLE_MCP_ENABLED: bool = os.environ.get("GOOGLE_MCP_ENABLED", "false").lower() in ("1", "true", "yes")
```

`.env.example`: a matching commented block under the admin-panel section, with the sentence "COMPANION_TOKEN is deliberately separate from CHAT_PASSWORD and ADMIN_PANEL_PASSWORD (decision 32)". `COMPANION_BIND` only changes the uvicorn `host` when the companion is enabled — today's `127.0.0.1` stays the default; refuse `0.0.0.0` at startup with a clear log line.

### P5 — test scaffold (XS)

`tests/test_companion_api.py` with a fixture that builds `admin_panel.create_app(PASSWORD, schedule_manager=None)` under `COMPANION_ENABLED=True, COMPANION_TOKEN="t0k"` (monkeypatch the `fritz_utils` names, then reload-free: `companion_api` must read them at call time like `is_admin` does) and a `fake_ask_stuff` that replays scripted deltas through `streaming_callback` and honours `cancel_event`. Use `with client.websocket_connect("/companion") as ws:`. **Acceptance:** the fixture connects and is closed cleanly with 4401 on a bad token — the first real test lands in PR 1.

### P6 — the VRM (owner)

VRoid Studio → VRM 1.0 → `%APPDATA%\com.divora.fritz\avatar\fritz.vrm`. Confirm in VRoid's export dialog that expressions `aa ih ou ee oh happy angry sad relaxed surprised blink blinkLeft blinkRight lookUp lookDown lookLeft lookRight` are present (they are by default). Commit `desktop/assets/avatar/placeholder.vrm` (any CC0 VRM 1.0 under 2 MB) for CI and first run.

### PR 1 — `companion_api.py` (brain, M) · depends on P0–P5

```python
# companion_api.py — structure
import asyncio, base64, json, secrets, threading, time, uuid
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.routing import WebSocketRoute
from fritz_utils import (MessageSource, canonical_user_id, resolve_identity, ROOT_USER)
import fritz_utils  # read COMPANION_* at call time
from bot_adapters import run_blocking, fritz_error
from companion_protocol import *

_SESSIONS: dict[str, "Session"] = {}        # keyed by ws id; guarded by _SESSIONS_LOCK

class Session:
    ws: WebSocket; canonical_id: str; identity: str; display_name: str; thread_id: str   # identity = resolve_identity(canonical_id)
    caps: set[str]; out: asyncio.Queue[ServerFrame]
    active: "Turn | None"; audio_buf: bytearray; last_pong: float

class Turn:
    turn_id: str; cancel: threading.Event; started: float

async def companion_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hello = await _handshake(ws)                  # 5 s, 4400/4401/4409/4426
    session = _open_session(ws, hello)            # identity_store.record(identity, name, "desktop")
    sender = asyncio.create_task(_drain(session)) # out → ws.send_text(frame.model_dump_json())
    heart = asyncio.create_task(_heartbeat(session))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect": break
            if "bytes" in msg and msg["bytes"] is not None:
                session.audio_buf += msg["bytes"]; continue
            frame = parse_client_frame(msg["text"])  # Error(bad_frame) on failure
            await _dispatch(session, frame)
    finally:
        _close_session(session); sender.cancel(); heart.cancel()

def routes() -> list[WebSocketRoute]:
    return [WebSocketRoute("/companion", companion_endpoint, name="companion")]
```

`_dispatch`: `say` → `_start_turn(session, text, MessageSource.DESKTOP_TEXT)`; `audio_end` → PR 9; `cancel` → `session.active.cancel.set()` if ids match; `subscribe` → Ph3; `ping` → `Pong`.

`_start_turn` refuses with `Error(busy)` if `session.active`; otherwise builds `Turn`, emits `State(thinking)`, and runs on the bounded pool:

```python
loop = asyncio.get_running_loop()
def emit(frame): loop.call_soon_threadsafe(session.out.put_nowait, frame)
def streaming_callback(delta, accumulated, restart): emit(Token(turn_id=t.turn_id, delta=delta, restart=restart))
def progress_callback(text): emit(Progress(turn_id=t.turn_id, text=text))
result = await run_blocking(ask_stuff, text, source, session.canonical_id,   # the unresolved desktop-<id>; ask_stuff resolves one hop itself (:960)
                            progress_callback=progress_callback, streaming_callback=streaming_callback,
                            thread_id=session.thread_id, display_name=session.display_name,
                            cancel_event=t.cancel)
emit(Done(turn_id=t.turn_id, text=result["text"], images=[...], interrupted=result.get("interrupted", False)))
emit(State(turn_id=t.turn_id, state="idle"))
```

Exceptions → `Error(code="turn_failed", text=fritz_error("companion_turn", e))`, then `State(idle)`. `audit_log("companion_turn", user_id=identity, ...)` on both paths, mirroring `chat_stream`.

*Registration.* In `admin_panel.create_app`, after the `/static` mount: `if fritz_utils.COMPANION_ENABLED and fritz_utils.COMPANION_TOKEN: routes += companion_api.routes()` (import inside the function to keep `admin_panel` importable without the new module's dependencies). In `start_admin_panel`, `host=fritz_utils.COMPANION_BIND if COMPANION_ENABLED else "127.0.0.1"`. All **three** middlewares (`_SecurityHeadersMiddleware`, `_BasicAuthMiddleware`, `_SameOriginMiddleware`, `admin_panel.py:1327-1332`) are `BaseHTTPMiddleware`, and Starlette passes non-`http` scopes straight through it (`starlette/middleware/base.py`, `if scope["type"] != "http"`), so no exemption code is needed — which also means **the token is the only gate on `/companion`**: no Basic auth, no CSP, no same-origin check. Treat `COMPANION_TOKEN` accordingly (≥ 32 random chars; `scripts/setup.py` generates it with `secrets.token_urlsafe(32)`).

**Tests** (`tests/test_companion_api.py`): handshake ok → `welcome.identity == "desktop-nick-pc"` (no links configured) and `thread_id` ends `#desktop`; bad token → close 4401; no hello in time → 4408 (use a short monkeypatched deadline); third client → 4409; one streamed turn → `state thinking`, tokens in order with `restart` on the first, `done`, `state idle`; `busy` on overlapping `say`; cancel mid-stream → `done.interrupted is True` and the fake saw its event; heartbeat `ping`→`pong`; **middleware** — the connect succeeds with no `Authorization` header and no `Sec-Fetch-Site`; `monkeypatch.setitem(fritz_utils.IDENTITY_LINKS, "desktop-nick-pc", "discord-1")` → `welcome.identity == "discord-1"` (the dict is parsed at import from `alias=primary,…`; setting the env var in a test does nothing). **Acceptance:** `websocat ws://127.0.0.1:8001/companion` by hand completes the § 2.3 exchange against the live brain.

*As built (PR 1), where it departs from the text above:*

- **`COMPANION_BIND` is not applied yet; it stays reserved until Phase 5.** `host=COMPANION_BIND` would move the whole app, so the admin panel and the web chat would leave loopback with `/companion`, and a desk on the same PC could no longer reach `127.0.0.1`. The server stays on `127.0.0.1` (`admin_panel.server_config`). Phase 5 has to listen on a second address for `/companion` alone.
- **`ping` and `pong` are in both unions**, since § 2.2 has the client mirror the server's heartbeat.
- **The client's name is the machine's, so it is recorded against `desktop-<id>`**, never against a linked identity, whose name is a person's. Recording it against the linked identity would rename the owner "Nick's PC". For the same reason `ask_stuff` gets no `display_name`; it looks the name up itself.
- **`done.images` are `data:` URLs**, read from the web chat's asset roots and types only, each up to 8 MiB. `/chat/assets` needs the chat cookie, which the desk doesn't have.
- **Bounds:** `turn_id` is 1–64 of `[A-Za-z0-9_-]`, and `say.text` is stripped before its length is checked. A broken `say` that names a well-formed `turn_id` is answered `bad_frame` under that id, so the turn the client is waiting on still ends.
- **Barge-in works in one go, as § 2.2 has the body send it.** A `say` that arrives once the running turn is cancelled waits for that turn to end and then starts: `done` and `idle` for the old turn, then `thinking` for the new. Refusing it `busy` would lose it, because a cancel only takes effect at the model's next token, or when the tool in flight returns. One `say` waits like this; another gets `busy`, and a `cancel` naming the waiting one drops it (`done` with empty text and `interrupted`, then `idle`). A turn cancelled, or whose desk left, before a pool thread is free never reaches `ask_stuff`. That matches how `ask_stuff` treats a turn cancelled while queued.
- **A newer client's hello gets 4426 even when version 1 cannot parse it** (a cap it does not know, say), but only when the token is right, so a stranger still learns nothing. Other malformed hellos get 4400.
- **Frame limits:** uvicorn's `ws_max_size` is 1 MiB (its default was 16 MiB, read whole before the token is seen); 1 MiB still fits a 30-second PCM16 clip in one frame for PR 9. Over it, the close is 1009. Text frames over 128 Ki characters are refused unread, answered under the turn they name if that is in their first or last KiB. `permessage-deflate` is off: over loopback it saves nothing, and it let a 16 KiB frame on the wire become 16 MiB to parse. A refused frame is logged with `%r`, because pydantic quotes the client's `type`.
- **Not served yet:** `server_caps` is `["text"]`; a binary frame is dropped unread; `audio_end` and `subscribe` are answered `bad_frame`.
- **uvicorn runs with `ws="websockets-sansio"`**, because `"auto"` picks the deprecated `websockets.legacy`. The sans-I/O protocol sends no keepalive pings, so the app's heartbeat is the only one.

### PR 9 — audio turn + brain-side TTS (brain, M) · depends on PR 1 · Phase 2

`_dispatch(audio_end)`: write `session.audio_buf` as a WAV (`wave` module, 16 kHz mono 16-bit) to `temp_audio/companion_<turn>.wav`; clear the buffer; `State(transcribing)`; `text = await run_blocking(stt.transcribe, path)`; `None`/`""` → `Error(stt_failed, fritz_error(...))`; else `Stt(text)` and `_start_turn(..., MessageSource.DESKTOP_VOICE)`. Delete the temp file in `finally`.

`companion_tts.py`:

```python
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
class SentenceChunker:
    """Feed streaming deltas; yields complete sentences ≥ MIN_CHARS; flush() yields the tail."""
    def __init__(self, min_chars: int = 12): ...
    def feed(self, delta: str, restart: bool) -> list[str]: ...   # restart drops the buffer
    def flush(self) -> list[str]: ...
```

Mode by `COMPANION_TTS`: `"local"` → each sentence becomes `Speak(seq, text, final)` immediately (no audio work in the brain); `"xtts"` → a per-session single-worker thread calls a shared `TTSEngine` (`get_tts_engine()` lazy accessor added to `tts.py`, replacing the construction at `main_discord.py:249-250` so both entry points share one instance), reads the returned **WAV path** and emits `Speech(seq, b64 wav, final)`. `TTS_MAX_CONCURRENCY` is today an `asyncio.Semaphore` on the Discord cog (`bot_commands.py:581`), not on the engine — move the gate into `tts.py` as a `threading.Semaphore(TTS_MAX_CONCURRENCY)` taken inside `generate_speech`, so Discord and the companion share one limit; `"off"` → nothing. `State(speaking)` is emitted before the first `speech`/`speak`; `Done` follows the last one, which carries `final=true`. Cancel flushes the chunker and stops the worker between sentences.

**Tests:** `SentenceChunker` on a scripted delta sequence (sentence splits, restart drops, flush tail, min-chars merge); audio turn with a fake `transcribe`; `COMPANION_TTS=local` emits `speak` frames in order ending `final`; `xtts` with a fake engine emits `speech` with increasing `seq`; cancel between sentences stops at the next boundary. **Acceptance:** § 4.6.

### PR 14 — widget feed + `DeliverySink` (brain, M) · Phase 3

`companion_api`: `subscribe` starts one `asyncio.Task` per session that calls `widgets.compute(name)` on a per-widget cadence (`clock` 60 s — the body ticks seconds itself; `weather` 15 min via Open-Meteo `httpx`; `desk` on every `done`; `agenda`/`inbox` 5 min/2 min in Phase 4) and emits `Widget(name, payload, ttl_s)`. Payload schemas live in `companion_protocol.py` as `WeatherPayload`, `DeskPayload`, … and are generated to TS.

`scheduler.py`: introduce

```python
class DeliverySink(Protocol):
    def deliver(self, identity: str, text: str, *, schedule_id: int) -> bool: ...
```

with `DiscordChannelSink(bot, channel_id)` wrapping today's `channel.send` path and `CompanionSink()` that emits `Reminder` to every live session whose identity matches. `_run_task` iterates the sinks registered for the identity; the Discord sink remains first so behaviour is unchanged when no desk is connected. Answer plan 13 Q1 (timezone) before this lands; store the zone the reminder was created in.

### PR 15–17 — Google, read-only, owner-only (brain, M+S+M) · Phase 4

`google_mcp.py`:

```python
async def load_google_tools() -> dict[str, tuple[BaseTool, str]]:
    client = MultiServerMCPClient({"google": {
        "transport": "stdio", "command": "uvx",
        "args": ["workspace-mcp", "--tools", "gmail", "drive", "calendar"],
        "env": {**os.environ, "WORKSPACE_MCP_PERMISSIONS": "gmail:readonly drive:readonly calendar:readonly",
                "GOOGLE_MCP_CREDENTIALS_DIR": str(GOOGLE_CRED_DIR)},
    }})
    tools = await client.get_tools()
    return {t.name: (t, t.description) for t in tools}
```

Loaded once at startup when `GOOGLE_MCP_ENABLED` (cache in a module global; a failure logs and leaves the dict empty — Fritz must boot without Google). **Gate:** in `mister_fritz.executor`, `extra_tools` gains the Google dict **only if** `resolve_identity(user_id) == ROOT_USER`. Put the check in one helper, `owner_only_tools(user_id) -> dict`, and nowhere else. **Tests** (`tests/test_google_mcp.py`): with a fake loaded dict, the tool list for `ROOT_USER` contains `gmail_*`; for `discord-999` it contains none; with `GOOGLE_MCP_ENABLED=false` nothing is attempted. Verify the exact CLI flags against the installed `workspace-mcp --help` before relying on them — the permission variable name was read from its README, not run.

PR 16 is the owner's OAuth setup (plan 14 PR 16; the one non-negotiable step is *Publishing status → In production*). PR 17 adds `agenda`/`inbox` widget computations that call the loaded MCP tools directly with `tool.ainvoke({...})` on the widget cadence — never through the LLM.

---

## 4. Body-side work (`desktop/`)

### 4.0 Scaffold (part of Spike 0, kept for PR 2)

`pnpm create tauri-app` is not used; copy Lorraine's `package.json`/`vite.config.ts`/`tsconfig.json`/`vitest.config.ts` shapes (multi-page Vite, dev port **1431** so both apps can run), then add `solid-js`, `vite-plugin-solid`, `tailwindcss@3`, `three`, `@pixiv/three-vrm@^3`, `@pixiv/three-vrm-animation@^3`, `@tauri-apps/api`, `@tauri-apps/plugin-global-shortcut`. `tsconfig`: `"jsx": "preserve", "jsxImportSource": "solid-js"` (Lorraine's leftover, now true).

`src-tauri/tauri.conf.json` essentials:

```json
{
  "productName": "Mister Fritz", "identifier": "com.divora.fritz", "version": "0.1.0",
  "build": { "beforeDevCommand": "pnpm dev", "devUrl": "http://localhost:1431", "beforeBuildCommand": "pnpm build", "frontendDist": "../dist" },
  "app": { "windows": [
    { "label": "control", "url": "control.html", "title": "Mister Fritz", "width": 900, "height": 640, "minWidth": 760, "minHeight": 520, "decorations": false, "visible": false, "center": true, "additionalBrowserArgs": "<LORRAINE_ARGS>" },
    { "label": "prompt", "url": "prompt.html", "width": 640, "height": 120, "decorations": false, "transparent": true, "alwaysOnTop": true, "skipTaskbar": true, "shadow": false, "resizable": false, "focus": false, "visible": false, "additionalBrowserArgs": "<LORRAINE_ARGS>" }
  ], "security": { "csp": null } },
  "bundle": { "active": true, "targets": ["nsis"], "windows": { "nsis": { "installMode": "currentUser" } } }
}
```

`<LORRAINE_ARGS>` is Lorraine's string, byte-identical in every window **and** in `overlay_windows.rs`'s builder (`.additional_browser_args(...)`), or WebView2 forks environments. The avatar window is built at runtime by the lifted overlay code (label `avatar`), as DivoraVoice does.

`capabilities/default.json`: `core:default`, `core:window:allow-set-ignore-cursor-events`, `core:window:allow-set-position`, `core:window:allow-set-size`, `core:window:allow-show`, `core:window:allow-hide`, `core:window:allow-set-focus` (prompt only — scope by window label `prompt`), `global-shortcut:allow-register`, `global-shortcut:allow-unregister`, `core:event:default`, for windows `["control","prompt","avatar"]`.

### 4.1 Spike 0 (1 day) — gates PR 2

Build the scaffold with only the avatar window and this `index.html`: a three.js scene (see `VrmActor` § 4.3) loading `placeholder.vrm`, idle breathing, `PacedLoop` at 30 fps. Run the table from plan 14 § Spike 0 for a working session. Record results in `desktop/docs/SPIKE0.md` (foreground log, PresentMon capture summary, GPU %, screenshots of edges). **Pass → PR 2. Fail → stop; plan 14 Fallback A/B is an owner decision.**

Foreground logger for the spike (Rust, debug-only): a thread calling `GetForegroundWindow` at 10 Hz, mapping to PID with `GetWindowThreadProcessId`, writing a line whenever the PID equals ours.

### 4.2 Rust: the overlay (lifted) and the shell — PR 2

**`overlay_machine.rs`** — copy from DivoraVoice verbatim, then delete the recorder paths: `Input::HideWanted(bool)`, `Input::Capturing(bool)`, `CAPTURE_ACK`, the `affinity_call` helper and the tests that exercise them (about 18 references; `affinity_call` tests near `:2109`). Rename `OVERLAY_TITLE` to `"Mister Fritz"`, `TRAY_LOCK_LABEL` to "Lock Fritz in place", `HOTKEY_ARRANGE` to `"avatar:arrange"`. Keep `Input::{Open, Built, BuildFailed, Ready, Painted, Placed, Watchdog, Close, CloseRequested, Destroyed, Mode, Alive, LeaseTick}` (`Placed { fit }` is used by several tests — keep it), `Effect::{Build, ApplyWindow, Show, Focus, ReturnFocus, ArmWatchdog, ArmLease, CloseWindow, DestroyWindow, Emit, RefreshTray}`, `Mode::{Locked, Arrange}`, `ModeRequest::{Arrange, Toggle, …}`, `WATCHDOG` 10 s, `LEASE_TICK`/`LEASE_MISSES`, and **every test for the inputs you kept**. The charm/asset code is not in this file (it lives in `overlay_windows.rs`). Add `Input::Park { on: bool }` → `Effect::Park { on: bool }` (position change; no show/hide) and a `parked: bool` on `WindowState`.

**`overlay_windows.rs`** — copy, then: drop charm serving, `ServeGate`, recorder hooks and `divora_capture` imports (replace `mark_non_rude` with `crate::non_rude::mark_non_rude`); keep `build_window` with the flags exactly as DivoraVoice has them plus `.focusable(false)` (WebviewWindowBuilder method since Tauri 2.8 — verify the exact name against the pinned tauri docs; DivoraVoice doesn't use it) and `.additional_browser_args(LORRAINE_ARGS)`; keep `set_ignore_cursor_events(true)` *before* the HWND is stored; keep the ready handshake and lease beats — DivoraVoice's commands are `overlay_handshake` / `overlay_alive`; rename to `avatar_ready` / `avatar_beat` (Tauri commands are named after the Rust fn), called from the page on first paint and every 1 s from the paced loop — the watchdog, `focus`/`return_focus` via `SetForegroundWindow`, and the test `the_overlay_never_calls_set_focus` (the needles stay — the file still must not call it). Implement `Effect::Park` as `set_position(PhysicalPosition{x: -10000, y: 0})` and restore to the saved rect.

**`non_rude.rs`** — `mark_non_rude(hwnd: isize) -> bool` / `unmark_non_rude` from `divora-capture/src/native/window.rs` (`SetPropW(hwnd, w!("NonRudeHWND"), 1)`), `cfg(windows)`.

**`windows_ext.rs`**:

```rust
pub fn apply_toolwindow(hwnd: isize);          // GWL_EXSTYLE |= WS_EX_TOOLWINDOW, &= !WS_EX_APPWINDOW; SetWindowPos(... SWP_FRAMECHANGED|SWP_NOACTIVATE|SWP_NOMOVE|SWP_NOSIZE)
pub fn monitor_covered(hwnd: isize) -> bool;   // Lorraine workerw.rs::monitor_covered, generalised to "monitor containing hwnd"
pub fn work_area_of(hwnd: isize) -> Rect;      // for default placement bottom-right
```

**`cursor.rs`** — thread, 15 Hz, `GetCursorPos`; emits `cursor {x, y, over_avatar: bool}` to `avatar` only when moved ≥ 2 px; `over_avatar` from the current hit rect (so the page needn't do the math).

**`hook.rs`** — Lorraine's, with `HitTarget` reduced to `enum HitTarget { Avatar }` and the exposure test replaced by "rect only" (the avatar is topmost; `WindowFromPoint` would skip our click-through window). Right-button-up inside → `tray::popup_at(x, y)`; left double-click → `prompt::open()`; both return `LRESULT(1)` to swallow the click so the window behind Fritz doesn't also receive it (Lorraine's buddy-drag does the same). Left-drag is **not** handled here in Locked mode (moving is Arrange's job).

**`hotkeys.rs`** — registers from `settings.hotkeys` via `tauri_plugin_global_shortcut::Builder::with_handler`; emits `hotkey {id, state: "pressed"|"released"}`; ids `prompt`, `ptt`, `arrange`, `park`. PTT is a **held level**, the DivoraVoice way (`set_ptm_held`-style, `lib.rs:3413`): `pressed` sets `ptt_held = true`, `released` sets it `false`, both idempotent, and the page reacts to level changes — so a reordered or duplicated edge cannot leave the mic stuck on.

**`tray.rs`** — `Show/Hide Fritz` (park), `Arrange…`/`Lock in place`, `Mute`, `Appearance ▸` (Mood: Ink/Dusk/Midnight · Accent: Brand/Abyssal/Ember · Light theme ✓ · Motion: Functional/Ambient/Rich), `Settings…`, `Quit`. Re-created on `TaskbarCreated` (Tauri handles this; verify).

**`settings.rs`** — Lorraine's store (load/get/atomic save/corrupt-quarantine) with this schema:

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
  pub version: u32,                                  // 1
  pub brain: Brain { url: String /* ws://127.0.0.1:8001/companion */, client_id: String, client_name: String },
  pub avatar: Avatar { x: i32, y: i32, w: u32, h: u32, scale: f32, parked: bool, vrm_path: Option<String>, lookat_cursor: bool },
  pub theme: Theme { theme: String /*dark|light*/, mood: String /*ink*/, accent: String /*brand*/, motion: f32 /*0.6*/, grain: bool, motion_user: bool },
  pub hotkeys: Hotkeys { prompt: String, ptt: String, arrange: String, park: String },  // "Ctrl+Alt+F", "Ctrl+Alt+Space", "Ctrl+Alt+A", "Ctrl+Alt+H"
  pub audio: Audio { input_device: Option<String>, output_device: Option<String>, muted: bool, tts_voice: String /* "bm_george" */ },
  pub widget_layout: HashMap<String, WidgetPos>,     // Ph3, Lorraine's WidgetPos
  pub run_at_startup: bool,                          // Ph5
}
```

The token is **not** in settings: `companion.token` beside it, `0600`-equivalent (owner-only ACL via `icacls` is overkill — document that `%APPDATA%` is already per-user), written by the control window's "Connect" section.

**Commands** (all `#[tauri::command]`, sync unless noted): `get_settings() -> Settings`, `set_setting(path: String, value: serde_json::Value) -> Settings` (one generic setter; emits `settings:changed` to all windows), `set_avatar_rect(rect)` (from the page; physical px, window-local → stored as screen px), `avatar_ready()`, `avatar_beat()`, `set_arrange(on: bool)`, `set_parked(on: bool)`, `open_prompt()`, `close_prompt()`, `get_token_present() -> bool`, `set_token(token: String)`, `companion_url() -> String`. **Events:** `cursor`, `hotkey`, `power:display`, `settings:changed`, `avatar:status` (the lifted `OverlayStatus`), `monitor:covered`.

**Startup (`lib.rs::run`)**: `set_dpi_awareness()` (Lorraine) → settings load → tray → hotkeys → `power::start` → overlay machine `Input::Open` (builds `avatar`) → cursor thread → hook (if enabled). `control` window shows on first run (no token) or from the tray.

**Tests (`cargo test --features custom-protocol --lib`, never bare)**: all lifted `overlay_machine` tests; `Park` round-trip; `the_overlay_never_calls_set_focus`; settings load/save/corrupt; hotkey edge guard; `apply_toolwindow` style math on a fake ex-style value.

*As built (PR 2, first commit: settings, logs, the display watch).* PR 2 lands in three commits, each reviewed: this one, the cursor look-at, then the control window.
- **Settings** are this schema plus `diagnostics.foregroundLog`. `avatar.rect` is an `Option` (`{x, y, width, height}`) rather than bare `x, y, w, h`, so "never placed" has a value. A file that doesn't parse goes to `settings.corrupt-<unix seconds>.json`, each under its own name; one that exists but can't be read is never saved over (this run's changes stay in memory). Saves are serialised, durable and atomic, and a failed one is tried again by the next change or the saver's next look. A value serde accepts but Fritz can't use (a hand-typed `Win+Shift+H`: global-hotkey says `Super`) is put back to its default at load, alone, with a `startup.log` line, so it can't block later changes. Spike 0's `avatar-state.json` is imported once (then renamed `avatar-state.imported.json`). `set_setting(path, value)` takes a camelCase path that must already exist, replaces an object only whole, re-checks the result (client id pattern, theme names, ranges, hotkeys that parse), and refuses `version`, `avatar.rect`, `avatar.parked` and any group holding them: those are the shell's, saved as they change (every 5 s and at Quit, as Spike 0 did). A change is applied to the store as it is under the lock, so concurrent changes (async commands run in parallel) and the saver never undo each other, and every window hears `settings:changed` with all the settings, in the order the changes were made.
- **Commands** keep Spike 0's names (`avatar_handshake` with its two stages, `avatar_beat`, `set_avatar_mode`, `avatar_status`, `avatar_vrm`) rather than this section's `avatar_ready`/`set_arrange`; added `get_settings`, `set_setting`, `power_display`. All are `async fn` in `lib.rs`, held to the wire fixture by a test. `avatar_vrm` reads `avatar.vrmPath` when it is set.
- **Hotkeys**: `arrange` and `park` from the settings, read at start; one that doesn't parse, or that another of ours has, falls back to its default. Live rebinding comes with the Hotkeys section (PR 4). The ids stay `avatar:arrange` and `avatar:park`.
- **`power.rs`** (Lorraine's) watches the display and resume only. The lock screen stays the cover watch's (WTS), since Lorraine's merged "locked" into "display off" and a display waking to show the lock screen would have unpaused him. `power:display` is sent only on a change, `power_display` answers a page that starts later, and the page draws only while `shouldDraw(status, displayOn)` (`src/avatar/run.ts`), so a status event no longer restarts the loop with the display off. Arrange draws on with the display off: the loop beats its lease, which would otherwise end it as "stopped responding".
- **Logs**: `startup.log` and `crash.log` beside the settings, the last kept as `.old` past 64 KiB; the panic hook chains to the default; `guard` wraps the shell's main-thread effects, the tray, the hotkey handler and the power watch's window procedure.
- **Spike 0's foreground logger** stays, opt-in through `diagnostics.foregroundLog` rather than debug-only (the Phase 1 gate runs it in the release exe), writing `diagnostics\foreground.log`. The tray's *Open the spike logs* is *Open Fritz's logs*.
- **Not in PR 2:** Mute (PR 8: there is nothing to mute until then), the Appearance submenu (PR 3, with the theme), the prompt window (PR 4), `hook.rs` (PR 5).

*As built (PR 2, second commit: the cursor look-at).*
- **`cursor.rs`** polls at 15 Hz and sends `cursor {active, x, y, width, height, over}` to the avatar window only: x and y from his window's top-left in physical pixels, with the window's size so the page scales to CSS pixels without knowing the monitor's DPI. Only a move of 2 px or more, or a change in his window, is sent, and only while he is open, not parked, the display is on and `avatar.lookatCursor` is set. A pause (parked, display off) sends nothing and forgets nothing, so shown again over a still cursor he hasn't glanced away; a stop (no window, or look-at turned off) sends one `active: false` and the page looks ahead. `over` is the window's rectangle; the model's own hit rect is PR 5's. Since a still cursor sends nothing, the page also asks `cursor_now` at start (taken only if no event came while asking), and keeps the last frame, working out the gaze from it every frame: a cursor that hadn't moved since he opened, or an event that came before the model loaded, is still looked at (the live check found the first missing), and the gaze follows his head when Arrange resizes him.
- **The page** (`src/avatar/look.ts`) turns the cursor's offset from his eyes on the page into a gaze, `atan` over a viewing distance of 1.2 window heights, clamped to yaw ±35° and pitch ±20°, and eases toward it (eyes τ 0.08 s, head 0.3 s). `VrmActor.setLook` gives three-vrm's look-at a world target the whole gaze away, which turns the eyes by what the head hasn't, and turns the head 40% of the yaw and 30% of the pitch on top of the breathing nod. The head's share is what shows on the placeholder, which has no eyes or look expressions; a VRoid export's eyes do the rest. Tested on the placeholder in a 1.0 rig and a 0.x rig.
- `boundingRect`, `setPhase`, `setMouth`, `setMood` and `setTint` arrive with the PRs that use them (5, 3, 10, 11, 3). `setLookAt(screenX, screenY)` became `setLook(gaze)` (the page works out the gaze, in degrees) and `headScreenPosition(camera)` became `headScreen(camera, width, height)`; PR 3's bubble uses the latter.
- **A Spike 0 fix it needed:** the shell knew he was parked by his window's left edge being at or beyond −5000 px, which a monitor far to the left of the primary also has, so a Fritz dragged there was "parked" to `standing()` (the cursor measured from his old spot, and the old spot saved). The shell now records its own park move (`window_parked`, set before the move off screen, cleared after the move back).

*As built (PR 2, third commit: the control window).*
- **Declared in tauri.conf.json** (`control`, 900×640, min 760×520, undecorated, hidden, centred, the shared browser arguments, and `"focus": false`: hidden is not enough, since a webview made with focus takes the keyboard from the owner's app at every start, unseen; a test pins it), shown by the tray's *Settings…* (show, unminimise, then `SetForegroundWindow`, which the tray's click allows: never tao's `set_focus`, which taps Alt into the owner's app when Windows refuses), and only hidden on close. It is an ordinary window, in the taskbar and Alt-Tab; Arrange started from it, or Arrange or Lock from the tray while it was the app in use, hands the keyboard back to it, not to the app beneath (of Fritz's windows, the z-order walk counts it alone as an app). Not shown on first run yet: that needs the token, PR 3. `FRITZ_SHOW_SETTINGS` shows it at start, for a test run.
- **Solid** (§ 4.0, Q12), pinned exactly to releases weeks old (`solid-js` 1.9.16, `vite-plugin-solid` 2.11.14, which allows Vite 8); Tailwind and DivoraVoice's Solid components (`Toggle`, `Segmented`, `HotkeyCapture`) are left for PR 3, with the theme and fonts. The styling is DivoraVoice's tokens (`src/divora/tokens.css`, lines 17–265) and its eyebrow, button and toggle classes, with a little layout of its own; `theme.ts` sets the mood, accent and light-theme attributes from `settings.theme` (PR 3's `theme.ts` adds motion and the avatar's tint).
- **Sections in PR 2:** *Fritz* (Show/Hide, Arrange…, "his eyes follow the cursor"), *Model* (which model he wears: the export, one too large, nothing at `avatar.vrmPath`, or the placeholder; its folder; *Load it again*, which disposes of the old model once the new one is ready), *Diagnostics* (the logs folder; the foreground log, from the next start), *About*. Brain and Appearance are PR 3's, Hotkeys PR 4's, Voice PR 8's; the VRM file picker (`tauri-plugin-dialog`) waits for a section that needs a dependency for it, the folder button covering the export for now.
- **Commands:** `set_parked(on)` (the tray's Hide and Show), `avatar_model()`, `open_folder(which)` (`"logs"` or `"avatar"` only), `reload_model()` (sends `avatar:reload` to the avatar's page), and `avatar_loaded(wearing, error)`, by which the page says after every load whether it wears the export or the placeholder, and why not the export when one is there: the file on disk can't say that a VRoid export failed to parse. The shell keeps it for `avatar_model` and tells the windows (`avatar:loaded`), with which file `avatar_vrm` handed the page for that load (where, its size and when it was written), so a report is never pinned on another file: an export dropped in since, taken away, or a changed `avatar.vrmPath` is `changed` ("A new model is here", until it is loaded), and an old error is never shown against a new file. What the window says is `view.ts`, pure and tested; the components are thin over it.
- **The foreground logger** treats the control window having the keyboard, while it is visible, as deliberate: a `SETTINGS` line (counted in the heartbeat as such), never a `HIT`, and Arrange handing the keyboard back to it isn't Fritz keeping it. Hidden, it holding the keyboard is a `HIT`, like any theft.

### 4.3 Frontend: the avatar page — PR 2 (render) and PR 3 (chrome)

**`PacedLoop.ts`**

```ts
export class PacedLoop {
  constructor(private tick: (dt: number) => void, fps = 30) {}
  start(); stop(); setFps(fps: number); get running(): boolean;
  // setTimeout toward an absolute next-frame time; resync if > 2 periods behind (Lorraine wallpaper.ts:1189-1292). Never rAF.
}
```

**`VrmActor.ts`**

```ts
export type Phase = 'idle'|'listening'|'transcribing'|'thinking'|'speaking';
export class VrmActor {
  static async load(url: string, renderer: THREE.WebGLRenderer): Promise<VrmActor>;   // GLTFLoader + VRMLoaderPlugin; VRMUtils.rotateVRM0 if 0.x
  update(dt: number): void;                   // breathing (chest/upperChest sin), blink scheduler, lookAt easing, expression easing, vrm.update(dt)
  setPhase(p: Phase): void;                   // thinking: head tilt + lookAt away; listening: lean in; speaking: handled by setMouth
  setLookAt(screenX: number, screenY: number): void;   // window-local px → world point on a plane 1 m in front; yaw ±35°, pitch ±20°
  setMouth(v: number): void;                  // expressionManager.setValue('aa', v) with attack 80 ms / decay 150 ms
  setMood(name: 'happy'|'angry'|'sad'|'relaxed'|'surprised'|'neutral', weight: number): void;
  setTint(outline: string, rim: string): void;// MToon outlineColorFactor / rim tint per theme
  headScreenPosition(camera): {x,y};          // for the bubble anchor
  boundingRect(camera): {x,y,w,h};            // projected mesh AABB for hit rects
}
```

Renderer: `new THREE.WebGLRenderer({ alpha: true, antialias: true, premultipliedAlpha: true })`, `setClearColor(0x000000, 0)`, `outputColorSpace = SRGBColorSpace`; camera framing from the VRM's `head` and `hips` bone heights so any VRoid export fits. Lighting: one hemisphere + one directional; MToon doesn't need more.

**`state.ts`** — the plan 14 state machine as a Solid store: `{ phase, turnId, transcript, reply, progress[], mood, connected }`; transitions driven by `CompanionClient` events; `cancel()` returns to idle and clears the audio queue.

**`hitRect.ts`** — `publishHitRect(actor, camera)`: project `boundingRect` each frame, call `invoke('set_avatar_rect', …)` only when any edge moved ≥ 2 px (idempotent — safe under Tauri's invoke reordering).

**`Bubble.tsx`** (PR 3) — Solid component anchored at `headScreenPosition` + offset; a `--surface-2` glass panel with `--line` border and `--shadow-2`; body text `--text-hi` in Space Grotesk 14 px; `progress` as `.eyebrow` lines; streams `reply` with `restart` clearing. Materialise: scale 0.6→1, blur 8→0 over `400 * motion` ms, **driven by the paced loop** (write `style.transform/filter` from `tick`), never CSS transitions. Fades 6 s after `done` unless hovered (hover known from `cursor.over_avatar`).

**`companion.ts`** (PR 3)

```ts
export class CompanionClient {
  constructor(url: string, token: string, client: ClientInfo);
  connect(): void;                 // backoff 1,2,4,…,30 s; sends hello; resolves welcome
  say(text: string): string;       // returns turn_id (crypto.randomUUID())
  sendAudio(pcm16: ArrayBuffer, meta: Omit<AudioEnd,'type'|'turn_id'>): string;
  cancel(turnId: string): void;
  subscribe(widgets: string[]): void;
  on<K extends ServerFrame['type']>(type: K, cb: (f: Extract<ServerFrame,{type:K}>) => void): () => void;
  readonly status: Accessor<'offline'|'connecting'|'online'>;
}
```

Heartbeat: answer `ping` with `pong`; send `ping` every 20 s; two misses → reconnect.

**`theme.ts`** — `applyTheme(doc, settings.theme)` sets `data-theme`, `data-mood`, `data-accent`, `style.setProperty('--motion', …)`, `data-motion-user`; listens to `settings:changed`. The avatar page also calls `actor.setTint(...)` from a small per-mood/accent table.

**Build gates (`scripts/`)**: `check-no-input.mjs` (Lorraine's, pointed at `src/avatar/**` and `src/wallpaper/**` — the avatar page is click-through, so pointer handlers there are dead code); `check-no-css-animation.mjs` — fails on `@keyframes`, `animation:`, `transition:` in any CSS or `style=` reached by `index.html`/`wallpaper.html` (allow-list `prompt.html`/`control.html`); `sync-divora-tokens.mjs` — downloads `src/styles.css` from `NickSanft/DivoraVoiceMod@<pinned sha>` and diffs against `src/divora/tokens.css`, `--write` to update.

**Tests (vitest)**: `PacedLoop` with fake timers (30 fps → 30 ticks/s ± 1, resync after a 200 ms stall, `stop()` stops); `state.ts` transitions incl. cancel; `LipSync` smoothing (PR 10); `hitRect` 2-px hysteresis; `companion.ts` against a `ws` mock (hello/welcome, token order, busy, reconnect backoff, heartbeat); `theme.ts` attribute writes; `protocol.ts` type guard against fixtures exported by `gen_protocol_ts.py --fixtures`.

*As built (PR 3, first commit: the theme on every window).*
- **Fonts:** self-hosted woff2 in `src/divora/fonts/`, the latin subsets of the Fontsource packages DivoraVoice pins (`@fontsource-variable/bricolage-grotesque` 5.2.10, `@fontsource-variable/space-grotesk` 5.2.10, `@fontsource/space-mono` 5.2.9, each tarball's sha512 checked against DivoraVoice's lockfile). Our own `@font-face` rules (`src/divora/fonts.css`) use the family names tokens.css expects. OFL texts in `desktop/LICENSES/`. Both pages load them, 97 KB in all.
- **`src/divora/theme.ts`** (replacing the control window's): `data-theme`, `data-mood`, `data-accent`, `data-motion` and `data-motion-user` on every page. Motion is a level, never an inline `--motion`: tokens.css maps `data-motion` to `--motion`, and an inline value or the attribute both outrank its reduced-motion rule. So reduced motion is worked out in JS, once, for CSS and the paced loop alike: Windows asking for less (WebView2 follows its animation switch), without `theme.motionUser`, is functional. `theme.motion` snaps to a level as DivoraVoice does (0 functional, below 0.8 ambient, else rich), with factors 0, 0.6 and 1. The event is the existing `settings:changed`, not a new `settings:theme`. `watchTheme` (listen, then ask, `afterAsking`, re-apply on the reduced-motion switch) is what PR 4's prompt and Phase 3's wallpaper call too.
- **The avatar:** `setTint(outline, rim, strength)` (Q10): his MToon outlines lean 35% toward the mood's `--text-lo` and his rim light gains the accent's `--indigo` (0.25 on dark, 0.15 on light), always from the model's own colours. Where the model's own rim is off (black, as a VRoid 0.x export leaves it, with a falloff of 1 that would spread the accent over half of him), the tint gives it a rim's falloff (fresnel 5, no lift); a rim the model lights itself keeps its shape. Colours an expression owns (three-vrm's colour binds, put back to the file's every frame) get the tint again after each update. `tint.ts` holds the colours, and a test checks each against tokens.css. `setMotion(factor)`: the sway scales with it, and at 0 he looks ahead whatever the cursor does, leaving only the blink and the breathing (the DoD's "--motion: 0"). That stills his eyes under Windows' reduced motion unless the owner keeps his motion. Grain is never on the avatar.
- **Gates:** `scripts/reach.mjs` follows `avatar.html`'s imports (into `src/divora`, `src/shared`) with the TypeScript compiler's own reading of imports and comments, and both checks read exactly that. `check-paced.mjs` reads our own modules (packages are listed, not read) and also forbids motion set from code (`style.transition =`, `transitionDuration:`, a transition or animation property named in a string, quoted or vendor-prefixed, as a Solid style key would be, and smooth scrolling). `check-no-css-animation.mjs` checks the reached stylesheets and avatar.html's own, then (`--dist`, after `vite build`) the built page's stylesheets, packages included; a built page it refuses is removed, so a cargo build after it stops instead of embedding it. `pnpm build` runs both. `check-no-input.mjs` waits for the wallpaper (PR 12): pointed at `src/avatar/**` it would fail on Arrange's handlers. `scripts/sync-divora-tokens.mjs` (`pnpm sync:divora`, `--write`) compares tokens.css with DivoraVoice's token block at the commit tokens.css's header names (the one place the pin is kept; `--sha <sha> --write` moves it and the line range), through `gh`, by markers, not line numbers. It needs the network, so it is not in the build.
- **Settings:** no new fields. The wire fixture pins `themeDefaults` (Q9), `motionLevels` and `motionSnap` for both sides. Tailwind is not added (Q12, the owner's choice).

*As built (PR 3, second commit: Appearance, in Settings and the tray).*
- **Lifted:** DivoraVoice's `Toggle`, `Segmented` and `blurOnPointer` (`src/divora/components/`) and `Sigil` (`src/divora/sigils/`, an inline Solid component rather than `sigils/{index.ts,*.svg}`, since `currentColor` needs inline SVG), with their own tests, its test setup (jsdom, `@solidjs/testing-library`, jest-dom; vitest's "dom" project beside the "node" one) and the CSS for the segmented control, the grain and the functional-motion switch. `Sigil` loses `github` (no borrowed artwork) and gains the body's eight glyphs in the same hand: `butler`, `desk`, `calendar`, `inbox`, `weather`, `listening`, `thinking`, `speaking`.
- **Settings › Appearance:** Theme, Colour mood, Accent and Motion as radio groups, "Keep his motion when Windows asks for less" (`theme.motionUser`, the plan's "unless the user lifts it in settings"; the Motion row says when Windows has him keeping still) and Parchment grain, on this window only. Sections now carry a sigil (Fritz's is `butler`).
- **The tray:** an Appearance submenu of check items (the moods, the accents, Light theme, the three motion levels), built once and held by every rebuilt menu. A pick and a change in Settings both go through `change_setting`, which tells every window and, for the theme, moves the submenu's ticks in place, so they follow Settings. Nothing is rebuilt for it: muda never reuses a menu id and matches clicks on 16 bits, so each object built shortens the tray's life (the Show/Hide rebuild PR 2 does on park and Arrange still spends about eight a time). The tray ticks the level `theme.motion` stands for, as the Motion row does. `settings::motion_level` snaps as theme.ts does, both held to the fixture's rows; `settings::{THEMES, MOODS, ACCENTS}` are what the settings accept and what the tray offers.
- **`control_opening`:** `FRITZ_SHOW_SETTINGS=<section>` opens the control window at that section (any other value only shows it, as before): the first-run Brain opening (commit 3) and the live checks use it.

### 4.4 The prompt window — PR 4

`Prompt.tsx`: a single-line input (Space Grotesk 16 px) on a `--surface-1` panel with the brand gradient as a 2 px bottom rule while connected; placeholder "Ask Fritz…"; Enter → `say`, hide; Shift+Enter → newline; ↑ recalls last prompt; Esc hides; a `.eyebrow` shows `LOCAL · NO ACCOUNT` and the connection dot (`--success` online, `--warning` connecting, `--danger` offline).

Rust `prompt::open()`: `snapshot_foreground()` (Lorraine `workerw.rs:480`; `restore_foreground()` at `:515`) → position centred horizontally 160 px above the avatar's top edge (clamped to the work area) → `show()` → `set_focus()` (this window is allowed to; it is not the overlay) → emit `prompt:open`. `prompt::close()`: `hide()` → `restore_foreground()`. The page sends `invoke('close_prompt')` on Enter/Esc. Hotkey `prompt` toggles.

**Test (vitest):** input semantics (Enter/Shift+Enter/↑/Esc) via DOM events. **Manual:** opening from a fullscreen game does nothing while `monitor:covered` is true.

### 4.5 Avatar interaction — PR 5

Arrange: `set_arrange(true)` → `Input::Mode(ModeRequest::Arrange)` (lifted) → the overlay lifts click-through, takes the foreground with `try_foreground`, the page shows a dashed `--line-glow` frame and a resize grip; drag moves the window (Tauri `startDragging` works here because the window is now focusable? — **no**: keep `focusable(false)` and move with `SetWindowPos` from mouse deltas the page reports via `invoke('arrange_drag', dx, dy)`); Esc, the tray or a 30 s lease without a beat → `Locked`, `return_focus`, save the rect. Locked: hook right-click → tray popup; double-click → prompt. **Tests:** machine transitions (cargo); drag maths (vitest).

### 4.6 Voice — PR 8 (capture) and PR 10 (playback + lip-sync) · Phase 2 · depend on PR 9

**`audio_in.rs`**: `cpal` default input device (DivoraVoice's path) → resample to 16 kHz mono (`rubato`, as DivoraVoice) → PCM16 ring buffer; `start()` on `hotkey ptt pressed`, `stop() -> Vec<u8>` on `released` or at 30 s; the bytes go to the page via a `ptt:clip` event (base64; a 30 s clip is ~960 KB — acceptable; or write to a temp file and pass the path), and the page sends one binary WS frame + `audio_end`. Emit `ptt:level` RMS at 20 Hz while recording for the bubble's meter (DivoraVoice's emerald→gold→crimson meter component). Tray `Mute` unregisters the PTT hotkey and shows the `muted` sigil on the avatar's bubble.

**`LipSync.ts`**:

```ts
export class LipSync {
  constructor(ctx: AudioContext) { analyser = ctx.createAnalyser(); analyser.fftSize = 1024; }
  attach(node: AudioNode): void;          // playback graph: source → analyser → destination
  sample(dt: number): number;             // RMS of time-domain data → target = clamp((rms - 0.02) * 6, 0, 1); attack 80 ms, decay 150 ms
}
```

Playback queue: `speech` frames decoded with `ctx.decodeAudioData` and scheduled back-to-back on `ctx.currentTime`; `speak` frames (local TTS) go to `invoke('tts_synthesize', {text, voice})` which returns WAV bytes from `kokoro_tts.rs` (lifted `divora-core/src/tts/{kokoro,phonemize,speaker,tokens}.rs`, models in `%APPDATA%\com.divora.fritz\tts\`, fetched by a `scripts/fetch-voice-assets.ps1` adapted from DivoraVoice's). `state → speaking` while the queue is non-empty; `cancel` (PTT pressed while speaking) stops all sources, clears the queue, sends `cancel`, then starts the new recording — barge-in.

**Acceptance (Phase 2):** hold PTT, "what time is it?", release → `stt` text in the bubble ≤ 1 s after release on the owner's machine; first audio ≤ 3 s after release with `COMPANION_TTS=local`; mouth moves with the audio; PTT mid-sentence stops playback and `ollama ps` shows the model idle within a second.

### 4.7 Ambient layer — PR 12 (wallpaper) and PR 13 (widgets) · Phase 3 · depend on PR 14

Lift `workerw.rs`, `host.rs`, the slot/`ensure_attached`/watchdog code from Lorraine's `lib.rs:387-559` into `wallpaper.rs`, `power.rs` wiring, and declare `wallpaper-0..3` windows (transparent, `visible:false`) in `tauri.conf.json` with the shared browser args. `wallpaper/main.tsx`: the study scene (`StudyCircle.ts` draws the ring, ticks and threads on a canvas inside the paced loop; widgets are Solid components absolutely positioned from `widget_layout`), `power:display` pauses the loop, `check-no-input.mjs` guards the page. Widgets consume `Widget` frames from a second `CompanionClient` instance (caps `["widgets"]`) or the avatar window re-broadcasts them via Tauri events — choose the re-broadcast (one brain connection per device) and document it. `control/sections/Layout.tsx` is Lorraine's `control.ts:400-490` editor ported to Solid.

### 4.8 Control window — spread across PRs 2–4, 8

Sections (DivoraVoice's Settings page shape: `sigil + title + desc + panel of rows`, max-width 680): **Brain** (URL, token field with "Test connection", status line, `LOCAL · NO ACCOUNT` eyebrow, on-device shield); **Avatar** (VRM file picker via `tauri-plugin-dialog`, scale slider, "Look at cursor" toggle, "Arrange…" button); **Hotkeys** (four `HotkeyCapture` rows, lifted); **Appearance** (Theme segmented Dark/Light · Mood segmented · Accent segmented · Motion segmented Functional/Ambient/Rich · Parchment grain toggle); **Voice** (Ph2: input/output device `Select`s, PTT meter, TTS voice `Select` listing Kokoro presets with `bm_george` default, "Audition"); **Layout** (Ph3); **About** (version, MIT, GitHub, "Replay first run").

First-run wizard (DivoraVoice's shell, 3 steps): *Brain* (paste token, test) → *Avatar* (pick VRM or keep the placeholder) → *Hotkeys* (confirm/change) → "Summon Fritz".

---

## 5. CI

`.github/workflows/ci.yml` gains a job:

```yaml
  desktop:
    runs-on: windows-latest
    defaults: { run: { working-directory: desktop } }
    steps:
      # Every `uses:` MUST be pinned to a full commit SHA with an exact-version comment —
      # tests/test_workflows.py fails the suite on tag pins or `# v4`-style comments.
      # Copy the checkout pin from the existing jobs; resolve the others with
      # `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` at PR time.
      - uses: actions/checkout@<sha>            # v7.0.1 (same pin as the core job)
      - uses: pnpm/action-setup@<sha>           # v4.x.y
        with: { version: 10 }
      - uses: actions/setup-node@<sha>          # v4.x.y
        with: { node-version: 20, cache: pnpm, cache-dependency-path: desktop/pnpm-lock.yaml }
      - uses: dtolnay/rust-toolchain@<sha>      # stable, pinned by commit
        with: { toolchain: "1.89" }
      - uses: Swatinem/rust-cache@<sha>         # v2.x.y
        with: { workspaces: desktop/src-tauri }
      - run: pnpm install --frozen-lockfile
      - run: pnpm test
      - run: pnpm build            # runs the three check scripts
      - run: cargo check --manifest-path src-tauri/Cargo.toml
      - run: cargo test --features custom-protocol --lib
        working-directory: desktop/src-tauri
      - run: python ../scripts/gen_protocol_ts.py --check
```

The Python jobs add `python scripts/gen_protocol_ts.py --check` after `ruff`. No overlay/WorkerW behaviour is tested in CI; the Spike 0 table is the manual checklist and lives in `desktop/docs/SPIKE0.md`.

---

## 6. Phase gates (manual acceptance scripts)

**Phase 1.** (1) Brain: `python main_local.py` with `COMPANION_ENABLED=true`. (2) Body: first run → wizard → token → "Summon Fritz" → avatar appears bottom-right, breathing, eyes following the cursor. (3) Alt-Tab: Fritz is not listed. (4) Click through Fritz onto a window behind him: the click lands behind. (5) `Ctrl+Alt+F` from Notepad, type "good evening", Enter: prompt vanishes, bubble streams, Notepad keeps focus (caret still blinking). (6) `Ctrl+Alt+A`, drag Fritz to another monitor, Esc: position persists across restart. (7) Borderless-fullscreen game: Fritz parks; Alt-Tab out: Fritz returns. (8) Sleep/resume and explorer restart (`taskkill /f /im explorer.exe && start explorer`): Fritz is still there. (9) Tray → Appearance → Midnight: bubble, prompt and control recolour live. (10) Spike 0's foreground logger run for the whole session shows zero hits.

**Phase 2.** § 4.6 acceptance, plus Mute hides the PTT path and the bubble shows the muted sigil.

**Phase 3.** Study scene on every monitor behind icons; widgets drag/resize from Layout and survive explorer restart, sleep, DPI change; `/schedule` from Discord lands in the desk widget and bubble.

**Phase 4.** "What's on my calendar tomorrow?" answers from Google at the desk; agenda widget refreshes without a turn; a non-owner Discord user asking the same gets a refusal and the audit log shows no Google tool call.

---

## 7. Decision defaults applied by this plan (flip here, not in code)

| Q (plan 14) | Default this plan builds | How to flip |
|---|---|---|
| Q1 GPU/VRAM | Assume 16 GB; keep `WHISPER_DEVICE=cpu`; TTS off the GPU | Owner states the card; if ≥ 24 GB, `COMPANION_TTS=xtts` becomes viable |
| Q2 TTS | **(c) local Kokoro in the body**, voice `bm_george`; `COMPANION_TTS=local` | `COMPANION_TTS=xtts` for brain-side XTTS; the protocol carries both |
| Q3 mood source | None in Phase 2; expressions follow `state` only | PR 11 adds inline `[mood:…]` tags |
| Q4 mic | Rust `cpal` capture | WebView `getUserMedia` + Tauri 2.12 `on_permission_request` |
| Q5 cancel | Partial reply checkpointed with ` [interrupted]` | Drop the exchange in `executor` when interrupted |
| Q6 hotkeys | `Ctrl+Alt+F` prompt · `Ctrl+Alt+Space` PTT · `Ctrl+Alt+A` arrange · `Ctrl+Alt+H` park | Settings → Hotkeys |
| Q7 thread | Own thread `…#desktop` | Pass `thread_id=identity` in `companion_api` |
| Q8 scheduler in `main_local` | No (Phase 1) | After PR 14 |
| Q9 default look | `ink` · brand · dark · motion 0.6 · grain on (control only) | Settings → Appearance |
| Q10 VRM per theme | Tint only | Add `vrm_path_by_mood` to settings |
| Q11 UI sounds | None | A `sounds/` set behind a toggle |
| Q12 framework | SolidJS + Tailwind (DivoraVoice's). *As applied:* Solid, without Tailwind: DivoraVoice uses no Tailwind utility class anywhere (its look is its own CSS, lifted), so the owner chose to leave it out (PR 3) | Vanilla TS port of the CSS only; or add Tailwind 3.4.19 with postcss and autoprefixer and `@tailwind` at the top of `control.css` |
| Q13 glyphs | None | Lift DivoraVoice's `src/data/glyphs.ts` (`detectShape`) + `src/components/SparkLayer.tsx` later |

---

## 8. Appendix — command cheat sheet

```powershell
# brain
python -m pip install -c requirements.txt -e ".[dev,voice]"
python main_local.py                                   # COMPANION_ENABLED=true in .env
pytest tests/ -v ; ruff check . ; python scripts/gen_protocol_ts.py --check
# body
cd desktop ; pnpm install ; pnpm dev                   # Vite on :1431
pnpm tauri dev                                         # dev build (loads localhost:1431)
pnpm build ; cargo check --manifest-path src-tauri/Cargo.toml ; cd src-tauri ; cargo test --features custom-protocol --lib
pnpm tauri build                                       # NSIS per-user installer
node scripts/sync-divora-tokens.mjs --write            # refresh tokens.css from the pinned DivoraVoice sha
# protocol by hand
websocat ws://127.0.0.1:8001/companion
```

Lifted-file inventory with sources: `desktop/docs/LIFTED.md` is created by PR 2 and lists every file copied from LorraineAtelier or DivoraVoiceMod with the source path and commit, so a later sync is a diff rather than an archaeology.
