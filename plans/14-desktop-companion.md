# 14. Desktop companion — Fritz gets a body

[← back to index](README.md) · [decisions](DECISIONS.md) · [build plan for Claude Code](14-desktop-companion-build.md)

_Planned November 2026. Interview-driven: eight product decisions were taken with the owner first (recorded below and proposed as decisions 28–33, plus 34 for adopting the DivoraVoice design system, added at the owner's request), then readers went through this repo, `NickSanft/LorraineAtelier` and `NickSanft/DivoraVoiceMod` line by line, and the library landscape (three-vrm, Tauri 2, langchain-mcp-adapters, workspace-mcp, Ollama) was checked against current releases. Every file:line reference was verified against the working tree at `aecfe90` (2026-10-16), Lorraine at `d62b7ec` (2026-07-25) and DivoraVoice at `c9f066b` (2026-11-08)._

> **Read this first.** Phase 1 is gated on a one-day spike (§ Spike 0) because Lorraine's own `CLAUDE.md` says *never keep a top-level WebView2 window always shown* — the author tried a persistent always-on-top overlay and WebView2 kept stealing the foreground. That is exactly the avatar window this plan proposes — and the owner has since shipped one that works: DivoraVoice v1.55's stream overlay (`overlay_machine.rs` + `overlay_windows.rs`) is a persistent, transparent, always-on-top, click-through WebView2 window with a tested rule that it never calls `set_focus()`. Spike 0 now lifts that code and proves it carries a WebGL avatar; if it fails, the fallback is already specified and nothing else in the plan changes.

---

## Goal

Turn Mister Fritz from a chat bot into a desk-side assistant: a 3D character that lives on the Windows desktop, is talked to by hotkey or push-to-talk, answers in Fritz's voice with lip-sync, and (later) sits on a live wallpaper of movable widgets that show the owner's calendar, inbox and the weather without being asked. The brain stays where it is — this repo, LangGraph on local Ollama — and gains one new client surface. Nothing here changes what Discord or Telegram users see.

Six end goals from the owner, mapped to phases:

| # | End goal | Where it lands |
|---|---|---|
| 1 | A *true* virtual assistant built on Fritz | Phases 1–2 (reactive) + 3–4 (ambient) |
| 2 | A persistent 2D/3D model on the Windows desktop | Phase 1 (VRM, overlay) |
| 3 | Lives on a live wallpaper with movable widgets, as in Lorraine | Phase 3 |
| 4 | Eventually persists across devices | Phase 5 (one device through Phase 4) |
| 5 | Integrates with Gmail, Drive and Calendar | Phase 4 |
| 6 | One solid, iterable tech stack | § Architecture — the stack is decided, not grown |

## What was decided in the interview

These are recorded so the plan can be read without the chat. Each is proposed as a numbered entry for `DECISIONS.md` (§ Decisions to record).

| Decision | Choice | Rejected |
|---|---|---|
| Stack split | **Python brain (this repo) + Rust/Tauri 2/TypeScript body.** Two languages, one wire contract. | Porting the brain to TS (loses LangGraph/Chroma/faster-whisper and 288 commits); porting the shell to Python (throws away Lorraine's WorkerW work) |
| Avatar | **3D VRM** (VRoid Studio model) in a transparent, always-on-top, click-through overlay | Live2D (proprietary Core, Cubism Editor needed); pixel sprite; wallpaper-layer-only |
| Brain hosting | **Same Windows PC, fully local Ollama** | Home server; cloud fallback; cloud-primary |
| Google | **Read-only first, via `workspace-mcp`** | Write scopes day one; hand-written `googleapiclient` tools; defer Google |
| Assistant style | **Reactive + ambient widgets.** No unprompted speech. | Proactive nudges; acting on the owner's behalf (both later, both gated) |
| Input | **Global hotkey + text, and push-to-talk voice** | Wake word; clicking the avatar as the primary input |
| Repo | **Inside this repo, under `desktop/`** | New repo; monorepo split; forking Lorraine in place |
| Phase 1 | **Avatar + text chat only** | Wallpaper, widgets, Google or voice in the first milestone |

---

## Current state (verified against the working tree)

### The brain has no API a desktop client can use

- The only network surface is the admin panel: **Starlette** on uvicorn, bound `127.0.0.1:ADMIN_PANEL_PORT` (default 8001, `fritz_utils.py:278`), started **only** from `main_discord.py:305` inside `on_ready`, and refusing to start without `ADMIN_PANEL_PASSWORD` (`admin_panel.py:1349`). `main_telegram.py` never starts it. So today the web surface exists only while the Discord bot is logged in.
- Chat is `POST /chat/stream` (`admin_panel.py:844-954`): hand-formatted **SSE** (`reset`, `token` deltas, `progress`, one `done` with `{text, html, images}`, or `error`), polled off a `queue.Queue` every 50 ms. No WebSocket anywhere; `pyproject.toml` even pins uvicorn with the comment "plain Config: no reload, no websockets".
- Auth is a 30-day HMAC cookie (`chat_auth.py`, `fritz_chat_id`), `SameSite=Lax`, no CORS, and `_SameOriginMiddleware` (`admin_panel.py:87-144`) rejects any POST whose `Sec-Fetch-Site` is `cross-site`/`same-site`. A Tauri webview `fetch` from `http://tauri.localhost` **will be refused**. A Rust-side `reqwest` call would pass (no `Sec-Fetch-*` headers), but that still buys no cancel and no audio.
- **No cancel.** A disconnected client leaves the turn running server-side; the "Enough" button admits this honestly (its label was settled by the mockup; see the "Still open" list at the end of `DECISIONS.md`).
- `ask_stuff` (`mister_fritz.py:914-929`) is synchronous and blocking, returns `{"text","image_paths","timestamp"}`, and takes `streaming_callback(delta, accumulated, restart)` (contract on `_DeltaEmitter`, `:584-646`) and `progress_callback(text)`. One turn per thread with one waiter via `one_turn_at_a_time` (`:369`); a third concurrent turn gets the "One moment, sir…" text reply (`:1026-1038`).
- Identity: `canonical_user_id(platform, raw)` → `<platform>-<id>` (`fritz_utils.py:488`), `KNOWN_PLATFORMS = {"discord","telegram","web","local"}` (`:479`), `resolve_identity` follows one `IDENTITY_LINKS` hop (`:549-565`). `MessageSource` (`fritz_utils.py:866`) has no desktop member; `LOCAL` is described to the model as "via CLI".
- Tools: `get_conversation_tools_description(include_file_tools=False, extra_tools=None)` (`agent_tools.py:571-596`). Reminder tools are only attached when `channel_id` **and** `schedule_manager` are passed (`mister_fritz.py:683-695`), so the web chat has none. **No MCP support exists.** The `skills/` loader (`agent_tools.py:456-488`) has no contract (plan 13, open question 3).
- STT: `stt.transcribe(path) -> str|None`, **faster-whisper**, sync, file-path in, segment timestamps discarded (`stt.py:98`). Default `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, model `small`.
- TTS: `TTSEngine.generate_speech(message, speaker="Baldur Sanjin", ...) -> wav path`, **Coqui XTTS v2**, sync under an instance lock, cached by md5 in `output/`, `split_sentences=True` internally. No timing/viseme output. Constructed only in `main_discord.on_ready` (`:248-254`), used only by `/voice`. `.wav` is not in `_CHAT_ASSET_CONTENT_TYPES`, so the web surface cannot serve audio.
- Scheduler: `ScheduleManager(bot: discord.Client)` imports `discord` at module scope; `_run_task` (`scheduler.py:90-137`) posts straight to `bot.get_channel(channel_id)`. **No delivery abstraction**; `channel_id INTEGER NOT NULL`. Plan 13's timezone question (Q1) is still open and will matter for an agenda widget.
- Extras: `voice` = `coqui-tts>=0.27` + `faster-whisper>=1.0`; `image` = diffusers/torch stack. Core stays torch-free. CI = compileall, ruff, `scripts/check_imports.py`, pytest.

### Lorraine has the Windows plumbing, and one hard-won warning

What is directly reusable (all paths under `LorraineAtelier/`):

- `src-tauri/src/workerw.rs` — WorkerW discovery (Progman `0x052C`, Win11 child path, Win10 enum fallback, never trusts the return value), `enum_monitors()` in physical pixels under PerMonitorV2, `monitor_covered()` fullscreen detection, `desktop_exposed_at()`, `snapshot_foreground()`/`restore_foreground()`.
- `host.rs` + `lib.rs:284-337` — per-monitor native host windows under WorkerW and `reparent_controller` (moves the WebView2 **controller**, not the Tauri HWND). `ensure_attached` (`lib.rs:387-497`) rebuilds on explorer restart, display/DPI change, monitor unplug; a 600 ms watchdog drives it.
- `hook.rs` — `WH_MOUSE_LL` on its own thread, cheap rect gate, work shipped over mpsc; this is how the wallpaper gets clicks the OS never sends it. `scripts/check-no-input.mjs` fails the build if `wallpaper.ts` registers pointer handlers (dead code there).
- `power.rs` — resume, display on/off, session lock/unlock via a hidden top-level `WS_EX_TOOLWINDOW` window (message-only windows don't get `WM_POWERBROADCAST`).
- `audio.rs` — `wasapi` crate (cpal loopback was unreliable), realfft, `viz:bands` at ~45 Hz. The `Direction::Capture`-on-render-endpoint trick is loopback; a microphone is the same crate on the default capture endpoint.
- `settings.rs` — `%APPDATA%\<id>\settings.json`, atomic temp-and-rename, corrupt file quarantined. `widget_layout: HashMap<String, WidgetPos{x,y,scale,vertical}>` in % of work area.
- The buddy engine (`lib.rs:700-1092`): a 30 Hz Rust-owned wander state machine across monitors, emitted per-window — a generic companion-roaming engine.
- `control.ts:400-490` — the widget layout editor (draggable chips on an aspect-correct stage, 60 ms debounce, `set_widget_layout`).
- `wallpaper.ts:1189-1292` — the **setTimeout-paced** render loop. PresentMon showed a rAF loop or any CSS animation makes Chromium present every vsync regardless of draw rate; that, not rasterization, was the idle GPU cost.
- Gotchas that will bite again: every window's `additionalBrowserArgs` must be byte-identical or WebView2 forks environments; `hwnd()` fails for runtime-created windows (declare a pool in config); `cargo test` without `--features custom-protocol` overwrites the exe with a dev build that loads `localhost:1430`; the unsigned exe trips a Defender ML false positive per SHA-256.

**The warning** (`CLAUDE.md:35`, `docs/DEV_NOTES.md:30,87`, `docs/ARCHITECTURE_EVALUATION.md:14,64`): the planned layered click-through always-on-top overlay was abandoned because WebView2 took and kept the foreground; `WS_EX_NOACTIVATE` applied by hand did not stop it, and a focus-restoring watchdog made it a fight. Lorraine's overlay is therefore modal: shown with `show()` + `set_focus()` (`lib.rs:141-155`), hidden with `hide()` + `restore_foreground()` (`:197-207`). Two things differ for the avatar: Tauri 2.8 (Aug 2025) added a first-class `focusable` window attribute and `set_focusable` (tao 0.34.1 `with_focusable`), and the avatar never needs focus — typing happens in a separate modal prompt window that follows Lorraine's proven show/hide pattern. And the owner's own DivoraVoice already ships the counter-example (next section). Zebar and SnapShelf (Tauri 2, persistent transparent always-on-top WebView2 windows) corroborate it, and the WebView2 focus bug Lorraine likely hit (WebView2Feedback #356) is reported fixed since runtime 1.0.1901.177. Spike 0 is therefore a port-and-verify, not an experiment.

Lorraine also locks versions worth matching: tauri **2.11.3**, tauri-build 2.6.3, tauri-plugin-global-shortcut **2.3.2**, windows **0.61.3**, webview2-com 0.38.2 (must match wry's), wasapi 0.15, Vite 5.4, Vitest 2.1, TypeScript 5.9, pnpm 10.

---

### DivoraVoice has the persistent overlay, the design system and a Rust TTS

`NickSanft/DivoraVoiceMod` (v1.60.0, Tauri 2 + SolidJS 1.9 + Tailwind 3.4 + Vite 5.4; Rust workspace `divora-core` / `divora-capture`; identifier `com.divora.voicemod`), verified at `c9f066b`:

- **The stream overlay** (v1.55.0) is the window Lorraine said couldn't exist. `src-tauri/src/overlay_windows.rs:164-180` builds it `transparent(true)`, `decorations(false)`, `shadow(false)` (an undecorated transparent window keeps a 1 px border otherwise — tauri #8308), `always_on_top(true)`, `skip_taskbar(true)`, `visible(false)`, `focused(false)`; `:199` makes it click-through with `set_ignore_cursor_events(true)` **before** anything can show it; `:203-207` marks the HWND *NonRude* through `divora_capture::mark_non_rude` so Windows' fullscreen-app detection ignores it. It is shown only after the page's ready handshake (10 s `WATCHDOG`, `overlay_machine.rs:28`), and locked again when the page's heartbeat stops (`LEASE_TICK`/`LEASE_MISSES`). All rules live in `overlay_machine.rs`, a pure, window-free state machine (`OverlayCore::step` → `Effect`s) so every transition is unit-tested off-Windows.
- **Focus discipline, enforced by a test.** `overlay_windows.rs:855-865` `the_overlay_never_calls_set_focus` greps its own source: tauri/tao's focus call injects an Alt key press when Windows refuses, so the overlay takes the foreground only with plain `SetForegroundWindow` (`try_foreground`, `:627-636`) and only for **Arrange** mode, remembering the previous foreground and handing it back (`return_focus`, `:641-646`). **Locked** mode (the default) is click-through; Arrange lifts `ignore_cursor` so pictures can be dragged, then re-locks. The module header (`:7-21`) records the threading rules — window calls on the main thread, getters never under the core lock, the page's handshake as commands not event listeners because tauri parks Rust listeners on lock contention.
- **An IPC ordering gotcha** worth inheriting (CHANGELOG 1.60.0, Architecture notes): Tauri 2 invokes are separate fetches and WebView2 does not keep their order — 40 fired at once arrived scrambled. `src/data/inOrder.ts` serialises per-feature command queues. The companion protocol is a single ordered WebSocket so the brain side is safe; the body's own `invoke`s (hit-rects, settings) must be idempotent or queued.
- **The design system** is in `src/styles.css` (tokens, three `data-mood` blocks `:117-143`, two `data-accent` blocks `:146-160`, `data-theme="light"` `:171-246`, `--motion` with the `prefers-reduced-motion` collapse `:249-252`) and `docs/mockups/README.md` (type roles, Sigil set, components, copy rules). See § Theme.
- **On-device TTS in Rust**: `divora-core/src/tts/{kokoro,phonemize,speaker,tokens}.rs` run Kokoro-82M through `ort` (`=2.0.0-rc.12`, `load-dynamic`) with espeak-ng phonemisation; `synthesize(text, voice_id, asset_dir) -> TtsAudio` (`mod.rs:243`); preset voices include `bm_george — crisp (UK)` and `bf_emma — soft (UK)` (`mod.rs:60-61`). Voice cloning (VoxCPM accent-preserving, OpenVoice timbre) sits beside it. Relevant to Q2.
- **Push-to-modulate** is a backend-side hold-key with press/release edges and an explicit fix for release-before-press reordering (1.60.0 Fixed) — the same shape as the desk's push-to-talk.

## Architecture

```
┌──────────────────────────── Windows PC ────────────────────────────┐
│                                                                     │
│  BRAIN (this repo, Python 3.12)             BODY (desktop/, Tauri 2)│
│  ┌──────────────────────────────┐   WS      ┌─────────────────────┐ │
│  │ main_discord.py / main_local │◄─────────►│ avatar window       │ │
│  │   admin_panel (Starlette)    │ /companion│  three.js + VRM     │ │
│  │   + companion_api.py  ◄──────┼───────────┤  click-through,     │ │
│  │ mister_fritz.ask_stuff()     │           │  never focusable    │ │
│  │   LangGraph ⇄ Ollama         │           ├─────────────────────┤ │
│  │   Chroma / SQLite            │           │ prompt window       │ │
│  │ stt.py (faster-whisper)      │           │  modal, hotkey      │ │
│  │ tts.py (XTTS v2) / chunker   │           ├─────────────────────┤ │
│  │ scheduler → DeliverySink     │           │ wallpaper + widgets │ │
│  │ MCP client → workspace-mcp ──┼─► Google  │  (Phase 3, WorkerW) │ │
│  └──────────────────────────────┘           ├─────────────────────┤ │
│         ▲ Discord / Telegram                │ Rust: hook, power,  │ │
│         │ (unchanged)                       │ tray, settings, mic │ │
└─────────┼───────────────────────────────────┴─────────────────────┴─┘
          ▼
   other devices later (Phase 5): same /companion protocol over Tailscale
```

**One rule makes goal 6 true:** the body contains no intelligence. It renders, captures audio, owns Windows plumbing, and speaks a small JSON protocol. Every decision about what Fritz says, remembers, fetches or schedules is in Python. A second device is another client of the same protocol; Discord stays an adapter of the same `ask_stuff`.

### The companion protocol

New endpoint on the existing Starlette app: `ws://127.0.0.1:8001/companion`. JSON text frames; binary frames reserved for audio in Phase 2.

| Dir | `type` | Payload | Notes |
|---|---|---|---|
| → | `hello` | `token, client{id, name, caps[]}` | First frame. `token` = `COMPANION_TOKEN`. `client.id` becomes `desktop-<machine-slug>`. Bad token → close 4401. |
| ← | `welcome` | `identity, display_name, server_caps[]` | After `identity_store.record(...)`. |
| → | `say` | `text, turn_id` | Text turn. |
| → | `audio` | binary frame, 16 kHz mono PCM16 WAV, ≤ 30 s | Push-to-talk clip (Phase 2). Followed by `{type:"audio_end", turn_id}`. |
| → | `cancel` | `turn_id` | Cooperative cancel (see P3). |
| → | `subscribe` | `widgets[]` | Phase 3: `clock, weather, desk, agenda, inbox`. |
| ← | `state` | `turn_id, state` ∈ `listening\|transcribing\|thinking\|speaking\|idle` | Drives the avatar state machine. |
| ← | `stt` | `turn_id, text` | What Whisper heard, echoed to the bubble. |
| ← | `token` | `turn_id, delta` | From `streaming_callback`. |
| ← | `progress` | `turn_id, text` | From `progress_callback` (tool notices). |
| ← | `speech` | `turn_id, seq, mime, b64` | Phase 2: one sentence of TTS audio. |
| ← | `done` | `turn_id, text, images[], interrupted` | Exactly one per turn. |
| ← | `error` | `turn_id, text` | Butler-voiced via `fritz_error`. |
| ← | `widget` | `name, payload, ttl_s` | Phase 3. Computed **without** the LLM. |
| ← | `reminder` | `schedule_id, text` | Phase 3/5 via `DeliverySink`. |
| ↔ | `ping`/`pong` | — | 20 s heartbeat; SSE had none. |

Shared schema: `companion_protocol.py` (pydantic models) is the source of truth; `desktop/src/shared/protocol.ts` is generated from it in CI (`scripts/gen_protocol_ts.py`) so the two sides cannot drift silently.

### Identity

`desktop` joins `KNOWN_PLATFORMS`. The owner links `desktop-<machine>` to their Discord root identity through the existing `IDENTITY_LINKS` so memories and `ROOT_USER` privileges follow them (one hop is already supported, `fritz_utils.py:549-565`). Threads: `THREADS_PER_CHANNEL` defaults to **false** (`fritz_utils.py:537`), and when it is off `thread_id_for` ignores `channel_key` and returns the bare identity (`:575-576`) — so once `IDENTITY_LINKS` maps the desk to the Discord root, the desk would silently share the owner's Discord DM thread. `companion_api` therefore passes an explicit `thread_id=f"{identity}#desktop"` to `ask_stuff` (the keyword-only override that already exists for the web chat), which works regardless of the flag. Q7 asks whether that separation is wanted.

### Avatar state machine (body side)

`idle → listening (PTT held) → transcribing → thinking (tokens arriving) → speaking (audio queue non-empty) → idle`. `cancel` from any state returns to `idle` and flushes the audio queue. Each state maps to a VRM pose/expression set; `thinking` plays a short look-away, `speaking` drives the mouth from an `AnalyserNode` on the playing buffer.

---

## Theme — the body joins the Divora design system

The owner already has a shipped, tested design system in a Tauri 2 app of his own: **DivoraVoice** (`NickSanft/DivoraVoiceMod`, v1.60.0, identifier `com.divora.voicemod`). Its handoff (`docs/mockups/README.md`) describes the identity as *"spellcraft for your voice" — a calm, precise utility (Linear/Raycast bones) wearing a dusk-lit arcane skin*, and its production stylesheet (`src/styles.css`) implements that as a token sheet with **orthogonal axes**, switched by root attributes:

| Axis | Attribute | Values in DivoraVoice | What it changes |
|---|---|---|---|
| Theme | `data-theme` | dark (default), `light` | Surface/text polarity; light is a lilac-tinted ramp where surfaces *darken* with elevation |
| Mood | `data-mood` | Dusk Violet (default), `ink` (ink + candle), `midnight` | The surface/line/text hue family; composes with light (parchment & candle, overcast slate) |
| Accent | `data-accent` | brand (indigo→pink), `abyssal` (cyan), `ember` (amber→red) | `--indigo`/`--pink`/`--grad` — the one true accent, "alive / modulating" |
| Motion | `--motion` | functional 0 · ambient 0.6 · rich 1 | Every duration scales by it; `prefers-reduced-motion` collapses it to 0 unless `data-motion-user` |
| Texture | grain / vignette toggles | parchment grain, vignette | Overlays; grain switches to `multiply` on light |

The Fritz body **adopts this system rather than inventing a parallel one.** The same tokens, the same axes, the same type roles and the same component vocabulary — so the owner's two Tauri apps (and the brushbuddy-era Lorraine plumbing they both descend from) read as one family, and a token fix in DivoraVoice is a copy away from Fritz. Fritz's house look is then a *configuration* of that system, not a fork of it.

### What the body takes verbatim from DivoraVoice

- **Tokens.** The `:root` block of `src/styles.css`: surfaces `--bg #07060d` … `--surface-4 #2c2442`; lines `--line`/`--line-strong`/`--line-glow`; text `--text-hi #ECE7F8`, `--text-mid`, `--text-lo`, `--text-dim`; accent `--indigo #6D5BF0`, `--indigo-deep #4F46E5`, `--pink #EC4899`, `--pink-deep #DB2777`, `--grad: linear-gradient(120deg, #4F46E5 0%, #7C5CF6 42%, #DB2777 100%)`, `--grad-soft`; semantic `--success #34D9A0` (emerald: armed / ok / "on-device"), `--warning #E9B14C` (candlelit gold), `--danger #F2567A` (crimson-rose), `--info #58C6F2`, each with a `-bg` at ~12 % alpha; radii `--r-sm 7 · md 11 · lg 16 · xl 22 · pill`; a 4 px spacing scale `--s1…--s9`; shadows `--shadow-1..3` and `--glow-accent`. Plus the three mood blocks, the two accent blocks, the light theme and its two mood composites — copied as a file, with a `scripts/sync-divora-tokens.mjs` that diffs against a pinned DivoraVoice commit so drift is visible.
- **Type roles**: **Bricolage Grotesque** for display (headings, Fritz's name, "big moments"), **Space Grotesk** for UI/body with `tabular-nums` for anything that counts (clock, agenda times, unread badges), **Space Mono** for eyebrows, hotkeys, badges and technical readouts; the scale `--t-display 34 … --t-micro 9.5`; the `.eyebrow` (mono 9.5 px, 0.22 em tracking, uppercase, `--text-lo`), `.gradtext`, `.mono`, `.tnum` helpers. *Never Inter/Roboto/Arial; nothing under ~11 px.* Self-hosted woff2, as DivoraVoice and decision 13 both require.
- **The Sigil icon system** — custom geometric arcane SVGs on a 24×24 viewBox, stroked with `currentColor`, no stock icon set. The body reuses the existing `mic`, `output`, `monitor`, `muted`, `settings`, `lock`, `shield`, `keyboard`, `play`, `stop`, `refresh`, `info`, `warning`, `check`, `x`, `chevron*` glyphs and adds a handful of its own in the same hand: `butler` (Fritz), `desk`, `calendar`, `inbox`, `weather`, `listening`, `thinking`, `speaking`.
- **Components**: Button (primary = gradient + glow, secondary, ghost, danger), IconButton, Toggle (40×23, gradient when on), Slider, Badge, Kbd, Segmented, Select/device picker, **HotkeyCapture** (the body's hotkey settings are exactly DivoraVoice's problem — including the "field kept listening after cancel" bug fixed in 1.60.0), EmptyState, Tooltip, Card/Panel, and the two level meters (emerald→gold→crimson with a white peak-hold cap — perfect for the push-to-talk input meter).
- **Brand pillars in the UI.** DivoraVoice shows a persistent mono "LOCAL · NO ACCOUNT" affirmation and an "On-device" shield. Fritz's brain is fully local Ollama (decision 30), so the control window carries the same affirmation and the same shield — a true claim, and the family resemblance costs nothing.
- **Copy voice.** DivoraVoice's rule: eyebrows and labels may be arcane ("The Coven", "Glyph casting", "Hourglass"), actions say exactly what they do (Use, Stop all, Re-scan). The body follows it: themed nouns, plain verbs. Fritz's own butler voice in replies is unchanged by any of this.

### Fritz's house configuration

- **Mood `ink` (ink + candle) by default**, brand accent, dark theme, `--motion` ambient. `ink` is DivoraVoice's warm near-black with candle-gold lines and parchment-toned secondary text — it *is* the dark-academia register of Fritz's web chat (decision 12) expressed in DivoraVoice's token language, so the desk and the web chat agree without a second palette. Dusk Violet, Midnight and the light composites come along for free and are switchable from the tray and the control window (Q9 confirms the default).
- **Parchment grain on, vignette off** in the control window; **neither** in the avatar window (a grain overlay on a transparent window reads as dirt on the desktop).
- **The spell circle becomes the desk's signature.** DivoraVoice's hero is the Mixer's spell circle — a glowing voice core with effects orbiting as sigil-nodes on threads of light. The desk's equivalent, drawn on the wallpaper layer in Phase 3: Fritz's **study circle**, the avatar's position as the core and the ambient widgets (clock, agenda, inbox, weather, desk) as the orbiting sigil-nodes, threads lit when a widget has fresh data. It is the same visual grammar applied to information instead of effects, which is how the two apps stay siblings rather than twins.
- **Glyph casting is a shared lineage, not a copy.** DivoraVoice's background drag-to-cast (triangle/square/circle → preset) and Lorraine's sigil overlay are the same gesture family; the desk may bind glyphs to Fritz actions later (Q13) but ships none in Phases 1–4.

### Rules for every configuration

These are DivoraVoice's and Lorraine's rules, and they bind the body:

- **Reduced motion is a first-class variant.** `prefers-reduced-motion` maps to `--motion: 0` (functional) unless the user lifts it in settings, exactly as DivoraVoice does with `data-motion-user`; the avatar's idle collapses to blink + breathing; ambience off; crossfades only. Screenshot it in every visual PR.
- **Nothing flashes faster than 3 Hz.** Lorraine's `CLAUDE.md` rule; glow and pulse changes ease over ≥ 330 ms.
- **AA contrast on the actual background.** DivoraVoice's token ramps already pass on their own surfaces; the avatar bubble floats over arbitrary desktops, so it always carries its own `--surface-2` glass panel behind text and never trusts the desktop behind it.
- **All chrome is vector/CSS.** DivoraVoice's rule ("no raster images are required by the chrome"), plus: no borrowed artwork, no album art. The VRM and its textures are the owner's VRoid work.
- **Motion is JS-paced in always-visible windows.** This is where the body departs from DivoraVoice's main window, which animates in CSS under `--motion`. Lorraine's PresentMon work showed any CSS animation in an always-visible WebView2 window makes Chromium present every vsync; DivoraVoice's own overlay already obeys this (its timer face is explicitly unanimated so it keeps counting with motion turned down). So in the avatar and wallpaper windows the `--motion` factor scales durations inside the paced loop and CSS never animates; `check-no-css-animation.mjs` enforces it. The control and prompt windows are normal windows and may use DivoraVoice's CSS motion as-is.
- **The avatar is theme-aware only at the edges**: MToon outline colour, rim-light tint and bubble palette follow the active mood/accent so Fritz looks lit by the room; the model itself does not change per mood (per-mood outfits are Q10).

## Prerequisites — brain side, all before Phase 1 ships

These are small, independently revertible, and two of them fix live web-chat bugs on the way.

### P0 — `websockets` becomes a declared dependency. (XS)
It is already in the `requirements.txt` freeze (`:394`, transitive via langgraph-sdk). Add `websockets>=13` to core deps in `pyproject.toml`, update the uvicorn comment. Starlette's `WebSocketRoute` needs it or `wsproto`.

### P1 — The web server can start without Discord. (S)
Extract the `start_admin_panel()` call at `main_discord.py:305` into `serve_web(schedule_manager=None)` in `admin_panel.py`, and add **`main_local.py`**: loads config, starts the Starlette app (admin + chat + companion), no Discord, no Telegram. Keeps `main_discord.py` calling the same function so one process still serves everything when the bot is up. This is also what lets Phase 5's second device reach a brain whose Discord token has expired.

### P2 — `MessageSource.DESKTOP_TEXT` and `DESKTOP_VOICE`. (XS)
`fritz_utils.py:866` enum + a branch in `get_source_info` (`mister_fritz.py:132-156`): voice gets the existing "30 words or less" instruction; text is described to the model as "from the desk, in person". `"desktop"` added to `KNOWN_PLATFORMS` (`:479`).

### P3 — Cooperative cancel in `ask_stuff`. (M) — **also fixes the web chat's "Enough" button**
Add keyword-only `cancel_event: threading.Event | None = None`. The check must live in the **inner** loop: tokens are produced inside the executor node by `agent.stream(..., stream_mode=["values","messages"])` at `mister_fritz.py:844`, and the outer `app.stream(...)` loop at `:1018` runs with `stream_mode="values"`, which only yields after a node finishes — a check there could never interrupt a generation in progress. So thread the event into the executor (via `config["configurable"]["cancel_event"]`, which LangGraph passes to nodes), and inside the `:844` loop do `if cancel_event and cancel_event.is_set(): break`. Breaking that generator closes langchain-ollama's HTTP stream, which stops Ollama generating — a real cancel, not a UI hide. The partial reply is what gets checkpointed; append an `[interrupted]` marker to the stored AI message so the summariser and history window do not treat it as complete (decision needed: § Open questions, Q5). `admin_panel.chat_stream` wires a per-request event to it; the "Enough" notice copy can then be made truthful.

### P4 — Config knobs, one batched commit (house rule from README "one batched config-knob PR"). (XS)
`COMPANION_ENABLED=false`, `COMPANION_TOKEN` (required when enabled; **separate** from `CHAT_PASSWORD`, parallel to decision 2b), `COMPANION_BIND=127.0.0.1` (Phase 5 may widen it), `COMPANION_MAX_CLIENTS=2`, `COMPANION_TTS=xtts|kokoro|off` (Phase 2), `GOOGLE_MCP_ENABLED=false` (Phase 4). All in `fritz_utils.py` `# Tunables` + `.env.example`.

### P5 — `tests/test_companion_api.py` scaffold. (XS)
Starlette `TestClient.websocket_connect` works without a live Ollama when `ask_stuff` is monkeypatched (the eight existing `TestChatStream*` classes in `tests/test_admin_panel.py` show the pattern; the deliberate plain-daemon-thread comment is at `admin_panel.py:929-933` — `plans/README.md` still cites the old `:554-558` location and "five" classes, both stale).

### P6 — A VRM to look at. (owner, ~2 h, no code)
Make the Fritz model in VRoid Studio (free): a butler — waistcoat, grey at the temples, dry expression presets. Export **VRM 1.0**. Keep the file **out of git** (5–50 MB; loaded from `%APPDATA%\com.divora.fritz\avatar\fritz.vrm`, with a tiny CC0 placeholder VRM committed for CI). Confirm the export has the five viseme presets (`aa ih ou ee oh`) and `happy/angry/sad/relaxed/surprised`; VRoid emits them by default.

---

## Spike 0 — prove the persistent overlay (1 day, gates Phase 1)

A throwaway Tauri 2.11 app whose single window is **DivoraVoice's overlay, lifted**: copy `overlay_machine.rs` and `overlay_windows.rs` (and `mark_non_rude` from `divora-capture`), strip the charm/asset and recorder effects, keep the build flags (`transparent`, `decorations(false)`, `shadow(false)`, `always_on_top`, `skip_taskbar`, `visible(false)`, `focused(false)`), the click-through-before-show, the ready handshake + watchdog, the Locked/Arrange modes and `the_overlay_never_calls_set_focus`. Add `set_focusable(false)` (Tauri ≥ 2.8; DivoraVoice predates needing it because Locked never takes focus) and Lorraine's byte-identical `additionalBrowserArgs`. The page is a three.js scene rendering a VRM with a **setTimeout-paced 30 fps loop** (no rAF, no CSS animations — Lorraine's lesson; DivoraVoice's own overlay timer face is unanimated for the same reason), `WebGLRenderer({alpha:true})` with clear alpha 0. "Hide" parks the window at x = −10000 rather than `hide()`/`show()` (the SnapShelf pattern) so it is never re-shown and never re-activated.

What is new relative to DivoraVoice, and therefore what the spike is actually testing: a **continuously animating WebGL canvas** instead of mostly-static pictures, and a window that is **up all day** instead of during a stream.

Run it for a working session alongside normal use and record:

| Check | Pass |
|---|---|
| `GetForegroundWindow()` polled at 10 Hz never returns the avatar HWND or any WebView2 child of it | zero occurrences in 30 min, including at app start, after explorer restart, after sleep/resume |
| Typing in another app while the avatar animates | no dropped keystrokes, no caret loss |
| Clicks anywhere on the avatar window's rect | pass through to the app beneath |
| Transparent areas | no black/white halo; VRM edges premultiplied correctly |
| PresentMon on the avatar process | presents ≤ 30/s while animating; ≤ 1/s when the loop is paused |
| Borderless-fullscreen game on the same monitor | avatar parks itself (reuse `monitor_covered()`), game unaffected |
| GPU busy from Task Manager, avatar idle-animating | agree a threshold with the owner (suggest < 3 % on the dGPU) |

**If it fails** on the foreground check and no amount of `focusable:false` + `WS_EX_TOOLWINDOW` on top of DivoraVoice's discipline fixes it: **Fallback A** — render the avatar inside the WorkerW layer using Lorraine's host/controller reparenting (proven; the avatar is then behind windows, like the brushbuddy, and surfaces only when the desktop is exposed). **Fallback B** — a sidecar native renderer (`bevy_vrm1`, as Desktop Homunculus does) owning its own non-WebView window, driven by the same protocol. Both keep the brain and protocol unchanged.

---

## The PR sequence

Effort tags follow the README's scale — **M = half day, L = 1–3 days, XL = > 3 days** — plus the XS (under an hour) and S (a couple of hours) that plans 11–13 use informally. "Body" PRs live under `desktop/`; "brain" PRs in the root. The desktop toolchain is pnpm + Vite + Vitest + cargo, mirroring Lorraine so its tests and scripts port.

### Phase 1 — avatar + text chat (the milestone)

**PR 1 (brain, M) — `companion_api.py`: the WebSocket route.** `hello`/`welcome`, token check with `compare_digest`, `say` → `run_blocking(ask_stuff, ..., source=DESKTOP_TEXT, streaming_callback=…, cancel_event=…)`, deltas forwarded as `token`, `progress`, one `done`, `fritz_error` on exceptions, 20 s `ping`. Per-connection send queue drained by the websocket task; the agent runs on the bounded pool exactly like Discord does. Register the route in `create_app` behind `COMPANION_ENABLED`. Both `_SameOriginMiddleware` and `_BasicAuthMiddleware` are `BaseHTTPMiddleware`, and the installed Starlette passes non-`http` scopes straight through them (`starlette/middleware/base.py:102`), so a WebSocket route needs no exemption — but add a test that asserts the handshake succeeds with no Basic-auth header and no `Sec-Fetch-Site`, so a future middleware change cannot silently lock the desk out. Tests: handshake, bad token, one streamed turn, cancel mid-stream (asserts `interrupted: true`), two clients, heartbeat.

**PR 2 (body, L) — scaffold + avatar window.** `desktop/` with `tauri.conf.json` (windows: `avatar`, `prompt`, `control`; identical browser args), Rust `overlay_machine.rs` + `overlay_windows.rs` as proven in Spike 0 (Locked/Arrange, handshake, watchdog, NonRude, no-`set_focus` test), `windows.rs` for park/unpark, `set_focusable(false)` and `WS_EX_TOOLWINDOW` via the `windows` crate so Alt-Tab never lists it, tray (`Show/Hide Fritz`, `Mute`, `Settings`, `Quit`), `settings.rs` lifted from Lorraine with the gift fields removed, `log_startup`/`log_crash`/`guard`. Frontend `src/avatar/`: three.js + `@pixiv/three-vrm` 3.x loader, procedural idle (breathing on `chest/upperChest`, random blink every 2–6 s, `vrm.lookAt.target` following the cursor from a Rust-emitted `cursor` event at 15 Hz), the paced loop, pause on `power:display=false` and when the monitor is covered. Acceptance = Spike 0's table, now in the real app.

**PR 3 (body, M) — theme contract + companion client + speech bubble.** `src/divora/` with `tokens.css` (DivoraVoice's `src/styles.css` `:root`, mood, accent, light and motion blocks, copied under `scripts/sync-divora-tokens.mjs`), `fonts/` (Bricolage Grotesque, Space Grotesk, Space Mono as self-hosted woff2), `sigils/` (the lifted Sigil set + the body's additions) and `theme.ts` (sets `data-theme`/`data-mood`/`data-accent`/`--motion` on every window from settings and a `settings:theme` event). `src/shared/companion.ts` WebSocket client with reconnect/backoff, the state machine, and a DOM speech bubble anchored to the VRM head bone's projected position, streaming `token`s on its own glass panel; its materialise/fade timings follow the shared motion spec and its materials come from the active theme. `progress` shows as a small secondary orb/line; `error` as the butler's apology. Ships with the reduced-motion variant and `check-no-css-animation.mjs` in the build.

**PR 4 (body, M) — the prompt window.** `prompt` window: small, undecorated, transparent, always-on-top, **focusable**, hidden. Global shortcut `Ctrl+Alt+F` (tauri-plugin-global-shortcut 2.3.2) → `snapshot_foreground()` → show → focus input. Enter sends `say` and hides immediately (the reply streams in the avatar bubble); Escape hides. Hide → `restore_foreground()`. Exactly Lorraine's modal pattern, so it is known to work. Shift+Enter for newline; up-arrow recalls the last prompt.

**PR 5 (body, S) — avatar interaction: Arrange mode + the mouse hook.** Moving the avatar uses DivoraVoice's **Arrange** mode as shipped: a tray item / `Ctrl+Alt+A` lifts click-through for the avatar window, which takes the foreground via `try_foreground`, lets the owner drag it, and re-locks (`return_focus`) on Esc, the tray or a 30 s lease. Everyday interaction while **Locked** stays hook-based, lifted from Lorraine's `hook.rs`: the frontend publishes the VRM mesh's projected rect with `set_hit_rects` when it moves > 2 px; right-click inside it pops the tray menu, double-click opens the prompt. Toggling `ignore_cursor_events` therefore happens only on an explicit mode change, as DivoraVoice already does in production — not on hover, which is where Tauri #11461 bites.

**PR 6 (brain+body, S) — `IDENTITY_LINKS` for the desk + docs.** `.env.example` example linking `desktop-<machine>` → `discord-<root>`; `desktop/README.md` with the build gotchas (custom-protocol, exe lock, Defender); root `README.md` gains a "Desktop companion" section; `plans/README.md` index entry (Appendix D).

**PR 7 (CI, S) — `desktop-ci` job** on `windows-latest`: `pnpm install --frozen-lockfile`, `pnpm test`, `cargo check --features custom-protocol`, `scripts/gen_protocol_ts.py --check`. No WorkerW/overlay tests run headless; those stay manual with the Spike 0 table as the checklist.

**Phase 1 definition of done.** Ctrl+Alt+F from any app, type "good evening", Enter — the prompt vanishes, the avatar looks up, the bubble streams Fritz's reply, the foreground app never lost focus, and the exchange appears under the desk thread in the admin panel's user page. Idle for an hour: avatar breathing, GPU presents at the cap, no foreground steals logged. Switching mood (ink ↔ dusk ↔ midnight), accent or light/dark from the tray recolours bubble, prompt and control window without a restart; `--motion: 0` leaves only blink and breathing moving.

### Phase 2 — voice: push-to-talk in, spoken reply out

**PR 8 (body, M) — microphone capture in Rust.** `audio.rs` gains a capture path on the default **capture** endpoint — via `wasapi` (Lorraine's crate, `Direction::Capture`) or `cpal` (DivoraVoice's mic path; both are proven on the owner's hardware) — 16 kHz mono PCM16, started on `Ctrl+Alt+Space` **Pressed** and stopped on **Released** (the plugin emits both states since 2.x), 30 s cap, sent as one binary WS frame + `audio_end`. Rust capture avoids WebView2's microphone permission prompt entirely (Tauri 2.12 added `on_permission_request` if the getUserMedia route is ever preferred — see Q4). Tray `Mute` kills the hotkey.

**PR 9 (brain, M) — audio turn.** `companion_api` writes the clip to a temp file, `stt.transcribe`, emits `stt`, runs the turn as `DESKTOP_VOICE`. **Sentence-chunked TTS:** a `SentenceChunker` consumes the delta stream and, at each sentence boundary, calls `TTSEngine.generate_speech` on a dedicated single worker (respecting `TTS_MAX_CONCURRENCY`), reads the WAV, emits `speech{seq}`. First audio arrives after the first sentence rather than the whole reply. `TTSEngine` construction moves out of `main_discord.on_ready` into a lazy shared accessor so `main_local.py` has it too (this also makes plan 13 A1's "failed TTS load" path explicit). Cache hits (`output/speech_<md5>.wav`) make repeated butler phrases instant.

**PR 10 (body, M) — playback + lip-sync.** Ordered playback queue (Web Audio), `AnalyserNode` → RMS → `expressionManager.setValue('aa', v)` with attack/decay smoothing; a cheap three-band vowel estimate (`aa/ih/ou`) is a stretch goal. `speaking` state until the queue drains; `cancel` on PTT press while speaking (barge-in) flushes the queue and sends `cancel` — this is why P3 exists.

**PR 11 (brain, S, optional) — mood tags.** Decide Q3 first. If "inline tags": the desktop system-prompt branch asks for a leading `[mood:dry|pleased|alarmed|weary]` token, stripped before `token` events and emitted as `state`-adjacent `mood`; the Discord path strips it too. If "none": skip.

**Phase 2 DoD.** Hold Ctrl+Alt+Space, ask for the time, release: transcript appears in the bubble within ~1 s, Fritz begins speaking within ~3 s of release on the agreed hardware (see Q1), mouth moves with the audio, pressing PTT mid-sentence cuts him off and Ollama stops generating (verified with `ollama ps` load dropping).

### Phase 3 — the ambient layer: live wallpaper + widgets

**PR 12 (body, L) — WorkerW wallpaper.** Lift `workerw.rs`, `host.rs`, `power.rs`, the slot/`ensure_attached`/watchdog code from `lib.rs`, the paced loop, and `check-no-input.mjs`. Windows `wallpaper-0..3` declared in config. The backdrop is a **study scene** built from the Divora tokens — the active mood's `--bg`/`--surface` ramp as a slow radial gradient, the **study circle** (the desk's spell circle: a faint outer ring with runic ticks, the avatar's position as the core, widget sigil-nodes on the orbit with threads lit when their data is fresh), parchment grain where the mood wants it — driven by the paced loop with `--motion` scaling, and a static `--motion: 0` variant. It uses neither Lorraine's palette nor art; Lorraine's content never enters this repo.

**PR 13 (body, M) — widgets + layout editor.** Widget DOM in the wallpaper page; `control.ts` editor lifted. Widgets at this phase: **clock**, **weather** (Open-Meteo, no key — the owner's JUCE wallpaper plan already specified it), **Fritz's desk** (last exchange, current state, a one-line "note to self" Fritz can write via a new `desk_note` tool), **now playing / audio seal** (optional; `audio.rs` loopback already exists and feeds `--pulse-ambient`). Widget chrome, type roles and labels come from the active theme (clock and agenda times in the data monospace; titles in the display face, used nowhere else). Layout persisted via `settings.rs`.

**PR 14 (brain, M) — widget feed + `DeliverySink`.** `companion_api` runs a per-connection asyncio task that pushes `widget` frames on each widget's cadence, computed by plain Python (no LLM). `scheduler.py` gets a `DeliverySink` protocol (`deliver(identity, text) -> bool`) with the Discord channel as the first implementation and the companion connection as the second; `_run_task` tries the sinks registered for the identity. This is the first step of plan 13's deferred reminder/timezone work and should land with an answer to its Q1.

**Phase 3 DoD.** Desktop shows the study behind icons on every monitor; widgets drag/resize from the control window and survive explorer restart, sleep/resume and a DPI change; a `/schedule` reminder set from Discord pops in the desk widget and the bubble when the desk client is connected.

### Phase 4 — Google, read-only, owner-only

**PR 15 (brain, M) — MCP client.** `langchain-mcp-adapters` (0.3.x, `MultiServerMCPClient`) in core deps. `google_mcp.py` starts `workspace-mcp` (taylorwilsdon, MIT, Python, PyPI `workspace-mcp`) as a **stdio subprocess** via `uvx workspace-mcp --tools gmail drive calendar` with `WORKSPACE_MCP_PERMISSIONS="gmail:readonly drive:readonly calendar:readonly"`, loads tools once at startup, and exposes them as `extra_tools` to `get_conversation_tools_description` **only when `resolve_identity(user_id) == ROOT_USER`**. This gate is the whole security story: Fritz is a multi-user Discord bot, and nobody but the owner may ever trigger a Gmail tool, from any surface. Test: a non-root identity's tool list contains no `gmail_*`/`drive_*`/`calendar_*` names.

**PR 16 (owner + brain, S) — OAuth setup.** Google Cloud project → OAuth client type *Desktop app* → enable Gmail, Calendar, Drive APIs → scopes `gmail.readonly`, `calendar.readonly`, `drive.readonly` → **set the consent screen's publishing status to "In production"** (unverified-app warning is acceptable for personal use). Leaving it in *Testing* expires the refresh token every 7 days and the integration silently dies weekly. Credentials live under `%APPDATA%\com.divora.fritz\google\`, `.gitignore`d, never in `.env`. `scripts/setup.py` gains a `--google` step that runs the first consent flow.

**PR 17 (brain+body, M) — agenda + inbox widgets.** Widget feed calls the MCP tools directly every 5 min (calendar) / 2 min (gmail unread count + top 3 subjects), with results cached so the agent's own "what's on today?" turn reads the cache first. Nothing is written to Google in this phase; the `send`/`create` tools are simply not loaded.

**Phase 4 DoD.** "What's on my calendar tomorrow?" from the desk answers from Google; the agenda widget updates without a turn; a second Discord user asking the same question gets Fritz's refusal, not the owner's calendar.

### Phase 5 — hardening and the second device

- **Autostart** via Lorraine's Startup-folder `.lnk` (`autostart.rs`; the HKCU Run key silently failed there). **NSIS** per-user installer (MSI can't do per-user). Defender false-positive note in `desktop/README.md`.
- **Remote reach**: `COMPANION_BIND` may be set to the Tailscale interface; token auth is already required; TLS is Tailscale's. Never bind `0.0.0.0`.
- **Second Windows PC**: the same installer, a different `client.id`, the same `IDENTITY_LINKS` target. Settings stay per device (window positions are hardware-specific); memory and threads are the brain's.
- **Android spike (S)**: Tauri 2 mobile builds the same TypeScript frontend into an Android WebView; the avatar + prompt + companion client are platform-neutral, and Rust plumbing is `cfg(windows)`. Treat as a spike, not a promise — mobile Tauri is younger than desktop.
- **Proactive mode (decision-gated)**: with `DeliverySink` and the widget feed in place, "meeting in 10 minutes" is a scheduler job plus quiet-hours rules; it is deliberately not in Phases 1–4 because the interview chose reactive + ambient.

---

## What not to build

- **No second brain in TypeScript.** The body never calls Ollama, Chroma or Google directly.
- **No reuse of `/chat/stream` or the chat cookie for the desk.** Different origin, different lifetime, needs cancel and audio.
- **No `ignore_cursor_events` toggling** for hover; the hook + rect model is proven and sidesteps Tauri #11461.
- **No rAF render loop and no CSS animations** in any always-visible window — DivoraVoice's `--motion`-scaled CSS keyframes stay in the control and prompt windows; in the avatar and wallpaper windows the same factor scales JS-paced motion.
- **No second design system.** Tokens, type, sigils and components come from DivoraVoice; the body adds sigils and a study scene in that hand, never a parallel palette.
- **No photographic or borrowed textures, no album art**; every theme asset is vector/CSS or the owner's own work.
- **No Lorraine gift content**: recognizer, casting, dedication, spellbook, Witch Hat assets, `#0b0914` palette. Lift plumbing only (Appendix B).
- **No Google write scopes, no wake word, no unprompted speech** until a decision explicitly adds them.
- **No VRM in git** beyond the CC0 placeholder.

---

## The honest risk summary

| Risk | Likelihood | Blast radius | Mitigation |
|---|---|---|---|
| Persistent WebView2 overlay steals foreground (Lorraine's failure) | low — DivoraVoice's overlay has shipped since v1.55 with the same window flags and a no-`set_focus` test | Phase 1 architecture | Spike 0 lifts that code and adds only a live WebGL canvas; Fallback A (WorkerW layer) and B (`bevy_vrm1` sidecar) keep brain + protocol intact |
| **VRAM contention.** `gpt-oss` ≈ 14 GB; XTTS v2 on GPU ≈ 2–3 GB (estimate); SDXL ≈ 7 GB on demand; the VRM renderer a few hundred MB; games want the rest | high on a 16 GB card | voice latency, Ollama spilling to CPU (11× slower) | Q1 decides. Options: TTS on CPU (Kokoro-82M ONNX is near-realtime on CPU; XTTS is not), a smaller model for `DESKTOP_VOICE` turns, `OLLAMA_KEEP_ALIVE`/`OLLAMA_MAX_LOADED_MODELS=1` tuned, avatar parks + loop pauses while a game is fullscreen |
| XTTS first-sentence latency too high for conversation | medium | Phase 2 feel | Sentence chunking (PR 9) + cache; evaluate Kokoro/Piper behind `COMPANION_TTS` |
| Transparent WebGL quirks in WebView2 (halo, black background after resume) | medium | cosmetic → ugly | `alpha:true`, premultiplied output, clear alpha 0; Lorraine's resume path already rebuilds controllers after sleep |
| Neon theme fails contrast over real desktops (the bubble floats over anything) | high if unguarded | unreadable replies | Bubble always carries its own glass panel; neon tokens are accents only; AA checked against the panel, not the desktop; visual-regression shots on light and dark wallpapers |
| Theme motion creeps back into CSS and wakes the GPU | medium | idle GPU cost returns | `check-no-css-animation.mjs` in `pnpm build`, PresentMon re-check in each visual PR |
| Google consent screen left in *Testing* | certain if forgotten | integration dies every 7 days | PR 16 step is explicit; `setup.py --google` warns if status unknown |
| Owner-only gating missed on one path | low | another Discord user reads the owner's inbox | Gate at tool-list construction (PR 15), not in the prompt; a test asserts non-root lists are clean |
| Windows 11 updates change WorkerW again | low–medium | wallpaper layer (Phase 3 only) | Lorraine's watchdog + Progman-child path handled 24H2; avatar overlay does not depend on WorkerW |
| Plan 13's open timezone decision | certain to surface | agenda widget shows wrong hours | Answer Q1 of plan 13 before PR 17 |
| Defender ML false positive on the unsigned exe | high | install friction | Documented; per-SHA clearance; consider code signing in Phase 5 |

---

## Open questions, ranked by how much they block work

1. **What GPU and how much VRAM?** Decides Q1's model/TTS placement and the Spike 0 GPU threshold. Blocks Phase 2 planning, not Phase 1.
2. **TTS engine for the desk** — (a) XTTS v2 in the brain (same voice as `/voice`, GPU, slower); (b) Kokoro-82M in the brain via Python (CPU, fast); (c) **Kokoro-82M in the body**, lifting DivoraVoice's Rust pipeline (`divora-core/src/tts/{kokoro,phonemize}.rs`, `ort` + espeak-ng) — zero VRAM, already shipping, and its preset `bm_george — crisp (UK)` is a ready-made English-butler voice; the brain then sends text and the body speaks. (c) also opens DivoraVoice's accent-preserving cloning as a route to a bespoke Fritz voice later. The trade is that Discord `/voice` and the desk would speak with different voices unless `/voice` moves to Kokoro too. Could be any two behind `COMPANION_TTS`.
3. **Mood/expression source** — inline `[mood:…]` tag from the model (zero latency, touches Discord's prompt path), a FAST-model classifier per sentence (off the critical path, +latency), or none in Phase 2.
4. **Microphone path** — Rust `wasapi` capture (recommended; no permission prompt) vs WebView `getUserMedia` with Tauri 2.12's `on_permission_request`.
5. **Cancel semantics** — checkpoint the partial reply with an `[interrupted]` marker (recommended) vs drop the exchange.
6. **Hotkeys** — `Ctrl+Alt+F` prompt, `Ctrl+Alt+Space` PTT, `Ctrl+Alt+H` park/unpark? Lorraine owns `Ctrl+Alt+W` if both run.
7. **Desk thread** — its own thread via the explicit `thread_id=` override (recommended) vs sharing the owner's Discord DM thread so a conversation can continue across Discord and the desk.
8. **Does `main_local.py` also run the scheduler** (needs the `DeliverySink` work from PR 14 first) or only chat + companion in Phase 1?
9. **Default configuration on first run** — mood `ink` + brand accent + dark (recommended: it is the web chat's dark academia in Divora's token language) or DivoraVoice's own default, Dusk Violet. All moods ship either way.
10. **How far does theme reach the character** — tint only (outline + rim light + bubble; recommended for Phase 1) or per-theme VRM outfits (a second VRoid export per theme, swapped with the theme)?
11. **UI sounds** — a small themed set (prompt open, reply done, reminder) or silence except for Fritz's voice.
12. **Frontend framework for the body's chrome** — SolidJS + Tailwind, as DivoraVoice, so its components port as code (recommended; the avatar canvas is three.js either way and Solid's fine-grained signals suit a 30 fps HUD), or vanilla TypeScript as Lorraine, porting DivoraVoice's CSS only. Preact (the owner's web stack) is the third option but would make the components a rewrite.
13. **Glyph casting on the desk** — bind DivoraVoice/Lorraine-style drawn glyphs to Fritz actions (triangle → "what's next?", circle → mute) in a later phase, or leave gestures out of the companion entirely.

### Experiments, not questions

- PresentMon the Spike 0 window at 24 vs 30 fps and with/without spring-bone physics; pick the cap from data.
- Measure XTTS vs Kokoro first-sentence latency on the actual GPU with `gpt-oss` resident.
- Try `faster-whisper` `small` on GPU `float16` vs the current CPU `int8` for a 5 s PTT clip.

---

## Decisions to record (proposed for `DECISIONS.md`, continuing from 27)

**28. Brain/body split.** The brain is this repo; the body is a Tauri 2 app under `desktop/`. The body contains no intelligence and talks only the companion protocol. *(interview)*

**29. Avatar = VRM in a persistent, non-focusable, click-through overlay**, gated on Spike 0; Fallback A (WorkerW layer) if the spike fails. *(interview + Lorraine warning)*

**30. The brain runs on the owner's Windows PC with fully local Ollama.** No cloud model, no home server, in Phases 1–4. *(interview)*

**31. Google is read-only via `workspace-mcp`, loaded only for `ROOT_USER`.** Write scopes need a new decision. *(interview)*

**32. `COMPANION_TOKEN` is its own secret**, separate from `CHAT_PASSWORD` and `ADMIN_PANEL_PASSWORD`, by the same reasoning as 2b. *(proposed)*

**33. Phase 1 is avatar + text chat only.** Voice, wallpaper, widgets and Google are Phases 2–4. *(interview)*

**34. The body adopts the DivoraVoice design system** — its tokens, axes (`data-theme` × `data-mood` × `data-accent` × `--motion` × texture), type roles (Bricolage Grotesque / Space Grotesk / Space Mono), Sigil icon system and components — with Fritz's house configuration being mood `ink`, brand accent, dark. Reduced motion, the 3 Hz floor, AA contrast, vector-only chrome and plain-verb copy bind every configuration; in always-visible windows motion is JS-paced, never CSS. Decision 12 (chat-only scope for the dark-academia web theme) is unaffected: the desk expresses the same register through Divora's tokens, it does not extend the web theme's scope. *(owner request, Nov 2026)*

---

## Also worth doing, noticed in passing

- Plan 13's four undecided questions (timezone, stop path, skills contract, turn bounds) all touch this work: the agenda widget needs a timezone, `main_local.py` needs a stop path to drain a desk turn, and a per-turn deadline is what keeps one stuck voice turn from freezing the avatar in `thinking`. Answer them before Phase 3.
- Lorraine's `desktop_exposed_at` returns true for *any* window in its own process (`workerw.rs:360-364`); when lifting `hook.rs`, make the exposure test explicit per hit-target so a visible control window can't swallow clicks.
- Lorraine's `pause_on_fullscreen` default disagrees between `lib.rs:237-244` ("OFF") and `settings.rs:109` (`true`); pick one when lifting.
- `_CHAT_ASSET_CONTENT_TYPES` could gain `audio/wav` so the web chat can also play `/voice` output — cheap once PR 9 exists.
- **`/forget all` can be partly undone by the turn it raced** (found reviewing `3258948`, which made every conversation forget wait for the turn in flight; older than that commit). `privacy.forget_all` deletes memories *before* `forget_conversation` waits, and the turn's own memory writers run outside its gate: `extract_memories_background` starts after `ask_stuff` has released the gate, and `summarize_conversation`'s worker runs on a daemon thread. Both make an LLM call and then write to Chroma, so facts from the exchange being forgotten land after the delete — reproduced in review with a write 0.92 s after `forget_memories`, inside the same `/forget all`. Moving `forget_conversation` first in `forget_all` covers memory tools called during the turn, but not the background writers. Suggested fix: a per-identity erasure epoch in `storage`, bumped by `forget_memories` under a lock, captured by `ask_stuff` when the turn starts, and checked under the same lock just before each background write; plus a test that times the writer against the delete, which `tests/test_forget_race.py::test_forget_all` cannot see because its harness stubs the extractor.

---

## Appendix A — `desktop/` layout

```
desktop/
  package.json  vite.config.ts  tsconfig.json  vitest.config.ts
  index.html (avatar)  prompt.html  control.html  wallpaper.html (Ph3)
  src/
    avatar/   main.ts  vrm.ts  loop.ts  lipsync.ts  bubble.ts  state.ts
    prompt.ts  control.ts  wallpaper.ts (Ph3)
    shared/   companion.ts  protocol.ts (generated)  settings.ts  motion.ts (shared timings)
    divora/   tokens.css (synced from DivoraVoice)  theme.ts  fonts/ (woff2)  sigils/  components/ (ported)
  src-tauri/
    tauri.conf.json  capabilities/default.json  Cargo.toml
    src/  lib.rs  main.rs  overlay_machine.rs  overlay_windows.rs (from DivoraVoice)  windows.rs  tray.rs  hook.rs  settings.rs
          audio.rs (Ph2)  power.rs  workerw.rs (Ph3)  host.rs (Ph3)  autostart.rs (Ph5)
  assets/avatar/placeholder.vrm (CC0, tiny)
  scripts/check-no-input.mjs  scripts/check-no-css-animation.mjs  scripts/sync-divora-tokens.mjs
  tests/visual/   (Skia-canvas harness; bubble + widgets × {ink, dusk, midnight, light} × {motion 1, motion 0})
  README.md
```

Brain-side additions: `companion_api.py`, `companion_protocol.py`, `google_mcp.py` (Ph4), `main_local.py`, `scripts/gen_protocol_ts.py`, `tests/test_companion_api.py`, `tests/test_google_mcp.py`.

## Appendix B — what to lift from DivoraVoice and Lorraine, and what must not come

**From DivoraVoice (public, MIT):**

| Lift as-is | Lift and trim | Leave behind |
|---|---|---|
| `overlay_machine.rs` (pure machine + tests), `the_overlay_never_calls_set_focus`, `mark_non_rude`/`unmark_non_rude`, `src/styles.css` token/mood/accent/light/motion blocks, type roles + font files, the Sigil set and `DMark` pattern (new mark for Fritz), `HotkeyCapture`, `Kbd`, `Badge`, `Toggle`, `Segmented`, `Select`, meters, `inOrder.ts`, the push-to-modulate press/release edge handling | `overlay_windows.rs` (drop charm assets, recorder hooks, OBS title; keep build/handshake/watchdog/Locked-Arrange/focus), `tts/{kokoro,phonemize,speaker,tokens}.rs` **if Q2 picks (c)**, the first-run wizard shell (steps become: Fritz brain reachable → VRM → hotkeys → ready) | DSP chain, voice conversion, soundboard, recording/capture pipeline, MIDI/Stream Deck, Hourglass, Coven presets, the `DivoraVoice Overlay` window title (OBS matches it), identifier `com.divora.voicemod` |

**From Lorraine (private):**

| Lift as-is (rename identifiers) | Lift and trim | Leave behind |
|---|---|---|
| `workerw.rs`, `host.rs`, `power.rs`, `settings.rs` store + atomic save, `autostart.rs`, `log_*`/`guard`, watchdog + `ensure_attached`, `reparent_controller`, buddy roaming engine, widget layout editor, paced loop, `check-no-input.mjs`, visual test harness | `hook.rs` (generic mechanism; app-specific `HitTarget`), `audio.rs` (add capture path), `tauri.conf.json` window pool + browser args, `overlay.ts` → only the `overlay:mode` handshake | `recognizer.rs`, `casting.rs`, corpus + oracle, `dedication.ts`, `almanac.ts`, `palette.ts`, `elementSigil.ts`, `seal.ts`, `viz.ts`, `ambient.ts`, all `preview_*`/gift commands, `Gift` settings, `public/` art and fonts, `mockup/`, `WHA_MAGIC_RESEARCH.md`, `GIFT_BACKLOG.md`, identifiers `com.divora.lorraineatelier` / `LorraineWallpaperHost` / `LorraineSessionWatch` / `lorraine-tray`, colour `#0b0914` |

Lorraine's repo README asks that it stay private; nothing from its gift content, and no file history, is copied into this public repo.

## Appendix C — prior art checked

- **Open-LLM-VTuber** (MIT, Python FastAPI backend + separate desktop client, Live2D, "desktop pet" transparent always-on-top mode): the same brain/body split as this plan, so the architecture is not novel risk. Its LLM→expression mapping via inline tags informs Q3.
- **Desktop Homunculus** (Apache-2.0, Rust/Bevy, VRM + VRMA, built-in MCP server, Windows needs an NVIDIA config): Fallback B's renderer, and proof VRM mascots work on the Windows desktop natively.
- **DivoraVoice's stream overlay** (the owner's own, Tauri 2, v1.55+): the primary evidence that a persistent click-through always-on-top WebView2 window works when it never takes focus; Spike 0 lifts it.
- **Zebar** and **SnapShelf** (Tauri 2, persistent always-on-top transparent WebView2 windows in production): independent corroboration.
- **AITuberKit** and **three-vrm-utils** (VRM + LLM in the browser; vowel lip-sync from an `AnalyserNode`): reference implementations for PR 10.
- **Lively Wallpaper**: the WorkerW reference Lorraine already ported from; its 24H2 fixes are the regression suite to watch.

## Appendix D — index entry for `plans/README.md`

> [14. Desktop companion](14-desktop-companion.md) — Fritz gets a body: a VRM avatar in a Tauri 2 overlay talking to the brain over a new `/companion` WebSocket (text in Phase 1, push-to-talk voice in Phase 2), then Lorraine-style live wallpaper and widgets (Phase 3), read-only Gmail/Calendar/Drive through `workspace-mcp` gated to `ROOT_USER` (Phase 4), and a second device over the same protocol (Phase 5). Phase 1 is gated on a one-day overlay spike because of Lorraine's foreground-stealing finding. The overlay lifts DivoraVoice's shipped `overlay_machine`/`overlay_windows` and the chrome adopts DivoraVoice's design system (tokens, moods, sigils, components), with Fritz's house look as the `ink` mood. Decisions 28–34 proposed. The execution-level companion is [14-desktop-companion-build.md](14-desktop-companion-build.md).
