# Mister Fritz — the desktop body

Fritz's body on the Windows desktop: a Tauri 2 app that talks to the brain
over the `/companion` WebSocket ([plan 14](../plans/14-desktop-companion.md),
[build plan](../plans/14-desktop-companion-build.md)). Windows only.

Today it is one persistent, transparent, click-through, never-focused avatar
window with a WebGL VRM in it, a control window (tray > Settings…), a tray,
three hotkeys, and the prompt (*Ask Fritz*), Ctrl+Alt+F (PR 4 of the build
plan is under way). Spike 0 proved the window ([docs/SPIKE0.md](docs/SPIKE0.md)), and
its table is PR 2's acceptance.

## What it keeps, and where

Everything is in `%APPDATA%\com.divora.fritz\` (the tray's *Open Fritz's logs*
opens it):

| File | What |
|---|---|
| `settings.json` | The settings ([`settings.rs`](src-tauri/src/settings.rs)), saved as they change. A file that doesn't parse is moved to `settings.corrupt-<time>.json` and the defaults used; one that can't be read is left alone and not saved over that run |
| `companion.token` | The brain's `COMPANION_TOKEN`, saved by Settings › Brain ([`companion.rs`](src-tauri/src/companion.rs)). Never in the settings or a log, and given only to his own window |
| `startup.log`, `crash.log` | A line per start, stop and resume, and every panic. Kept across runs (the previous one as `.old` past 64 KiB): read these after a start that died |
| `logs\fritz.log` | This run's detailed log, started afresh each run |
| `diagnostics\foreground.log` | Spike 0's foreground logger, **off** unless `diagnostics.foregroundLog` is `true` (read at start): it writes other apps' window titles |
| `avatar\fritz.vrm` | The owner's model, if there is one (or `avatar.vrmPath`) |

The hotkeys are `hotkeys.arrange` (Ctrl+Alt+A), `hotkeys.park`
(Ctrl+Alt+H) and `hotkeys.prompt` (Ctrl+Alt+F), registered at start and
set in Settings › Hotkeys: click a key's field and press the new chord (Esc
cancels). It is bound the moment it is saved: checked (Ctrl and Alt
together, or Win, or F13 to F24 alone; no other action's key, nor
push-to-talk's Ctrl+Alt+Space), taken from Windows, saved, and only then is
the old key let go, so a key another app holds is refused on its row with
the old one still working. A chord Fritz or another app already holds does
its job instead of reaching the field. Each row says when the key set isn't
the one working (it fell back to its default at start, or Windows wouldn't
give it), and *Default* puts the default back. A key edited into
`settings.json` by hand takes effect at the next start. His eyes follow the cursor while
`avatar.lookatCursor` is `true` (the default): on a VRoid export the eyes
turn, and on any model his head turns part of the way, which is what shows
on the placeholder.

Both windows wear `settings.theme`: the mood (`ink`, `dusk`, `midnight`),
the accent (`brand`, `abyssal`, `ember`), dark or light, and the motion
(0 to 1: functional, ambient, rich), set in Settings › Appearance or the
tray's *Appearance* menu, and changed in both windows at once. He takes a
tint from it, his outlines leaning toward the mood and a rim light in the
accent. With no motion, or while Windows asks for less (its *Animation
effects* switch) and *Keep his motion when Windows asks for less*
(`theme.motionUser`) is off, he only blinks and breathes: no sway, and his
eyes stay ahead. The parchment grain (`theme.grain`) is the control
window's alone.

The tray's *Settings…* opens the control window: his brain (where it is,
the token, and whether he is connected), show or hide him, arrange him, his
eyes following the cursor, his hotkeys, the theme, which model he wears
(with its folder, and loading it again after you drop in a new export), his
logs, and the foreground log, and a question to ask him. Closing it only
hides it. The rest (`avatar.vrmPath`, `brain.clientId` and
`brain.clientName`) is still set by editing `settings.json` with Fritz quit,
until the sections for them land.

## His brain

He connects to the brain's `/companion` WebSocket at `brain.url`
(`ws://127.0.0.1:8001/companion`, the admin panel's port), with the brain's
`COMPANION_TOKEN`: start the brain with `COMPANION_ENABLED=true` and a
`COMPANION_TOKEN` of 32 characters or more in its `.env`, then paste the
token into Settings › Brain and press *Connect*. Until there is a token,
Settings shows at every start with the first run in place of its sections:
four steps (connect his brain, choose his look, learn his keys, summon him),
with *Skip setup* at every step and *Summon Fritz* at the last, which brings
him onto the desktop and puts Settings away. It waits to show while the lock
screen, a fullscreen app or a presentation is in front, and never opens over
them. Nothing about it is saved; About › *Replay the first run* shows it
again.

His own window holds the connection (the control window only hears of it):
it says hello, is online once the brain welcomes it, and answers the brain's
pings. A wrong token, a protocol version the brain doesn't speak, or a hello
it can't read is a refusal, which he never retries by himself (*Reconnect*,
or a new token, tries again). Anything else (the brain down, full or quiet)
he retries after 1, 2, 4, 8, 16, then every 30 seconds, and Settings says
why and when. The protocol's TypeScript (`src/shared/protocol.ts` and its
fixtures) is generated from the brain's `companion_protocol.py` by
`python scripts/gen_protocol_ts.py` at the repository's root; the brain's
pytest fails while they differ.

## His speech bubble

His replies show in a bubble over his head, on its own panel in the mood's
`--surface-2` (it floats over whatever is on the desktop, so it never
trusts the desktop behind it). The prompt (Ctrl+Alt+F, below) or Settings
› Brain's *Ask him something* asks him: he glances up while
he thinks, the reply streams in, a tool's notice shows as a smaller line,
and the brain's last word on the reply replaces what streamed. It holds 6
seconds, or longer for a long reply (50 ms a character, at most 30
seconds), and longer while the cursor is over his window (only while his
eyes follow the cursor: otherwise the app hears nothing of it), then
fades. A turn that fails shows his apology; asking again while he answers
stops the first reply. Pictures aren't shown in the bubble yet: it says
how many came. With no motion, the bubble only fades in and out, and he
doesn't glance or tilt. While he is hidden (the tray, the hotkey, or a
fullscreen app on his monitor) the bubble waits, so a reply that came
meanwhile is there to read when he is back.

His window is 340×640 the first time (shorter on a screen that hasn't the
room), the top 28% kept clear for the bubble (he stands as tall as he did in
300×460). A window you arranged before keeps its size, so he looks smaller
in it until you arrange him taller.

## The prompt

*Ask Fritz* is a small window, hidden until Ctrl+Alt+F (`hotkeys.prompt`)
opens it, just above Fritz on his monitor (below him if there's no room
above; where he would stand while he is hidden), with the keyboard in it.
The key again closes it. It doesn't open over the lock screen, another
app's fullscreen window (a game, a video, a presentation), or while you are
arranging him: the key then does nothing, and takes nothing. Enter asks him. The prompt closes
once his window has sent the question on to the brain; a refusal (he isn't
connected, his window is closed) leaves your words in it with the reason.
Shift+Enter is a new line, ↑ in an empty field brings back the last
question, and Esc closes it and keeps what you'd typed. Closed by Enter, Esc
or Alt+F4, it gives the keyboard back to the window you were in; clicking
away leaves it where you clicked. Its line says whether he can be asked now,
in Settings' words. Nothing in it moves, and it uses only the text colours
that read well on its panel in every mood.

## Build

Node ≥ 20.19, pnpm, Rust stable (MSRV 1.89, MSVC toolchain) and the Visual
Studio C++ Build Tools. From `desktop/`:

```powershell
pnpm install
pnpm build                                     # checks, type-checks and bundles the pages into dist/
cargo build --release --locked --features custom-protocol --manifest-path src-tauri/Cargo.toml
```

The app is `src-tauri/target/release/mister-fritz-desktop.exe`.

- **Always pass `--features custom-protocol`** to any `cargo build` or
  `cargo test` you run yourself. Without it the exe is rebuilt as a dev build
  that loads `http://localhost:1431`, and the window stays empty. To
  type-check only, `cargo check --manifest-path src-tauri/Cargo.toml` is safe.
- **`--locked`, and the exact Tauri pins in `Cargo.toml`, matter**: tauri
  2.11's own version ranges also accept the 2.12 runtime crates, and that mix
  does not compile. Move tauri, tauri-build, tauri-runtime, tauri-runtime-wry
  and tauri-utils together, never one at a time.
- **The pages are embedded in the exe at compile time**: after any change to
  a page (`avatar.html`, `control.html`) or under `src/` or `public/`, run
  `pnpm build` *and* the cargo build again.
- **Smart App Control** (Windows 11) blocks the Tauri CLI's unsigned native
  module on some machines, so `pnpm tauri build` fails with "An Application
  Control policy has blocked this file". The two commands above are what it
  runs, minus the installer, and need no CLI.

`pnpm build` refuses an avatar page that could animate outside its paced
loop. `scripts/check-paced.mjs` reads our own modules `avatar.html` reaches
(`scripts/reach.mjs` follows the imports; packages are listed, not read) for
requestAnimationFrame, setAnimationLoop, the Web Animations API and motion
set from code. `scripts/check-no-css-animation.mjs` reads every stylesheet
the page reaches for @keyframes, animation, transition and smooth
scrolling, then, after `vite build`, the built page's stylesheets, packages'
included; a built page it refuses is removed, so the cargo build that
follows stops rather than embed it. `pnpm sync:divora` compares
`src/divora/tokens.css` with DivoraVoice's tokens at the commit its header
pins (it reads GitHub, so it isn't part of the build; `--sha <sha> --write`
takes another commit's tokens and pins it).

## Test

```powershell
pnpm test:run                                                        # the pages ("node" and "dom" projects)
cargo test --locked --features custom-protocol --lib --manifest-path src-tauri/Cargo.toml   # the shell
```

`FRITZ_SHOW_SETTINGS` set to anything shows the control window at start, for
a run nobody clicks; set to a section (`brain`, `hotkeys`, `appearance`,
`model`, …) it opens there, and set to `wizard`, at the first run. Without a
token it opens at the first run anyway. In a debug build,
`FRITZ_PRETEND_SAY` asks the brain its text once he first connects, as Ask
would (a release build never reads it).

## The placeholder model

`public/placeholder.vrm` is generated, CC0, and committed so a fresh checkout
always has a body: `python scripts/make_placeholder_vrm.py` rewrites it,
byte for byte. The owner's real model (plan 14 P6, a VRoid export) is never
committed; drop it at `%APPDATA%\com.divora.fritz\avatar\fritz.vrm` (or point
`avatar.vrmPath` at it) and it is used instead.

## Where things come from

[docs/LIFTED.md](docs/LIFTED.md) lists every file lifted from DivoraVoiceMod
(MIT, see [LICENSES/DivoraVoiceMod.txt](LICENSES/DivoraVoiceMod.txt)) or
LorraineAtelier, with its source and commit; each lifted file says so at the
top.
