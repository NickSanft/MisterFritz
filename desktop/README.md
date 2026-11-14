# Mister Fritz — the desktop body

Fritz's body on the Windows desktop: a Tauri 2 app that will talk to the brain
over the `/companion` WebSocket ([plan 14](../plans/14-desktop-companion.md),
[build plan](../plans/14-desktop-companion-build.md)). Windows only.

Today it is one persistent, transparent, click-through, never-focused avatar
window with a WebGL VRM in it, a control window (tray > Settings…), a tray and
two hotkeys: PR 2 of the build plan. Spike 0 proved the window ([docs/SPIKE0.md](docs/SPIKE0.md)), and
its table is PR 2's acceptance.

## What it keeps, and where

Everything is in `%APPDATA%\com.divora.fritz\` (the tray's *Open Fritz's logs*
opens it):

| File | What |
|---|---|
| `settings.json` | The settings ([`settings.rs`](src-tauri/src/settings.rs)), saved as they change. A file that doesn't parse is moved to `settings.corrupt-<time>.json` and the defaults used; one that can't be read is left alone and not saved over that run |
| `startup.log`, `crash.log` | A line per start, stop and resume, and every panic. Kept across runs (the previous one as `.old` past 64 KiB): read these after a start that died |
| `logs\fritz.log` | This run's detailed log, started afresh each run |
| `diagnostics\foreground.log` | Spike 0's foreground logger, **off** unless `diagnostics.foregroundLog` is `true` (read at start): it writes other apps' window titles |
| `avatar\fritz.vrm` | The owner's model, if there is one (or `avatar.vrmPath`) |

The hotkeys are `hotkeys.arrange` (Ctrl+Alt+A) and `hotkeys.park`
(Ctrl+Alt+H), read at start. His eyes follow the cursor while
`avatar.lookatCursor` is `true` (the default): on a VRoid export the eyes
turn, and on any model his head turns part of the way, which is what shows
on the placeholder.

Both windows wear `settings.theme`: the mood (`ink`, `dusk`, `midnight`),
the accent (`brand`, `abyssal`, `ember`), dark or light, and the motion
(0 to 1: functional, ambient, rich). He takes a tint from it, his outlines
leaning toward the mood and a rim light in the accent. With no motion, or
while Windows asks for less (its *Animation effects* switch) and
`theme.motionUser` isn't set, he only blinks and breathes: no sway, and his
eyes stay ahead.

The tray's *Settings…* opens the control window: show or hide him, arrange
him, his eyes following the cursor, which model he wears (with its folder,
and loading it again after you drop in a new export), his logs, and the
foreground log. Closing it only hides it. The rest (the hotkeys,
`avatar.vrmPath`) is still set by editing `settings.json` with Fritz quit,
until the sections for them land.

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
pnpm test:run                                                        # the page
cargo test --locked --features custom-protocol --lib --manifest-path src-tauri/Cargo.toml   # the shell
```

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
