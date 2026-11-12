# Mister Fritz — the desktop body

Fritz's body on the Windows desktop: a Tauri 2 app that will talk to the brain
over the `/companion` WebSocket ([plan 14](../plans/14-desktop-companion.md),
[build plan](../plans/14-desktop-companion-build.md)). Windows only.

Today it is one persistent, transparent, click-through, never-focused avatar
window with a WebGL VRM in it, a tray and two hotkeys: PR 2 of the build plan,
in progress. Spike 0 proved the window ([docs/SPIKE0.md](docs/SPIKE0.md)), and
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
(Ctrl+Alt+H), read at start. Until the control window lands, change a setting
by editing `settings.json` with Fritz quit.

## Build

Node ≥ 20.19, pnpm, Rust stable (MSRV 1.89, MSVC toolchain) and the Visual
Studio C++ Build Tools. From `desktop/`:

```powershell
pnpm install
pnpm build                                     # checks, type-checks and bundles the page into dist/
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
- **The page is embedded in the exe at compile time**: after any change under
  `src/` or `avatar.html`, run `pnpm build` *and* the cargo build again.
- **Smart App Control** (Windows 11) blocks the Tauri CLI's unsigned native
  module on some machines, so `pnpm tauri build` fails with "An Application
  Control policy has blocked this file". The two commands above are what it
  runs, minus the installer, and need no CLI.

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
