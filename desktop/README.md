# Mister Fritz — the desktop body

Fritz's body on the Windows desktop: a Tauri 2 app that will talk to the brain
over the `/companion` WebSocket ([plan 14](../plans/14-desktop-companion.md),
[build plan](../plans/14-desktop-companion-build.md)). Windows only.

**Today this is Spike 0**: one persistent, transparent, click-through,
never-focused avatar window with a WebGL VRM in it, instrumented for the
working session that decides whether the rest of Phase 1 is built this way.
How to run that session, and where its results go: [docs/SPIKE0.md](docs/SPIKE0.md).

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
committed; drop it at `%APPDATA%\com.divora.fritz\avatar\fritz.vrm` and it is
used instead.

## Where things come from

`src-tauri/src/overlay_machine.rs`, `overlay_windows.rs`, `non_rude.rs`, part
of `windows_ext.rs`, `src/avatar/status.ts` and (cut down)
`src/avatar/wire.fixture.json` are lifted from DivoraVoiceMod (MIT, see
[LICENSES/DivoraVoiceMod.txt](LICENSES/DivoraVoiceMod.txt)); each says so at
the top. The WebView2 arguments, the paced loop and the fullscreen test
follow LorraineAtelier.
