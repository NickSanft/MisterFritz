# What is lifted, and from where

Plan 14 builds the body from two working apps rather than from scratch. This
is every file here that came from one of them, what it came from, and what
changed on the way. Each file also says so at its top.

## DivoraVoiceMod (MIT)

`github.com/NickSanft/DivoraVoiceMod` at **62ed6a5**; the licence is
[../LICENSES/DivoraVoiceMod.txt](../LICENSES/DivoraVoiceMod.txt). Upstream has
not touched these files since.

| Here | There | Changed |
|---|---|---|
| `src-tauri/src/overlay_machine.rs` | `src-tauri/src/overlay_machine.rs` | The recorder's inputs, state and tests removed; names are Fritz's; `Input::Park` / `Effect::Park` added (a move off screen, never a hide), with `parked` on the core and in the status; Arrange refused while parked |
| `src-tauri/src/overlay_windows.rs` | `src-tauri/src/overlay_windows.rs` | Charm serving and the recorder removed; `.focusable(false)` and Lorraine's browser arguments added; parking, the cover watch's park and the saved spot; the status emitted from the main thread at the time it runs; handing the keyboard back rewritten; effects run under `logs::guard` |
| `src-tauri/src/non_rude.rs` | `divora-capture/src/native/window.rs` | `mark_non_rude` / `unmark_non_rude` only |
| `src-tauri/src/windows_ext.rs` (part) | `divora-capture/src/native/window.rs` | `foreground_window`, `try_foreground`, `can_take_foreground`, `make_layered_opaque`; the rest of the file is Fritz's own or follows Lorraine (below) |
| `src/avatar/status.ts` | `src/overlay/status.ts` | Cut to the avatar's status and commands |
| `src/avatar/wire.fixture.json` | `src/overlay/wire.fixture.json` | Cut to what the avatar keeps; `parked`, the commands and Fritz's events added |

## LorraineAtelier (the owner's own app, private)

At **d62b7ec**. Plumbing only: no gift content, of its settings only
`WidgetPos` (below), and none of its history (these are fresh files, not
copies with their past).

| Here | There | Changed |
|---|---|---|
| `src-tauri/src/settings.rs` | `src-tauri/src/settings.rs` | The store: a `OnceLock<Mutex<…>>`, the path from `%APPDATA%` so it loads before any window exists, serde defaults, a corrupt file set aside. Of its fields, only `WidgetPos { x, y, scale, vertical }` and its defaults (0, 0, 1, false), as plan 14 § 4.2 asks for Phase 3's widgets; the rest of the schema is plan 14's. Fixed: a file that can't be read is never saved over; each corrupt file keeps its own name; saves are durable (`sync_all`), atomic with no direct-write fallback, serialised, each from its own temporary file, and tried again after a failure; a value that can't be used is put back to its default at load, alone |
| `src-tauri/src/logs.rs` | `src-tauri/src/lib.rs` (`log_startup`, `log_crash`, `guard`, the panic hook) | One capped writer for both logs, keeping the last as `.old`; readable UTC times; one line per call; the panic hook chains to the previous one |
| `src-tauri/src/power.rs` | `src-tauri/src/power.rs` | Display state and resume only: no session handling (the cover watch owns the lock screen); the event sent only on a change, and the state kept for a page that asks; the window procedure guarded |
| `src-tauri/src/windows_ext.rs` (part) | `src-tauri/src/workerw.rs` | `set_dpi_awareness`; the fullscreen test, generalised to every monitor Fritz overlaps and made blind to maximised windows |
| `src-tauri/src/overlay_windows.rs` (`BROWSER_ARGS`) | the WebView2 arguments its windows share | Byte-identical, in every window |
| `src/avatar/PacedLoop.ts` | `src/wallpaper.ts`'s paced loop | Rewritten as a class with an injectable clock |
