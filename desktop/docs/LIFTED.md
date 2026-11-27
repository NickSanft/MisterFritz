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
| `src-tauri/src/overlay_windows.rs` | `src-tauri/src/overlay_windows.rs` | Charm serving and the recorder removed; `.focusable(false)` and Lorraine's browser arguments added (with Fritz's own autoplay flag, PR 10); parking, the cover watch's park and the saved spot; the status emitted from the main thread at the time it runs; handing the keyboard back rewritten, and its rules moved to `keyboard.rs`; effects run under `logs::guard`; each step mirrored into one atomic that the mouse hook's procedure reads without a lock (the live HWND into a second, at build, adopt and forget), and the hook's owner thread poked to decide again whether the hook should be in |
| `src-tauri/src/keyboard.rs` (part) | `src-tauri/src/overlay_windows.rs` (`Native`) | `foreground`, `try_foreground` and `can_take_foreground`, as methods of `Keyboard`; `the_shell_never_calls_set_focus` is its `the_overlay_never_calls_set_focus`, run over every module; who gets the keyboard back (`return_target`, `hand_back`) is Fritz's own |
| `src-tauri/src/non_rude.rs` | `divora-capture/src/native/window.rs` | `mark_non_rude` / `unmark_non_rude` only |
| `src-tauri/src/windows_ext.rs` (part) | `divora-capture/src/native/window.rs` | `foreground_window`, `try_foreground`, `can_take_foreground`, `make_layered_opaque`; the rest of the file is Fritz's own or follows Lorraine (below) |
| `src-tauri/src/tts_tokens.rs` | `divora-core/src/tts/tokens.rs` | Verbatim: phonemes to Kokoro's ids, the text cut into chunks, the vocab read from Kokoro's config |
| `src-tauri/src/tts_phonemize.rs` | `divora-core/src/tts/phonemize.rs` | The text goes to espeak-ng on its stdin as UTF-8 (`--stdin`), never as an argument, which Windows passes through the ANSI code page, with a newline after it, the byte `--stdin` takes for its line end and overwrites; its errors written out by hand rather than with thiserror; its real test reads a fetched folder (`FRITZ_TTS_TEST_DIR`) and checks text the code page can't carry |
| `src-tauri/src/kokoro_tts.rs` (part) | `divora-core/src/tts/kokoro.rs`; `divora-core/src/tts/mod.rs` | `StylePack` and `PRESET_VOICES` verbatim, and `infer`'s inputs and output, its tensors a shape and a vector each rather than ndarray's; the rest is Fritz's own: one Kokoro loaded and kept (DivoraVoice loads it for every line), a sentence said whole (DivoraVoice's `split_text`, which cuts at every stop, only for a text too long to say at once) and halved at a word when its phonemes pass Kokoro's 510, ONNX Runtime opened only by its full path from the data folder, four threads, none spinning, a voice he doesn't have spoken as George |
| `src/avatar/status.ts` | `src/overlay/status.ts` | Cut to the avatar's status and commands |
| `src/avatar/wire.fixture.json` | `src/overlay/wire.fixture.json` | Cut to what the avatar keeps; `parked`, the commands and Fritz's events added |
| `src/divora/tokens.css` | `src/styles.css`, lines 17-265 | Verbatim: the tokens, moods, accents, light theme and motion levels; `scripts/sync-divora-tokens.mjs` (`pnpm sync:divora`) checks it against the pinned commit |
| `src/divora/components.css` | `src/styles.css`, lines 375-381, 496-672, 1079-1121, 1319-1329, 267-281, 1161-1169, 367-369, 980-1011, 776-791 and 447-457; `src/screens/SettingsScreen.tsx`, lines 3298-3306 | `.eyebrow` (from Tailwind's `@layer utilities`, which comes after `components` in the cascade), the buttons, the toggle, the segmented control, and the hotkey capture with its keycaps (from `@layer components`), taken out of their layers, and the capture's shimmer keyframes (in no layer there either); the capture's hint in `--text-mid`, not `--text-lo` (below AA at 11 px on `--surface-2`), and its ring in the accent, not a fixed violet; the grain on the body, fixed to the window, above everything (z-index 100, not 60), with no vignette; the functional-motion switch verbatim; the titlebar's `LOCAL · NO ACCOUNT` mark (`.titlebar-local`) and `.mono`; the text field's inline style as a `.field` class; the control window only |
| `src/divora/components/Toggle.tsx`, `Segmented.tsx`, `blurOnPointer.ts` and the two tests, `Toggle.test.tsx` and `Segmented.test.tsx` | `src/components/` | Verbatim |
| `src/divora/components/HotkeyCapture.tsx`, `Kbd.tsx` and `HotkeyCapture.test.tsx` | `src/components/` | Verbatim |
| `src/divora/hotkeys.ts` and its test, `hotkeys.test.ts` | `src/data/hotkeys.ts`, lines 1-15 and 45-82; `hotkeys.test.ts`, lines 1-61 | The three converters (`keysToAccelerator`, `acceleratorToKeys`, `normalizeAccelerator`) and their tests, verbatim; DivoraVoice's actions and the reading of its saved bindings left out, and the test's import cut to match |
| `src/divora/sigils/Sigil.tsx` and its test, `Sigil.test.tsx` | `src/components/Sigil.tsx`, `Sigil.test.tsx` | `github` removed (GitHub's own mark); Fritz's own glyphs (`butler`, `desk`, `calendar`, `inbox`, `weather`, `listening`, `thinking`, `speaking`) added in the same hand, with a test of their own |
| `src/avatar/BubbleView.ts` (part) | `src/components/Sigil.tsx` (`muted`); `src/components/Meters.tsx` (`HMeter`) | The `muted` sigil's marks only, built as DOM in `mutedSigil` (his page has no Solid); `src/avatar/mutedSign.test.tsx` holds them to the lifted Sigil's. `HMeter`'s three elements, built as DOM, its level and cap set each paint from the shell's levels in decibels (`src/avatar/meter.ts`), the cap held for a second rather than given by the engine |
| `src/avatar/bubble.css` (part) | `src/styles.css`, lines 866-891 | The horizontal meter (`.hmeter`, `.hmeter-level`, `.hmeter-peak`), verbatim but taken out of its `@layer`; Fritz's own `.bubble-meter` gives it the `border-box` DivoraVoice's `* { box-sizing: border-box }` reset gives every element there |
| `src/test-setup.ts` | `src/test-setup.ts` | Verbatim; vitest's "dom" project follows DivoraVoice's config (jsdom, the Solid plugin), without hot reload |

Not lifted, but following DivoraVoice: `src/divora/theme.ts` maps `theme.motion` to a motion level as its `src/App.tsx` (lines 168-177) and `src/overlay.tsx` (lines 111-124) do; the rest of that file is Fritz's own.

Not lifted, but following DivoraVoice: the first run's layout (`src/control/Wizard.tsx`), a rail of the steps, the step, and a footer of *Skip setup*, *Back* and *Continue*, as its `src/components/Wizard.tsx` (lines 191-436) lays its own out; no code, and its steps are Settings' own rows.

## The fonts (SIL Open Font License 1.1)

DivoraVoice loads its faces from Fontsource packages; the body self-hosts the
same files, at the versions DivoraVoice's lockfile pins, each package's
sha512 checked against it. Only the latin subset is kept.

| Here | From | Licence |
|---|---|---|
| `src/divora/fonts/bricolage-grotesque-latin-wght-normal.woff2` | `@fontsource-variable/bricolage-grotesque` 5.2.10, `files/` | [../LICENSES/BricolageGrotesque-OFL.txt](../LICENSES/BricolageGrotesque-OFL.txt) |
| `src/divora/fonts/space-grotesk-latin-wght-normal.woff2` | `@fontsource-variable/space-grotesk` 5.2.10, `files/` | [../LICENSES/SpaceGrotesk-OFL.txt](../LICENSES/SpaceGrotesk-OFL.txt) |
| `src/divora/fonts/space-mono-latin-{400,700}-normal.woff2` | `@fontsource/space-mono` 5.2.9, `files/` | [../LICENSES/SpaceMono-OFL.txt](../LICENSES/SpaceMono-OFL.txt) |

`src/divora/fonts.css` holds those packages' latin `@font-face` rules, with
the family names tokens.css expects.

## LorraineAtelier (the owner's own app, private)

At **d62b7ec**. Plumbing only: no gift content, of its settings only
`WidgetPos` (below), and none of its history (these are fresh files, not
copies with their past).

| Here | There | Changed |
|---|---|---|
| `src-tauri/src/settings.rs` | `src-tauri/src/settings.rs` | The store: a `OnceLock<Mutex<…>>`, the path from `%APPDATA%` so it loads before any window exists, serde defaults, a corrupt file set aside. Of its fields, only `WidgetPos { x, y, scale, vertical }` and its defaults (0, 0, 1, false), as plan 14 § 4.2 asks for Phase 3's widgets; the rest of the schema is plan 14's. Fixed: a file that can't be read is never saved over; each corrupt file keeps its own name; saves are durable (`sync_all`), atomic with no direct-write fallback, serialised, each from its own temporary file, and tried again after a failure; a value that can't be used is put back to its default at load, alone |
| `src-tauri/src/logs.rs` | `src-tauri/src/lib.rs` (`log_startup`, `log_crash`, `guard`, the panic hook) | One capped writer for both logs, keeping the last as `.old`; readable UTC times; one line per call; the panic hook chains to the previous one, and writes nothing for the mouse hook's thread (whose procedure may not wait) |
| `src-tauri/src/power.rs` | `src-tauri/src/power.rs` | Display state and resume only: no session handling (the cover watch owns the lock screen); the event sent only on a change, and the state kept for a page that asks; the window procedure guarded; a change pokes the mouse hook |
| `src-tauri/src/hook.rs` | `src-tauri/src/hook.rs` | The plumbing only, as a fresh file: its own thread and message loop, `WH_MOUSE_LL` on every thread, the unhook as the thread ends, the stop by `WM_QUIT`, lParam read only for `HC_ACTION`, `CallNextHookEx` unless swallowing, moves never swallowed, the work handed to another thread over a channel. Changed: one thread installs and removes it, and the hook thread's queue is made before its id is known (both of Lorraine's stop races closed); the module handle passed; the hook thread at the highest priority; in only while Fritz can be clicked; its exposure test is Fritz's own (no window above him at the point, and the app in front not busy with the mouse), not `desktop_exposed_at`; no gift targets, hover, buddy or dock; injected input ignored; no lock, log or wait in the procedure, and its panics caught; a watch that puts it back when it goes quiet while the mouse moves (Windows may have dropped it). Fritz's own, not Lorraine's: a press decided as his takes its release with it wherever it lands (Lorraine swallows its action's press and lets the release pass) |
| `src-tauri/src/windows_ext.rs` (part) | `src-tauri/src/workerw.rs` | `set_dpi_awareness`; the fullscreen test, generalised to every monitor Fritz overlaps and made blind to framed windows merely maximised (Chromium's rule) |
| `src-tauri/src/overlay_windows.rs` (`BROWSER_ARGS`) | the WebView2 arguments its windows share | Byte-identical, in every window; `--autoplay-policy=no-user-gesture-required` appended (PR 10), Fritz's own, so his voice plays in a window nobody clicks |
| `src/avatar/PacedLoop.ts` | `src/wallpaper.ts`'s paced loop | Rewritten as a class with an injectable clock |
| `src-tauri/src/workerw.rs` | `src-tauri/src/workerw.rs` | Finding the desktop's wallpaper layer (Progman asked with `0x052C`, its answer never trusted; Progman's own WorkerW child, then the top-level WorkerW without the icons, then Progman itself), the answer kept between looks, and the monitors in physical pixels with their DPI. Changed: the choice is a pure function; Progman (the fallback) is kept for ten looks at most, so a WorkerW Explorer makes later is found; monitors carry Fritz's `Bounds`. None of its casting overlay, foreground or process-matching code |
| `src-tauri/src/host.rs` | `src-tauri/src/host.rs` | The per-monitor host window in the wallpaper layer, placed from the layer's client origin, that a webview's controller is moved into. Changed: Fritz's own class name; it paints the owner's mood's `--bg` before the page does (Lorraine's colour isn't here); it clips its children, sits below its siblings, tells its parent nothing as it comes and goes (`WS_EX_NOPARENTNOTIFY`: no message to Explorer, waited for, from the main thread), and can be put back over its monitor |
| `src-tauri/src/wallpaper.rs` | `src-tauri/src/lib.rs` (the slots, `reparent_controller`, `ensure_attached`, `update_pause`, the watchdog, `power_resume`) | The model: one hidden webview window per monitor (at most four), its controller moved into a host, a 600 ms watch that builds a slot anew when its host died (an Explorer restart), its monitor moved or changed scale, or the PC woke. Changed: the decision is one pure function; the watch is its own thread and only window work goes to the main thread (windows are asked for from the watch's thread, which waits there for each to be made, never from the main thread's own work); a slot is left alone while its homing is out, and built anew if it isn't heard from, or if its window was closed by anyone else (a close is refused); a host off its monitor (the desktop's top-left moved, or wry moved it with its hidden window) is put back, which Lorraine never did; an unplugged monitor's window is closed too; windows are built only for the monitors there are, while the setting is on, never declared in tauri.conf.json; the fullscreen pause is the page's own (nothing hidden, so nothing flashes), and the lock screen pauses it too. No gift, buddy, casting or hit-rect code |
| `scripts/check-no-input.mjs` | `scripts/check-no-input.mjs` | The gate: no click, pointer or mouse handler in the wallpaper's page. Changed: it reads every module the page loads (`reach.mjs`), comments blanked; double-click, context-menu, wheel, touch, drag and key handlers are refused too; a missing file is an error |

Not lifted, but following Lorraine: a modal window hidden at rest, the window
in use saved before it shows and handed back as it closes
(`src-tauri/src/prompt.rs` and `keyboard.rs`, Fritz's own code).
