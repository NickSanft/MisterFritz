# Spike 0 — does a persistent WebGL avatar overlay behave?

Plan 14 gates all of Phase 1 on this ([plan 14 § Spike 0](../../plans/14-desktop-companion.md),
[build plan § 4.1](../../plans/14-desktop-companion-build.md)). Lorraine's own notes say a
persistent always-on-top WebView2 window steals the foreground; DivoraVoice's
stream overlay shows it needn't, if it never takes focus. What is new here is a
**continuously animating WebGL canvas**, in a window that is **up all day**.

**Pass → PR 2. Fail on the foreground row → stop: Fallback A (WorkerW layer) or B
(native `bevy_vrm1` sidecar) is the owner's call.**

## What the spike app is

One window, built at runtime by DivoraVoice's lifted overlay shell:
transparent, undecorated, shadowless, always on top, off the taskbar and out of
Alt-Tab, hidden until its page is ready, never focused, **not focusable**
(`WS_EX_NOACTIVATE`), click-through before it is first shown, marked
`NonRudeHWND`, with Lorraine's WebView2 arguments. The page is three.js +
three-vrm drawing Fritz at 30 fps from a `setTimeout` loop (no
`requestAnimationFrame`, no CSS animation), alpha-cleared and premultiplied.

| Control | Does |
|---|---|
| Tray → Hide / Show Fritz, or **Ctrl+Alt+H** | parks him at x = −10000 (a move, never a hide) and stops the loop; Show moves him back, and reopens his window if it was closed. A Hide is remembered across restarts |
| Tray → Arrange Fritz…, or **Ctrl+Alt+A** | (only while he is on screen) takes the foreground, lifts click-through after the page's first beat: drag to move, the corner grip to resize, **Esc** (or the tray's Lock, or Ctrl+Alt+A) to lock; the keyboard goes back to the app you were in (or, if that one has gone, the app in use — never an always-on-top window if there's another — or the desktop) |
| A fullscreen app on any monitor he touches | he parks himself within a second, and comes back when it goes — not if you hid him yourself, and not while you are arranging him (he steps aside once you lock him). A maximised window doesn't count (with an auto-hiding taskbar it covers the monitor too); the lock screen does |
| Tray → Open the spike logs | `%APPDATA%\com.divora.fritz\` |
| A saved spot on a monitor that's gone | he comes back onto the nearest one |

`spike\foreground.log` has four kinds of line. **`HIT`** is a failure of the
first row: Fritz's process had the foreground when nobody asked it to —
including `HIT … kept after Arrange`, when Arrange ended but he kept the
keyboard. **`ARRANGE`** is Arrange taking the keyboard on purpose, **`TRAY`**
the tray menu doing the same (Windows requires it before showing a menu), and
**`alive`** the minute heartbeat with running counts. Every `HIT`, `ARRANGE`
and `TRAY` line is followed by the windows that had the foreground just before.

The model is `%APPDATA%\com.divora.fritz\avatar\fritz.vrm` if it exists (plan 14 P6,
the VRoid export), else the committed placeholder. **Run the GPU and PresentMon
rows with the real VRoid model**: the placeholder is lighter than an export.

## The session

Build it (see [../README.md](../README.md)), start
`src-tauri\target\release\mister-fritz-desktop.exe`, and work normally for a
session — at least 30 minutes, ideally a working day. Along the way:

1. **Foreground** — nothing to do but work, using the tray and the hotkeys as
   you like; `spike\foreground.log` records any moment the foreground is
   Fritz's. During the session, also: restart Explorer (Task Manager → Windows
   Explorer → Restart), sleep and resume, and quit and restart Fritz — and
   note the times, since the log can't tell those apart from an ordinary
   quiet stretch.
2. **Typing** — type a few paragraphs in an editor while he animates, some of
   it with the caret right behind him.
3. **Clicks** — click, double-click and drag on things beneath his window,
   both on him and on the empty corners of his rectangle.
4. **Edges** — screenshot him at 200 % zoom over a white page and over a dark
   one; look for a black or white fringe.
5. **Presents** — with Intel PresentMon, capture 60 s animating and 60 s
   hidden. The presents come from WebView2's GPU process, not from
   `mister-fritz-desktop.exe`: it is the `msedgewebview2.exe` whose command line
   has `--type=gpu-process` and `--webview-exe-name=mister-fritz-desktop.exe`.
6. **Fullscreen** — a borderless-fullscreen game (or a browser in F11) on his
   monitor: he should step aside, and the game should notice nothing.
7. **GPU** — on an otherwise idle desktop, run
   `powershell -ExecutionPolicy Bypass -File desktop\scripts\spike0-gpu.ps1 -Seconds 60`.
   It samples the GPU for a minute with him animating, parks him (Ctrl+Alt+H),
   samples a minute more and brings him back, then prints each adapter's total
   3D use in both states. The **difference on his adapter** is the number for
   the row: it counts the desktop compositor's work of drawing him, which is
   billed to `dwm.exe`, not to Fritz. Note which GPU it is on (the integrated
   one is the aim; the discrete one should stay asleep).

Also note, though the table doesn't ask: whether Alt-Tab or the taskbar ever
shows him, and whether Arrange hands the keyboard back.

## Results

| Check | Pass | Result |
|---|---|---|
| `GetForegroundWindow()` at 10 Hz never returns Fritz's window or a WebView2 child of it | zero `HIT` lines in 30 min, incl. app start, Explorer restart, sleep/resume | |
| Typing in another app while he animates | no dropped keystrokes, no caret loss | |
| Clicks anywhere on his window's rectangle | pass through to the app beneath | |
| Transparent areas | no black/white halo; VRM edges premultiplied correctly | |
| PresentMon on the WebView2 GPU process | ≤ 30 presents/s animating; ≤ 1/s parked | |
| Borderless-fullscreen game on the same monitor | he parks himself, game unaffected | |
| GPU busy, idle-animating (`spike0-gpu.ps1`, animating − parked) | agree a threshold (suggested < 3 % on the dGPU) | |

Session: date, length, Windows build, GPU(s), monitor(s) and scaling, model used.

Attach: `foreground.log`, the PresentMon summaries, the edge screenshots, the
`spike0-gpu.ps1` output.

## Measured while building it (not the session)

A short automated check on the build machine (Windows 11 build 26300, Intel
graphics + RTX 5080 Laptop, one 3840×2400 panel at 200 %, auto-hiding
taskbar), with the placeholder model, 2026-11-11:

- Started without taking the foreground; the window is topmost, layered,
  click-through, `WS_EX_NOACTIVATE` and a tool window (no taskbar button, not
  in Alt-Tab), 600×920 px at the bottom right.
- The window under the middle of him is the app beneath: clicks pass through.
- Ctrl+Alt+H moved him to x = −10000 and back to the same spot; Ctrl+Alt+A
  took the foreground and lifted click-through and `NOACTIVATE`; Ctrl+Alt+A
  again locked him and handed the foreground back to the app that had it.
- `foreground.log`: 0 `HIT`, 1 `ARRANGE` (the deliberate one).
- Rendered on the **Intel** GPU (the one DWM uses); the RTX stayed asleep
  (0 % throughout). `spike0-gpu.ps1 -Seconds 5`: the Intel adapter's total 3D
  use was 13.9 % animating and 8.3 % parked, a **difference of 5.6 %** —
  Fritz's WebView2 process 4.2 % and DWM 1.8 % of it. Five-second samples on
  a desktop that wasn't idle, with the light placeholder: a first reading
  only. The session's row wants 60 s samples and the real model, and the
  threshold (the plan suggests < 3 % on the *discrete* GPU, which he doesn't
  touch at all) is yours to set for the integrated one.
- Edges over a white window: no fringe at 1:1.

Found and fixed on the way, so the session doesn't trip on them: tao rewrites
the window's extended style on every change and dropped it back into Alt-Tab
(now pinned by a subclass), and a maximised window under an auto-hiding
taskbar was taken for a fullscreen app.
