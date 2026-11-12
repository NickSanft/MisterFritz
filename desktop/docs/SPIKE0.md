# Spike 0 — does a persistent WebGL avatar overlay behave?

Plan 14 gates all of Phase 1 on this ([plan 14 § Spike 0](../../plans/14-desktop-companion.md),
[build plan § 4.1](../../plans/14-desktop-companion-build.md)). Lorraine's own notes say a
persistent always-on-top WebView2 window steals the foreground; DivoraVoice's
stream overlay shows it needn't, if it never takes focus. What is new here is a
**continuously animating WebGL canvas**, in a window that is **up all day**.

**Pass → PR 2. Fail on the foreground row → stop: Fallback A (WorkerW layer) or B
(native `bevy_vrm1` sidecar) is the owner's call.**

> **Running this table again for PR 2 (its acceptance), in the real app:** the
> foreground logger is now off unless it is switched on (tray > Settings… >
> Diagnostics, or `diagnostics.foregroundLog` in
> `%APPDATA%\com.divora.fritz\settings.json`), from the next start, and it
> writes to `diagnostics\foreground.log`, not `spike\foreground.log`. The tray's
> *Open the spike logs* is now *Open Fritz's logs*. A fifth kind of line,
> **`SETTINGS`**, is the control window having the keyboard, which it should
> when you open it: not a `HIT`. New with it: Arrange (its button, or the
> tray's) and the tray's Lock, with Settings open and in use, hand the
> keyboard back to Settings, never to the app beneath it. Everything below is
> the spike as it was run.

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
| A fullscreen app on any monitor he touches, or the lock screen | he parks himself within a second, and comes back when it goes — not if you hid him yourself, and not while you are arranging him (he steps aside once you lock him). A maximised window doesn't count (with an auto-hiding taskbar it covers the monitor too). The lock screen is Windows' word, not a window's: from the lock until you sign back in |
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
| `GetForegroundWindow()` at 10 Hz never returns Fritz's window or a WebView2 child of it | zero `HIT` lines in 30 min, incl. app start, Explorer restart, sleep/resume | **0 `HIT`** in 16 min over two runs, through both starts, the lock screen, a sleep, an Explorer restart and a quit and restart. 4 `ARRANGE` lines over 3 Arranges, all yours (the 20:35:11 one is a click on him mid-Arrange), and none kept the keyboard afterwards (no `kept after Arrange`); 1 `TRAY`. After the second and third Arranges the session's checks found Claude, the app you'd been in, back in the foreground. Short of 30 min |
| Typing in another app while he animates | no dropped keystrokes, no caret loss | Not checked on its own. Notepad had the keyboard for half a minute while he animated; nothing was reported |
| Clicks anywhere on his window's rectangle | pass through to the app beneath | Pass by the automated check: at every look (both starts, after the sleep, after the Explorer restart) the window under his middle was the app beneath. Not clicked by hand |
| Transparent areas | no black/white halo; VRM edges premultiplied correctly | No fringe at 1:1 over a white window (the build) or the dark Claude window (the session). Not looked at 200 % |
| PresentMon on the WebView2 GPU process | ≤ 30 presents/s animating; ≤ 1/s parked | Not run: PresentMon isn't installed |
| Borderless-fullscreen game on the same monitor | he parks himself, game unaffected | Not run. The same watch parks him for the lock screen, and did so unreliably; fixed (below) |
| GPU busy, idle-animating (`spike0-gpu.ps1`, animating − parked) | agree a threshold (suggested < 3 % on the dGPU) | On the Intel iGPU (the one DWM uses); the RTX stayed asleep. His WebView2 GPU process: 4.3 % of its 3D engine animating (a 3 s sample in the second run, as in the build's check). The build's 5 s A/B: 5.6 % difference. Placeholder model; no threshold agreed |

Session: 2026-11-11, 20:33–20:50 (UTC−6), Windows 11 build 26300, Intel
graphics + RTX 5080 Laptop, one 3840×2400 panel at 200 % with an auto-hiding
taskbar, the placeholder model (there is no VRoid Studio export yet).

From `foreground.log`, `fritz.log`, the Windows System log and the checks
made during the session (local times):

- 20:33:09 — started; the foreground stayed with Claude.
- 20:35:05 — Arrange (Ctrl+Alt+A); moved to the middle of the screen.
  Mid-Arrange the keyboard went to Windows Search (20:35:09), then Claude
  (20:35:10); at 20:35:11 he took it again (a click on him). The Arrange was
  over before the lock screen at 20:35:30 (the watch parked him, which it
  doesn't mid-Arrange); which window got the keyboard then isn't in the logs.
- 20:35:30–20:36:05 — the lock screen: the cover watch parked him and brought
  him back five times (`by="LockScreenBackstopFrame (pid 7104)"`).
- 20:36:10–20:36:17 — Modern Standby (Kernel-Power 506/507). The heartbeat
  lost 8 s of polls; afterwards the same window and WebView2 GPU process, still
  drawn and animating.
- 20:37:04 — Explorer restarted from Task Manager. He stayed off the taskbar
  and out of Alt-Tab; his tray icon came back (in the overflow).
- 20:37:28 — Arrange again; locked, and Claude had the keyboard back by
  20:38:20.
- 20:41:25 — Quit from the tray (the one `TRAY` line).
- 20:42:31 — started again: back at his saved spot, without the foreground.
  20:42:38, Arrange; moved back to the bottom right, and Claude had the
  keyboard back by 20:43:13.
- 20:50 — stopped for the lock-screen fix.

`fritz.log` starts afresh each run, so the first run's cover lines above were
read before the restart. The raw logs stay on the owner's machine: they name,
with their titles, the windows that had the foreground just before each
`ARRANGE` and `TRAY` line (up to five each). They are not a full foreground
history.

**Owner's decision (2026-11-11): proceed on these results** — PR 1, then
PR 2, in the build plan's order. What the table marks not run or not checked,
and the full 30 minutes, are left to PR 2, whose acceptance is this table in
the real app.

### Found in the session

The lock screen didn't keep him parked. The watch tested the window at each
monitor's centre, and under the lock screen that was Explorer's
`LockScreenBackstopFrame` only some of the time, so he went and came back five
times in 35 s. Nothing showed (the lock screen is above every always-on-top
window), but each return restarted his 30 fps loop behind it. Now the watch
asks Windows first: while the session is locked (`WTSQuerySessionInformationW`,
`WTS_SESSIONSTATE_LOCK`) he is covered by "the lock screen", whatever window is
at the centre, until you sign back in. On start the watch logs what Windows
says (`cover watch started locked=Some(false)`).

Seen live at 21:11–21:12 on the fixed build: two locks (Winlogon's Lock and
Unlock notifications), the first ended after 5 s by Windows Hello face
sign-in, the second held for 16 s. Each gave exactly one
`parking him by="the lock screen"`, within 2.6 s of the lock, and one
`bringing him back`, within 0.8 s of the sign-in; afterwards he was drawn,
animating and click-through at his spot.

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
