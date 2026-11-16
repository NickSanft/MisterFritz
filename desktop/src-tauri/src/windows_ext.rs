//! Win32 for the avatar window: the foreground without injected input, Alt-Tab,
//! the monitor Fritz stands on, whether a fullscreen app or the lock screen
//! covers it, and what is under the mouse (for the mouse hook's procedure:
//! plain reads, no message sent, no log).
//!
//! Windows are passed as integers (`HWND.0 as isize`), so callers need
//! neither `unsafe` nor the `windows` crate. 0 is no window, and every call
//! is a no-op, `false` or `None` for it.
//!
//! Lifted in part from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora
//! Contributors), divora-capture/src/native/window.rs at 62ed6a5:
//! `foreground_window`, `try_foreground`, `can_take_foreground` and
//! `make_layered_opaque`. See desktop/LICENSES/DivoraVoiceMod.txt.
//! `set_dpi_awareness` and the cover test follow Lorraine's workerw.rs.

use std::ffi::c_void;

use windows::core::{BOOL, PWSTR};
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_CLOAKED};
use windows::Win32::Graphics::Gdi::{
    EnumDisplayMonitors, GetMonitorInfoW, MonitorFromPoint, HDC, HMONITOR, MONITORINFO,
    MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY,
};
use windows::Win32::System::RemoteDesktop::{
    WTSFreeMemory, WTSQuerySessionInformationW, WTSSessionInfoEx, WTSINFOEXW,
    WTS_CURRENT_SERVER_HANDLE, WTS_CURRENT_SESSION, WTS_SESSIONSTATE_LOCK, WTS_SESSIONSTATE_UNLOCK,
};
use windows::Win32::UI::HiDpi::{
    GetDpiForMonitor, GetDpiForWindow, GetSystemMetricsForDpi, SetProcessDpiAwarenessContext,
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, MDT_EFFECTIVE_DPI,
};
use windows::Win32::UI::Input::KeyboardAndMouse::GetDoubleClickTime;
use windows::Win32::UI::Shell::{
    DefSubclassProc, RemoveWindowSubclass, SHQueryUserNotificationState, SetWindowSubclass,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumThreadWindows, GetMenuItemID, PostMessageW, SendMessageW, TrackPopupMenu, HMENU, TPM_LEFTALIGN, TPM_RETURNCMD,
    TPM_RIGHTBUTTON, TPM_TOPALIGN, WM_COMMAND, WM_NULL,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetClassNameW, GetCursorPos, GetForegroundWindow, GetGUIThreadInfo, GetShellWindow,
    GetTopWindow, GetWindow, GetWindowLongPtrW, GetWindowRect, GetWindowThreadProcessId, IsIconic,
    IsWindow, IsWindowVisible, IsZoomed, SetForegroundWindow, SetLayeredWindowAttributes,
    SetWindowLongPtrW, SetWindowPos, WindowFromPoint, GA_ROOT, GA_ROOTOWNER, GUITHREADINFO,
    GUI_INMENUMODE, GUI_INMOVESIZE, GUI_POPUPMENUMODE, GWL_EXSTYLE, GW_HWNDNEXT, GW_HWNDPREV,
    GW_OWNER, LWA_ALPHA, SM_CXDOUBLECLK, SM_CYDOUBLECLK, STYLESTRUCT, SWP_FRAMECHANGED,
    SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOOWNERZORDER, SWP_NOSIZE, SWP_NOZORDER, WA_INACTIVE,
    WM_ACTIVATE, WM_NCDESTROY, WM_STYLECHANGING, GWL_STYLE, WS_DLGFRAME, WS_EX_APPWINDOW,
    WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_THICKFRAME, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    WS_EX_TRANSPARENT,
};

/// A screen rectangle in physical pixels: left, top, right, bottom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Bounds {
    pub fn width(self) -> i32 {
        self.right - self.left
    }
    pub fn height(self) -> i32 {
        self.bottom - self.top
    }
    fn of(r: RECT) -> Self {
        Self {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        }
    }
}

pub(crate) fn hwnd(h: isize) -> HWND {
    HWND(std::ptr::without_provenance_mut::<c_void>(
        h.cast_unsigned(),
    ))
}

/// Per-monitor v2 DPI awareness, before any window exists. Harmless when the
/// windowing layer (or the manifest) already set it: the call just fails.
pub fn set_dpi_awareness() {
    // SAFETY: a process-wide setting with a predefined context.
    let _ = unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
}

/// The window that has the foreground, as an integer; 0 when none.
pub fn foreground_window() -> isize {
    // SAFETY: no arguments.
    unsafe { GetForegroundWindow() }.0 as isize
}

/// `SetForegroundWindow`, and nothing else: unlike tao's `set_focus` it never
/// injects input when Windows refuses. False for 0 or a refusal.
pub fn try_foreground(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    // SAFETY: a window handle; a stale one only makes the call fail.
    unsafe { SetForegroundWindow(hwnd(window)) }.as_bool()
}

/// Whether `window` may take the foreground back: it exists, is visible and
/// isn't minimised.
pub fn can_take_foreground(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    let h = hwnd(window);
    // SAFETY: plain queries on a window handle; a stale one answers false.
    unsafe { IsWindow(Some(h)).as_bool() && IsWindowVisible(h).as_bool() && !IsIconic(h).as_bool() }
}

/// R1 rung 1 only: `SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA)`,
/// so a layered window that nobody gave attributes still draws.
pub fn make_layered_opaque(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    // SAFETY: a window handle; a stale or unlayered one only makes it fail.
    unsafe { SetLayeredWindowAttributes(hwnd(window), COLORREF(0), 255, LWA_ALPHA) }.is_ok()
}

/// An extended style with `WS_EX_TOOLWINDOW` on and `WS_EX_APPWINDOW` off:
/// what Alt-Tab and the taskbar leave out.
pub fn toolwindow_style(ex: u32) -> u32 {
    (ex | WS_EX_TOOLWINDOW.0) & !WS_EX_APPWINDOW.0
}

/// Set `WS_EX_TOOLWINDOW` (and clear `WS_EX_APPWINDOW`) now. Writes only
/// when the style differs. Main thread. [`pin_toolwindow`] keeps it set.
pub fn apply_toolwindow(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    let h = hwnd(window);
    // SAFETY: our own window's style, read and written on its thread; the
    // reframe neither moves, sizes, reorders nor activates it.
    unsafe {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ex = GetWindowLongPtrW(h, GWL_EXSTYLE) as u32;
        let want = toolwindow_style(ex);
        if want == ex {
            return true;
        }
        SetWindowLongPtrW(h, GWL_EXSTYLE, want as isize);
        SetWindowPos(
            h,
            None,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED
                | SWP_NOACTIVATE
                | SWP_NOMOVE
                | SWP_NOSIZE
                | SWP_NOZORDER
                | SWP_NOOWNERZORDER,
        )
        .is_ok()
    }
}

/// The subclass id: "FRIT".
pub(crate) const PIN_ID: usize = 0x4652_4954;

/// Keeps every rewrite of the extended style out of Alt-Tab, and takes
/// itself off as the window goes.
unsafe extern "system" fn pin_proc(
    h: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
    _id: usize,
    _data: usize,
) -> LRESULT {
    #[allow(clippy::cast_possible_truncation)]
    let index = wp.0 as i32;
    if msg == WM_STYLECHANGING && index == GWL_EXSTYLE.0 && lp.0 != 0 {
        // SAFETY: for WM_STYLECHANGING, lParam points at the STYLESTRUCT the
        // change is about to apply; Windows reads styleNew back after we return.
        let s = unsafe { &mut *(lp.0 as *mut STYLESTRUCT) };
        s.styleNew = toolwindow_style(s.styleNew);
    } else if msg == WM_NCDESTROY {
        // SAFETY: our own subclass on the window it was set on.
        let _ = unsafe { RemoveWindowSubclass(h, Some(pin_proc), PIN_ID) };
    }
    // SAFETY: passes the message on down the subclass chain unchanged.
    unsafe { DefSubclassProc(h, msg, wp, lp) }
}

/// Keep the window out of Alt-Tab for good. tao rewrites the whole extended
/// style from its own flags on every change — click-through, focusable, even
/// showing the window — and puts `WS_EX_APPWINDOW` back as it does, so
/// setting the bit once is not enough. A subclass corrects each rewrite as it
/// happens (`WM_STYLECHANGING`), then the style is applied once. Must run on
/// the window's own thread (the main thread).
pub fn pin_toolwindow(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    // SAFETY: a window of this thread and a static procedure; the procedure
    // removes itself at WM_NCDESTROY.
    let pinned = unsafe { SetWindowSubclass(hwnd(window), Some(pin_proc), PIN_ID, 0) }.as_bool();
    pinned && apply_toolwindow(window)
}

/// The activation watch's subclass id: "FRDA".
pub(crate) const ACTIVATION_ID: usize = 0x4652_4441;

/// `WM_ACTIVATE`'s wParam as active or not: its low word `WA_ACTIVE` or
/// `WA_CLICKACTIVE` (a click activated it) is active, `WA_INACTIVE` is not;
/// the high word (minimised) is no part of it. Pure.
pub(crate) fn activation_state(wp: usize) -> bool {
    #[allow(clippy::cast_possible_truncation)]
    let state = (wp & 0xFFFF) as u32;
    state != WA_INACTIVE
}

/// Calls the watcher stored with the subclass each time the window becomes
/// the active one or stops being it, and takes itself off as the window
/// goes.
unsafe extern "system" fn activate_proc(
    h: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
    _id: usize,
    data: usize,
) -> LRESULT {
    if msg == WM_ACTIVATE && data != 0 {
        // SAFETY: `data` is the `fn(bool)` that `watch_activation` stored, as
        // an integer the size of a pointer; nothing else sets this id.
        let on_change = unsafe { std::mem::transmute::<usize, fn(bool)>(data) };
        on_change(activation_state(wp.0));
    } else if msg == WM_NCDESTROY {
        // SAFETY: our own subclass on the window it was set on.
        let _ = unsafe { RemoveWindowSubclass(h, Some(activate_proc), ACTIVATION_ID) };
    }
    // SAFETY: passes the message on down the subclass chain unchanged.
    unsafe { DefSubclassProc(h, msg, wp, lp) }
}

/// Call `on_change` each time `window` becomes the active window (true:
/// `WA_ACTIVE`, or `WA_CLICKACTIVE` for a click into it) or stops being it
/// (false), from inside its window procedure, so it must only store and
/// post, never act there. Windows says so whatever the webview inside was
/// told, which is why a window can't rely on the webview's own focus alone.
/// Must run on the window's own thread (the main thread).
pub fn watch_activation(window: isize, on_change: fn(bool)) -> bool {
    if window == 0 {
        return false;
    }
    // SAFETY: a window of this thread and a static procedure, which removes
    // itself at WM_NCDESTROY; the data is a plain function pointer.
    unsafe { SetWindowSubclass(hwnd(window), Some(activate_proc), ACTIVATION_ID, on_change as usize) }.as_bool()
}

/// The window at the top of `window`'s owner chain (itself, with no owner);
/// 0 for 0 or a window that is gone.
pub fn root_owner(window: isize) -> isize {
    if window == 0 {
        return 0;
    }
    // SAFETY: a plain query; a stale handle answers null.
    unsafe { GetAncestor(hwnd(window), GA_ROOTOWNER) }.0 as isize
}

/// Whether `window` is visible (WS_VISIBLE): false for 0 or a hidden one.
pub fn is_visible(window: isize) -> bool {
    // SAFETY: any handle; a stale one is just not visible.
    window != 0 && unsafe { IsWindowVisible(hwnd(window)) }.as_bool()
}

/// Where the cursor is on screen, physical pixels (the process is
/// per-monitor DPI aware, so no monitor's scale bends it).
pub fn cursor_pos() -> Option<(i32, i32)> {
    let mut p = POINT::default();
    // SAFETY: a valid out-pointer.
    unsafe { GetCursorPos(&raw mut p) }.ok().map(|()| (p.x, p.y))
}

/// The window's rectangle on screen, physical pixels.
pub fn window_bounds(window: isize) -> Option<Bounds> {
    if window == 0 {
        return None;
    }
    let mut r = RECT::default();
    // SAFETY: a window handle and a valid out-pointer.
    unsafe { GetWindowRect(hwnd(window), &raw mut r) }
        .ok()
        .map(|()| Bounds::of(r))
}

fn monitor_info(mon: HMONITOR) -> Option<MONITORINFO> {
    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: a monitor handle and a sized out-struct.
    unsafe { GetMonitorInfoW(mon, &raw mut info) }
        .as_bool()
        .then_some(info)
}

/// A monitor's work area and its effective DPI (at least 48).
fn work_and_dpi(mon: HMONITOR) -> Option<(Bounds, u32)> {
    let info = monitor_info(mon)?;
    let (mut dx, mut dy) = (96u32, 96u32);
    // SAFETY: a monitor handle and two out-pointers.
    let _ = unsafe { GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &raw mut dx, &raw mut dy) };
    Some((Bounds::of(info.rcWork), dx.max(48)))
}

/// The primary monitor's work area (the desktop less the taskbar) and its
/// effective DPI, for Fritz's first placement.
pub fn primary_work_area() -> Option<(Bounds, u32)> {
    // SAFETY: a point and a flag; the primary monitor always exists.
    work_and_dpi(unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) })
}

/// The work area and DPI of the monitor at (`x`, `y`), or of the nearest
/// one when no monitor has that point.
pub fn work_area_at(x: i32, y: i32) -> Option<(Bounds, u32)> {
    // SAFETY: a point and a flag; NEAREST always answers a monitor.
    work_and_dpi(unsafe { MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTONEAREST) })
}

/// Shell surfaces that never count as covering a monitor.
pub fn is_shell_class(class: &str) -> bool {
    matches!(
        class,
        "Progman"
            | "WorkerW"
            | "SHELLDLL_DefView"
            | "Shell_TrayWnd"
            | "Shell_SecondaryTrayWnd"
            | "Windows.UI.Core.CoreWindow"
    )
}

/// Whether `window` covers all of `monitor`. A maximised window leaves the
/// taskbar showing, so it doesn't.
pub fn covers(window: Bounds, monitor: Bounds) -> bool {
    window.left <= monitor.left
        && window.top <= monitor.top
        && window.right >= monitor.right
        && window.bottom >= monitor.bottom
}

/// Whether `window` takes `monitor` over the way a fullscreen app does: it
/// covers all of it and is not a framed window merely maximised
/// (`framed_maximized`). Lorraine's test (covers the monitor) assumed a
/// maximised window leaves the taskbar showing; with an auto-hiding taskbar
/// the work area is the whole monitor, so a maximised editor would send
/// Fritz away all day. A borderless window maximised to fill its monitor is
/// a fullscreen app all the same (a player, a kiosk, a presentation).
pub fn fullscreen_on(window: Bounds, monitor: Bounds, framed_maximized: bool) -> bool {
    !framed_maximized && covers(window, monitor)
}

/// Whether a window style has a frame (a sizing border or a caption), as an
/// editor or a browser has and a borderless fullscreen app hasn't
/// (Chromium's test for a fullscreen window).
pub fn framed_style(style: u32) -> bool {
    style & (WS_THICKFRAME.0 | WS_DLGFRAME.0) != 0
}

/// Whether `h` is maximised with a frame: the one kind of window covering a
/// monitor that isn't fullscreen.
///
/// # Safety
/// `h` may be stale, which only makes the queries fail.
unsafe fn framed_maximized(h: HWND) -> bool {
    // SAFETY: plain queries on a window handle.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    unsafe {
        IsZoomed(h).as_bool() && framed_style(GetWindowLongPtrW(h, GWL_STYLE) as u32)
    }
}

/// Whether `window` takes over any of `monitors` the way a fullscreen app
/// does ([`fullscreen_on`]).
pub fn fullscreen_among(window: Bounds, framed_maximized: bool, monitors: &[Bounds]) -> bool {
    monitors.iter().any(|&m| fullscreen_on(window, m, framed_maximized))
}

/// The foreground window, if it is another app's fullscreen window on any
/// monitor (a game, a video, a presentation), named for the log as
/// `"{class} (pid {pid})"`. Never ours, and never the shell's.
pub fn fullscreen_foreground() -> Option<String> {
    // SAFETY: plain queries; every handle below may be stale, which only
    // makes the calls fail.
    unsafe {
        let fg = GetForegroundWindow();
        if fg.0.is_null() {
            return None;
        }
        let root = GetAncestor(fg, GA_ROOT);
        let top = if root.0.is_null() { fg } else { root };
        let mut pid = 0u32;
        GetWindowThreadProcessId(top, Some(&raw mut pid));
        if pid == 0 || pid == std::process::id() {
            return None;
        }
        let class = class_name(top);
        if is_shell_class(&class) {
            return None;
        }
        let mut w = RECT::default();
        GetWindowRect(top, &raw mut w).ok()?;
        let all: Vec<Bounds> = monitors().into_iter().map(|(m, _)| m).collect();
        fullscreen_among(Bounds::of(w), framed_maximized(top), &all).then(|| format!("{class} (pid {pid})"))
    }
}

/// What the shell's notification state says the owner is busy with, when it
/// is something nothing should open over: a Direct3D fullscreen app (3) or
/// presentation mode (4). Pure.
pub fn busy_state(quns: i32) -> Option<&'static str> {
    match quns {
        3 => Some("a Direct3D fullscreen app"),
        4 => Some("presentation mode"),
        _ => None,
    }
}

/// What the shell says the owner is busy with now ([`busy_state`] of
/// [`notification_state`]).
pub fn busy_now() -> Option<&'static str> {
    notification_state().and_then(busy_state)
}

/// The shell's notification state (`SHQueryUserNotificationState`), or
/// `None` if it won't say.
pub fn notification_state() -> Option<i32> {
    // SAFETY: no arguments; the state comes back by value.
    unsafe { SHQueryUserNotificationState() }.ok().map(|s| s.0)
}

/// Whether two rectangles share any area.
pub fn intersects(a: Bounds, b: Bounds) -> bool {
    a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
}

/// Every monitor as (whole monitor, work area).
pub fn monitors() -> Vec<(Bounds, Bounds)> {
    unsafe extern "system" fn each(mon: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
        // SAFETY: `data` is the Vec below, alive for the whole enumeration.
        let out = unsafe { &mut *(data.0 as *mut Vec<(Bounds, Bounds)>) };
        if let Some(info) = monitor_info(mon) {
            out.push((Bounds::of(info.rcMonitor), Bounds::of(info.rcWork)));
        }
        BOOL(1)
    }
    let mut out: Vec<(Bounds, Bounds)> = Vec::new();
    // SAFETY: a callback that only pushes into `out`, which outlives the call.
    let _ = unsafe { EnumDisplayMonitors(None, None, Some(each), LPARAM(&raw mut out as isize)) };
    out
}

/// `b` unchanged if its middle is on one of the work areas `works`; else
/// moved (and shrunk if need be) into the nearest one. A saved spot on a
/// monitor that has since been unplugged comes back to the desktop.
pub fn settle(b: Bounds, works: &[Bounds]) -> Bounds {
    let (cx, cy) = (b.left + b.width() / 2, b.top + b.height() / 2);
    let inside = |w: &Bounds| cx >= w.left && cx < w.right && cy >= w.top && cy < w.bottom;
    if works.is_empty() || works.iter().any(inside) {
        return b;
    }
    let gap = |w: &Bounds| {
        let dx = i64::from((w.left - cx).max(0).max(cx - w.right));
        let dy = i64::from((w.top - cy).max(0).max(cy - w.bottom));
        dx * dx + dy * dy
    };
    let w = *works.iter().min_by_key(|w| gap(w)).unwrap_or(&works[0]);
    let (width, height) = (b.width().min(w.width()), b.height().min(w.height()));
    let left = b.left.clamp(w.left, w.right - width);
    let top = b.top.clamp(w.top, w.bottom - height);
    Bounds {
        left,
        top,
        right: left + width,
        bottom: top + height,
    }
}

/// [`settle`] against the monitors connected now.
pub fn onto_screen(b: Bounds) -> Bounds {
    let works: Vec<Bounds> = monitors().into_iter().map(|(_, w)| w).collect();
    settle(b, &works)
}

/// The monitors `at` overlaps (any part of him), in enumeration order.
pub fn touching(at: Bounds, monitors: &[Bounds]) -> Vec<Bounds> {
    monitors
        .iter()
        .copied()
        .filter(|m| intersects(at, *m))
        .collect()
}

/// The fullscreen app on any monitor any part of `at` is on, described for
/// the log, or `None` (Lorraine's `monitor_covered`, generalised from a
/// wallpaper's monitor to the ones Fritz stands on, and blind to maximised
/// windows). Fritz is always on top, so a sliver of him over a fullscreen
/// game is a sliver drawn over the game. Each monitor is tested at its
/// centre, so it holds whichever window has the focus: a fullscreen game
/// keeps Fritz parked while you look at another monitor.
pub fn covering_app(at: Bounds) -> Option<String> {
    let all: Vec<Bounds> = monitors().into_iter().map(|(m, _)| m).collect();
    touching(at, &all).into_iter().find_map(covered_monitor)
}

/// The window at `m`'s centre, if another app's fullscreen window. Our own
/// windows (Fritz is click-through, so the point passes through him anyway)
/// and the shell's never count.
fn covered_monitor(m: Bounds) -> Option<String> {
    let centre = POINT {
        x: m.left + m.width() / 2,
        y: m.top + m.height() / 2,
    };
    // SAFETY: plain queries; every handle below may be stale, which only
    // makes the calls fail.
    unsafe {
        let hit = WindowFromPoint(centre);
        if hit.0.is_null() {
            return None;
        }
        let root = GetAncestor(hit, GA_ROOT);
        let top = if root.0.is_null() { hit } else { root };
        let mut pid = 0u32;
        GetWindowThreadProcessId(top, Some(&raw mut pid));
        if pid == std::process::id() {
            return None;
        }
        let class = class_name(top);
        if is_shell_class(&class) {
            return None;
        }
        let mut w = RECT::default();
        let full = GetWindowRect(top, &raw mut w).is_ok() && fullscreen_on(Bounds::of(w), m, framed_maximized(top));
        full.then(|| format!("{class} (pid {pid})"))
    }
}

/// Whether this session is locked, as Windows tells it: `Some(true)` from a
/// lock (Win+L, or a sleep that asks for sign-in on wake) until the owner
/// signs back in, `Some(false)` otherwise, `None` if Windows won't say. The
/// lock screen is drawn above every always-on-top window, but its windows are
/// not ones [`covering_app`] can rely on: in Spike 0's session it found
/// Explorer's `LockScreenBackstopFrame` at the monitor's centre only some of
/// the time, and parked and unparked Fritz five times under one lock screen.
pub fn session_locked() -> Option<bool> {
    let mut buf = PWSTR::null();
    let mut len = 0u32;
    // SAFETY: on success Windows allocates `buf`, freed below exactly once;
    // it holds a WTSINFOEXW when at least that many bytes come back, and is
    // read unaligned so its alignment needn't be trusted. On failure nothing
    // was allocated.
    unsafe {
        WTSQuerySessionInformationW(
            Some(WTS_CURRENT_SERVER_HANDLE),
            WTS_CURRENT_SESSION,
            WTSSessionInfoEx,
            &raw mut buf,
            &raw mut len,
        )
        .ok()?;
        if buf.is_null() {
            return None;
        }
        let whole = usize::try_from(len).is_ok_and(|n| n >= std::mem::size_of::<WTSINFOEXW>());
        let flags = if whole {
            let info: WTSINFOEXW = std::ptr::read_unaligned(buf.0.cast());
            (info.Level == 1).then_some(info.Data.WTSInfoExLevel1.SessionFlags)
        } else {
            None
        };
        WTSFreeMemory(buf.0.cast());
        flags.and_then(lock_state)
    }
}

/// `WTSINFOEX_LEVEL1_W::SessionFlags` read as a lock: `WTS_SESSIONSTATE_LOCK`
/// is locked, `WTS_SESSIONSTATE_UNLOCK` unlocked, and anything else
/// (`WTS_SESSIONSTATE_UNKNOWN`) says nothing. (Windows 7 had the two the
/// wrong way round; Tauri 2 needs Windows 10.)
pub fn lock_state(flags: i32) -> Option<bool> {
    match flags.cast_unsigned() {
        WTS_SESSIONSTATE_LOCK => Some(true),
        WTS_SESSIONSTATE_UNLOCK => Some(false),
        _ => None,
    }
}

fn class_name(h: HWND) -> String {
    let mut name = [0u16; 128];
    // SAFETY: a sized buffer; a stale handle only makes the call fail.
    let n = usize::try_from(unsafe { GetClassNameW(h, &mut name) }).unwrap_or(0);
    String::from_utf16_lossy(&name[..n])
}

/// A window of this process (the tray's hidden window, the avatar...).
pub fn owned_by_us(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    let mut pid = 0u32;
    // SAFETY: a plain query; a stale handle leaves pid 0.
    unsafe { GetWindowThreadProcessId(hwnd(window), Some(&raw mut pid)) };
    pid == std::process::id()
}

/// What makes a top-level window one the keyboard can be handed to: what
/// Alt-Tab would list, less our own windows and the shell's, and less the
/// windows that won't take it — click-through overlays (another app's
/// stream overlay) and no-activate ones that don't ask for the taskbar.
#[allow(clippy::fn_params_excessive_bools)]
pub fn is_app_window(
    visible: bool,
    minimised: bool,
    cloaked: bool,
    ex: u32,
    owned: bool,
    ours: bool,
    class: &str,
) -> bool {
    let asks_for_taskbar = ex & WS_EX_APPWINDOW.0 != 0;
    visible
        && !minimised
        && !cloaked
        && ex & WS_EX_TOOLWINDOW.0 == 0
        && ex & WS_EX_TRANSPARENT.0 == 0
        && (ex & WS_EX_NOACTIVATE.0 == 0 || asks_for_taskbar)
        && (!owned || asks_for_taskbar)
        && !ours
        && !is_shell_class(class)
}

/// Of the app windows in z-order (handle, always-on-top), the one in use:
/// the first that isn't always on top. Topmost windows sit above every
/// other whatever was used last, so one (Task Manager on top, a pinned
/// window) is only the answer when there is no other app; then `desktop`.
pub fn pick_app(in_z_order: &[(isize, bool)], desktop: isize) -> isize {
    in_z_order
        .iter()
        .find(|(_, topmost)| !topmost)
        .or_else(|| in_z_order.first())
        .map_or(desktop, |(h, _)| *h)
}

/// Whether the z-order walk passes `id` over as ours: every window of this
/// process but `also` (the control window, an ordinary app window).
fn passed_over_as_ours(id: isize, also: isize) -> bool {
    id != also && owned_by_us(id)
}

/// The app in use, from the top of the z-order, that isn't `skip`; failing
/// any, the desktop (`GetShellWindow`), so a Lock always has somewhere to
/// hand the keyboard. 0 only with no shell at all. Our own windows are
/// passed over, but `also` (0: none).
pub fn next_app_window(skip: isize, also: isize) -> isize {
    let mut apps: Vec<(isize, bool)> = Vec::new();
    // SAFETY: plain queries walking the top-level z-order; a window that goes
    // mid-walk only ends the walk early.
    unsafe {
        let mut h = GetTopWindow(None).unwrap_or_default();
        for _ in 0..2048 {
            if h.0.is_null() {
                break;
            }
            let id = h.0 as isize;
            if id != skip {
                let mut cloaked = 0u32;
                let _ = DwmGetWindowAttribute(
                    h,
                    DWMWA_CLOAKED,
                    (&raw mut cloaked).cast(),
                    size_of::<u32>() as u32,
                );
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let ex = GetWindowLongPtrW(h, GWL_EXSTYLE) as u32;
                let owned = GetWindow(h, GW_OWNER).is_ok_and(|o| !o.0.is_null());
                if is_app_window(
                    IsWindowVisible(h).as_bool(),
                    IsIconic(h).as_bool(),
                    cloaked != 0,
                    ex,
                    owned,
                    passed_over_as_ours(id, also),
                    &class_name(h),
                ) {
                    let topmost = ex & WS_EX_TOPMOST.0 != 0;
                    apps.push((id, topmost));
                    if !topmost {
                        break; // the first non-topmost one is the answer
                    }
                }
            }
            h = GetWindow(h, GW_HWNDNEXT).unwrap_or_default();
        }
        pick_app(&apps, GetShellWindow().0 as isize)
    }
}

// What is under the mouse, for the mouse hook's procedure: plain reads, no
// message sent, no log, no allocation.

/// Whether an extended style passes clicks through (`WS_EX_TRANSPARENT`
/// with `WS_EX_LAYERED`): his window when Locked, another app's overlay.
pub(crate) fn passes_clicks(ex: u32) -> bool {
    ex & WS_EX_TRANSPARENT.0 != 0 && ex & WS_EX_LAYERED.0 != 0
}

/// Whether `window` is shown and passes clicks through it: his, Locked,
/// but never his in Arrange, even between the Lock and its style.
pub fn click_through(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    let h = hwnd(window);
    // SAFETY: plain queries on a window handle; a stale one answers false.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    unsafe {
        IsWindowVisible(h).as_bool() && passes_clicks(GetWindowLongPtrW(h, GWL_EXSTYLE) as u32)
    }
}

/// A window above his, as the walk reads it: each read made only when the
/// one before it didn't settle the question.
pub(crate) trait Above {
    fn visible(&self) -> bool;
    fn rect(&self) -> Option<Bounds>;
    fn ex(&self) -> u32;
    fn cloaked(&self) -> bool;
}

/// Whether one of `above` (from just above his window up to the top) takes
/// a click at `at`: the first that is visible, holds the point, doesn't pass
/// clicks through and isn't cloaked. `None` is the walk losing its place (a
/// window gone mid-walk), and past `cap` windows: either counts as taken
/// (he can't tell, so it isn't his). At the top, not.
pub(crate) fn first_over<A: Above>(above: impl Iterator<Item = Option<A>>, at: (i32, i32), cap: usize) -> bool {
    for (seen, w) in above.enumerate() {
        let Some(w) = w else {
            return true;
        };
        if seen >= cap {
            return true;
        }
        if w.visible() && w.rect().is_some_and(|r| crate::hit_shape::holds(r, at)) && !passes_clicks(w.ex()) && !w.cloaked() {
            return true;
        }
    }
    false
}

/// A top-level window, read lazily.
struct Win(HWND);

impl Above for Win {
    fn visible(&self) -> bool {
        // SAFETY: a plain query; a stale handle is just not visible.
        unsafe { IsWindowVisible(self.0) }.as_bool()
    }
    fn rect(&self) -> Option<Bounds> {
        let mut r = RECT::default();
        // SAFETY: a window handle and a valid out-pointer.
        unsafe { GetWindowRect(self.0, &raw mut r) }.ok().map(|()| Bounds::of(r))
    }
    fn ex(&self) -> u32 {
        // SAFETY: a plain query; a stale handle answers 0.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        unsafe {
            GetWindowLongPtrW(self.0, GWL_EXSTYLE) as u32
        }
    }
    fn cloaked(&self) -> bool {
        let mut cloaked = 0u32;
        // SAFETY: a window handle and a sized out-value; a failure leaves 0.
        let _ = unsafe {
            DwmGetWindowAttribute(self.0, DWMWA_CLOAKED, (&raw mut cloaked).cast(), size_of::<u32>() as u32)
        };
        cloaked != 0
    }
}

/// The top-level windows above `window`, nearest first; then, if the walk
/// ended because the window it stood on had gone (rather than at the top),
/// one `None`.
fn windows_above(window: isize) -> impl Iterator<Item = Option<Win>> {
    let mut h = hwnd(window);
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        // SAFETY: walking the z-order with plain queries.
        match unsafe { GetWindow(h, GW_HWNDPREV) } {
            Ok(p) if !p.0.is_null() => {
                h = p;
                Some(Some(Win(h)))
            }
            _ => {
                done = true;
                // SAFETY: a plain query; a stale handle answers false.
                let gone = !unsafe { IsWindow(Some(h)) }.as_bool();
                gone.then_some(None)
            }
        }
    })
}

/// Whether a window above `window` takes a click at `at` ([`first_over`]).
/// His window gone, or one above it gone mid-walk, counts as covered.
pub fn covered_at(window: isize, at: (i32, i32), cap: usize) -> bool {
    // SAFETY: a plain query; a stale handle answers false.
    if window == 0 || !unsafe { IsWindow(Some(hwnd(window))) }.as_bool() {
        return true;
    }
    first_over(windows_above(window), at, cap)
}

/// Whether a thread's GUI state says it is busy with the mouse: in a menu,
/// moving or sizing a window, or holding the capture. Pure.
pub(crate) fn busy_gui(flags: u32, capture: bool) -> bool {
    flags & (GUI_INMENUMODE.0 | GUI_POPUPMENUMODE.0 | GUI_INMOVESIZE.0) != 0 || capture
}

/// The shell's flyouts in front (Start, Search, the notification centre)
/// are this class; they close as they lose the foreground.
const FLYOUT_CLASS: &str = "Windows.UI.Core.CoreWindow";

/// Whether `h`'s class is `name`, without allocating.
fn class_is(h: HWND, name: &str) -> bool {
    let mut buf = [0u16; 64];
    // SAFETY: a sized buffer; a stale handle only makes the call fail.
    let n = usize::try_from(unsafe { GetClassNameW(h, &mut buf) }).unwrap_or(0);
    n == name.len() && name.encode_utf16().zip(&buf[..n]).all(|(a, &b)| a == b)
}

/// Whether the app in front is busy with the mouse, so a press goes to it
/// whatever is beneath: in a menu, moving or sizing a window, holding the
/// capture, or a shell flyout open. Never for no foreground or one of ours.
pub fn foreground_busy() -> bool {
    // SAFETY: plain queries; every handle may be stale, which only makes the
    // calls fail.
    unsafe {
        let fg = GetForegroundWindow();
        if fg.0.is_null() {
            return false;
        }
        let mut pid = 0u32;
        let thread = GetWindowThreadProcessId(fg, Some(&raw mut pid));
        if pid == 0 || pid == std::process::id() {
            return false;
        }
        if class_is(fg, FLYOUT_CLASS) {
            return true;
        }
        let mut info = GUITHREADINFO {
            cbSize: size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        GetGUIThreadInfo(thread, &raw mut info).is_ok() && busy_gui(info.flags.0, !info.hwndCapture.0.is_null())
    }
}

// His menu (popup.rs): on the tray's window, on the main thread.

/// Show `menu` at `at` on `owner` by mouse, and the id chosen (0 for none):
/// `TPM_RETURNCMD`, so nothing is queued and the caller sends the choice on
/// at once; `TPM_RIGHTBUTTON`, so either button picks. Main thread: the
/// menu's modal loop runs here.
pub fn track_menu(menu: isize, owner: isize, at: (i32, i32)) -> u32 {
    if menu == 0 || owner == 0 {
        return 0;
    }
    let flags = TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RIGHTBUTTON | TPM_RETURNCMD;
    // SAFETY: a menu and a window of this thread; a stale one only fails.
    let id = unsafe { TrackPopupMenu(HMENU(menu as *mut c_void), flags, at.0, at.1, None, hwnd(owner), None) };
    u32::try_from(id.0).unwrap_or(0)
}

/// The command id of the item at `pos` in menu `menu`, for the log; 0 for
/// no menu (and Windows' own answers for a separator or a submenu).
pub fn menu_item_id(menu: isize, pos: i32) -> u32 {
    if menu == 0 {
        return 0;
    }
    // SAFETY: a read of a menu handle; a stale one only fails.
    unsafe { GetMenuItemID(HMENU(menu as *mut c_void), pos) }
}

/// A menu item's id to `owner` as `WM_COMMAND`, at once (muda's subclass on
/// the tray's window turns it into the tray's own menu event).
pub fn send_command(owner: isize, id: u32) {
    if owner == 0 {
        return;
    }
    // SAFETY: a window of this thread; the id is a menu item's.
    unsafe { SendMessageW(hwnd(owner), WM_COMMAND, Some(WPARAM(id as usize)), Some(LPARAM(0))) };
}

/// The benign message Windows documents after a popup menu.
pub fn post_null(owner: isize) {
    if owner == 0 {
        return;
    }
    // SAFETY: a window handle; a stale one only makes the post fail.
    let _ = unsafe { PostMessageW(Some(hwnd(owner)), WM_NULL, WPARAM(0), LPARAM(0)) };
}

/// Whether any of `rects` holds `at`.
pub(crate) fn any_holds(rects: &[Bounds], at: (i32, i32)) -> bool {
    rects.iter().any(|&r| crate::hit_shape::holds(r, at))
}

/// The class Windows gives a popup menu's window.
const MENU_CLASS: &str = "#32768";

/// Whether a window of the menu's thread is one of its menus: shown, and of
/// the menu class (his own window and Settings are that thread's too).
pub(crate) fn is_menu_window(visible: bool, menu_class: bool) -> bool {
    visible && menu_class
}

/// Whether `at` is on a menu `thread` shows (his menu, or one of its
/// submenus): the visible menu windows of that thread. For the hook's
/// owner thread, never the procedure.
pub fn our_menu_at(thread: u32, at: (i32, i32)) -> bool {
    unsafe extern "system" fn each(h: HWND, data: LPARAM) -> BOOL {
        // SAFETY: `data` is the Vec below, alive for the whole enumeration.
        let rects = unsafe { &mut *(data.0 as *mut Vec<Bounds>) };
        // SAFETY: plain queries on a window of the thread being walked.
        if is_menu_window(unsafe { IsWindowVisible(h) }.as_bool(), class_is(h, MENU_CLASS)) {
            let mut r = RECT::default();
            // SAFETY: a window handle and a valid out-pointer.
            if unsafe { GetWindowRect(h, &raw mut r) }.is_ok() {
                rects.push(Bounds::of(r));
            }
        }
        BOOL(1)
    }
    if thread == 0 {
        return false;
    }
    let mut rects: Vec<Bounds> = Vec::new();
    // SAFETY: a callback that only pushes into `rects`, which outlives it.
    let _ = unsafe { EnumThreadWindows(thread, Some(each), LPARAM(&raw mut rects as isize)) };
    any_holds(&rects, at)
}

/// Half a double-click rectangle's side, at least 1 px.
pub(crate) fn half(cx: i32) -> i32 {
    (cx / 2).max(1)
}

/// Windows' double-click: its time (ms), and half its rectangle's width and
/// height in physical pixels at `window`'s DPI (96 for none).
pub fn double_click_limits(window: isize) -> (u32, i32, i32) {
    // SAFETY: plain queries; a stale handle answers DPI 0.
    unsafe {
        let dpi = if window == 0 { 0 } else { GetDpiForWindow(hwnd(window)) };
        let dpi = if dpi == 0 { 96 } else { dpi };
        (
            GetDoubleClickTime(),
            half(GetSystemMetricsForDpi(SM_CXDOUBLECLK, dpi)),
            half(GetSystemMetricsForDpi(SM_CYDOUBLECLK, dpi)),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{
        busy_state, framed_style, fullscreen_among, primary_work_area, work_area_at,
        apply_toolwindow, can_take_foreground, covers, foreground_window, fullscreen_on, hwnd,
        intersects, is_app_window, is_shell_class, lock_state, make_layered_opaque, monitors,
        next_app_window, owned_by_us, passed_over_as_ours, pick_app, pin_toolwindow, session_locked,
        settle,
        toolwindow_style, touching, try_foreground, window_bounds, Bounds,
    };
    use windows::core::w;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, GetShellWindow, GetWindowLongPtrW, SetWindowLongPtrW, GWL_EXSTYLE,
        WINDOW_EX_STYLE, WS_EX_APPWINDOW, WS_EX_LAYERED, WS_EX_TOOLWINDOW, WS_POPUP,
    };

    /// A hidden window of the predefined STATIC class (nothing to register),
    /// destroyed on drop.
    pub(crate) struct Hidden(pub(crate) HWND);

    impl Hidden {
        pub(crate) fn new(ex: WINDOW_EX_STYLE) -> Self {
            // SAFETY: a predefined class, no parent, menu or creation data.
            let h = unsafe {
                CreateWindowExW(
                    ex,
                    w!("STATIC"),
                    w!("fritz window test"),
                    WS_POPUP,
                    10,
                    20,
                    30,
                    40,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .unwrap();
            Self(h)
        }

        pub(crate) fn id(&self) -> isize {
            self.0 .0 as isize
        }
    }

    impl Drop for Hidden {
        fn drop(&mut self) {
            // SAFETY: our own window, destroyed once on the thread that made it.
            let _ = unsafe { DestroyWindow(self.0) };
        }
    }

    #[test]
    fn no_window_is_nothing() {
        assert!(!try_foreground(0));
        assert!(!can_take_foreground(0));
        assert!(!make_layered_opaque(0));
        assert!(!apply_toolwindow(0));
        assert_eq!(window_bounds(0), None);
    }

    #[test]
    fn handles_round_trip() {
        let win = Hidden::new(WINDOW_EX_STYLE(0));
        assert_eq!(hwnd(win.id()), win.0);
    }

    /// A hidden window can't take the foreground back, is never the
    /// foreground, and only a layered one takes layered attributes.
    #[test]
    fn queries_on_a_real_window() {
        let plain = Hidden::new(WINDOW_EX_STYLE(0));
        assert!(!can_take_foreground(plain.id()), "hidden");
        assert!(!make_layered_opaque(plain.id()), "not layered");
        assert_ne!(foreground_window(), plain.id());
        let layered = Hidden::new(WS_EX_LAYERED);
        assert!(make_layered_opaque(layered.id()));
        assert_eq!(
            window_bounds(plain.id()),
            Some(Bounds {
                left: 10,
                top: 20,
                right: 40,
                bottom: 60
            })
        );
    }

    #[test]
    fn toolwindow_style_math() {
        let (tool, app) = (WS_EX_TOOLWINDOW.0, WS_EX_APPWINDOW.0);
        assert_eq!(toolwindow_style(0), tool);
        assert_eq!(toolwindow_style(app), tool);
        assert_eq!(
            toolwindow_style(app | 0x8),
            tool | 0x8,
            "other bits are kept"
        );
        assert_eq!(toolwindow_style(tool), tool);
    }

    /// The bit goes on a real window, and a second call changes nothing.
    #[test]
    fn toolwindow_goes_on_and_stays() {
        let win = Hidden::new(WS_EX_APPWINDOW);
        let ex = || {
            // SAFETY: our own live window.
            unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) as u32 }
        };
        assert!(apply_toolwindow(win.id()));
        assert_eq!(ex() & WS_EX_TOOLWINDOW.0, WS_EX_TOOLWINDOW.0);
        assert_eq!(ex() & WS_EX_APPWINDOW.0, 0);
        let once = ex();
        assert!(apply_toolwindow(win.id()));
        assert_eq!(ex(), once);
    }

    #[test]
    fn covering_means_all_four_edges() {
        let mon = Bounds {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1080,
        };
        assert!(covers(mon, mon));
        assert!(covers(
            Bounds {
                left: -8,
                top: -8,
                right: 1928,
                bottom: 1088
            },
            mon
        ));
        // Maximised: the taskbar is still showing.
        assert!(!covers(
            Bounds {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1032
            },
            mon
        ));
        // Another monitor.
        assert!(!covers(
            Bounds {
                left: 1920,
                top: 0,
                right: 3840,
                bottom: 1080
            },
            mon
        ));
    }

    /// Pinned, the bit survives a rewrite of the whole extended style, as tao
    /// does on every change; the other bits in the rewrite still land.
    #[test]
    fn a_pinned_window_stays_out_of_alt_tab() {
        let win = Hidden::new(WS_EX_APPWINDOW);
        let ex = || {
            // SAFETY: our own live window.
            unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) as u32 }
        };
        assert!(pin_toolwindow(win.id()));
        let rewrite = (WS_EX_APPWINDOW.0 | WS_EX_LAYERED.0) as isize;
        // SAFETY: our own live window, on its own thread.
        unsafe { SetWindowLongPtrW(win.0, GWL_EXSTYLE, rewrite) };
        assert_eq!(
            ex() & WS_EX_TOOLWINDOW.0,
            WS_EX_TOOLWINDOW.0,
            "tool window kept"
        );
        assert_eq!(ex() & WS_EX_APPWINDOW.0, 0, "app window refused");
        assert_eq!(
            ex() & WS_EX_LAYERED.0,
            WS_EX_LAYERED.0,
            "the rest of the rewrite lands"
        );
        assert!(!pin_toolwindow(0));
    }

    /// Unpinned, the same rewrite drops it: what the pin is for.
    #[test]
    fn an_unpinned_window_loses_it_to_a_rewrite() {
        let win = Hidden::new(WS_EX_APPWINDOW);
        assert!(apply_toolwindow(win.id()));
        // SAFETY: our own live window, on its own thread.
        unsafe { SetWindowLongPtrW(win.0, GWL_EXSTYLE, WS_EX_APPWINDOW.0 as isize) };
        // SAFETY: our own live window.
        let ex = unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) as u32 };
        assert_eq!(ex & WS_EX_TOOLWINDOW.0, 0);
    }

    /// The values Windows documents for `SessionFlags`: 0 locked, 1 unlocked,
    /// 0xFFFFFFFF unknown.
    #[test]
    fn session_flags_read_as_a_lock() {
        assert_eq!(lock_state(0), Some(true));
        assert_eq!(lock_state(1), Some(false));
        assert_eq!(lock_state(-1), None, "WTS_SESSIONSTATE_UNKNOWN");
        assert_eq!(lock_state(2), None);
    }

    /// This desktop session's lock can be read at all: the call, the
    /// buffer's size and its level-1 layout. (The tests run unlocked or
    /// locked; either is an answer.)
    #[test]
    fn the_session_lock_can_be_read() {
        assert!(session_locked().is_some());
    }

    /// Another app's window is fullscreen if it takes over any one monitor
    /// whole, unmaximised; one spanning two monitors takes over neither.
    #[test]
    fn fullscreen_among_table() {
        let one = Bounds { left: 0, top: 0, right: 1920, bottom: 1080 };
        let two = Bounds { left: 1920, top: 0, right: 4480, bottom: 1440 };
        let both = [one, two];
        assert!(!fullscreen_among(one, true, &both), "maximised");
        assert!(fullscreen_among(two, false, &both), "covering monitor 2");
        let span = Bounds { left: 960, top: 0, right: 3000, bottom: 1440 };
        assert!(!fullscreen_among(span, false, &both), "spanning two, covering neither whole");
        let windowed = Bounds { left: 100, top: 100, right: 900, bottom: 700 };
        assert!(!fullscreen_among(windowed, false, &both));
        assert!(!fullscreen_among(one, false, &[]), "no monitors");
    }

    /// Both fullscreen tests, the foreground's and a monitor's, exempt only
    /// a framed window merely maximised, never any maximised one.
    #[test]
    fn both_fullscreen_tests_exempt_only_framed_maximised_windows() {
        let all = include_str!("windows_ext.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        for name in ["pub fn fullscreen_foreground(", "fn covered_monitor("] {
            let at = src.find(name).unwrap();
            let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
            assert!(body.contains("framed_maximized(top)"), "{name}");
            assert!(!body.contains("IsZoomed"), "{name} reads IsZoomed only through framed_maximized");
        }
    }

    /// A sizing border or a caption is a frame; a bare popup has none.
    #[test]
    fn a_frame_is_a_sizing_border_or_a_caption() {
        use windows::Win32::UI::WindowsAndMessaging::{WS_CAPTION, WS_OVERLAPPEDWINDOW, WS_POPUP, WS_THICKFRAME, WS_VISIBLE};
        assert!(framed_style(WS_OVERLAPPEDWINDOW.0), "an editor");
        assert!(framed_style((WS_POPUP | WS_THICKFRAME).0), "a frameless app that can be resized");
        assert!(framed_style((WS_POPUP | WS_CAPTION).0));
        assert!(!framed_style((WS_POPUP | WS_VISIBLE).0), "a borderless fullscreen app");
        assert!(!framed_style(0));
    }

    /// Only a Direct3D fullscreen app and presentation mode are busy.
    #[test]
    fn only_d3d_and_presentation_mode_are_busy() {
        for quns in -1..=7 {
            let want = match quns {
                3 => Some("a Direct3D fullscreen app"),
                4 => Some("presentation mode"),
                _ => None,
            };
            assert_eq!(busy_state(quns), want, "{quns}");
        }
    }

    /// The work area at the primary's centre is the primary's.
    #[test]
    fn work_area_at_matches_the_primary_at_its_centre() {
        let (work, dpi) = primary_work_area().unwrap();
        let at = work_area_at(work.left + work.width() / 2, work.top + work.height() / 2).unwrap();
        assert_eq!(at, (work, dpi));
        assert!(dpi >= 48);
    }

    /// A point on no monitor finds the nearest one.
    #[test]
    fn a_point_off_every_monitor_finds_the_nearest() {
        let (work, _) = work_area_at(-1_000_000, -1_000_000).unwrap();
        assert!(monitors().iter().any(|&(_, w)| w == work));
    }

    /// Fullscreen is covering the monitor without being maximised. With an
    /// auto-hiding taskbar a maximised window covers it too (this is the
    /// owner's Claude window, measured), and stays a working window.
    #[test]
    fn a_maximised_window_is_not_fullscreen() {
        let mon = Bounds {
            left: 0,
            top: 0,
            right: 3840,
            bottom: 2400,
        };
        let maximised = Bounds {
            left: -13,
            top: -13,
            right: 3853,
            bottom: 2413,
        };
        assert!(covers(maximised, mon));
        assert!(!fullscreen_on(maximised, mon, true));
        assert!(fullscreen_on(mon, mon, false), "borderless fullscreen");
        let short = Bounds {
            left: 0,
            top: 0,
            right: 3840,
            bottom: 2352,
        };
        assert!(!fullscreen_on(short, mon, false));
    }

    fn b(left: i32, top: i32, right: i32, bottom: i32) -> Bounds {
        Bounds {
            left,
            top,
            right,
            bottom,
        }
    }

    /// A spot still on a monitor stays put; one whose monitor is gone (the
    /// owner undocked) comes back into the nearest work area, shrunk if it
    /// can't fit.
    #[test]
    fn settle_brings_a_lost_spot_back() {
        let laptop = b(0, 0, 3840, 2352);
        let gone = b(4000, 500, 4600, 1420);
        assert_eq!(
            settle(b(3192, 1432, 3792, 2352), &[laptop]),
            b(3192, 1432, 3792, 2352)
        );
        assert_eq!(settle(gone, &[laptop]), b(3240, 500, 3840, 1420));
        assert_eq!(
            settle(b(-900, -50, -300, 870), &[laptop]),
            b(0, 0, 600, 920)
        );
        let tall = b(5000, 0, 5600, 5000);
        assert_eq!(settle(tall, &[laptop]), b(3240, 0, 3840, 2352));
        // Two monitors: the nearer one wins.
        let left = b(-1920, 0, 0, 1080);
        assert_eq!(
            settle(b(-2600, 100, -2000, 1020), &[laptop, left]),
            b(-1920, 100, -1320, 1020)
        );
        // With no monitors known, nothing moves.
        assert_eq!(settle(gone, &[]), gone);
    }

    /// Straddling two monitors, both are tested: a sliver over a fullscreen
    /// game is a sliver drawn over it.
    #[test]
    fn touching_finds_every_monitor_he_overlaps() {
        let a = b(0, 0, 1920, 1080);
        let c = b(1920, 0, 3840, 1080);
        assert_eq!(touching(b(1700, 300, 2000, 760), &[a, c]), [a, c]);
        assert_eq!(touching(b(100, 300, 400, 760), &[a, c]), [a]);
        assert!(intersects(a, a));
        assert!(!intersects(a, c), "edges touching is no overlap");
    }

    /// The live desktop has at least one monitor, and every work area sits
    /// inside its monitor.
    #[test]
    fn monitors_are_enumerated() {
        let all = monitors();
        assert!(!all.is_empty());
        for (m, w) in all {
            assert!(covers(m, w), "{m:?} {w:?}");
        }
    }

    #[test]
    fn app_windows_are_what_alt_tab_lists_less_ours() {
        let ok = |ex: u32, owned: bool, ours: bool, class: &str| {
            is_app_window(true, false, false, ex, owned, ours, class)
        };
        assert!(ok(0, false, false, "Notepad"));
        assert!(
            !ok(WS_EX_TOOLWINDOW.0, false, false, "Notepad"),
            "tool window"
        );
        assert!(!ok(0, true, false, "#32770"), "an owned dialog");
        assert!(
            ok(WS_EX_APPWINDOW.0, true, false, "Owned app"),
            "owned but asks for the taskbar"
        );
        assert!(!ok(0, false, true, "Tauri Window"), "ours");
        assert!(!ok(0, false, false, "Shell_TrayWnd"), "the taskbar");
        assert!(
            !is_app_window(false, false, false, 0, false, false, "Notepad"),
            "hidden"
        );
        assert!(
            !is_app_window(true, true, false, 0, false, false, "Notepad"),
            "minimised"
        );
        assert!(
            !is_app_window(true, false, true, 0, false, false, "Notepad"),
            "cloaked"
        );
    }

    /// Click-through overlays and no-activate windows don't take the
    /// keyboard, unless they ask for the taskbar.
    #[test]
    fn windows_that_wont_take_the_keyboard_are_skipped() {
        use windows::Win32::UI::WindowsAndMessaging::{WS_EX_NOACTIVATE, WS_EX_TRANSPARENT};
        let ok = |ex: u32| is_app_window(true, false, false, ex, false, false, "App");
        assert!(
            !ok(WS_EX_TRANSPARENT.0 | WS_EX_LAYERED.0),
            "a click-through overlay"
        );
        assert!(!ok(WS_EX_NOACTIVATE.0), "no-activate");
        assert!(
            ok(WS_EX_NOACTIVATE.0 | WS_EX_APPWINDOW.0),
            "no-activate but on the taskbar"
        );
    }

    /// The app in use is the first that isn't always on top; a topmost one
    /// only when it's all there is; the desktop when there's nothing.
    #[test]
    fn the_app_in_use_is_below_the_topmost_band() {
        const DESKTOP: isize = 1;
        assert_eq!(
            pick_app(&[(10, true), (20, false), (30, false)], DESKTOP),
            20
        );
        assert_eq!(pick_app(&[(10, true), (11, true)], DESKTOP), 10);
        assert_eq!(pick_app(&[], DESKTOP), DESKTOP);
    }

    /// On a live desktop there is always somewhere to hand the keyboard (an
    /// app, or the desktop itself), and it is never the window skipped.
    #[test]
    fn there_is_always_somewhere_to_hand_the_keyboard() {
        let win = Hidden::new(WINDOW_EX_STYLE(0));
        let to = next_app_window(win.id(), 0);
        assert_ne!(to, 0);
        assert_ne!(to, win.id());
    }

    /// The walk passes over every window of ours but the one named (the
    /// control window); no one else's is ever passed over as ours.
    #[test]
    fn only_the_named_window_of_ours_is_an_app() {
        let win = Hidden::new(WINDOW_EX_STYLE(0));
        let other = Hidden::new(WINDOW_EX_STYLE(0));
        assert!(passed_over_as_ours(win.id(), 0));
        assert!(passed_over_as_ours(win.id(), other.id()));
        assert!(!passed_over_as_ours(win.id(), win.id()));
        // SAFETY: a plain query.
        let shell = unsafe { GetShellWindow() }.0 as isize;
        assert!(!passed_over_as_ours(shell, 0));
    }

    #[test]
    fn our_own_windows_are_ours() {
        let win = Hidden::new(WINDOW_EX_STYLE(0));
        assert!(owned_by_us(win.id()));
        assert!(!owned_by_us(0));
    }

    #[test]
    fn the_shell_never_covers() {
        for class in [
            "Progman",
            "WorkerW",
            "Shell_TrayWnd",
            "Shell_SecondaryTrayWnd",
        ] {
            assert!(is_shell_class(class), "{class}");
        }
        for class in ["UnityWndClass", "Chrome_WidgetWin_1", ""] {
            assert!(!is_shell_class(class), "{class}");
        }
    }
}

/// The mouse hook's reads.
#[cfg(test)]
mod under_the_mouse {
    #![allow(clippy::unwrap_used)]

    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    use windows::Win32::UI::WindowsAndMessaging::{
        GetTopWindow, GetWindow, GW_HWNDNEXT, WINDOW_EX_STYLE, WS_EX_LAYERED, WS_EX_TOPMOST, WS_EX_TRANSPARENT,
    };

    use super::tests::Hidden;
    use super::{busy_gui, click_through, covered_at, double_click_limits, first_over, half, passes_clicks, Above, Bounds};

    /// A window above his, whose reads are logged by its name.
    struct Fake<'a> {
        name: &'static str,
        visible: bool,
        rect: Bounds,
        ex: u32,
        cloaked: bool,
        log: &'a RefCell<Vec<String>>,
    }

    impl Above for Fake<'_> {
        fn visible(&self) -> bool {
            self.log.borrow_mut().push(format!("{} visible", self.name));
            self.visible
        }
        fn rect(&self) -> Option<Bounds> {
            self.log.borrow_mut().push(format!("{} rect", self.name));
            Some(self.rect)
        }
        fn ex(&self) -> u32 {
            self.log.borrow_mut().push(format!("{} ex", self.name));
            self.ex
        }
        fn cloaked(&self) -> bool {
            self.log.borrow_mut().push(format!("{} cloaked", self.name));
            self.cloaked
        }
    }

    const HERE: Bounds = Bounds { left: 0, top: 0, right: 100, bottom: 100 };
    const ELSEWHERE: Bounds = Bounds { left: 200, top: 0, right: 300, bottom: 100 };
    const THROUGH: u32 = WS_EX_TRANSPARENT.0 | WS_EX_LAYERED.0;

    #[test]
    fn first_over_table() {
        let log = RefCell::new(Vec::new());
        let w = |name, visible, rect, ex, cloaked| Fake { name, visible, rect, ex, cloaked, log: &log };
        for (name, above, want) in [
            ("nothing above", vec![], false),
            ("hidden", vec![w("a", false, HERE, 0, false)], false),
            ("cloaked", vec![w("a", true, HERE, 0, true)], false),
            ("click-through", vec![w("a", true, HERE, THROUGH, false)], false),
            ("beside the point", vec![w("a", true, ELSEWHERE, 0, false)], false),
            ("transparent but not layered", vec![w("a", true, HERE, WS_EX_TRANSPARENT.0, false)], true),
            ("layered but not transparent", vec![w("a", true, HERE, WS_EX_LAYERED.0, false)], true),
            ("visible and holding the point", vec![w("a", true, HERE, WS_EX_TOPMOST.0, false)], true),
            ("past the passers-by", vec![w("a", false, HERE, 0, false), w("b", true, ELSEWHERE, 0, false), w("c", true, HERE, 0, false)], true),
        ] {
            assert_eq!(first_over(above.into_iter().map(Some), (50, 50), 512), want, "{name}");
        }
        // The walk losing its place (a window gone mid-walk) is covered,
        // whatever came before it; reaching the top is not.
        let lost = vec![Some(w("a", false, HERE, 0, false)), None];
        assert!(first_over(lost.into_iter(), (50, 50), 512), "a window gone mid-walk");
        assert!(first_over(vec![None::<Fake>].into_iter(), (50, 50), 512), "gone at the first step");
        // The reads are made only as needed: a hidden window's rectangle
        // isn't read, nor the style or cloak of one beside the point.
        log.borrow_mut().clear();
        let ws = vec![w("hidden", false, HERE, 0, false), w("beside", true, ELSEWHERE, 0, false), w("over", true, HERE, 0, false), w("never", true, HERE, 0, false)];
        assert!(first_over(ws.into_iter().map(Some), (50, 50), 512));
        assert_eq!(
            *log.borrow(),
            ["hidden visible", "beside visible", "beside rect", "over visible", "over rect", "over ex", "over cloaked"]
        );
        // The cap: past it, covered; up to it, as found.
        let many = |n: usize| (0..n).map(|_| w("x", false, HERE, 0, false)).collect::<Vec<_>>();
        assert!(first_over(many(5).into_iter().map(Some), (50, 50), 4), "past the cap");
        assert!(!first_over(many(4).into_iter().map(Some), (50, 50), 4), "the top reached at the cap");
        assert!(!first_over(many(3).into_iter().map(Some), (50, 50), 4));
    }

    #[test]
    fn passes_clicks_table() {
        assert!(passes_clicks(THROUGH));
        assert!(passes_clicks(THROUGH | WS_EX_TOPMOST.0));
        assert!(!passes_clicks(WS_EX_TRANSPARENT.0));
        assert!(!passes_clicks(WS_EX_LAYERED.0));
        assert!(!passes_clicks(0));
    }

    #[test]
    fn busy_gui_table() {
        for (flags, capture, want) in [
            (4, false, true),  // GUI_INMENUMODE
            (16, false, true), // GUI_POPUPMENUMODE
            (2, false, true),  // GUI_INMOVESIZE
            (0, true, true),   // holding the capture
            (0, false, false),
            (1, false, false), // GUI_CARETBLINKING
            (8, false, false), // GUI_SYSTEMMENUMODE alone
        ] {
            assert_eq!(busy_gui(flags, capture), want, "{flags} {capture}");
        }
    }

    #[test]
    fn any_holds_table() {
        let menus = [HERE, ELSEWHERE];
        assert!(super::any_holds(&menus, (50, 50)));
        assert!(super::any_holds(&menus, (250, 99)));
        assert!(!super::any_holds(&menus, (150, 50)), "between them");
        assert!(!super::any_holds(&menus, (100, 50)), "the right edge is outside");
        assert!(!super::any_holds(&[], (50, 50)));
        // No menu of this test's thread is open.
        // SAFETY: this thread's id.
        let me = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
        assert!(!super::our_menu_at(me, (50, 50)));
        assert!(!super::our_menu_at(0, (50, 50)));
    }

    #[test]
    fn only_a_shown_menu_window_counts_as_the_menu() {
        assert!(super::is_menu_window(true, true));
        assert!(!super::is_menu_window(true, false), "his window, Settings");
        assert!(!super::is_menu_window(false, true), "a menu gone");
        assert!(!super::is_menu_window(false, false));
        let src = include_str!("windows_ext.rs").replace("\r\n", "\n");
        let at = src.find("pub fn our_menu_at(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(body.contains("if is_menu_window(unsafe { IsWindowVisible(h) }.as_bool(), class_is(h, MENU_CLASS)) {\n            let mut r = RECT::default();"));
        assert!(body.contains("EnumThreadWindows(thread, Some(each),") && body.trim_end().ends_with("any_holds(&rects, at)"));
        // The class it looks for is the one Windows gives a hidden window's
        // class back as: a check on a real window of this test's thread.
        let w = super::tests::Hidden::new(windows::Win32::UI::WindowsAndMessaging::WINDOW_EX_STYLE(0));
        assert!(super::class_is(w.0, "Static") && !super::class_is(w.0, super::MENU_CLASS));
    }

    #[test]
    fn the_menu_calls_do_nothing_without_a_window() {
        assert_eq!(super::track_menu(0, 7, (0, 0)), 0);
        assert_eq!(super::track_menu(7, 0, (0, 0)), 0);
        super::send_command(0, 1003);
        super::post_null(0);
    }

    #[test]
    fn activation_state_table() {
        for (wp, active) in [
            (0usize, false),     // WA_INACTIVE
            (1, true),           // WA_ACTIVE
            (2, true),           // WA_CLICKACTIVE: a click into it
            (0x0001_0000, false), // inactive and minimised
            (0x0001_0002, true), // clicked into, minimised flag set
        ] {
            assert_eq!(super::activation_state(wp), active, "{wp:#x}");
        }
        // The procedure passes Windows' word on, read by that table.
        let src = include_str!("windows_ext.rs").replace("\r\n", "\n");
        let at = src.find("unsafe extern \"system\" fn activate_proc(").unwrap();
        let proc_ = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(proc_.contains("if msg == WM_ACTIVATE && data != 0 {"));
        assert!(proc_.contains("on_change(activation_state(wp.0));"));
    }

    #[test]
    fn half_table() {
        for (cx, want) in [(4, 2), (5, 2), (8, 4), (1, 1), (0, 1), (-6, 1)] {
            assert_eq!(half(cx), want, "{cx}");
        }
    }

    /// Never a window that isn't shown; nothing for no window; a window
    /// that is gone counts as covered.
    #[test]
    fn real_windows() {
        let hidden = Hidden::new(WINDOW_EX_STYLE(THROUGH));
        assert!(!click_through(hidden.id()), "hidden, so not his to click");
        assert!(!click_through(0));
        assert!(covered_at(0, (0, 0), 512));
        // From a window still there, the walk reaches the top without losing
        // its place.
        assert!(super::windows_above(hidden.id()).all(|w| w.is_some()));
        let gone = hidden.id();
        drop(hidden);
        assert!(covered_at(gone, (0, 0), 512));
        // From one that has gone, it says so at once.
        let walked: Vec<_> = super::windows_above(gone).collect();
        assert!(walked.len() == 1 && walked[0].is_none(), "{}", walked.len());
        // A window above gone mid-walk: the walk says it lost its place, not
        // that it reached the top.
        let below = Hidden::new(WINDOW_EX_STYLE(0));
        let above = Hidden::new(WINDOW_EX_STYLE(0)); // made later, so higher among the windows that aren't topmost
        let mut walk = super::windows_above(below.id());
        assert!(walk.by_ref().take(10_000).any(|w| w.is_some_and(|w| w.0 == above.0)), "the later window is above");
        drop(above);
        assert!(matches!(walk.next(), Some(None)), "lost its place, not the top");
        let (ms, w, h) = double_click_limits(0);
        assert!(ms > 0 && w >= 1 && h >= 1, "{ms} {w} {h}");
    }

    /// How long the walk takes on this desktop: run on purpose
    /// (`cargo test ... -- --ignored the_walk_is_quick --nocapture`), never
    /// in CI. Read-only; no window is made or shown, and only counts and
    /// times are printed.
    #[test]
    #[ignore = "timed on the owner's desktop, on purpose"]
    fn the_walk_is_quick() {
        // SAFETY: plain queries walking the z-order.
        let all: Vec<isize> = unsafe {
            let mut out = Vec::new();
            let mut h = GetTopWindow(None).unwrap_or_default();
            while !h.0.is_null() && out.len() < 10_000 {
                out.push(h.0 as isize);
                h = GetWindow(h, GW_HWNDNEXT).unwrap_or_default();
            }
            out
        };
        let topmost = all.iter().filter(|&&h| super::Win(super::hwnd(h)).ex() & WS_EX_TOPMOST.0 != 0).count();
        let at = super::cursor_pos().unwrap_or((0, 0));
        let time = |what: &str, n: u32, f: &mut dyn FnMut()| -> Duration {
            let (mut total, mut slowest) = (Duration::ZERO, Duration::ZERO);
            for _ in 0..n {
                let t = Instant::now();
                f();
                let d = t.elapsed();
                total += d;
                slowest = slowest.max(d);
            }
            let mean = total / n;
            println!("{what}: mean {} µs, slowest {} µs ({n} runs)", mean.as_micros(), slowest.as_micros());
            mean
        };
        println!("{} top-level windows, {topmost} of them topmost", all.len());
        // From the last topmost window, as his is: the usual walk.
        let from = all.get(topmost.saturating_sub(1)).copied().unwrap_or(0);
        println!("{} windows above the last topmost one", topmost.saturating_sub(1));
        let usual = time("covered_at from the last topmost window", 1_000, &mut || {
            std::hint::black_box(covered_at(from, at, 512));
        });
        // The longest walk: from the bottom, every window read, never stopping.
        let bottom = all.last().copied().unwrap_or(0);
        time("every window read, from the bottom up", 100, &mut || {
            for w in super::windows_above(bottom).flatten() {
                std::hint::black_box((w.visible(), w.rect(), w.ex(), w.cloaked()));
            }
        });
        let fg = super::foreground_window();
        time("the cloak alone", 1_000, &mut || {
            std::hint::black_box(super::Win(super::hwnd(fg)).cloaked());
        });
        time("foreground_busy", 1_000, &mut || {
            std::hint::black_box(super::foreground_busy());
        });
        time("double_click_limits", 1_000, &mut || {
            std::hint::black_box(double_click_limits(0));
        });
        assert!(usual < Duration::from_micros(500), "the usual walk takes {usual:?}");
    }
}
