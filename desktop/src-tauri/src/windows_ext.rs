//! Win32 for the avatar window: the foreground without injected input, Alt-Tab,
//! the monitor Fritz stands on, and whether a fullscreen app covers it.
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

use windows::core::BOOL;
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_CLOAKED};
use windows::Win32::Graphics::Gdi::{
    EnumDisplayMonitors, GetMonitorInfoW, MonitorFromPoint, HDC, HMONITOR, MONITORINFO,
    MONITOR_DEFAULTTOPRIMARY,
};
use windows::Win32::UI::HiDpi::{
    GetDpiForMonitor, SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    MDT_EFFECTIVE_DPI,
};
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetClassNameW, GetForegroundWindow, GetShellWindow, GetTopWindow, GetWindow,
    GetWindowLongPtrW, GetWindowRect, GetWindowThreadProcessId, IsIconic, IsWindow,
    IsWindowVisible, IsZoomed, SetForegroundWindow, SetLayeredWindowAttributes, SetWindowLongPtrW,
    SetWindowPos, WindowFromPoint, GA_ROOT, GWL_EXSTYLE, GW_HWNDNEXT, GW_OWNER, LWA_ALPHA,
    STYLESTRUCT, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOOWNERZORDER, SWP_NOSIZE,
    SWP_NOZORDER, WM_NCDESTROY, WM_STYLECHANGING, WS_EX_APPWINDOW, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT,
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
const PIN_ID: usize = 0x4652_4954;

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

/// The primary monitor's work area (the desktop less the taskbar) and its
/// effective DPI, for Fritz's first placement.
pub fn primary_work_area() -> Option<(Bounds, u32)> {
    // SAFETY: a point and a flag; the primary monitor always exists.
    let mon = unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) };
    let info = monitor_info(mon)?;
    let (mut dx, mut dy) = (96u32, 96u32);
    // SAFETY: a monitor handle and two out-pointers.
    let _ = unsafe { GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &raw mut dx, &raw mut dy) };
    Some((Bounds::of(info.rcWork), dx.max(48)))
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
/// covers all of it and is not merely maximised. Lorraine's test (covers the
/// monitor) assumed a maximised window leaves the taskbar showing; with an
/// auto-hiding taskbar the work area is the whole monitor, so a maximised
/// editor would send Fritz away all day.
pub fn fullscreen_on(window: Bounds, monitor: Bounds, maximized: bool) -> bool {
    !maximized && covers(window, monitor)
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
        let full = GetWindowRect(top, &raw mut w).is_ok()
            && fullscreen_on(Bounds::of(w), m, IsZoomed(top).as_bool());
        full.then(|| format!("{class} (pid {pid})"))
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

/// The app in use, from the top of the z-order, that isn't `skip`; failing
/// any, the desktop (`GetShellWindow`), so a Lock always has somewhere to
/// hand the keyboard. 0 only with no shell at all.
pub fn next_app_window(skip: isize) -> isize {
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
                    owned_by_us(id),
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

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{
        apply_toolwindow, can_take_foreground, covers, foreground_window, fullscreen_on, hwnd,
        intersects, is_app_window, is_shell_class, make_layered_opaque, monitors, next_app_window,
        owned_by_us, pick_app, pin_toolwindow, settle, toolwindow_style, touching, try_foreground,
        window_bounds, Bounds,
    };
    use windows::core::w;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, GetWindowLongPtrW, SetWindowLongPtrW, GWL_EXSTYLE,
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
        let to = next_app_window(win.id());
        assert_ne!(to, 0);
        assert_ne!(to, win.id());
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
