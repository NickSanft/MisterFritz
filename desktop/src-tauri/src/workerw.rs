//! The desktop's wallpaper layer, and the monitors to draw on it.
//!
//! Plumbing from LorraineAtelier's `workerw.rs` (the owner's own app) at
//! d62b7ec: ask Progman to make the wallpaper layer's WorkerW (the
//! undocumented `0x052C`, wParam 0xD, lParam 1, its answer never trusted),
//! then find it: on Windows 11 a child of Progman, on Windows 10 the
//! top-level WorkerW with no `SHELLDLL_DefView` (the icons) in it, and
//! failing both Progman itself. A window put there is a child of the shell,
//! below the icons. The process is per-monitor-v2 DPI aware
//! (`windows_ext::set_dpi_awareness`), so monitor rectangles are physical
//! pixels as they come, never scaled.
//!
//! Changed on the way: the answer is kept as Lorraine kept it, so Progman is
//! not sent the message on every look, but Progman itself (the fallback) is
//! kept only for a while: a WorkerW Explorer makes later (it is restarting,
//! say) is found then. Monitors carry their bounds as Fritz's `Bounds`.
//! Lorraine's casting overlay, foreground and process-matching code isn't
//! here; this app has its own where it needs them.

use std::sync::{Mutex, PoisonError};

use windows::core::{w, BOOL, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO,
};
use windows::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, FindWindowExW, FindWindowW, GetClassNameW, GetSystemMetrics, GetWindowRect,
    IsWindow, IsWindowVisible, SendMessageTimeoutW, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN,
    SMTO_NORMAL,
};

use crate::windows_ext::{hwnd, Bounds};

/// Progman's "make the wallpaper layer" message, and how long it may take.
const SPAWN_WORKERW: u32 = 0x052C;
const SPAWN_WAIT_MS: u32 = 1000;
/// `MONITORINFO.dwFlags`' primary-monitor bit.
const MONITORINFOF_PRIMARY: u32 = 1;
/// How many of the wallpaper watch's ticks Progman (the fallback) is kept
/// for before Progman is asked again: about six seconds.
const FALLBACK_LOOKS: u32 = 10;

/// A monitor: where it is, in physical pixels, its effective DPI, and
/// whether it is the primary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Monitor {
    pub bounds: Bounds,
    pub dpi: u32,
    pub primary: bool,
}

/// Where the wallpaper layer was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    /// A WorkerW: Progman's child (Windows 11) or the one beside the icons'
    /// (Windows 10).
    Worker(isize),
    /// Progman itself, when no WorkerW big enough was there.
    Progman(isize),
}

impl Found {
    pub fn window(self) -> isize {
        match self {
            Self::Worker(h) | Self::Progman(h) => h,
        }
    }
}

/// Which window holds the wallpaper, from what was found: Progman's own
/// WorkerW child if it covers at least `min_area` (Windows 11), else the
/// biggest top-level WorkerW without the icons in it, if it does (Windows
/// 10), else Progman. Areas are in square physical pixels. Pure.
pub fn choose(progman: isize, child: Option<(isize, i64)>, top_level: Option<(isize, i64)>, min_area: i64) -> Found {
    let big = |c: Option<(isize, i64)>| c.filter(|&(h, area)| h != 0 && area >= min_area).map(|(h, _)| h);
    big(child)
        .or_else(|| big(top_level))
        .map_or(Found::Progman(progman), Found::Worker)
}

/// The least a wallpaper layer may cover: a quarter of the virtual screen.
pub fn min_area(virtual_width: i32, virtual_height: i32) -> i64 {
    i64::from(virtual_width.max(0)) * i64::from(virtual_height.max(0)) / 4
}

fn area(h: HWND) -> i64 {
    let mut r = RECT::default();
    // SAFETY: a window handle that may be stale (the call then fails) and an
    // out-struct.
    if unsafe { GetWindowRect(h, &raw mut r) }.is_ok() {
        i64::from(r.right - r.left) * i64::from(r.bottom - r.top)
    } else {
        0
    }
}

fn class_of(h: HWND) -> String {
    let mut buf = [0u16; 64];
    // SAFETY: a window handle and a buffer it writes at most 64 units into.
    let n = unsafe { GetClassNameW(h, &mut buf) };
    String::from_utf16_lossy(&buf[..usize::try_from(n).unwrap_or(0)])
}

/// The biggest visible top-level WorkerW without the icons in it.
fn top_level_worker() -> Option<(isize, i64)> {
    unsafe extern "system" fn each(h: HWND, data: LPARAM) -> BOOL {
        // SAFETY: `data` is the Option below, alive for the whole enumeration.
        let best = unsafe { &mut *(data.0 as *mut Option<(isize, i64)>) };
        // SAFETY: plain queries on a window handle Windows just gave us.
        let visible = unsafe { IsWindowVisible(h) }.as_bool();
        if visible && class_of(h) == "WorkerW" {
            // SAFETY: as above; a null class name and no title.
            let icons = unsafe { FindWindowExW(Some(h), None, w!("SHELLDLL_DefView"), PCWSTR::null()) };
            if !matches!(icons, Ok(d) if !d.0.is_null()) {
                let a = area(h);
                if best.is_none_or(|(_, b)| a > b) {
                    *best = Some((h.0 as isize, a));
                }
            }
        }
        BOOL(1)
    }
    let mut best: Option<(isize, i64)> = None;
    // SAFETY: a callback that only writes `best`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(each), LPARAM(&raw mut best as isize)) };
    best
}

/// Ask Progman for the wallpaper layer and find it; `None` only when there
/// is no Progman (Explorer isn't running). Can wait up to a second on a
/// hung Explorer: never on the main thread.
pub fn find() -> Option<Found> {
    // SAFETY: plain Win32 calls; every handle may be stale, which only makes
    // a call fail.
    unsafe {
        let progman = FindWindowW(w!("Progman"), PCWSTR::null()).ok().filter(|h| !h.0.is_null())?;
        let mut answer: usize = 0;
        // Harmless when the layer is already there; the answer is ignored.
        let _ = SendMessageTimeoutW(
            progman,
            SPAWN_WORKERW,
            WPARAM(0xD),
            LPARAM(0x1),
            SMTO_NORMAL,
            SPAWN_WAIT_MS,
            Some(&raw mut answer),
        );
        let child = FindWindowExW(Some(progman), None, w!("WorkerW"), PCWSTR::null())
            .ok()
            .filter(|h| !h.0.is_null())
            .map(|h| (h.0 as isize, area(h)));
        let floor = min_area(GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN));
        let top = if child.is_some_and(|(_, a)| a >= floor) { None } else { top_level_worker() };
        Some(choose(progman.0 as isize, child, top, floor))
    }
}

/// Whether `h` is a window now.
pub fn is_window(h: isize) -> bool {
    // SAFETY: a plain query; a stale handle answers false.
    h != 0 && unsafe { IsWindow(Some(hwnd(h))) }.as_bool()
}

/// The last answer, and how many more looks it is good for if it is the
/// fallback (unused for a WorkerW).
static KEPT: Mutex<Option<(Found, u32)>> = Mutex::new(None);

/// Whether a kept answer is still good: it is a window, and if it is the
/// fallback, it hasn't been kept too long (`left` looks remain). Pure.
pub fn keep(kept: Found, alive: bool, left: u32) -> bool {
    alive && (matches!(kept, Found::Worker(_)) || left > 0)
}

/// [`find`], kept between looks while it stays a window, as Lorraine kept
/// it (so Progman isn't asked on every look); Progman itself is kept for
/// [`FALLBACK_LOOKS`] of the watch's ticks at most (`ticked`: this look is
/// one; a look a poke adds counts nothing).
pub fn layer(ticked: bool) -> Option<Found> {
    let mut kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((found, left)) = kept.as_mut() {
        if keep(*found, is_window(found.window()), *left) {
            if ticked {
                *left = left.saturating_sub(1);
            }
            return Some(*found);
        }
    }
    let found = find();
    *kept = found.map(|f| (f, FALLBACK_LOOKS));
    found
}

/// Where `parent`'s client area starts on the screen: the wallpaper layer
/// need not start at (0, 0) when a monitor is left of or above the primary.
pub fn client_origin(parent: isize) -> Option<(i32, i32)> {
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: a window handle that may be stale (the call then fails) and a
    // point.
    unsafe { ClientToScreen(hwnd(parent), &raw mut p) }
        .as_bool()
        .then_some((p.x, p.y))
}

/// Every monitor, in Windows' order, which is the slots' order. If Windows
/// lists none, the virtual screen as one primary at 96 DPI, as Lorraine did.
pub fn monitors() -> Vec<Monitor> {
    unsafe extern "system" fn each(mon: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
        // SAFETY: `data` is the Vec below, alive for the whole enumeration.
        let out = unsafe { &mut *(data.0 as *mut Vec<Monitor>) };
        let mut info = MONITORINFO {
            cbSize: u32::try_from(size_of::<MONITORINFO>()).unwrap_or(0),
            ..Default::default()
        };
        // SAFETY: a monitor handle Windows just gave us and a sized out-struct.
        if unsafe { GetMonitorInfoW(mon, &raw mut info) }.as_bool() {
            let (mut dx, mut dy) = (96u32, 96u32);
            // SAFETY: as above, and two out-pointers.
            let _ = unsafe { GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &raw mut dx, &raw mut dy) };
            let r = info.rcMonitor;
            out.push(Monitor {
                bounds: Bounds { left: r.left, top: r.top, right: r.right, bottom: r.bottom },
                dpi: dx.max(48),
                primary: info.dwFlags & MONITORINFOF_PRIMARY != 0,
            });
        }
        BOOL(1)
    }
    let mut out: Vec<Monitor> = Vec::new();
    // SAFETY: a callback that only pushes into `out`, which outlives the call.
    let _ = unsafe { EnumDisplayMonitors(None, None, Some(each), LPARAM(&raw mut out as isize)) };
    if out.is_empty() {
        use windows::Win32::UI::WindowsAndMessaging::{SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN};
        // SAFETY: plain metric queries.
        let (x, y, w, h) = unsafe {
            (
                GetSystemMetrics(SM_XVIRTUALSCREEN),
                GetSystemMetrics(SM_YVIRTUALSCREEN),
                GetSystemMetrics(SM_CXVIRTUALSCREEN),
                GetSystemMetrics(SM_CYVIRTUALSCREEN),
            )
        };
        out.push(Monitor {
            bounds: Bounds { left: x, top: y, right: x + w, bottom: y + h },
            dpi: 96,
            primary: true,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progmans_own_worker_comes_first() {
        let floor = min_area(1920, 1080);
        assert_eq!(choose(1, Some((2, floor)), Some((3, floor * 4)), floor), Found::Worker(2));
    }

    #[test]
    fn a_small_child_gives_way_to_the_top_level_worker() {
        let floor = min_area(1920, 1080);
        assert_eq!(choose(1, Some((2, floor - 1)), Some((3, floor)), floor), Found::Worker(3));
        assert_eq!(choose(1, None, Some((3, floor)), floor), Found::Worker(3));
    }

    #[test]
    fn progman_when_nothing_big_enough_is_there() {
        let floor = min_area(1920, 1080);
        assert_eq!(choose(1, Some((2, 10)), Some((3, 10)), floor), Found::Progman(1));
        assert_eq!(choose(1, None, None, floor), Found::Progman(1));
        assert_eq!(choose(1, Some((0, floor)), None, floor), Found::Progman(1), "a null child is none");
        assert_eq!(Found::Progman(1).window(), 1);
        assert_eq!(Found::Worker(2).window(), 2);
    }

    #[test]
    fn the_floor_is_a_quarter_of_the_virtual_screen() {
        assert_eq!(min_area(3840, 1080), 3840 * 1080 / 4);
        assert_eq!(min_area(-5, 100), 0);
        // Two 8K monitors side by side: no overflow.
        assert_eq!(min_area(15360, 4320), 15360_i64 * 4320 / 4);
    }

    #[test]
    fn an_answer_is_kept_while_it_is_good() {
        assert!(keep(Found::Worker(5), true, 0), "a live WorkerW, kept");
        assert!(!keep(Found::Worker(5), false, 0), "a dead one, found again");
        assert!(keep(Found::Progman(5), true, 3), "Progman, for a while");
        assert!(!keep(Found::Progman(5), true, 0), "Progman, asked again");
        assert!(!keep(Found::Progman(5), false, 3), "a dead Progman, found again");
    }

    #[test]
    fn the_monitors_are_listed_with_their_scale() {
        let all = monitors();
        assert!(!all.is_empty());
        for m in &all {
            assert!(m.bounds.width() > 0 && m.bounds.height() > 0, "{m:?}");
            assert!(m.dpi >= 48, "{m:?}");
        }
        assert!(all.iter().filter(|m| m.primary).count() <= 1);
    }
}
