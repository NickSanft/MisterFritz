//! The study scene's host windows: one plain child window per monitor in
//! the desktop's wallpaper layer, below the icons, that a wallpaper window's
//! WebView2 controller is moved into (`wallpaper.rs`).
//!
//! Plumbing from LorraineAtelier's `host.rs` (the owner's own app) at
//! d62b7ec: a window put in the wallpaper layer dies when Explorer restarts,
//! so the webview never lives there itself; it lives in its own hidden
//! window and only its controller is moved into a cheap host like this. An
//! Explorer restart kills the host alone, and a new window and host replace
//! them. Changed on the way: the class is Fritz's own, and the colour it
//! paints before the page does is the owner's mood's `--bg` (Lorraine's was
//! a fixed colour of hers), set by the wallpaper as the theme changes; it
//! clips its children, so it never paints over the page; it sits below its
//! siblings, so in Progman itself (the layer's fallback) the icons stay
//! above it.
//!
//! [`create`], [`place`] and [`destroy`] run on the main thread only (a host
//! is its maker's, whose thread pumps its messages; DestroyWindow fails from
//! any other). The rest only ask or compute, from any thread. A host tells
//! its parent nothing as it comes and goes (`WS_EX_NOPARENTNOTIFY`): that
//! would be a message to Explorer, sent and waited for with no time limit,
//! from the main thread.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{CreateSolidBrush, DeleteObject, FillRect, HDC, HGDIOBJ};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, GetClientRect, GetParent, RegisterClassW,
    SetWindowPos, HWND_BOTTOM, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER,
    WM_ERASEBKGND, WNDCLASSW, WS_CHILD, WS_CLIPCHILDREN, WS_EX_NOPARENTNOTIFY, WS_VISIBLE,
};

use crate::windows_ext::{hwnd, Bounds};

const CLASS: PCWSTR = w!("FritzStudySceneHost");
static REGISTERED: AtomicBool = AtomicBool::new(false);
/// What a host paints before the page does, as a COLORREF (0x00BBGGRR).
static BACKDROP: AtomicU32 = AtomicU32::new(0);

/// `(r, g, b)` as a COLORREF. Pure.
pub fn colorref((r, g, b): (u8, u8, u8)) -> u32 {
    u32::from(r) | (u32::from(g) << 8) | (u32::from(b) << 16)
}

/// The colour hosts paint from now on, until the page covers them.
pub fn set_backdrop(rgb: (u8, u8, u8)) {
    BACKDROP.store(colorref(rgb), Ordering::SeqCst);
}

unsafe extern "system" fn wndproc(h: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_ERASEBKGND {
        let erased = crate::logs::guard("study scene host", || {
            let mut r = RECT::default();
            // SAFETY: our own window, and for WM_ERASEBKGND wparam is the
            // DC Windows lends for this call.
            unsafe {
                if GetClientRect(h, &raw mut r).is_ok() {
                    let brush = CreateSolidBrush(COLORREF(BACKDROP.load(Ordering::SeqCst)));
                    if !brush.is_invalid() {
                        FillRect(HDC(wparam.0 as *mut _), &raw const r, brush);
                        let _ = DeleteObject(HGDIOBJ(brush.0));
                    }
                }
            }
        });
        if erased.is_some() {
            return LRESULT(1);
        }
    }
    // SAFETY: the default procedure, with what Windows sent.
    unsafe { DefWindowProcW(h, msg, wparam, lparam) }
}

fn register() -> Option<HINSTANCE> {
    // SAFETY: this module's own handle.
    let hinst: HINSTANCE = unsafe { GetModuleHandleW(None) }.ok().map(HINSTANCE::from)?;
    if !REGISTERED.swap(true, Ordering::SeqCst) {
        let class = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: hinst,
            lpszClassName: CLASS,
            ..Default::default()
        };
        // SAFETY: a class description that lives through the call. A second
        // registration (it can't happen) only fails.
        unsafe { RegisterClassW(&raw const class) };
    }
    Some(hinst)
}

/// Where a host for `monitor` goes in a parent whose client area starts at
/// `origin` on the screen: (x, y, width, height). Pure.
pub fn placement(monitor: Bounds, origin: (i32, i32)) -> (i32, i32, i32, i32) {
    (
        monitor.left - origin.0,
        monitor.top - origin.1,
        monitor.width(),
        monitor.height(),
    )
}

/// A host over `monitor`, a child of `parent` (the wallpaper layer) whose
/// client area starts at `origin`, below its siblings; `None` if Windows
/// won't make it.
pub fn create(parent: isize, monitor: Bounds, origin: (i32, i32)) -> Option<isize> {
    let hinst = register()?;
    let (x, y, width, height) = placement(monitor, origin);
    // SAFETY: a registered class, a parent that may be stale (creation then
    // fails), no menu or creation data.
    let h = unsafe {
        CreateWindowExW(
            WS_EX_NOPARENTNOTIFY,
            CLASS,
            PCWSTR::null(),
            WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN,
            x,
            y,
            width,
            height,
            Some(hwnd(parent)),
            None,
            Some(hinst),
            None,
        )
    }
    .ok()
    .filter(|h| !h.0.is_null())?;
    // SAFETY: our own new window.
    let _ = unsafe { SetWindowPos(h, Some(HWND_BOTTOM), 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE) };
    Some(h.0 as isize)
}

/// Move a host back over `monitor`, in a parent whose client area starts
/// at `origin`, without changing its place among its siblings; whether
/// Windows moved it.
pub fn place(host: isize, monitor: Bounds, origin: (i32, i32)) -> bool {
    let (x, y, width, height) = placement(monitor, origin);
    // SAFETY: a window of this thread's, or a stale handle (the call then
    // fails).
    unsafe { SetWindowPos(hwnd(host), None, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE) }.is_ok()
}

/// Destroy a host; nothing if it is gone already.
pub fn destroy(host: isize) {
    if host != 0 {
        // SAFETY: a window this thread made, or a stale handle (it then fails).
        let _ = unsafe { DestroyWindow(hwnd(host)) };
    }
}

/// A host's client size in physical pixels; (0, 0) if it is gone.
pub fn client_size(host: isize) -> (i32, i32) {
    let mut r = RECT::default();
    // SAFETY: a window handle that may be stale (the call then fails) and an
    // out-struct.
    if unsafe { GetClientRect(hwnd(host), &raw mut r) }.is_ok() {
        (r.right - r.left, r.bottom - r.top)
    } else {
        (0, 0)
    }
}

/// The window a host is a child of; 0 if it is gone.
pub fn parent_of(host: isize) -> isize {
    // SAFETY: a plain query; a stale handle answers an error.
    unsafe { GetParent(hwnd(host)) }.map_or(0, |p| p.0 as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::{GetWindowLongPtrW, GWL_EXSTYLE, WINDOW_EX_STYLE, WS_POPUP};

    /// A hidden top-level parent of the predefined STATIC class, destroyed
    /// on drop.
    struct Parent(isize);

    impl Parent {
        fn new() -> Self {
            // SAFETY: a predefined class, no parent, menu or creation data.
            let h = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("STATIC"),
                    w!("fritz host test"),
                    WS_POPUP,
                    -20_000,
                    0,
                    1200,
                    900,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .unwrap();
            Self(h.0 as isize)
        }
    }

    impl Drop for Parent {
        fn drop(&mut self) {
            // SAFETY: our own window, destroyed once on the thread that made it.
            let _ = unsafe { DestroyWindow(hwnd(self.0)) };
        }
    }

    #[test]
    fn a_host_is_placed_from_the_parents_origin() {
        let m = Bounds { left: -1920, top: 120, right: 0, bottom: 1200 };
        assert_eq!(placement(m, (-1920, 0)), (0, 120, 1920, 1080));
        assert_eq!(placement(m, (0, 0)), (-1920, 120, 1920, 1080));
    }

    #[test]
    fn colours_are_windows_order() {
        assert_eq!(colorref((0x08, 0x07, 0x0b)), 0x000b_0708);
        assert_eq!(colorref((0xff, 0, 0)), 0xff);
    }

    #[test]
    fn a_host_fills_its_monitor_as_a_child_of_the_layer() {
        let parent = Parent::new();
        let origin = crate::workerw::client_origin(parent.0).unwrap();
        let monitor = Bounds {
            left: origin.0 + 100,
            top: origin.1 + 50,
            right: origin.0 + 900,
            bottom: origin.1 + 650,
        };
        let host = create(parent.0, monitor, origin).expect("a host");
        assert!(crate::workerw::is_window(host));
        assert_eq!(client_size(host), (800, 600), "no scaling: physical pixels");
        assert_eq!(parent_of(host), parent.0);
        assert_eq!(crate::windows_ext::window_bounds(host), Some(monitor));
        // SAFETY: a plain query on our own window.
        let ex = unsafe { GetWindowLongPtrW(hwnd(host), GWL_EXSTYLE) } as u32;
        assert_ne!(ex & WS_EX_NOPARENTNOTIFY.0, 0, "it tells Explorer nothing");
        // Moved away (as wry moves it), then put back.
        let corner = Bounds { left: origin.0, top: origin.1, right: origin.0 + 300, bottom: origin.1 + 200 };
        assert!(place(host, corner, origin));
        assert_eq!(crate::windows_ext::window_bounds(host), Some(corner));
        assert!(place(host, monitor, origin));
        assert_eq!(crate::windows_ext::window_bounds(host), Some(monitor));
        destroy(host);
        assert!(!crate::workerw::is_window(host));
        assert_eq!(client_size(host), (0, 0));
        assert_eq!(parent_of(host), 0);
        destroy(host); // twice is nothing
        assert!(!place(host, monitor, origin));
    }

    #[test]
    fn no_host_without_a_parent() {
        let parent = Parent::new();
        let gone = parent.0;
        drop(parent);
        assert_eq!(create(gone, Bounds { left: 0, top: 0, right: 10, bottom: 10 }, (0, 0)), None);
    }
}
