// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// divora-capture/src/native/window.rs at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.

//! The shell's `NonRudeHWND` mark (Raymond Chen, 2025-05-22): a window
//! carrying it is never taken for a fullscreen app, so a large always-on-top
//! avatar doesn't hide the taskbar or hold back notifications. Read when the
//! window is shown, so it is set before the first show.

use std::ffi::c_void;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::UI::WindowsAndMessaging::{RemovePropW, SetPropW};

use crate::windows_ext::hwnd;

const NON_RUDE: PCWSTR = w!("NonRudeHWND");

/// Set the mark. Main thread. False for 0 or a refusal.
pub fn mark_non_rude(window: isize) -> bool {
    if window == 0 {
        return false;
    }
    // Any non-null value marks it; the shell only checks the property is there.
    let one = HANDLE(std::ptr::without_provenance_mut::<c_void>(1));
    // SAFETY: a window handle (a stale one only makes the call fail) and a
    // static wide string.
    unsafe { SetPropW(hwnd(window), NON_RUDE, Some(one)) }.is_ok()
}

/// Remove [`mark_non_rude`]'s property, releasing its atom reference. Never
/// set, or the window gone: nothing happens.
pub fn unmark_non_rude(window: isize) {
    if window == 0 {
        return;
    }
    // SAFETY: as in `mark_non_rude`. The returned value is the 1 set there,
    // not a handle to free.
    let _ = unsafe { RemovePropW(hwnd(window), NON_RUDE) };
}

#[cfg(test)]
mod tests {
    use super::{mark_non_rude, unmark_non_rude, NON_RUDE};
    use crate::windows_ext::tests::Hidden;
    use windows::Win32::UI::WindowsAndMessaging::{GetPropW, WINDOW_EX_STYLE};

    fn marked(win: &Hidden) -> bool {
        // SAFETY: our own live window and a static wide string.
        !unsafe { GetPropW(win.0, NON_RUDE) }.is_invalid()
    }

    #[test]
    fn no_window_is_nothing() {
        assert!(!mark_non_rude(0));
        unmark_non_rude(0);
    }

    /// The property goes on and comes off the real window.
    #[test]
    fn marks_and_unmarks() {
        let win = Hidden::new(WINDOW_EX_STYLE(0));
        assert!(!marked(&win));
        assert!(mark_non_rude(win.id()));
        assert!(marked(&win));
        unmark_non_rude(win.id());
        assert!(!marked(&win));
        // Twice is fine.
        unmark_non_rude(win.id());
    }
}
