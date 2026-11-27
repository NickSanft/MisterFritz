// `Keyboard`'s first three methods follow `Native`'s, and
// `the_shell_never_calls_set_focus` is its `the_overlay_never_calls_set_focus`
// run over every module, in DivoraVoiceMod (MIT, Copyright (c) 2026 Divora
// Contributors), src-tauri/src/overlay_windows.rs at 62ed6a5. See
// desktop/LICENSES/DivoraVoiceMod.txt. The rest is Fritz's own.

//! Who has the keyboard, and who gets it back. Arrange takes the keyboard
//! and hands it back on Lock, and the prompt takes it to be typed in; every
//! window of Fritz's that ever takes it saves who had it first and follows
//! these same rules when it is done. [`Desk`] is Windows' own answers, the
//! one place they are asked.
//!
//! The keyboard is taken with plain `SetForegroundWindow`, never tauri's or
//! tao's focus call, which taps Alt into the owner's app when Windows
//! refuses (`the_shell_never_calls_set_focus`).

use tauri::Manager;

use crate::windows_ext;

/// The desktop calls the hand-back rules make. A trait so the rules are
/// tested without a window.
pub(crate) trait Keyboard {
    fn foreground(&self) -> isize;
    fn try_foreground(&self, hwnd: isize) -> bool;
    fn can_take_foreground(&self, hwnd: isize) -> bool;
    /// A window of this process (the tray's hidden window, the avatar, the
    /// control window).
    fn owned_by_us(&self, hwnd: isize) -> bool;
    /// The control window (tray > Settings), or 0: the one window of ours
    /// the keyboard is handed back to after Arrange.
    fn settings_window(&self) -> isize;
    /// The app in use: the top ordinary app window that isn't `skip`
    /// (always-on-top ones only if there is no other), else the desktop.
    /// Where the keyboard goes when there's nobody saved to give it back to.
    /// Our own windows never count, except `also` (the control window, or
    /// 0): visible and not minimised, it is an app like any other.
    fn next_app_window(&self, skip: isize, also: isize) -> isize;
}

/// The window that should get the keyboard back after Arrange: `fg` if it
/// is someone else's usable window, else the app beneath in z-order. Arrange
/// from the tray finds the tray's own hidden window in the foreground (the
/// menu put it there), which can never take the keyboard back.
///
/// Of our own windows, only the control window counts, both as `fg` and in
/// the z-order: Arrange started from it, or from the tray while it was in
/// use, hands the keyboard back to it, not to the app beneath (which would
/// come up over it).
pub(crate) fn return_target(fg: isize, ours: isize, n: &impl Keyboard) -> isize {
    let settings = n.settings_window();
    let ours_but_returnable = |w: isize| !n.owned_by_us(w) || (w != 0 && w == settings);
    if fg != 0 && fg != ours && ours_but_returnable(fg) && n.can_take_foreground(fg) {
        fg
    } else {
        n.next_app_window(ours, settings)
    }
}

/// Who gets the keyboard from `ours` when it is done: the window saved
/// before it took it, if that can still take it, else the app in use (the
/// desktop, with none open). `saved` is 0 when nobody was saved.
pub(crate) fn hand_back(saved: isize, ours: isize, n: &impl Keyboard) -> isize {
    if saved != 0 && n.can_take_foreground(saved) {
        saved
    } else {
        n.next_app_window(ours, n.settings_window())
    }
}

/// Windows' own answers, with the control window's HWND (0 when there is
/// none): the one real [`Keyboard`].
pub(crate) struct Desk(pub(crate) isize);

impl Desk {
    /// The desktop now, with the control window the app declares.
    pub(crate) fn of(app: &tauri::AppHandle) -> Self {
        let settings = app
            .get_webview_window(crate::CONTROL_LABEL)
            .and_then(|w| w.hwnd().ok())
            .map_or(0, |h| h.0 as isize);
        Self(settings)
    }
}

impl Keyboard for Desk {
    fn foreground(&self) -> isize {
        windows_ext::foreground_window()
    }

    fn try_foreground(&self, hwnd: isize) -> bool {
        windows_ext::try_foreground(hwnd)
    }

    fn can_take_foreground(&self, hwnd: isize) -> bool {
        windows_ext::can_take_foreground(hwnd)
    }

    fn owned_by_us(&self, hwnd: isize) -> bool {
        windows_ext::owned_by_us(hwnd)
    }

    fn settings_window(&self) -> isize {
        self.0
    }

    fn next_app_window(&self, skip: isize, also: isize) -> isize {
        windows_ext::next_app_window(skip, also)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeSet;

    use super::{hand_back, return_target, Keyboard};

    const OURS: isize = 7;
    const GAME: isize = 42;
    /// The tray's hidden window: ours, and never able to take the keyboard.
    const TRAY: isize = 9;
    /// The control window.
    const SETTINGS: isize = 30;
    /// The app beneath in z-order, which `next_app_window` answers.
    const BENEATH: isize = 77;

    /// A desktop where `next_app_window` is always the app beneath, and
    /// remembers what it was asked.
    struct Desk {
        takeable: Vec<isize>,
        ours: Vec<isize>,
        settings: isize,
        asked: std::cell::RefCell<Vec<(isize, isize)>>,
    }

    fn desk(takeable: &[isize], settings: isize) -> Desk {
        Desk {
            takeable: takeable.to_vec(),
            ours: vec![OURS, TRAY, SETTINGS],
            settings,
            asked: std::cell::RefCell::default(),
        }
    }

    impl Keyboard for Desk {
        fn foreground(&self) -> isize {
            unreachable!("the rules are given who is in front")
        }
        fn try_foreground(&self, _hwnd: isize) -> bool {
            unreachable!("the rules only choose")
        }
        fn can_take_foreground(&self, hwnd: isize) -> bool {
            self.takeable.contains(&hwnd)
        }
        fn owned_by_us(&self, hwnd: isize) -> bool {
            self.ours.contains(&hwnd)
        }
        fn settings_window(&self) -> isize {
            self.settings
        }
        fn next_app_window(&self, skip: isize, also: isize) -> isize {
            self.asked.borrow_mut().push((skip, also));
            BENEATH
        }
    }

    /// Someone else's usable window is the one given back; ours, the
    /// tray's, nobody's or one that can't take it any more give the app
    /// beneath; the control window counts only while it can take it.
    #[test]
    fn return_target_table() {
        let usable = [GAME, BENEATH, SETTINGS, TRAY, OURS];
        for (fg, takeable, settings, want) in [
            (GAME, &usable[..], SETTINGS, GAME),
            (GAME, &[BENEATH][..], SETTINGS, BENEATH),
            (OURS, &usable[..], SETTINGS, BENEATH),
            (TRAY, &usable[..], SETTINGS, BENEATH),
            (0, &usable[..], SETTINGS, BENEATH),
            // Nobody in front is never saved, even on a desk that says 0
            // could take the keyboard.
            (0, &[0, BENEATH][..], SETTINGS, BENEATH),
            (SETTINGS, &usable[..], SETTINGS, SETTINGS),
            (SETTINGS, &[BENEATH][..], SETTINGS, BENEATH),
            (SETTINGS, &usable[..], 0, BENEATH),
        ] {
            let d = desk(takeable, settings);
            let case = format!("fg {fg} takeable {takeable:?} settings {settings}");
            assert_eq!(return_target(fg, OURS, &d), want, "{case}");
            if want == BENEATH {
                assert_eq!(*d.asked.borrow(), [(OURS, settings)], "{case}");
            }
        }
        // The window giving the keyboard back is never handed it, even on a
        // desk that doesn't count it as ours.
        let d = Desk { ours: vec![], ..desk(&usable, SETTINGS) };
        assert_eq!(return_target(OURS, OURS, &d), BENEATH);
    }

    /// The saved window if it can still take the keyboard; else the app in
    /// use, passing over `ours` and counting the control window.
    #[test]
    fn hand_back_table() {
        for (saved, takeable, want) in [
            (GAME, &[GAME, BENEATH][..], GAME),
            (GAME, &[BENEATH][..], BENEATH),
            (SETTINGS, &[SETTINGS][..], SETTINGS),
            (0, &[0, GAME, BENEATH][..], BENEATH),
        ] {
            let d = desk(takeable, SETTINGS);
            let case = format!("saved {saved} takeable {takeable:?}");
            assert_eq!(hand_back(saved, OURS, &d), want, "{case}");
            let asked: &[(isize, isize)] = if want == BENEATH { &[(OURS, SETTINGS)] } else { &[] };
            assert_eq!(*d.asked.borrow(), asked, "{case}");
        }
    }

    /// Every module of the shell, by name, with its source (the mouse
    /// hook's scans read it too).
    pub(crate) const SOURCES: [(&str, &str); 28] = [
        ("audio_in", include_str!("audio_in.rs")),
        ("avatar_state", include_str!("avatar_state.rs")),
        ("companion", include_str!("companion.rs")),
        ("cover", include_str!("cover.rs")),
        ("cursor", include_str!("cursor.rs")),
        ("foreground_log", include_str!("foreground_log.rs")),
        ("hit_shape", include_str!("hit_shape.rs")),
        ("hook", include_str!("hook.rs")),
        ("host", include_str!("host.rs")),
        ("hotkeys", include_str!("hotkeys.rs")),
        ("keyboard", include_str!("keyboard.rs")),
        ("kokoro_tts", include_str!("kokoro_tts.rs")),
        ("lib", include_str!("lib.rs")),
        ("logs", include_str!("logs.rs")),
        ("main", include_str!("main.rs")),
        ("non_rude", include_str!("non_rude.rs")),
        ("overlay_machine", include_str!("overlay_machine.rs")),
        ("overlay_windows", include_str!("overlay_windows.rs")),
        ("popup", include_str!("popup.rs")),
        ("power", include_str!("power.rs")),
        ("prompt", include_str!("prompt.rs")),
        ("settings", include_str!("settings.rs")),
        ("tray", include_str!("tray.rs")),
        ("tts_phonemize", include_str!("tts_phonemize.rs")),
        ("tts_tokens", include_str!("tts_tokens.rs")),
        ("wallpaper", include_str!("wallpaper.rs")),
        ("windows_ext", include_str!("windows_ext.rs")),
        ("workerw", include_str!("workerw.rs")),
    ];

    /// A source scan, the honest limit without a window: tauri's and tao's
    /// focus call taps Alt into the owner's app when Windows refuses, so no
    /// window of Fritz's may make it. The needles are built from halves so
    /// this test doesn't find itself, and are calls (a method or a path), so
    /// its own name doesn't either.
    #[test]
    fn the_shell_never_calls_set_focus() {
        for (name, src) in SOURCES {
            for needle in [concat!(".set_", "focus("), concat!("::set_", "focus(")] {
                assert!(
                    !src.contains(needle),
                    "{name}.rs takes the keyboard with try_foreground, never {needle}"
                );
            }
        }
    }

    /// The scan covers every module lib.rs declares, so a new one can't
    /// slip past it.
    #[test]
    fn the_scan_covers_every_module() {
        let lib = SOURCES.iter().find(|(n, _)| *n == "lib").unwrap().1;
        let declared: BTreeSet<&str> = lib
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                let l = l.strip_prefix("pub(crate) ").or_else(|| l.strip_prefix("pub ")).unwrap_or(l);
                l.strip_prefix("mod ")?.strip_suffix(';')
            })
            .chain(["lib", "main"])
            .collect();
        let scanned: BTreeSet<&str> = SOURCES.iter().map(|(n, _)| *n).collect();
        assert_eq!(scanned, declared);
    }
}
