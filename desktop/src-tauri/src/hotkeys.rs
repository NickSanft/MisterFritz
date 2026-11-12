//! The global hotkeys, from `settings.hotkeys`: Arrange and Hide/Show in
//! this PR (prompt and push-to-talk arrive with PRs 4 and 8). Read once at
//! start; a key changed in the settings takes effect on the next start.
//!
//! A binding that doesn't parse, or that another of ours already has, falls
//! back to its default (and says so) rather than leaving the action without
//! a key.

use std::str::FromStr;
use std::sync::Mutex;

use tauri::AppHandle;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};

use crate::overlay_machine::HOTKEY_ARRANGE;
use crate::settings::Hotkeys;

/// The id the park hotkey is handled under (the machine's own is Arrange's).
pub const PARK: &str = "avatar:park";

/// The registered bindings, for the handler to tell which fired.
static BOUND: Mutex<Vec<(Shortcut, &'static str)>> = Mutex::new(Vec::new());

/// An accelerator as the settings hold it (`"Ctrl+Alt+A"`).
pub fn parse(accel: &str) -> Result<Shortcut, String> {
    Shortcut::from_str(accel.trim()).map_err(|e| format!("{accel:?} isn't a hotkey: {e}"))
}

/// The bindings to register: `(shortcut, id, accelerator)`, each from the
/// settings or, when that one can't be used, the default. Pure, for tests.
pub fn bindings(h: &Hotkeys) -> Vec<(Shortcut, &'static str, String)> {
    let defaults = Hotkeys::default();
    let wanted = [
        (HOTKEY_ARRANGE, h.arrange.as_str(), defaults.arrange.clone()),
        (PARK, h.park.as_str(), defaults.park.clone()),
    ];
    let mut out: Vec<(Shortcut, &'static str, String)> = Vec::new();
    for (id, accel, fallback) in wanted {
        let taken = |s: &Shortcut| out.iter().any(|(t, _, _)| t == s);
        let chosen = match parse(accel) {
            Ok(s) if !taken(&s) => Some((s, accel.trim().to_owned())),
            Ok(_) => {
                tracing::warn!(id, accel, "this hotkey is already Fritz's for something else; using the default");
                None
            }
            Err(e) => {
                tracing::warn!(id, error = %e, "using the default hotkey");
                None
            }
        };
        let chosen = chosen.or_else(|| {
            parse(&fallback)
                .ok()
                .filter(|s| !taken(s))
                .map(|s| (s, fallback.clone()))
        });
        match chosen {
            Some((s, accel)) => out.push((s, id, accel)),
            None => tracing::warn!(id, "no hotkey for this; the tray still works"),
        }
    }
    out
}

/// Register the settings' hotkeys. A key another app holds is logged, not
/// fatal: the tray does everything the hotkeys do.
pub fn register(app: &AppHandle, h: &Hotkeys) {
    let mut bound = Vec::new();
    for (shortcut, id, accel) in bindings(h) {
        match app.global_shortcut().register(shortcut) {
            Ok(()) => {
                tracing::info!(id, accel, "hotkey registered");
                bound.push((shortcut, id));
            }
            Err(e) => tracing::warn!(id, accel, error = %e, "couldn't register a hotkey; the tray still works"),
        }
    }
    set_bound(bound);
}

/// What `id_of` answers from: the bindings that registered.
fn set_bound(bound: Vec<(Shortcut, &'static str)>) {
    *BOUND.lock().unwrap_or_else(|e| e.into_inner()) = bound;
}

/// Which of ours `shortcut` is.
pub fn id_of(shortcut: &Shortcut) -> Option<&'static str> {
    BOUND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(s, _)| s == shortcut)
        .map(|(_, id)| *id)
}

#[cfg(test)]
mod tests {
    use super::{bindings, id_of, parse, set_bound, PARK};
    use crate::overlay_machine::HOTKEY_ARRANGE;
    use crate::settings::Hotkeys;

    fn ids(h: &Hotkeys) -> Vec<(&'static str, String)> {
        bindings(h).into_iter().map(|(_, id, a)| (id, a)).collect()
    }

    #[test]
    fn the_defaults_parse() {
        let d = Hotkeys::default();
        for accel in [&d.prompt, &d.ptt, &d.arrange, &d.park] {
            parse(accel).unwrap();
        }
        assert_eq!(
            ids(&d),
            vec![(HOTKEY_ARRANGE, "Ctrl+Alt+A".to_owned()), (PARK, "Ctrl+Alt+H".to_owned())]
        );
    }

    #[test]
    fn the_settings_choose_the_keys() {
        let h = Hotkeys {
            arrange: "Ctrl+Shift+F9".into(),
            park: " Alt+F10 ".into(),
            ..Hotkeys::default()
        };
        assert_eq!(
            ids(&h),
            vec![(HOTKEY_ARRANGE, "Ctrl+Shift+F9".to_owned()), (PARK, "Alt+F10".to_owned())]
        );
        assert_eq!(bindings(&h)[0].0, parse("Ctrl+Shift+F9").unwrap());
    }

    #[test]
    fn a_key_that_doesnt_parse_falls_back_to_its_default() {
        let h = Hotkeys { park: "Ctrl+Alt+Nope".into(), ..Hotkeys::default() };
        assert_eq!(ids(&h)[1], (PARK, "Ctrl+Alt+H".to_owned()));
    }

    /// A press is told apart by its key: each bound key is its own action,
    /// and a key that isn't ours is nothing.
    #[test]
    fn a_pressed_key_is_the_action_bound_to_it() {
        let h = Hotkeys { arrange: "Ctrl+Shift+F9".into(), ..Hotkeys::default() };
        set_bound(bindings(&h).into_iter().map(|(s, id, _)| (s, id)).collect());
        assert_eq!(id_of(&parse("Ctrl+Shift+F9").unwrap()), Some(HOTKEY_ARRANGE));
        assert_eq!(id_of(&parse("Ctrl+Alt+H").unwrap()), Some(PARK));
        assert_eq!(id_of(&parse("Ctrl+Alt+A").unwrap()), None, "no longer arrange's");
        set_bound(Vec::new());
        assert_eq!(id_of(&parse("Ctrl+Alt+H").unwrap()), None);
    }

    /// Two actions on one key: the second gets its default, never the same
    /// key twice (one press would do both).
    #[test]
    fn one_key_never_does_two_things() {
        let h = Hotkeys { park: "Ctrl+Alt+A".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![(HOTKEY_ARRANGE, "Ctrl+Alt+A".to_owned()), (PARK, "Ctrl+Alt+H".to_owned())]
        );
        // And if the default is the one taken, the action goes without.
        let h = Hotkeys { arrange: "Ctrl+Alt+H".into(), park: "Ctrl+Alt+H".into(), ..Hotkeys::default() };
        assert_eq!(ids(&h), vec![(HOTKEY_ARRANGE, "Ctrl+Alt+H".to_owned())]);
    }
}
