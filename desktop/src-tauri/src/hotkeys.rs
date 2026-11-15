//! The global hotkeys, from `settings.hotkeys`: Arrange, Hide/Show and the
//! prompt (push-to-talk arrives with PR 8). Registered at start; a key set
//! with `set_hotkey` is bound the moment it is saved, and one edited into
//! settings.json by hand takes effect on the next start.
//!
//! At start, a binding that doesn't parse, or that another of ours already
//! has, falls back to its default (and says so) rather than leaving the
//! action without a key. A key set live is checked first (`check`) and, if
//! it can't be had, refused with the reason, the old key left working.
//!
//! The bound list is locked only in `with_bound`, never across a call to
//! the global-shortcut plugin: the plugin calls our handler on the main
//! thread under its own lock, and its register and unregister wait on the
//! main thread. Rebinding runs on a command's worker thread, one at a time
//! (`REBINDING`); the handler never rebinds.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Mutex;

use tauri::AppHandle;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

use crate::overlay_machine::HOTKEY_ARRANGE;
use crate::settings::{Hotkeys, Settings};

/// The id the park hotkey is handled under (the machine's own is Arrange's).
pub const PARK: &str = "avatar:park";
/// The id the prompt's hotkey is handled under.
pub const PROMPT: &str = "prompt:toggle";

/// The actions a key may be set for: the setting's name, the id its press
/// is handled under, and what it does (for a refusal naming it).
pub const ACTIONS: [(&str, &str, &str); 3] = [
    ("prompt", PROMPT, "ask him something"),
    ("arrange", HOTKEY_ARRANGE, "arrange him"),
    ("park", PARK, "hide or show him"),
];

/// A key held: `(shortcut, id, accelerator)`, the accelerator as it was
/// set (for `bound_now`).
type Entry = (Shortcut, &'static str, String);

/// The keys registered, for the handler to tell which fired.
static BOUND: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Taken by `rebind` alone, so two rebinds never interleave.
static REBINDING: Mutex<()> = Mutex::new(());

/// Refused: no Ctrl, Alt or Win at all, and not a key that may stand alone.
pub const NEEDS_A_MODIFIER: &str = "A hotkey needs Ctrl and Alt together, or Win, with it (F13 to F24 may \
     stand alone): on its own it would be taken from every app.";
/// Refused: Ctrl or Alt alone (with or without Shift), which apps use.
pub const APPS_OWN_KEYS: &str = "Ctrl or Alt by itself is what apps use for their own keys (copy, save, \
     their menus), and Fritz would take it from every app: use Ctrl and Alt together, or Win.";

/// The only code that locks a bound list. Never call the plugin, or
/// anything that waits on the main thread, inside `f`.
fn with_bound<T>(store: &Mutex<Vec<Entry>>, f: impl FnOnce(&mut Vec<Entry>) -> T) -> T {
    f(&mut store.lock().unwrap_or_else(|e| e.into_inner()))
}

/// An accelerator as the settings hold it (`"Ctrl+Alt+A"`).
pub fn parse(accel: &str) -> Result<Shortcut, String> {
    Shortcut::from_str(accel.trim()).map_err(|e| format!("{accel:?} isn't a hotkey: {e}"))
}

/// The id an action's press is handled under.
pub fn id_for(action: &str) -> Option<&'static str> {
    ACTIONS.iter().find(|(name, _, _)| *name == action).map(|(_, id, _)| *id)
}

/// What the action an id belongs to does.
fn does(id: &str) -> &'static str {
    ACTIONS.iter().find(|(_, i, _)| *i == id).map_or("do something else", |(_, _, d)| *d)
}

/// An action's key as the settings hold it.
fn setting<'a>(h: &'a Hotkeys, action: &str) -> Option<&'a str> {
    match action {
        "prompt" => Some(&h.prompt),
        "arrange" => Some(&h.arrange),
        "park" => Some(&h.park),
        _ => None,
    }
}

/// F13 to F24: no ordinary keyboard types them, so one may stand alone.
fn is_spare(key: Code) -> bool {
    use Code::*;
    matches!(key, F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | F21 | F22 | F23 | F24)
}

/// Why a chord can't be one of Fritz's keys, or `None` if it can. Windows
/// gives a global hotkey to its owner in every app, so a chord an app uses
/// itself (Ctrl+C, Alt+F, Ctrl+F5) would stop working everywhere: a key
/// needs Ctrl and Alt together, or Win, unless it is F13 to F24. (On a
/// layout with AltGr, Ctrl+Alt and a letter types a character: such a
/// chord is the owner's call, as the defaults are.)
pub fn chord_problem(s: &Shortcut) -> Option<&'static str> {
    let win = s.mods.intersects(Modifiers::SUPER | Modifiers::META);
    let ctrl = s.mods.contains(Modifiers::CONTROL);
    let alt = s.mods.contains(Modifiers::ALT);
    if is_spare(s.key) || win || (ctrl && alt) {
        None
    } else if ctrl || alt {
        Some(APPS_OWN_KEYS)
    } else {
        Some(NEEDS_A_MODIFIER)
    }
}

/// The key `accel` would be for `action`, or why it can't be: it must
/// parse, pass `chord_problem`, and be no other action's (as set, or as
/// bound), nor push-to-talk's. The action's own key, however it is spelled, passes.
/// Pure: `bound` is a copy of what is registered.
pub fn check(h: &Hotkeys, action: &str, accel: &str, bound: &[Entry]) -> Result<Shortcut, String> {
    let id = id_for(action).ok_or_else(|| format!("Fritz has no hotkey {action:?}."))?;
    let accel = accel.trim();
    let s = Shortcut::from_str(accel).map_err(|e| format!("Fritz can't use {accel:?} as a key: {e}"))?;
    if let Some(why) = chord_problem(&s) {
        return Err(why.to_owned());
    }
    for (other, other_id, other_does) in ACTIONS {
        if other_id != id && setting(h, other).and_then(|a| parse(a).ok()) == Some(s) {
            return Err(format!("{accel} is already Fritz's key to {other_does}."));
        }
    }
    if let Some((_, other_id, _)) = bound.iter().find(|(b, i, _)| *i != id && *b == s) {
        return Err(format!("{accel} is already Fritz's key to {}.", does(other_id)));
    }
    if parse(&h.ptt).ok() == Some(s) {
        return Err(format!("{accel} is kept for push-to-talk."));
    }
    Ok(s)
}

/// The owner's words for a key Windows wouldn't register. The plugin's
/// errors are flattened to text, and global-hotkey's own for a key another
/// app holds starts "HotKey already registered" (its error.rs).
pub fn refusal(err: &str, accel: &str) -> String {
    if err.starts_with("HotKey already registered") {
        format!("Another app already uses {accel}: pick another.")
    } else {
        format!("Windows wouldn't give Fritz {accel}: {err}")
    }
}

/// Registering keys with Windows: the plugin, or a test's fake.
pub(crate) trait Keys {
    fn register(&self, s: Shortcut) -> Result<(), String>;
    fn unregister(&self, s: Shortcut) -> Result<(), String>;
}

/// The global-shortcut plugin. Its register and unregister wait on the main
/// thread; `register_multiple` and `unregister_all` would hold its lock
/// across that wait, so they are never used.
struct Plugin<'a>(&'a AppHandle);

impl Keys for Plugin<'_> {
    fn register(&self, s: Shortcut) -> Result<(), String> {
        self.0.global_shortcut().register(s).map_err(|e| e.to_string())
    }

    fn unregister(&self, s: Shortcut) -> Result<(), String> {
        self.0.global_shortcut().unregister(s).map_err(|e| e.to_string())
    }
}

/// The bindings to register: `(shortcut, id, accelerator)`, each from the
/// settings or, when that one can't be used, the default. Pure, for tests.
pub fn bindings(h: &Hotkeys) -> Vec<Entry> {
    let defaults = Hotkeys::default();
    // In this order: an older action keeps its key over a newer one.
    let wanted = [
        (HOTKEY_ARRANGE, h.arrange.as_str(), defaults.arrange.clone()),
        (PARK, h.park.as_str(), defaults.park.clone()),
        (PROMPT, h.prompt.as_str(), defaults.prompt.clone()),
    ];
    let mut out: Vec<Entry> = Vec::new();
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
/// fatal: the tray or Settings does everything the hotkeys do.
pub fn register(app: &AppHandle, h: &Hotkeys) {
    register_with(&Plugin(app), &BOUND, h);
}

fn register_with(keys: &impl Keys, store: &Mutex<Vec<Entry>>, h: &Hotkeys) {
    let mut held = Vec::new();
    for (shortcut, id, accel) in bindings(h) {
        match keys.register(shortcut) {
            Ok(()) => {
                tracing::info!(id, accel, "hotkey registered");
                held.push((shortcut, id, accel));
            }
            Err(e) => tracing::warn!(id, accel, error = %e, "couldn't register a hotkey; the tray still works"),
        }
    }
    with_bound(store, |b| *b = held);
}

/// Which of ours `shortcut` is.
pub fn id_of(shortcut: &Shortcut) -> Option<&'static str> {
    id_in(&BOUND, shortcut)
}

fn id_in(store: &Mutex<Vec<Entry>>, shortcut: &Shortcut) -> Option<&'static str> {
    with_bound(store, |b| b.iter().find(|(s, _, _)| s == shortcut).map(|(_, id, _)| *id))
}

/// Each action's key as registered now, or `None` if it has none: at start
/// a setting may have fallen back to its default, or Windows refused it.
pub fn bound_now() -> BTreeMap<&'static str, Option<String>> {
    bound_in(&BOUND)
}

fn bound_in(store: &Mutex<Vec<Entry>>) -> BTreeMap<&'static str, Option<String>> {
    with_bound(store, |b| {
        ACTIONS
            .iter()
            .map(|(name, id, _)| (*name, b.iter().find(|(_, i, _)| i == id).map(|(_, _, a)| a.clone())))
            .collect()
    })
}

/// Bind `accel` to `action` and save it. From a command's worker thread
/// only (the plugin waits on the main thread), one rebind at a time.
pub fn rebind(app: &AppHandle, action: &str, accel: &str) -> Result<Settings, String> {
    let _one = REBINDING.lock().unwrap_or_else(|e| e.into_inner());
    let h = crate::settings::get().hotkeys;
    rebind_with(&Plugin(app), &BOUND, &h, action, accel, |path, accel| {
        crate::change_setting(app, path, serde_json::json!(accel))
    })
}

/// The rebind, in an order that never leaves the action without a key: the
/// new key is registered, then saved, then swapped into `store`, and only
/// then is the old one let go. A key Windows refuses is never saved; a
/// failed save lets the new key go again. The same key spelled anew is only
/// saved. `save` is given the setting's path and the accelerator.
pub(crate) fn rebind_with(
    keys: &impl Keys,
    store: &Mutex<Vec<Entry>>,
    h: &Hotkeys,
    action: &str,
    accel: &str,
    save: impl FnOnce(&str, &str) -> Result<Settings, String>,
) -> Result<Settings, String> {
    let accel = accel.trim();
    let bound = with_bound(store, |b| b.clone());
    let new = check(h, action, accel, &bound)?;
    let Some(id) = id_for(action) else {
        return Err(format!("Fritz has no hotkey {action:?}."));
    };
    let old = bound.iter().find(|(_, i, _)| *i == id).map(|(s, _, _)| *s);
    let path = format!("hotkeys.{action}");
    if old == Some(new) {
        let after = save(&path, accel)?;
        put(store, id, new, accel);
        tracing::info!(action, accel, "hotkey saved; it was already bound");
        return Ok(after);
    }
    keys.register(new).map_err(|e| refusal(&e, accel))?;
    let after = match save(&path, accel) {
        Ok(after) => after,
        Err(e) => {
            if let Err(u) = keys.unregister(new) {
                tracing::warn!(action, accel, error = %u, "couldn't let go of a key that wasn't saved");
            }
            return Err(e);
        }
    };
    put(store, id, new, accel);
    if let Some(old) = old {
        if let Err(e) = keys.unregister(old) {
            tracing::warn!(action, error = %e, "couldn't let go of the old hotkey");
        }
    }
    tracing::info!(action, accel, "hotkey rebound");
    Ok(after)
}

/// `id`'s entry in `store`, replaced or added.
fn put(store: &Mutex<Vec<Entry>>, id: &'static str, s: Shortcut, accel: &str) {
    with_bound(store, |b| match b.iter_mut().find(|(_, i, _)| *i == id) {
        Some(entry) => *entry = (s, id, accel.to_owned()),
        None => b.push((s, id, accel.to_owned())),
    });
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Mutex;

    use super::{
        bindings, bound_in, check, chord_problem, id_in, parse, rebind_with, refusal, register_with, Entry, Keys,
        ACTIONS, APPS_OWN_KEYS, NEEDS_A_MODIFIER, PARK, PROMPT,
    };
    use crate::overlay_machine::HOTKEY_ARRANGE;
    use crate::settings::{Hotkeys, Settings};
    use tauri_plugin_global_shortcut::Shortcut;

    fn ids(h: &Hotkeys) -> Vec<(&'static str, String)> {
        bindings(h).into_iter().map(|(_, id, a)| (id, a)).collect()
    }

    fn key(accel: &str) -> Shortcut {
        parse(accel).unwrap()
    }

    #[test]
    fn the_defaults_parse() {
        let d = Hotkeys::default();
        for accel in [&d.prompt, &d.ptt, &d.arrange, &d.park] {
            parse(accel).unwrap();
        }
        assert_eq!(
            ids(&d),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Alt+A".to_owned()),
                (PARK, "Ctrl+Alt+H".to_owned()),
                (PROMPT, "Ctrl+Alt+F".to_owned()),
            ]
        );
    }

    #[test]
    fn the_settings_choose_the_keys() {
        let h = Hotkeys {
            arrange: "Ctrl+Shift+F9".into(),
            park: " Alt+F10 ".into(),
            prompt: "Ctrl+Alt+Shift+F13".into(),
            ..Hotkeys::default()
        };
        assert_eq!(
            ids(&h),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Shift+F9".to_owned()),
                (PARK, "Alt+F10".to_owned()),
                (PROMPT, "Ctrl+Alt+Shift+F13".to_owned()),
            ]
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
        let store = Mutex::new(bindings(&h));
        assert_eq!(id_in(&store, &key("Ctrl+Shift+F9")), Some(HOTKEY_ARRANGE));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+H")), Some(PARK));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+A")), None, "no longer arrange's");
        *store.lock().unwrap() = Vec::new();
        assert_eq!(id_in(&store, &key("Ctrl+Alt+H")), None);
    }

    /// Two actions on one key: the second gets its default, never the same
    /// key twice (one press would do both).
    #[test]
    fn one_key_never_does_two_things() {
        let h = Hotkeys { park: "Ctrl+Alt+A".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Alt+A".to_owned()),
                (PARK, "Ctrl+Alt+H".to_owned()),
                (PROMPT, "Ctrl+Alt+F".to_owned()),
            ]
        );
        // And if the default is the one taken, the action goes without.
        let h = Hotkeys { arrange: "Ctrl+Alt+H".into(), park: "Ctrl+Alt+H".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![(HOTKEY_ARRANGE, "Ctrl+Alt+H".to_owned()), (PROMPT, "Ctrl+Alt+F".to_owned())]
        );
    }

    /// The prompt came last, so it never takes an older action's key: set
    /// to Arrange's, it gets its own default, and Arrange keeps its key.
    #[test]
    fn the_prompt_never_takes_an_older_actions_key() {
        let h = Hotkeys { prompt: "Ctrl+Alt+A".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Alt+A".to_owned()),
                (PARK, "Ctrl+Alt+H".to_owned()),
                (PROMPT, "Ctrl+Alt+F".to_owned()),
            ]
        );
        // Arrange set to the prompt's default: the prompt goes without.
        let h = Hotkeys { arrange: "Ctrl+Alt+F".into(), ..Hotkeys::default() };
        assert_eq!(ids(&h), vec![(HOTKEY_ARRANGE, "Ctrl+Alt+F".to_owned()), (PARK, "Ctrl+Alt+H".to_owned())]);
    }

    /// A chord Fritz may hold: Ctrl and Alt together, or Win, or F13 to F24
    /// alone. Ctrl or Alt by itself is an app's own key; no modifier at all
    /// is a key typed everywhere.
    #[test]
    fn what_a_chord_must_be() {
        for ok in [
            "Ctrl+Alt+F", "Ctrl+Alt+Shift+F13", "Ctrl+Super+P", "Super+P", "Shift+Super+K", "F13", "F24",
            "Shift+F24", "Ctrl+F13", "Ctrl+Alt+Space",
        ] {
            assert_eq!(chord_problem(&key(ok)), None, "{ok}");
        }
        for (no, why) in [
            ("F", NEEDS_A_MODIFIER),
            ("Shift+F", NEEDS_A_MODIFIER),
            ("F12", NEEDS_A_MODIFIER),
            ("Shift+F1", NEEDS_A_MODIFIER),
            ("Space", NEEDS_A_MODIFIER),
            ("Ctrl+C", APPS_OWN_KEYS),
            ("Ctrl+Shift+T", APPS_OWN_KEYS),
            ("Ctrl+F5", APPS_OWN_KEYS),
            ("Alt+F", APPS_OWN_KEYS),
            ("Alt+F4", APPS_OWN_KEYS),
            ("Alt+Shift+F9", APPS_OWN_KEYS),
        ] {
            assert_eq!(chord_problem(&key(no)), Some(why), "{no}");
        }
        // Both refusals give the rule itself, so following one never leads
        // to the other.
        for why in [NEEDS_A_MODIFIER, APPS_OWN_KEYS] {
            assert!(why.contains("Ctrl and Alt together") && why.contains("or Win"), "{why}");
        }
        // A Win written as Meta is a Win (the library registers it as one).
        let mut meta = key("Ctrl+P");
        meta.mods = tauri_plugin_global_shortcut::Modifiers::META;
        assert_eq!(chord_problem(&meta), None);
    }

    fn defaults_bound() -> Vec<Entry> {
        bindings(&Hotkeys::default())
    }

    #[test]
    fn a_key_is_checked_before_it_is_bound() {
        let d = Hotkeys::default();
        let bound = defaults_bound();
        // Each action's own default passes, however it is spelled.
        for (action, accel) in [("prompt", "Ctrl+Alt+F"), ("arrange", "ctrl+alt+a"), ("park", " Alt + Ctrl + H ")] {
            assert!(check(&d, action, accel, &bound).is_ok(), "{action} {accel}");
        }
        assert_eq!(check(&d, "prompt", "Ctrl+Alt+Shift+F13", &bound), Ok(key("Ctrl+Alt+Shift+F13")));
        assert_eq!(check(&d, "prompt", "F13", &bound), Ok(key("F13")));
        assert_eq!(check(&d, "park", "Ctrl+Super+P", &bound), Ok(key("Ctrl+Super+P")));
        for (action, accel, says) in [
            ("prompt", "Ctrl+Alt+A", "Ctrl+Alt+A is already Fritz's key to arrange him."),
            ("arrange", "Ctrl+Alt+H", "Ctrl+Alt+H is already Fritz's key to hide or show him."),
            ("park", " Ctrl+Alt+F ", "Ctrl+Alt+F is already Fritz's key to ask him something."),
            ("prompt", "Ctrl+Alt+Space", "Ctrl+Alt+Space is kept for push-to-talk."),
            ("prompt", "F", NEEDS_A_MODIFIER),
            ("prompt", "Shift+F", NEEDS_A_MODIFIER),
            ("prompt", "Ctrl+S", APPS_OWN_KEYS),
            ("prompt", "Ctrl+Alt+Nope", "Fritz can't use \"Ctrl+Alt+Nope\" as a key: "),
            ("prompt", "", "Fritz can't use \"\" as a key: "),
            ("ptt", "Ctrl+Alt+Shift+F13", "Fritz has no hotkey \"ptt\"."),
            ("nope", "Ctrl+Alt+Shift+F13", "Fritz has no hotkey \"nope\"."),
        ] {
            let err = check(&d, action, accel, &bound).unwrap_err();
            assert!(err.starts_with(says), "{action} {accel:?}: {err}");
        }
    }

    /// A key is another action's both as set and as bound: one that fell
    /// back at start holds its default, and one Windows refused still has
    /// its setting.
    #[test]
    fn another_actions_key_is_its_setting_and_what_it_holds() {
        // Arrange's setting didn't parse, so it holds its default.
        let h = Hotkeys { arrange: "Ctrl+Alt+Nope".into(), ..Hotkeys::default() };
        let bound = bindings(&h);
        let err = check(&h, "prompt", "Ctrl+Alt+A", &bound).unwrap_err();
        assert!(err.contains("to arrange him"), "{err}");
        // A key no action has passes.
        assert!(check(&h, "prompt", "Ctrl+Alt+Shift+F14", &bound).is_ok());
        // Each action's key refused by Windows: not bound, but still its
        // own, named to every other action that would take it.
        let spare = "Ctrl+Alt+Shift+F15";
        for (name, id, does) in ACTIONS {
            let mut h = Hotkeys::default();
            match name {
                "prompt" => h.prompt = spare.into(),
                "arrange" => h.arrange = spare.into(),
                "park" => h.park = spare.into(),
                _ => unreachable!("{name}"),
            }
            let bound: Vec<Entry> = bindings(&h).into_iter().filter(|(_, i, _)| *i != id).collect();
            assert!(check(&h, name, spare, &bound).is_ok(), "{name} may still set its own");
            for (other, _, _) in ACTIONS.iter().filter(|(n, _, _)| *n != name) {
                let err = check(&h, other, spare, &bound).unwrap_err();
                assert_eq!(err, format!("{spare} is already Fritz's key to {does}."), "{other} taking {name}'s");
            }
        }
    }

    /// global-hotkey's own text for a key another app holds (error.rs:
    /// `HotKey already registered: {0:?}`), as the plugin flattens it.
    #[test]
    fn a_key_another_app_holds_is_said_so() {
        let held = key("Ctrl+Alt+Shift+F13");
        let err = tauri_plugin_global_shortcut::Error::GlobalHotkey(format!("HotKey already registered: {held:?}"));
        assert_eq!(
            refusal(&err.to_string(), "Ctrl+Alt+Shift+F13"),
            "Another app already uses Ctrl+Alt+Shift+F13: pick another."
        );
        assert_eq!(
            refusal("Unable to register hotkey: Unknown VKCode for F35", "Ctrl+Alt+F35"),
            "Windows wouldn't give Fritz Ctrl+Alt+F35: Unable to register hotkey: Unknown VKCode for F35"
        );
    }

    /// Windows, as a test sees it: every call logged; keys in `held` are
    /// another app's, and `stuck` keys can't be let go. With `watch`, each
    /// call also logs what that list held as it was made.
    #[derive(Default)]
    struct FakeKeys<'a> {
        calls: RefCell<Vec<String>>,
        held: Vec<Shortcut>,
        stuck: Vec<Shortcut>,
        watch: Option<&'a Mutex<Vec<Entry>>>,
    }

    impl FakeKeys<'_> {
        fn holding(held: &[&str]) -> Self {
            Self { held: held.iter().map(|a| key(a)).collect(), ..Self::default() }
        }

        fn name(s: Shortcut) -> String {
            for accel in ["Ctrl+Alt+A", "Ctrl+Alt+H", "Ctrl+Alt+F", "Ctrl+Alt+Shift+F13", "Ctrl+Alt+Shift+F14"] {
                if key(accel) == s {
                    return accel.to_owned();
                }
            }
            format!("{s:?}")
        }

        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }

        /// What the watched list held for the prompt, as a call was made.
        fn seen(&self) -> String {
            self.watch.map_or_else(String::new, |store| {
                format!(" (the prompt's then: {})", bound_in(store)["prompt"].as_deref().unwrap_or("none"))
            })
        }
    }

    impl Keys for FakeKeys<'_> {
        fn register(&self, s: Shortcut) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("register {}{}", Self::name(s), self.seen()));
            if self.held.contains(&s) {
                return Err(format!("HotKey already registered: {s:?}"));
            }
            Ok(())
        }

        fn unregister(&self, s: Shortcut) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("unregister {}{}", Self::name(s), self.seen()));
            if self.stuck.contains(&s) {
                return Err(format!("Failed to unregister hotkey: {s:?}"));
            }
            Ok(())
        }
    }

    /// A save that logs itself into a fake's calls and answers `ok`.
    fn saver(calls: &RefCell<Vec<String>>, ok: bool) -> impl FnOnce(&str, &str) -> Result<Settings, String> + '_ {
        move |path, accel| {
            calls.borrow_mut().push(format!("save {path}={accel}"));
            if ok {
                Ok(Settings::default())
            } else {
                Err("Couldn't save the settings: disk full".to_owned())
            }
        }
    }

    fn store() -> Mutex<Vec<Entry>> {
        Mutex::new(defaults_bound())
    }

    /// The new key is taken, saved, swapped in, and only then the old one
    /// let go: the action always has a key.
    #[test]
    fn a_rebind_takes_the_new_key_before_it_lets_the_old_go() {
        let keys = FakeKeys::default();
        let store = store();
        let d = Hotkeys::default();
        assert!(rebind_with(&keys, &store, &d, "prompt", " Ctrl+Alt+Shift+F13 ", saver(&keys.calls,true)).is_ok());
        assert_eq!(
            keys.calls(),
            ["register Ctrl+Alt+Shift+F13", "save hotkeys.prompt=Ctrl+Alt+Shift+F13", "unregister Ctrl+Alt+F"]
        );
        assert_eq!(id_in(&store, &key("Ctrl+Alt+Shift+F13")), Some(PROMPT));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+F")), None);
        assert_eq!(bound_in(&store)["prompt"].as_deref(), Some("Ctrl+Alt+Shift+F13"));
        assert_eq!(bound_in(&store)["arrange"].as_deref(), Some("Ctrl+Alt+A"), "the others are untouched");
    }

    /// The new key is registered while the old one is still the prompt's,
    /// and the old one let go only once the new one is: at every step a
    /// press finds the prompt by one of them.
    #[test]
    fn the_old_key_is_let_go_only_once_the_new_one_is_bound() {
        let store = store();
        let keys = FakeKeys { watch: Some(&store), ..FakeKeys::default() };
        rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+Shift+F13", saver(&keys.calls, true)).unwrap();
        assert_eq!(
            keys.calls(),
            [
                "register Ctrl+Alt+Shift+F13 (the prompt's then: Ctrl+Alt+F)",
                "save hotkeys.prompt=Ctrl+Alt+Shift+F13",
                "unregister Ctrl+Alt+F (the prompt's then: Ctrl+Alt+Shift+F13)",
            ]
        );
    }

    /// A key another app holds is refused with the reason: never saved, and
    /// the old key still works.
    #[test]
    fn a_refused_key_leaves_the_old_one_bound() {
        let keys = FakeKeys::holding(&["Ctrl+Alt+Shift+F13"]);
        let store = store();
        let err = rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+Shift+F13", saver(&keys.calls,true))
            .unwrap_err();
        assert_eq!(err, "Another app already uses Ctrl+Alt+Shift+F13: pick another.");
        assert_eq!(keys.calls(), ["register Ctrl+Alt+Shift+F13"]);
        assert_eq!(id_in(&store, &key("Ctrl+Alt+F")), Some(PROMPT));
        assert_eq!(*store.lock().unwrap(), defaults_bound());
    }

    /// A save that fails lets the new key go again; the old one is kept.
    #[test]
    fn a_failed_save_lets_the_new_key_go() {
        let keys = FakeKeys::default();
        let store = store();
        let err = rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+Shift+F13", saver(&keys.calls,false))
            .unwrap_err();
        assert_eq!(err, "Couldn't save the settings: disk full");
        assert_eq!(
            keys.calls(),
            ["register Ctrl+Alt+Shift+F13", "save hotkeys.prompt=Ctrl+Alt+Shift+F13", "unregister Ctrl+Alt+Shift+F13"]
        );
        assert_eq!(*store.lock().unwrap(), defaults_bound());
    }

    /// The same key spelled anew is only saved (registering it again would
    /// be refused as already held), and reported as it is now spelled.
    #[test]
    fn the_same_key_respelled_is_only_saved() {
        let keys = FakeKeys::default();
        let store = store();
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "alt+ctrl+f", saver(&keys.calls,true)).is_ok());
        assert_eq!(keys.calls(), ["save hotkeys.prompt=alt+ctrl+f"]);
        assert_eq!(bound_in(&store)["prompt"].as_deref(), Some("alt+ctrl+f"));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+F")), Some(PROMPT));
        // A failed save changes nothing.
        let keys = FakeKeys::default();
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+F", saver(&keys.calls,false)).is_err());
        assert_eq!(bound_in(&store)["prompt"].as_deref(), Some("alt+ctrl+f"));
    }

    /// An action that had no key (Windows refused it at start) gets one,
    /// with nothing to let go.
    #[test]
    fn an_action_without_a_key_gets_one() {
        let keys = FakeKeys::default();
        let store = Mutex::new(defaults_bound().into_iter().filter(|(_, id, _)| *id != PARK).collect());
        assert_eq!(bound_in(&store)["park"], None);
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "park", "Ctrl+Alt+Shift+F14", saver(&keys.calls,true)).is_ok());
        assert_eq!(keys.calls(), ["register Ctrl+Alt+Shift+F14", "save hotkeys.park=Ctrl+Alt+Shift+F14"]);
        assert_eq!(id_in(&store, &key("Ctrl+Alt+Shift+F14")), Some(PARK));
    }

    /// A key the check refuses never reaches Windows or the settings; an
    /// old key Windows won't let go is only logged (the new one is bound).
    #[test]
    fn a_refused_check_touches_nothing_and_a_stuck_old_key_is_only_logged() {
        let keys = FakeKeys::default();
        let store = store();
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+A", saver(&keys.calls,true)).is_err());
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+C", saver(&keys.calls,true)).is_err());
        assert!(keys.calls().is_empty(), "{:?}", keys.calls());
        let keys = FakeKeys { stuck: vec![key("Ctrl+Alt+F")], ..FakeKeys::default() };
        assert!(rebind_with(&keys, &store, &Hotkeys::default(), "prompt", "Ctrl+Alt+Shift+F13", saver(&keys.calls,true)).is_ok());
        assert_eq!(keys.calls().last().map(String::as_str), Some("unregister Ctrl+Alt+F"));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+Shift+F13")), Some(PROMPT));
    }

    /// At start, a key Windows refuses is left out of what is bound, and
    /// reported as no key.
    #[test]
    fn at_start_a_refused_key_is_left_out() {
        let keys = FakeKeys::holding(&["Ctrl+Alt+A"]);
        let store = Mutex::new(Vec::new());
        register_with(&keys, &store, &Hotkeys::default());
        assert_eq!(keys.calls(), ["register Ctrl+Alt+A", "register Ctrl+Alt+H", "register Ctrl+Alt+F"]);
        let now = bound_in(&store);
        assert_eq!(now.keys().copied().collect::<Vec<_>>(), ["arrange", "park", "prompt"]);
        assert_eq!(now["arrange"], None);
        assert_eq!(now["park"].as_deref(), Some("Ctrl+Alt+H"));
        assert_eq!(now["prompt"].as_deref(), Some("Ctrl+Alt+F"));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+A")), None);
    }

    /// The actions a window may name are the fixture's, and each is handled
    /// under its own id.
    #[test]
    fn the_actions_are_the_wire_contract() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let mut names: Vec<&str> = ACTIONS.iter().map(|(n, _, _)| *n).collect();
        names.sort_unstable();
        assert_eq!(fixture["hotkeyActions"], serde_json::json!(names));
        // Settings' Default sends these: the settings' own defaults.
        let d = Hotkeys::default();
        assert_eq!(
            fixture["hotkeyDefaults"],
            serde_json::json!({ "prompt": d.prompt, "arrange": d.arrange, "park": d.park })
        );
        assert_eq!(super::id_for("prompt"), Some(PROMPT));
        assert_eq!(super::id_for("arrange"), Some(HOTKEY_ARRANGE));
        assert_eq!(super::id_for("park"), Some(PARK));
        assert_eq!(super::id_for("ptt"), None, "push-to-talk comes with PR 8");
    }

    /// This module's code, without its tests.
    fn code(src: &str) -> String {
        let src = src.replace("\r\n", "\n");
        src[..src.find("#[cfg(test)]").unwrap()].to_owned()
    }

    /// The text from `open` (just after a `(`) to its matching `)`.
    fn to_close(s: &str) -> &str {
        let mut depth = 1;
        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return &s[..i];
                    }
                }
                _ => {}
            }
        }
        panic!("unclosed")
    }

    /// The bound list is locked in `with_bound` alone, and nothing given to
    /// it calls the plugin: the handler takes that lock on the main thread,
    /// where the plugin's calls wait.
    #[test]
    fn bound_is_never_held_across_a_plugin_call() {
        let src = code(include_str!("hotkeys.rs"));
        let def = src.find("fn with_bound<").unwrap();
        let body = &src[def..def + src[def..].find("\n}\n").unwrap()];
        let locks: Vec<usize> = src.match_indices(".lock()").map(|(i, _)| i).collect();
        assert_eq!(locks.len(), 2, "one in with_bound, one for REBINDING");
        for at in locks {
            let in_with_bound = at > def && at < def + body.len();
            assert!(in_with_bound || src[..at].ends_with("REBINDING"), "a lock outside with_bound: {}", &src[at - 40..at]);
        }
        let calls: Vec<usize> = src.match_indices("with_bound(").map(|(i, _)| i).collect();
        assert!(calls.len() >= 5, "{}", calls.len());
        for at in calls {
            let args = to_close(&src[at + "with_bound(".len()..]);
            for plugin in ["register(", "unregister(", "global_shortcut", "keys.", "save(", "change_setting"] {
                assert!(!args.contains(plugin), "with_bound given {plugin}: {args}");
            }
        }
    }

    /// One rebind at a time, and only `rebind` takes that turn.
    #[test]
    fn only_rebind_takes_the_rebinding_lock() {
        let src = code(include_str!("hotkeys.rs"));
        let uses = src.lines().filter(|l| !l.trim_start().starts_with("//")).filter(|l| l.contains("REBINDING"));
        assert_eq!(uses.count(), 2, "its static, and rebind");
        let at = src.find("pub fn rebind(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(body.contains("let _one = REBINDING.lock()"), "{body}");
    }

    /// The handler never rebinds or calls the plugin: it runs on the main
    /// thread, inside the plugin's lock, where the plugin's calls wait.
    /// Rebinding runs on a blocking worker, from `set_hotkey` alone.
    #[test]
    fn the_handler_never_rebinds() {
        let lib = code(include_str!("lib.rs"));
        let at = lib.find("fn on_shortcut(").unwrap();
        let handler = &lib[at..at + lib[at..].find("\n}\n").unwrap()];
        for (name, src) in [
            ("on_shortcut", handler.to_owned()),
            ("prompt.rs", code(include_str!("prompt.rs"))),
            ("overlay_windows.rs", code(include_str!("overlay_windows.rs"))),
            ("overlay_machine.rs", code(include_str!("overlay_machine.rs"))),
        ] {
            for call in ["hotkeys::rebind", "global_shortcut()", "hotkeys::register("] {
                assert!(!src.contains(call), "{name} calls {call}");
            }
        }
        assert_eq!(lib.matches("hotkeys::rebind(").count(), 1, "one caller");
        let at = lib.find("async fn set_hotkey(").unwrap();
        let body = &lib[at..at + lib[at..].find("\n}\n").unwrap()];
        assert!(body.contains("spawn_blocking(move || hotkeys::rebind("), "{body}");
    }
}
