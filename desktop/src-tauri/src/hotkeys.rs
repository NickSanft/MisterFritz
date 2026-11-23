//! The global hotkeys, from `settings.hotkeys`: Arrange, Hide/Show, the
//! prompt and push-to-talk. Registered at start; a key set with
//! `set_hotkey` is bound the moment it is saved, and one edited into
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
/// The id push-to-talk is handled under: held, not pressed (its release
/// matters as much as its press).
pub const PTT: &str = "ptt:hold";

/// The actions a key may be set for: the setting's name, the id its press
/// is handled under, and what it does (for a refusal naming it).
pub const ACTIONS: [(&str, &str, &str); 4] = [
    ("prompt", PROMPT, "ask him something"),
    ("arrange", HOTKEY_ARRANGE, "arrange him"),
    ("park", PARK, "hide or show him"),
    ("ptt", PTT, "talk to him"),
];

/// A key held: `(shortcut, id, accelerator)`, the accelerator as it was
/// set (for `bound_now`).
type Entry = (Shortcut, &'static str, String);

/// The keys registered, for the handler to tell which fired.
static BOUND: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Taken by `rebind` and `mute` alone, so two rebinds never interleave, and
/// a mute never comes between a rebind's look at the keys and its swap.
/// Both run on a worker thread, never the main one, where the plugin's
/// calls inside them wait.
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
        "ptt" => Some(&h.ptt),
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
/// bound). The action's own key, however it is spelled, passes.
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
        (PTT, h.ptt.as_str(), defaults.ptt.clone()),
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

/// Register the settings' hotkeys, all but push-to-talk's while it is muted
/// (`audio.muted`): its chord is left to other apps. A key another app
/// holds is logged, not fatal: the tray or Settings does everything the
/// hotkeys do.
pub fn register(app: &AppHandle, h: &Hotkeys, muted: bool) {
    register_with(&Plugin(app), &BOUND, h, muted);
}

fn register_with(keys: &impl Keys, store: &Mutex<Vec<Entry>>, h: &Hotkeys, muted: bool) {
    let mut held = Vec::new();
    for (shortcut, id, accel) in bindings(h) {
        if muted && id == PTT {
            tracing::info!(accel, "push-to-talk is muted: its key is left to other apps");
            continue;
        }
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

/// Bind `accel` to `action` and save it: the settings saved, and whether
/// the action moved to a new key (`rebind_with`). From a command's worker
/// thread only (the plugin waits on the main thread), one rebind at a time.
pub fn rebind(app: &AppHandle, action: &str, accel: &str) -> Result<(Settings, bool), String> {
    let _one = REBINDING.lock().unwrap_or_else(|e| e.into_inner());
    let s = crate::settings::get();
    muted_refusal(action, s.audio.muted)?;
    rebind_with(&Plugin(app), &BOUND, &s.hotkeys, action, accel, |path, accel| {
        crate::change_setting(app, path, serde_json::json!(accel))
    })
}

/// Setting push-to-talk's key while it is muted, refused: the owner's call,
/// so nothing is saved or taken from Windows until it is unmuted.
pub const PTT_MUTED: &str = "Push-to-talk is muted: unmute it from the tray first.";

/// Whether `action`'s key may be set now: push-to-talk's not while muted.
pub fn muted_refusal(action: &str, muted: bool) -> Result<(), String> {
    if muted && action == "ptt" {
        Err(PTT_MUTED.to_owned())
    } else {
        Ok(())
    }
}

/// Push-to-talk muted: its key let go of, so other apps get the chord. Or
/// unmuted: its key taken from Windows again, the setting's or, as at
/// start, the default when that one can't be used. In a rebind's turn, from
/// a worker thread only (the plugin waits on the main thread). Answers the
/// key push-to-talk holds now, if any.
pub fn mute(app: &AppHandle, on: bool) -> Option<String> {
    let _one = REBINDING.lock().unwrap_or_else(|e| e.into_inner());
    let h = crate::settings::get().hotkeys;
    mute_with(&Plugin(app), &BOUND, &h, on)
}

/// `mute`'s work. Muting takes push-to-talk's entry out of `store` before
/// letting go of its key, so a press or release still coming for it is no
/// key of ours; one Windows won't let go of is only logged (muted, it
/// opens nothing).
pub(crate) fn mute_with(keys: &impl Keys, store: &Mutex<Vec<Entry>>, h: &Hotkeys, on: bool) -> Option<String> {
    let bound = with_bound(store, |b| b.clone());
    let held = bound.iter().find(|(_, id, _)| *id == PTT).cloned();
    if on {
        if let Some((s, _, accel)) = held {
            with_bound(store, |b| b.retain(|(_, id, _)| *id != PTT));
            if let Err(e) = keys.unregister(s) {
                tracing::warn!(accel, error = %e, "couldn't let go of push-to-talk's key; muted, it opens nothing");
            }
            tracing::info!(accel, "push-to-talk muted: its key let go of");
        }
        return None;
    }
    if let Some((_, _, accel)) = held {
        return Some(accel);
    }
    let taken = |s: &Shortcut| bound.iter().any(|(b, _, _)| b == s);
    let defaults = Hotkeys::default();
    let chosen = [h.ptt.as_str(), defaults.ptt.as_str()]
        .into_iter()
        .find_map(|a| parse(a).ok().filter(|s| !taken(s)).map(|s| (s, a.trim().to_owned())));
    let Some((s, accel)) = chosen else {
        tracing::warn!(accel = h.ptt.as_str(), "unmuted, but push-to-talk's key is another of Fritz's; the tray still works");
        return None;
    };
    match keys.register(s) {
        Ok(()) => {
            put(store, PTT, s, &accel);
            tracing::info!(accel, "push-to-talk unmuted: its key taken again");
            Some(accel)
        }
        Err(e) => {
            tracing::warn!(accel, error = %e, "unmuted, but couldn't take push-to-talk's key from Windows");
            None
        }
    }
}

/// The rebind, in an order that never leaves the action without a key: the
/// new key is registered, then saved, then swapped into `store`, and only
/// then is the old one let go. A key Windows refuses is never saved; a
/// failed save lets the new key go again. The same key spelled anew is only
/// saved. `save` is given the setting's path and the accelerator. Answers
/// what `save` did, and whether the action moved to a new key (false for
/// the same key respelled): push-to-talk moved must be let go, as its old
/// key's release no longer reaches Fritz.
pub(crate) fn rebind_with(
    keys: &impl Keys,
    store: &Mutex<Vec<Entry>>,
    h: &Hotkeys,
    action: &str,
    accel: &str,
    save: impl FnOnce(&str, &str) -> Result<Settings, String>,
) -> Result<(Settings, bool), String> {
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
        return Ok((after, false));
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
    Ok((after, true))
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
        bindings, bound_in, check, chord_problem, id_in, muted_refusal, mute_with, parse, rebind_with, refusal,
        register_with, Entry, Keys, ACTIONS, APPS_OWN_KEYS, NEEDS_A_MODIFIER, PARK, PROMPT, PTT, PTT_MUTED,
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
                (PTT, "Ctrl+Alt+V".to_owned()),
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
                (PTT, "Ctrl+Alt+V".to_owned()),
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
                (PTT, "Ctrl+Alt+V".to_owned()),
            ]
        );
        // And if the default is the one taken, the action goes without.
        let h = Hotkeys { arrange: "Ctrl+Alt+H".into(), park: "Ctrl+Alt+H".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Alt+H".to_owned()),
                (PROMPT, "Ctrl+Alt+F".to_owned()),
                (PTT, "Ctrl+Alt+V".to_owned()),
            ]
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
                (PTT, "Ctrl+Alt+V".to_owned()),
            ]
        );
        // Arrange set to the prompt's default: the prompt goes without.
        let h = Hotkeys { arrange: "Ctrl+Alt+F".into(), ..Hotkeys::default() };
        assert_eq!(
            ids(&h),
            vec![
                (HOTKEY_ARRANGE, "Ctrl+Alt+F".to_owned()),
                (PARK, "Ctrl+Alt+H".to_owned()),
                (PTT, "Ctrl+Alt+V".to_owned()),
            ]
        );
    }

    /// Push-to-talk came last of all, so it never takes another's key: set
    /// to the prompt's, it gets its own default; a key that doesn't parse
    /// does the same; and with its default taken too, it goes without.
    #[test]
    fn push_to_talk_never_takes_another_actions_key() {
        let h = Hotkeys { ptt: "Ctrl+Alt+F".into(), ..Hotkeys::default() };
        assert_eq!(ids(&h)[3], (PTT, "Ctrl+Alt+V".to_owned()));
        assert_eq!(ids(&h)[2], (PROMPT, "Ctrl+Alt+F".to_owned()));
        let h = Hotkeys { ptt: "Ctrl+Alt+Nope".into(), ..Hotkeys::default() };
        assert_eq!(ids(&h)[3], (PTT, "Ctrl+Alt+V".to_owned()));
        let h = Hotkeys { ptt: "Ctrl+Alt+A".into(), park: "Ctrl+Alt+V".into(), ..Hotkeys::default() };
        assert!(ids(&h).iter().all(|(id, _)| *id != PTT), "{:?}", ids(&h));
    }

    /// Push-to-talk's key, bound under its default because its setting
    /// doesn't parse, is still named as push-to-talk's to a rebind.
    #[test]
    fn a_rebind_onto_push_to_talks_bound_key_names_it() {
        let h = Hotkeys { ptt: "Ctrl+Alt+Nope".into(), ..Hotkeys::default() };
        let bound = bindings(&h);
        assert_eq!(
            check(&h, "prompt", "Ctrl+Alt+V", &bound),
            Err("Ctrl+Alt+V is already Fritz's key to talk to him.".to_owned())
        );
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
            ("prompt", "Ctrl+Alt+V", "Ctrl+Alt+V is already Fritz's key to talk to him."),
            ("ptt", "Ctrl+Alt+F", "Ctrl+Alt+F is already Fritz's key to ask him something."),
            ("prompt", "F", NEEDS_A_MODIFIER),
            ("prompt", "Shift+F", NEEDS_A_MODIFIER),
            ("prompt", "Ctrl+S", APPS_OWN_KEYS),
            ("prompt", "Ctrl+Alt+Nope", "Fritz can't use \"Ctrl+Alt+Nope\" as a key: "),
            ("prompt", "", "Fritz can't use \"\" as a key: "),
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
                "ptt" => h.ptt = spare.into(),
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
    /// call also logs what that list held for that action as it was made.
    #[derive(Default)]
    struct FakeKeys<'a> {
        calls: RefCell<Vec<String>>,
        held: Vec<Shortcut>,
        stuck: Vec<Shortcut>,
        watch: Option<(&'a Mutex<Vec<Entry>>, &'static str)>,
    }

    impl FakeKeys<'_> {
        fn holding(held: &[&str]) -> Self {
            Self { held: held.iter().map(|a| key(a)).collect(), ..Self::default() }
        }

        fn name(s: Shortcut) -> String {
            for accel in ["Ctrl+Alt+A", "Ctrl+Alt+H", "Ctrl+Alt+F", "Ctrl+Alt+V", "Ctrl+Alt+Shift+F13", "Ctrl+Alt+Shift+F14"] {
                if key(accel) == s {
                    return accel.to_owned();
                }
            }
            format!("{s:?}")
        }

        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }

        /// What the watched list held for the watched action, as a call was
        /// made.
        fn seen(&self) -> String {
            self.watch.map_or_else(String::new, |(store, action)| {
                format!(" (the {action}'s then: {})", bound_in(store)[action].as_deref().unwrap_or("none"))
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
        let keys = FakeKeys { watch: Some((&store, "prompt")), ..FakeKeys::default() };
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

    /// A rebind says whether the action moved to a new key (`set_hotkey`
    /// lets go of push-to-talk moved): not when the key is respelled, and
    /// a refused one is an error, push-to-talk kept on its old key.
    #[test]
    fn a_rebind_says_whether_the_key_moved() {
        let store = store();
        let d = Hotkeys::default();
        let moved = |r: Result<(Settings, bool), String>| r.map(|(_, moved)| moved);
        let keys = FakeKeys::default();
        assert_eq!(moved(rebind_with(&keys, &store, &d, "ptt", "alt+ctrl+v", saver(&keys.calls, true))), Ok(false), "respelled");
        assert_eq!(keys.calls(), ["save hotkeys.ptt=alt+ctrl+v"]);
        let keys = FakeKeys::holding(&["Ctrl+Alt+Shift+F13"]);
        assert!(rebind_with(&keys, &store, &d, "ptt", "Ctrl+Alt+Shift+F13", saver(&keys.calls, true)).is_err());
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), Some(PTT), "refused, it keeps its key");
        let keys = FakeKeys::default();
        assert!(rebind_with(&keys, &store, &d, "ptt", "Ctrl+Alt+Shift+F13", saver(&keys.calls, false)).is_err());
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), Some(PTT), "unsaved, it keeps its key");
        let keys = FakeKeys::default();
        assert_eq!(moved(rebind_with(&keys, &store, &d, "ptt", "Ctrl+Alt+Shift+F13", saver(&keys.calls, true))), Ok(true));
        assert_eq!(keys.calls(), ["register Ctrl+Alt+Shift+F13", "save hotkeys.ptt=Ctrl+Alt+Shift+F13", "unregister Ctrl+Alt+V"]);
        assert_eq!(id_in(&store, &key("Ctrl+Alt+Shift+F13")), Some(PTT));
        let keys = FakeKeys::default();
        assert_eq!(moved(rebind_with(&keys, &store, &d, "prompt", "Ctrl+Alt+Shift+F14", saver(&keys.calls, true))), Ok(true));
        // An action that had no key gets one: moved, with nothing let go.
        let store = Mutex::new(defaults_bound().into_iter().filter(|(_, id, _)| *id != PTT).collect());
        let keys = FakeKeys::default();
        assert_eq!(moved(rebind_with(&keys, &store, &d, "ptt", "Ctrl+Alt+Shift+F13", saver(&keys.calls, true))), Ok(true));
        assert_eq!(keys.calls(), ["register Ctrl+Alt+Shift+F13", "save hotkeys.ptt=Ctrl+Alt+Shift+F13"]);
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
        register_with(&keys, &store, &Hotkeys::default(), false);
        assert_eq!(
            keys.calls(),
            ["register Ctrl+Alt+A", "register Ctrl+Alt+H", "register Ctrl+Alt+F", "register Ctrl+Alt+V"]
        );
        let now = bound_in(&store);
        assert_eq!(now.keys().copied().collect::<Vec<_>>(), ["arrange", "park", "prompt", "ptt"]);
        assert_eq!(now["arrange"], None);
        assert_eq!(now["park"].as_deref(), Some("Ctrl+Alt+H"));
        assert_eq!(now["prompt"].as_deref(), Some("Ctrl+Alt+F"));
        assert_eq!(id_in(&store, &key("Ctrl+Alt+A")), None);
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), Some(PTT));
        assert_eq!(now["ptt"].as_deref(), Some("Ctrl+Alt+V"), "push-to-talk's row shows its key");
    }

    /// Muted at start, push-to-talk's key is left to other apps; the others
    /// are taken as ever.
    #[test]
    fn muted_at_start_push_to_talk_s_key_is_left_alone() {
        let keys = FakeKeys::default();
        let store = Mutex::new(Vec::new());
        register_with(&keys, &store, &Hotkeys::default(), true);
        assert_eq!(keys.calls(), ["register Ctrl+Alt+A", "register Ctrl+Alt+H", "register Ctrl+Alt+F"]);
        assert_eq!(bound_in(&store)["ptt"], None);
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), None);
    }

    /// Muting lets go of push-to-talk's key, out of the bound list first;
    /// unmuting takes it again, into the list once Windows has given it;
    /// each twice is once.
    #[test]
    fn muting_lets_go_of_push_to_talk_s_key_and_unmuting_takes_it_again() {
        let store = store();
        let d = Hotkeys::default();
        let keys = FakeKeys { watch: Some((&store, "ptt")), ..FakeKeys::default() };
        assert_eq!(mute_with(&keys, &store, &d, true), None);
        assert_eq!(keys.calls(), ["unregister Ctrl+Alt+V (the ptt's then: none)"], "out of the list before");
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), None);
        assert_eq!(bound_in(&store)["prompt"].as_deref(), Some("Ctrl+Alt+F"), "the others are untouched");
        let keys = FakeKeys { watch: Some((&store, "ptt")), ..FakeKeys::default() };
        assert_eq!(mute_with(&keys, &store, &d, true), None);
        assert!(keys.calls().is_empty(), "muted again: {:?}", keys.calls());
        assert_eq!(mute_with(&keys, &store, &d, false).as_deref(), Some("Ctrl+Alt+V"));
        assert_eq!(keys.calls(), ["register Ctrl+Alt+V (the ptt's then: none)"], "into the list after");
        assert_eq!(id_in(&store, &key("Ctrl+Alt+V")), Some(PTT));
        let keys = FakeKeys::default();
        assert_eq!(mute_with(&keys, &store, &d, false).as_deref(), Some("Ctrl+Alt+V"), "unmuted again");
        assert!(keys.calls().is_empty(), "{:?}", keys.calls());
    }

    /// Unmuted, push-to-talk takes the setting's key, or, as at start, the
    /// default when the setting's can't be used; none when neither can, and
    /// none when Windows refuses it. A key Windows won't let go of at mute is
    /// only logged: it is no key of ours.
    #[test]
    fn unmuting_takes_the_key_the_start_would() {
        let unbound = || Mutex::new(defaults_bound().into_iter().filter(|(_, id, _)| *id != PTT).collect::<Vec<_>>());
        let with_ptt = |ptt: &str| Hotkeys { ptt: ptt.into(), ..Hotkeys::default() };
        let keys = FakeKeys::default();
        let list = unbound();
        assert_eq!(mute_with(&keys, &list, &with_ptt(" Ctrl+Alt+J "), false).as_deref(), Some("Ctrl+Alt+J"));
        assert_eq!(id_in(&list, &key("Ctrl+Alt+J")), Some(PTT));
        for (ptt, why) in [("Ctrl+Alt+Nope", "unparsed"), ("Ctrl+Alt+F", "the prompt's")] {
            let keys = FakeKeys::default();
            let list = unbound();
            assert_eq!(mute_with(&keys, &list, &with_ptt(ptt), false).as_deref(), Some("Ctrl+Alt+V"), "{why}");
            assert_eq!(keys.calls(), ["register Ctrl+Alt+V"], "{why}");
        }
        let both = Mutex::new(vec![
            (key("Ctrl+Alt+F"), PROMPT, "Ctrl+Alt+F".to_owned()),
            (key("Ctrl+Alt+V"), PARK, "Ctrl+Alt+V".to_owned()),
        ]);
        let keys = FakeKeys::default();
        assert_eq!(mute_with(&keys, &both, &with_ptt("Ctrl+Alt+F"), false), None);
        assert!(keys.calls().is_empty(), "{:?}", keys.calls());
        let keys = FakeKeys::holding(&["Ctrl+Alt+V"]);
        let list = unbound();
        assert_eq!(mute_with(&keys, &list, &Hotkeys::default(), false), None);
        assert_eq!(bound_in(&list)["ptt"], None, "refused, it has no key");
        let keys = FakeKeys { stuck: vec![key("Ctrl+Alt+V")], ..FakeKeys::default() };
        let list = store();
        assert_eq!(mute_with(&keys, &list, &Hotkeys::default(), true), None);
        assert_eq!(id_in(&list, &key("Ctrl+Alt+V")), None, "stuck in Windows, but no key of ours");
    }

    /// Push-to-talk's key isn't set while it is muted; the others' are.
    #[test]
    fn push_to_talk_s_key_is_not_set_while_muted() {
        assert_eq!(muted_refusal("ptt", true), Err(PTT_MUTED.to_owned()));
        assert_eq!(muted_refusal("ptt", false), Ok(()));
        for other in ["prompt", "arrange", "park"] {
            assert_eq!(muted_refusal(other, true), Ok(()), "{other}");
        }
        let src = code(include_str!("hotkeys.rs"));
        let at = src.find("pub fn rebind(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        let lock = body.find("let _one = REBINDING.lock()").unwrap();
        let refusal = body.find("muted_refusal(action, s.audio.muted)?;").expect("refused in rebind");
        let work = body.find("rebind_with(").unwrap();
        assert!(lock < refusal && refusal < work, "{body}");
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
            serde_json::json!({ "prompt": d.prompt, "arrange": d.arrange, "park": d.park, "ptt": d.ptt })
        );
        assert_eq!(super::id_for("prompt"), Some(PROMPT));
        assert_eq!(super::id_for("arrange"), Some(HOTKEY_ARRANGE));
        assert_eq!(super::id_for("park"), Some(PARK));
        assert_eq!(super::id_for("ptt"), Some(PTT));
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
        assert_eq!(locks.len(), 3, "one in with_bound, and REBINDING's for rebind and mute");
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

    /// One rebind at a time, and only `rebind` and `mute` take that turn:
    /// a mute can't come between a rebind's look at the keys and its swap.
    #[test]
    fn only_rebind_and_mute_take_the_rebinding_lock() {
        let src = code(include_str!("hotkeys.rs"));
        let uses = src.lines().filter(|l| !l.trim_start().starts_with("//")).filter(|l| l.contains("REBINDING"));
        assert_eq!(uses.count(), 3, "its static, rebind and mute");
        for call in ["pub fn rebind(", "pub fn mute("] {
            let at = src.find(call).unwrap();
            let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
            assert!(body.contains("let _one = REBINDING.lock()"), "{body}");
        }
    }

    /// What the app binds, reads and rebinds is the one list the handler
    /// finds its keys in: each of the app's calls hands its pure half `BOUND`.
    #[test]
    fn the_app_s_calls_use_the_one_bound_list() {
        let src = code(include_str!("hotkeys.rs"));
        for (call, inner) in [
            ("pub fn register(", "register_with(&Plugin(app), &BOUND, h, muted);"),
            ("pub fn id_of(", "id_in(&BOUND, shortcut)"),
            ("pub fn bound_now(", "bound_in(&BOUND)"),
            ("pub fn rebind(", "rebind_with(&Plugin(app), &BOUND, &s.hotkeys, action, accel,"),
            ("pub fn mute(", "mute_with(&Plugin(app), &BOUND, &h, on)"),
        ] {
            let at = src.find(call).unwrap_or_else(|| panic!("{call}"));
            let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
            assert!(body.contains(inner), "{body}");
        }
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
