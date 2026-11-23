//! The tray: Ask Fritz…, Show / Hide Fritz, Arrange / Lock, Mute
//! push-to-talk, Appearance ▸, Settings, Fritz's logs, Quit. Mute
//! push-to-talk is a check item built once and held, as Ask Fritz… is; its
//! tick follows `audio.muted`. The menu is rebuilt when Arrange starts or ends (the
//! machine's `RefreshTray`) and when Fritz is parked or shown, since its
//! items follow both. Show and Hide are separate ids, so a click does what
//! the label it showed said, whatever the state is by the time it lands.
//!
//! Ask Fritz… opens the prompt with the keyboard in it: a click on Fritz's
//! own menu is input his process received, so Windows lets the prompt take
//! the keyboard. It and the separator after it, like the Appearance submenu,
//! are built once and held by every rebuilt menu.
//!
//! The Appearance submenu is built once and held by every rebuilt menu; a
//! theme change (from here or the control window) only moves its ticks. muda
//! gives each menu object it makes an id that is never reused, and matches a
//! click by the id's low 16 bits, so every object built shortens the tray's
//! life: rebuilding the submenu's fourteen objects for each change would
//! spend them three times as fast.
//!
//! His menu (a right-click on him) is this menu, shown by popup.rs on the
//! tray's own window. A rebuild waits while any menu of Fritz's is open:
//! the check and the swap both run on the main thread, where a menu opens,
//! so one can't open between them.

use std::sync::atomic::{AtomicIsize, Ordering};

use serde_json::{json, Value};
use tauri::menu::{
    CheckMenuItem, CheckMenuItemBuilder, ContextMenu, Menu, MenuBuilder, MenuEvent, MenuItem, MenuItemBuilder,
    PredefinedMenuItem, Submenu, SubmenuBuilder,
};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, Wry};

use crate::overlay_machine::{ModeRequest, TRAY_LOCK_ID, TRAY_LOCK_LABEL};
use crate::overlay_windows;
use crate::settings::{self, Theme};

const TRAY_ID: &str = "fritz";
const ASK: &str = "ask";
const ASK_LABEL: &str = "Ask Fritz…";
const SHOW: &str = "show";
const HIDE: &str = "hide";
const ARRANGE: &str = "arrange";
const MUTE: &str = "mute";
const MUTE_LABEL: &str = "Mute push-to-talk";
const SETTINGS: &str = "settings";
const LOGS: &str = "logs";
const QUIT: &str = "quit";

/// The menu the tray holds now, as Win32 knows it, for his menu; 0 before
/// the first. Set on the main thread with the menu.
static HMENU: AtomicIsize = AtomicIsize::new(0);

/// The Appearance submenu's choices, as Settings › Appearance names them:
/// (the setting's value, the item's id, its label).
const MOODS: [(&str, &str, &str); 3] = [
    ("ink", "appearance:mood:ink", "Ink + candle"),
    ("dusk", "appearance:mood:dusk", "Dusk violet"),
    ("midnight", "appearance:mood:midnight", "Midnight"),
];
const ACCENTS: [(&str, &str, &str); 3] = [
    ("brand", "appearance:accent:brand", "Brand accent"),
    ("abyssal", "appearance:accent:abyssal", "Abyssal accent"),
    ("ember", "appearance:accent:ember", "Ember accent"),
];
const LIGHT: (&str, &str) = ("appearance:light", "Light theme");
/// (the level, the value a pick sets: the level's own, as Settings sets it,
/// the item's id, its label).
const MOTIONS: [(&str, f64, &str, &str); 3] = [
    ("functional", 0.0, "appearance:motion:functional", "Functional motion"),
    ("ambient", 0.6, "appearance:motion:ambient", "Ambient motion"),
    ("rich", 1.0, "appearance:motion:rich", "Rich motion"),
];

/// One check item in the Appearance submenu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppearanceItem {
    pub id: &'static str,
    pub label: &'static str,
    pub checked: bool,
}

/// The Appearance submenu for `t`: the mood, the accent, the light theme and
/// the motion level, each group checked where the theme stands.
pub fn appearance(t: &Theme) -> [Vec<AppearanceItem>; 4] {
    let pick = |group: &[(&'static str, &'static str, &'static str)], now: &str| {
        group
            .iter()
            .map(|&(value, id, label)| AppearanceItem { id, label, checked: value == now })
            .collect()
    };
    let level = settings::motion_level(t.motion);
    [
        pick(&MOODS, &t.mood),
        pick(&ACCENTS, &t.accent),
        vec![AppearanceItem { id: LIGHT.0, label: LIGHT.1, checked: t.theme == "light" }],
        MOTIONS
            .iter()
            .map(|&(value, _, id, label)| AppearanceItem { id, label, checked: value == level })
            .collect(),
    ]
}

/// The setting a click on Appearance item `id` changes, and to what, from
/// theme `t`: Light theme turns the light theme on or off; the others set
/// what they name. `None` for an id that isn't one of them.
pub fn appearance_change(id: &str, t: &Theme) -> Option<(&'static str, Value)> {
    if id == LIGHT.0 {
        let next = if t.theme == "light" { "dark" } else { "light" };
        return Some(("theme.theme", json!(next)));
    }
    if let Some(&(value, ..)) = MOODS.iter().find(|m| m.1 == id) {
        return Some(("theme.mood", json!(value)));
    }
    if let Some(&(value, ..)) = ACCENTS.iter().find(|a| a.1 == id) {
        return Some(("theme.accent", json!(value)));
    }
    MOTIONS
        .iter()
        .find(|m| m.2 == id)
        .map(|&(_, motion, ..)| ("theme.motion", json!(motion)))
}

/// Whether a change to setting `path` moves the menu's ticks: the theme's do.
pub fn follows(path: &str) -> bool {
    path == "theme" || path.starts_with("theme.")
}

/// Ask Fritz… and the separator after it, built once and held by every
/// rebuilt menu: a separator is a menu object with an id of its own too.
struct Ask {
    item: MenuItem<Wry>,
    after: PredefinedMenuItem<Wry>,
}

fn build_ask(app: &AppHandle) -> tauri::Result<Ask> {
    Ok(Ask {
        item: MenuItemBuilder::with_id(ASK, ASK_LABEL).build(app)?,
        after: PredefinedMenuItem::separator(app)?,
    })
}

/// Mute push-to-talk, built once and held by every rebuilt menu.
struct Mute {
    item: CheckMenuItem<Wry>,
}

fn build_mute(app: &AppHandle) -> tauri::Result<Mute> {
    Ok(Mute {
        item: CheckMenuItemBuilder::with_id(MUTE, MUTE_LABEL)
            .checked(settings::get().audio.muted)
            .build(app)?,
    })
}

/// The Appearance submenu, built once, and its items by id.
struct Look {
    submenu: Submenu<Wry>,
    items: Vec<(&'static str, CheckMenuItem<Wry>)>,
}

fn build_look(app: &AppHandle) -> tauri::Result<Look> {
    let mut sub = SubmenuBuilder::new(app, "Appearance");
    let mut items = Vec::new();
    for (i, group) in appearance(&settings::get().theme).iter().enumerate() {
        if i > 0 {
            sub = sub.separator();
        }
        for item in group {
            let check = CheckMenuItemBuilder::with_id(item.id, item.label)
                .checked(item.checked)
                .build(app)?;
            sub = sub.item(&check);
            items.push((item.id, check));
        }
    }
    Ok(Look { submenu: sub.build()?, items })
}

/// Move the Appearance ticks to where the theme stands, in place: no menu
/// object is made. Logs what is ticked now, as the menu itself says.
pub fn retick(app: &AppHandle) {
    let Some(look) = app.try_state::<Look>() else {
        return;
    };
    for item in appearance(&settings::get().theme).iter().flatten() {
        if let Some((_, check)) = look.items.iter().find(|(id, _)| *id == item.id) {
            if let Err(e) = check.set_checked(item.checked) {
                tracing::warn!(id = item.id, error = %e, "couldn't tick the tray's item");
            }
        }
    }
    let ticked: Vec<&str> = look
        .items
        .iter()
        .filter(|(_, c)| c.is_checked().unwrap_or(false))
        .map(|(id, _)| *id)
        .collect();
    tracing::info!(?ticked, "the tray's Appearance ticks follow the theme");
    if let Some(mute) = app.try_state::<Mute>() {
        let muted = settings::get().audio.muted;
        if let Err(e) = mute.item.set_checked(muted) {
            tracing::warn!(error = %e, "couldn't tick the tray's Mute push-to-talk");
        }
        tracing::info!(muted, "the tray's Mute push-to-talk follows the setting");
    }
}

/// The state-dependent items: (id, label) for Show / Hide, and for Arrange /
/// Lock when one is on offer. Arrange is only offered while he is on screen,
/// since the machine refuses it while he is hidden or gone.
pub fn items(
    open: bool,
    parked: bool,
    arranging: bool,
) -> (
    (&'static str, &'static str),
    Option<(&'static str, &'static str)>,
) {
    let shown = open && !parked;
    let show_hide = if shown {
        (HIDE, "Hide Fritz")
    } else {
        (SHOW, "Show Fritz")
    };
    let arrange = if arranging {
        Some((TRAY_LOCK_ID, TRAY_LOCK_LABEL))
    } else if shown {
        Some((ARRANGE, "Arrange Fritz…"))
    } else {
        None
    };
    (show_hide, arrange)
}

fn menu(app: &AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let status = overlay_windows::status(app);
    let ((sh_id, sh_label), arrange) =
        items(status.open, status.parked, overlay_windows::arranging(app));
    let mut m = MenuBuilder::new(app);
    // Ask Fritz… first, and its separator: the held ones, never built here.
    if let Some(ask) = app.try_state::<Ask>() {
        m = m.item(&ask.item).item(&ask.after);
    }
    m = m.text(sh_id, sh_label);
    if let Some((id, label)) = arrange {
        m = m.text(id, label);
    }
    // The held Mute push-to-talk, never built here.
    if let Some(mute) = app.try_state::<Mute>() {
        m = m.item(&mute.item);
    }
    m = m.separator();
    // The one Appearance submenu: a menu dropped for the next only detaches
    // it (muda removes a child submenu before destroying its parent).
    if let Some(look) = app.try_state::<Look>() {
        m = m.item(&look.submenu);
    }
    m.text(SETTINGS, "Settings…")
        .text(LOGS, "Open Fritz's logs")
        .separator()
        .text(QUIT, "Quit Mister Fritz")
        .build()
}

pub fn build(app: &AppHandle) -> tauri::Result<()> {
    app.manage(build_ask(app)?);
    app.manage(build_mute(app)?);
    app.manage(build_look(app)?);
    let held = menu(app)?;
    HMENU.store(held.hpopupmenu().unwrap_or(0), Ordering::SeqCst);
    let mut tray = TrayIconBuilder::with_id(TRAY_ID)
        .tooltip("Mister Fritz")
        .menu(&held)
        .show_menu_on_left_click(true)
        .on_menu_event(on_menu);
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// Rebuild the menu for the current state, on the main thread: inline
/// when already there, else posted.
pub fn refresh(app: &AppHandle) {
    let h = app.clone();
    let _ = app.run_on_main_thread(move || rebuild(&h));
}

/// Main thread. While a menu of Fritz's is open the rebuild only marks
/// itself due, and runs once the menu has closed (popup.rs).
fn rebuild(app: &AppHandle) {
    if !crate::popup::may_rebuild() {
        tracing::debug!("the tray menu waits to be rebuilt: a menu of Fritz's is open");
        return;
    }
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    match menu(app) {
        Ok(m) => {
            let held = m.hpopupmenu().unwrap_or(0);
            let _ = tray.set_menu(Some(m));
            HMENU.store(held, Ordering::SeqCst);
            // The first item's id is Ask Fritz…'s, the same at every rebuild.
            tracing::debug!("the tray menu was rebuilt (its first item's id {})", crate::windows_ext::menu_item_id(held, 0));
        }
        Err(e) => tracing::warn!(error = %e, "couldn't rebuild the tray menu"),
    }
}

/// The menu the tray holds now; 0 before the first.
pub fn hmenu() -> isize {
    HMENU.load(Ordering::SeqCst)
}

/// The tray's own hidden window (tray-icon's `tray_icon_app`), as an
/// integer; 0 when there is no tray. On the main thread it runs inline.
pub(crate) fn window(app: &AppHandle) -> isize {
    app.tray_by_id(TRAY_ID)
        .and_then(|t| t.with_inner_tray_icon(|i| i.window_handle() as isize).ok())
        .unwrap_or(0)
}

fn on_menu(app: &AppHandle, event: MenuEvent) {
    crate::logs::guard("tray menu", || on_menu_item(app, event.id().as_ref()));
}

/// Whether choosing `id` closes a prompt waiting for a click: every
/// choice but Ask Fritz…, which takes the keyboard into it. (A press on his
/// menu spares a waiting prompt so that its choice decides.)
fn closes_wait(id: &str) -> bool {
    id != ASK
}

fn on_menu_item(app: &AppHandle, id: &str) {
    if closes_wait(id) {
        let open = crate::prompt::waiting_open();
        if open != 0 {
            crate::prompt::close_later(app, crate::prompt::Why::Chose, open);
        }
    }
    match id {
        // The menu shows nothing that changes with the prompt: no rebuild.
        ASK => {
            crate::prompt::toggle(app, crate::prompt::Opener::Menu);
            return;
        }
        SHOW => crate::show_avatar(app),
        HIDE => {
            overlay_windows::set_parked(app, true);
        }
        ARRANGE => {
            if let Err(e) = overlay_windows::set_mode(app, ModeRequest::Arrange) {
                tracing::info!(error = %e, "arrange refused");
            }
        }
        TRAY_LOCK_ID => {
            let _ = overlay_windows::set_mode(app, ModeRequest::Locked);
        }
        // The other of what its tick showed, which is the setting; the tick
        // is put back where the setting says once it has changed. Nothing
        // else in the menu follows it: no rebuild.
        MUTE => {
            crate::set_muted(app, !settings::get().audio.muted);
            return;
        }
        SETTINGS => crate::show_control(app),
        LOGS => {
            let _ = std::process::Command::new("explorer")
                .arg(crate::settings::app_dir())
                .spawn();
        }
        QUIT => crate::quit(app),
        _ => {
            if let Some((path, value)) = appearance_change(id, &settings::get().theme) {
                if let Err(e) = crate::change_setting(app, path, value) {
                    tracing::warn!(path, error = %e, "the tray's change was refused");
                }
                // A check item ticks or unticks itself when clicked: put
                // every tick where the settings say, whatever the click did.
                retick(app);
                return;
            }
        }
    }
    refresh(app);
}

#[cfg(test)]
mod tests {
    use super::{
        appearance, appearance_change, follows, items, ACCENTS, ARRANGE, ASK, ASK_LABEL, HIDE, LIGHT, LOGS,
        MOODS, MOTIONS, MUTE, MUTE_LABEL, QUIT, SETTINGS, SHOW, TRAY_ID,
    };
    use crate::overlay_machine::{TRAY_LOCK_ID, TRAY_LOCK_LABEL};
    use crate::settings::{self, Theme};
    use serde_json::{json, Value};

    #[test]
    fn the_menu_says_what_a_click_does() {
        assert_eq!(items(true, false, false).0, (HIDE, "Hide Fritz"));
        assert_eq!(items(true, true, false).0, (SHOW, "Show Fritz"));
        // Closed (Alt+F4, or the watchdog): Show unparks and reopens him.
        assert_eq!(items(false, false, false).0, (SHOW, "Show Fritz"));
        assert_eq!(items(false, true, false).0, (SHOW, "Show Fritz"));
    }

    /// Arrange only while he is on screen (the machine refuses it otherwise);
    /// Lock whenever arranging.
    #[test]
    fn arrange_is_offered_only_when_it_would_work() {
        assert_eq!(items(true, false, false).1.map(|i| i.0), Some(ARRANGE));
        assert_eq!(items(true, true, false).1, None, "hidden");
        assert_eq!(items(false, false, false).1, None, "closed");
        assert_eq!(
            items(true, false, true).1,
            Some((TRAY_LOCK_ID, TRAY_LOCK_LABEL))
        );
    }

    fn theme(theme: &str, mood: &str, accent: &str, motion: f32) -> Theme {
        Theme {
            theme: theme.into(),
            mood: mood.into(),
            accent: accent.into(),
            motion,
            ..Theme::default()
        }
    }

    fn checked(group: &[super::AppearanceItem]) -> Vec<&'static str> {
        group.iter().filter(|i| i.checked).map(|i| i.id).collect()
    }

    /// The submenu offers exactly what the settings accept: every mood and
    /// accent, and the two themes as Light theme on or off.
    #[test]
    fn appearance_offers_what_the_settings_accept() {
        let offered = |group: &[(&'static str, &'static str, &'static str)]| {
            let mut v: Vec<&str> = group.iter().map(|g| g.0).collect();
            v.sort_unstable();
            v
        };
        let sorted = |group: &[&'static str]| {
            let mut v = group.to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(offered(&MOODS), sorted(&settings::MOODS));
        assert_eq!(offered(&ACCENTS), sorted(&settings::ACCENTS));
        assert_eq!(sorted(&settings::THEMES), ["dark", "light"], "Light theme is on or off");
    }

    /// Each group ticks exactly the one the theme is on, for every value the
    /// settings accept, and the motion group the level `theme.motion` stands
    /// for (the fixture's snap rows, which theme.test.ts holds the pages to).
    #[test]
    fn appearance_ticks_where_the_theme_stands() {
        for mood in settings::MOODS {
            for accent in settings::ACCENTS {
                for th in settings::THEMES {
                    let t = theme(th, mood, accent, 0.6);
                    let [moods, accents, lights, motions] = appearance(&t);
                    assert_eq!(checked(&moods), [format!("appearance:mood:{mood}")]);
                    assert_eq!(checked(&accents), [format!("appearance:accent:{accent}")]);
                    assert_eq!(lights.len(), 1);
                    assert_eq!(lights[0].checked, th == "light");
                    assert_eq!(checked(&motions), ["appearance:motion:ambient"]);
                }
            }
        }
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        for row in fixture["motionSnap"].as_array().unwrap() {
            let m = row[0].as_f64().unwrap() as f32;
            let t = theme("dark", "ink", "brand", m);
            let want = format!("appearance:motion:{}", row[1].as_str().unwrap());
            assert_eq!(checked(&appearance(&t)[3]), [want.as_str()], "{m}");
        }
        // Every pick is one the settings take.
        let d = Theme::default();
        for item in appearance(&d).iter().flatten() {
            let (path, value) = appearance_change(item.id, &d).unwrap();
            settings::with_value(&settings::Settings::default(), path, value)
                .unwrap_or_else(|e| panic!("{}: {e}", item.id));
        }
    }

    /// A click sets what it names; Light theme turns the light theme on, or
    /// off; a motion item sets its level's own value, as Settings does.
    #[test]
    fn a_click_changes_what_it_says() {
        let d = theme("dark", "ink", "brand", 0.6);
        for (value, id, _) in MOODS {
            assert_eq!(appearance_change(id, &d), Some(("theme.mood", json!(value))));
        }
        for (value, id, _) in ACCENTS {
            assert_eq!(appearance_change(id, &d), Some(("theme.accent", json!(value))));
        }
        assert_eq!(appearance_change(LIGHT.0, &d), Some(("theme.theme", json!("light"))));
        let l = theme("light", "ink", "brand", 0.6);
        assert_eq!(appearance_change(LIGHT.0, &l), Some(("theme.theme", json!("dark"))));
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let levels = fixture["motionLevels"].as_object().unwrap();
        assert_eq!(levels.len(), MOTIONS.len());
        for (level, _, id, _) in MOTIONS {
            let (path, value) = appearance_change(id, &d).unwrap();
            assert_eq!(path, "theme.motion");
            assert_eq!(value.as_f64(), levels[level].as_f64(), "{level}");
            // and that value is the level it says
            let m = levels[level].as_f64().unwrap() as f32;
            assert_eq!(settings::motion_level(m), level);
        }
        for other in ["", "appearance:", "appearance:mood:violet", "appearance:light:x", SHOW, QUIT] {
            assert_eq!(appearance_change(other, &d), None, "{other}");
        }
    }

    /// Every id in the menu is its own: no Appearance id is taken for
    /// another item, or another's for one of them.
    #[test]
    fn every_menu_id_is_unique() {
        let mut ids: Vec<&str> = vec![ASK, SHOW, HIDE, ARRANGE, TRAY_LOCK_ID, MUTE, SETTINGS, LOGS, QUIT, TRAY_ID];
        ids.extend(appearance(&Theme::default()).iter().flatten().map(|i| i.id));
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "{ids:?}");
        let labels: Vec<&str> = appearance(&Theme::default()).iter().flatten().map(|i| i.label).collect();
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), labels.len(), "{labels:?}");
    }

    /// A rebuild runs on the main thread, and checks for an open menu first.
    #[test]
    fn a_rebuild_checks_for_an_open_menu_first() {
        let src = include_str!("tray.rs").replace("\r\n", "\n");
        let at = src.find("pub fn refresh(").unwrap();
        let refresh = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(refresh.contains("run_on_main_thread(move || rebuild(&h))"), "{refresh}");
        let at = src.find("fn rebuild(").unwrap();
        let rebuild = &src[at..at + src[at..].find("\n}\n").unwrap()];
        let first = rebuild[rebuild.find('{').unwrap() + 1..].trim_start();
        assert!(
            first.starts_with("if !crate::popup::may_rebuild() {\n        tracing::debug!(\"the tray menu waits to be rebuilt: a menu of Fritz's is open\");\n        return;\n    }"),
            "{first}"
        );
        assert!(rebuild.find("HMENU.store(held").unwrap() > rebuild.find("tray.set_menu(Some(m))").unwrap());
    }

    /// Ask Fritz… and its separator are built once, at the tray's build,
    /// and every menu holds those two, first: a rebuild spends no id on
    /// them.
    #[test]
    fn ask_is_built_once_and_held_first() {
        let src = include_str!("tray.rs").replace("\r\n", "\n");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        let body = |sig: &str| {
            let at = code.find(sig).unwrap_or_else(|| panic!("no {sig}"));
            &code[at..at + code[at..].find("\n}\n").unwrap()]
        };
        for built in ["MenuItemBuilder::with_id(ASK, ASK_LABEL)", "PredefinedMenuItem::separator(app)"] {
            assert_eq!(code.matches(built).count(), 1, "{built}");
            assert!(body("fn build_ask(").contains(built), "{built} only in build_ask");
        }
        // Its id and label nowhere else: no other builder makes another.
        let words = |w: &str| {
            code.match_indices(w)
                .filter(|&(i, _)| {
                    let after = code[i + w.len()..].chars().next().unwrap_or(' ');
                    let before = code[..i].chars().next_back().unwrap_or(' ');
                    !(after.is_alphanumeric() || after == '_' || before.is_alphanumeric() || before == '_')
                })
                .count()
        };
        assert_eq!(words("ASK"), 4, "its const, build_ask, closes_wait, and the click's arm");
        assert_eq!(words("ASK_LABEL"), 2, "its const and build_ask");
        let menu = body("fn menu(");
        let held = menu.find("m = m.item(&ask.item).item(&ask.after);").expect("held, appended");
        assert!(held < menu.find("m = m.text(sh_id, sh_label);").unwrap(), "first");
        let build = body("pub fn build(");
        assert!(build.find("app.manage(build_ask(app)?);").unwrap() < build.find("let held = menu(app)?;").unwrap());
        assert_eq!(code.matches("build_ask(").count(), 2, "its definition, and the one call, in build");
        assert_eq!(ASK_LABEL, "Ask Fritz…");
    }

    /// A click on it opens the prompt from his menu, and rebuilds nothing;
    /// any other choice first closes a prompt waiting for a click.
    #[test]
    fn ask_opens_the_prompt_and_never_rebuilds() {
        let src = include_str!("tray.rs").replace("\r\n", "\n");
        let at = src.find("fn on_menu_item(").unwrap();
        let f = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(f.contains("        ASK => {\n            crate::prompt::toggle(app, crate::prompt::Opener::Menu);\n            return;\n        }"));
        let close = f
            .find("    if closes_wait(id) {\n        let open = crate::prompt::waiting_open();\n        if open != 0 {\n            crate::prompt::close_later(app, crate::prompt::Why::Chose, open);\n        }\n    }")
            .expect("a choice closes a waiting prompt");
        assert!(close < f.find("    match id {").unwrap(), "before the choice runs");
        assert!(!super::closes_wait(ASK));
        for other in [SHOW, HIDE, ARRANGE, TRAY_LOCK_ID, SETTINGS, LOGS, QUIT, LIGHT.0, MOODS[0].1] {
            assert!(super::closes_wait(other), "{other}");
        }
    }

    /// Mute push-to-talk is built once, at the tray's build, ticked as the
    /// setting stands, and held by every menu after Show / Hide and Arrange;
    /// retick moves its tick to the setting.
    #[test]
    fn mute_is_built_once_and_held_after_show_and_arrange() {
        let src = include_str!("tray.rs").replace("\r\n", "\n");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        let body = |sig: &str| {
            let at = code.find(sig).unwrap_or_else(|| panic!("no {sig}"));
            &code[at..at + code[at..].find("\n}\n").unwrap()]
        };
        let built = "CheckMenuItemBuilder::with_id(MUTE, MUTE_LABEL)\n            .checked(settings::get().audio.muted)";
        assert_eq!(code.matches("CheckMenuItemBuilder::with_id(MUTE").count(), 1);
        assert!(body("fn build_mute(").contains(built));
        assert_eq!(code.matches("build_mute(").count(), 2, "its definition, and the one call, in build");
        let build = body("pub fn build(");
        assert!(build.find("app.manage(build_mute(app)?);").unwrap() < build.find("let held = menu(app)?;").unwrap());
        let menu = body("fn menu(");
        let held = menu.find("m = m.item(&mute.item);").expect("held, appended");
        assert!(held > menu.find("m = m.text(id, label);").unwrap(), "after Arrange");
        assert!(held < menu.find("m = m.separator();").unwrap(), "before Appearance");
        assert!(body("pub fn retick(").contains("mute.item.set_checked(muted)"));
        assert_eq!(MUTE_LABEL, "Mute push-to-talk");
    }

    /// A click on it sets the other of what the setting says, and rebuilds
    /// nothing; it closes a waiting prompt as any choice but Ask does.
    #[test]
    fn mute_sets_the_other_and_never_rebuilds() {
        let src = include_str!("tray.rs").replace("\r\n", "\n");
        let at = src.find("fn on_menu_item(").unwrap();
        let f = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(f.contains("        MUTE => {\n            crate::set_muted(app, !settings::get().audio.muted);\n            return;\n        }"), "{f}");
        assert!(super::closes_wait(MUTE));
    }

    #[test]
    fn the_tray_follows_the_theme_only() {
        for path in ["theme", "theme.mood", "theme.motion", "theme.motionUser"] {
            assert!(follows(path), "{path}");
        }
        for path in ["avatar.lookatCursor", "themes", "theme_x", "diagnostics.foregroundLog", ""] {
            assert!(!follows(path), "{path}");
        }
    }
}
