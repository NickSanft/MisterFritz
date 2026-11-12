//! Mister Fritz's body on the Windows desktop (plan 14): one persistent,
//! transparent, click-through, never-focused avatar window with a WebGL VRM
//! in it, a tray, and global hotkeys. Windows only.
//!
//! What it keeps lives in `%APPDATA%\com.divora.fritz\`: `settings.json`,
//! `startup.log` and `crash.log` (`logs.rs`), `logs\fritz.log` (this run's),
//! and, if the owner turns it on, `diagnostics\foreground.log`. A VRoid
//! export at `avatar\fritz.vrm`, or wherever `avatar.vrmPath` says, replaces
//! the committed placeholder.

mod avatar_state;
mod cover;
mod cursor;
mod foreground_log;
mod hotkeys;
mod logs;
mod non_rude;
mod overlay_machine;
mod overlay_windows;
mod power;
mod settings;
mod tray;
mod windows_ext;

use std::fs::File;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, RunEvent, WindowEvent};
use tauri_plugin_global_shortcut::{Shortcut, ShortcutEvent, ShortcutState};

use overlay_machine::{
    HandshakeStage, ModeRequest, OverlayRect, OverlayStatus, Refusal, BAD_STAGE, OVERLAY_LABEL,
};
use settings::Settings;

/// The largest VRM the page is handed. VRoid exports are 5–50 MB.
const MAX_VRM_BYTES: u64 = 256 * 1024 * 1024;

/// Fritz's first spot: the bottom-right of the primary work area, this many
/// logical pixels in, and this big.
const FIRST_SIZE: (f64, f64) = (300.0, 460.0);
const FIRST_MARGIN: f64 = 24.0;

/// Sent to every window when a setting changes, with all of them.
pub const SETTINGS_EVENT: &str = "settings:changed";

fn data_dir() -> PathBuf {
    settings::app_dir()
}

/// The model the page draws: `avatar.vrmPath`, or `avatar\fritz.vrm` in the
/// data folder.
fn model_path(s: &Settings) -> PathBuf {
    s.avatar
        .vrm_path
        .as_deref()
        .map_or_else(|| data_dir().join("avatar").join("fritz.vrm"), PathBuf::from)
}

/// Where Spike 0's foreground logger writes, if the owner has turned it on:
/// it records other apps' window titles, so it is off unless asked for.
fn foreground_log_path(s: &Settings) -> Option<PathBuf> {
    s.diagnostics
        .foreground_log
        .then(|| data_dir().join("diagnostics").join("foreground.log"))
}

/// The first placement, in physical pixels.
fn first_rect() -> Option<OverlayRect> {
    let (work, dpi) = windows_ext::primary_work_area()?;
    let s = f64::from(dpi) / 96.0;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (w, h, m) = (
        (FIRST_SIZE.0 * s) as u32,
        (FIRST_SIZE.1 * s) as u32,
        (FIRST_MARGIN * s) as i32,
    );
    Some(OverlayRect {
        x: work.right - i32::try_from(w).ok()? - m,
        y: work.bottom - i32::try_from(h).ok()? - m,
        width: w,
        height: h,
    })
}

/// Open (or reopen) the avatar window where Fritz last stood, brought onto a
/// live monitor if that spot's monitor is gone.
pub(crate) fn open_avatar(app: &AppHandle) {
    let rect = overlay_windows::standing(app)
        .or_else(avatar_state::saved_rect)
        .map(overlay_windows::onto_screen)
        .or_else(first_rect);
    let h = app.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = overlay_windows::open(h.clone(), rect).await {
            tracing::warn!(error = %e, "couldn't open the avatar");
        }
        tray::refresh(&h);
    });
}

/// Show Fritz, whatever state he is in: unparked, and reopened if his window
/// is gone (Alt+F4 in Arrange, the watchdog). Unparking first means a
/// reopened window isn't built off screen.
pub(crate) fn show_avatar(app: &AppHandle) {
    overlay_windows::set_parked(app, false);
    if !overlay_windows::status(app).open {
        open_avatar(app);
    }
}

/// The tray's Quit: save, take the shell's mark off, and go.
pub(crate) fn quit(app: &AppHandle) {
    avatar_state::save_now(app);
    logs::startup("quit from the tray");
    overlay_windows::before_exit(app);
    app.exit(0);
}

/// Main thread, inside WM_HOTKEY (presses) or global-hotkey's poller
/// (releases).
fn on_shortcut(app: &AppHandle, shortcut: &Shortcut, event: ShortcutEvent) {
    logs::guard("hotkey", || {
        let pressed = event.state == ShortcutState::Pressed;
        let Some(id) = hotkeys::id_of(shortcut) else {
            return;
        };
        if id == hotkeys::PARK {
            if pressed {
                let status = overlay_windows::status(app);
                if status.open && !status.parked {
                    overlay_windows::set_parked(app, true);
                } else {
                    show_avatar(app);
                }
            }
        } else {
            overlay_windows::hotkey(app, id, pressed);
        }
    });
}

fn init_logging() {
    let dir = data_dir().join("logs");
    let _ = std::fs::create_dir_all(&dir);
    // One session's log: started fresh each run.
    let Ok(file) = File::create(dir.join("fritz.log")) else {
        return;
    };
    let level = if cfg!(debug_assertions) {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    let _ = tracing_subscriber::fmt()
        .with_writer(Mutex::new(file))
        .with_ansi(false)
        .with_max_level(level)
        .try_init();
}

#[tauri::command]
async fn avatar_handshake(
    app: AppHandle,
    instance: u32,
    stage: String,
) -> Result<OverlayStatus, String> {
    let stage = HandshakeStage::parse(&stage).ok_or_else(|| BAD_STAGE.to_owned())?;
    Ok(overlay_windows::handshake(&app, instance, stage))
}

#[tauri::command]
async fn avatar_beat(app: AppHandle, instance: u32, gen: u32) {
    overlay_windows::alive(&app, instance, gen);
}

#[tauri::command]
async fn set_avatar_mode(app: AppHandle, mode: String) -> Result<OverlayStatus, String> {
    let req = ModeRequest::parse(&mode).ok_or_else(|| Refusal::BadMode.message().to_owned())?;
    let status = overlay_windows::set_mode(&app, req);
    let h = app.clone();
    let _ = app.run_on_main_thread(move || tray::refresh(&h));
    status
}

#[tauri::command]
async fn avatar_status(app: AppHandle) -> OverlayStatus {
    overlay_windows::status(&app)
}

/// The owner's VRoid export, as raw bytes; an error ("none") when there is
/// none, and the page falls back to the placeholder it ships with. Which
/// file it was is kept for the page's report that follows (`HANDED`).
#[tauri::command]
async fn avatar_vrm() -> Result<tauri::ipc::Response, String> {
    let path = model_path(&settings::get());
    let stamp = stamp(&path);
    *HANDED.lock().unwrap_or_else(|e| e.into_inner()) = Some(ModelFile {
        path: path.clone(),
        stamp,
    });
    let size = stamp.ok_or_else(|| "none".to_owned())?.len;
    if size > MAX_VRM_BYTES {
        return Err(format!(
            "{} is larger than {MAX_VRM_BYTES} bytes",
            path.display()
        ));
    }
    std::fs::read(&path)
        .map(tauri::ipc::Response::new)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_settings() -> Settings {
    settings::get()
}

/// Set one setting by its camelCase path; every window hears the result, in
/// the order the changes were made.
#[tauri::command]
async fn set_setting(
    app: AppHandle,
    path: String,
    value: serde_json::Value,
) -> Result<Settings, String> {
    let after = settings::set(&path, value, |after| {
        let _ = app.emit(SETTINGS_EVENT, after);
    })?;
    tracing::info!(path, "setting changed");
    Ok(after)
}

/// Whether the display is on: for a page that starts after the last
/// `power:display`.
#[tauri::command]
async fn power_display() -> bool {
    power::display_on()
}

/// The cursor now: for a page that starts after the last `cursor` event
/// (sent only on a move, so a still cursor would otherwise never be looked
/// at). Where it is even while he is parked or the display is off, so the
/// page has it when he is seen again; `active: false` only when look-at is
/// off or there is no window.
#[tauri::command]
async fn cursor_now(app: AppHandle) -> cursor::Cursor {
    cursor::now(&app)
}

/// The control window (tray > Settings…), declared in tauri.conf.json and
/// shown on demand.
pub const CONTROL_LABEL: &str = "control";
/// Sent to the avatar's page to load its model again (the control window's
/// "Load it again", after the owner drops a new export in).
pub const RELOAD_EVENT: &str = "avatar:reload";
/// Shows the control window at start: for a test run, which can't click the
/// tray.
const SHOW_SETTINGS_ENV: &str = "FRITZ_SHOW_SETTINGS";

/// Show the control window and give it the keyboard (it is an ordinary
/// window; only the avatar's must never take focus). By SetForegroundWindow,
/// which Windows allows from the tray's click and refuses otherwise (the
/// window then waits behind, its taskbar button flashing); never tao's
/// `set_focus`, which taps Alt into the owner's app to get its way.
pub(crate) fn show_control(app: &AppHandle) {
    let Some(w) = app.get_webview_window(CONTROL_LABEL) else {
        tracing::warn!("there is no control window");
        return;
    };
    let _ = w.show();
    let _ = w.unminimize();
    if let Ok(h) = w.hwnd() {
        windows_ext::try_foreground(h.0 as isize);
    }
}

/// Whether closing the window `label` only hides it: the control window
/// does, so the tray can bring it back; the avatar's has its own shell.
pub fn hides_on_close(label: &str) -> bool {
    label == CONTROL_LABEL
}

/// Which model the page draws, for the control window.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// `"setting"` when `avatar.vrmPath` names it, `"folder"` for
    /// `avatar\fritz.vrm` in the data folder.
    pub from: &'static str,
    pub path: String,
    /// Whether there is a file there to load (if not, he wears the
    /// placeholder).
    pub found: bool,
    pub bytes: Option<u64>,
    /// Larger than the page is handed: he wears the placeholder.
    pub too_big: bool,
    /// What the avatar's page last loaded, as it said (`avatar_loaded`):
    /// `"export"` or `"placeholder"`; `None` before its first load.
    pub wearing: Option<&'static str>,
    /// Why the export didn't load, when the page fell back to the
    /// placeholder for the file there now (it isn't a VRM, say).
    pub load_error: Option<String>,
    /// The file there now isn't the one the page last loaded (an export
    /// dropped in or taken away, or `avatar.vrmPath` changed): what he
    /// wears is from before, and loading again wears what's there.
    pub changed: bool,
}

/// A model file as it was seen: its size and when it was last written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    pub len: u64,
    pub modified: Option<std::time::SystemTime>,
}

/// The model file at `path` (a file, not a folder), if there is one.
fn stamp(path: &std::path::Path) -> Option<Stamp> {
    std::fs::metadata(path)
        .ok()
        .filter(std::fs::Metadata::is_file)
        .map(|m| Stamp { len: m.len(), modified: m.modified().ok() })
}

/// Which file a load was of: where the page was told to look, and what was
/// there (`None`: nothing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub path: PathBuf,
    pub stamp: Option<Stamp>,
}

/// What `avatar_vrm` last handed the page, taken by the report that
/// follows it (the page loads one model at a time).
static HANDED: std::sync::Mutex<Option<ModelFile>> = std::sync::Mutex::new(None);

/// What the avatar's page last loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wearing {
    pub what: &'static str,
    pub error: Option<String>,
    /// The file it was of; `None` when the page never asked for one, so it
    /// matches no file.
    pub file: Option<ModelFile>,
}

/// The longest load error kept, in characters: it comes from the page and
/// goes into a window.
const LOAD_ERROR_CHARS: usize = 300;

/// The page's report, checked: `"export"` or `"placeholder"`, and an error
/// cut to printable text of a sensible length.
pub fn wearing_from(what: &str, error: Option<String>) -> Result<Wearing, String> {
    let what = match what {
        "export" => "export",
        "placeholder" => "placeholder",
        other => return Err(format!("{other:?} isn't a model the page wears")),
    };
    let error = error
        .map(|e| e.chars().filter(|c| !c.is_control()).take(LOAD_ERROR_CHARS).collect::<String>())
        .filter(|e| !e.trim().is_empty());
    Ok(Wearing { what, error, file: None })
}

/// The page's report, checked, and tied to the file `avatar_vrm` handed it
/// for that load.
fn take_report(what: &str, error: Option<String>) -> Result<Wearing, String> {
    let mut w = wearing_from(what, error)?;
    w.file = HANDED.lock().unwrap_or_else(|e| e.into_inner()).take();
    Ok(w)
}

/// The page's last report, kept for `avatar_model`.
static WEARING: std::sync::Mutex<Option<Wearing>> = std::sync::Mutex::new(None);

/// Sent to every window when the avatar's page has loaded a model.
pub const LOADED_EVENT: &str = "avatar:loaded";

/// The model for the control window: the file there now (`stamp` looks),
/// and the page's last report, whose error counts only if it was of that
/// same file.
fn model_info(
    s: &Settings,
    stamp: impl FnOnce(&std::path::Path) -> Option<Stamp>,
    wearing: Option<&Wearing>,
) -> ModelInfo {
    let path = model_path(s);
    let now = stamp(&path);
    let bytes = now.map(|st| st.len);
    let display = path.display().to_string();
    let here = ModelFile { path, stamp: now };
    let fresh = wearing.filter(|w| w.file.as_ref() == Some(&here));
    ModelInfo {
        from: if s.avatar.vrm_path.is_some() { "setting" } else { "folder" },
        path: display,
        found: bytes.is_some(),
        bytes,
        too_big: bytes.is_some_and(|b| b > MAX_VRM_BYTES),
        wearing: wearing.map(|w| w.what),
        load_error: fresh.and_then(|w| w.error.clone()),
        changed: wearing.is_some() && fresh.is_none(),
    }
}

/// The folders the control window opens: `"logs"` is the data folder (the
/// logs and settings are in it), `"avatar"` where his model goes.
fn folder(which: &str) -> Option<PathBuf> {
    match which {
        "logs" => Some(data_dir()),
        "avatar" => Some(data_dir().join("avatar")),
        _ => None,
    }
}

/// Hide or show him, from the control window: as the tray's Hide and Show.
#[tauri::command]
async fn set_parked(app: AppHandle, on: bool) -> OverlayStatus {
    if on {
        overlay_windows::set_parked(&app, true);
    } else {
        show_avatar(&app);
    }
    let h = app.clone();
    let _ = app.run_on_main_thread(move || tray::refresh(&h));
    overlay_windows::status(&app)
}

#[tauri::command]
async fn avatar_model() -> ModelInfo {
    let wearing = WEARING.lock().unwrap_or_else(|e| e.into_inner()).clone();
    model_info(&settings::get(), stamp, wearing.as_ref())
}

/// The avatar's page says what it has loaded (after every load): kept for
/// `avatar_model`, and every window is told, so the control window can say
/// when an export didn't load.
#[tauri::command]
async fn avatar_loaded(app: AppHandle, wearing: String, error: Option<String>) -> Result<(), String> {
    let w = take_report(&wearing, error)?;
    match &w.error {
        Some(e) => tracing::warn!(wearing = w.what, error = %e, "the avatar's model"),
        None => tracing::info!(wearing = w.what, "the avatar's model"),
    }
    *WEARING.lock().unwrap_or_else(|e| e.into_inner()) = Some(w);
    let _ = app.emit(LOADED_EVENT, ());
    Ok(())
}

/// Open one of Fritz's folders in Explorer (made first if it isn't there).
#[tauri::command]
async fn open_folder(which: String) -> Result<(), String> {
    let dir = folder(&which).ok_or_else(|| format!("there is no folder {which:?}"))?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::process::Command::new("explorer")
        .arg(&dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Have the avatar's page load its model again.
#[tauri::command]
async fn reload_model(app: AppHandle) {
    let _ = app.emit_to(OVERLAY_LABEL, RELOAD_EVENT, ());
}

pub fn run() {
    logs::install_panic_hook();
    logs::startup(&format!(
        "=== start {} pid {} exe {}",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
        std::env::current_exe().map_or_else(|_| "?".to_owned(), |p| p.display().to_string())
    ));
    windows_ext::set_dpi_awareness();
    // Before the app is built, so a window's first get_settings sees them.
    let (loaded, repaired) = settings::load();
    logs::startup(&format!("settings: {}", loaded.describe()));
    for problem in &repaired {
        logs::startup(&format!("settings: {problem}; put back to its default"));
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().with_handler(on_shortcut).build())
        .invoke_handler(tauri::generate_handler![
            avatar_handshake,
            avatar_beat,
            set_avatar_mode,
            avatar_status,
            avatar_vrm,
            get_settings,
            set_setting,
            power_display,
            cursor_now,
            set_parked,
            avatar_model,
            open_folder,
            reload_model,
            avatar_loaded
        ])
        .setup(move |app| {
            let handle = app.handle().clone();
            init_logging();
            tracing::info!(version = env!("CARGO_PKG_VERSION"), "Mister Fritz starting");
            tracing::info!(settings = %loaded.describe(), "settings");
            for problem in &repaired {
                tracing::warn!(%problem, "a setting was put back to its default");
            }
            let s = settings::get();
            overlay_windows::init(app);
            if s.avatar.parked {
                overlay_windows::set_parked(&handle, true);
            }
            tray::build(&handle)?;
            hotkeys::register(&handle, &s.hotkeys);
            if let Some(path) = foreground_log_path(&s) {
                foreground_log::start(handle.clone(), path);
            } else {
                tracing::info!("the foreground log is off (diagnostics.foregroundLog)");
            }
            cover::start(handle.clone());
            power::start(handle.clone());
            avatar_state::start_saver(handle.clone());
            open_avatar(&handle);
            cursor::start(handle.clone());
            if std::env::var_os(SHOW_SETTINGS_ENV).is_some() {
                show_control(&handle);
            }
            logs::startup("started");
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == OVERLAY_LABEL {
                overlay_windows::on_window_event(window.app_handle(), event);
            } else if hides_on_close(window.label()) {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("Mister Fritz couldn't start");
    app.run(|_app, event| match event {
        // The avatar window closing (Alt+F4 in Arrange) leaves the tray up;
        // only Quit ends the app.
        RunEvent::ExitRequested {
            code: None, api, ..
        } => api.prevent_exit(),
        RunEvent::Exit => logs::startup("stopped"),
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    /// The key Tauri looks up for a parameter: `#[tauri::command]` defaults to
    /// lowerCamelCase (`arrange_gen` is sent as `arrangeGen`), so the fixture
    /// holds the wire keys, and this maps the Rust names onto them.
    fn wire_key(param: &str) -> String {
        let mut parts = param.split('_').filter(|s| !s.is_empty());
        let mut key = parts.next().unwrap_or_default().to_owned();
        for part in parts {
            let mut c = part.chars();
            if let Some(first) = c.next() {
                key.extend(first.to_uppercase());
                key.push_str(c.as_str());
            }
        }
        key
    }

    /// The owner's model: `avatar.vrmPath` when set, else `avatar\fritz.vrm`
    /// in the data folder.
    #[test]
    fn the_model_is_where_the_settings_say() {
        let mut s = crate::settings::Settings::default();
        assert_eq!(
            super::model_path(&s),
            crate::settings::app_dir().join("avatar").join("fritz.vrm")
        );
        s.avatar.vrm_path = Some("D:\\models\\fritz-v2.vrm".into());
        assert_eq!(super::model_path(&s), std::path::PathBuf::from("D:\\models\\fritz-v2.vrm"));
    }

    /// The fixture's events are the shell's: each has its Rust half here or
    /// in overlay_machine.rs (status), and its page half in the TS tests.
    #[test]
    fn the_events_are_the_wire_contract() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let events = fixture["events"].as_object().unwrap();
        assert_eq!(events["settings"], super::SETTINGS_EVENT);
        assert_eq!(events["power"], crate::power::DISPLAY_EVENT);
        assert_eq!(events["status"], crate::overlay_machine::OVERLAY_STATUS_EVENT);
        assert_eq!(events["cursor"], crate::cursor::CURSOR_EVENT);
        assert_eq!(events["reload"], super::RELOAD_EVENT);
        assert_eq!(events["loaded"], super::LOADED_EVENT);
        assert_eq!(events.len(), 6, "an event with no Rust half: {events:?}");
    }

    #[test]
    fn only_the_control_window_hides_on_close() {
        assert!(super::hides_on_close("control"));
        assert!(!super::hides_on_close("avatar"));
        assert!(!super::hides_on_close("prompt"));
    }

    /// The control window is the one tauri.conf.json declares, so the tray
    /// and the close rule find it by that label.
    #[test]
    fn the_control_window_is_declared() {
        let conf: Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let windows = conf["app"]["windows"].as_array().unwrap();
        let control = windows.iter().find(|w| w["label"] == super::CONTROL_LABEL).unwrap();
        assert_eq!(control["url"], "control.html");
        assert_eq!(control["visible"], false, "shown from the tray, never at start");
        // Hidden is not enough: a webview made with focus takes the keyboard
        // from the owner's app at every start, though no one can see it.
        assert_eq!(control["focus"], false, "a hidden window must not take the keyboard");
    }

    /// A file of `len` bytes, last written at second `at`.
    fn file(len: u64, at: u64) -> Option<super::Stamp> {
        Some(super::Stamp {
            len,
            modified: Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(at)),
        })
    }

    #[test]
    fn the_control_window_says_which_model_he_wears() {
        let mut s = crate::settings::Settings::default();
        let found = super::model_info(&s, |_| file(12_000_000, 1), None);
        assert_eq!(found.from, "folder");
        assert!(found.found && !found.too_big);
        assert_eq!(found.bytes, Some(12_000_000));
        assert!(found.path.ends_with("fritz.vrm"), "{}", found.path);
        let none = super::model_info(&s, |_| None, None);
        assert!(!none.found && !none.too_big && none.bytes.is_none());
        s.avatar.vrm_path = Some(r"D:\models\big.vrm".into());
        let big = super::model_info(&s, |_| file(super::MAX_VRM_BYTES + 1, 1), None);
        assert_eq!(big.from, "setting");
        assert!(big.found && big.too_big);
        assert_eq!(big.path, r"D:\models\big.vrm");
    }

    /// The control window reads these keys, and asks for these folders.
    #[test]
    fn the_model_and_folders_are_the_wire_contract() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let info = serde_json::to_value(super::model_info(&crate::settings::Settings::default(), |_| None, None)).unwrap();
        let mut keys: Vec<&str> = info.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let want: Vec<&str> =
            fixture["modelInfoKeys"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
        assert_eq!(keys, want);
        for which in fixture["folders"].as_array().unwrap() {
            assert!(super::folder(which.as_str().unwrap()).is_some(), "{which}");
        }
    }

    /// The page's report, as of the file `stamp` at the model's place.
    fn report(s: &crate::settings::Settings, what: &str, error: Option<&str>, stamp: Option<super::Stamp>) -> super::Wearing {
        super::Wearing {
            file: Some(super::ModelFile { path: super::model_path(s), stamp }),
            ..super::wearing_from(what, error.map(str::to_owned)).unwrap()
        }
    }

    /// What the page said it loaded reaches the control window: an export
    /// that didn't parse is the placeholder, with the page's reason.
    #[test]
    fn the_page_says_what_he_wears() {
        let s = crate::settings::Settings::default();
        let before = super::model_info(&s, |_| file(5, 1), None);
        assert_eq!((before.wearing, before.changed), (None, false), "before any load");
        let failed = report(&s, "placeholder", Some("That file has no VRM in it."), file(5, 1));
        let info = super::model_info(&s, |_| file(5, 1), Some(&failed));
        assert_eq!(info.wearing, Some("placeholder"));
        assert_eq!(info.load_error.as_deref(), Some("That file has no VRM in it."));
        assert!(!info.changed);
        let ok = report(&s, "export", None, file(5, 1));
        let info = super::model_info(&s, |_| file(5, 1), Some(&ok));
        assert_eq!((info.wearing, info.load_error, info.changed), (Some("export"), None, false));
        // Nothing there at the last load, and nothing now: the placeholder,
        // and no news.
        let nothing = report(&s, "placeholder", None, None);
        assert!(!super::model_info(&s, |_| None, Some(&nothing)).changed);
    }

    /// A report is of the file the page was handed: once the file there
    /// isn't that one, the window hears that it changed, and an old error
    /// is never pinned on a new file. What he wears is still as reported.
    #[test]
    fn a_report_is_of_the_file_it_was_handed() {
        let mut s = crate::settings::Settings::default();
        let failed = report(&s, "placeholder", Some("That file has no VRM in it."), file(5, 1));
        let ok = report(&s, "export", None, file(5, 1));
        let nothing = report(&s, "placeholder", None, None);
        let cases = [
            (&nothing, file(5, 1), "an export dropped in where there was none"),
            (&failed, file(6, 1), "the broken export replaced by a larger one"),
            (&failed, file(5, 2), "the broken export written again"),
            (&ok, None, "the export he wears taken away"),
        ];
        for (was, now, case) in cases {
            let info = super::model_info(&s, |_| now, Some(was));
            assert!(info.changed, "{case}");
            assert_eq!(info.load_error, None, "{case}");
            assert_eq!(info.wearing, Some(was.what), "{case}");
        }
        // A report the page never asked a file for matches none.
        let unasked = super::wearing_from("placeholder", None).unwrap();
        assert!(super::model_info(&s, |_| None, Some(&unasked)).changed);
        // avatar.vrmPath changed: the same stamp somewhere else is another file.
        s.avatar.vrm_path = Some(r"D:\models\other.vrm".into());
        let info = super::model_info(&s, |_| file(5, 1), Some(&failed));
        assert!(info.changed && info.load_error.is_none());
    }

    /// A file is stamped with its size and when it was written; a folder
    /// or nothing at all is no model.
    #[test]
    fn a_model_file_is_stamped() {
        let dir = std::env::temp_dir().join(format!("fritz-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("fritz.vrm");
        std::fs::write(&f, b"vrm").unwrap();
        let st = super::stamp(&f).unwrap();
        assert_eq!(st.len, 3);
        assert_eq!(st.modified, std::fs::metadata(&f).unwrap().modified().ok());
        assert!(st.modified.is_some());
        assert_eq!(super::stamp(&dir), None, "a folder");
        assert_eq!(super::stamp(&dir.join("none.vrm")), None, "nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `avatar_vrm` keeps which file it handed over (here, none at the
    /// model's place), and the page's report takes it, once.
    #[test]
    fn the_report_takes_what_was_handed() {
        let none = tauri::async_runtime::block_on(super::avatar_vrm()).err();
        assert_eq!(none.as_deref(), Some("none"));
        let s = crate::settings::get();
        let w = super::take_report("placeholder", None).unwrap();
        assert_eq!(w.file, Some(super::ModelFile { path: super::model_path(&s), stamp: None }));
        assert!(!super::model_info(&s, |_| None, Some(&w)).changed);
        assert!(super::model_info(&s, |_| file(5, 1), Some(&w)).changed);
        assert_eq!(super::take_report("placeholder", None).unwrap().file, None, "taken");
    }

    #[test]
    fn the_pages_report_is_checked() {
        assert!(super::wearing_from("a hat", None).is_err());
        let w = super::wearing_from("placeholder", Some(format!("bad\r\nfile\u{0}{}", "x".repeat(500)))).unwrap();
        let e = w.error.unwrap();
        assert!(!e.contains(['\r', '\n', '\u{0}']), "{e}");
        assert_eq!(e.chars().count(), super::LOAD_ERROR_CHARS);
        assert_eq!(super::wearing_from("export", Some("  ".into())).unwrap().error, None);
    }

    #[test]
    fn the_control_window_opens_only_fritzs_folders() {
        let data = crate::settings::app_dir();
        assert_eq!(super::folder("logs"), Some(data.clone()));
        assert_eq!(super::folder("avatar"), Some(data.join("avatar")));
        assert_eq!(super::folder(r"C:\Windows"), None);
        assert_eq!(super::folder(""), None);
    }

    #[test]
    fn the_foreground_log_runs_only_when_asked_for() {
        let mut s = crate::settings::Settings::default();
        assert_eq!(super::foreground_log_path(&s), None);
        s.diagnostics.foreground_log = true;
        assert_eq!(
            super::foreground_log_path(&s),
            Some(crate::settings::app_dir().join("diagnostics").join("foreground.log"))
        );
    }

    #[test]
    fn wire_keys_are_lower_camel_case() {
        assert_eq!(wire_key("instance"), "instance");
        assert_eq!(wire_key("arrange_gen"), "arrangeGen");
        assert_eq!(wire_key("client_id_2"), "clientId2");
    }

    /// Every command the page calls is registered, with the argument keys
    /// the page sends (Tauri matches them to these parameters, in
    /// lowerCamelCase).
    /// The page's half is src/avatar/status.test.ts and run.test.ts, against the same
    /// fixture, so renaming one side alone fails a test.
    #[test]
    fn the_commands_are_the_wire_contract() {
        let src: String = include_str!("lib.rs")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let handlers = {
            let at = src.find("generate_handler![").expect("a handler list") + 18;
            let end = at + src[at..].find(']').unwrap();
            src[at..end]
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        };
        let commands = fixture["commands"].as_object().expect("a commands block");
        assert_eq!(commands.len(), handlers.len(), "{handlers:?}");
        for cmd in commands.values() {
            let name = cmd["name"].as_str().unwrap();
            assert!(handlers.contains(&name), "{name} isn't registered");
            let sig = format!("async fn {name}(");
            let at = src
                .find(&sig)
                .unwrap_or_else(|| panic!("no command {name}"))
                + sig.len();
            let params: Vec<String> = src[at..at + src[at..].find(')').unwrap()]
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(|p| p.split(':').next().unwrap().trim())
                .filter(|p| *p != "app")
                .map(wire_key)
                .collect();
            let want: Vec<&str> = cmd["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_str().unwrap())
                .collect();
            assert_eq!(params, want, "{name}");
        }
    }
}
