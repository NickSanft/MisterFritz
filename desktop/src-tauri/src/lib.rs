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

use tauri::{AppHandle, Emitter, Manager, RunEvent};
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
/// none, and the page falls back to the placeholder it ships with.
#[tauri::command]
async fn avatar_vrm() -> Result<tauri::ipc::Response, String> {
    let path = model_path(&settings::get());
    let size = std::fs::metadata(&path)
        .map_err(|_| "none".to_owned())?
        .len();
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
            cursor_now
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
            logs::startup("started");
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == OVERLAY_LABEL {
                overlay_windows::on_window_event(window.app_handle(), event);
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
        assert_eq!(events.len(), 4, "an event with no Rust half: {events:?}");
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
