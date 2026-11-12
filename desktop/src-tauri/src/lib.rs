//! Mister Fritz's body on the Windows desktop — Spike 0 (plan 14): one
//! persistent, transparent, click-through, never-focused avatar window with a
//! WebGL VRM in it, instrumented for a working session. Windows only.
//!
//! What the spike records lives under `%APPDATA%\com.divora.fritz\`:
//! `spike\foreground.log` (the first row of the table), `logs\fritz.log`, and
//! `avatar-state.json`. A VRoid export at `avatar\fritz.vrm` replaces the
//! committed placeholder.

mod avatar_state;
mod cover;
mod foreground_log;
mod non_rude;
mod overlay_machine;
mod overlay_windows;
mod tray;
mod windows_ext;

use std::fs::File;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager, RunEvent};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

use overlay_machine::{
    HandshakeStage, ModeRequest, OverlayRect, OverlayStatus, Refusal, BAD_STAGE, HOTKEY_ARRANGE,
    OVERLAY_LABEL,
};

/// The largest VRM the page is handed. VRoid exports are 5–50 MB.
const MAX_VRM_BYTES: u64 = 256 * 1024 * 1024;

/// Fritz's first spot: the bottom-right of the primary work area, this many
/// logical pixels in, and this big.
const FIRST_SIZE: (f64, f64) = (300.0, 460.0);
const FIRST_MARGIN: f64 = 24.0;

const PARK_HOTKEY: &str = "avatar:park";

fn shortcuts() -> [(Shortcut, &'static str); 2] {
    let ctrl_alt = Some(Modifiers::CONTROL | Modifiers::ALT);
    [
        (Shortcut::new(ctrl_alt, Code::KeyA), HOTKEY_ARRANGE),
        (Shortcut::new(ctrl_alt, Code::KeyH), PARK_HOTKEY),
    ]
}

fn data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("com.divora.fritz"))
}

fn state_file(app: &AppHandle) -> PathBuf {
    data_dir(app).join("avatar-state.json")
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
        .or_else(|| avatar_state::load(&state_file(app)).rect.map(Into::into))
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
    let state = avatar_state::current(app);
    if state.rect.is_some() {
        let _ = avatar_state::save(&state_file(app), &state);
    }
    overlay_windows::before_exit(app);
    app.exit(0);
}

fn on_shortcut(app: &AppHandle, shortcut: &Shortcut, event: ShortcutEvent) {
    let pressed = event.state == ShortcutState::Pressed;
    let Some((_, id)) = shortcuts().into_iter().find(|(s, _)| s == shortcut) else {
        return;
    };
    if id == PARK_HOTKEY {
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
}

fn init_logging(app: &AppHandle) {
    let dir = data_dir(app).join("logs");
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
async fn avatar_vrm(app: AppHandle) -> Result<tauri::ipc::Response, String> {
    let path = data_dir(&app).join("avatar").join("fritz.vrm");
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

pub fn run() {
    windows_ext::set_dpi_awareness();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().with_handler(on_shortcut).build())
        .invoke_handler(tauri::generate_handler![
            avatar_handshake,
            avatar_beat,
            set_avatar_mode,
            avatar_status,
            avatar_vrm
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            init_logging(&handle);
            tracing::info!(version = env!("CARGO_PKG_VERSION"), "Mister Fritz starting (Spike 0)");
            overlay_windows::init(app);
            let saved = avatar_state::load(&state_file(&handle));
            if saved.parked {
                overlay_windows::set_parked(&handle, true);
            }
            tray::build(&handle)?;
            for (shortcut, id) in shortcuts() {
                if let Err(e) = handle.global_shortcut().register(shortcut) {
                    tracing::warn!(id, error = %e, "couldn't register a hotkey; the tray still works");
                }
            }
            foreground_log::start(handle.clone(), data_dir(&handle).join("spike").join("foreground.log"));
            cover::start(handle.clone());
            avatar_state::start_saver(handle.clone(), state_file(&handle), saved);
            open_avatar(&handle);
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == OVERLAY_LABEL {
                overlay_windows::on_window_event(window.app_handle(), event);
            }
        })
        .build(tauri::generate_context!())
        .expect("Mister Fritz couldn't start");
    app.run(|_app, event| {
        // The avatar window closing (Alt+F4 in Arrange) leaves the tray up;
        // only Quit ends the app.
        if let RunEvent::ExitRequested {
            code: None, api, ..
        } = event
        {
            api.prevent_exit();
        }
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

    #[test]
    fn wire_keys_are_lower_camel_case() {
        assert_eq!(wire_key("instance"), "instance");
        assert_eq!(wire_key("arrange_gen"), "arrangeGen");
        assert_eq!(wire_key("client_id_2"), "clientId2");
    }

    /// Every command the page calls is registered, with the argument keys
    /// the page sends (Tauri matches them to these parameters, in
    /// lowerCamelCase).
    /// The page's half is src/avatar/status.test.ts, against the same
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
