//! Mister Fritz's body on the Windows desktop (plan 14): one persistent,
//! transparent, click-through, never-focused avatar window with a WebGL VRM
//! in it, a tray, global hotkeys, the control window (Settings) and the
//! prompt (`prompt.rs`), both declared and hidden at rest. Windows only.
//!
//! What it keeps lives in `%APPDATA%\com.divora.fritz\`: `settings.json`,
//! `companion.token` (the brain's token, `companion.rs`), `startup.log` and
//! `crash.log` (`logs.rs`), `logs\fritz.log` (this run's), and, if the owner
//! turns it on, `diagnostics\foreground.log`. A VRoid export at
//! `avatar\fritz.vrm`, or wherever `avatar.vrmPath` says, replaces the
//! committed placeholder.

mod avatar_state;
mod companion;
mod cover;
mod cursor;
mod foreground_log;
mod hit_shape;
mod hotkeys;
mod keyboard;
mod logs;
mod non_rude;
mod overlay_machine;
mod overlay_windows;
mod power;
mod prompt;
mod settings;
mod tray;
mod windows_ext;

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, MutexGuard};
use std::time::Duration;

use tauri::webview::PageLoadEvent;
use tauri::{AppHandle, Emitter, Manager, RunEvent, WindowEvent};
use tauri_plugin_global_shortcut::{Shortcut, ShortcutEvent, ShortcutState};

use overlay_machine::{
    HandshakeStage, ModeRequest, OverlayRect, OverlayStatus, Refusal, BAD_STAGE, OVERLAY_LABEL,
};
use settings::Settings;

/// The largest VRM the page is handed. VRoid exports are 5–50 MB.
const MAX_VRM_BYTES: u64 = 256 * 1024 * 1024;

/// Fritz's first spot: the bottom-right of the primary work area, this many
/// logical pixels in, and this big. The top 28% is room for his speech
/// bubble (the page's `HEADROOM`), so he stands as tall as PR 2's 300×460
/// framed him; a window opened with no place yet is built this size too.
pub(crate) const FIRST_SIZE: (f64, f64) = (340.0, 640.0);
const FIRST_MARGIN: f64 = 24.0;

/// Sent to every window when a setting changes, with all of them.
pub const SETTINGS_EVENT: &str = "settings:changed";
/// Sent to every window when the avatar's page reports the brain's link.
pub const COMPANION_STATUS_EVENT: &str = "companion:status";
/// Sent to the avatar's page to ask for its config again and connect anew:
/// the token was saved, or the owner pressed Reconnect.
pub const COMPANION_CONFIG_EVENT: &str = "companion:config";
/// Sent to the avatar's page with `{text, id}`: a turn to ask the brain,
/// from Settings › Brain's Ask or the prompt. The page takes it
/// (`companion_take`), asks it only if the shell still waits for it, and
/// answers `companion_said` with whether it went on.
pub const COMPANION_SAY_EVENT: &str = "companion:say";
/// A debug build only: once the link is first online, ask the brain this,
/// through the same path as Ask (a live check's turn with no click).
pub const PRETEND_SAY_ENV: &str = "FRITZ_PRETEND_SAY";
/// The pretend say is asked once a run.
static PRETENDED: AtomicBool = AtomicBool::new(false);

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
pub(crate) fn first_rect() -> Option<OverlayRect> {
    let (work, dpi) = windows_ext::primary_work_area()?;
    first_rect_in(work, dpi)
}

/// The first placement in work area `work` at `dpi`: `FIRST_SIZE` at its
/// bottom right, `FIRST_MARGIN` in. On a work area too short (or narrow)
/// for it, the window is made to fit rather than placed off its top: the
/// top of his window is the bubble's room.
fn first_rect_in(work: windows_ext::Bounds, dpi: u32) -> Option<OverlayRect> {
    let s = f64::from(dpi) / 96.0;
    #[allow(clippy::cast_possible_truncation)]
    let (w, h, m) = (
        (FIRST_SIZE.0 * s) as i32,
        (FIRST_SIZE.1 * s) as i32,
        (FIRST_MARGIN * s) as i32,
    );
    let (w, h) = (w.min(work.width() - 2 * m), h.min(work.height() - 2 * m));
    let (width, height) = (u32::try_from(w).ok()?, u32::try_from(h).ok()?);
    if width == 0 || height == 0 {
        return None;
    }
    Some(OverlayRect {
        x: work.right - w - m,
        y: work.bottom - h - m,
        width,
        height,
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
        // Inline, inside WM_HOTKEY: the press's right to the foreground is
        // fresh only here, and the prompt needs it to take the keyboard.
        if id == hotkeys::PROMPT {
            if prompt::hotkey_request(id, pressed) {
                prompt::toggle(app, prompt::Opener::Hotkey);
            }
        } else if id == hotkeys::PARK {
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

/// Set one setting by its camelCase path, from a window or the tray: every
/// window hears the result, in the order the changes were made, and the
/// tray's Appearance ticks follow a change to the theme (moved in place,
/// never a rebuilt menu: see tray.rs).
pub(crate) fn change_setting(
    app: &AppHandle,
    path: &str,
    value: serde_json::Value,
) -> Result<Settings, String> {
    let after = settings::set(path, value, |after| {
        let _ = app.emit(SETTINGS_EVENT, after);
    })?;
    tracing::info!(path, "setting changed");
    if tray::follows(path) {
        let handle = app.clone();
        let _ = app.run_on_main_thread(move || tray::retick(&handle));
    }
    Ok(after)
}

#[tauri::command]
async fn set_setting(
    app: AppHandle,
    path: String,
    value: serde_json::Value,
) -> Result<Settings, String> {
    window_may_set(&path)?;
    change_setting(&app, &path, value)
}

/// Why a window may not set the hotkeys with `set_setting`.
const HOTKEYS_ELSEWHERE: &str = "Hotkeys are set in Settings › Hotkeys, which binds them as it saves them.";

/// What a window may not set by its path: the hotkeys, which `set_hotkey`
/// binds as it saves them (one saved by its path alone would wait for the
/// next start, unchecked). The shell's own `change_setting` is unaffected.
fn window_may_set(path: &str) -> Result<(), String> {
    if path == "hotkeys" || path.starts_with("hotkeys.") {
        Err(HOTKEYS_ELSEWHERE.to_owned())
    } else {
        Ok(())
    }
}

/// Only the control window (Settings, and its first run) sets hotkeys.
fn from_the_control(label: &str) -> Result<(), String> {
    if label == CONTROL_LABEL {
        Ok(())
    } else {
        Err(format!("only Fritz's Settings may do that, not {label:?}"))
    }
}

/// Set one of Fritz's keys (`prompt`, `arrange` or `park`): checked, taken
/// from Windows, saved, and only then is the old key let go. Refused with
/// the reason, the old key still working. On a blocking worker: the plugin
/// waits on the main thread.
#[tauri::command]
async fn set_hotkey(
    app: AppHandle,
    webview: tauri::Webview,
    action: String,
    accelerator: String,
) -> Result<Settings, String> {
    from_the_control(webview.label())?;
    tauri::async_runtime::spawn_blocking(move || hotkeys::rebind(&app, &action, &accelerator))
        .await
        .map_err(|e| e.to_string())?
}

/// Each action's key as Windows holds it now, or null: a setting that fell
/// back at start, or that Windows refused, shows here as it really is.
#[tauri::command]
async fn hotkeys_bound() -> std::collections::BTreeMap<&'static str, Option<String>> {
    hotkeys::bound_now()
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
/// tray. Set to a section's name (`appearance`), it opens there; set to
/// `wizard`, at the first run.
const SHOW_SETTINGS_ENV: &str = "FRITZ_SHOW_SETTINGS";
/// The control window's sections, by the ids its page gives them, in order.
pub const CONTROL_SECTIONS: [&str; 7] = ["brain", "fritz", "hotkeys", "appearance", "model", "diagnostics", "about"];
/// What `control_opening` answers for the first run, which takes the
/// control window's place of its sections.
pub const CONTROL_WIZARD: &str = "wizard";

/// Where the control window opens: the section `FRITZ_SHOW_SETTINGS` names,
/// if it names one; else the first run, when it names that or there is no
/// token yet; else nowhere in particular.
fn opening(env: Option<&str>, token_present: bool) -> Option<&'static str> {
    let want = env.map(str::trim);
    let named = want.and_then(|want| CONTROL_SECTIONS.iter().copied().find(|s| s.eq_ignore_ascii_case(want)));
    let wizard = want.is_some_and(|w| w.eq_ignore_ascii_case(CONTROL_WIZARD)) || !token_present;
    named.or_else(|| wizard.then_some(CONTROL_WIZARD))
}

/// What Settings must not open over at start, or `None`, as the prompt's
/// gate has it: the lock screen, another app's fullscreen window in front,
/// the shell's busy states (a Direct3D fullscreen app, presentation mode),
/// or a fullscreen window on the monitor where Settings will stand (its
/// hidden window, `control`, already stands there). Asked of Windows
/// itself: the cover watch may not have looked yet.
fn in_front_now(control: isize) -> Option<String> {
    cover::covered_by(windows_ext::session_locked(), || {
        windows_ext::fullscreen_foreground()
            .or_else(|| windows_ext::busy_now().map(str::to_owned))
            .or_else(|| windows_ext::window_bounds(control).and_then(windows_ext::covering_app))
    })
}

/// Settings has been shown since start, by any way (the tray, the start).
static CONTROL_SHOWN: AtomicBool = AtomicBool::new(false);

/// How long Settings waits at start before looking again at what is in front.
const SETTINGS_WAIT: Duration = Duration::from_secs(2);

/// Wait while `in_front` names something, telling `waiting` once what it
/// waits for, and `pause` between looks. Pure, for tests.
fn wait_until_clear(
    mut in_front: impl FnMut() -> Option<String>,
    mut waiting: impl FnMut(&str),
    mut pause: impl FnMut(),
) {
    let mut said = false;
    while let Some(what) = in_front() {
        if !said {
            waiting(&what);
            said = true;
        }
        pause();
    }
}

/// Show Settings at start, as the first run or when asked to: at once, or,
/// while something it mustn't open over is in front, once that has gone;
/// and not at all if the owner opened it meanwhile (from the tray), which
/// would only show it again, unasked. Never over a game, a presentation or
/// the lock screen; the looks are made off the main thread, and the show
/// posted back to it, where the tray's own show runs too.
fn show_control_when_clear(app: AppHandle) {
    let control = app.get_webview_window(CONTROL_LABEL).and_then(|w| w.hwnd().ok()).map_or(0, |h| h.0 as isize);
    let spawned = std::thread::Builder::new().name("settings-at-start".into()).spawn(move || {
        wait_until_clear(
            || in_front_now(control),
            |what| {
                logs::startup(&format!("Settings waits: {what} is in front"));
                tracing::info!(what, "Settings waits until nothing it mustn't open over is in front");
            },
            || std::thread::sleep(SETTINGS_WAIT),
        );
        let h = app.clone();
        let _ = app.run_on_main_thread(move || {
            if CONTROL_SHOWN.load(Ordering::SeqCst) {
                tracing::info!("Settings was opened meanwhile: not shown again");
            } else {
                show_control(&h);
            }
        });
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the wait for Settings");
    }
}

fn token_present() -> bool {
    companion::read_token(&data_dir()).is_some()
}

/// Where the control window's page scrolls to once it has loaded.
#[tauri::command]
async fn control_opening() -> Option<&'static str> {
    opening(std::env::var(SHOW_SETTINGS_ENV).ok().as_deref(), token_present())
}

/// Show the control window and give it the keyboard (it is an ordinary
/// window; only the avatar's must never take focus). By SetForegroundWindow,
/// which Windows allows from the tray's click and refuses otherwise (the
/// window then waits behind, its taskbar button flashing); never tao's
/// `set_focus`, which taps Alt into the owner's app to get its way.
pub(crate) fn show_control(app: &AppHandle) {
    CONTROL_SHOWN.store(true, Ordering::SeqCst);
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
/// does, so the tray can bring it back; the avatar's and the prompt's have
/// their own shells (a bare hide would leave the keyboard with a hidden
/// prompt).
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

/// `s` without its control characters (line breaks, escapes), cut to `max`
/// characters: text a page reports, made fit for a log line or a window.
pub(crate) fn printable(s: &str, max: usize) -> String {
    s.chars().filter(|c| !c.is_control()).take(max).collect()
}

/// The page's report, checked: `"export"` or `"placeholder"`, and an error
/// cut to printable text of a sensible length.
pub fn wearing_from(what: &str, error: Option<String>) -> Result<Wearing, String> {
    let what = match what {
        "export" => "export",
        "placeholder" => "placeholder",
        other => return Err(format!("{other:?} isn't a model the page wears")),
    };
    let error = error
        .map(|e| printable(&e, LOAD_ERROR_CHARS))
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

/// His window as far as a shape goes: the one that is open and not on its
/// way out. A shape for any other is nobody's.
fn live_window(open: bool, phase: overlay_machine::PhaseName, instance: u32) -> Option<u32> {
    (open && phase != overlay_machine::PhaseName::Closing).then_some(instance)
}

/// Which load of his page this is, by the shell's count: his page asks
/// once, and stamps every shape it sends with the answer.
#[tauri::command]
async fn avatar_page(webview: tauri::Webview) -> Result<u32, String> {
    from_the_avatar(webview.label())?;
    Ok(hit_shape::page())
}

/// His page says where his body is on his window: up to six boxes, in CSS
/// pixels of a page `view` big. Whether it was kept: a send that isn't the
/// live window's, this load's and the latest is dropped, which is no
/// error. Refused with why when it can't be a shape (hit_shape.rs).
#[tauri::command]
async fn set_hit_rects(
    app: AppHandle,
    webview: tauri::Webview,
    instance: u32,
    page: u32,
    seq: u32,
    view: hit_shape::View,
    rects: Vec<hit_shape::PageRect>,
) -> Result<bool, String> {
    from_the_avatar(webview.label())?;
    let packed = hit_shape::pack(&view, &rects)?;
    let s = overlay_windows::status(&app);
    let live = live_window(s.open, s.phase, s.instance);
    Ok(match hit_shape::keep(live, hit_shape::Stamp { instance, page, seq }, &packed) {
        // His shape, never the owner's input: nothing here says where a click was.
        hit_shape::Kept::Changed { boxes, percent, news: true } => {
            tracing::info!("his hit shape: {boxes} boxes, {percent}% of his window (window {instance})");
            true
        }
        hit_shape::Kept::Changed { boxes, percent, news: false } => {
            tracing::debug!("his hit shape: {boxes} boxes, {percent}% of his window (window {instance})");
            true
        }
        hit_shape::Kept::Same => true,
        hit_shape::Kept::Older => false,
    })
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

/// What the avatar's page connects with: the brain's URL, the token (null
/// when there is none) and this desk's id and name.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompanionConfig {
    pub url: String,
    pub token: Option<String>,
    pub client_id: String,
    pub client_name: String,
}

fn companion_config_for(s: &Settings, token: Option<String>) -> CompanionConfig {
    CompanionConfig {
        url: s.brain.url.clone(),
        token,
        client_id: s.brain.client_id.clone(),
        client_name: s.brain.client_name.clone(),
    }
}

/// The brain's session is the avatar page's alone (the control window
/// hears of it through the shell): only that page is given the token, and
/// only it reports the link.
fn from_the_avatar(label: &str) -> Result<(), String> {
    if label == OVERLAY_LABEL {
        Ok(())
    } else {
        Err(format!("only Fritz's own window may do that, not {label:?}"))
    }
}

/// Whether there is a token, for the control window (never the token).
#[tauri::command]
async fn get_token_present() -> bool {
    token_present()
}

/// Save the token the owner pasted (checked: `companion::token_from`), and
/// have the avatar's page connect with it. The token is never logged.
#[tauri::command]
async fn set_token(app: AppHandle, token: String) -> Result<(), String> {
    let t = companion::token_from(&token)?;
    companion::write_token(&data_dir(), &t).map_err(|e| format!("Couldn't save the token: {e}"))?;
    tracing::info!("the companion token was saved");
    let _ = app.emit_to(OVERLAY_LABEL, COMPANION_CONFIG_EVENT, ());
    Ok(())
}

/// The avatar's page asks what to connect with; any other window is refused.
#[tauri::command]
async fn companion_config(webview: tauri::Webview) -> Result<CompanionConfig, String> {
    from_the_avatar(webview.label())?;
    Ok(companion_config_for(&settings::get(), companion::read_token(&data_dir())))
}

/// The avatar's page says where the link stands (on every change): kept for
/// `companion_status`, logged when the link or its reason changes (a new
/// wait between tries is no news), and every window is told.
#[tauri::command]
async fn companion_report(app: AppHandle, webview: tauri::Webview, report: serde_json::Value) -> Result<(), String> {
    from_the_avatar(webview.label())?;
    let s = companion::status_from(report)?;
    let was = companion::keep(s.clone());
    if was.as_ref().map(companion::CompanionStatus::headline) != Some(s.headline()) {
        if s.link == companion::Link::Refused {
            tracing::warn!("{}", s.describe());
        } else {
            tracing::info!("{}", s.describe());
        }
    }
    let _ = app.emit(COMPANION_STATUS_EVENT, &s);
    if s.link == companion::Link::Online && !PRETENDED.load(Ordering::SeqCst) {
        let env = std::env::var(PRETEND_SAY_ENV).ok();
        if let Some(text) = companion::pretend_say_text(cfg!(debug_assertions), env) {
            PRETENDED.store(true, Ordering::SeqCst);
            // Not awaited here: the page answers the turn with a command of
            // its own, while this report's is still open.
            let h = app.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = ask_the_brain(&h, &text, PRETEND_SAY_ENV).await {
                    tracing::warn!("{PRETEND_SAY_ENV}: {e}");
                }
            });
        }
    }
    Ok(())
}

/// His window is gone, and the brain's session with it (his page held it):
/// its last report is forgotten and every window told, so nothing is asked
/// of a page that isn't there, nor of the next one before it is online.
fn companion_page_gone(app: &AppHandle) {
    companion::forget();
    let _ = app.emit(COMPANION_STATUS_EVENT, companion::current(token_present()));
}

/// How long the avatar's page has to take a turn it was handed.
const SAY_WAIT: Duration = Duration::from_secs(5);
/// How much longer a turn the page has taken is waited for: it is asking
/// the brain, which is at once, unless the page dies meanwhile.
const SAY_GRACE: Duration = Duration::from_secs(5);
/// Why a turn wasn't taken when his page never took it (so it wasn't asked).
const SAY_UNANSWERED: &str = "His window didn't answer just now: try again.";
/// Why a turn wasn't taken when his page took it but never said whether it
/// went on: it may have been asked, so this never says to try again.
const SAY_UNSURE: &str = "His window took it but never said whether it went on: see his bubble before asking again.";

/// What the shell heard of a turn it handed to the avatar's page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Heard {
    /// The page's answer: whether the link took it.
    Answered(bool),
    /// The page never took it, and now never will: it wasn't asked.
    NotTaken,
    /// The page took it and never said: it may have been asked.
    Silent,
}

/// Where a turn handed to the avatar's page stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Handed over, not yet taken: the shell may still give it up.
    Handed,
    /// Taken by the page, which is asking it: never given up now.
    Taken,
}

/// The turns handed to the avatar's page that it hasn't yet answered, by
/// their number, each with its stage and where its answer goes.
static SAYS: Mutex<Vec<(u64, Stage, mpsc::Sender<bool>)>> = Mutex::new(Vec::new());
static SAY_NUMBER: AtomicU64 = AtomicU64::new(0);

fn says() -> MutexGuard<'static, Vec<(u64, Stage, mpsc::Sender<bool>)>> {
    SAYS.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A number for the next turn, and where the page's answer to it arrives.
fn await_say() -> (u64, mpsc::Receiver<bool>) {
    let id = SAY_NUMBER.fetch_add(1, Ordering::SeqCst) + 1;
    let (tx, rx) = mpsc::channel();
    says().push((id, Stage::Handed, tx));
    (id, rx)
}

/// The page takes turn `id` before asking it: true while the shell still
/// waits for it (it then waits for its answer), false once it has given it
/// up, or taken it already, and the page must not ask it. The shell alone
/// decides, so no clock of the page's is trusted.
fn take_say(id: u64) -> bool {
    let mut all = says();
    match all.iter_mut().find(|(n, _, _)| *n == id) {
        Some(entry) if entry.1 == Stage::Handed => {
            entry.1 = Stage::Taken;
            true
        }
        _ => false,
    }
}

/// Turn `id` off the list, at whatever stage; where its answer went, if it
/// was there.
fn stop_awaiting(id: u64) -> Option<mpsc::Sender<bool>> {
    let mut all = says();
    let at = all.iter().position(|(n, _, _)| *n == id)?;
    Some(all.swap_remove(at).2)
}

/// The wait is over: a turn the page hasn't taken is taken off the list, so
/// it is never asked (true). One it has taken, or already answered, isn't
/// given up: its answer is coming (false).
fn give_up(id: u64) -> bool {
    let mut all = says();
    match all.iter().position(|(n, stage, _)| *n == id && *stage == Stage::Handed) {
        Some(at) => {
            all.swap_remove(at);
            true
        }
        None => false,
    }
}

/// What the shell heard means for the turn.
fn say_outcome(heard: Heard) -> Result<(), &'static str> {
    match heard {
        Heard::Answered(true) => Ok(()),
        Heard::Answered(false) => Err(companion::NOT_CONNECTED),
        Heard::NotTaken => Err(SAY_UNANSWERED),
        Heard::Silent => Err(SAY_UNSURE),
    }
}

/// What `companion:say` carries: the turn's text and number.
fn say_payload(text: &str, id: u64) -> serde_json::Value {
    serde_json::json!({ "text": text, "id": id })
}

/// Wait for the page's answer to turn `id`. Once `wait` is over, a turn the
/// page never took is given up, so it is never asked; one it took is being
/// asked, and its answer is waited for `grace` longer.
fn wait_for_answer(id: u64, answer: &mpsc::Receiver<bool>, wait: Duration, grace: Duration) -> Heard {
    if let Ok(sent) = answer.recv_timeout(wait) {
        return Heard::Answered(sent);
    }
    if give_up(id) {
        return Heard::NotTaken;
    }
    answer.recv_timeout(grace).map_or(Heard::Silent, Heard::Answered)
}

/// Ask the brain `text`, through the avatar's page (it holds the session).
/// Refused, with why, when there is nothing to ask, his window is closed or
/// the link isn't online; and taken only once the page says it sent it on,
/// since the link can drop between its last report and the turn. Logs the
/// length, never the text.
async fn ask_the_brain(app: &AppHandle, text: &str, from: &str) -> Result<(), String> {
    let avatar = overlay_windows::status(app);
    let open = avatar.open && avatar.phase != overlay_machine::PhaseName::Closing;
    if let Some(why) = companion::say_refusal(text, &companion::current(token_present()), open) {
        return Err(why.to_owned());
    }
    let (id, answer) = await_say();
    if let Err(e) = app.emit_to(OVERLAY_LABEL, COMPANION_SAY_EVENT, say_payload(text, id)) {
        stop_awaiting(id);
        return Err(format!("Couldn't reach his window: {e}"));
    }
    // A wait that failed (it can't, short of a panic) may have let the page
    // take the turn: unsure, never "try again".
    let heard = tauri::async_runtime::spawn_blocking(move || wait_for_answer(id, &answer, SAY_WAIT, SAY_GRACE))
        .await
        .unwrap_or(Heard::Silent);
    stop_awaiting(id);
    let chars = text.trim().chars().count();
    match heard {
        Heard::Answered(true) => tracing::info!("a turn asked from {from} ({chars} characters)"),
        Heard::Silent => tracing::warn!("a turn from {from} ({chars} characters) was taken by his page, which never said whether it went on"),
        Heard::Answered(false) | Heard::NotTaken => {
            tracing::warn!("a turn from {from} ({chars} characters) wasn't asked: {}", say_outcome(heard).unwrap_err());
        }
    }
    say_outcome(heard).map_err(str::to_owned)
}

/// The avatar's page takes turn `id` before asking it: whether the shell
/// still waits for it (if not, the page doesn't ask).
#[tauri::command]
async fn companion_take(webview: tauri::Webview, id: u64) -> Result<bool, String> {
    from_the_avatar(webview.label())?;
    Ok(take_say(id))
}

/// The avatar's page says whether it sent turn `id` on to the brain.
#[tauri::command]
async fn companion_said(webview: tauri::Webview, id: u64, sent: bool) -> Result<(), String> {
    from_the_avatar(webview.label())?;
    match stop_awaiting(id) {
        Some(waiting) => {
            let _ = waiting.send(sent);
        }
        None => tracing::warn!(id, sent, "his page answered a turn after its wait"),
    }
    Ok(())
}

/// A turn asked from a window (Settings › Brain's Ask): the avatar's page
/// asks it, and his bubble shows the reply.
#[tauri::command]
async fn companion_say(app: AppHandle, text: String) -> Result<(), String> {
    ask_the_brain(&app, &text, "Settings").await
}

/// The prompt's commands are its own.
fn from_the_prompt(label: &str) -> Result<(), String> {
    if label == prompt::LABEL {
        Ok(())
    } else {
        Err(format!("only Fritz's prompt may do that, not {label:?}"))
    }
}

/// The prompt's Enter: ask the brain, and close the prompt once the turn is
/// taken. A refusal leaves it open, with the draft and the shell's words.
#[tauri::command]
async fn prompt_say(app: AppHandle, webview: tauri::Webview, text: String) -> Result<(), String> {
    from_the_prompt(webview.label())?;
    let open = prompt::opens();
    ask_the_brain(&app, &text, "the prompt").await?;
    prompt::close_later(&app, prompt::Why::Sent, open);
    Ok(())
}

/// The prompt's Esc.
#[tauri::command]
async fn prompt_close(app: AppHandle, webview: tauri::Webview) -> Result<(), String> {
    from_the_prompt(webview.label())?;
    prompt::close_later(&app, prompt::Why::Esc, prompt::opens());
    Ok(())
}

/// The prompt's page has mounted and listens: its hotkey may open it now.
#[tauri::command]
async fn prompt_ready(app: AppHandle, webview: tauri::Webview) -> Result<(), String> {
    from_the_prompt(webview.label())?;
    let pretend = prompt::pretend_wanted(cfg!(debug_assertions), std::env::var_os(prompt::PRETEND_ENV));
    prompt::page_ready(&app, pretend);
    Ok(())
}

/// The link now, for a window that opens after the last report.
#[tauri::command]
async fn companion_status() -> companion::CompanionStatus {
    companion::current(token_present())
}

/// The owner's Reconnect: the avatar's page asks for its config again and
/// connects anew, even with nothing changed.
#[tauri::command]
async fn companion_connect(app: AppHandle) {
    let _ = app.emit_to(OVERLAY_LABEL, COMPANION_CONFIG_EVENT, ());
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
            set_hotkey,
            hotkeys_bound,
            power_display,
            cursor_now,
            set_parked,
            avatar_model,
            open_folder,
            reload_model,
            avatar_loaded,
            avatar_page,
            set_hit_rects,
            control_opening,
            get_token_present,
            set_token,
            companion_config,
            companion_report,
            companion_status,
            companion_connect,
            companion_say,
            companion_take,
            companion_said,
            prompt_say,
            prompt_close,
            prompt_ready
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
            hit_shape::init();
            overlay_windows::init(app);
            prompt::adopt(app);
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
            let has_token = token_present();
            if companion::shows_control_at_start(has_token, std::env::var_os(SHOW_SETTINGS_ENV).is_some()) {
                if !has_token {
                    logs::startup("no companion token yet: the first run opens in Settings");
                    tracing::info!("no companion token yet: the first run opens in Settings");
                }
                show_control_when_clear(handle.clone());
            }
            logs::startup("started");
            Ok(())
        })
        .on_page_load(|webview, payload| {
            if webview.label() == prompt::LABEL && payload.event() == PageLoadEvent::Started {
                prompt::on_page_started(webview.app_handle());
            }
            // His page loading, for the first time or again: the load is
            // numbered, and the shape the page before it sent forgotten.
            if webview.label() == OVERLAY_LABEL && payload.event() == PageLoadEvent::Started {
                let page = hit_shape::page_started();
                tracing::debug!(page, "his page starts loading");
            }
        })
        .on_window_event(|window, event| {
            if window.label() == OVERLAY_LABEL {
                overlay_windows::on_window_event(window.app_handle(), event);
                if matches!(event, WindowEvent::Destroyed) {
                    companion_page_gone(window.app_handle());
                    hit_shape::forget();
                }
            } else if window.label() == prompt::LABEL {
                prompt::on_window_event(window.app_handle(), event);
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
        assert_eq!(events["companionStatus"], super::COMPANION_STATUS_EVENT);
        assert_eq!(events["companionConfig"], super::COMPANION_CONFIG_EVENT);
        assert_eq!(events["companionSay"], super::COMPANION_SAY_EVENT);
        assert_eq!(events.len(), 9, "an event with no Rust half: {events:?}");
        assert_eq!(fixture["debug"]["pretendSayEnv"], super::PRETEND_SAY_ENV);
        assert_eq!(fixture["debug"]["pretendPromptEnv"], crate::prompt::PRETEND_ENV);
        let say = super::say_payload("good evening", 7);
        let mut keys: Vec<&str> = say.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(serde_json::json!(keys), fixture["sayKeys"]);
        assert_eq!((say["text"].as_str(), say["id"].as_u64()), (Some("good evening"), Some(7)));
    }

    /// `FRITZ_SHOW_SETTINGS=<section>` opens the control window there, and
    /// `wizard` at the first run; any other value only shows it. With no
    /// token yet, it opens at the first run unless a section is named. The
    /// sections are the page's ids (the fixture holds both sides to them).
    #[test]
    fn the_control_window_opens_where_it_is_told() {
        let opening = |env| super::opening(env, true);
        assert_eq!(opening(None), None);
        assert_eq!(opening(Some("1")), None);
        assert_eq!(opening(Some("")), None);
        assert_eq!(opening(Some("appearance")), Some("appearance"));
        assert_eq!(opening(Some(" Appearance ")), Some("appearance"));
        assert_eq!(opening(Some("model")), Some("model"));
        assert_eq!(opening(Some("appearances")), None);
        assert_eq!(opening(Some("brain")), Some("brain"));
        // The first run: no token, so the wizard, unless a section is named;
        // or when asked for, token or not.
        assert_eq!(super::opening(None, false), Some("wizard"));
        assert_eq!(super::opening(Some("1"), false), Some("wizard"));
        assert_eq!(super::opening(Some("wizard"), true), Some("wizard"));
        assert_eq!(super::opening(Some(" Wizard "), true), Some("wizard"));
        assert_eq!(super::opening(Some("wizards"), true), None);
        assert_eq!(super::CONTROL_WIZARD, "wizard");
        assert_eq!(super::opening(Some("model"), false), Some("model"));
        assert_eq!(super::CONTROL_SECTIONS[0], "brain", "first in the window");
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let want: Vec<&str> = fixture["controlSections"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert_eq!(super::CONTROL_SECTIONS.to_vec(), want);
        for s in super::CONTROL_SECTIONS {
            assert_eq!(opening(Some(s)), Some(s));
        }
        assert_eq!(fixture["controlWizard"], super::CONTROL_WIZARD);
        assert!(!super::CONTROL_SECTIONS.contains(&super::CONTROL_WIZARD), "the wizard isn't a section");
    }

    /// Settings at start waits while something it mustn't open over is in
    /// front, says once what, and shows when that has gone; at once if
    /// nothing is.
    #[test]
    fn settings_at_start_waits_for_a_clear_screen() {
        let mut looks = vec![None, Some("Game".to_owned()), Some("the lock screen".to_owned()), Some("Game".to_owned())];
        let (mut said, mut pauses) = (Vec::new(), 0);
        super::wait_until_clear(|| looks.pop().flatten(), |w| said.push(w.to_owned()), || pauses += 1);
        assert_eq!(said, ["Game"], "said once, for the first");
        assert_eq!(pauses, 3);
        let (mut said, mut pauses) = (Vec::<String>::new(), 0);
        super::wait_until_clear(|| None, |w| said.push(w.to_owned()), || pauses += 1);
        assert!(said.is_empty() && pauses == 0, "nothing in front: at once");
    }

    /// The show at start goes through the wait, never straight to the
    /// window: the first run must not open over a game or the lock screen.
    #[test]
    fn settings_at_start_is_shown_only_through_the_wait() {
        let all = include_str!("lib.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let at = src.find(".setup(move |app| {").unwrap();
        let setup = &src[at..];
        assert!(setup.contains("show_control_when_clear(handle.clone());"));
        assert!(!setup.contains("show_control(&handle)"), "straight to the window");
        let at = src.find("fn show_control_when_clear(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(body.contains("wait_until_clear(\n            || in_front_now(control),"), "{body}");
        // Posted to the main thread, where the tray's show runs, and only if
        // Settings wasn't opened meanwhile.
        assert!(body.contains("run_on_main_thread(move || {\n            if CONTROL_SHOWN.load(Ordering::SeqCst) {"), "{body}");
        assert!(body.contains("            } else {\n                show_control(&h);"), "{body}");
        let at = src.find("pub(crate) fn show_control(").unwrap();
        let first = src[at..].lines().nth(1).unwrap().trim();
        assert_eq!(first, "CONTROL_SHOWN.store(true, Ordering::SeqCst);", "every show marks it");
        let at = src.find("fn in_front_now(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        for gate in [
            "windows_ext::session_locked()",
            "windows_ext::fullscreen_foreground()",
            "windows_ext::busy_now()",
            "windows_ext::window_bounds(control).and_then(windows_ext::covering_app)",
        ] {
            assert!(body.contains(gate), "{gate}");
        }
    }

    #[test]
    fn only_the_control_window_hides_on_close() {
        assert!(super::hides_on_close("control"));
        assert!(!super::hides_on_close("avatar"));
        assert!(!super::hides_on_close("prompt"));
    }

    /// The prompt's close comes before the bare hide, which would leave the
    /// keyboard with a hidden window.
    #[test]
    fn the_prompt_branch_comes_before_the_hide_rule() {
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        let events = &code[code.find(".on_window_event(").unwrap()..];
        let prompt = events.find("prompt::on_window_event(").expect("a prompt branch");
        let hide = events.find("hides_on_close(window.label())").expect("the hide rule");
        assert!(prompt < hide);
    }

    /// The prompt is declared hidden, and never takes the keyboard by
    /// itself: hidden is not enough (a webview made with focus takes it at
    /// every start). It can be typed in, so it is focusable; it is out of
    /// the taskbar, on top, and drawn by its page alone.
    #[test]
    fn the_prompt_window_is_declared() {
        let conf: Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let windows = conf["app"]["windows"].as_array().unwrap();
        let prompt = windows.iter().find(|w| w["label"] == crate::prompt::LABEL).expect("a prompt window");
        assert_eq!(prompt["url"], "prompt.html");
        assert_eq!(prompt["visible"], false);
        assert_eq!(prompt["focus"], false, "a hidden window must not take the keyboard");
        assert_ne!(prompt["focusable"], false, "it is typed in");
        assert_eq!(prompt["transparent"], true);
        assert_eq!(prompt["alwaysOnTop"], true);
        assert_eq!(prompt["skipTaskbar"], true);
        assert_eq!(prompt["decorations"], false);
        assert_eq!(prompt["shadow"], false);
        assert_eq!(prompt["resizable"], false);
        assert_eq!(prompt["generalAutofillEnabled"], false, "nothing it is asked is kept for autofill");
        assert_eq!((prompt["width"].as_f64(), prompt["height"].as_f64()), (Some(640.0), Some(120.0)));
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
                // The app and the calling webview are Tauri's, not the page's.
                .filter(|p| *p != "app" && *p != "webview")
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

    /// Only the avatar's page is given the token, and only it reports.
    #[test]
    fn only_the_avatars_page_holds_the_brains_session() {
        assert!(super::from_the_avatar(crate::overlay_machine::OVERLAY_LABEL).is_ok());
        for other in [super::CONTROL_LABEL, "prompt", ""] {
            assert!(super::from_the_avatar(other).is_err(), "{other:?}");
        }
    }

    #[test]
    fn only_the_control_window_sets_hotkeys() {
        assert!(super::from_the_control(super::CONTROL_LABEL).is_ok());
        for other in [crate::overlay_machine::OVERLAY_LABEL, crate::prompt::LABEL, ""] {
            assert!(super::from_the_control(other).is_err(), "{other:?}");
        }
    }

    /// A window sets a hotkey with `set_hotkey`, which binds it; by its
    /// path it would only wait, unchecked, for the next start. The check
    /// is `set_setting`'s first line.
    #[test]
    fn a_window_may_not_set_hotkeys_directly() {
        for path in ["hotkeys", "hotkeys.prompt", "hotkeys.arrange", "hotkeys.park", "hotkeys.ptt"] {
            assert_eq!(super::window_may_set(path), Err(super::HOTKEYS_ELSEWHERE.to_owned()), "{path}");
        }
        for path in ["avatar.lookatCursor", "theme.mood", "hotkeysExtra", "diagnostics.foregroundLog"] {
            assert_eq!(super::window_may_set(path), Ok(()), "{path}");
        }
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        let at = src.find("async fn set_setting(").unwrap();
        let body = &src[at + src[at..].find('{').unwrap() + 1..];
        let first = body.lines().map(str::trim).find(|l| !l.is_empty()).unwrap();
        assert_eq!(first, "window_may_set(&path)?;");
    }

    #[test]
    fn only_the_prompt_may_use_its_commands() {
        assert!(super::from_the_prompt(crate::prompt::LABEL).is_ok());
        for other in [crate::overlay_machine::OVERLAY_LABEL, super::CONTROL_LABEL, ""] {
            assert!(super::from_the_prompt(other).is_err(), "{other:?}");
        }
    }

    /// A turn is taken once the page says it sent it on: not when it says
    /// it couldn't, nor when it never says.
    #[test]
    fn a_turn_is_taken_once_the_page_sent_it() {
        use super::Heard;
        assert_eq!(super::say_outcome(Heard::Answered(true)), Ok(()));
        assert_eq!(super::say_outcome(Heard::Answered(false)), Err(crate::companion::NOT_CONNECTED));
        assert_eq!(super::say_outcome(Heard::NotTaken), Err(super::SAY_UNANSWERED));
        // Taken and never answered: it may have been asked, so never "try again".
        assert_eq!(super::say_outcome(Heard::Silent), Err(super::SAY_UNSURE));
        assert!(super::SAY_UNANSWERED.contains("try again") && !super::SAY_UNSURE.contains("try again"));
    }

    /// Each turn is waited for by its own number; an answer reaches only
    /// that turn's waiter, and an answer to a turn no longer waited for
    /// (or never sent) reaches nobody.
    #[test]
    fn a_pages_answer_reaches_its_own_turn() {
        let (a, ra) = super::await_say();
        let (b, rb) = super::await_say();
        assert_ne!(a, b);
        super::stop_awaiting(b).unwrap().send(true).unwrap();
        assert_eq!(rb.try_recv(), Ok(true));
        assert!(ra.try_recv().is_err(), "the other turn heard nothing");
        assert!(super::stop_awaiting(b).is_none(), "answered once");
        assert!(super::stop_awaiting(a).is_some());
        assert!(super::stop_awaiting(a).is_none());
        assert!(super::stop_awaiting(u64::MAX).is_none());
    }

    /// The page may take a turn once, and only while the shell still waits
    /// for it; a turn given up can't be taken, so it is never asked.
    #[test]
    fn a_turn_is_taken_once_and_only_while_waited_for() {
        let (id, _rx) = super::await_say();
        assert!(super::take_say(id));
        assert!(!super::take_say(id), "taken once");
        assert!(!super::give_up(id), "a taken turn isn't given up");
        assert!(super::stop_awaiting(id).is_some());
        let (id, _rx) = super::await_say();
        assert!(super::give_up(id));
        assert!(!super::take_say(id), "given up, so never asked");
        assert!(!super::take_say(u64::MAX));
    }

    /// An answer in time is the answer. With none, a turn the page never
    /// took is given up; one it took is waited for a while longer, so an
    /// answer from a page that took it late is still heard.
    #[test]
    fn the_wait_hears_every_answer_the_page_sent() {
        use super::Heard;
        let short = std::time::Duration::from_millis(20);
        let long = std::time::Duration::from_secs(5);
        let (id, rx) = super::await_say();
        super::stop_awaiting(id).unwrap().send(false).unwrap();
        assert_eq!(super::wait_for_answer(id, &rx, short, short), Heard::Answered(false));
        // The real order: the wait starts, then the page takes the turn and
        // answers, and is heard at once, never sitting out the grace.
        let (id, rx) = super::await_say();
        let page = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(super::take_say(id), "taken during the wait");
            super::stop_awaiting(id).unwrap().send(true).unwrap();
        });
        let started = std::time::Instant::now();
        assert_eq!(super::wait_for_answer(id, &rx, std::time::Duration::from_secs(2), long), Heard::Answered(true));
        assert!(started.elapsed() < std::time::Duration::from_secs(1), "{:?}", started.elapsed());
        page.join().unwrap();
        // Never taken: given up after the wait, and too late to take.
        let (id, rx) = super::await_say();
        assert_eq!(super::wait_for_answer(id, &rx, short, long), Heard::NotTaken);
        assert!(!super::take_say(id));
        assert!(super::stop_awaiting(id).is_none(), "taken off the list");
        // Taken in time, answered after the wait: heard in the grace.
        let (id, rx) = super::await_say();
        assert!(super::take_say(id));
        let page = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(80));
            super::stop_awaiting(id).unwrap().send(true).unwrap();
        });
        assert_eq!(super::wait_for_answer(id, &rx, short, long), Heard::Answered(true));
        page.join().unwrap();
        // Taken, and never answered: silent, after the grace.
        let (id, rx) = super::await_say();
        assert!(super::take_say(id));
        assert_eq!(super::wait_for_answer(id, &rx, short, short), Heard::Silent);
        assert!(super::stop_awaiting(id).is_some(), "left for the caller to take off");
    }

    /// The handshake's links, held in place: the shell waits for the
    /// page's answer and goes by it, and `companion_said` passes the page's
    /// word on as it came.
    #[test]
    fn a_turn_goes_by_the_pages_answer() {
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        let body = |name: &str| {
            let at = src.find(&format!("fn {name}(")).unwrap_or_else(|| panic!("no fn {name}"));
            src[at..at + src[at..].find("\n}\n").unwrap()].to_owned()
        };
        let ask = body("ask_the_brain");
        let waits = ask.find("wait_for_answer(id, &answer, SAY_WAIT, SAY_GRACE)").expect("it waits for the answer");
        assert!(ask[waits..].contains("stop_awaiting(id);"), "then takes the turn off the list, whatever was heard");
        assert!(ask.trim_end().ends_with("say_outcome(heard).map_err(str::to_owned)"), "and goes by it");
        assert!(body("companion_said").contains("waiting.send(sent)"), "the page's word, as it came");
        assert!(body("companion_take").trim_end().ends_with("Ok(take_say(id))"), "the shell decides whether it may ask");
    }

    /// The prompt closes only once its turn is taken, and with the open it
    /// was asked in (read before the wait, which a new open can outlast).
    #[test]
    fn a_turn_from_the_prompt_closes_it_only_once_taken() {
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        let at = src.find("async fn prompt_say(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        let open = body.find("let open = prompt::opens();").expect("the open read first");
        let asked = body.find("ask_the_brain(&app, &text, \"the prompt\").await?;").expect("the turn asked");
        let closed = body.find("prompt::close_later(&app, prompt::Why::Sent, open);").expect("then closed");
        assert!(open < asked && asked < closed);
    }

    /// The page reads these keys from `companion_config`, and the control
    /// window these from a report; every sample the fixture holds is a
    /// report the shell keeps, as it was sent.
    #[test]
    fn the_companion_config_and_status_are_the_wire_contract() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let keys = |v: &Value| {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort_unstable();
            k
        };
        let want = |name: &str| -> Vec<String> {
            fixture[name].as_array().unwrap().iter().map(|k| k.as_str().unwrap().to_owned()).collect()
        };
        let config = serde_json::to_value(super::companion_config_for(
            &crate::settings::Settings::default(),
            Some("t0k".into()),
        ))
        .unwrap();
        assert_eq!(keys(&config), want("companionConfigKeys"));
        assert_eq!(config["token"], "t0k");
        let none = serde_json::to_value(super::companion_config_for(&crate::settings::Settings::default(), None)).unwrap();
        assert_eq!(none["token"], Value::Null);
        let status = serde_json::to_value(crate::companion::CompanionStatus::before_any_report(true)).unwrap();
        assert_eq!(keys(&status), want("companionStatusKeys"));
        for (name, sample) in fixture["companionStatusSamples"].as_object().unwrap() {
            let kept = crate::companion::status_from(sample.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(&serde_json::to_value(kept).unwrap(), sample, "{name}");
        }
    }

    /// The check is in the commands, before anything else: there is no app
    /// manifest, so nothing else stops another window calling them (and
    /// `companion_config` hands over the token).
    #[test]
    fn who_is_calling_is_checked_first() {
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        for (name, guard) in [
            ("companion_config", "from_the_avatar"),
            ("companion_report", "from_the_avatar"),
            ("companion_take", "from_the_avatar"),
            ("companion_said", "from_the_avatar"),
            ("prompt_say", "from_the_prompt"),
            ("prompt_close", "from_the_prompt"),
            ("prompt_ready", "from_the_prompt"),
            ("set_hotkey", "from_the_control"),
            ("avatar_page", "from_the_avatar"),
            ("set_hit_rects", "from_the_avatar"),
        ] {
            let at = src.find(&format!("async fn {name}(")).unwrap_or_else(|| panic!("no command {name}"));
            let body = &src[at + src[at..].find('{').unwrap() + 1..];
            let first = body.lines().map(str::trim).find(|l| !l.is_empty()).unwrap();
            assert_eq!(first, format!("{guard}(webview.label())?;"), "{name}");
        }
    }

    /// The pretend say is read only where the build says debug: the one
    /// call passes `cfg!(debug_assertions)`, so a release build never asks.
    #[test]
    fn the_pretend_say_is_read_only_in_a_debug_build() {
        let all = include_str!("lib.rs").replace("\r\n", "\n");
        // The code, not these tests (which name the call too).
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let calls: Vec<_> = src.match_indices("companion::pretend_say_text(").collect();
        assert_eq!(calls.len(), 1, "one caller");
        let call = &src[calls[0].0..];
        assert!(call.starts_with("companion::pretend_say_text(cfg!(debug_assertions), "), "{}", &call[..80]);
    }

    /// A shape is kept only for his window as it stands: open, and not on
    /// its way out (a closing window's page may still be sending).
    #[test]
    fn a_shape_is_only_for_his_open_window() {
        use crate::overlay_machine::PhaseName;
        for (open, phase, want) in [
            (true, PhaseName::Starting, Some(7)),
            (true, PhaseName::Shown, Some(7)),
            (true, PhaseName::Painted, Some(7)),
            (true, PhaseName::Closing, None),
            (false, PhaseName::Closed, None),
            (false, PhaseName::Painted, None),
        ] {
            assert_eq!(super::live_window(open, phase, 7), want, "{open} {phase:?}");
        }
    }

    /// His page starting to load numbers the load and forgets the shape of
    /// the page before it, and his window going forgets it too: a shape
    /// must never outlive the page that measured it.
    #[test]
    fn his_shape_is_forgotten_as_his_page_loads_again() {
        let all = include_str!("lib.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let at = src.find(".on_page_load(|webview, payload| {").unwrap();
        let body = &src[at..at + src[at..].find("\n        })\n").unwrap()];
        assert!(
            body.contains(
                "if webview.label() == OVERLAY_LABEL && payload.event() == PageLoadEvent::Started {\n                let page = hit_shape::page_started();"
            ),
            "{body}"
        );
        let at = src.find(".on_window_event(|window, event| {").unwrap();
        let body = &src[at..at + src[at..].find("\n        })\n").unwrap()];
        let gone = &body[body.find("if matches!(event, WindowEvent::Destroyed) {").unwrap()..];
        assert!(gone[..gone.find('}').unwrap()].contains("hit_shape::forget();"), "{gone}");
        // The page's number is the shell's count, asked by his page alone.
        let at = src.find("async fn avatar_page(").unwrap();
        assert!(src[at..at + 200].contains("Ok(hit_shape::page())"));
    }

    /// The pretend prompt is read only where the build says debug: the one
    /// call passes `cfg!(debug_assertions)`, so a release build never opens it.
    #[test]
    fn the_pretend_prompt_is_read_only_in_a_debug_build() {
        let all = include_str!("lib.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let calls: Vec<_> = src.match_indices("prompt::pretend_wanted(").collect();
        assert_eq!(calls.len(), 1, "one caller");
        assert!(src[calls[0].0..].starts_with("prompt::pretend_wanted(cfg!(debug_assertions), "));
        assert!(!crate::prompt::pretend_wanted(false, Some("1".into())), "never in a release build");
        assert!(crate::prompt::pretend_wanted(true, Some("1".into())));
        assert!(!crate::prompt::pretend_wanted(true, Some("".into())));
        assert!(!crate::prompt::pretend_wanted(true, None));
    }

    /// The prompt's key opens it right there in the hotkey handler, inside
    /// WM_HOTKEY, never posted or spawned: only there may it take the
    /// keyboard.
    #[test]
    fn the_prompt_key_opens_it_inline() {
        let src = include_str!("lib.rs").replace("\r\n", "\n");
        let at = src.find("fn on_shortcut(").unwrap();
        let body = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(body.contains("if prompt::hotkey_request(id, pressed) {"), "on its key's press only");
        assert!(body.contains("prompt::toggle(app, prompt::Opener::Hotkey);"));
        for posted in ["run_on_main_thread", "spawn", "close_later"] {
            assert!(!body.contains(posted), "on_shortcut must not {posted}");
        }
        assert_eq!(crate::hotkeys::PROMPT, "prompt:toggle");
    }

    /// The avatar window is built at the first size, which leaves the
    /// bubble its room (the page's HEADROOM is 0.28: 640 × 0.72 ≈ 460).
    #[test]
    fn the_first_size_leaves_room_for_the_bubble() {
        assert!((super::FIRST_SIZE.0 - 340.0).abs() < f64::EPSILON && (super::FIRST_SIZE.1 - 640.0).abs() < f64::EPSILON);
        let src = include_str!("overlay_windows.rs");
        assert!(src.contains(".inner_size(crate::FIRST_SIZE.0, crate::FIRST_SIZE.1)"), "build_window uses FIRST_SIZE");
        let framing = include_str!("../../src/avatar/framing.ts");
        assert!(framing.contains("export const HEADROOM = 0.28;"), "the page's headroom");
        assert!((super::FIRST_SIZE.1 * (1.0 - 0.28) - 460.0_f64).abs() < 1.0);
    }

    /// He starts at FIRST_SIZE in the work area's bottom right; a work area
    /// too short for it gets a shorter window, never one placed off its top
    /// (where the bubble goes).
    #[test]
    fn he_first_stands_inside_the_work_area() {
        use crate::windows_ext::Bounds;
        let big = Bounds { left: 0, top: 0, right: 3840, bottom: 2400 };
        let r = super::first_rect_in(big, 192).unwrap();
        assert_eq!((r.x, r.y, r.width, r.height), (3840 - 680 - 48, 2400 - 1280 - 48, 680, 1280));
        // 1366x768 at 125%, a 48-pixel taskbar: 800 pixels tall won't fit.
        let short = Bounds { left: 0, top: 0, right: 1366, bottom: 720 };
        let r = super::first_rect_in(short, 120).unwrap();
        assert_eq!(r.y, 30, "a margin from the top, not above it");
        assert_eq!(i64::from(r.y) + i64::from(r.height), 720 - 30);
        assert_eq!(r.width, 425);
        // A second monitor's work area, left of and above the primary's origin.
        let off = Bounds { left: -1920, top: -200, right: 0, bottom: 880 };
        let r = super::first_rect_in(off, 96).unwrap();
        assert!(r.y >= off.top + 24 && r.x >= off.left + 24, "{r:?}");
        assert!(super::first_rect_in(Bounds { left: 0, top: 0, right: 40, bottom: 40 }, 96).is_none());
    }

    /// The shell's limit on a turn's length is the protocol's.
    #[test]
    fn a_turns_length_is_the_protocols() {
        let ts = include_str!("../../src/shared/protocol.ts");
        assert!(ts.contains(&format!("export const SAY_MAX_CHARS = {};", crate::companion::SAY_MAX_CHARS)));
    }

    #[test]
    fn printable_text_has_no_control_characters_and_a_length() {
        assert_eq!(super::printable("a\r\nb\u{1b}c", 10), "abc");
        assert_eq!(super::printable("😀😀😀", 2), "😀😀");
    }
}
