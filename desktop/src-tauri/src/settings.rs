//! Fritz's settings: `%APPDATA%\com.divora.fritz\settings.json`, the schema
//! of plan 14's build § 4.2, loaded once before the app is built (so a
//! window's first `get_settings` sees them) and saved on every change.
//!
//! The store is plumbing from LorraineAtelier (the owner's own app), and of
//! its fields only `WidgetPos`: a `OnceLock<Mutex<…>>`, the path from
//! `%APPDATA%` rather than an `AppHandle`, serde defaults so an old file
//! still loads, and a file that doesn't parse set aside rather than lost.
//! Fixed on the way:
//! - a file that exists but can't be *read* (a sharing violation from a sync
//!   client, bad permissions) is never saved over: this run keeps its changes
//!   in memory only, and says so;
//! - each corrupt file keeps its own name, so a second never replaces the
//!   first;
//! - a save is durable (`sync_all` before the rename) and atomic, with no
//!   direct-write fallback that would undo both, and one that fails is tried
//!   again by the next change or the saver's next look;
//! - changes are serialised with their saves and applied to the store as it
//!   is under the lock, so no change undoes another;
//! - a value Fritz can't use (a hand edit gone wrong) is put back to its
//!   default at load, alone, rather than blocking every later change.
//!
//! The companion token is not here (plan 14 build § 4.2): it gets a file of
//! its own in PR 3.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The app's data folder under `%APPDATA%`: tauri.conf.json's `identifier`,
/// which is also what Tauri's `app_data_dir` resolves to (a test holds them
/// together).
pub const APP_DIR: &str = "com.divora.fritz";
pub const FILE: &str = "settings.json";
/// Spike 0's stand-in, imported once into `avatar` when there is no
/// settings file yet.
pub const SPIKE_STATE: &str = "avatar-state.json";

pub const VERSION: u32 = 1;

/// Paths a window may not set: the shell owns where Fritz stands and whether
/// he is hidden (it saves them as they change), and the version is the
/// file's, not a preference.
const READ_ONLY: [&str; 3] = ["version", "avatar.rect", "avatar.parked"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    pub version: u32,
    /// PR 3: where the brain is, and who this desk is to it.
    pub brain: Brain,
    pub avatar: Avatar,
    /// PR 3.
    pub theme: Theme,
    pub hotkeys: Hotkeys,
    /// PR 8 and 10.
    pub audio: Audio,
    /// Phase 3: Lorraine's widget positions, by widget id.
    pub widget_layout: BTreeMap<String, WidgetPos>,
    /// Phase 5.
    pub run_at_startup: bool,
    pub diagnostics: Diagnostics,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: VERSION,
            brain: Brain::default(),
            avatar: Avatar::default(),
            theme: Theme::default(),
            hotkeys: Hotkeys::default(),
            audio: Audio::default(),
            widget_layout: BTreeMap::new(),
            run_at_startup: false,
            diagnostics: Diagnostics::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Brain {
    pub url: String,
    /// The protocol's `ClientId`: `^[a-z0-9][a-z0-9-]{0,31}$`.
    pub client_id: String,
    pub client_name: String,
}

impl Default for Brain {
    fn default() -> Self {
        let machine = std::env::var("COMPUTERNAME").unwrap_or_default();
        Self {
            url: "ws://127.0.0.1:8001/companion".to_owned(),
            client_id: client_id_from(&machine),
            client_name: if machine.trim().is_empty() {
                "this PC".to_owned()
            } else {
                machine.trim().to_owned()
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Avatar {
    /// Where he last stood, in physical pixels; `None` until he has stood
    /// anywhere (he then starts bottom-right of the primary work area). The
    /// plan's `x, y, w, h`, with room to say "never placed".
    pub rect: Option<SavedRect>,
    /// The page's model scale. Not read yet: the model fills whatever size
    /// Arrange gives the window.
    pub scale: f32,
    /// The owner's Hide. A park by the cover watch is never saved.
    pub parked: bool,
    /// A VRM to use instead of `avatar\fritz.vrm` in the data folder.
    pub vrm_path: Option<String>,
    /// Whether his eyes follow the cursor (PR 2's second part).
    pub lookat_cursor: bool,
}

impl Default for Avatar {
    fn default() -> Self {
        Self {
            rect: None,
            scale: 1.0,
            parked: false,
            vrm_path: None,
            lookat_cursor: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Theme {
    pub theme: String,
    pub mood: String,
    pub accent: String,
    pub motion: f32,
    pub grain: bool,
    pub motion_user: bool,
}

impl Default for Theme {
    /// Plan 14 build § 7, Q9.
    fn default() -> Self {
        Self {
            theme: "dark".to_owned(),
            mood: "ink".to_owned(),
            accent: "brand".to_owned(),
            motion: 0.6,
            grain: true,
            motion_user: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Hotkeys {
    /// PR 4.
    pub prompt: String,
    /// PR 8.
    pub ptt: String,
    pub arrange: String,
    pub park: String,
}

impl Default for Hotkeys {
    /// Plan 14 build § 7, Q6.
    fn default() -> Self {
        Self {
            prompt: "Ctrl+Alt+F".to_owned(),
            ptt: "Ctrl+Alt+Space".to_owned(),
            arrange: "Ctrl+Alt+A".to_owned(),
            park: "Ctrl+Alt+H".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Audio {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub muted: bool,
    /// Plan 14 build § 7, Q2.
    pub tts_voice: String,
}

impl Default for Audio {
    fn default() -> Self {
        Self {
            input_device: None,
            output_device: None,
            muted: false,
            tts_voice: "bm_george".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct WidgetPos {
    pub x: f64,
    pub y: f64,
    pub scale: f64,
    pub vertical: bool,
}

impl Default for WidgetPos {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            scale: 1.0,
            vertical: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Diagnostics {
    /// Spike 0's foreground logger (`diagnostics\foreground.log`). Off by
    /// default: it writes other apps' window titles to disk. Read at start.
    pub foreground_log: bool,
}

/// A `ClientId` from a machine name: lower case, anything else a dash, no
/// leading or doubled dashes, at most 32 characters; `"desk"` if nothing is
/// left.
pub fn client_id_from(machine: &str) -> String {
    let mut id = String::new();
    for c in machine.trim().chars().flat_map(char::to_lowercase) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            id.push(c);
        } else if !id.is_empty() && !id.ends_with('-') {
            id.push('-');
        }
    }
    id.truncate(32);
    let id = id.trim_end_matches('-').to_owned();
    if id.is_empty() {
        "desk".to_owned()
    } else {
        id
    }
}

fn is_client_id(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 32
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b
            .iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
}

/// Put back to its default every value serde accepts but Fritz can't use,
/// one field at a time, and say what each was. Run on what is loaded, so a
/// hand edit gone wrong (`"Win+Shift+H"`: the hotkey library says `Super`)
/// costs that one setting, not every later change to any other.
pub fn repair(s: &mut Settings) -> Vec<String> {
    let d = Settings::default();
    let mut fixed = Vec::new();
    if !is_client_id(&s.brain.client_id) {
        fixed.push(format!(
            "brain.clientId {:?} isn't a client id: lower-case letters, digits and dashes, \
             starting with a letter or digit, at most 32",
            s.brain.client_id
        ));
        s.brain.client_id = d.brain.client_id.clone();
    }
    for (field, value, ok, default) in [
        ("theme.theme", &mut s.theme.theme, &["dark", "light"][..], &d.theme.theme),
        ("theme.mood", &mut s.theme.mood, &["ink", "dusk", "midnight"][..], &d.theme.mood),
        ("theme.accent", &mut s.theme.accent, &["brand", "abyssal", "ember"][..], &d.theme.accent),
    ] {
        if !ok.contains(&value.as_str()) {
            fixed.push(format!("{field} is {value:?}, not one of {ok:?}"));
            value.clone_from(default);
        }
    }
    if !(0.0..=1.0).contains(&s.theme.motion) {
        fixed.push(format!("theme.motion {} isn't between 0 and 1", s.theme.motion));
        s.theme.motion = d.theme.motion;
    }
    if !(0.25..=4.0).contains(&s.avatar.scale) {
        fixed.push(format!("avatar.scale {} isn't between 0.25 and 4", s.avatar.scale));
        s.avatar.scale = d.avatar.scale;
    }
    if s.avatar.vrm_path.as_deref().is_some_and(|p| p.trim().is_empty()) {
        fixed.push("avatar.vrmPath is empty; use null for the usual place".to_owned());
        s.avatar.vrm_path = None;
    }
    for (field, accel, default) in [
        ("hotkeys.prompt", &mut s.hotkeys.prompt, &d.hotkeys.prompt),
        ("hotkeys.ptt", &mut s.hotkeys.ptt, &d.hotkeys.ptt),
        ("hotkeys.arrange", &mut s.hotkeys.arrange, &d.hotkeys.arrange),
        ("hotkeys.park", &mut s.hotkeys.park, &d.hotkeys.park),
    ] {
        if let Err(e) = crate::hotkeys::parse(accel) {
            fixed.push(format!("{field}: {e}"));
            accel.clone_from(default);
        }
    }
    fixed
}

/// What `set` refuses that serde would accept: anything `repair` would put
/// back.
pub fn validate(s: &Settings) -> Result<(), String> {
    match repair(&mut s.clone()).into_iter().next() {
        Some(problem) => Err(problem),
        None => Ok(()),
    }
}

/// The data folder: `%APPDATA%\com.divora.fritz`. A test build uses a
/// folder of its own, so no test reads or writes the owner's.
pub fn app_dir() -> PathBuf {
    if cfg!(test) {
        return std::env::temp_dir().join(format!("fritz-test-appdir-{}", std::process::id()));
    }
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(APP_DIR)
}

/// What `load` found, for the start-up log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loaded {
    /// No settings file: the defaults.
    Missing,
    Read,
    /// No settings file, but Spike 0's `avatar-state.json`: imported.
    Imported,
    /// The file didn't parse: moved here, and the defaults used.
    SetAside(PathBuf),
    /// The file exists but couldn't be read: the defaults, and nothing saved
    /// over it this run.
    Unreadable(String),
}

impl Loaded {
    pub fn describe(&self) -> String {
        match self {
            Loaded::Missing => "none yet: the defaults".to_owned(),
            Loaded::Read => "read".to_owned(),
            Loaded::Imported => format!("imported Spike 0's {SPIKE_STATE}"),
            Loaded::SetAside(p) => format!(
                "didn't parse, so it was moved to {} and the defaults used",
                p.display()
            ),
            Loaded::Unreadable(e) => format!(
                "couldn't be read ({e}), so the defaults are used and nothing is saved over \
                 it this run"
            ),
        }
    }
}

struct Store {
    settings: Settings,
    path: PathBuf,
    writable: bool,
    /// The last save failed: the file is behind the store, and the next
    /// change, or the saver's next look, writes again.
    dirty: bool,
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(Store {
            settings: Settings::default(),
            path: app_dir().join(FILE),
            writable: true,
            dirty: false,
        })
    })
}

fn lock() -> MutexGuard<'static, Store> {
    store().lock().unwrap_or_else(|e| e.into_inner())
}

/// Changes one at a time, each with its save: taken before the store lock,
/// never after it.
static SAVING: Mutex<()> = Mutex::new(());

fn saving() -> MutexGuard<'static, ()> {
    SAVING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Read `path`. Pure apart from the file system; `load` adds Spike 0's
/// import and installs the result. Bytes that aren't UTF-8 are a corrupt
/// file, set aside like any other; only a read the system refused (a sync
/// client or scanner holding the file) is "unreadable".
pub fn read(path: &Path) -> (Settings, Loaded) {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return (Settings::default(), Loaded::Missing),
        Err(e) => return (Settings::default(), Loaded::Unreadable(e.to_string())),
    };
    // A byte-order mark is what Windows PowerShell's `-Encoding utf8` writes,
    // and the README says to edit this file by hand.
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&bytes);
    match serde_json::from_slice::<Settings>(body) {
        Ok(s) => (s, Loaded::Read),
        Err(e) => {
            tracing::warn!(error = %e, "settings didn't parse");
            match set_aside(path) {
                Ok(to) => (Settings::default(), Loaded::SetAside(to)),
                // Couldn't move it: don't save over it either.
                Err(e) => (Settings::default(), Loaded::Unreadable(e.to_string())),
            }
        }
    }
}

/// Move a corrupt file to `settings.corrupt-<unix seconds>[-n].json`, a name
/// no earlier one has.
fn set_aside(path: &Path) -> io::Result<PathBuf> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("settings");
    for n in 0..1000u32 {
        let name = if n == 0 {
            format!("{stem}.corrupt-{secs}.json")
        } else {
            format!("{stem}.corrupt-{secs}-{n}.json")
        };
        let to = path.with_file_name(name);
        if !to.exists() {
            fs::rename(path, &to)?;
            return Ok(to);
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists, "no free name to set it aside"))
}

/// Spike 0's `{ rect, parked }`, as it wrote them.
#[derive(Deserialize)]
struct SpikeState {
    rect: Option<SavedRect>,
    #[serde(default)]
    parked: bool,
}

/// Load the settings into the store, from `dir`. Once, first thing in
/// `run()`, before any window exists. Also returns what `repair` put back,
/// one line each, for the start-up log; the repaired values are saved.
pub fn load_from(dir: &Path) -> (Loaded, Vec<String>) {
    let path = dir.join(FILE);
    let (mut settings, mut loaded) = read(&path);
    if loaded == Loaded::Missing {
        let spike = dir.join(SPIKE_STATE);
        if let Some(old) = fs::read_to_string(&spike)
            .ok()
            .and_then(|t| serde_json::from_str::<SpikeState>(&t).ok())
        {
            settings.avatar.rect = old.rect;
            settings.avatar.parked = old.parked;
            loaded = Loaded::Imported;
        }
    }
    let repaired = repair(&mut settings);
    let writable = !matches!(loaded, Loaded::Unreadable(_));
    {
        let mut s = lock();
        s.settings = settings;
        s.path = path.clone();
        s.writable = writable;
        s.dirty = false;
    }
    if loaded == Loaded::Imported {
        // Saved at once, and only then the old file renamed, so it is
        // imported once and never lost: if the save fails, the next start
        // imports again.
        if save_current() {
            let _ = fs::rename(dir.join(SPIKE_STATE), dir.join("avatar-state.imported.json"));
        }
    } else if !repaired.is_empty() && writable {
        save_current();
    }
    (loaded, repaired)
}

pub fn load() -> (Loaded, Vec<String>) {
    load_from(&app_dir())
}

/// The settings now.
pub fn get() -> Settings {
    lock().settings.clone()
}

/// Change the settings with `f`, and save them if they changed (or if the
/// last save failed). Returns what `f` returned and the settings after it.
pub fn update<R>(f: impl FnOnce(&mut Settings) -> R) -> (R, Settings) {
    let _saving = saving();
    apply(f)
}

/// `update`'s work; the caller holds `SAVING`.
fn apply<R>(f: impl FnOnce(&mut Settings) -> R) -> (R, Settings) {
    let (r, after, write, path, writable) = {
        let mut s = lock();
        let before = s.settings.clone();
        let r = f(&mut s.settings);
        let after = s.settings.clone();
        let write = after != before || s.dirty;
        (r, after, write, s.path.clone(), s.writable)
    };
    if write {
        let saved = save_to(&path, &after, writable);
        if writable {
            lock().dirty = !saved;
        }
    }
    (r, after)
}

fn save_current() -> bool {
    let _saving = saving();
    let (settings, path, writable) = {
        let s = lock();
        (s.settings.clone(), s.path.clone(), s.writable)
    };
    let saved = save_to(&path, &settings, writable);
    if writable {
        lock().dirty = !saved;
    }
    saved
}

fn save_to(path: &Path, settings: &Settings, writable: bool) -> bool {
    if !writable {
        tracing::warn!(
            "settings.json couldn't be read at start, so this change isn't saved over it"
        );
        return false;
    }
    match write(path, settings) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "couldn't save the settings");
            false
        }
    }
}

/// Write `settings` to `path` atomically and durably: a temporary file of
/// this save's own, flushed to disk, then renamed over the old file. A
/// rename refused for a moment (a virus scanner or sync client holding the
/// file) is tried again; a write that can't be made atomic isn't made.
pub fn write(path: &Path, settings: &Settings) -> io::Result<()> {
    static N: AtomicU64 = AtomicU64::new(0);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or(FILE);
    let tmp = path.with_file_name(format!(
        "{name}.{}-{}.tmp",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes = serde_json::to_vec_pretty(settings)?;
    let written = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    #[cfg(test)]
    {
        // A test's slow save: the next write, once, waits before its rename.
        let ms = tests::DELAY_NEXT_WRITE_MS.swap(0, Ordering::SeqCst);
        if ms > 0 {
            thread::sleep(Duration::from_millis(ms));
        }
    }
    const TRIES: u32 = 5;
    let mut last = None;
    for attempt in 0..TRIES {
        match fs::rename(&tmp, path) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Some(e);
                if attempt + 1 < TRIES {
                    thread::sleep(Duration::from_millis(20 << attempt));
                }
            }
        }
    }
    let _ = fs::remove_file(&tmp);
    Err(last.unwrap_or_else(|| io::Error::other("rename failed")))
}

/// Set one value by its camelCase path, as a window sends it
/// (`"avatar.lookatCursor"`, `false`). The path must already exist; an
/// object may only be replaced by one with the same keys; the result must
/// still be settings serde accepts and `validate` passes. The shell's own
/// fields, and any group holding one, are refused.
///
/// Applied to the settings as they are under the lock, never to an earlier
/// copy, so two windows' changes, or a change and the shell's saver, can't
/// undo each other. `announce` runs with the result before the next change
/// may start, so announcements go out in the order the changes were made.
pub fn set(path: &str, value: Value, announce: impl FnOnce(&Settings)) -> Result<Settings, String> {
    let _saving = saving();
    let (r, after) = apply(|s| with_value(s, path, value).map(|next| *s = next));
    r?;
    announce(&after);
    Ok(after)
}

/// `current` with `path` set to `value`, checked; the store is not touched.
pub fn with_value(current: &Settings, path: &str, value: Value) -> Result<Settings, String> {
    if let Some(kept) = READ_ONLY.iter().find(|r| {
        **r == path || path.starts_with(&format!("{r}.")) || r.starts_with(&format!("{path}."))
    }) {
        return Err(if *kept == path || path.starts_with(&format!("{kept}.")) {
            format!("{path} is kept by Fritz himself and can't be set")
        } else {
            format!("{path} holds {kept}, which Fritz keeps himself: set its other parts one by one")
        });
    }
    let mut tree = serde_json::to_value(current).map_err(|e| e.to_string())?;
    let mut at = &mut tree;
    for key in path.split('.') {
        at = at
            .as_object_mut()
            .and_then(|o| o.get_mut(key))
            .ok_or_else(|| format!("there is no setting {path}"))?;
    }
    if let (Some(old), Some(new)) = (at.as_object(), value.as_object()) {
        let (mut a, mut b): (Vec<_>, Vec<_>) = (old.keys().collect(), new.keys().collect());
        a.sort();
        b.sort();
        if a != b {
            return Err(format!("{path} must be set whole, with the keys {a:?}"));
        }
    } else if at.is_object() {
        return Err(format!("{path} is a group of settings; set its parts"));
    }
    *at = value;
    let next: Settings =
        serde_json::from_value(tree).map_err(|e| format!("{path}: {e}"))?;
    validate(&next)?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Read and cleared by `write`: makes the next save slow.
    pub(super) static DELAY_NEXT_WRITE_MS: AtomicU64 = AtomicU64::new(0);

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fritz-settings-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// The store is one per process, so tests that load into it take turns.
    fn serial() -> MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_folder_is_the_apps_identifier() {
        let conf: Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        assert_eq!(conf["identifier"], APP_DIR);
    }

    #[test]
    fn the_defaults_are_the_plans() {
        let s = Settings::default();
        assert_eq!(s.version, 1);
        assert_eq!(s.brain.url, "ws://127.0.0.1:8001/companion");
        assert_eq!(
            (s.hotkeys.prompt.as_str(), s.hotkeys.ptt.as_str(), s.hotkeys.arrange.as_str(), s.hotkeys.park.as_str()),
            ("Ctrl+Alt+F", "Ctrl+Alt+Space", "Ctrl+Alt+A", "Ctrl+Alt+H")
        );
        assert_eq!(
            (s.theme.theme.as_str(), s.theme.mood.as_str(), s.theme.accent.as_str()),
            ("dark", "ink", "brand")
        );
        assert!((s.theme.motion - 0.6).abs() < 1e-6 && s.theme.grain && !s.theme.motion_user);
        assert_eq!(s.audio.tts_voice, "bm_george");
        assert!(s.avatar.lookat_cursor && !s.avatar.parked && s.avatar.rect.is_none());
        assert!(!s.diagnostics.foreground_log, "the foreground log is opt-in");
        validate(&s).unwrap();
    }

    /// The file is camelCase, as the pages see it.
    #[test]
    fn the_file_is_camel_case() {
        let v = serde_json::to_value(Settings::default()).unwrap();
        assert!(v["avatar"].get("lookatCursor").is_some(), "{v}");
        assert!(v["brain"].get("clientId").is_some(), "{v}");
        assert!(v.get("runAtStartup").is_some() && v.get("widgetLayout").is_some(), "{v}");
        assert!(v["diagnostics"].get("foregroundLog").is_some(), "{v}");
    }

    #[test]
    fn client_ids_follow_the_protocols_pattern() {
        let cases: Vec<(String, String)> = vec![
            ("NICK-LAPTOP".into(), "nick-laptop".into()),
            ("  Desk_01 (2)  ".into(), "desk-01-2".into()),
            ("---".into(), "desk".into()),
            (String::new(), "desk".into()),
            ("Ünïcode PC".into(), "n-code-pc".into()),
            ("a".repeat(40), "a".repeat(32)),
            ("abc-------------------------------def".into(), "abc-def".into()),
            // Cut at 32 just after a dash: the dash goes too.
            (format!("{} b", "a".repeat(31)), "a".repeat(31)),
        ];
        for (machine, want) in cases {
            let got = client_id_from(&machine);
            assert_eq!(got, want, "{machine:?}");
            assert!(is_client_id(&got), "{got:?}");
        }
    }

    #[test]
    fn a_missing_file_is_the_defaults() {
        let d = dir("missing");
        assert_eq!(read(&d.join(FILE)), (Settings::default(), Loaded::Missing));
    }

    /// An old or partial file still loads: missing fields take their
    /// defaults, unknown ones are ignored.
    #[test]
    fn a_partial_file_keeps_what_it_has() {
        let d = dir("partial");
        fs::write(
            d.join(FILE),
            r#"{"avatar":{"parked":true},"someday":42,"hotkeys":{"park":"Ctrl+Shift+H"}}"#,
        )
        .unwrap();
        let (s, loaded) = read(&d.join(FILE));
        assert_eq!(loaded, Loaded::Read);
        assert!(s.avatar.parked && s.avatar.lookat_cursor);
        assert_eq!(s.hotkeys.park, "Ctrl+Shift+H");
        assert_eq!(s.hotkeys.arrange, "Ctrl+Alt+A");
    }

    /// A file that doesn't parse is moved aside under a name of its own; a
    /// second one doesn't replace the first.
    #[test]
    fn corrupt_files_are_set_aside_each_under_its_own_name() {
        let d = dir("corrupt");
        let path = d.join(FILE);
        fs::write(&path, b"{ not json").unwrap();
        let (s, first) = read(&path);
        assert_eq!(s, Settings::default());
        fs::write(&path, br#"{"avatar":{"parked":"yes"}}"#).unwrap();
        let (_, second) = read(&path);
        let (Loaded::SetAside(a), Loaded::SetAside(b)) = (&first, &second) else {
            panic!("{first:?} {second:?}");
        };
        assert_ne!(a, b);
        assert_eq!(fs::read(a).unwrap(), b"{ not json");
        assert_eq!(fs::read(b).unwrap(), br#"{"avatar":{"parked":"yes"}}"#);
        assert!(!path.exists());
    }

    /// A file that is there but can't be read is never saved over, even
    /// once whatever held it has let go.
    #[test]
    fn an_unreadable_file_is_never_saved_over() {
        use std::os::windows::fs::OpenOptionsExt;
        let _one = serial();
        let d = dir("unreadable");
        let path = d.join(FILE);
        let mine = br#"{"avatar":{"lookatCursor":true},"hotkeys":{"park":"Ctrl+Shift+H"}}"#;
        fs::write(&path, mine).unwrap();
        // Held with no sharing, as a sync client or a scanner may hold it.
        let held = fs::OpenOptions::new().read(true).share_mode(0).open(&path).unwrap();
        let (loaded, _) = load_from(&d);
        drop(held);
        assert!(matches!(loaded, Loaded::Unreadable(_)), "{loaded:?}");
        let (_, after) = update(|s| s.avatar.lookat_cursor = false);
        assert!(!after.avatar.lookat_cursor, "kept in memory");
        assert_eq!(fs::read(&path).unwrap(), mine, "not saved over");
    }

    #[test]
    fn bytes_that_arent_utf8_are_a_corrupt_file() {
        let d = dir("utf8");
        let path = d.join(FILE);
        fs::write(&path, [0xFF, 0xFE, b'{', b'}']).unwrap();
        let (s, loaded) = read(&path);
        assert_eq!(s, Settings::default());
        assert!(matches!(loaded, Loaded::SetAside(_)), "{loaded:?}");
    }

    /// Windows PowerShell's `-Encoding utf8` starts the file with a BOM.
    #[test]
    fn a_hand_edit_with_a_byte_order_mark_still_reads() {
        let d = dir("bom");
        let path = d.join(FILE);
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(br#"{"diagnostics":{"foregroundLog":true}}"#);
        fs::write(&path, bytes).unwrap();
        let (s, loaded) = read(&path);
        assert_eq!(loaded, Loaded::Read);
        assert!(s.diagnostics.foreground_log);
    }

    /// Saves are one at a time: a change made while an older one is still
    /// being written is saved after it, never overtaken by it.
    #[test]
    fn a_slow_save_is_never_overtaken() {
        let _one = serial();
        let d = dir("ordered");
        load_from(&d);
        DELAY_NEXT_WRITE_MS.store(300, Ordering::SeqCst);
        let slow = thread::spawn(|| update(|s| s.audio.tts_voice = "first".into()));
        thread::sleep(Duration::from_millis(60));
        update(|s| s.audio.tts_voice = "second".into());
        slow.join().unwrap();
        assert_eq!(get().audio.tts_voice, "second");
        assert_eq!(read(&d.join(FILE)).0.audio.tts_voice, "second", "the older save landed last");
    }

    #[test]
    fn saves_are_atomic_and_leave_nothing_behind() {
        let d = dir("save");
        let path = d.join(FILE);
        let mut s = Settings::default();
        s.avatar.rect = Some(SavedRect { x: -1200, y: 40, width: 300, height: 460 });
        write(&path, &s).unwrap();
        s.avatar.parked = true;
        write(&path, &s).unwrap();
        assert_eq!(read(&path), (s, Loaded::Read));
        let left: Vec<_> = fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from(FILE)], "{left:?}");
    }

    /// Many threads changing settings at once: the file ends as the last
    /// change left the store, and parses.
    #[test]
    fn concurrent_changes_end_as_the_store_does() {
        let _one = serial();
        let d = dir("concurrent");
        load_from(&d);
        let threads: Vec<_> = (0..8)
            .map(|t| {
                thread::spawn(move || {
                    for i in 0..10 {
                        update(|s| {
                            s.widget_layout.insert(
                                format!("w{t}"),
                                WidgetPos { x: f64::from(i), ..WidgetPos::default() },
                            );
                        });
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let (on_disk, loaded) = read(&d.join(FILE));
        assert_eq!(loaded, Loaded::Read);
        assert_eq!(on_disk, get());
        assert_eq!(on_disk.widget_layout.len(), 8);
        assert!(on_disk.widget_layout.values().all(|w| (w.x - 9.0).abs() < 1e-9));
    }

    #[test]
    fn spike_zeros_state_is_imported_once() {
        let _one = serial();
        let d = dir("import");
        fs::write(
            d.join(SPIKE_STATE),
            r#"{"rect":{"x":-1200,"y":40,"width":300,"height":460},"parked":true}"#,
        )
        .unwrap();
        assert_eq!(load_from(&d).0, Loaded::Imported);
        let s = get();
        assert_eq!(s.avatar.rect, Some(SavedRect { x: -1200, y: 40, width: 300, height: 460 }));
        assert!(s.avatar.parked);
        assert!(!d.join(SPIKE_STATE).exists());
        assert!(d.join("avatar-state.imported.json").exists());
        assert_eq!(read(&d.join(FILE)).0, s, "saved at once");
        assert_eq!(load_from(&d).0, Loaded::Read, "not imported again");
    }

    /// Spike 0's file is only ever a first start's: a settings file wins.
    #[test]
    fn existing_settings_win_over_spike_zeros_file() {
        let _one = serial();
        let d = dir("both");
        let mut mine = Settings::default();
        mine.avatar.rect = Some(SavedRect { x: 10, y: 20, width: 300, height: 460 });
        write(&d.join(FILE), &mine).unwrap();
        fs::write(d.join(SPIKE_STATE), r#"{"rect":{"x":-1,"y":-1,"width":9,"height":9},"parked":true}"#)
            .unwrap();
        assert_eq!(load_from(&d).0, Loaded::Read);
        assert_eq!(get(), mine);
        assert!(d.join(SPIKE_STATE).exists(), "left alone");
    }

    /// A hand edit gone wrong costs that one setting: it is put back to its
    /// default at load, said so, saved, and every other change still works.
    #[test]
    fn a_bad_value_is_put_back_at_load_and_blocks_nothing() {
        let _one = serial();
        let d = dir("repair");
        fs::write(
            d.join(FILE),
            r#"{"hotkeys":{"park":"Win+Shift+H","arrange":"Ctrl+Shift+F9"},
               "brain":{"clientId":"Nick-Laptop"},"theme":{"mood":"violet"}}"#,
        )
        .unwrap();
        let (loaded, repaired) = load_from(&d);
        assert_eq!(loaded, Loaded::Read);
        assert_eq!(repaired.len(), 3, "{repaired:?}");
        for field in ["hotkeys.park", "brain.clientId", "theme.mood"] {
            assert!(repaired.iter().any(|r| r.contains(field)), "{field}: {repaired:?}");
        }
        let s = get();
        assert_eq!(s.hotkeys.park, "Ctrl+Alt+H");
        assert_eq!(s.hotkeys.arrange, "Ctrl+Shift+F9", "the good ones are kept");
        assert_eq!(s.theme.mood, "ink");
        assert!(is_client_id(&s.brain.client_id));
        assert_eq!(read(&d.join(FILE)).0, s, "the repair is saved");
        set("avatar.lookatCursor", json!(false), |_| {}).unwrap();
        assert!(!get().avatar.lookat_cursor);
    }

    /// Two windows changing different settings at once, during a slow save,
    /// and the shell's saver moving him: every change survives, in the store
    /// and on disk.
    #[test]
    fn concurrent_sets_and_the_saver_never_undo_each_other() {
        let _one = serial();
        let d = dir("atomic");
        load_from(&d);
        DELAY_NEXT_WRITE_MS.store(300, Ordering::SeqCst);
        let slow = thread::spawn(|| set("avatar.lookatCursor", json!(false), |_| {}).unwrap());
        thread::sleep(Duration::from_millis(60));
        let other = thread::spawn(|| set("audio.muted", json!(true), |_| {}).unwrap());
        let saver = thread::spawn(|| {
            update(|s| s.avatar.rect = Some(SavedRect { x: 1, y: 2, width: 3, height: 4 }))
        });
        for t in [slow.join().is_ok(), other.join().is_ok(), saver.join().is_ok()] {
            assert!(t);
        }
        for s in [get(), read(&d.join(FILE)).0] {
            assert!(!s.avatar.lookat_cursor && s.audio.muted, "{s:?}");
            assert_eq!(s.avatar.rect, Some(SavedRect { x: 1, y: 2, width: 3, height: 4 }));
        }
    }

    /// Announcements go out in the order the changes were made: the last one
    /// a window hears is the settings as they are.
    #[test]
    fn changes_are_announced_in_the_order_they_were_made() {
        let _one = serial();
        let d = dir("announce");
        load_from(&d);
        let heard = std::sync::Arc::new(Mutex::new(Vec::<String>::new()));
        let h = heard.clone();
        let first = thread::spawn(move || {
            set("audio.ttsVoice", json!("first"), |s| {
                // A slow announcement: a second change must wait for it.
                thread::sleep(Duration::from_millis(200));
                h.lock().unwrap().push(s.audio.tts_voice.clone());
            })
        });
        thread::sleep(Duration::from_millis(60));
        let h = heard.clone();
        set("audio.ttsVoice", json!("second"), move |s| {
            h.lock().unwrap().push(s.audio.tts_voice.clone());
        })
        .unwrap();
        first.join().unwrap().unwrap();
        assert_eq!(*heard.lock().unwrap(), ["first", "second"]);
        assert_eq!(get().audio.tts_voice, "second");
    }

    /// A save that fails is tried again by the next change, or by the
    /// saver's next look, even when that changes nothing.
    #[test]
    fn a_failed_save_is_tried_again() {
        use std::os::windows::fs::OpenOptionsExt;
        let _one = serial();
        let d = dir("retry");
        load_from(&d);
        update(|s| s.audio.tts_voice = "saved".into());
        // Held without delete-sharing, the file can't be renamed over.
        const SHARE_READ_WRITE: u32 = 0x1 | 0x2;
        let held = fs::OpenOptions::new()
            .read(true)
            .share_mode(SHARE_READ_WRITE)
            .open(d.join(FILE))
            .unwrap();
        update(|s| s.audio.tts_voice = "pending".into());
        drop(held);
        assert_eq!(read(&d.join(FILE)).0.audio.tts_voice, "saved", "the save failed");
        update(|_| {});
        assert_eq!(read(&d.join(FILE)).0.audio.tts_voice, "pending", "and was tried again");
    }

    #[test]
    fn a_window_sets_one_value_by_its_path() {
        let s = Settings::default();
        let next = with_value(&s, "avatar.lookatCursor", json!(false)).unwrap();
        assert!(!next.avatar.lookat_cursor);
        let next = with_value(&s, "hotkeys.park", json!("Ctrl+Shift+F12")).unwrap();
        assert_eq!(next.hotkeys.park, "Ctrl+Shift+F12");
        let next = with_value(&s, "avatar.vrmPath", json!("C:\\models\\fritz.vrm")).unwrap();
        assert_eq!(next.avatar.vrm_path.as_deref(), Some("C:\\models\\fritz.vrm"));
        let next = with_value(&next, "avatar.vrmPath", Value::Null).unwrap();
        assert_eq!(next.avatar.vrm_path, None);
    }

    #[test]
    fn what_a_window_cant_set() {
        let s = Settings::default();
        for (path, value, says) in [
            ("version", json!(2), "kept by Fritz"),
            ("avatar.rect", Value::Null, "kept by Fritz"),
            ("avatar.rect.x", json!(5), "kept by Fritz"),
            ("avatar.parked", json!(true), "kept by Fritz"),
            ("avatar.nonsense", json!(1), "no setting"),
            ("avatar.lookatCursor", json!("yes"), "avatar.lookatCursor"),
            ("brain", json!({"url": "ws://elsewhere/companion"}), "set whole"),
            ("avatar", serde_json::to_value(Avatar::default()).unwrap(), "Fritz keeps himself"),
            ("theme", json!("dark"), "group of settings"),
            ("theme.mood", json!("violet"), "theme.mood"),
            ("theme.motion", json!(3), "between 0 and 1"),
            ("avatar.scale", json!(0), "between 0.25 and 4"),
            ("avatar.vrmPath", json!("  "), "empty"),
            ("brain.clientId", json!("Bad Id"), "client id"),
            ("hotkeys.arrange", json!("Ctrl+Alt+Nope"), "hotkeys.arrange"),
        ] {
            let err = with_value(&s, path, value).unwrap_err();
            assert!(err.contains(says), "{path}: {err}");
        }
    }
}
