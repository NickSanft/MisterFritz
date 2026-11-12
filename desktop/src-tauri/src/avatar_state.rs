//! Where Fritz stands and whether he's hidden, kept across restarts in
//! `%APPDATA%\com.divora.fritz\avatar-state.json`. A stand-in for PR 2's
//! settings store: written atomically, and a corrupt file is set aside, not
//! fatal.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::overlay_machine::OverlayRect;

/// How often the saver looks for a change.
const SAVE_EVERY: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl From<OverlayRect> for Rect {
    fn from(r: OverlayRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

impl From<Rect> for OverlayRect {
    fn from(r: Rect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvatarState {
    pub rect: Option<Rect>,
    pub parked: bool,
}

/// The saved state; the default when there is none. A file that doesn't
/// parse is renamed `.bad` so the next save doesn't destroy the evidence.
pub fn load(path: &Path) -> AvatarState {
    let Ok(text) = fs::read_to_string(path) else {
        return AvatarState::default();
    };
    serde_json::from_str(&text).unwrap_or_else(|e| {
        tracing::warn!(error = %e, "avatar state unreadable; starting fresh");
        let _ = fs::rename(path, path.with_extension("json.bad"));
        AvatarState::default()
    })
}

/// Write `state` atomically: a temporary file, then a rename over the old.
pub fn save(path: &Path, state: &AvatarState) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(state)?)?;
    fs::rename(&tmp, path)
}

/// The state as the shell holds it now.
pub fn current(app: &AppHandle) -> AvatarState {
    AvatarState {
        rect: crate::overlay_windows::standing(app).map(Rect::from),
        // The owner's Hide only: a park by the cover watch is undone when
        // the fullscreen app goes, and must not outlive a restart.
        parked: crate::overlay_windows::owner_parked(app),
    }
}

/// Save whenever the state changes (checked every few seconds), so a drag in
/// Arrange or a park survives a crash as well as a Quit.
pub fn start_saver(app: AppHandle, path: PathBuf, mut last: AvatarState) {
    let spawned = thread::Builder::new()
        .name("fritz-state-saver".into())
        .spawn(move || loop {
            thread::sleep(SAVE_EVERY);
            let now = current(&app);
            if now != last && now.rect.is_some() {
                match save(&path, &now) {
                    Ok(()) => last = now,
                    Err(e) => tracing::warn!(error = %e, "couldn't save the avatar state"),
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the state saver");
    }
}

#[cfg(test)]
mod tests {
    use super::{load, save, AvatarState, Rect};
    use std::fs;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fritz-state-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips() {
        let path = dir("round").join("avatar-state.json");
        let state = AvatarState {
            rect: Some(Rect {
                x: -1200,
                y: 40,
                width: 300,
                height: 460,
            }),
            parked: true,
        };
        save(&path, &state).unwrap();
        assert_eq!(load(&path), state);
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn missing_is_the_default() {
        assert_eq!(
            load(&dir("missing").join("nope.json")),
            AvatarState::default()
        );
    }

    #[test]
    fn corrupt_is_set_aside_not_fatal() {
        let path = dir("corrupt").join("avatar-state.json");
        fs::write(&path, b"{ not json").unwrap();
        assert_eq!(load(&path), AvatarState::default());
        assert!(!path.exists());
        assert!(path.with_extension("json.bad").exists());
    }
}
