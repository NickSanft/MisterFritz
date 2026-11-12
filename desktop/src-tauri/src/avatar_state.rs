//! Where Fritz stands and whether the owner has hidden him, kept in
//! `settings.avatar` as they change: checked every few seconds, since
//! Arrange moves the window natively and no command fires per move, and
//! saved at Quit. (Spike 0 kept these in `avatar-state.json`, which the
//! settings import once.)

use std::thread;
use std::time::Duration;

use tauri::AppHandle;

use crate::overlay_machine::OverlayRect;
use crate::settings::{self, Avatar, SavedRect};

/// How often the saver looks for a change.
const SAVE_EVERY: Duration = Duration::from_secs(5);

impl From<OverlayRect> for SavedRect {
    fn from(r: OverlayRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

impl From<SavedRect> for OverlayRect {
    fn from(r: SavedRect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

/// `avatar` with what the shell holds now. A spot only once he has stood
/// somewhere (a window that never opened keeps the last saved one), and the
/// owner's Hide only: a park by the cover watch is undone when the
/// fullscreen app goes, and must not outlive a restart.
pub fn next(avatar: &Avatar, standing: Option<OverlayRect>, owner_parked: bool) -> Avatar {
    Avatar {
        rect: standing.map(SavedRect::from).or(avatar.rect),
        parked: owner_parked,
        ..avatar.clone()
    }
}

/// Save the shell's state into the settings, if it changed.
pub fn save_now(app: &AppHandle) {
    let standing = crate::overlay_windows::standing(app);
    let owner_parked = crate::overlay_windows::owner_parked(app);
    settings::update(|s| s.avatar = next(&s.avatar, standing, owner_parked));
}

/// Where he last stood, if anywhere.
pub fn saved_rect() -> Option<OverlayRect> {
    settings::get().avatar.rect.map(OverlayRect::from)
}

/// Save whenever the state changes (checked every few seconds), so a drag in
/// Arrange or a park survives a crash as well as a Quit.
pub fn start_saver(app: AppHandle) {
    let spawned = thread::Builder::new()
        .name("fritz-state-saver".into())
        .spawn(move || loop {
            thread::sleep(SAVE_EVERY);
            save_now(&app);
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the state saver");
    }
}

#[cfg(test)]
mod tests {
    use super::next;
    use crate::overlay_machine::OverlayRect;
    use crate::settings::{Avatar, SavedRect};

    const AT: OverlayRect = OverlayRect { x: -1200, y: 40, width: 300, height: 460 };

    #[test]
    fn where_he_stands_is_kept() {
        let a = next(&Avatar::default(), Some(AT), false);
        assert_eq!(a.rect, Some(SavedRect { x: -1200, y: 40, width: 300, height: 460 }));
        assert!(!a.parked);
    }

    /// A window that isn't open says nowhere; the last spot stays.
    #[test]
    fn no_window_keeps_the_last_spot() {
        let before = next(&Avatar::default(), Some(AT), false);
        assert_eq!(next(&before, None, false).rect, before.rect);
    }

    #[test]
    fn the_owners_hide_is_kept_and_so_is_its_undoing() {
        let hidden = next(&Avatar::default(), Some(AT), true);
        assert!(hidden.parked);
        assert!(!next(&hidden, Some(AT), false).parked);
    }

    /// Only the shell's two fields change: the owner's preferences stay.
    #[test]
    fn the_rest_of_avatar_is_untouched() {
        let mine = Avatar {
            lookat_cursor: false,
            vrm_path: Some("C:\\m\\fritz.vrm".into()),
            scale: 1.5,
            ..Avatar::default()
        };
        let a = next(&mine, Some(AT), true);
        assert!(!a.lookat_cursor);
        assert_eq!(a.vrm_path, mine.vrm_path);
        assert!((a.scale - 1.5).abs() < f32::EPSILON);
    }
}
