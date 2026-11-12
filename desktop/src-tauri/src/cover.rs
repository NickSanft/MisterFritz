//! Fritz steps aside for a fullscreen app: once a second, if another app's
//! window covers a monitor he stands on (a borderless-fullscreen game, a
//! fullscreen video), he is parked; when it goes, he comes back — but only if
//! it was this watch that parked him. A Hide from the tray or hotkey is the
//! owner's: the watch never undoes it, and only the owner's is saved
//! (`overlay_windows::owner_parked`).

use std::thread;
use std::time::Duration;

use tauri::AppHandle;

use crate::windows_ext::{self, Bounds};

const EVERY: Duration = Duration::from_secs(1);

/// The watch's one decision: `Some(on)` to park or unpark him, `None` to
/// leave him be. `cover_parked` says the current park is the watch's own.
/// Never while he is being arranged: the window is the owner's then, and a
/// park mid-drag would be dragged straight back on screen by the move loop.
pub fn decide(covered: bool, parked: bool, cover_parked: bool, arranging: bool) -> Option<bool> {
    if arranging {
        return None;
    }
    match (covered, parked) {
        (true, false) => Some(true),
        (false, true) if cover_parked => Some(false),
        _ => None,
    }
}

pub fn start(app: AppHandle) {
    let spawned = thread::Builder::new()
        .name("fritz-cover-watch".into())
        .spawn(move || loop {
            thread::sleep(EVERY);
            let status = crate::overlay_windows::status(&app);
            if !status.open {
                continue;
            }
            let Some(r) = crate::overlay_windows::standing(&app) else {
                continue;
            };
            let at = Bounds {
                left: r.x,
                top: r.y,
                right: r.x.saturating_add(i32::try_from(r.width).unwrap_or(0)),
                bottom: r.y.saturating_add(i32::try_from(r.height).unwrap_or(0)),
            };
            let covering = windows_ext::covering_app(at);
            // Decided under the shell's park lock, against the state as it is
            // then, not as it was when this tick began.
            match crate::overlay_windows::set_covered(&app, covering.is_some()) {
                Some(true) => {
                    let by = covering.as_deref().unwrap_or("?");
                    tracing::info!(by, "a fullscreen app covers Fritz's monitor: parking him");
                }
                Some(false) => tracing::info!("Fritz's monitor is clear again: bringing him back"),
                None => {}
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the cover watch");
    }
}

#[cfg(test)]
mod tests {
    use super::decide;

    #[test]
    fn steps_aside_for_fullscreen_and_comes_back() {
        assert_eq!(decide(false, false, false, false), None);
        assert_eq!(decide(true, false, false, false), Some(true));
        assert_eq!(
            decide(true, true, true, false),
            None,
            "still covered: stays parked"
        );
        assert_eq!(decide(false, true, true, false), Some(false));
    }

    #[test]
    fn never_undoes_the_owners_hide() {
        assert_eq!(
            decide(false, true, false, false),
            None,
            "hidden from the tray"
        );
        assert_eq!(decide(true, true, false, false), None);
    }

    /// Shown by the owner while the game is still up: the watch parks him
    /// again, as the fullscreen app still covers him.
    #[test]
    fn shown_over_a_game_steps_aside_again() {
        assert_eq!(decide(true, false, false, false), Some(true));
    }

    /// While the owner is arranging him, the watch keeps its hands off,
    /// whatever covers what; it acts on the first tick after Lock.
    #[test]
    fn never_while_arranging() {
        for (covered, parked, cover_parked) in [(true, false, false), (false, true, true)] {
            assert_eq!(decide(covered, parked, cover_parked, true), None);
        }
    }
}
