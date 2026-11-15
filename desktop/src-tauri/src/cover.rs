//! Fritz steps aside for a fullscreen app and for the lock screen: once a
//! second, if the session is locked, or another app's window covers a monitor
//! he stands on (a borderless-fullscreen game, a fullscreen video), he is
//! parked; when it goes, he comes back — but only if it was this watch that
//! parked him. A Hide from the tray or hotkey is the owner's: the watch never
//! undoes it, and only the owner's is saved (`overlay_windows::owner_parked`).

use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};
use std::thread;
use std::time::Duration;

use tauri::AppHandle;

use crate::windows_ext::{self, Bounds};

const EVERY: Duration = Duration::from_secs(1);

/// What the watch last heard of the lock, for a hotkey's gate, which runs
/// inside `WM_HOTKEY` and must never wait on Windows' session service:
/// 0 unknown, 1 unlocked, 2 locked.
static LOCKED: AtomicU8 = AtomicU8::new(0);
/// The shell's notification state as the watch last heard it; -1 unknown.
static NOTIFICATION: AtomicI32 = AtomicI32::new(-1);

/// Keep what the watch heard, for [`locked`] and [`busy`].
fn remember(locked: Option<bool>, notification: Option<i32>) {
    LOCKED.store(locked.map_or(0, |l| if l { 2 } else { 1 }), Ordering::SeqCst);
    NOTIFICATION.store(notification.unwrap_or(-1), Ordering::SeqCst);
}

/// Whether the session was locked at the watch's last look (about a second
/// ago, longer just after a wake); `None` if Windows won't say, or before
/// its first look.
pub fn locked() -> Option<bool> {
    match LOCKED.load(Ordering::SeqCst) {
        2 => Some(true),
        1 => Some(false),
        _ => None,
    }
}

/// The lock for a hotkey's gate: the watch's `cached` answer, unless it
/// says locked. Then Windows is asked (`ask`), since the cache can still say
/// so just after the owner signed back in; a key can't be pressed while the
/// session really is locked, so this is rare, and never slows a press.
pub fn lock_now(cached: Option<bool>, ask: impl FnOnce() -> Option<bool>) -> Option<bool> {
    if cached == Some(true) {
        ask()
    } else {
        cached
    }
}

/// What the owner is busy with, for a hotkey's gate: the watch's `cached`
/// answer, unless it says busy. Then the shell is asked again (`ask`), since
/// a game quit or presentation mode turned off a moment ago can still be in
/// the cache; that costs a call only while it says busy.
pub fn busy_now(cached: Option<&'static str>, ask: impl FnOnce() -> Option<&'static str>) -> Option<&'static str> {
    if cached.is_some() {
        ask()
    } else {
        None
    }
}

/// What the shell said the owner was busy with at the watch's last look
/// (a Direct3D fullscreen app, presentation mode), if anything.
pub fn busy() -> Option<&'static str> {
    windows_ext::busy_state(NOTIFICATION.load(Ordering::SeqCst))
}

/// What the log names as covering him while the session is locked.
const LOCK_SCREEN: &str = "the lock screen";

/// What covers Fritz, for the log, or `None`. While the session is locked it
/// is the lock screen, whatever window `app` would find (and `app` isn't
/// asked); otherwise it is `app`'s answer, another app's fullscreen window.
pub fn covered_by(locked: Option<bool>, app: impl FnOnce() -> Option<String>) -> Option<String> {
    if locked == Some(true) {
        Some(LOCK_SCREEN.to_owned())
    } else {
        app()
    }
}

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
        .spawn(move || {
            // Whether Windows answers here at all: `None` means the lock screen
            // can only be seen as a fullscreen window, as before.
            let locked = windows_ext::session_locked();
            tracing::info!(?locked, "cover watch started");
            remember(locked, windows_ext::notification_state());
            loop {
                thread::sleep(EVERY);
                // Each tick, whatever Fritz is doing, for the prompt's gate.
                let locked = windows_ext::session_locked();
                remember(locked, windows_ext::notification_state());
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
                let covering = covered_by(locked, || windows_ext::covering_app(at));
                // Decided under the shell's park lock, against the state as it
                // is then, not as it was when this tick began.
                match crate::overlay_windows::set_covered(&app, covering.is_some()) {
                    Some(true) => {
                        let by = covering.as_deref().unwrap_or("?");
                        tracing::info!(by, "Fritz's monitor is covered: parking him");
                    }
                    Some(false) => {
                        tracing::info!("Fritz's monitor is clear again: bringing him back");
                    }
                    None => {}
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the cover watch");
    }
}

#[cfg(test)]
mod tests {
    use super::{busy, busy_now, covered_by, decide, lock_now, locked, remember};

    /// Not busy in the cache is trusted; busy is checked with the shell, and
    /// a game quit a moment ago no longer refuses the press.
    #[test]
    fn a_cached_busy_state_is_checked_with_the_shell() {
        assert_eq!(busy_now(Some("a Direct3D fullscreen app"), || None), None, "just quit");
        assert_eq!(busy_now(Some("presentation mode"), || Some("presentation mode")), Some("presentation mode"));
        assert_eq!(busy_now(None, || panic!("asked the shell")), None);
    }

    /// The cache is trusted unless it says locked; then Windows is asked,
    /// and a lock lifted a moment ago no longer refuses the press.
    #[test]
    fn a_cached_lock_is_checked_with_windows() {
        assert_eq!(lock_now(Some(true), || Some(false)), Some(false), "just signed in");
        assert_eq!(lock_now(Some(true), || Some(true)), Some(true));
        assert_eq!(lock_now(Some(true), || None), None);
        for cached in [Some(false), None] {
            assert_eq!(lock_now(cached, || panic!("asked Windows")), cached);
        }
    }

    /// The prompt's gate reads what the watch last heard, never Windows:
    /// the lock as it was, and the shell's busy states only.
    #[test]
    fn the_watch_remembers_for_the_prompts_gate() {
        remember(Some(true), Some(3));
        assert_eq!((locked(), busy()), (Some(true), Some("a Direct3D fullscreen app")));
        remember(Some(false), Some(2));
        assert_eq!((locked(), busy()), (Some(false), None));
        remember(None, None);
        assert_eq!((locked(), busy()), (None, None));
    }

    /// Locked, he is covered by the lock screen whatever is at a monitor's
    /// centre, and the window walk isn't made.
    #[test]
    fn the_lock_screen_covers_him() {
        let covering = covered_by(Some(true), || {
            panic!("the fullscreen test was asked while locked")
        });
        assert_eq!(covering.as_deref(), Some("the lock screen"));
    }

    /// Unlocked, or when Windows won't say, it is the fullscreen test alone.
    #[test]
    fn unlocked_it_is_the_fullscreen_test() {
        for locked in [Some(false), None] {
            assert_eq!(covered_by(locked, || None), None, "{locked:?}");
            assert_eq!(
                covered_by(locked, || Some("Game (pid 1)".to_owned())).as_deref(),
                Some("Game (pid 1)"),
                "{locked:?}"
            );
        }
    }

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
