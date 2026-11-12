//! Where the cursor is, for Fritz's eyes: polled at 15 Hz and sent to the
//! avatar's page as `cursor`, in physical pixels from his window's top-left
//! (plan 14 build § 4.2). Only a real move is sent, at least 2 px, or a
//! change in his window; and only while he can be seen looking: on screen,
//! not parked, the display on, and `avatar.lookatCursor` set.
//!
//! A pause (parked, or the display off) sends nothing and forgets nothing:
//! when he is seen again, a cursor that hasn't moved sends nothing either,
//! so his eyes are where they were, with no glance ahead and back. A stop
//! (his window gone, or the owner turning it off) sends one
//! `{ active: false }`, and the page looks ahead.
//!
//! `over` says whether the cursor is over his window's rectangle; hit rects
//! for the model itself are PR 5's.

use std::thread;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::overlay_machine::{OverlayRect, OVERLAY_LABEL};

pub const CURSOR_EVENT: &str = "cursor";

/// 15 Hz.
const EVERY: Duration = Duration::from_millis(66);
/// The smallest move worth telling the page, in physical pixels.
const MIN_MOVE: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cursor {
    /// Whether to look at it at all.
    pub active: bool,
    /// From the window's top-left, physical pixels; outside the window is
    /// negative or past its size.
    pub x: i32,
    pub y: i32,
    /// The window's size, physical pixels, so the page can scale to its own.
    pub width: u32,
    pub height: u32,
    pub over: bool,
}

impl Cursor {
    pub const IDLE: Cursor = Cursor { active: false, x: 0, y: 0, width: 0, height: 0, over: false };
}

/// Whether he is watching the cursor now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Watch {
    /// Seen, and asked to.
    Watching,
    /// Asked to, but not seen just now: parked, or the display off.
    Paused,
    /// Not asked to, or no window.
    Stopped,
}

pub fn watch(open: bool, parked: bool, display_on: bool, lookat: bool) -> Watch {
    if !open || !lookat {
        Watch::Stopped
    } else if parked || !display_on {
        Watch::Paused
    } else {
        Watch::Watching
    }
}

/// What the page is to hear for `watch`, with the cursor `located`: nothing
/// while paused (`None`), `IDLE` when stopped, the cursor while watching.
pub fn seen(watch: Watch, located: Option<Cursor>) -> Option<Cursor> {
    match watch {
        Watch::Paused => None,
        Watch::Stopped => Some(Cursor::IDLE),
        Watch::Watching => Some(located.unwrap_or(Cursor::IDLE)),
    }
}

/// The cursor at `at` (screen, physical) as the page wants it, relative to
/// his window at `rect`.
pub fn locate(at: (i32, i32), rect: OverlayRect) -> Cursor {
    let x = at.0.saturating_sub(rect.x);
    let y = at.1.saturating_sub(rect.y);
    let inside = |v: i32, size: u32| v >= 0 && i64::from(v) < i64::from(size);
    Cursor {
        active: true,
        x,
        y,
        width: rect.width,
        height: rect.height,
        over: inside(x, rect.width) && inside(y, rect.height),
    }
}

/// Whether `now` is news after `last`: a start or stop, a move of
/// `MIN_MOVE` or more either way, or a change in his window or `over`.
/// Nothing is sent before the first look, since the page starts looking
/// ahead.
pub fn worth_sending(last: Option<&Cursor>, now: &Cursor) -> bool {
    let Some(last) = last else {
        return now.active;
    };
    if last.active != now.active {
        return true;
    }
    now.active
        && ((last.x - now.x).abs() >= MIN_MOVE
            || (last.y - now.y).abs() >= MIN_MOVE
            || last.width != now.width
            || last.height != now.height
            || last.over != now.over)
}

fn located(app: &AppHandle) -> Option<Cursor> {
    match (crate::windows_ext::cursor_pos(), crate::overlay_windows::standing(app)) {
        (Some(at), Some(rect)) => Some(locate(at, rect)),
        _ => None,
    }
}

fn watch_now(app: &AppHandle) -> Watch {
    let status = crate::overlay_windows::status(app);
    let lookat = crate::settings::peek(|s| s.avatar.lookat_cursor);
    watch(status.open, status.parked, crate::power::display_on(), lookat)
}

/// What the watch would send the page now: `None` while paused.
fn sample(app: &AppHandle) -> Option<Cursor> {
    let w = watch_now(app);
    seen(w, (w == Watch::Watching).then(|| located(app)).flatten())
}

/// The cursor for a page that asks (`cursor_now`): where it is, even while
/// he is paused (the page draws nothing then, and has it ready when he is
/// seen again); `IDLE` when stopped.
pub fn now(app: &AppHandle) -> Cursor {
    match watch_now(app) {
        Watch::Stopped => Cursor::IDLE,
        Watch::Watching | Watch::Paused => located(app).unwrap_or(Cursor::IDLE),
    }
}

pub fn start(app: AppHandle) {
    let spawned = thread::Builder::new()
        .name("fritz-cursor".into())
        .spawn(move || {
            tracing::info!("cursor watch started (15 Hz; sent only while he looks)");
            let mut last: Option<Cursor> = None;
            loop {
                thread::sleep(EVERY);
                crate::logs::guard("cursor", || {
                    let Some(now) = sample(&app) else {
                        return; // paused: nothing said, nothing forgotten
                    };
                    if worth_sending(last.as_ref(), &now) {
                        let _ = app.emit_to(OVERLAY_LABEL, CURSOR_EVENT, now);
                        last = Some(now);
                    }
                });
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the cursor watch; his eyes stay ahead");
    }
}

#[cfg(test)]
mod tests {
    use super::{locate, seen, watch, worth_sending, Cursor, Watch};
    use crate::overlay_machine::OverlayRect;

    const AT: OverlayRect = OverlayRect { x: 3000, y: 1400, width: 600, height: 920 };

    #[test]
    fn only_while_he_can_be_seen_looking() {
        assert_eq!(watch(true, false, true, true), Watch::Watching);
        assert_eq!(watch(true, true, true, true), Watch::Paused, "parked");
        assert_eq!(watch(true, false, false, true), Watch::Paused, "display off");
        assert_eq!(watch(false, false, true, true), Watch::Stopped, "no window");
        assert_eq!(watch(true, false, true, false), Watch::Stopped, "the owner turned it off");
        assert_eq!(watch(true, true, false, false), Watch::Stopped, "turned off wins over a pause");
    }

    /// Hidden and shown again with the cursor where it was: the page hears
    /// nothing at all, so his eyes stay on it (no glance ahead and back).
    #[test]
    fn a_pause_says_nothing_and_forgets_nothing() {
        let a = locate((3100, 1500), AT);
        let mut last = Some(a);
        for (w, located) in [
            (Watch::Paused, None),
            (Watch::Paused, None),
            (Watch::Watching, Some(a)),
        ] {
            if let Some(now) = seen(w, located) {
                assert!(!worth_sending(last.as_ref(), &now), "{w:?} sent {now:?}");
                last = Some(now);
            }
        }
        assert_eq!(last, Some(a));
        // A stop does say so, once.
        let stopped = seen(Watch::Stopped, None).unwrap();
        assert_eq!(stopped, Cursor::IDLE);
        assert!(worth_sending(Some(&a), &stopped));
        // And watching with nowhere to measure from is a stop too.
        assert_eq!(seen(Watch::Watching, None), Some(Cursor::IDLE));
    }

    #[test]
    fn the_cursor_is_measured_from_his_window() {
        let c = locate((3100, 1500), AT);
        assert_eq!((c.x, c.y, c.width, c.height), (100, 100, 600, 920));
        assert!(c.active && c.over);
        // Elsewhere on the desktop: negative, or past his size.
        let c = locate((10, 20), AT);
        assert_eq!((c.x, c.y), (-2990, -1380));
        assert!(!c.over);
        assert!(!locate((3600, 1500), AT).over, "the right edge is outside");
        assert!(locate((3599, 2319), AT).over, "the last pixel is inside");
    }

    #[test]
    fn only_a_real_move_is_news() {
        let a = locate((3100, 1500), AT);
        assert!(worth_sending(None, &a), "the first look");
        assert!(!worth_sending(None, &Cursor::IDLE), "nothing to stop before a start");
        assert!(!worth_sending(Some(&a), &a));
        assert!(!worth_sending(Some(&a), &locate((3101, 1501), AT)), "a pixel's jitter");
        assert!(worth_sending(Some(&a), &locate((3102, 1500), AT)));
        assert!(worth_sending(Some(&a), &locate((3100, 1498), AT)));
    }

    /// Arrange moving or resizing him changes where the cursor is from him.
    #[test]
    fn his_window_moving_is_news() {
        let a = locate((3100, 1500), AT);
        let bigger = OverlayRect { width: 700, ..AT };
        assert!(worth_sending(Some(&a), &locate((3100, 1500), bigger)));
        let moved = OverlayRect { x: 2000, ..AT };
        assert!(worth_sending(Some(&a), &locate((3100, 1500), moved)));
    }

    /// One pixel across his window's edge is less than a move, but `over`
    /// changes, and the page is told.
    #[test]
    fn crossing_his_edge_is_news_however_small() {
        let inside = locate((3599, 1500), AT);
        let outside = locate((3600, 1500), AT);
        assert!(inside.over && !outside.over);
        assert!(worth_sending(Some(&inside), &outside));
        assert!(worth_sending(Some(&outside), &inside));
    }

    #[test]
    fn stopping_is_said_once() {
        let a = locate((3100, 1500), AT);
        assert!(worth_sending(Some(&a), &Cursor::IDLE));
        assert!(!worth_sending(Some(&Cursor::IDLE), &Cursor::IDLE));
        assert!(worth_sending(Some(&Cursor::IDLE), &a), "and starting again");
    }

    /// The page reads these keys; the fixture holds them for both sides.
    #[test]
    fn the_payload_is_the_wire_contract() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let v = serde_json::to_value(locate((3100, 1500), AT)).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let want: Vec<&str> =
            fixture["cursorKeys"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
        assert_eq!(keys, want);
    }
}
