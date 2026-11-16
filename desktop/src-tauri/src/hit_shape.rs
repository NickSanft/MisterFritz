//! Where Fritz's body is on his window, as his page measures it
//! (`set_hit_rects`): up to six boxes, kept as fractions of his window in
//! atomics, so the mouse hook's procedure (`hook.rs`) can ask whether a
//! point is on him without taking a lock.
//!
//! Fractions, not screen pixels: a drag, a park or a change of DPI moves
//! his window, and the boxes are mapped onto wherever it stands at the
//! moment of asking, with no new send.
//!
//! Which send is the newest is the shell's to say, with no clock compared
//! between processes. It numbers every load of his page (`page_started`),
//! the page asks its number once (`avatar_page`) and stamps each send with
//! it and a count of its own. A send is kept only if it is the live
//! window's, this load's, and later than the one kept. So a send still on
//! its way from the page before a reload is never taken for the new one's.
//!
//! A shape not heard again for `STALE` counts as gone (the page sends it
//! again every few seconds while it draws): a page that hung or crashed
//! can't leave an invisible patch of the desktop his for long.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::windows_ext::Bounds;

/// The most boxes a shape has.
pub const MAX_RECTS: usize = 6;
/// A shape not heard again for this long counts as gone.
pub const STALE: Duration = Duration::from_secs(15);
/// The most of his window a shape may cover: a page's mistake must never
/// make his whole window solid.
pub const MOST: f64 = 0.85;
/// The largest view a page can mean, in CSS pixels a side.
pub const VIEW_MAX: f64 = 16_384.0;
/// A whole side of his window, as a box's edges count it.
const FULL: u16 = u16::MAX;

/// His page's size, in CSS pixels, as it measured the boxes.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct View {
    pub width: f64,
    pub height: f64,
}

/// One box of his shape, in CSS pixels from his page's top-left.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PageRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Which send this is: his window, the load of its page, and the page's
/// own count of its sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stamp {
    pub instance: u32,
    pub page: u32,
    pub seq: u32,
}

/// What became of a send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kept {
    /// Kept, and not the shape held before. `news` when the number of boxes
    /// changed, or this is the first from this load of his page.
    Changed { boxes: usize, percent: u32, news: bool },
    /// The same shape again: only its time is new.
    Same,
    /// Not the live window's, not this load's, or no later than the one
    /// kept: dropped.
    Older,
}

/// A share of a side, as an edge is kept: near edges rounded down and far
/// edges up, so the box kept is never smaller than the page's.
fn share(v: f64, side: f64, far: bool) -> u16 {
    let s = v / side * f64::from(FULL);
    let s = if far { s.ceil() } else { s.floor() };
    // In range by the clamp before it; the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let out = s.clamp(0.0, f64::from(FULL)) as u16;
    out
}

pub(crate) fn pack_one(left: u16, top: u16, right: u16, bottom: u16) -> u64 {
    u64::from(left) | u64::from(top) << 16 | u64::from(right) << 32 | u64::from(bottom) << 48
}

/// Left, top, right, bottom.
pub(crate) fn unpack(v: u64) -> [u16; 4] {
    // Each a 16-bit field of the word.
    #[allow(clippy::cast_possible_truncation)]
    let field = |shift: u32| (v >> shift) as u16;
    [field(0), field(16), field(32), field(48)]
}

/// The page's boxes as they are kept, or why they can't be a shape: more
/// than `MAX_RECTS`; a number that isn't one; a view no window has; or
/// boxes covering more than `MOST` of the view. Each box is clamped to the
/// view, and one under a pixel either way is dropped.
pub(crate) fn pack(view: &View, rects: &[PageRect]) -> Result<Vec<u64>, String> {
    if rects.len() > MAX_RECTS {
        return Err(format!("his shape is at most {MAX_RECTS} boxes, not {}", rects.len()));
    }
    let side = |v: f64| v.is_finite() && (1.0..=VIEW_MAX).contains(&v);
    if !side(view.width) || !side(view.height) {
        return Err(format!("his page can't be {} by {} pixels", view.width, view.height));
    }
    let mut packed = Vec::with_capacity(rects.len());
    let mut area = 0.0;
    for r in rects {
        if ![r.x, r.y, r.width, r.height].iter().all(|v| v.is_finite()) {
            return Err("a box of his shape has an edge that isn't a number".to_owned());
        }
        let left = r.x.clamp(0.0, view.width);
        let top = r.y.clamp(0.0, view.height);
        let right = (r.x + r.width).clamp(0.0, view.width);
        let bottom = (r.y + r.height).clamp(0.0, view.height);
        if right - left < 1.0 || bottom - top < 1.0 {
            continue;
        }
        area += (right - left) * (bottom - top);
        packed.push(pack_one(
            share(left, view.width, false),
            share(top, view.height, false),
            share(right, view.width, true),
            share(bottom, view.height, true),
        ));
    }
    if area > MOST * view.width * view.height {
        return Err(format!("his shape can't cover more than {:.0}% of his window", MOST * 100.0));
    }
    Ok(packed)
}

/// A kept box on the screen, for a window standing at `window` (physical
/// pixels): near edges rounded down, far edges up.
pub(crate) fn on_screen(f: [u16; 4], window: Bounds) -> Bounds {
    let full = i64::from(FULL);
    let (w, h) = (i64::from(window.width().max(0)), i64::from(window.height().max(0)));
    let near = |f: u16, side: i64| i32::try_from(i64::from(f) * side / full).unwrap_or(i32::MAX);
    let far = |f: u16, side: i64| i32::try_from((i64::from(f) * side + full - 1) / full).unwrap_or(i32::MAX);
    Bounds {
        left: window.left.saturating_add(near(f[0], w)),
        top: window.top.saturating_add(near(f[1], h)),
        right: window.left.saturating_add(far(f[2], w)),
        bottom: window.top.saturating_add(far(f[3], h)),
    }
}

/// Whether `at` is in `b`: its right and bottom edges are outside, as
/// `cursor::locate` has a window's.
pub(crate) fn holds(b: Bounds, at: (i32, i32)) -> bool {
    at.0 >= b.left && at.0 < b.right && at.1 >= b.top && at.1 < b.bottom
}

/// Whether the send `next` is one to keep: the live window's, of the load
/// of its page that is the current one (`page`), and later than `kept` if
/// that is the same window's and load's.
pub(crate) fn newer(kept: Option<Stamp>, live: Option<u32>, page: u32, next: Stamp) -> bool {
    if live != Some(next.instance) || next.page != page {
        return false;
    }
    match kept {
        Some(k) if k.instance == next.instance && k.page == next.page => next.seq > k.seq,
        _ => true,
    }
}

/// The share of his window the boxes cover, in whole percent (boxes that
/// overlap count twice: it is for the log).
fn percent(packed: &[u64]) -> u32 {
    let full = f64::from(FULL);
    let area: f64 = packed
        .iter()
        .map(|&v| {
            let [l, t, r, b] = unpack(v);
            f64::from(r.saturating_sub(l)) * f64::from(b.saturating_sub(t))
        })
        .sum();
    // 0 to 600 at most.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let out = (area / (full * full) * 100.0).round() as u32;
    out
}

/// The last send kept: its stamp, and its boxes (to tell the same shape
/// again from a new one).
struct Last {
    stamp: Stamp,
    packed: Vec<u64>,
}

/// His shape, and the count of his page's loads. The boxes, their number,
/// their window and their time are atomics, read without a lock
/// (`contains`); `last` is taken only by the commands and the page's
/// events, never by a reader.
pub(crate) struct Store {
    rects: [AtomicU64; MAX_RECTS],
    count: AtomicUsize,
    instance: AtomicU32,
    heard_ms: AtomicU64,
    pages: AtomicU32,
    last: Mutex<Option<Last>>,
}

impl Store {
    pub(crate) const fn new() -> Self {
        Self {
            rects: [const { AtomicU64::new(0) }; MAX_RECTS],
            count: AtomicUsize::new(0),
            instance: AtomicU32::new(0),
            heard_ms: AtomicU64::new(0),
            pages: AtomicU32::new(0),
            last: Mutex::new(None),
        }
    }

    fn last(&self) -> MutexGuard<'_, Option<Last>> {
        self.last.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// His page starts loading, for the first time or again: the load is
    /// numbered, and the shape from the page before it forgotten. The
    /// number.
    pub(crate) fn page_started(&self) -> u32 {
        let mut last = self.last();
        self.count.store(0, Ordering::Release);
        *last = None;
        self.pages.fetch_add(1, Ordering::SeqCst).wrapping_add(1)
    }

    /// The number of the load of his page that is the current one (0
    /// before the first).
    pub(crate) fn page(&self) -> u32 {
        self.pages.load(Ordering::SeqCst)
    }

    /// No shape: his window is gone.
    pub(crate) fn forget(&self) {
        let mut last = self.last();
        self.count.store(0, Ordering::Release);
        *last = None;
    }

    /// Keep the send `next`, heard at `now_ms`, if it is one to keep
    /// (`newer`). The count is zero while the boxes are written, so a read
    /// that lands in the middle finds no shape rather than half of one.
    pub(crate) fn keep(&self, live: Option<u32>, next: Stamp, packed: &[u64], now_ms: u64) -> Kept {
        let mut last = self.last();
        if !newer(last.as_ref().map(|l| l.stamp), live, self.page(), next) {
            return Kept::Older;
        }
        let packed = &packed[..packed.len().min(MAX_RECTS)];
        let same_page = last
            .as_ref()
            .is_some_and(|l| l.stamp.instance == next.instance && l.stamp.page == next.page);
        let same = same_page && last.as_ref().is_some_and(|l| l.packed == packed);
        let news = !same_page || last.as_ref().is_none_or(|l| l.packed.len() != packed.len());
        if !same {
            self.count.store(0, Ordering::Release);
            for (slot, v) in self.rects.iter().zip(packed) {
                slot.store(*v, Ordering::Relaxed);
            }
            self.instance.store(next.instance, Ordering::Relaxed);
        }
        self.heard_ms.store(now_ms, Ordering::Relaxed);
        if !same {
            self.count.store(packed.len(), Ordering::Release);
        }
        *last = Some(Last { stamp: next, packed: packed.to_vec() });
        if same {
            Kept::Same
        } else {
            Kept::Changed { boxes: packed.len(), percent: percent(packed), news }
        }
    }

    /// Whether a shape of the live window `live`, heard within `STALE`,
    /// is held.
    pub(crate) fn held(&self, live: u32, now_ms: u64) -> bool {
        let stale = u64::try_from(STALE.as_millis()).unwrap_or(u64::MAX);
        self.count.load(Ordering::Acquire) > 0
            && self.instance.load(Ordering::Relaxed) == live
            && now_ms.saturating_sub(self.heard_ms.load(Ordering::Relaxed)) <= stale
    }

    /// Whether the screen point `at` is on him, for the live window
    /// `live` standing at `window`, at `now_ms`: inside a box of a shape
    /// that is that window's and was heard within `STALE`. Atomics only.
    pub(crate) fn contains(&self, at: (i32, i32), window: Bounds, live: u32, now_ms: u64) -> bool {
        let n = self.count.load(Ordering::Acquire).min(MAX_RECTS);
        if n == 0 || self.instance.load(Ordering::Relaxed) != live {
            return false;
        }
        let stale = u64::try_from(STALE.as_millis()).unwrap_or(u64::MAX);
        if now_ms.saturating_sub(self.heard_ms.load(Ordering::Relaxed)) > stale {
            return false;
        }
        self.rects[..n]
            .iter()
            .any(|r| holds(on_screen(unpack(r.load(Ordering::Relaxed)), window), at))
    }
}

/// His shape.
static STORE: Store = Store::new();
/// When the shell started, for `now_ms`. Set at setup (`init`), never
/// lazily: a reader may be the hook's procedure, which waits on nothing.
static START: OnceLock<Instant> = OnceLock::new();

/// At setup, before anything asks the time.
pub fn init() {
    START.get_or_init(Instant::now);
}

/// Milliseconds since `init` (0 before it): this process's own clock.
pub fn now_ms() -> u64 {
    START
        .get()
        .map_or(0, |s| u64::try_from(s.elapsed().as_millis()).unwrap_or(u64::MAX))
}

/// His page starts loading: see `Store::page_started`. Safe inside
/// WebView2's own callback: one leaf lock, held for a few stores.
pub fn page_started() -> u32 {
    STORE.page_started()
}

/// The current load of his page, for `avatar_page`.
pub fn page() -> u32 {
    STORE.page()
}

/// His window is gone.
pub fn forget() {
    STORE.forget();
}

/// Keep a send from his page: see `Store::keep`.
pub(crate) fn keep(live: Option<u32>, next: Stamp, packed: &[u64]) -> Kept {
    STORE.keep(live, next, packed, now_ms())
}

/// Whether a screen point is on him now: the mouse hook's question
/// (`hook::Live::shape_holds`).
pub fn contains(at: (i32, i32), window: Bounds, live: u32) -> bool {
    STORE.contains(at, window, live, now_ms())
}

/// Whether a fresh shape of the live window `live` is held: the pretend
/// click waits for one.
pub fn held(live: u32) -> bool {
    STORE.held(live, now_ms())
}

#[cfg(test)]
mod tests {
    use super::{
        holds, newer, on_screen, pack, pack_one, percent, unpack, Kept, PageRect, Stamp, Store, View, MAX_RECTS, MOST,
        STALE,
    };
    use crate::windows_ext::Bounds;

    const VIEW: View = View { width: 340.0, height: 640.0 };

    fn rect(x: f64, y: f64, width: f64, height: f64) -> PageRect {
        PageRect { x, y, width, height }
    }

    fn stamp(instance: u32, page: u32, seq: u32) -> Stamp {
        Stamp { instance, page, seq }
    }

    fn window(left: i32, top: i32, width: i32, height: i32) -> Bounds {
        Bounds { left, top, right: left + width, bottom: top + height }
    }

    /// What a page can't mean is refused with why; what it can is clamped
    /// to the view, and a box under a pixel dropped.
    #[test]
    fn pack_table() {
        let body = rect(120.0, 190.0, 100.0, 400.0);
        for (view, rects, says) in [
            (VIEW, vec![body; MAX_RECTS + 1], "at most 6 boxes, not 7"),
            (View { width: f64::NAN, height: 640.0 }, vec![body], "his page can't be"),
            (View { width: 340.0, height: f64::INFINITY }, vec![body], "his page can't be"),
            (View { width: 0.0, height: 640.0 }, vec![body], "his page can't be 0 by 640"),
            (View { width: 340.0, height: 20_000.0 }, vec![body], "his page can't be"),
            (VIEW, vec![rect(f64::NAN, 0.0, 10.0, 10.0)], "isn't a number"),
            (VIEW, vec![rect(0.0, 0.0, f64::INFINITY, 10.0)], "isn't a number"),
            (VIEW, vec![rect(0.0, 0.0, 340.0, 640.0 * 0.9)], "more than 85% of his window"),
            (VIEW, vec![rect(0.0, 0.0, 340.0, 300.0), rect(0.0, 300.0, 340.0, 300.0)], "more than 85%"),
        ] {
            let err = pack(&view, &rects).unwrap_err();
            assert!(err.contains(says), "{says}: {err}");
        }
        // None is a shape: nothing of his window is his.
        assert_eq!(pack(&VIEW, &[]), Ok(vec![]));
        assert_eq!(pack(&VIEW, &[rect(120.0, 190.0, 100.0, 40.0); MAX_RECTS]).unwrap().len(), MAX_RECTS);
        // Just under the most is allowed.
        assert!(pack(&VIEW, &[rect(0.0, 0.0, 340.0, 640.0 * MOST - 1.0)]).is_ok());
        // Past the right and bottom edges: clamped to them.
        let over = pack(&VIEW, &[rect(300.0, 600.0, 100.0, 100.0)]).unwrap();
        assert_eq!(unpack(over[0])[2..], [u16::MAX, u16::MAX]);
        // Before the left and top edges: clamped to them.
        let before = pack(&VIEW, &[rect(-50.0, -50.0, 100.0, 100.0)]).unwrap();
        assert_eq!(unpack(before[0])[..2], [0, 0]);
        // Wholly outside, or under a pixel either way: dropped.
        assert_eq!(pack(&VIEW, &[rect(400.0, 10.0, 50.0, 50.0), rect(10.0, 10.0, 0.5, 50.0), body]).unwrap().len(), 1);
        assert_eq!(pack(&VIEW, &[rect(10.0, 10.0, 50.0, 0.9)]), Ok(vec![]));
    }

    #[test]
    fn pack_round_trips() {
        assert_eq!(unpack(pack_one(1, 2, 3, 4)), [1, 2, 3, 4]);
        assert_eq!(unpack(pack_one(u16::MAX, 0, 0x1234, 0xfedc)), [u16::MAX, 0, 0x1234, 0xfedc]);
        // A box comes back within a pixel, and never smaller than it was.
        let page = rect(120.0, 190.0, 100.0, 400.0);
        let kept = pack(&VIEW, &[page]).unwrap();
        let back = on_screen(unpack(kept[0]), window(0, 0, 340, 640));
        assert!(back.left <= 120 && back.left >= 119, "{back:?}");
        assert!(back.top <= 190 && back.top >= 189, "{back:?}");
        assert!(back.right >= 220 && back.right <= 221, "{back:?}");
        assert!(back.bottom >= 590 && back.bottom <= 591, "{back:?}");
        assert_eq!(percent(&kept), 18, "100 × 400 of 340 × 640");
        // As kept: a near edge's share rounded down, a far edge's up (a whole
        // side is 65 535: 101 of 340 is 19 467.75, 201 of 640 is 20 582.1).
        let kept = pack(&VIEW, &[rect(101.0, 101.0, 100.0, 100.0)]).unwrap();
        assert_eq!(unpack(kept[0]), [19_467, 10_342, 38_743, 20_583]);
    }

    /// The boxes follow his window: wherever it stands, at whatever scale.
    #[test]
    fn on_screen_table() {
        let half = [0, 0, 0x8000, 0x8000];
        let all = [0, 0, u16::MAX, u16::MAX];
        for (f, w, want) in [
            (all, window(100, 200, 340, 640), window(100, 200, 340, 640)),
            (all, window(100, 200, 510, 960), window(100, 200, 510, 960)),
            (all, window(100, 200, 680, 1280), window(100, 200, 680, 1280)),
            // A monitor to the left of and above the primary one.
            (all, window(-1920, -400, 340, 640), window(-1920, -400, 340, 640)),
            // Near edges down, far edges up: 0x8000 of 341 is 170.5.
            (half, window(0, 0, 341, 641), window(0, 0, 171, 321)),
            ([0x8000, 0x8000, u16::MAX, u16::MAX], window(0, 0, 341, 641), Bounds { left: 170, top: 320, right: 341, bottom: 641 }),
            // Parked: wherever the boxes are, they are off every monitor.
            (all, window(-10_000, -10_000, 340, 640), window(-10_000, -10_000, 340, 640)),
            // No window to speak of: no box.
            (all, window(5, 5, 0, 0), window(5, 5, 0, 0)),
        ] {
            assert_eq!(on_screen(f, w), want, "{f:?} on {w:?}");
        }
    }

    #[test]
    fn holds_edges() {
        let b = window(10, 20, 30, 40);
        for (at, yes) in [
            ((10, 20), true),
            ((39, 59), true),
            ((40, 30), false),
            ((20, 60), false),
            ((9, 30), false),
            ((20, 19), false),
        ] {
            assert_eq!(holds(b, at), yes, "{at:?}");
        }
        assert!(!holds(window(5, 5, 0, 0), (5, 5)), "an empty box holds nothing");
    }

    #[test]
    fn newer_table() {
        let live = Some(1);
        for (kept, live, page, next, yes, why) in [
            (None, live, 3, stamp(1, 3, 1), true, "the first"),
            (Some(stamp(1, 3, 4)), live, 3, stamp(1, 3, 5), true, "a later send"),
            (Some(stamp(1, 3, 4)), live, 3, stamp(1, 3, 4), false, "the same send again"),
            (Some(stamp(1, 3, 4)), live, 3, stamp(1, 3, 3), false, "an earlier send"),
            (Some(stamp(1, 3, 900)), live, 4, stamp(1, 4, 1), true, "the page after a reload starts its count again"),
            (Some(stamp(1, 4, 2)), live, 4, stamp(1, 3, 901), false, "the page before the reload, late"),
            (None, live, 4, stamp(1, 3, 901), false, "the page before the reload, with nothing kept"),
            (None, live, 3, stamp(1, 4, 1), false, "a load the shell hasn't seen start"),
            (Some(stamp(1, 3, 900)), Some(2), 5, stamp(2, 5, 1), true, "a new window's first"),
            (None, Some(2), 5, stamp(1, 5, 7), false, "not the live window's"),
            (None, None, 3, stamp(1, 3, 1), false, "no live window"),
        ] {
            assert_eq!(newer(kept, live, page, next), yes, "{why}");
        }
    }

    /// A store with his page loaded once, and the boxes of one body.
    fn loaded() -> (Store, Vec<u64>) {
        let s = Store::new();
        assert_eq!(s.page(), 0);
        assert_eq!(s.page_started(), 1);
        (s, pack(&VIEW, &[rect(120.0, 190.0, 100.0, 400.0)]).unwrap())
    }

    #[test]
    fn keep_reports_changes_only() {
        let (s, body) = loaded();
        assert_eq!(s.keep(Some(1), stamp(1, 1, 1), &[], 10), Kept::Changed { boxes: 0, percent: 0, news: true });
        assert_eq!(s.keep(Some(1), stamp(1, 1, 2), &body, 20), Kept::Changed { boxes: 1, percent: 18, news: true });
        // The same again (the page's keep-alive): only its time is new.
        assert_eq!(s.keep(Some(1), stamp(1, 1, 3), &body, 30), Kept::Same);
        // A box moved: changed, but no news.
        let moved = pack(&VIEW, &[rect(124.0, 190.0, 100.0, 400.0)]).unwrap();
        assert_eq!(s.keep(Some(1), stamp(1, 1, 4), &moved, 40), Kept::Changed { boxes: 1, percent: 18, news: false });
        // No later than the one kept, another load's, or not the live window's: dropped.
        assert_eq!(s.keep(Some(1), stamp(1, 1, 4), &body, 50), Kept::Older);
        assert_eq!(s.keep(Some(1), stamp(1, 2, 9), &body, 50), Kept::Older);
        assert_eq!(s.keep(Some(2), stamp(1, 1, 9), &body, 50), Kept::Older);
        assert_eq!(s.keep(None, stamp(1, 1, 9), &body, 50), Kept::Older);
        // A dropped send changes nothing: the moved box is still the one held.
        assert!(!s.contains((122, 200), window(0, 0, 340, 640), 1, 60));
        assert!(s.contains((223, 200), window(0, 0, 340, 640), 1, 60));
    }

    #[test]
    fn contains_table() {
        let (s, body) = loaded();
        let at = window(1000, 500, 680, 1280); // his window, at twice the page's scale
        assert!(!s.contains((1340, 1000), at, 1, 0), "no shape yet");
        assert!(!s.held(1, 0), "no shape yet");
        assert_eq!(s.keep(Some(1), stamp(1, 1, 1), &body, 1_000), Kept::Changed { boxes: 1, percent: 18, news: true });
        let stale = u64::try_from(STALE.as_millis()).unwrap();
        // Held while fresh, for its own window only.
        assert!(s.held(1, 1_000 + stale));
        assert!(!s.held(1, 1_001 + stale));
        assert!(!s.held(2, 1_000));
        for (point, live, now, yes, why) in [
            ((1340, 1000), 1, 1_000, true, "on him"),
            ((1240, 880), 1, 1_000, true, "his box's top-left corner"),
            ((1440, 1000), 1, 1_000, false, "past his box's right edge"),
            ((1100, 1000), 1, 1_000, false, "beside him"),
            ((1340, 700), 1, 1_000, false, "in the headroom over him"),
            ((340, 500), 1, 1_000, false, "where his window isn't"),
            ((1340, 1000), 2, 1_000, false, "another window's shape"),
            ((1340, 1000), 1, 1_000 + stale, true, "heard just within the time"),
            ((1340, 1000), 1, 1_001 + stale, false, "not heard for too long"),
        ] {
            assert_eq!(s.contains(point, at, live, now), yes, "{why}");
        }
        // Heard again, it is his again; parked, nothing on a monitor is.
        assert_eq!(s.keep(Some(1), stamp(1, 1, 2), &body, 20_000), Kept::Same);
        assert!(s.contains((1340, 1000), at, 1, 20_500));
        assert!(!s.contains((1340, 1000), window(-10_000, -10_000, 680, 1280), 1, 20_500));
    }

    /// His page loading again, and his window going, both leave no shape;
    /// and the page before a reload can't put its own back.
    #[test]
    fn a_reload_and_a_closed_window_forget_his_shape() {
        let (s, body) = loaded();
        let at = window(0, 0, 340, 640);
        s.keep(Some(1), stamp(1, 1, 5), &body, 0);
        assert!(s.contains((170, 300), at, 1, 0));
        assert!(s.held(1, 0));
        assert_eq!(s.page_started(), 2);
        assert!(!s.held(1, 0), "forgotten as the page loads again");
        assert!(!s.contains((170, 300), at, 1, 0), "forgotten as the page starts again");
        assert_eq!(s.keep(Some(1), stamp(1, 1, 6), &body, 0), Kept::Older, "the old page, late");
        assert!(!s.contains((170, 300), at, 1, 0));
        // The new page's first is kept, though its count starts again.
        assert_eq!(s.keep(Some(1), stamp(1, 2, 1), &body, 0), Kept::Changed { boxes: 1, percent: 18, news: true });
        assert!(s.contains((170, 300), at, 1, 0));
        s.forget();
        assert!(!s.contains((170, 300), at, 1, 0), "his window is gone");
        assert_eq!(s.page(), 2, "the count of loads is kept");
        // A smaller shape after a larger one leaves none of the larger behind.
        let two = pack(&VIEW, &[rect(0.0, 0.0, 50.0, 50.0), rect(200.0, 500.0, 50.0, 50.0)]).unwrap();
        s.keep(Some(1), stamp(1, 2, 2), &two, 0);
        assert!(s.contains((225, 525), at, 1, 0));
        s.keep(Some(1), stamp(1, 2, 3), &two[..1], 0);
        assert!(!s.contains((225, 525), at, 1, 0));
        assert!(s.contains((25, 25), at, 1, 0));
    }

    /// The page and the shell agree on the numbers, and the fixture's
    /// sample is a shape the shell keeps.
    #[test]
    fn the_constants_are_the_fixtures() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let shape = &fixture["hitShape"];
        assert_eq!(shape["maxRects"], MAX_RECTS);
        assert_eq!(shape["staleMs"], u64::try_from(STALE.as_millis()).unwrap());
        assert_eq!(shape["most"], MOST);
        let sample = &fixture["hitRectsSample"];
        let view: View = serde_json::from_value(sample["view"].clone()).unwrap();
        let rects: Vec<PageRect> = serde_json::from_value(sample["rects"].clone()).unwrap();
        assert!(!rects.is_empty() && rects.len() <= MAX_RECTS);
        assert_eq!(pack(&view, &rects).unwrap().len(), rects.len());
        // A number JSON can't hold arrives as null, which is no number.
        assert!(serde_json::from_value::<View>(serde_json::json!({"width": null, "height": 640})).is_err());
    }

    /// The time is set at setup, never by whoever asks first: the hook's
    /// procedure will ask, and it waits on nothing.
    #[test]
    fn the_clock_starts_at_setup() {
        let src = include_str!("hit_shape.rs").replace("\r\n", "\n");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        let at = code.find("pub fn now_ms()").unwrap();
        let body = &code[at..at + code[at..].find("\n}\n").unwrap()];
        assert!(body.contains("START\n        .get()"), "{body}");
        assert!(!body.contains("get_or_init"), "{body}");
        let lib = include_str!("lib.rs").replace("\r\n", "\n");
        let setup = &lib[lib.find(".setup(move |app| {").unwrap()..];
        assert!(setup[..setup.find("hotkeys::register(").unwrap()].contains("hit_shape::init();"));
    }
}
