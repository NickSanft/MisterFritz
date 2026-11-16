//! The mouse hook: a press on Fritz's body is his. A `WH_MOUSE_LL` hook sees
//! every mouse button pressed on the desktop and decides, as it is pressed,
//! whether it is on him. A right-click on him opens his menu (popup.rs); a
//! double-click opens the prompt (prompt.rs), which takes the keyboard or,
//! refused, waits for a click. While it waits, a press into it is counted as
//! the owner's open there and then, before Windows activates it (the
//! foreground log's PROMPT), and a press anywhere but in it or his (a left
//! or right press on his body) closes it; one on his menu, while it is up,
//! leaves it to the menu's choice (tray.rs).
//!
//! **What it sees, and what it keeps.** Button presses and releases. Moves
//! and the wheel are only counted, never looked at. It keeps the last left
//! press on him (its time and point, for a double-click) and counts:
//! events, presses, presses on his shape, the slowest answer. The log names
//! gestures, never a point.
//!
//! **When it is in.** Only while he can be clicked (`why_off`): the switch
//! (Settings › Fritz › Clicks on him) on, his window open, drawn and
//! Locked, not hidden, the display on, the session unlocked, and nothing
//! the shell calls busy or fullscreen in front. Under a fullscreen game, on
//! the lock screen, with the display asleep or while he is arranged, there
//! is no hook at all.
//!
//! **How a press becomes his** (`look`, cheapest first, any doubt and it
//! isn't): the hook is armed; the press is real, not injected; the gate is
//! open; his window is shown and passes clicks through; the point is in his
//! window, and inside the shape his page last sent; the app in front isn't
//! in a menu, moving a window or holding the mouse; and no window above
//! his takes the click there.
//!
//! **Threads.** `fritz-mouse` owns everything: it puts the hook in and takes
//! it out, puts it back if it goes quiet while the mouse moves (Windows
//! may have dropped it), and hears what the
//! procedure decided. `fritz-mouse-hook` only installs the hook and pumps
//! its messages, at the highest priority. One owner, so an off-on race
//! can't leave two hooks.
//!
//! **The procedure's rules** (`proc:begin` to `proc:end`, scanned by a
//! test, with every function of another module it calls): no lock, no log,
//! no wait, and no allocation but the one channel send; lParam is read
//! only for `HC_ACTION`; a panic is caught, and disarms the hook until it
//! is put in again.
//!
//! The plumbing follows LorraineAtelier's `hook.rs` (the owner's own app):
//! its own thread and pump, the unhook as the thread ends, the stop by
//! `WM_QUIT`, the work handed to another thread. See docs/LIFTED.md.

use std::cell::RefCell;
use std::ffi::OsString;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use windows::Win32::Foundation::{LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{
    GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, PeekMessageW, PostThreadMessageW,
    SetWindowsHookExW, TranslateMessage, UnhookWindowsHookEx, HC_ACTION, HHOOK, LLMHF_INJECTED,
    LLMHF_LOWER_IL_INJECTED, MSG, MSLLHOOKSTRUCT, PM_NOREMOVE, WH_MOUSE_LL, WM_LBUTTONDOWN,
    WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_QUIT, WM_RBUTTONDOWN, WM_RBUTTONUP,
    WM_XBUTTONDOWN, WM_XBUTTONUP,
};

use crate::overlay_windows::{self, Mirror};
use crate::windows_ext::{self, Bounds};

/// How often the owner thread looks at whether the hook should be in, when
/// nothing pokes it sooner.
const LOOK: Duration = Duration::from_millis(500);
/// How often, while it is in, the owner thread checks it hasn't gone
/// quiet.
const WATCH_EVERY: Duration = Duration::from_secs(2);
/// Looks in a row with the cursor moved and nothing heard before the hook
/// is put back.
const QUIET_LOOKS: u32 = 2;
/// How far the cursor must have moved for a look to count as quiet.
const MOVED_PX: u32 = 8;
/// The least time between two put-backs.
const PUT_BACK_GAP: Duration = Duration::from_secs(60);
/// How often, while it is in, the counts go to the log.
const SUMMARY_EVERY: Duration = Duration::from_secs(600);
/// How long `stop` waits for the owner thread to take the hook out.
const STOP_WAIT: Duration = Duration::from_millis(500);
/// The most windows above his the procedure looks at; past them, a press
/// counts as covered (he can't tell, so it isn't his).
pub const ABOVE_CAP: usize = 512;
/// The owner thread's name.
pub const OWNER_THREAD: &str = "fritz-mouse";
/// The hook thread's name: the panic hook writes nothing for it
/// (`logs::install_panic_hook`), since its procedure may not wait.
pub const HOOK_THREAD: &str = "fritz-mouse-hook";
/// Sent to every window with the status whenever the hook goes in or out,
/// is put back or fails: Settings › Diagnostics follows it.
pub const STATUS_EVENT: &str = "hook:status";
/// Debug builds only: the hook goes in while he is hidden, undrawn or the
/// display is off, for a live check of the install, the removal and the
/// watch with nothing shown. Nothing is his then: the procedure's own gate
/// (`Live::gate_open`) always asks `why_off` with `while_hidden` false.
pub const WHILE_HIDDEN_ENV: &str = "FRITZ_HOOK_WHILE_HIDDEN";
/// Debug builds only, `right` or `double`: once his shape is held, a
/// pretend click on him (in-process values, never input) runs through the
/// decision, the channel and the owner thread.
pub const PRETEND_CLICK_ENV: &str = "FRITZ_PRETEND_CLICK";

/// Why the hook is out, in the words the log and Settings use.
pub const OFF_SWITCH: &str = "clicks on him are off in Settings";
pub const OFF_NOT_OPEN: &str = "his window isn't open";
pub const OFF_NOT_DRAWN: &str = "his window isn't drawn yet";
pub const OFF_ARRANGING: &str = "he is being arranged";
pub const OFF_HIDDEN: &str = "he's hidden";
pub const OFF_DISPLAY: &str = "the display is off";
pub const OFF_LOCKED: &str = "the session is locked";
pub const OFF_FULLSCREEN: &str = "another app is fullscreen in front";
pub const OFF_REFUSED: &str = "Windows refused it";

/// Whether the procedure may decide: set when the hook goes in, cleared
/// when the owner thread ends or the procedure panics.
pub(crate) struct Flags {
    armed: AtomicBool,
    panicked: AtomicBool,
}

impl Flags {
    pub(crate) const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
        }
    }
}

/// What the procedure counts. Atomics only.
pub(crate) struct Counts {
    events: AtomicU64,
    presses: AtomicU64,
    on_shape: AtomicU64,
    covered: AtomicU64,
    taken: AtomicU64,
    gestures: AtomicU64,
    slowest_ns: AtomicU64,
    put_back: AtomicU32,
    /// His menu was asked for and no real press has come since: a menu
    /// late to open (the main thread busy) opens only while this holds.
    asked: AtomicBool,
}

impl Counts {
    pub(crate) const fn new() -> Self {
        Self {
            events: AtomicU64::new(0),
            presses: AtomicU64::new(0),
            on_shape: AtomicU64::new(0),
            covered: AtomicU64::new(0),
            taken: AtomicU64::new(0),
            gestures: AtomicU64::new(0),
            slowest_ns: AtomicU64::new(0),
            put_back: AtomicU32::new(0),
            asked: AtomicBool::new(false),
        }
    }
}

static FLAGS: Flags = Flags::new();
static COUNTS: Counts = Counts::new();
/// The switch, as the owner thread last read it.
static CLICKS_ON: AtomicBool = AtomicBool::new(false);
/// The hook is in now.
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// Why it is out, as the owner thread last decided; `None` while it is in
/// (or before the first look). Never read by the procedure.
static WHY: Mutex<Option<&'static str>> = Mutex::new(None);
/// The owner thread's mailbox, set once at start.
static TO_OWNER: OnceLock<Sender<Msg>> = OnceLock::new();

/// What reaches the owner thread.
enum Msg {
    /// Look now: an overlay step, a display change, a setting.
    Poke,
    /// The procedure decided a gesture: acted on, or (`would`) only watched;
    /// `menu` is his menu's number when it was made (0: none up), `waiting`
    /// the open of the prompt waiting for a click (0: none), and `pretend`
    /// marks debug builds' pretend click.
    Gesture { g: Gesture, would: bool, menu: u32, waiting: u64, pretend: bool },
    /// Take the hook out and end; acked.
    Stop(mpsc::SyncSender<()>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Button {
    Left,
    Right,
    Middle,
    X,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Edge {
    Down,
    Up,
}

/// A button pressed or released, as the hook heard it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Event {
    pub button: Button,
    pub edge: Edge,
    pub at: (i32, i32),
    /// The event's tick (`MSLLHOOKSTRUCT.time`), in ms; it wraps.
    pub time: u32,
    pub injected: bool,
}

/// Windows' double-click: the time between two presses, and half the
/// rectangle the second must fall in, centred on the first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Limits {
    pub ms: u32,
    pub half_w: i32,
    pub half_h: i32,
}

/// What a press on him, or one while his menu is up or the prompt waits,
/// asks of the shell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Gesture {
    /// A right press and release on him: his menu, where it was released.
    Menu { at: (i32, i32) },
    /// Two left presses on him within Windows' time and rectangle.
    DoubleClick,
    /// A press while his menu is up or the prompt waits: on him (`his`) or
    /// elsewhere. It ends the menu (unless it is on it), and, elsewhere, the
    /// waiting prompt unless it spares it (`spares`).
    Pressed { at: (i32, i32), his: bool },
    /// A press into the prompt while it waits, counted as the owner's open
    /// already; `fg` had the keyboard then. It ends his menu.
    IntoPrompt { at: (i32, i32), fg: isize },
}

/// What a button's release goes with.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
enum Held {
    /// Its press wasn't his: the release passes.
    #[default]
    No,
    /// Its press was his.
    Yes,
    /// A right press on him: the release opens his menu.
    Menu,
}

/// What else is up that a press may end: his menu (its number; 0 when
/// none is up) and the prompt waiting for a click (its open; 0 when none).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Ctx {
    pub menu: u32,
    pub waiting: u64,
}

/// The presses and releases so far: which buttons' presses were his, and
/// the last left press on him.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Tracker {
    left: Held,
    right: Held,
    first: Option<(u32, (i32, i32))>,
}

/// Which buttons are acted on (swallowed, and their gestures done) rather
/// than only watched: the right, for his menu, and the left, for the
/// double-click.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Acts {
    pub left: bool,
    pub right: bool,
}

pub(crate) const ACTS: Acts = Acts { left: true, right: true };

/// What becomes of one event.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Applied {
    pub swallow: bool,
    pub act: Option<Gesture>,
    pub would: Option<Gesture>,
}

/// Where a press stops, in `look`'s order: the first check it fails, or
/// `Him`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Look {
    Disarmed,
    Gate,
    Window,
    Shape,
    Busy,
    Covered,
    Him,
}

/// What `look` asks of the desktop: atomics and plain reads.
pub(crate) trait Scene {
    fn armed(&self) -> bool;
    /// The switch is on, `why_off` says nothing, and no other menu of
    /// Fritz's (the tray icon's own) is open.
    fn gate_open(&self) -> bool;
    /// His window; 0 when none.
    fn avatar(&self) -> isize;
    fn click_through(&self, w: isize) -> bool;
    fn bounds(&self, w: isize) -> Option<Bounds>;
    /// The point is inside his shape, his window standing at `window`.
    fn shape_holds(&self, at: (i32, i32), window: Bounds) -> bool;
    fn foreground_busy(&self) -> bool;
    fn covered(&self, w: isize, at: (i32, i32)) -> bool;
    fn limits(&self) -> Limits;
    /// The point is on the prompt: inside its window, and no window above
    /// it takes the press there.
    fn prompt_at(&self, at: (i32, i32)) -> bool;
    /// Count a press into the waiting prompt as the owner's open.
    fn count_into_prompt(&self);
    /// Who has the keyboard.
    fn foreground(&self) -> isize;
}

/// The gate's inputs.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inputs {
    pub clicks: bool,
    pub m: Mirror,
    pub display_on: bool,
    pub locked: Option<bool>,
    pub busy: Option<&'static str>,
    pub fullscreen: bool,
    pub while_hidden: bool,
}

thread_local! {
    /// The hook thread's tracker: the procedure only runs there.
    static TRACKER: RefCell<Tracker> = const { RefCell::new(Tracker::NEW) };
}

// proc:begin — everything the procedure runs: no lock, no log, no wait, no
// allocation but the one send.

/// The hook's procedure.
unsafe extern "system" fn mouse_proc(code: i32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    let started = Instant::now();
    COUNTS.events.fetch_add(1, Ordering::Relaxed);
    #[allow(clippy::cast_possible_wrap)]
    let action = code == HC_ACTION as i32;
    let swallow = action && lp.0 != 0 && FLAGS.armed.load(Ordering::Relaxed) && {
        // SAFETY: for HC_ACTION, lParam points at this event's
        // MSLLHOOKSTRUCT, which Windows owns for the call.
        let info = unsafe { &*(lp.0 as *const MSLLHOOKSTRUCT) };
        #[allow(clippy::cast_possible_truncation)]
        let msg = wp.0 as u32;
        guarded(&FLAGS, || {
            TRACKER.with(|t| {
                // Never re-entered, but a busy tracker only lets it pass.
                t.try_borrow_mut()
                    .is_ok_and(|mut t| decide(&Live, &mut t, ctx_now, &COUNTS, msg, info, deliver))
            })
        })
    };
    note_slowest(&COUNTS, started);
    if swallow {
        LRESULT(1)
    } else {
        // SAFETY: the arguments Windows gave this procedure, passed on.
        unsafe { CallNextHookEx(None, code, wp, lp) }
    }
}

/// `f`'s answer; `false` on a panic, which disarms the hook until it is put
/// in again (the owner thread logs it). Never logs.
fn guarded(flags: &Flags, f: impl FnOnce() -> bool) -> bool {
    if let Ok(v) = panic::catch_unwind(AssertUnwindSafe(f)) {
        v
    } else {
        flags.panicked.store(true, Ordering::SeqCst);
        flags.armed.store(false, Ordering::SeqCst);
        false
    }
}

fn note_slowest(counts: &Counts, started: Instant) {
    let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    counts.slowest_ns.fetch_max(ns, Ordering::Relaxed);
}

/// His menu, if it is up, and the prompt's open, if it waits.
fn ctx_now() -> Ctx {
    Ctx {
        menu: crate::popup::his_menu(),
        waiting: crate::prompt::waiting_open(),
    }
}

/// The one send: a gesture to the owner thread.
fn deliver(m: Msg) {
    if let Some(to) = TO_OWNER.get() {
        let _ = to.send(m);
    }
}

/// One event: tracked, looked at only if it is a real left or right press,
/// counted, and its gesture delivered. Whether to swallow it.
fn decide(
    s: &impl Scene,
    t: &mut Tracker,
    cx: impl FnOnce() -> Ctx,
    counts: &Counts,
    msg: u32,
    info: &MSLLHOOKSTRUCT,
    send: impl FnOnce(Msg),
) -> bool {
    let Some(e) = event_of(msg, info.pt.x, info.pt.y, info.time, info.flags) else {
        return false;
    };
    if e.edge == Edge::Down && !e.injected {
        counts.presses.fetch_add(1, Ordering::Relaxed);
        // A press after a right-click: a menu not yet open stays shut.
        counts.asked.store(false, Ordering::SeqCst);
    }
    // Which menu is up, read after the answer is cleared: a menu opening
    // now either finds the answer gone, or is seen here, and this press
    // ends it (popup::show stores, then reads, too).
    let cx = cx();
    let mut seen = None;
    let (his, g) = t.step(
        &e,
        cx,
        || {
            let l = look(s, e.at);
            seen = Some(l);
            l == Look::Him
        },
        || s.limits(),
    );
    if matches!(seen, Some(Look::Busy | Look::Covered | Look::Him)) {
        counts.on_shape.fetch_add(1, Ordering::Relaxed);
    }
    if seen == Some(Look::Covered) {
        counts.covered.fetch_add(1, Ordering::Relaxed);
    }
    let a = applied(his, g, &e, ACTS);
    if a.swallow {
        counts.taken.fetch_add(1, Ordering::Relaxed);
    }
    if matches!(a.act, Some(Gesture::Menu { .. })) {
        counts.asked.store(true, Ordering::SeqCst);
    }
    // A press into the waiting prompt: the owner's open, counted now, before
    // it reaches the prompt and Windows activates it. It passes.
    let a = match a.act {
        Some(Gesture::Pressed { at, his: false }) if cx.waits() && s.prompt_at(at) => {
            s.count_into_prompt();
            Applied { act: Some(Gesture::IntoPrompt { at, fg: s.foreground() }), ..a }
        }
        _ => a,
    };
    if let Some(g) = a.act.or(a.would) {
        counts.gestures.fetch_add(1, Ordering::Relaxed);
        send(Msg::Gesture {
            g,
            would: a.act.is_none(),
            menu: cx.menu,
            waiting: cx.waiting,
            pretend: false,
        });
    }
    a.swallow
}

/// A button message as an event; moves and the wheel are `None`.
pub(crate) fn event_of(msg: u32, x: i32, y: i32, time: u32, flags: u32) -> Option<Event> {
    let (button, edge) = match msg {
        WM_LBUTTONDOWN => (Button::Left, Edge::Down),
        WM_LBUTTONUP => (Button::Left, Edge::Up),
        WM_RBUTTONDOWN => (Button::Right, Edge::Down),
        WM_RBUTTONUP => (Button::Right, Edge::Up),
        WM_MBUTTONDOWN => (Button::Middle, Edge::Down),
        WM_MBUTTONUP => (Button::Middle, Edge::Up),
        WM_XBUTTONDOWN => (Button::X, Edge::Down),
        WM_XBUTTONUP => (Button::X, Edge::Up),
        _ => return None,
    };
    Some(Event {
        button,
        edge,
        at: (x, y),
        time,
        injected: flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED) != 0,
    })
}

impl Ctx {
    fn menu_up(self) -> bool {
        self.menu != 0
    }

    fn waits(self) -> bool {
        self.waiting != 0
    }
}

impl Tracker {
    pub(crate) const NEW: Self = Self {
        left: Held::No,
        right: Held::No,
        first: None,
    };

    fn hold(&mut self, b: Button) -> Option<&mut Held> {
        match b {
            Button::Left => Some(&mut self.left),
            Button::Right => Some(&mut self.right),
            Button::Middle | Button::X => None,
        }
    }

    /// One event: whether it is his, and the gesture it makes. `on_him` is
    /// asked only for a real left or right press; `limits` only for a left
    /// press on him with a first press held.
    pub(crate) fn step(
        &mut self,
        e: &Event,
        cx: Ctx,
        on_him: impl FnOnce() -> bool,
        limits: impl FnOnce() -> Limits,
    ) -> (bool, Option<Gesture>) {
        if e.edge == Edge::Up {
            // A release goes with its press, wherever it lands, injected
            // or not.
            let held = self.hold(e.button).map(std::mem::take).unwrap_or_default();
            return match held {
                Held::No => (false, None),
                Held::Yes => (true, None),
                Held::Menu => (true, Some(Gesture::Menu { at: e.at })),
            };
        }
        if e.injected {
            return (false, None);
        }
        let his = matches!(e.button, Button::Left | Button::Right) && on_him();
        if !his {
            // A lost release never leaves a stale hold.
            if let Some(h) = self.hold(e.button) {
                *h = Held::No;
            }
            self.first = None;
            let g = (cx.menu_up() || cx.waits()).then_some(Gesture::Pressed { at: e.at, his: false });
            return (false, g);
        }
        if cx.menu_up() {
            if let Some(h) = self.hold(e.button) {
                *h = Held::Yes;
            }
            self.first = None;
            return (true, Some(Gesture::Pressed { at: e.at, his: true }));
        }
        if e.button == Button::Right {
            self.right = Held::Menu;
            self.first = None;
            return (true, None);
        }
        self.left = Held::Yes;
        if let Some(first) = self.first {
            if within(first, e, limits()) {
                self.first = None;
                return (true, Some(Gesture::DoubleClick));
            }
        }
        self.first = Some((e.time, e.at));
        (true, None)
    }
}

/// Whether `e` is a double-click's second press after `first`: down to down
/// within the time (the tick wraps), and within half the rectangle each way.
fn within(first: (u32, (i32, i32)), e: &Event, l: Limits) -> bool {
    let (t0, p0) = first;
    e.time.wrapping_sub(t0) <= l.ms
        && e.at.0.abs_diff(p0.0) <= l.half_w.unsigned_abs()
        && e.at.1.abs_diff(p0.1) <= l.half_h.unsigned_abs()
}

/// Swallow only a press or release that is his on a button in `acts`. A
/// menu acts with `acts.right`, a double-click with `acts.left`, and each is
/// otherwise only what it would do; a press that ends a menu or a waiting
/// prompt, or goes into it, always acts.
pub(crate) fn applied(his: bool, g: Option<Gesture>, e: &Event, acts: Acts) -> Applied {
    let on = match e.button {
        Button::Left => acts.left,
        Button::Right => acts.right,
        Button::Middle | Button::X => false,
    };
    let (act, would) = match g {
        Some(Gesture::Menu { .. }) if acts.right => (g, None),
        Some(Gesture::DoubleClick) if acts.left => (g, None),
        Some(Gesture::Pressed { .. } | Gesture::IntoPrompt { .. }) => (g, None),
        Some(_) => (None, g),
        None => (None, None),
    };
    Applied {
        swallow: his && on,
        act,
        would,
    }
}

/// The checks, in order, stopping at the first that fails.
pub(crate) fn look(s: &impl Scene, at: (i32, i32)) -> Look {
    if !s.armed() {
        return Look::Disarmed;
    }
    if !s.gate_open() {
        return Look::Gate;
    }
    let w = s.avatar();
    if w == 0 || !s.click_through(w) {
        return Look::Window;
    }
    let Some(window) = s.bounds(w) else {
        return Look::Window;
    };
    if !crate::hit_shape::holds(window, at) {
        return Look::Window;
    }
    if !s.shape_holds(at, window) {
        return Look::Shape;
    }
    if s.foreground_busy() {
        return Look::Busy;
    }
    if s.covered(w, at) {
        return Look::Covered;
    }
    Look::Him
}

/// Why the hook must be out, or `None` when he can be clicked. In order:
/// the switch; his window open, not hidden, the display on, his window
/// drawn (a window hidden from the start never draws: it says hidden) and
/// not arranged; the lock; what the shell says the owner is busy with;
/// another app fullscreen in front. `while_hidden` (debug builds) overrides
/// only what keeps him unseen: the hiding, the display and the drawing.
pub(crate) fn why_off(i: &Inputs) -> Option<&'static str> {
    if !i.clicks {
        return Some(OFF_SWITCH);
    }
    if !i.m.open {
        return Some(OFF_NOT_OPEN);
    }
    if i.m.parked && !i.while_hidden {
        return Some(OFF_HIDDEN);
    }
    if !i.display_on && !i.while_hidden {
        return Some(OFF_DISPLAY);
    }
    if !i.m.painted && !i.while_hidden {
        return Some(OFF_NOT_DRAWN);
    }
    if i.m.arranging {
        return Some(OFF_ARRANGING);
    }
    if i.locked == Some(true) {
        return Some(OFF_LOCKED);
    }
    if let Some(busy) = i.busy {
        return Some(busy);
    }
    if i.fullscreen {
        return Some(OFF_FULLSCREEN);
    }
    None
}

/// The gate's inputs now, from atomics.
fn inputs_now(clicks: bool, while_hidden: bool) -> Inputs {
    Inputs {
        clicks,
        m: overlay_windows::mirror(),
        display_on: crate::power::display_on(),
        locked: crate::cover::locked(),
        busy: crate::cover::busy(),
        fullscreen: crate::cover::fullscreen(),
        while_hidden,
    }
}

/// The desktop as it is.
struct Live;

impl Scene for Live {
    fn armed(&self) -> bool {
        FLAGS.armed.load(Ordering::Relaxed)
    }
    fn gate_open(&self) -> bool {
        // The tray icon's own menu open: presses are its, so it closes as
        // any menu does.
        let clicks = CLICKS_ON.load(Ordering::Relaxed);
        why_off(&inputs_now(clicks, false)).is_none() && !crate::popup::other_menu_up()
    }
    fn avatar(&self) -> isize {
        overlay_windows::mirror().hwnd
    }
    fn click_through(&self, w: isize) -> bool {
        windows_ext::click_through(w)
    }
    fn bounds(&self, w: isize) -> Option<Bounds> {
        windows_ext::window_bounds(w)
    }
    fn shape_holds(&self, at: (i32, i32), window: Bounds) -> bool {
        crate::hit_shape::contains(at, window, overlay_windows::mirror().instance)
    }
    fn foreground_busy(&self) -> bool {
        windows_ext::foreground_busy()
    }
    fn covered(&self, w: isize, at: (i32, i32)) -> bool {
        windows_ext::covered_at(w, at, ABOVE_CAP)
    }
    fn limits(&self) -> Limits {
        let (ms, half_w, half_h) = windows_ext::double_click_limits(self.avatar());
        Limits { ms, half_w, half_h }
    }
    fn prompt_at(&self, at: (i32, i32)) -> bool {
        let w = crate::prompt::hwnd();
        w != 0
            && windows_ext::window_bounds(w).is_some_and(|b| crate::hit_shape::holds(b, at))
            && !windows_ext::covered_at(w, at, ABOVE_CAP)
    }
    fn count_into_prompt(&self) {
        crate::prompt::pressed_into();
    }
    fn foreground(&self) -> isize {
        windows_ext::foreground_window()
    }
}

// proc:end

/// The hook thread: install, say so, pump until `WM_QUIT`, and unhook as
/// it ends. Its queue is made before its id is known, so the stop can
/// never be posted to a thread without one.
fn run_hook(ready: &Sender<Result<u32, String>>) {
    // SAFETY: this thread's own priority, queue and hook; the procedure is
    // a plain function for the whole process's life.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        let mut msg = MSG::default();
        let _ = PeekMessageW(&raw mut msg, None, 0, 0, PM_NOREMOVE);
        let module = match GetModuleHandleW(None) {
            Ok(m) => m,
            Err(e) => {
                let _ = ready.send(Err(format!("no module handle: {e}")));
                return;
            }
        };
        let hook = match SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_proc), Some(module.into()), 0) {
            Ok(h) => h,
            Err(e) => {
                let _ = ready.send(Err(format!("Windows refused the hook: {e}")));
                return;
            }
        };
        let _unhook = Unhook(hook);
        let _ = ready.send(Ok(GetCurrentThreadId()));
        while GetMessageW(&raw mut msg, None, 0, 0).0 > 0 {
            let _ = TranslateMessage(&raw const msg);
            DispatchMessageW(&raw const msg);
        }
    }
}

/// Unhooks as the hook thread ends, however it ends.
struct Unhook(HHOOK);

impl Drop for Unhook {
    fn drop(&mut self) {
        // SAFETY: the hook this thread set.
        let _ = unsafe { UnhookWindowsHookEx(self.0) };
    }
}

/// Putting the hook in and taking it out, so the owner's rules are tested
/// without a hook.
pub(crate) trait Hooks {
    type Running;
    fn install(&self) -> Result<Self::Running, String>;
    fn remove(&self, r: Self::Running);
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Change {
    Put,
    Removed,
    Failed(String),
    Same,
}

/// In when wanted, out when not, and nothing twice. A failed install is
/// tried again at the next look, never in a loop.
pub(crate) fn reconcile<H: Hooks>(want: bool, running: &mut Option<H::Running>, h: &H) -> Change {
    match (want, running.take()) {
        (true, Some(r)) => {
            *running = Some(r);
            Change::Same
        }
        (false, None) => Change::Same,
        (true, None) => match h.install() {
            Ok(r) => {
                *running = Some(r);
                Change::Put
            }
            Err(e) => Change::Failed(e),
        },
        (false, Some(r)) => {
            h.remove(r);
            Change::Removed
        }
    }
}

/// The real hook: a thread of its own.
struct LiveHooks;

struct Running {
    thread: u32,
    join: JoinHandle<()>,
}

impl Hooks for LiveHooks {
    type Running = Running;

    fn install(&self) -> Result<Running, String> {
        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name(HOOK_THREAD.into())
            .spawn(move || run_hook(&tx))
            .map_err(|e| format!("couldn't start its thread: {e}"))?;
        // It always answers, or ends (and the channel closes).
        match rx.recv() {
            Ok(Ok(thread)) => Ok(Running { thread, join }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err("its thread ended before it said".into())
            }
        }
    }

    fn remove(&self, r: Running) {
        // SAFETY: a thread id; its queue exists before the id is known.
        let posted = unsafe { PostThreadMessageW(r.thread, WM_QUIT, WPARAM(0), LPARAM(0)) };
        match posted {
            Ok(()) => {
                let _ = r.join.join();
            }
            Err(e) => {
                // Never a wait that can't end: the thread is let go.
                tracing::warn!(error = %e, "couldn't stop the mouse hook's thread");
            }
        }
    }
}

/// Whether the hook has gone quiet: Windows can drop a slow procedure
/// silently (`LowLevelHooksTimeout`), and the mouse over an app run as
/// administrator, or a cursor another app moves, looks the same. The
/// cursor moved and the procedure heard nothing, twice in a row.
#[derive(Debug, Default)]
pub(crate) struct Witness {
    at: Option<(i32, i32)>,
    heard: u64,
    quiet: u32,
    last_put: Option<Instant>,
}

impl Witness {
    /// A fresh hook: nothing seen yet (the last put-back is kept).
    fn reset(&mut self) {
        self.at = None;
        self.quiet = 0;
    }

    /// True when the hook should be put back: moved `MOVED_PX` or more with
    /// nothing heard, `QUIET_LOOKS` in a row, and no put-back within
    /// `PUT_BACK_GAP`.
    pub(crate) fn look(&mut self, at: Option<(i32, i32)>, heard: u64, now: Instant) -> bool {
        let moved = match (self.at, at) {
            (Some(a), Some(b)) => a.0.abs_diff(b.0) >= MOVED_PX || a.1.abs_diff(b.1) >= MOVED_PX,
            _ => false,
        };
        let quiet = moved && heard == self.heard;
        self.at = at;
        self.heard = heard;
        self.quiet = if quiet { self.quiet + 1 } else { 0 };
        let gap = self.last_put.is_none_or(|t| now.duration_since(t) >= PUT_BACK_GAP);
        if self.quiet >= QUIET_LOOKS && gap {
            self.quiet = 0;
            self.last_put = Some(now);
            return true;
        }
        false
    }
}

/// Clears `armed` when the owner thread ends, however it ends: the
/// procedure then passes everything.
struct Alive<'a>(&'a Flags);

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        self.0.armed.store(false, Ordering::SeqCst);
    }
}

/// A pretend click (debug builds).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pretend {
    Right,
    Double,
}

/// Whether the hook may go in while he is hidden: in a debug build, with
/// `FRITZ_HOOK_WHILE_HIDDEN` set (`debug` is `cfg!(debug_assertions)` at
/// the one caller, so a release build never reads it).
pub fn while_hidden_wanted(debug: bool, env: Option<OsString>) -> bool {
    debug && env.is_some_and(|v| !v.is_empty())
}

/// The pretend click asked for: in a debug build, `FRITZ_PRETEND_CLICK` set
/// to `right` or `double`.
pub fn pretend_click(debug: bool, env: Option<OsString>) -> Option<Pretend> {
    if !debug {
        return None;
    }
    match env?.to_str()?.trim().to_ascii_lowercase().as_str() {
        "right" => Some(Pretend::Right),
        "double" => Some(Pretend::Double),
        _ => None,
    }
}

/// A desktop where every press is on him, for the pretend click, with
/// Windows' double-click as given.
struct OnHim {
    avatar: isize,
    limits: Limits,
}

impl Scene for OnHim {
    fn armed(&self) -> bool {
        true
    }
    fn gate_open(&self) -> bool {
        true
    }
    fn avatar(&self) -> isize {
        self.avatar
    }
    fn click_through(&self, _: isize) -> bool {
        true
    }
    fn bounds(&self, _: isize) -> Option<Bounds> {
        Some(Bounds { left: -1, top: -1, right: 1, bottom: 1 })
    }
    fn shape_holds(&self, _: (i32, i32), _: Bounds) -> bool {
        true
    }
    fn foreground_busy(&self) -> bool {
        false
    }
    fn covered(&self, _: isize, _: (i32, i32)) -> bool {
        false
    }
    fn limits(&self) -> Limits {
        self.limits
    }
    fn prompt_at(&self, _: (i32, i32)) -> bool {
        false
    }
    fn count_into_prompt(&self) {}
    fn foreground(&self) -> isize {
        0
    }
}

/// The events of a pretend click: a right press and release, or two left
/// pairs, 80 ms apart, at (0, 0).
fn pretend_events(p: Pretend) -> &'static [(u32, u32)] {
    match p {
        Pretend::Right => &[(WM_RBUTTONDOWN, 0), (WM_RBUTTONUP, 80)],
        Pretend::Double => &[
            (WM_LBUTTONDOWN, 0),
            (WM_LBUTTONUP, 80),
            (WM_LBUTTONDOWN, 160),
            (WM_LBUTTONUP, 240),
        ],
    }
}

/// A pretend click through the decision, with the spot forced to "him" and
/// Windows' own double-click: its gesture reaches the owner thread over the
/// channel, as a real one does. Counted apart, so the hook's counts are
/// only real presses.
fn run_pretend(p: Pretend, avatar: isize, send: impl Fn(Msg)) {
    let (ms, half_w, half_h) = windows_ext::double_click_limits(avatar);
    run_pretend_with(p, avatar, Limits { ms, half_w, half_h }, send);
}

fn run_pretend_with(p: Pretend, avatar: isize, limits: Limits, send: impl Fn(Msg)) {
    let scene = OnHim { avatar, limits };
    let counts = Counts::new();
    let mut t = Tracker::NEW;
    for &(msg, dt) in pretend_events(p) {
        let info = MSLLHOOKSTRUCT {
            pt: POINT { x: 0, y: 0 },
            time: 1_000 + dt,
            ..Default::default()
        };
        decide(&scene, &mut t, ctx_now, &counts, msg, &info, |m| {
            send(match m {
                Msg::Gesture { g, would, menu, waiting, .. } => Msg::Gesture { g, would, menu, waiting, pretend: true },
                other => other,
            });
        });
    }
}

/// What the log says of a gesture.
fn gesture_line(g: Gesture, would: bool) -> &'static str {
    match (g, would) {
        (Gesture::Menu { .. }, true) => "a right-click on him: his menu would open (clicks on him are only watched)",
        (Gesture::DoubleClick, true) => "a double-click on him: the prompt would open (clicks on him are only watched)",
        (Gesture::Menu { .. }, false) => "a right-click on him: his menu",
        (Gesture::DoubleClick, false) => "a double-click on him: the prompt",
        (Gesture::Pressed { his: true, .. }, _) => "a press on him while his menu or the prompt was up",
        (Gesture::Pressed { his: false, .. }, _) => "a press elsewhere while his menu or the prompt was up",
        (Gesture::IntoPrompt { .. }, _) => "a press into the waiting prompt",
    }
}

/// The hook's state as the log says it: in, or out and why.
fn state_line(why: Option<&str>) -> String {
    why.map_or_else(|| "the mouse hook is in".to_owned(), |w| format!("the mouse hook is out ({w})"))
}

/// Start the owner thread, once, at setup: it puts the hook in whenever he
/// can be clicked.
pub fn start(app: AppHandle, while_hidden: bool, pretend: Option<Pretend>) {
    let (tx, rx) = mpsc::channel();
    if TO_OWNER.set(tx).is_err() {
        return;
    }
    if while_hidden {
        tracing::warn!("{WHILE_HIDDEN_ENV} is set: the mouse hook goes in while he's hidden");
    }
    if let Some(p) = pretend {
        tracing::warn!(?p, "{PRETEND_CLICK_ENV} is set: a pretend click on him once his shape is held");
    }
    let spawned = thread::Builder::new()
        .name(OWNER_THREAD.into())
        .spawn(move || owner(&app, &rx, while_hidden, pretend));
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the mouse hook's thread; clicks on him pass through");
    }
}

/// The owner thread's loop.
fn owner(app: &AppHandle, rx: &mpsc::Receiver<Msg>, while_hidden: bool, mut pretend: Option<Pretend>) {
    let _alive = Alive(&FLAGS);
    let hooks = LiveHooks;
    let mut running: Option<Running> = None;
    let mut said: Option<String> = None;
    let mut witness = Witness::default();
    let mut watched = Instant::now();
    let mut summed = Instant::now();
    let mut panic_said = false;
    let mut back_said = false;
    loop {
        match rx.recv_timeout(LOOK) {
            Ok(Msg::Stop(ack)) => {
                reconcile(false, &mut running, &hooks);
                INSTALLED.store(false, Ordering::SeqCst);
                tracing::info!("the mouse hook is out (Fritz is quitting)");
                let _ = ack.send(());
                return;
            }
            Ok(Msg::Gesture { g, would, menu, waiting, pretend }) => act(app, g, would, menu, waiting, pretend),
            Ok(Msg::Poke) | Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        let clicks = crate::settings::peek(|s| s.avatar.clicks);
        CLICKS_ON.store(clicks, Ordering::SeqCst);
        let why = why_off(&inputs_now(clicks, while_hidden));
        let line = match reconcile(why.is_none(), &mut running, &hooks) {
            Change::Put => {
                FLAGS.panicked.store(false, Ordering::SeqCst);
                FLAGS.armed.store(true, Ordering::SeqCst);
                panic_said = false;
                witness.reset();
                watched = Instant::now();
                state_line(None)
            }
            Change::Failed(e) => format!("the mouse hook couldn't go in: {e}"),
            Change::Removed => {
                // No hook to report a press elsewhere: his menu goes with
                // it, and one asked for and not yet open never opens.
                COUNTS.asked.store(false, Ordering::SeqCst);
                if let Some(up) = end_on_out(crate::popup::his_menu()) {
                    crate::popup::end(up, crate::popup::Ended::HookOut);
                }
                // Nor a click elsewhere: a prompt waiting for a click goes too.
                if let Some(open) = wait_on_out(crate::prompt::waiting_open()) {
                    crate::prompt::close_later(app, crate::prompt::Why::HookOut, open);
                }
                state_line(why)
            }
            Change::Same => state_line(why),
        };
        INSTALLED.store(running.is_some(), Ordering::SeqCst);
        // A double-click's open posted before the hook went out can land
        // after the Removed arm looked: while the hook is out, a real open
        // still waiting goes too (a pretend one waits out its time).
        if running.is_none() {
            if let Some(open) = wait_on_out(crate::prompt::waiting_click_open()) {
                crate::prompt::close_later(app, crate::prompt::Why::HookOut, open);
            }
        }
        *WHY.lock().unwrap_or_else(PoisonError::into_inner) = if running.is_some() { None } else { why.or(Some(OFF_REFUSED)) };
        if said.as_deref() != Some(line.as_str()) {
            tracing::info!("{line}");
            said = Some(line);
            let _ = app.emit(STATUS_EVENT, status());
        }
        say_panic(app, &mut panic_said);
        let now = Instant::now();
        if running.is_some() && now.duration_since(watched) >= WATCH_EVERY {
            watched = now;
            if witness.look(windows_ext::cursor_pos(), COUNTS.events.load(Ordering::Relaxed), now) {
                // Windows may have dropped it (an answer too slow), or the
                // mouse was over an app run as administrator, whose input
                // Windows never shows a hook of ours: in again either way,
                // at most once a minute. Said once; the count says the rest.
                if back_said {
                    tracing::debug!("the mouse hook heard nothing while the mouse moved: putting it in again");
                } else {
                    back_said = true;
                    tracing::warn!("the mouse hook heard nothing while the mouse moved (Windows may have dropped it, or the mouse was over an app run as administrator): putting it in again");
                }
                reconcile(false, &mut running, &hooks);
                // The old hook's thread is gone now: a panic it had in the
                // meantime is said before a fresh hook clears it.
                say_panic(app, &mut panic_said);
                match reconcile(true, &mut running, &hooks) {
                    Change::Put => {
                        // A fresh hook, armed as any is; only a put-back
                        // that went in is counted.
                        FLAGS.panicked.store(false, Ordering::SeqCst);
                        FLAGS.armed.store(true, Ordering::SeqCst);
                        panic_said = false;
                        COUNTS.put_back.fetch_add(1, Ordering::Relaxed);
                    }
                    Change::Failed(e) => {
                        // Out, and said so: the next good install is news.
                        let line = format!("the mouse hook couldn't go in: {e}");
                        tracing::warn!("{line}");
                        *WHY.lock().unwrap_or_else(PoisonError::into_inner) = Some(OFF_REFUSED);
                        said = Some(line);
                    }
                    Change::Removed | Change::Same => {}
                }
                INSTALLED.store(running.is_some(), Ordering::SeqCst);
                witness.reset();
                let _ = app.emit(STATUS_EVENT, status());
            }
        }
        if running.is_some() && now.duration_since(summed) >= SUMMARY_EVERY {
            summed = now;
            let s = status();
            tracing::info!(
                "the mouse hook: {} events, {} presses, {} on his shape, slowest answer {} µs",
                s.events,
                s.presses,
                s.on_shape,
                s.slowest_micros
            );
        }
        if let Some(p) = pretend {
            let m = overlay_windows::mirror();
            if m.open && crate::hit_shape::held(m.instance) {
                pretend = None;
                tracing::info!(
                    "pretend: {}",
                    match p {
                        Pretend::Right => "a right-click on him",
                        Pretend::Double => "a double-click on him",
                    }
                );
                run_pretend(p, m.hwnd, deliver);
            }
        }
    }
}

/// What the owner thread does with a gesture: his menu opened (or, for a
/// pretend, said), his menu ended by a press on him or off it; the prompt
/// opened (posted to the main thread), a waiting prompt closed by a press
/// elsewhere, or told who had the keyboard at a press into it; and the rest
/// logged. Gestures only, never a point.
fn act(app: &AppHandle, g: Gesture, would: bool, menu: u32, waiting: u64, pretend: bool) {
    if matches!(g, Gesture::Pressed { .. } | Gesture::IntoPrompt { .. }) {
        tracing::debug!("{}", gesture_line(g, would));
    } else {
        tracing::info!("{}", gesture_line(g, would));
    }
    let on_menu = |at| windows_ext::our_menu_at(crate::popup::main_thread(), at);
    match act_of(g, would, menu, pretend, on_menu) {
        Do::Ask(at) => {
            crate::popup::ask(at);
        }
        Do::Pretend(at) => {
            crate::popup::pretend_ask(at);
        }
        Do::End(n) => {
            crate::popup::end(n, crate::popup::Ended::Elsewhere);
        }
        Do::Nothing => {}
    }
    let in_prompt = |at| windows_ext::window_bounds(crate::prompt::hwnd()).is_some_and(|b| crate::hit_shape::holds(b, at));
    let spared = |at| spares(in_prompt(at), menu, || on_menu(at));
    match prompt_of(g, would, waiting, pretend, spared) {
        ToPrompt::Open => crate::prompt::toggle_later(app, crate::prompt::Opener::Click),
        ToPrompt::Pretend => crate::prompt::toggle_later(app, crate::prompt::Opener::PretendWait),
        ToPrompt::Close(open) => crate::prompt::close_later(app, crate::prompt::Why::Elsewhere, open),
        ToPrompt::Saw(open, fg) => crate::prompt::saw_press_later(app, open, fg),
        ToPrompt::Nothing => {}
    }
}

/// What the owner thread does to the prompt with a gesture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ToPrompt {
    /// Open it (or, open, what its gesture does to it).
    Open,
    /// Debug builds' pretend: a pretend open that waits, off screen.
    Pretend,
    /// Close the open waiting for a click: a press elsewhere.
    Close(u64),
    /// The open waiting was pressed into while this window had the
    /// keyboard.
    Saw(u64, isize),
    Nothing,
}

/// A gesture's act on the prompt: an acted double-click opens it (a
/// pretend, its pretend); while it waits (`waiting`, its open), a press it
/// isn't spared from closes it (`spared`, asked only then; see [`spares`])
/// and one into it says who had the keyboard; a press on him, or a gesture
/// only watched, does nothing to it.
pub(crate) fn prompt_of(g: Gesture, would: bool, waiting: u64, pretend: bool, spared: impl FnOnce((i32, i32)) -> bool) -> ToPrompt {
    match g {
        Gesture::DoubleClick if !would => {
            if pretend {
                ToPrompt::Pretend
            } else {
                ToPrompt::Open
            }
        }
        Gesture::Pressed { at, his: false } if waiting != 0 && !spared(at) => ToPrompt::Close(waiting),
        Gesture::IntoPrompt { fg, .. } if waiting != 0 => ToPrompt::Saw(waiting, fg),
        _ => ToPrompt::Nothing,
    }
}

/// Whether a press that isn't his spares the waiting prompt: one inside its
/// window (the procedure couldn't place it: neither counts nor closes), or
/// one on his own menu while it is up (`menu`; `on_menu` asked only then),
/// whose choice says what happens to it (Ask Fritz… takes the keyboard in).
pub(crate) fn spares(in_prompt: bool, menu: u32, on_menu: impl FnOnce() -> bool) -> bool {
    in_prompt || (menu != 0 && on_menu())
}

/// The waiting prompt to end as the hook goes out: its open, if one waits.
pub(crate) fn wait_on_out(open: u64) -> Option<u64> {
    (open != 0).then_some(open)
}

/// What the owner thread does with a gesture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Do {
    /// Show his menu at the release point.
    Ask((i32, i32)),
    /// Debug builds' pretend: say what would open, draw nothing.
    Pretend((i32, i32)),
    /// End his menu numbered so.
    End(u32),
    Nothing,
}

/// A gesture's act on his menu: an acted right-click asks for it (or, a
/// pretend, only says it); a press while it (`menu`) is up ends it unless it
/// is on the menu itself (`on_menu`, asked only for a press beside him), and
/// a press into the waiting prompt ends it; anything else, or a gesture only
/// watched, does nothing.
pub(crate) fn act_of(g: Gesture, would: bool, menu: u32, pretend: bool, on_menu: impl FnOnce((i32, i32)) -> bool) -> Do {
    match g {
        Gesture::Menu { at } if !would => {
            if pretend {
                Do::Pretend(at)
            } else {
                Do::Ask(at)
            }
        }
        Gesture::Pressed { at, his } if menu != 0 && ends_menu(his, || on_menu(at)) => Do::End(menu),
        // On the prompt, so not on his menu: elsewhere.
        Gesture::IntoPrompt { .. } if menu != 0 => Do::End(menu),
        _ => Do::Nothing,
    }
}

/// His menu to end as the hook goes out: the one up, if any.
pub(crate) fn end_on_out(up: u32) -> Option<u32> {
    (up != 0).then_some(up)
}

/// Whether his menu was asked for with no real press since, taking the
/// answer: `popup::show`'s question on the main thread.
pub fn take_asked() -> bool {
    COUNTS.asked.swap(false, Ordering::SeqCst)
}

/// Whether a press while his menu is up ends it: one on him does, and one
/// elsewhere unless it is on the menu itself (`on_menu`, asked only then).
pub(crate) fn ends_menu(his: bool, on_menu: impl FnOnce() -> bool) -> bool {
    his || !on_menu()
}

/// A panic of the procedure, said once: to the log, to crash.log and to
/// Settings (the panic hook writes nothing for the hook's thread).
fn say_panic(app: &AppHandle, said: &mut bool) {
    if FLAGS.panicked.load(Ordering::SeqCst) && !*said {
        *said = true;
        tracing::error!("the mouse hook's procedure failed; it passes every click until it is put in again");
        crate::logs::crash("the mouse hook's procedure failed; it passes every click until it is put in again");
        let _ = app.emit(STATUS_EVENT, status());
    }
}

/// Look now (an overlay step, a display change, a setting changed). Never
/// waits; from any thread.
pub fn poke() {
    if let Some(to) = TO_OWNER.get() {
        let _ = to.send(Msg::Poke);
    }
}

/// Take the hook out before Fritz quits: waits at most `STOP_WAIT`.
pub fn stop() {
    let Some(to) = TO_OWNER.get() else {
        return;
    };
    let (ack, done) = mpsc::sync_channel(1);
    if to.send(Msg::Stop(ack)).is_ok() && done.recv_timeout(STOP_WAIT).is_err() {
        tracing::warn!("the mouse hook didn't come out in time");
    }
}

/// `mouse_hook_status`, for Settings › Diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[allow(clippy::struct_excessive_bools)]
pub struct HookStatus {
    /// The switch (Settings › Fritz › Clicks on him).
    pub on: bool,
    pub installed: bool,
    /// Why it is out; null while it is in.
    pub why: Option<String>,
    pub events: u64,
    pub presses: u64,
    pub on_shape: u64,
    pub covered: u64,
    pub taken: u64,
    pub gestures: u64,
    pub slowest_micros: u64,
    pub put_back: u32,
    pub panicked: bool,
}

pub fn status() -> HookStatus {
    let installed = INSTALLED.load(Ordering::SeqCst);
    let why = *WHY.lock().unwrap_or_else(PoisonError::into_inner);
    HookStatus {
        on: CLICKS_ON.load(Ordering::SeqCst),
        installed,
        why: if installed { None } else { why.map(str::to_owned) },
        events: COUNTS.events.load(Ordering::Relaxed),
        presses: COUNTS.presses.load(Ordering::Relaxed),
        on_shape: COUNTS.on_shape.load(Ordering::Relaxed),
        covered: COUNTS.covered.load(Ordering::Relaxed),
        taken: COUNTS.taken.load(Ordering::Relaxed),
        gestures: COUNTS.gestures.load(Ordering::Relaxed),
        slowest_micros: COUNTS.slowest_ns.load(Ordering::Relaxed) / 1_000,
        put_back: COUNTS.put_back.load(Ordering::Relaxed),
        panicked: FLAGS.panicked.load(Ordering::SeqCst),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::cell::{Cell, RefCell};
    use std::time::{Duration, Instant};

    use super::*;

    const L: Limits = Limits { ms: 500, half_w: 2, half_h: 3 };
    const FREE: Ctx = Ctx { menu: 0, waiting: 0 };

    fn ev(button: Button, edge: Edge, at: (i32, i32), time: u32) -> Event {
        Event { button, edge, at, time, injected: false }
    }
    fn down(button: Button, at: (i32, i32), time: u32) -> Event {
        ev(button, Edge::Down, at, time)
    }
    fn up(button: Button, at: (i32, i32), time: u32) -> Event {
        ev(button, Edge::Up, at, time)
    }

    /// Steps `events` with `on_him` deciding by the point, and the given
    /// context: (his, gesture) for each.
    fn run(t: &mut Tracker, cx: Ctx, on_him: impl Fn((i32, i32)) -> bool, events: &[Event]) -> Vec<(bool, Option<Gesture>)> {
        events.iter().map(|e| t.step(e, cx, || on_him(e.at), || L)).collect()
    }
    fn him(at: (i32, i32)) -> bool {
        at.0 < 100
    }
    const ON: (i32, i32) = (10, 10);
    const OFF: (i32, i32) = (500, 10);

    #[test]
    fn event_of_table() {
        for (msg, button, edge) in [
            (WM_LBUTTONDOWN, Button::Left, Edge::Down),
            (WM_LBUTTONUP, Button::Left, Edge::Up),
            (WM_RBUTTONDOWN, Button::Right, Edge::Down),
            (WM_RBUTTONUP, Button::Right, Edge::Up),
            (WM_MBUTTONDOWN, Button::Middle, Edge::Down),
            (WM_MBUTTONUP, Button::Middle, Edge::Up),
            (WM_XBUTTONDOWN, Button::X, Edge::Down),
            (WM_XBUTTONUP, Button::X, Edge::Up),
        ] {
            let e = event_of(msg, 3, -4, 99, 0).unwrap();
            assert_eq!(e, Event { button, edge, at: (3, -4), time: 99, injected: false }, "{msg:#x}");
        }
        // Moves, the wheel and the horizontal wheel aren't buttons.
        for msg in [0x200, 0x20A, 0x20E, 0x203, 0] {
            assert_eq!(event_of(msg, 0, 0, 0, 0), None, "{msg:#x}");
        }
        for flags in [LLMHF_INJECTED, LLMHF_LOWER_IL_INJECTED, LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED] {
            assert!(event_of(WM_LBUTTONDOWN, 0, 0, 0, flags).unwrap().injected, "{flags}");
        }
        assert!(!event_of(WM_LBUTTONDOWN, 0, 0, 0, 0x10).unwrap().injected);
    }

    #[test]
    fn tracker_right_table() {
        let menu = |at| Some(Gesture::Menu { at });
        // Pressed and released on him: his, then his menu where it was released.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Right, ON, 0), up(Button::Right, (12, 14), 50)]), [(true, None), (true, menu((12, 14)))]);
        // Released elsewhere: still his, the menu at the release.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Right, ON, 0), up(Button::Right, OFF, 50)]), [(true, None), (true, menu(OFF))]);
        // Pressed elsewhere: both pass, wherever it is released.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Right, OFF, 0), up(Button::Right, ON, 50)]), [(false, None), (false, None)]);
        // The menu opens on the release, never the press, and only once.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Right, ON, 0), up(Button::Right, ON, 9), up(Button::Right, ON, 20)]), [(true, None), (true, menu(ON)), (false, None)]);
    }

    #[test]
    fn tracker_left_table() {
        let pairs = |times: &[u32], at2: (i32, i32)| -> Vec<Event> {
            vec![down(Button::Left, ON, times[0]), up(Button::Left, ON, times[0] + 5), down(Button::Left, at2, times[1]), up(Button::Left, at2, times[1] + 5)]
        };
        let dbl = Some(Gesture::DoubleClick);
        // One click: his, his, and nothing.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Left, ON, 0), up(Button::Left, ON, 5)]), [(true, None), (true, None)]);
        // A double-click: on the second press, all four his.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &pairs(&[0, 200], ON)), [(true, None), (true, None), (true, dbl), (true, None)]);
        // Exactly the time: still one. One ms over: none.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &pairs(&[0, 500], ON))[2], (true, dbl));
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &pairs(&[0, 501], ON))[2], (true, None));
        // Exactly half the rectangle each way: yes; one more: no.
        for (at, want) in [
            ((ON.0 + 2, ON.1), dbl),
            ((ON.0 - 2, ON.1 + 3), dbl),
            ((ON.0, ON.1 - 3), dbl),
            ((ON.0 + 3, ON.1), None),
            ((ON.0, ON.1 + 4), None),
            ((ON.0 - 3, ON.1 - 3), None),
        ] {
            let mut t = Tracker::NEW;
            assert_eq!(run(&mut t, FREE, him, &pairs(&[0, 100], at))[2], (true, want), "{at:?}");
        }
        // The tick wrapping: a 301 ms gap.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Left, ON, u32::MAX - 100), down(Button::Left, ON, 200)])[1], (true, dbl));
        // ... and a long one that only looks short the wrong way round.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Left, ON, 200), down(Button::Left, ON, u32::MAX - 100)])[1], (true, None));
        // A triple click: one double-click; the third press starts afresh.
        let mut t = Tracker::NEW;
        let three = [down(Button::Left, ON, 0), down(Button::Left, ON, 100), down(Button::Left, ON, 200), down(Button::Left, ON, 300)];
        let g: Vec<_> = run(&mut t, FREE, him, &three).into_iter().map(|(_, g)| g).collect();
        assert_eq!(g, [None, dbl, None, dbl]);
        // A right press between, or a press elsewhere, or the middle button: reset.
        for between in [down(Button::Right, ON, 50), down(Button::Left, OFF, 50), down(Button::Middle, ON, 50), down(Button::X, ON, 50)] {
            let mut t = Tracker::NEW;
            let out = run(&mut t, FREE, him, &[down(Button::Left, ON, 0), between, down(Button::Left, ON, 100)]);
            assert_eq!(out[2], (true, None), "{between:?}");
        }
    }

    #[test]
    fn tracker_never_held_table() {
        // The middle and X buttons on him pass, and are never asked about.
        for b in [Button::Middle, Button::X] {
            let mut t = Tracker::NEW;
            let out: Vec<_> = [down(b, ON, 0), up(b, ON, 5)].iter().map(|e| t.step(e, FREE, || panic!("asked"), || L)).collect();
            assert_eq!(out, [(false, None), (false, None)], "{b:?}");
        }
        // An injected press passes and changes nothing: a real double-click
        // around it still counts, and it is never looked at.
        let mut t = Tracker::NEW;
        let fake = Event { injected: true, ..down(Button::Left, ON, 50) };
        assert_eq!(t.step(&down(Button::Left, ON, 0), FREE, || true, || L), (true, None));
        assert_eq!(t.step(&fake, FREE, || panic!("asked"), || panic!("asked")), (false, None));
        assert_eq!(t.step(&down(Button::Left, ON, 100), FREE, || true, || L), (true, Some(Gesture::DoubleClick)));
        // Nor does it let go of a real press it lands during: that press's
        // release is still his (and a right one still opens his menu).
        for (b, last) in [(Button::Left, None), (Button::Right, Some(Gesture::Menu { at: ON }))] {
            let mut t = Tracker::NEW;
            let fake = Event { injected: true, ..down(b, ON, 50) };
            assert_eq!(t.step(&down(b, ON, 0), FREE, || true, || L), (true, None), "{b:?}");
            assert_eq!(t.step(&fake, FREE, || panic!("asked"), || panic!("asked")), (false, None), "{b:?}");
            assert_eq!(t.step(&up(b, ON, 100), FREE, || panic!("asked"), || L), (true, last), "{b:?}");
        }
        // An injected right press on him opens nothing.
        let mut t = Tracker::NEW;
        let fake = Event { injected: true, ..down(Button::Right, ON, 0) };
        assert_eq!(t.step(&fake, FREE, || panic!("asked"), || L), (false, None));
        assert_eq!(t.step(&up(Button::Right, ON, 5), FREE, || panic!("asked"), || L), (false, None));
        // An injected release still goes with a real press on him.
        let mut t = Tracker::NEW;
        t.step(&down(Button::Right, ON, 0), FREE, || true, || L);
        let fake_up = Event { injected: true, ..up(Button::Right, ON, 5) };
        assert_eq!(t.step(&fake_up, FREE, || panic!("asked"), || L), (true, Some(Gesture::Menu { at: ON })));
        // A lost release, then a press elsewhere of that button: its release passes.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Left, ON, 0), down(Button::Left, OFF, 900), up(Button::Left, OFF, 905)]), [(true, None), (false, None), (false, None)]);
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Right, ON, 0), down(Button::Right, OFF, 900), up(Button::Right, OFF, 905)]), [(true, None), (false, None), (false, None)]);
    }

    #[test]
    fn tracker_with_a_menu_or_a_waiting_prompt() {
        let menu = Ctx { menu: 3, waiting: 0 };
        let waiting = Ctx { menu: 0, waiting: 1 };
        // A press on him with a menu up: his, it ends the menu, and its
        // release opens no other.
        let mut t = Tracker::NEW;
        assert_eq!(
            run(&mut t, menu, him, &[down(Button::Right, ON, 0), up(Button::Right, ON, 5)]),
            [(true, Some(Gesture::Pressed { at: ON, his: true })), (true, None)]
        );
        // ... and resets the double-click.
        let mut t = Tracker::NEW;
        assert_eq!(t.step(&down(Button::Left, ON, 0), FREE, || true, || L), (true, None));
        assert_eq!(t.step(&down(Button::Left, ON, 50), menu, || true, || L), (true, Some(Gesture::Pressed { at: ON, his: true })));
        assert_eq!(t.step(&down(Button::Left, ON, 100), FREE, || true, || L), (true, None));
        // Elsewhere, with a menu up or the prompt waiting: not his, and it says so.
        for cx in [menu, waiting] {
            let mut t = Tracker::NEW;
            assert_eq!(run(&mut t, cx, him, &[down(Button::Left, OFF, 0)]), [(false, Some(Gesture::Pressed { at: OFF, his: false }))], "{cx:?}");
            // The middle button elsewhere too.
            assert_eq!(run(&mut t, cx, him, &[down(Button::Middle, ON, 0)]), [(false, Some(Gesture::Pressed { at: ON, his: false }))], "{cx:?}");
        }
        // Nothing up: a press elsewhere says nothing.
        let mut t = Tracker::NEW;
        assert_eq!(run(&mut t, FREE, him, &[down(Button::Left, OFF, 0)]), [(false, None)]);
        // Waiting: a double-click on him is a double-click alone.
        let mut t = Tracker::NEW;
        let out = run(&mut t, waiting, him, &[down(Button::Left, ON, 0), up(Button::Left, ON, 5), down(Button::Left, ON, 100)]);
        assert_eq!(out, [(true, None), (true, None), (true, Some(Gesture::DoubleClick))]);
    }

    #[test]
    fn the_spot_and_limits_are_asked_only_when_needed() {
        let (spot, lim) = (Cell::new(0), Cell::new(0));
        let mut t = Tracker::NEW;
        let mut step = |e: Event| {
            t.step(&e, FREE, || { spot.set(spot.get() + 1); him(e.at) }, || { lim.set(lim.get() + 1); L })
        };
        step(up(Button::Left, ON, 0)); // a release: neither
        step(down(Button::Middle, ON, 0)); // the middle button: neither
        step(Event { injected: true, ..down(Button::Left, ON, 0) }); // injected: neither
        assert_eq!((spot.get(), lim.get()), (0, 0));
        step(down(Button::Left, ON, 0)); // the first press: the spot only
        assert_eq!((spot.get(), lim.get()), (1, 0));
        step(down(Button::Left, ON, 100)); // the second: both
        assert_eq!((spot.get(), lim.get()), (2, 1));
        step(down(Button::Right, ON, 150)); // a right press: the spot only
        step(down(Button::Left, OFF, 200)); // a press elsewhere: the spot only
        assert_eq!((spot.get(), lim.get()), (4, 1));
    }

    #[test]
    fn applied_table() {
        let none = Acts { left: false, right: false };
        let right = Acts { left: false, right: true };
        let left = Acts { left: true, right: false };
        let menu = Gesture::Menu { at: ON };
        let pressed = Gesture::Pressed { at: ON, his: true };
        let r = up(Button::Right, ON, 0);
        let l = down(Button::Left, ON, 0);
        let a = |swallow, act, would| Applied { swallow, act, would };
        // Nothing acted on: nothing swallowed, the gestures only watched.
        assert_eq!(applied(true, Some(menu), &r, none), a(false, None, Some(menu)));
        assert_eq!(applied(true, Some(Gesture::DoubleClick), &l, none), a(false, None, Some(Gesture::DoubleClick)));
        assert_eq!(ACTS, Acts { left: true, right: true }, "both buttons act: his menu and the double-click");
        // The right button acted on.
        assert_eq!(applied(true, Some(menu), &r, right), a(true, Some(menu), None));
        assert_eq!(applied(true, Some(Gesture::DoubleClick), &l, right), a(false, None, Some(Gesture::DoubleClick)));
        assert_eq!(applied(true, None, &l, right), a(false, None, None));
        // The left, the mirror.
        assert_eq!(applied(true, Some(menu), &r, left), a(false, None, Some(menu)));
        assert_eq!(applied(true, Some(Gesture::DoubleClick), &l, left), a(true, Some(Gesture::DoubleClick), None));
        // Never one that isn't his, nor the middle button.
        assert_eq!(applied(false, None, &l, left), a(false, None, None));
        let both = Acts { left: true, right: true };
        assert_eq!(applied(true, None, &down(Button::Middle, ON, 0), both), a(false, None, None));
        // A press ending a menu always acts.
        assert_eq!(applied(true, Some(pressed), &l, none), a(false, Some(pressed), None));
        assert_eq!(applied(true, Some(pressed), &l, left), a(true, Some(pressed), None));
        // So does one into the waiting prompt, never his.
        let into = Gesture::IntoPrompt { at: ON, fg: 9 };
        assert_eq!(applied(false, Some(into), &l, none), a(false, Some(into), None));
        assert_eq!(applied(false, Some(into), &l, left), a(false, Some(into), None));
    }

    /// A desktop whose answers are set, which logs what it is asked.
    struct FakeScene {
        armed: bool,
        gate: bool,
        avatar: isize,
        through: bool,
        bounds: Option<Bounds>,
        shape: bool,
        busy: bool,
        covered: bool,
        /// The point is on the prompt.
        prompt: bool,
        /// Presses counted into the waiting prompt.
        into: Cell<u32>,
        fg: isize,
        asked: RefCell<Vec<&'static str>>,
    }

    fn scene() -> FakeScene {
        FakeScene {
            armed: true,
            gate: true,
            avatar: 7,
            through: true,
            bounds: Some(Bounds { left: 0, top: 0, right: 100, bottom: 200 }),
            shape: true,
            busy: false,
            covered: false,
            prompt: false,
            into: Cell::new(0),
            fg: 42,
            asked: RefCell::default(),
        }
    }

    impl Scene for FakeScene {
        fn armed(&self) -> bool {
            self.asked.borrow_mut().push("armed");
            self.armed
        }
        fn gate_open(&self) -> bool {
            self.asked.borrow_mut().push("gate");
            self.gate
        }
        fn avatar(&self) -> isize {
            self.asked.borrow_mut().push("avatar");
            self.avatar
        }
        fn click_through(&self, w: isize) -> bool {
            assert_eq!(w, self.avatar);
            self.asked.borrow_mut().push("through");
            self.through
        }
        fn bounds(&self, _: isize) -> Option<Bounds> {
            self.asked.borrow_mut().push("bounds");
            self.bounds
        }
        fn shape_holds(&self, _: (i32, i32), window: Bounds) -> bool {
            assert_eq!(Some(window), self.bounds);
            self.asked.borrow_mut().push("shape");
            self.shape
        }
        fn foreground_busy(&self) -> bool {
            self.asked.borrow_mut().push("busy");
            self.busy
        }
        fn covered(&self, _: isize, _: (i32, i32)) -> bool {
            self.asked.borrow_mut().push("covered");
            self.covered
        }
        fn limits(&self) -> Limits {
            L
        }
        fn prompt_at(&self, _: (i32, i32)) -> bool {
            self.asked.borrow_mut().push("prompt");
            self.prompt
        }
        fn count_into_prompt(&self) {
            self.into.set(self.into.get() + 1);
        }
        fn foreground(&self) -> isize {
            self.fg
        }
    }

    #[test]
    fn look_table() {
        let all = ["armed", "gate", "avatar", "through", "bounds", "shape", "busy", "covered"];
        type Case = (&'static str, fn(&mut FakeScene), Look, usize);
        let cases: [Case; 9] = [
            ("disarmed", |s| s.armed = false, Look::Disarmed, 1),
            ("gate shut", |s| s.gate = false, Look::Gate, 2),
            ("no window", |s| s.avatar = 0, Look::Window, 3),
            ("not click-through", |s| s.through = false, Look::Window, 4),
            ("no rectangle", |s| s.bounds = None, Look::Window, 5),
            ("parked off every monitor", |s| s.bounds = Some(Bounds { left: -10_000, top: 0, right: -9_660, bottom: 640 }), Look::Window, 5),
            ("beside his shape", |s| s.shape = false, Look::Shape, 6),
            ("the app in front is busy", |s| s.busy = true, Look::Busy, 7),
            ("a window above", |s| s.covered = true, Look::Covered, 8),
        ];
        for (name, set, want, calls) in cases {
            let mut s = scene();
            set(&mut s);
            assert_eq!(look(&s, (50, 50)), want, "{name}");
            assert_eq!(*s.asked.borrow(), all[..calls], "{name}: no later call is made");
        }
        let s = scene();
        assert_eq!(look(&s, (50, 50)), Look::Him);
        assert_eq!(*s.asked.borrow(), all);
        // Outside his window, by its edges: right and bottom are outside.
        for at in [(100, 50), (50, 200), (-1, 50)] {
            assert_eq!(look(&scene(), at), Look::Window, "{at:?}");
        }
    }

    fn open() -> Inputs {
        Inputs {
            clicks: true,
            m: Mirror { open: true, painted: true, arranging: false, parked: false, instance: 1, hwnd: 7 },
            display_on: true,
            locked: Some(false),
            busy: None,
            fullscreen: false,
            while_hidden: false,
        }
    }

    #[test]
    fn why_off_table() {
        assert_eq!(why_off(&open()), None);
        assert_eq!(why_off(&Inputs { locked: None, ..open() }), None, "an unknown lock isn't a lock");
        type Off = (fn(&mut Inputs), &'static str);
        let off: [Off; 9] = [
            (|i| i.clicks = false, OFF_SWITCH),
            (|i| i.m.open = false, OFF_NOT_OPEN),
            (|i| i.m.parked = true, OFF_HIDDEN),
            (|i| i.display_on = false, OFF_DISPLAY),
            (|i| i.m.painted = false, OFF_NOT_DRAWN),
            (|i| i.m.arranging = true, OFF_ARRANGING),
            (|i| i.locked = Some(true), OFF_LOCKED),
            (|i| i.busy = Some("a Direct3D fullscreen app"), "a Direct3D fullscreen app"),
            (|i| i.fullscreen = true, OFF_FULLSCREEN),
        ];
        // Each alone, and each before every one after it.
        for (n, (set, want)) in off.iter().enumerate() {
            let mut i = open();
            set(&mut i);
            assert_eq!(why_off(&i), Some(*want));
            let mut all = open();
            for (later, _) in &off[n..] {
                later(&mut all);
            }
            assert_eq!(why_off(&all), Some(*want), "{want} comes first");
        }
        // Hidden from the start, he is never drawn: hidden is the reason.
        let mut start = open();
        start.m.parked = true;
        start.m.painted = false;
        assert_eq!(why_off(&start), Some(OFF_HIDDEN));
        // While hidden (debug builds): only what keeps him unseen.
        let hidden = |set: fn(&mut Inputs)| {
            let mut i = Inputs { while_hidden: true, ..open() };
            set(&mut i);
            why_off(&i)
        };
        assert_eq!(hidden(|i| i.m.parked = true), None);
        assert_eq!(hidden(|i| i.display_on = false), None);
        assert_eq!(hidden(|i| {
            i.m.parked = true;
            i.m.painted = false;
        }), None, "hidden from the start, never drawn");
        assert_eq!(hidden(|i| i.m.open = false), Some(OFF_NOT_OPEN));
        assert_eq!(hidden(|i| i.m.painted = false), None, "undrawn while shown");
        assert_eq!(hidden(|i| i.busy = Some("x")), Some("x"), "busy is never overridden");
        assert_eq!(hidden(|i| i.m.arranging = true), Some(OFF_ARRANGING), "Arrange is always off");
        assert_eq!(hidden(|i| i.clicks = false), Some(OFF_SWITCH));
        assert_eq!(hidden(|i| i.locked = Some(true)), Some(OFF_LOCKED));
        assert_eq!(hidden(|i| i.fullscreen = true), Some(OFF_FULLSCREEN));
    }

    /// Hooks that log what they are asked, failing the installs `fail` names.
    struct FakeHooks {
        log: RefCell<Vec<&'static str>>,
        fail: Cell<u32>,
    }

    impl Hooks for FakeHooks {
        type Running = u32;
        fn install(&self) -> Result<u32, String> {
            self.log.borrow_mut().push("install");
            if self.fail.get() > 0 {
                self.fail.set(self.fail.get() - 1);
                return Err("refused".into());
            }
            Ok(1)
        }
        fn remove(&self, _: u32) {
            self.log.borrow_mut().push("remove");
        }
    }

    #[test]
    fn reconcile_table() {
        let h = FakeHooks { log: RefCell::default(), fail: Cell::new(0) };
        let mut r = None;
        assert_eq!(reconcile(false, &mut r, &h), Change::Same);
        assert_eq!(reconcile(true, &mut r, &h), Change::Put);
        assert_eq!(reconcile(true, &mut r, &h), Change::Same, "never twice");
        assert_eq!(reconcile(false, &mut r, &h), Change::Removed);
        assert_eq!(reconcile(false, &mut r, &h), Change::Same);
        assert_eq!(*h.log.borrow(), ["install", "remove"]);
        // Twenty alternating wants: twenty calls, never two installs in a row.
        h.log.borrow_mut().clear();
        for n in 0..20 {
            reconcile(n % 2 == 0, &mut r, &h);
        }
        let log = h.log.borrow();
        assert_eq!(log.len(), 20);
        assert!(log.windows(2).all(|w| w[0] != w[1]), "{log:?}");
        drop(log);
        // A failed install: out, and tried again at the next look.
        let h = FakeHooks { log: RefCell::default(), fail: Cell::new(1) };
        let mut r = None;
        assert_eq!(reconcile(true, &mut r, &h), Change::Failed("refused".into()));
        assert!(r.is_none());
        assert_eq!(reconcile(true, &mut r, &h), Change::Put);
        assert_eq!(*h.log.borrow(), ["install", "install"]);
    }

    #[test]
    fn witness_table() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        // Still: never.
        let mut w = Witness::default();
        for s in 0..10 {
            assert!(!w.look(Some((0, 0)), 5, at(s)));
        }
        // Moved and heard: never.
        let mut w = Witness::default();
        for s in 0..10 {
            assert!(!w.look(Some((i32::try_from(s).unwrap() * 50, 0)), s, at(s)));
        }
        // Moved less than 8 px with nothing heard: never.
        let mut w = Witness::default();
        for s in 0..10 {
            assert!(!w.look(Some((i32::try_from(s).unwrap() * 7, 0)), 0, at(s)));
        }
        // Quiet while moving: one look isn't enough, two are.
        let mut w = Witness::default();
        assert!(!w.look(Some((0, 0)), 3, at(0)));
        assert!(!w.look(Some((0, 8)), 3, at(2)));
        assert!(w.look(Some((0, 16)), 3, at(4)));
        // Again within a minute: no, however quiet.
        for s in 1..=10 {
            assert!(!w.look(Some((0, 16 + i32::try_from(s).unwrap() * 10)), 3, at(4 + s * 2)));
        }
        // After it: yes.
        let mut w2 = Witness { last_put: Some(at(4)), ..Witness::default() };
        assert!(!w2.look(Some((0, 0)), 3, at(60)));
        assert!(!w2.look(Some((0, 10)), 3, at(62)));
        assert!(w2.look(Some((0, 20)), 3, at(64)));
        // A heard event between breaks the run.
        let mut w = Witness::default();
        w.look(Some((0, 0)), 1, at(0));
        assert!(!w.look(Some((9, 0)), 1, at(2)));
        assert!(!w.look(Some((18, 0)), 2, at(4)));
        assert!(!w.look(Some((27, 0)), 2, at(6)));
        assert!(w.look(Some((36, 0)), 2, at(8)));
        // No cursor: never.
        let mut w = Witness::default();
        for s in 0..5 {
            assert!(!w.look(None, 0, at(s)));
        }
    }

    #[test]
    fn a_panic_in_the_decision_passes_the_event_and_disarms() {
        let f = Flags::new();
        f.armed.store(true, Ordering::SeqCst);
        assert!(guarded(&f, || true));
        assert!(!guarded(&f, || false));
        assert!(f.armed.load(Ordering::SeqCst) && !f.panicked.load(Ordering::SeqCst));
        assert!(!guarded(&f, || panic!("a test's panic")));
        assert!(f.panicked.load(Ordering::SeqCst));
        assert!(!f.armed.load(Ordering::SeqCst));
    }

    #[test]
    fn alive_disarms_the_hook_when_dropped() {
        let f = Flags::new();
        f.armed.store(true, Ordering::SeqCst);
        drop(Alive(&f));
        assert!(!f.armed.load(Ordering::SeqCst));
        // However it ends: a panic too.
        f.armed.store(true, Ordering::SeqCst);
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _alive = Alive(&f);
            panic!("a test's panic");
        }));
        assert!(!f.armed.load(Ordering::SeqCst));
    }

    /// A right-click on him is his: its press and release swallowed, his
    /// menu asked for at the release. With his menu up, a press says which
    /// menu it was made under.
    #[test]
    fn a_right_click_on_him_is_his_and_opens_his_menu() {
        let info = |x: i32, time: u32| MSLLHOOKSTRUCT { pt: POINT { x, y: 50 }, time, ..Default::default() };
        let s = scene();
        let c = Counts::new();
        let mut t = Tracker::NEW;
        let sent = RefCell::new(Vec::new());
        let send = |m: Msg| {
            if let Msg::Gesture { g, would, menu, .. } = m {
                sent.borrow_mut().push((g, would, menu));
            }
        };
        assert!(decide(&s, &mut t, || FREE, &c, WM_RBUTTONDOWN, &info(50, 0), send), "the press swallowed");
        assert!(decide(&s, &mut t, || FREE, &c, WM_RBUTTONUP, &info(60, 5), send), "the release swallowed");
        assert_eq!(*sent.borrow(), [(Gesture::Menu { at: (60, 50) }, false, 0)]);
        assert_eq!(c.taken.load(Ordering::Relaxed), 2);
        assert!(c.asked.load(Ordering::SeqCst), "his menu asked for");
        // A left press on him is his too; as any real press, it keeps a menu
        // not yet open from opening.
        assert!(decide(&s, &mut t, || FREE, &c, WM_LBUTTONDOWN, &info(50, 100), send));
        assert!(!c.asked.load(Ordering::SeqCst));
        // An injected press doesn't.
        c.asked.store(true, Ordering::SeqCst);
        let fake = MSLLHOOKSTRUCT { flags: LLMHF_INJECTED, ..info(500, 200) };
        decide(&s, &mut t, || FREE, &c, WM_LBUTTONDOWN, &fake, send);
        assert!(c.asked.load(Ordering::SeqCst));
        // His menu (number 4) up: a press beside him says so, and passes.
        sent.borrow_mut().clear();
        let up = Ctx { menu: 4, waiting: 0 };
        let beside = FakeScene { shape: false, ..scene() };
        assert!(!decide(&beside, &mut t, || up, &c, WM_LBUTTONDOWN, &info(50, 900), send));
        assert_eq!(*sent.borrow(), [(Gesture::Pressed { at: (50, 50), his: false }, false, 4)]);
        assert!(!c.asked.load(Ordering::SeqCst), "a press elsewhere keeps a late menu shut");
        // So does any other real button.
        c.asked.store(true, Ordering::SeqCst);
        decide(&beside, &mut t, || up, &c, WM_MBUTTONDOWN, &info(50, 950), send);
        assert!(!c.asked.load(Ordering::SeqCst), "a middle press too");
    }

    /// `popup::show` takes the "asked" answer: a menu opens once per
    /// right-click, never again from an answer already taken. (The live
    /// cell: no other test touches it.)
    #[test]
    fn the_asked_answer_is_taken_once() {
        COUNTS.asked.store(true, Ordering::SeqCst);
        assert!(take_asked());
        assert!(!take_asked(), "taken");
        assert!(!COUNTS.asked.load(Ordering::SeqCst));
    }

    /// What the owner thread does with each gesture, and only that.
    #[test]
    fn act_of_table() {
        const AT: (i32, i32) = (12, 34);
        let menu = Gesture::Menu { at: AT };
        let on = Gesture::Pressed { at: AT, his: true };
        let beside = Gesture::Pressed { at: AT, his: false };
        let never = |_: (i32, i32)| -> bool { panic!("the menu isn't asked about") };
        assert_eq!(act_of(menu, false, 0, false, never), Do::Ask(AT));
        assert_eq!(act_of(menu, false, 0, true, never), Do::Pretend(AT), "a pretend never draws a menu");
        assert_eq!(act_of(menu, true, 0, false, never), Do::Nothing, "only watched");
        assert_eq!(act_of(on, false, 4, false, never), Do::End(4), "a press on him");
        assert_eq!(act_of(beside, false, 4, false, |at| { assert_eq!(at, AT); false }), Do::End(4), "elsewhere");
        assert_eq!(act_of(beside, false, 4, false, |_| true), Do::Nothing, "on the menu itself");
        assert_eq!(act_of(on, false, 0, false, never), Do::Nothing, "no menu up");
        assert_eq!(act_of(Gesture::DoubleClick, true, 4, false, never), Do::Nothing);
        assert_eq!(act_of(Gesture::DoubleClick, false, 4, false, never), Do::Nothing, "a double-click leaves his menu be");
        let into = Gesture::IntoPrompt { at: AT, fg: 9 };
        assert_eq!(act_of(into, false, 4, false, never), Do::End(4), "a press into the prompt is off his menu");
        assert_eq!(act_of(into, false, 0, false, never), Do::Nothing);
        assert_eq!(end_on_out(0), None);
        assert_eq!(end_on_out(9), Some(9));
        // act carries each out, and the procedure asks which menu is up.
        let src = source("hook");
        let act = body(&src, "fn act(");
        for arm in [
            "let on_menu = |at| windows_ext::our_menu_at(crate::popup::main_thread(), at);",
            "match act_of(g, would, menu, pretend, on_menu) {",
            "Do::Ask(at) => {\n            crate::popup::ask(at);",
            "Do::Pretend(at) => {\n            crate::popup::pretend_ask(at);",
            "Do::End(n) => {\n            crate::popup::end(n, crate::popup::Ended::Elsewhere);",
        ] {
            assert!(act.contains(arm), "{arm}");
        }
        assert!(body(&src, "fn ctx_now(").contains("menu: crate::popup::his_menu(),"));
    }

    #[test]
    fn a_press_ends_his_menu_unless_it_is_on_the_menu() {
        assert!(ends_menu(true, || panic!("a press on him isn't asked about the menu")));
        assert!(ends_menu(false, || false), "elsewhere");
        assert!(!ends_menu(false, || true), "on the menu itself");
    }

    /// The owner thread asks for his menu; it never shows one itself.
    #[test]
    fn the_menu_is_asked_for_never_shown_from_the_owner_thread() {
        let src = source("hook");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(code.contains("crate::popup::ask(at)") && code.contains("crate::popup::pretend_ask(at)"));
        for never in [concat!("TrackPopup", "Menu"), concat!("track_", "menu("), concat!("EndM", "enu(")] {
            assert!(!code.contains(never), "hook.rs uses {never}");
        }
        // His menu ends as the hook goes out, and one not yet open never opens.
        let owner = body(&src, "fn owner(");
        assert!(
            owner.contains("Ok(Msg::Gesture { g, would, menu, waiting, pretend }) => act(app, g, would, menu, waiting, pretend),"),
            "acted under the menu and the wait the press was made under"
        );
        // The procedure reads which menu is up after clearing the answer.
        let decide_ = body(&src, "fn decide(");
        assert!(decide_.find("counts.asked.store(false, Ordering::SeqCst);").unwrap() < decide_.find("let cx = cx();").unwrap());
        let removed = &owner[owner.find("Change::Removed => {").unwrap()..];
        let removed = &removed[..removed.find("Change::Same").unwrap()];
        assert!(removed.contains("COUNTS.asked.store(false, Ordering::SeqCst);"));
        assert!(removed.contains("if let Some(up) = end_on_out(crate::popup::his_menu()) {\n                    crate::popup::end(up, crate::popup::Ended::HookOut);"));
    }

    #[test]
    fn decide_counts_and_delivers() {
        let info = |msg_x: i32, time: u32, flags: u32| MSLLHOOKSTRUCT { pt: POINT { x: msg_x, y: 50 }, time, flags, ..Default::default() };
        let s = scene();
        let c = Counts::new();
        let mut t = Tracker::NEW;
        let sent = RefCell::new(Vec::new());
        let send = |m: Msg| {
            if let Msg::Gesture { g, would, menu, waiting, pretend } = m {
                assert!(!pretend && menu == 0 && waiting == 0);
                sent.borrow_mut().push((g, would));
            }
        };
        // A move: nothing, and never looked at.
        assert!(!decide(&s, &mut t, || FREE, &c, 0x200, &info(50, 0, 0), send));
        assert!(s.asked.borrow().is_empty());
        // A double-click on him: two presses, two on his shape, one gesture,
        // acted on, every press and release his.
        for (msg, time) in [(WM_LBUTTONDOWN, 0), (WM_LBUTTONUP, 5), (WM_LBUTTONDOWN, 100), (WM_LBUTTONUP, 105)] {
            assert!(decide(&s, &mut t, || FREE, &c, msg, &info(50, time, 0), send));
        }
        assert_eq!(*sent.borrow(), [(Gesture::DoubleClick, false)]);
        let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
        assert_eq!((n(&c.presses), n(&c.on_shape), n(&c.covered), n(&c.taken), n(&c.gestures)), (2, 2, 0, 4, 1));
        // A press covered by a window above: on his shape, covered, not his.
        let s = FakeScene { covered: true, ..scene() };
        decide(&s, &mut t, || FREE, &c, WM_RBUTTONDOWN, &info(50, 300, 0), send);
        decide(&s, &mut t, || FREE, &c, WM_RBUTTONUP, &info(50, 305, 0), send);
        assert_eq!((n(&c.presses), n(&c.on_shape), n(&c.covered)), (3, 3, 1));
        assert_eq!(sent.borrow().len(), 1, "no menu for a covered press");
        // An injected press is no press.
        decide(&scene(), &mut t, || FREE, &c, WM_LBUTTONDOWN, &info(50, 400, LLMHF_INJECTED), send);
        assert_eq!(n(&c.presses), 3);
        // Beside his shape: a press, not on it.
        let s = FakeScene { shape: false, ..scene() };
        decide(&s, &mut t, || FREE, &c, WM_LBUTTONDOWN, &info(50, 500, 0), send);
        assert_eq!((n(&c.presses), n(&c.on_shape)), (4, 3));
    }

    #[test]
    fn a_pretend_click_runs_through_the_decision() {
        // Some window (with none, nothing is his, pretend or not), and a
        // double-click time of its own, whatever this desktop's is.
        let pretend = |p, ms| {
            let sent = RefCell::new(Vec::new());
            run_pretend_with(p, 1, Limits { ms, half_w: 1, half_h: 1 }, |m| {
                if let Msg::Gesture { g, would, pretend, .. } = m {
                    assert!(pretend, "marked as a pretend");
                    sent.borrow_mut().push((g, would));
                }
            });
            sent.into_inner()
        };
        assert_eq!(pretend(Pretend::Right, 500), [(Gesture::Menu { at: (0, 0) }, false)], "his menu acts");
        assert_eq!(pretend(Pretend::Double, 500), [(Gesture::DoubleClick, false)], "the double-click acts");
        // Its two presses are 160 ms apart, down to down.
        assert_eq!(pretend(Pretend::Double, 160), [(Gesture::DoubleClick, false)]);
        assert_eq!(pretend(Pretend::Double, 159), []);
    }

    /// The guards are held where they guard: the owner thread holds `Alive`
    /// for its whole life, the procedure decides only inside `guarded`, and
    /// the hook thread holds `Unhook` until its pump ends.
    #[test]
    fn the_guards_are_held_where_they_guard() {
        let src = source("hook");
        let owner = body(&src, "fn owner(");
        let alive = owner.find("let _alive = Alive(&FLAGS);").expect("the owner thread holds Alive");
        assert!(alive < owner.find("loop {").unwrap(), "held before the loop");
        let proc_ = body(&src, "fn mouse_proc(");
        let inside = body(proc_, "guarded(&FLAGS, || {");
        assert!(inside.contains("decide(") && proc_.matches("decide(").count() == 1, "decide only inside guarded");
        let run = body(&src, "fn run_hook(");
        let unhook = run.find("let _unhook = Unhook(hook);").expect("the hook thread holds Unhook");
        assert!(unhook < run.find("GetMessageW(").unwrap(), "held through the pump");
        // A put-back says a panic of the old hook before a fresh one clears
        // it, and counts itself only once it went in.
        let back = &owner[owner.find("if witness.look(").unwrap()..];
        let removed = back.find("reconcile(false, &mut running, &hooks);").unwrap();
        let said = back.find("say_panic(app, &mut panic_said);").unwrap();
        let put = back.find("match reconcile(true, &mut running, &hooks)").unwrap();
        assert!(removed < said && said < put, "the old hook's panic said before the new one goes in");
        let counted = back.find("COUNTS.put_back.fetch_add(1").unwrap();
        assert!(counted > back.find("Change::Put =>").unwrap() && counted < back.find("Change::Failed(e) =>").unwrap(), "counted only when in");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        assert_eq!(code.matches("COUNTS.put_back.fetch_add(").count(), 1);
        // And the procedure's own gate never overrides the hiding, and lets
        // presses pass while the tray icon's own menu is open.
        let live = body(&src, "impl Scene for Live");
        assert!(live.contains("why_off(&inputs_now(clicks, false))"), "the procedure's gate never takes while_hidden");
        assert!(live.contains("&& !crate::popup::other_menu_up()"), "the tray icon's menu takes its own presses");
    }

    #[test]
    fn the_pretend_click_and_the_hidden_hook_are_read_only_in_a_debug_build() {
        let lib = include_str!("lib.rs");
        for call in ["hook::while_hidden_wanted(", "hook::pretend_click("] {
            let at: Vec<_> = lib.match_indices(call).collect();
            assert_eq!(at.len(), 1, "{call} is read at one place");
            assert!(lib[at[0].0..].starts_with(&format!("{call}cfg!(debug_assertions), std::env::var_os(")), "{call}");
        }
        assert!(!while_hidden_wanted(false, Some("1".into())), "never in a release build");
        assert!(while_hidden_wanted(true, Some("1".into())));
        assert!(!while_hidden_wanted(true, Some("".into())));
        assert!(!while_hidden_wanted(true, None));
        assert_eq!(pretend_click(false, Some("right".into())), None, "never in a release build");
        assert_eq!(pretend_click(true, Some("right".into())), Some(Pretend::Right));
        assert_eq!(pretend_click(true, Some(" Double ".into())), Some(Pretend::Double));
        assert_eq!(pretend_click(true, Some("left".into())), None);
        assert_eq!(pretend_click(true, None), None);
    }

    #[test]
    fn the_log_names_gestures_never_points() {
        for g in [
            Gesture::Menu { at: (1234, 567) },
            Gesture::DoubleClick,
            Gesture::Pressed { at: (1234, 567), his: true },
            Gesture::Pressed { at: (1234, 567), his: false },
            Gesture::IntoPrompt { at: (1234, 567), fg: 0x1234 },
        ] {
            for would in [true, false] {
                let line = gesture_line(g, would);
                assert!(!line.contains("1234") && !line.contains("567"), "{line}");
            }
        }
        assert_eq!(gesture_line(Gesture::Menu { at: (0, 0) }, true), "a right-click on him: his menu would open (clicks on him are only watched)");
        assert_eq!(gesture_line(Gesture::DoubleClick, true), "a double-click on him: the prompt would open (clicks on him are only watched)");
        assert_eq!(gesture_line(Gesture::DoubleClick, false), "a double-click on him: the prompt");
        assert_eq!(gesture_line(Gesture::IntoPrompt { at: (0, 0), fg: 0 }, false), "a press into the waiting prompt");
        assert_eq!(state_line(None), "the mouse hook is in");
        assert_eq!(state_line(Some(OFF_HIDDEN)), "the mouse hook is out (he's hidden)");
    }

    /// While the prompt waits, a real press into it is counted there, before
    /// it reaches the prompt, and passes, reported with who had the keyboard;
    /// one elsewhere passes and says so; one on him is his and touches
    /// neither. With nothing waiting, the prompt isn't asked about.
    #[test]
    fn a_press_into_the_waiting_prompt_is_counted_there_and_passes() {
        let info = |x: i32, time: u32, flags: u32| MSLLHOOKSTRUCT { pt: POINT { x, y: 50 }, time, flags, ..Default::default() };
        let waits = Ctx { menu: 0, waiting: 5 };
        let sent = RefCell::new(Vec::new());
        let send = |m: Msg| {
            if let Msg::Gesture { g, would, waiting, .. } = m {
                sent.borrow_mut().push((g, would, waiting));
            }
        };
        let c = Counts::new();
        let mut t = Tracker::NEW;
        // On the prompt (and not on him).
        let s = FakeScene { shape: false, prompt: true, ..scene() };
        assert!(!decide(&s, &mut t, || waits, &c, WM_LBUTTONDOWN, &info(50, 0, 0), send), "it passes");
        assert_eq!(s.into.get(), 1, "counted in the procedure");
        assert_eq!(*sent.borrow(), [(Gesture::IntoPrompt { at: (50, 50), fg: 42 }, false, 5)]);
        assert!(!decide(&s, &mut t, || waits, &c, WM_LBUTTONUP, &info(50, 5, 0), send), "its release too");
        // The middle button too: any press activates it.
        decide(&s, &mut t, || waits, &c, WM_MBUTTONDOWN, &info(50, 10, 0), send);
        assert_eq!(s.into.get(), 2);
        // An injected press counts for nothing.
        decide(&s, &mut t, || waits, &c, WM_LBUTTONDOWN, &info(50, 20, LLMHF_INJECTED), send);
        assert_eq!(s.into.get(), 2);
        // Elsewhere: not counted, and it says so.
        sent.borrow_mut().clear();
        let s = FakeScene { shape: false, ..scene() };
        assert!(!decide(&s, &mut t, || waits, &c, WM_LBUTTONDOWN, &info(50, 100, 0), send));
        assert_eq!(s.into.get(), 0);
        assert_eq!(*sent.borrow(), [(Gesture::Pressed { at: (50, 50), his: false }, false, 5)]);
        // On him: his, and nothing for the prompt (a first press of a
        // double-click, say), and the prompt never asked about.
        sent.borrow_mut().clear();
        let s = FakeScene { prompt: true, ..scene() };
        assert!(decide(&s, &mut t, || waits, &c, WM_LBUTTONDOWN, &info(50, 200, 0), send));
        assert_eq!(s.into.get(), 0);
        assert!(sent.borrow().is_empty());
        assert!(!s.asked.borrow().contains(&"prompt"));
        // Nothing waiting: never asked, never counted.
        let s = FakeScene { shape: false, prompt: true, ..scene() };
        decide(&s, &mut t, || FREE, &c, WM_LBUTTONDOWN, &info(50, 300, 0), send);
        assert_eq!(s.into.get(), 0);
        assert!(!s.asked.borrow().contains(&"prompt"));
        // Only his menu up: a press where the prompt stands is no press into a
        // waiting prompt; never counted, never asked.
        sent.borrow_mut().clear();
        let menu_only = Ctx { menu: 4, waiting: 0 };
        let s = FakeScene { shape: false, prompt: true, ..scene() };
        decide(&s, &mut t, || menu_only, &c, WM_LBUTTONDOWN, &info(50, 350, 0), send);
        assert_eq!(s.into.get(), 0);
        assert!(!s.asked.borrow().contains(&"prompt"));
        assert_eq!(*sent.borrow(), [(Gesture::Pressed { at: (50, 50), his: false }, false, 0)]);
        // His menu up too: a press into the prompt is counted, and ends it.
        sent.borrow_mut().clear();
        let both = Ctx { menu: 4, waiting: 5 };
        let s = FakeScene { shape: false, prompt: true, ..scene() };
        decide(&s, &mut t, || both, &c, WM_LBUTTONDOWN, &info(50, 400, 0), send);
        assert_eq!(s.into.get(), 1);
        assert_eq!(*sent.borrow(), [(Gesture::IntoPrompt { at: (50, 50), fg: 42 }, false, 5)]);
    }

    /// What the owner thread does to the prompt with each gesture.
    #[test]
    fn prompt_of_table() {
        const AT: (i32, i32) = (12, 34);
        let into = Gesture::IntoPrompt { at: AT, fg: 9 };
        let elsewhere = Gesture::Pressed { at: AT, his: false };
        let on = Gesture::Pressed { at: AT, his: true };
        let never = |_: (i32, i32)| -> bool { panic!("the prompt's window isn't asked about") };
        let outside = |at: (i32, i32)| {
            assert_eq!(at, AT);
            false
        };
        assert_eq!(prompt_of(Gesture::DoubleClick, false, 0, false, never), ToPrompt::Open);
        assert_eq!(prompt_of(Gesture::DoubleClick, false, 3, false, never), ToPrompt::Open, "open, its gesture closes it");
        assert_eq!(prompt_of(Gesture::DoubleClick, false, 0, true, never), ToPrompt::Pretend, "a pretend opens a pretend");
        assert_eq!(prompt_of(Gesture::DoubleClick, true, 0, false, never), ToPrompt::Nothing, "only watched");
        assert_eq!(prompt_of(elsewhere, false, 3, false, outside), ToPrompt::Close(3));
        assert_eq!(prompt_of(elsewhere, false, 3, false, |_| true), ToPrompt::Nothing, "spared: inside its window, or on his menu");
        // What spares it.
        let never_menu = || -> bool { panic!("the menu isn't asked about") };
        assert!(spares(true, 0, never_menu), "inside the prompt's window");
        assert!(spares(true, 4, never_menu), "inside the prompt's window, his menu up");
        assert!(spares(false, 4, || true), "on his menu, up");
        assert!(!spares(false, 4, || false), "beside his menu");
        assert!(!spares(false, 0, never_menu), "no menu up");
        assert_eq!(prompt_of(elsewhere, false, 0, false, never), ToPrompt::Nothing, "nothing waiting (his menu's press)");
        assert_eq!(prompt_of(on, false, 3, false, never), ToPrompt::Nothing, "a press on him never closes it");
        assert_eq!(prompt_of(into, false, 3, false, never), ToPrompt::Saw(3, 9));
        assert_eq!(prompt_of(into, false, 0, false, never), ToPrompt::Nothing);
        assert_eq!(prompt_of(Gesture::Menu { at: AT }, false, 3, false, never), ToPrompt::Nothing);
        assert_eq!(wait_on_out(0), None);
        assert_eq!(wait_on_out(7), Some(7));
    }

    /// The prompt is opened, closed and told from the main thread, posted;
    /// the hook going out ends a waiting prompt; the procedure reads which
    /// open waits, and counts a press into it through the prompt's own word.
    #[test]
    fn the_prompt_is_reached_from_the_main_thread() {
        let src = source("hook");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(!code.contains(concat!("prompt::", "toggle(")), "never toggled on the owner thread");
        let act = body(&src, "fn act(");
        for arm in [
            "let in_prompt = |at| windows_ext::window_bounds(crate::prompt::hwnd()).is_some_and(|b| crate::hit_shape::holds(b, at));",
            "let spared = |at| spares(in_prompt(at), menu, || on_menu(at));",
            "match prompt_of(g, would, waiting, pretend, spared) {",
            "ToPrompt::Open => crate::prompt::toggle_later(app, crate::prompt::Opener::Click),",
            "ToPrompt::Pretend => crate::prompt::toggle_later(app, crate::prompt::Opener::PretendWait),",
            "ToPrompt::Close(open) => crate::prompt::close_later(app, crate::prompt::Why::Elsewhere, open),",
            "ToPrompt::Saw(open, fg) => crate::prompt::saw_press_later(app, open, fg),",
        ] {
            assert!(act.contains(arm), "{arm}");
        }
        let owner = body(&src, "fn owner(");
        let removed = &owner[owner.find("Change::Removed => {").unwrap()..];
        let removed = &removed[..removed.find("Change::Same").unwrap()];
        assert!(removed.contains("if let Some(open) = wait_on_out(crate::prompt::waiting_open()) {\n                    crate::prompt::close_later(app, crate::prompt::Why::HookOut, open);"));
        // And while it is out, a real open still waiting goes (one posted
        // before it went out can land after the Removed arm looked).
        assert!(owner.contains("if running.is_none() {\n            if let Some(open) = wait_on_out(crate::prompt::waiting_click_open()) {\n                crate::prompt::close_later(app, crate::prompt::Why::HookOut, open);"));
        assert!(body(&src, "fn ctx_now(").contains("waiting: crate::prompt::waiting_open(),"));
        let live = body(&src, "impl Scene for Live");
        let at = body(live, "fn prompt_at(");
        for part in ["crate::prompt::hwnd()", "w != 0", "windows_ext::window_bounds(w).is_some_and(|b| crate::hit_shape::holds(b, at))", "!windows_ext::covered_at(w, at, ABOVE_CAP)"] {
            assert!(at.contains(part), "{part}");
        }
        assert!(body(live, "fn count_into_prompt(").contains("crate::prompt::pressed_into();"));
        assert!(body(live, "fn foreground(").contains("windows_ext::foreground_window()"));
        // The procedure counts the press before it reports it.
        let decide_ = body(&src, "fn decide(");
        let counted = decide_.find("s.count_into_prompt();").unwrap();
        assert!(counted < decide_.find("send(Msg::Gesture {").unwrap());
        assert!(decide_.contains("Some(Gesture::Pressed { at, his: false }) if cx.waits() && s.prompt_at(at) => {"));
    }

    // The source scans.

    /// The source of `name`'s module.
    fn source(name: &str) -> String {
        crate::keyboard::tests::SOURCES
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no module {name}"))
            .1
            .replace("\r\n", "\n")
    }

    /// The body of the item starting at `sig` in `src`, to its matching brace.
    fn body<'a>(src: &'a str, sig: &str) -> &'a str {
        let at = src.find(sig).unwrap_or_else(|| panic!("no {sig}"));
        let open = at + src[at..].find('{').unwrap();
        let mut depth = 0;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &src[at..=open + i];
                    }
                }
                _ => {}
            }
        }
        panic!("{sig} never closes")
    }

    /// What the procedure, and everything it runs, may never contain. The
    /// needles are built from halves, so this test doesn't find itself.
    const NEVER: [&str; 22] = [
        concat!(".lo", "ck("),
        concat!("trac", "ing::"),
        concat!("lo", "gs::"),
        concat!("for", "mat!"),
        concat!("Str", "ing"),
        concat!("to_", "owned"),
        concat!("to_", "string"),
        concat!("Ve", "c"),
        concat!("Box", "::new"),
        concat!("run_on_", "main_thread"),
        concat!("emi", "t"),
        concat!("sle", "ep"),
        concat!("rec", "v"),
        concat!(".jo", "in("),
        concat!("sett", "ings::"),
        concat!("pee", "k("),
        concat!("SendMes", "sage"),
        concat!(".unw", "rap()"),
        concat!(".exp", "ect("),
        concat!("get_or", "_init"),
        concat!("Mut", "ex"),
        concat!("Has", "hMap"),
    ];

    fn assert_clean(what: &str, text: &str) {
        for needle in NEVER {
            assert!(!text.contains(needle), "{what} contains {needle}");
        }
    }

    #[test]
    fn the_procedure_takes_no_lock_and_never_waits() {
        let src = source("hook");
        let begin = src.find("// proc:begin").unwrap();
        let end = src.find("// proc:end").unwrap();
        let region = &src[begin..end];
        // Comments aside: only the code is held to it.
        let code: String = region
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_clean("the procedure's region", &code);
        assert_eq!(code.matches(".send(").count(), 1, "one send");
        for f in ["fn mouse_proc(", "fn guarded(", "fn decide(", "fn event_of(", "fn step(", "fn applied(", "fn look(", "fn why_off(", "fn inputs_now(", "impl Scene for Live", "fn within(", "fn deliver(", "fn note_slowest(", "fn ctx_now(", "fn menu_up(", "fn waits("] {
            assert!(code.contains(f), "{f} lies between the markers");
        }
    }

    /// Every function of another module that the procedure calls, by name:
    /// each held to the same rules.
    #[test]
    fn the_procedures_helpers_never_wait() {
        for (module, sig) in [
            ("overlay_windows", "pub(crate) fn mirror("),
            ("overlay_windows", "pub(crate) fn unpack_mirror("),
            ("power", "pub fn display_on("),
            ("cover", "pub fn locked("),
            ("cover", "pub fn busy("),
            ("cover", "pub fn fullscreen("),
            ("popup", "pub fn his_menu("),
            ("popup", "pub fn other_menu_up("),
            ("popup", "pub(crate) fn other_up("),
            ("popup", "pub(crate) fn up(&self)"),
            ("windows_ext", "pub fn busy_state("),
            ("windows_ext", "pub(crate) fn hwnd("),
            ("windows_ext", "pub fn click_through("),
            ("windows_ext", "pub(crate) fn passes_clicks("),
            ("windows_ext", "pub fn window_bounds("),
            ("windows_ext", "pub fn foreground_busy("),
            ("windows_ext", "pub(crate) fn busy_gui("),
            ("windows_ext", "fn class_is("),
            ("windows_ext", "pub fn covered_at("),
            ("windows_ext", "pub(crate) fn first_over<"),
            ("windows_ext", "fn windows_above("),
            ("windows_ext", "impl Above for Win"),
            ("windows_ext", "pub fn double_click_limits("),
            ("windows_ext", "pub(crate) fn half("),
            ("hit_shape", "pub fn contains("),
            ("hit_shape", "pub(crate) fn contains("),
            ("hit_shape", "pub(crate) fn on_screen("),
            ("hit_shape", "pub(crate) fn unpack("),
            ("hit_shape", "pub(crate) fn holds("),
            ("hit_shape", "pub fn now_ms("),
            ("prompt", "pub fn waiting_open("),
            ("prompt", "pub(crate) fn waiting_open_of("),
            ("prompt", "pub fn pressed_into("),
            ("prompt", "pub(crate) fn pressed_into_on("),
            ("prompt", "fn uncount_open("),
            ("prompt", "pub fn hwnd("),
            ("windows_ext", "pub fn foreground_window("),
        ] {
            let src = source(module);
            assert_clean(&format!("{module}::{sig}"), body(&src, sig));
        }
    }

    /// One place sets the hook, on its own thread, after its queue is made.
    #[test]
    fn only_the_hook_thread_sets_the_hook() {
        let needle = concat!("SetWindows", "HookExW(");
        let mut found = Vec::new();
        for (name, src) in crate::keyboard::tests::SOURCES {
            found.extend(src.match_indices(needle).map(|(i, _)| (name, i)));
        }
        assert_eq!(found.len(), 1, "{found:?}");
        let src = source("hook");
        let run = body(&src, "fn run_hook(");
        let set = run.find(needle).unwrap();
        let peek = run.find(concat!("PeekMes", "sageW(")).unwrap();
        let ready = run.find("ready.send(Ok(").unwrap();
        assert!(peek < set && set < ready, "the queue, then the hook, then its id");
    }

    /// The panic hook writes nothing for the hook's thread: its procedure
    /// may not wait on a file (the owner thread writes the line instead).
    #[test]
    fn the_panic_hook_passes_over_the_hook_thread() {
        let logs = source("logs");
        let hook = body(&logs, "pub fn install_panic_hook(");
        let skip = hook.find("if thread == crate::hook::HOOK_THREAD {\n            return;\n        }").expect("the hook's thread is passed over");
        assert!(skip < hook.find("crash(").unwrap() && skip < hook.find("before(info)").unwrap());
        assert_eq!(HOOK_THREAD, "fritz-mouse-hook");
    }

    /// Whatever can change the gate pokes the owner thread, so the hook
    /// goes in or out at once rather than at the next look.
    #[test]
    fn every_change_to_the_gate_pokes_the_hook() {
        let lib = source("lib");
        assert!(body(&lib, "pub(crate) fn change_setting(").contains("hook::poke();"), "a setting");
        assert!(body(&source("power"), "pub fn start(").contains("crate::hook::poke();"), "the display");
        let cover = body(&source("cover"), "pub fn start(").to_owned();
        assert!(cover.contains("if remember(") && cover.matches("crate::hook::poke();").count() == 2, "the lock, the busy state, a fullscreen app");
        assert!(body(&source("overlay_windows"), "fn step_effects(").contains("crate::hook::poke();"), "every overlay step");
    }

    #[test]
    fn quit_takes_the_hook_out_first() {
        let lib = source("lib");
        let quit = body(&lib, "pub(crate) fn quit(");
        let stop = quit.find("hook::stop();").expect("quit stops the hook");
        assert!(stop < quit.find("app.exit(0)").unwrap());
    }

    #[test]
    fn the_status_is_the_wire_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!("../../src/avatar/wire.fixture.json")).unwrap();
        let v = serde_json::to_value(status()).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut want: Vec<&str> = fixture["hookStatusKeys"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
        want.sort_unstable();
        assert_eq!(keys, want);
        assert_eq!(fixture["debug"]["hookWhileHiddenEnv"], WHILE_HIDDEN_ENV);
        assert_eq!(fixture["debug"]["pretendClickEnv"], PRETEND_CLICK_ENV);
        assert_eq!(fixture["events"]["hookStatus"], STATUS_EVENT);
        assert_eq!(fixture["hookOff"]["switch"], OFF_SWITCH);
        assert_eq!(fixture["hookOff"]["hidden"], OFF_HIDDEN);
    }
}
