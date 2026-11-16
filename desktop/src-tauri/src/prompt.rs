//! The prompt ("Ask Fritz"): a small window, declared in tauri.conf.json and
//! hidden at rest, that takes the keyboard so the owner can ask him
//! something, and hands it back as it closes.
//!
//! Its hotkey opens it, inside `WM_HOTKEY` on the main thread, while the
//! press's right to the foreground is fresh (the module never calls the
//! global-shortcut plugin, whose lock is held there). Nothing opens it
//! before its page has said it is ready, while Fritz is being arranged, over
//! the lock screen, another app's fullscreen window, a Direct3D fullscreen
//! app or presentation mode; a refusal shows nothing and takes nothing. It
//! stands centred over his window, just above it (below, with no room
//! above), on his monitor, at that monitor's DPI. It saves who had the
//! keyboard (`keyboard::return_target`), shows without activating, then
//! takes the keyboard with plain `SetForegroundWindow`; if Windows refuses,
//! it hides again at once. The key again closes it. Ask Fritz… in his menu
//! (the tray icon's, or a right-click on him) opens it as the key does, since
//! a click on Fritz's own menu is his process's input; it never closes it.
//!
//! **A prompt that waits for a click.** Opened by a gesture on him (a
//! double-click, which the mouse hook reports) that Windows won't give the
//! keyboard to, it stays shown and waits: its page says "Click here to
//! type…", and the owner's click into it is a click on a window of ours,
//! which Windows activates. It closes unanswered after `WAIT_FOR_CLICK`, on
//! a click elsewhere, on the double-click again, and as the hook goes out;
//! its key, or Ask Fritz…, takes the keyboard into it. A waiting open isn't
//! the owner's for the foreground log until the hook reports a press into it
//! (counted there, before Windows activates it) or its key or Ask Fritz…
//! takes it (`takes`): the prompt coming to the front otherwise is a HIT.
//!
//! **The owner's count** is one word, `credit`: the open counted in `takes`.
//! The main thread counts its opens, the hook's procedure a press into a
//! waiting one; whoever claims it first counts it, a claim from an open that
//! has ended is taken back, and an open that never had the keyboard gives
//! its count back as it closes.
//!
//! Every way out is one close, which runs once however many ask for it: a
//! question taken (Enter), Esc, Alt+F4, its key again, its page loading
//! again, the owner going to another window, the double-click on him again,
//! or, while it waits for a click, a click elsewhere, nobody clicking it in
//! time, or the hook going out.
//! 1. While the prompt still has the keyboard, it hands it back, by
//!    [`crate::keyboard`]'s rules: to the window saved before it took it,
//!    else the app in use.
//! 2. It hides.
//! 3. If the hidden prompt still has the keyboard, the app in use is given
//!    it. A hidden window never keeps it.
//!
//! The owner going elsewhere only hides the prompt: the keyboard is where
//! they sent it, and nothing takes it back. Windows says so (`WM_ACTIVATE`),
//! and so does the webview, which alone can miss it. A webview blur while
//! the prompt, or a window it owns, is still in front changes nothing.
//!
//! Closes are posted to the main thread, never run where they are asked
//! for: on a command's thread, inside the window procedure, or inside
//! WebView2's own callbacks. The one exception is its key again, which
//! closes it inline in `WM_HOTKEY`, as the key opens it. Each close carries
//! the open it was asked in, so a close left over from an earlier open never
//! closes a later one.

use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, WebviewWindow, WindowEvent};

use crate::keyboard::{self, Keyboard};
use crate::overlay_machine::should_return_focus;
use crate::windows_ext::Bounds;
use crate::{cover, logs, overlay_windows, windows_ext};

/// The window's label in tauri.conf.json.
pub const LABEL: &str = "prompt";
/// Its size in logical pixels: tauri.conf.json's (a test holds them together).
pub const SIZE: (f64, f64) = (640.0, 120.0);
/// Logical pixels between the prompt and his window, and kept clear of the
/// work area's edges.
const GAP: f64 = 12.0;
/// A debug build only: once the prompt's page is first ready, open it as
/// the hotkey would, but off screen and without taking the keyboard (a live
/// check's open, with no key pressed).
pub const PRETEND_ENV: &str = "FRITZ_PRETEND_PROMPT";
/// The pretend open happens once a run.
static PRETENDED: AtomicBool = AtomicBool::new(false);
/// `FRITZ_PRETEND_PROMPT`'s value for a pretend open that waits for a click.
pub const PRETEND_WAIT: &str = "wait";
/// Tells the prompt's page whether it waits for a click (`true`) or not.
pub const WAITING_EVENT: &str = "prompt:waiting";
/// How long a prompt waits for a click before it closes.
pub const WAIT_FOR_CLICK: Duration = Duration::from_secs(20);

/// Where the prompt stands: its HWND, whether its page is ready and whether
/// it is open, who had the keyboard before it took it, how many times it has
/// been opened (each close carries the open it was asked in), and how many
/// of those opens are the owner's (the foreground log's fresh open). For the
/// open showing now: whether it waits for a click, whether it ever had the
/// keyboard, whether it is a pretend, and whether its page was told it
/// waits. And which open is counted in `takes` (`credit`; 0: none), the
/// last open the hook saw a press into while it waited (`pressed`), and
/// whether a take is under way (`taking`).
pub(crate) struct State {
    hwnd: AtomicIsize,
    ready: AtomicBool,
    open: AtomicBool,
    back_to: AtomicIsize,
    opens: AtomicU64,
    takes: AtomicU64,
    waiting: AtomicBool,
    took: AtomicBool,
    credit: AtomicU64,
    pressed: AtomicU64,
    taking: AtomicBool,
    pretend: AtomicBool,
    told: AtomicBool,
}

impl State {
    const fn new() -> Self {
        Self {
            hwnd: AtomicIsize::new(0),
            ready: AtomicBool::new(false),
            open: AtomicBool::new(false),
            back_to: AtomicIsize::new(0),
            opens: AtomicU64::new(0),
            takes: AtomicU64::new(0),
            waiting: AtomicBool::new(false),
            took: AtomicBool::new(false),
            credit: AtomicU64::new(0),
            pressed: AtomicU64::new(0),
            taking: AtomicBool::new(false),
            pretend: AtomicBool::new(false),
            told: AtomicBool::new(false),
        }
    }
}

static STATE: State = State::new();
/// The app, for the activation watch (a plain `fn(bool)` in a window procedure).
static APP: OnceLock<AppHandle> = OnceLock::new();

/// Why the prompt is closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// Its question was taken (Enter).
    Sent,
    Esc,
    AltF4,
    /// The owner went to another window.
    Blur,
    /// Its page started loading again.
    Reload,
    /// Its key, pressed again.
    Toggle,
    /// The double-click on him, again.
    Gesture,
    /// A click elsewhere, while it waited for one (the hook's report).
    Elsewhere,
    /// Another choice from his menu than Ask Fritz…, while it waited for a
    /// click (tray.rs): the press on the menu spared it, so the choice
    /// decides.
    Chose,
    /// Nobody clicked into it in time.
    Unanswered,
    /// The mouse hook went out while it waited: a click elsewhere would go
    /// unseen.
    HookOut,
}

impl Why {
    fn says(self) -> &'static str {
        match self {
            Self::Sent => "its question taken",
            Self::Esc => "Esc",
            Self::AltF4 => "Alt+F4",
            Self::Blur => "the owner went elsewhere",
            Self::Reload => "its page loaded again",
            Self::Toggle => "its key again",
            Self::Gesture => "a double-click on him again",
            Self::Elsewhere => "a click elsewhere",
            Self::Chose => "another choice from his menu",
            Self::Unanswered => "nobody clicked it",
            Self::HookOut => "the mouse hook went out",
        }
    }
}

/// What a close did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Closed {
    /// It wasn't open, or that open is over: nothing was touched.
    NotOpen,
    /// A blur, or a close for a waiting prompt, while the prompt (or a
    /// window it owns) is in front: kept, in use.
    StillHasKeyboard,
    /// Hidden, with the keyboard left where it was.
    Hidden,
    /// Hidden, and the keyboard handed back to this window.
    HandedTo(isize),
    /// The hidden prompt still had the keyboard: this window, the app in
    /// use, was given it.
    Stranded(isize),
}

/// Who opens the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opener {
    /// Its hotkey: it takes the keyboard.
    Hotkey,
    /// A gesture on him (the double-click): it takes the keyboard, or,
    /// refused, waits for a click.
    Click,
    /// Ask Fritz… in his menu (the tray icon's, or a right-click on him): a
    /// click on Fritz's own menu, so it takes the keyboard. Open already, it
    /// leaves a prompt in use alone and takes the keyboard into one waiting.
    Menu,
    /// `FRITZ_PRETEND_PROMPT`, in a debug build: off screen, the keyboard
    /// untouched, and never counted as the owner's open.
    Pretend,
    /// `FRITZ_PRETEND_PROMPT=wait`: a pretend open that waits for a click,
    /// as a refused gesture's does.
    PretendWait,
}

/// What is open when an opener comes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenKind {
    /// It has (or had) the keyboard.
    Holding,
    /// It waits for a click.
    Waiting,
    /// A pretend open.
    Pretend,
}

/// What an opener does to a prompt already open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhenOpen {
    Close,
    /// Take the keyboard into the waiting prompt.
    Take,
    Refuse,
}

impl Opener {
    /// Shown where the owner sees it, the keyboard taken (or asked for).
    pub(crate) fn on_screen(self) -> bool {
        matches!(self, Self::Hotkey | Self::Click | Self::Menu)
    }

    /// Counted as the owner's open before the take, so the foreground log
    /// never sees the prompt in front without it.
    pub(crate) fn counts(self) -> bool {
        self.on_screen()
    }

    /// Refused the keyboard, it waits for a click rather than hiding.
    pub(crate) fn may_wait(self) -> bool {
        matches!(self, Self::Click | Self::PretendWait)
    }

    /// The opener, for the log.
    pub(crate) fn says(self) -> &'static str {
        match self {
            Self::Hotkey => "hotkey",
            Self::Click => "a double-click on him",
            Self::Menu => "his menu",
            Self::Pretend => "pretend",
            Self::PretendWait => "pretend double-click",
        }
    }

    /// Why its closing of an open prompt says it closed.
    pub(crate) fn closes_as(self) -> Why {
        match self {
            Self::Click => Why::Gesture,
            Self::Hotkey | Self::Menu | Self::Pretend | Self::PretendWait => Why::Toggle,
        }
    }

    /// Its key again closes the prompt, but takes the keyboard into one that
    /// waits (the owner asked to type); a gesture again closes it; his menu
    /// takes the keyboard into a waiting one and leaves one in use alone; a
    /// pretend never touches an open prompt.
    pub(crate) fn when_open(self, open: OpenKind) -> WhenOpen {
        match (self, open) {
            (Self::Hotkey | Self::Menu, OpenKind::Waiting) => WhenOpen::Take,
            (Self::Hotkey | Self::Click, _) => WhenOpen::Close,
            (Self::Menu, _) => WhenOpen::Refuse,
            (Self::Pretend | Self::PretendWait, _) => WhenOpen::Refuse,
        }
    }
}

/// Why an open is refused (the prompt left as it was).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refusal {
    NoWindow,
    NotReady,
    /// A pretend open, or his menu, on a prompt open already (left as it
    /// is).
    AlreadyOpen,
    Arranging,
    NoPlace,
    /// Something nothing should open over: the lock screen, another app's
    /// fullscreen window, a Direct3D fullscreen app, presentation mode.
    Blocked(String),
    /// Windows wouldn't give it the keyboard.
    NotGiven,
}

impl Refusal {
    fn says(&self) -> String {
        match self {
            Self::NoWindow => "there is no prompt window".into(),
            Self::NotReady => "its page isn't ready".into(),
            Self::AlreadyOpen => "it is open already".into(),
            Self::Arranging => "Fritz is being arranged".into(),
            Self::NoPlace => "there is no monitor to put it on".into(),
            Self::Blocked(by) => format!("{by} is in front"),
            Self::NotGiven => "Windows wouldn't give it the keyboard".into(),
        }
    }
}

/// Where the prompt would stand, and what it was placed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Place {
    pub at: Bounds,
    /// His window (where he stands, or would when hidden), if known.
    pub anchor: Option<Bounds>,
    /// The work area of the monitor it stands on.
    pub work: Bounds,
}

/// What an open did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Opened {
    Shown(Place),
    /// Shown, waiting for a click: Windows kept the keyboard (or a pretend).
    Waiting(Place),
    /// Its key, or his menu, took the keyboard into the prompt waiting for a
    /// click.
    Taken,
    /// It was open: the key again closed it.
    Closed(Closed),
    Refused(Refusal),
}

/// What opening and closing ask of the desktop beyond the keyboard. A trait
/// so both are tested without a window.
pub(crate) trait PromptDesk {
    fn hide(&self);
    /// The window at the top of `hwnd`'s owner chain.
    fn root_owner(&self, hwnd: isize) -> isize;
    /// Whether Fritz is being arranged.
    fn arranging(&self) -> bool;
    /// Where the prompt would stand now.
    fn where_to(&self) -> Option<Place>;
    /// What, if anything, nothing should open over, for a prompt at `at`.
    fn blocked(&self, at: Bounds) -> Option<String>;
    /// Size and move the prompt to `at` (physical pixels).
    fn place(&self, at: Bounds);
    /// Show it, without activating it.
    fn show(&self);
    /// Tell its page whether it waits for a click.
    fn tell_waiting(&self, on: bool);
}

/// Open `open` counted as the owner's (`takes`), once. `takes` goes up
/// before any claim is made and down only after one is taken out, so it
/// never falls below the claims (nor wraps below zero) whichever thread
/// races which. Claimed for it already (the hook counted a press into it), the count is
/// taken back; claimed for an earlier open (a press that raced that open's
/// close), that count is taken back and the claim made. Whether this call
/// counted it. Main thread.
fn count_open(s: &State, open: u64) -> bool {
    s.takes.fetch_add(1, Ordering::SeqCst);
    loop {
        match s.credit.compare_exchange(0, open, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return true,
            Err(c) if c == open => {
                s.takes.fetch_sub(1, Ordering::SeqCst);
                return false;
            }
            Err(c) => uncount_open(s, c),
        }
    }
}

/// Open `open`'s count given back, if it holds the claim.
fn uncount_open(s: &State, open: u64) {
    if s.credit.compare_exchange(open, 0, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
        s.takes.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The claim taken out as open `open` ends: kept in `takes` only if it is
/// `open`'s and that open had the keyboard (`had`). At a fresh open, with
/// `(0, false)`: any claim left is a press that raced the last close.
fn settle_count(s: &State, open: u64, had: bool) {
    let c = s.credit.swap(0, Ordering::SeqCst);
    if c != 0 && !(c == open && had) {
        s.takes.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A real press into the prompt while it waits, from the hook's procedure
/// before the press reaches it (so before Windows activates it): the
/// owner's open, counted. Atomics only. The open counted, or 0 (not
/// waiting, counted already, or that open ended meanwhile, when the claim is
/// taken back).
pub(crate) fn pressed_into_on(s: &State) -> u64 {
    if !s.waiting.load(Ordering::SeqCst) {
        return 0;
    }
    let open = s.opens.load(Ordering::SeqCst);
    // Seen, whether or not this claim lands: a take refused meanwhile, whose
    // claim blocked it, counts it again.
    s.pressed.store(open, Ordering::SeqCst);
    s.takes.fetch_add(1, Ordering::SeqCst);
    if s.credit.compare_exchange(0, open, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        s.takes.fetch_sub(1, Ordering::SeqCst);
        return 0;
    }
    // A close or a new open raced it: the claim goes back, unless the main
    // thread has taken it out already.
    if !s.open.load(Ordering::SeqCst) || s.opens.load(Ordering::SeqCst) != open {
        uncount_open(s, open);
        return 0;
    }
    open
}

/// The take, for open `open`, counted just before (`counted_now`: by that
/// count, not a press's). `taking` marks it, so the prompt's activation
/// inside it (Windows activates a window of this thread inside the call,
/// granted or, perhaps, refused) is the take's, not a click's, and changes
/// nothing. Refused, the count goes back; but a press the
/// hook saw into the waiting prompt meanwhile, whose claim this count
/// blocked, is counted again: the owner clicked into it. Main thread.
fn take(s: &State, k: &impl Keyboard, ours: isize, open: u64, counted_now: bool) -> bool {
    s.taking.store(true, Ordering::SeqCst);
    let ok = k.try_foreground(ours);
    s.taking.store(false, Ordering::SeqCst);
    if !ok && counted_now {
        uncount_open(s, open);
        if s.pressed.load(Ordering::SeqCst) == open {
            count_open(s, open);
        }
    }
    ok
}

/// The open waiting for a click; 0 when none. Atomics only.
pub(crate) fn waiting_open_of(s: &State) -> u64 {
    if s.waiting.load(Ordering::SeqCst) {
        s.opens.load(Ordering::SeqCst)
    } else {
        0
    }
}

/// The open waiting for a click after a double-click (never a pretend);
/// 0 when none.
pub(crate) fn waiting_click_open_of(s: &State) -> u64 {
    if s.pretend.load(Ordering::SeqCst) {
        0
    } else {
        waiting_open_of(s)
    }
}

/// The hook saw a press into the waiting prompt, open `open`, while `fg`
/// had the keyboard: that is who gets it back, if the open is still
/// showing and `fg` is the owner's (they may have moved on during the
/// wait). If the prompt is in front by now, it is in use. Main thread.
pub(crate) fn saw_press_on(s: &State, k: &impl Keyboard, w: &impl PromptDesk, open: u64, fg: isize) -> bool {
    if !s.open.load(Ordering::SeqCst) || s.opens.load(Ordering::SeqCst) != open {
        return false;
    }
    // Nobody, or a window of ours but Settings (the prompt itself, in use
    // already; the tray's menu): the press says nothing of the owner's app,
    // and the save from its open stands.
    let saved = !(fg == 0 || (k.owned_by_us(fg) && fg != k.settings_window()));
    if saved {
        let ours = s.hwnd.load(Ordering::SeqCst);
        s.back_to.store(keyboard::return_target(fg, ours, k), Ordering::SeqCst);
    }
    // In front by now, the click went in, its activation perhaps unseen.
    in_use_if_in_front(s, k, w);
    saved
}

/// A waiting prompt in front (it, or a window it owns, has the keyboard)
/// is in use, its activation unseen (Windows may send none for a click into
/// a prompt it already made its thread's active window): it no longer
/// waits, it has had the keyboard, and its page is told. Whether it was.
/// Main thread.
fn in_use_if_in_front(s: &State, k: &impl Keyboard, w: &impl PromptDesk) -> bool {
    let ours = s.hwnd.load(Ordering::SeqCst);
    let fg = k.foreground();
    if !s.waiting.load(Ordering::SeqCst) || ours == 0 || fg == 0 || !(fg == ours || w.root_owner(fg) == ours) {
        return false;
    }
    s.took.store(true, Ordering::SeqCst);
    s.waiting.store(false, Ordering::SeqCst);
    tell(s, w, false);
    true
}

/// Tell the page it waits, or no longer does, only when that changes.
fn tell(s: &State, w: &impl PromptDesk, on: bool) {
    if s.told.swap(on, Ordering::SeqCst) != on {
        w.tell_waiting(on);
    }
}

/// Close the prompt, if open `open` is still the one showing: the order the
/// module's doc gives, deciding by who has the keyboard now. Main thread.
pub(crate) fn close_on(s: &State, k: &impl Keyboard, w: &impl PromptDesk, why: Why, open: u64) -> Closed {
    if !s.open.load(Ordering::SeqCst) || s.opens.load(Ordering::SeqCst) != open {
        return Closed::NotOpen;
    }
    let ours = s.hwnd.load(Ordering::SeqCst);
    let holds = |h: isize| ours != 0 && h != 0 && (h == ours || w.root_owner(h) == ours);
    let fg = k.foreground();
    // A click elsewhere, none in time, or the hook going out closes only a
    // prompt still waiting: never one the owner has clicked into, or its key
    // took. One in front is in use, its activation unseen: it no longer
    // waits.
    if matches!(why, Why::Elsewhere | Why::Chose | Why::Unanswered | Why::HookOut) {
        if !s.waiting.load(Ordering::SeqCst) {
            return Closed::NotOpen;
        }
        if in_use_if_in_front(s, k, w) {
            return Closed::StillHasKeyboard;
        }
    }
    if why == Why::Blur && holds(fg) {
        return Closed::StillHasKeyboard;
    }
    if !s.open.swap(false, Ordering::SeqCst) {
        return Closed::NotOpen;
    }
    // Still waiting, it never had the keyboard (the tray's menu may have it
    // now): nothing to hand back, unless it has it after all (its
    // activation unseen: the watch didn't go in).
    let waited = s.waiting.swap(false, Ordering::SeqCst);
    // Counted, but the keyboard never went in: not the owner's after all. A
    // prompt that holds the keyboard at its close had it (its activation
    // may have gone unseen).
    settle_count(s, open, s.took.load(Ordering::SeqCst) || holds(fg));
    let saved = s.back_to.swap(0, Ordering::SeqCst);
    let settings = k.settings_window();
    let target = if why == Why::Blur || (waited && !holds(fg)) {
        0
    } else if holds(fg) {
        keyboard::hand_back(saved, ours, k)
    } else if fg != 0 && k.owned_by_us(fg) && !k.can_take_foreground(fg) {
        // The tray's hidden window, which stands in for the prompt (its menu
        // opened over it): the app beneath, as Lock from the tray does.
        k.next_app_window(ours, settings)
    } else {
        0
    };
    // The prompt has the keyboard here, or the tray's window stands in for it.
    let handed = should_return_focus(ours, ours, target, k.can_take_foreground(target)) && k.try_foreground(target);
    w.hide();
    tell(s, w, false);
    if holds(k.foreground()) {
        let to = k.next_app_window(ours, settings);
        k.try_foreground(to);
        return Closed::Stranded(to);
    }
    if handed {
        Closed::HandedTo(target)
    } else {
        Closed::Hidden
    }
}

/// Where the prompt stands, in physical pixels, for a monitor's work area
/// `work` at `dpi`: centred on his window `anchor`, its bottom a gap above
/// his top, or a gap below him when there is no room above; with no anchor,
/// centred, its middle a third of the way down. Shrunk to fit a small work
/// area, and kept inside it, a gap from its edges. Pure.
pub(crate) fn prompt_rect(anchor: Option<Bounds>, work: Bounds, dpi: u32) -> Bounds {
    let s = f64::from(dpi.max(48)) / 96.0;
    #[allow(clippy::cast_possible_truncation)]
    let (m, w, h) = ((GAP * s).round() as i32, (SIZE.0 * s).round() as i32, (SIZE.1 * s).round() as i32);
    let w = w.min(work.width() - 2 * m).max(1);
    let h = h.min(work.height() - 2 * m).max(1);
    let (left, top) = match anchor {
        Some(a) => {
            let above = a.top - m - h;
            (a.left + a.width() / 2 - w / 2, if above < work.top + m { a.bottom + m } else { above })
        }
        None => (work.left + (work.width() - w) / 2, work.top + work.height() / 3 - h / 2),
    };
    // Never `clamp`, which panics on a work area too small for the gaps.
    let left = left.min(work.right - m - w).max(work.left + m);
    let top = top.min(work.bottom - m - h).max(work.top + m);
    Bounds { left, top, right: left + w, bottom: top + h }
}

/// `at` moved off every monitor, its size kept: where a pretend open stands.
fn off_screen(at: Bounds) -> Bounds {
    let (x, y) = overlay_windows::PARK_AT;
    Bounds { left: x, top: y, right: x + at.width(), bottom: y + at.height() }
}

/// Open the prompt, or, open already, close it or take the keyboard into
/// it (`Opener::when_open`). The gates come first and touch no window; then
/// who has the keyboard is saved, the prompt placed and shown, and (an
/// on-screen open) the keyboard taken. Refused, the hotkey's or his menu's
/// open is undone; a gesture's waits for a click. Main thread.
pub(crate) fn open_on(s: &State, k: &impl Keyboard, d: &impl PromptDesk, how: Opener) -> Opened {
    let ours = s.hwnd.load(Ordering::SeqCst);
    if ours == 0 {
        return Opened::Refused(Refusal::NoWindow);
    }
    if !s.ready.load(Ordering::SeqCst) {
        return Opened::Refused(Refusal::NotReady);
    }
    if s.open.load(Ordering::SeqCst) {
        // A waiting prompt in front is in use: its key closes it.
        let kind = if s.pretend.load(Ordering::SeqCst) {
            OpenKind::Pretend
        } else if s.waiting.load(Ordering::SeqCst) && !in_use_if_in_front(s, k, d) {
            OpenKind::Waiting
        } else {
            OpenKind::Holding
        };
        return match how.when_open(kind) {
            WhenOpen::Close => Opened::Closed(close_on(s, k, d, how.closes_as(), s.opens.load(Ordering::SeqCst))),
            WhenOpen::Take => take_into(s, k, d, ours),
            WhenOpen::Refuse => Opened::Refused(Refusal::AlreadyOpen),
        };
    }
    if d.arranging() {
        return Opened::Refused(Refusal::Arranging);
    }
    let Some(place) = d.where_to() else {
        return Opened::Refused(Refusal::NoPlace);
    };
    if let Some(by) = d.blocked(place.at) {
        return Opened::Refused(Refusal::Blocked(by));
    }
    let on_screen = how.on_screen();
    let back_to = if on_screen { keyboard::return_target(k.foreground(), ours, k) } else { 0 };
    s.back_to.store(back_to, Ordering::SeqCst);
    d.place(if on_screen { place.at } else { off_screen(place.at) });
    // Every open starts afresh.
    s.waiting.store(false, Ordering::SeqCst);
    s.took.store(false, Ordering::SeqCst);
    settle_count(s, 0, false);
    s.pretend.store(!on_screen, Ordering::SeqCst);
    s.open.store(true, Ordering::SeqCst);
    let open = s.opens.fetch_add(1, Ordering::SeqCst) + 1;
    // Counted before the take, so the foreground log, polling on its own
    // thread, never sees the prompt in front without it.
    let counted_now = how.counts() && count_open(s, open);
    // Told before it shows: if it is refused the keyboard, the page already
    // asks for a click.
    if how.may_wait() {
        tell(s, d, true);
    }
    d.show();
    match how {
        Opener::Hotkey | Opener::Click | Opener::Menu => {
            // A gesture's open waits from the take on: a press into it
            // meanwhile is the owner's click, and one elsewhere closes it.
            if how.may_wait() {
                s.waiting.store(true, Ordering::SeqCst);
            }
            if take(s, k, ours, open, counted_now) {
                s.waiting.store(false, Ordering::SeqCst);
                s.took.store(true, Ordering::SeqCst);
                tell(s, d, false);
                Opened::Shown(place)
            } else if how.may_wait() {
                // Not the owner's open (its count went back with the take)
                // until they click into it, or its key takes the keyboard in.
                Opened::Waiting(place)
            } else {
                s.open.store(false, Ordering::SeqCst);
                s.back_to.store(0, Ordering::SeqCst);
                d.hide();
                Opened::Refused(Refusal::NotGiven)
            }
        }
        Opener::Pretend => Opened::Shown(place),
        Opener::PretendWait => {
            s.waiting.store(true, Ordering::SeqCst);
            Opened::Waiting(place)
        }
    }
}

/// Its key (or his menu) on the prompt waiting for a click: the keyboard
/// taken into it, counted first, and who had it saved afresh (the wait may
/// have been long, and the owner may have moved on; with the tray's window
/// in front, the app in use now, as a fresh open from it saves); refused, it
/// goes on waiting,
/// uncounted (unless the owner clicked into it meanwhile), the save from
/// its open kept.
fn take_into(s: &State, k: &impl Keyboard, d: &impl PromptDesk, ours: isize) -> Opened {
    let open = s.opens.load(Ordering::SeqCst);
    let counted_now = count_open(s, open);
    let back_to = keyboard::return_target(k.foreground(), ours, k);
    // Still waiting through the take: a press into it meanwhile is the
    // owner's click, and its activation inside the take is the take's.
    if take(s, k, ours, open, counted_now) {
        s.waiting.store(false, Ordering::SeqCst);
        s.back_to.store(back_to, Ordering::SeqCst);
        s.took.store(true, Ordering::SeqCst);
        tell(s, d, false);
        Opened::Taken
    } else {
        Opened::Refused(Refusal::NotGiven)
    }
}

/// The prompt and the desktop, for `open_on` and `close_on`.
struct Live<'a> {
    app: &'a AppHandle,
    window: &'a WebviewWindow,
}

impl PromptDesk for Live<'_> {
    fn hide(&self) {
        let _ = self.window.hide();
    }

    fn root_owner(&self, hwnd: isize) -> isize {
        windows_ext::root_owner(hwnd)
    }

    fn arranging(&self) -> bool {
        overlay_windows::arranging(self.app)
    }

    fn where_to(&self) -> Option<Place> {
        let anchor = overlay_windows::standing(self.app)
            .or_else(crate::first_rect)
            .map(overlay_windows::bounds_of);
        let (work, dpi) = anchor
            .and_then(|a| windows_ext::work_area_at(a.left + a.width() / 2, a.top + a.height() / 2))
            .or_else(windows_ext::primary_work_area)?;
        Some(Place { at: prompt_rect(anchor, work, dpi), anchor, work })
    }

    /// The lock and the shell's busy states as the cover watch last heard
    /// them (inside `WM_HOTKEY`, nothing waits on Windows' services, but a
    /// lock or a busy state it still says is checked); the foreground and the
    /// monitors, looked at now.
    fn blocked(&self, at: Bounds) -> Option<String> {
        cover::covered_by(cover::lock_now(cover::locked(), windows_ext::session_locked), || {
            windows_ext::fullscreen_foreground()
                .or_else(|| cover::busy_now(cover::busy(), windows_ext::busy_now).map(str::to_owned))
                .or_else(|| windows_ext::covering_app(at))
        })
    }

    /// On the main thread, where tao's setters are synchronous, so Win32
    /// reads back where it stands at once. Up to three passes: a move onto
    /// a monitor of another DPI can resize it as it lands.
    fn place(&self, at: Bounds) {
        let hwnd = STATE.hwnd.load(Ordering::SeqCst);
        let size = PhysicalSize::new(u32::try_from(at.width()).unwrap_or(1), u32::try_from(at.height()).unwrap_or(1));
        let pos = PhysicalPosition::new(at.left, at.top);
        for _ in 0..3 {
            let _ = self.window.set_size(size);
            let _ = self.window.set_position(pos);
            if windows_ext::window_bounds(hwnd) == Some(at) {
                return;
            }
        }
        tracing::debug!("the prompt didn't settle at {}", rect(at));
    }

    fn show(&self) {
        let _ = self.window.show();
    }

    fn tell_waiting(&self, on: bool) {
        let _ = self.window.emit_to(LABEL, WAITING_EVENT, on);
    }
}

/// A rectangle for the log.
fn rect(b: Bounds) -> String {
    format!("({}, {}) {}x{}", b.left, b.top, b.width(), b.height())
}

/// How many times the prompt has been opened: what a close asked for now
/// carries, so it can't close a later open.
pub fn opens() -> u64 {
    STATE.opens.load(Ordering::SeqCst)
}

/// How many opens took the keyboard (the owner's, never a pretend one), for
/// the foreground log: a PROMPT line needs a new one.
pub fn takes() -> u64 {
    STATE.takes.load(Ordering::SeqCst)
}

/// The open waiting for a click; 0 when none. For the hook's procedure.
pub fn waiting_open() -> u64 {
    waiting_open_of(&STATE)
}

/// The open waiting for a click after a double-click, never a pretend; 0
/// when none. For the hook's owner thread while the hook is out.
pub fn waiting_click_open() -> u64 {
    waiting_click_open_of(&STATE)
}

/// A real press into the waiting prompt, counted as the owner's open: the
/// hook's procedure, before the press reaches it. The open counted, or 0.
pub fn pressed_into() -> u64 {
    pressed_into_on(&STATE)
}

/// The hook saw a press into the waiting prompt, open `open`, while `fg`
/// had the keyboard: posted to the main thread (see [`saw_press_on`]).
pub fn saw_press_later(app: &AppHandle, open: u64, fg: isize) {
    post(app, move |h| {
        let Some(w) = h.get_webview_window(LABEL) else {
            return;
        };
        if saw_press_on(&STATE, &keyboard::Desk::of(h), &Live { app: h, window: &w }, open, fg) {
            tracing::debug!("a press into the waiting prompt: the keyboard goes back to the app it was pressed from");
        }
    });
}

/// `toggle`, posted to the main thread: the double-click's, from the hook's
/// thread.
pub fn toggle_later(app: &AppHandle, how: Opener) {
    post(app, move |h| toggle(h, how));
}

/// The prompt's HWND; 0 when there is none. For the foreground log.
pub fn hwnd() -> isize {
    STATE.hwnd.load(Ordering::SeqCst)
}

/// Whether a hotkey event opens (or closes) the prompt: its key's press.
pub fn hotkey_request(id: &str, pressed: bool) -> bool {
    pressed && id == crate::hotkeys::PROMPT
}

/// The pretend open asked for, if any: in a debug build, with
/// `FRITZ_PRETEND_PROMPT` set, `wait` for one that waits for a click, any
/// other value for PR 4's (`debug` is `cfg!(debug_assertions)` at the one
/// caller, so a release build never reads it).
pub fn pretend_kind(debug: bool, env: Option<OsString>) -> Option<Opener> {
    if !debug {
        return None;
    }
    let v = env.filter(|v| !v.is_empty())?;
    Some(if v == PRETEND_WAIT { Opener::PretendWait } else { Opener::Pretend })
}

/// Open or close the prompt, and say what happened. The hotkey's runs inline
/// in `WM_HOTKEY`. Main thread.
pub fn toggle(app: &AppHandle, how: Opener) {
    let Some(w) = app.get_webview_window(LABEL) else {
        tracing::warn!("there is no prompt window");
        return;
    };
    let opened = open_on(&STATE, &keyboard::Desk::of(app), &Live { app, window: &w }, how);
    // A new open waiting for a click is timed; one its key failed to take
    // into is timed already.
    let waits = matches!(opened, Opened::Waiting(_));
    match opened {
        Opened::Shown(p) if how.on_screen() => {
            tracing::info!("the prompt opened ({}) and took the keyboard at {}", how.says(), rect(p.at));
        }
        Opened::Shown(p) | Opened::Waiting(p) if !how.on_screen() => tracing::info!(
            "the prompt opened ({}: off screen, keyboard untouched){}; it would stand at {} in the work area {}, by his window at {}",
            how.says(),
            if how.may_wait() { " and waits for a click" } else { "" },
            rect(p.at),
            rect(p.work),
            p.anchor.map_or_else(|| "(none)".to_owned(), rect)
        ),
        Opened::Shown(_) => {}
        Opened::Waiting(_) => {
            tracing::info!("the prompt opened ({}) and waits for a click: Windows kept the keyboard where it was", how.says());
        }
        Opened::Taken => tracing::info!("the keyboard went into the waiting prompt ({})", how.says()),
        Opened::Closed(c) => log_closed(how.closes_as(), c),
        Opened::Refused(r @ Refusal::NotGiven) if STATE.waiting.load(Ordering::SeqCst) => {
            tracing::info!("the waiting prompt goes on waiting for a click: {}", r.says());
        }
        Opened::Refused(r @ Refusal::NotGiven) => tracing::warn!("the prompt stays hidden: {}", r.says()),
        Opened::Refused(r @ Refusal::AlreadyOpen) => tracing::info!("the prompt stays as it is ({}): {}", how.says(), r.says()),
        Opened::Refused(r) => tracing::info!("the prompt stays hidden: {}", r.says()),
    }
    if waits {
        unanswered_after(app, opens(), WAIT_FOR_CLICK);
    }
}

/// Close the prompt waiting for a click, open `open`, if nobody has clicked
/// into it after `wait`. A sleeper of its own; the close is posted, and is
/// ignored by a later open or a prompt no longer waiting.
fn unanswered_after(app: &AppHandle, open: u64, wait: Duration) {
    let h = app.clone();
    let spawned = std::thread::Builder::new().name("prompt-waits".into()).spawn(move || {
        std::thread::sleep(wait);
        close_later(&h, Why::Unanswered, open);
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't time the waiting prompt: it closes on a click elsewhere or its key");
    }
}

/// The prompt's page says it is ready (`prompt_ready`): the hotkey may open
/// it. With `pretend`, the first time, that pretend open, posted.
pub fn page_ready(app: &AppHandle, pretend: Option<Opener>) {
    if !STATE.ready.swap(true, Ordering::SeqCst) {
        tracing::info!("the prompt's page is ready");
    }
    if let Some(how) = pretend {
        if !PRETENDED.swap(true, Ordering::SeqCst) {
            post(app, move |h| toggle(h, how));
        }
    }
}

/// Run `f` on the main thread, posted through the async runtime from
/// wherever it is asked. Tauri runs a main-thread task inline when it is
/// asked from the main thread, so one asked inside a window procedure or a
/// WebView2 callback would otherwise run there.
fn post(app: &AppHandle, f: impl FnOnce(&AppHandle) + Send + 'static) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let h = app.clone();
        let _ = app.run_on_main_thread(move || {
            logs::guard("prompt", || f(&h));
        });
    });
}

/// At setup, on the main thread: remember the prompt's HWND, keep it out of
/// Alt-Tab (as the avatar is), and watch its activation (the owner clicking
/// into it while it waits, or going elsewhere).
pub fn adopt(app: &tauri::App) {
    let _ = APP.set(app.handle().clone());
    let Some(hwnd) = app
        .get_webview_window(LABEL)
        .and_then(|w| w.hwnd().ok())
        .map(|h| h.0 as isize)
    else {
        tracing::warn!("there is no prompt window");
        return;
    };
    STATE.hwnd.store(hwnd, Ordering::SeqCst);
    if !windows_ext::pin_toolwindow(hwnd) {
        tracing::warn!("couldn't keep the prompt out of Alt-Tab");
    }
    if !windows_ext::watch_activation(hwnd, activation) {
        tracing::warn!(
            "couldn't watch the prompt's activation: a click into it while it waits won't be seen (it closes unanswered), and the owner going elsewhere only by its page"
        );
    }
}

/// Windows says the prompt became the active window, or stopped being it:
/// from inside its window procedure, so only store and post. Active outside
/// a take of ours, it has had the keyboard (a waiting prompt clicked into
/// stops waiting); inside one, it changes nothing; inactive, the owner went
/// elsewhere.
fn activation(active: bool) {
    logs::guard("prompt", || {
        let Some(app) = APP.get() else {
            return;
        };
        if active {
            if activated(&STATE) {
                post(app, no_longer_waits);
            }
        } else {
            close_later(app, Why::Blur, opens());
        }
    });
}

/// The waiting prompt became the active window (the owner's click into it,
/// or Windows' own doing; never a take of ours, its open's or its key's,
/// whose activation comes inside the take and is told apart by `taking`):
/// its page told it no longer waits. Main thread, posted.
fn no_longer_waits(h: &AppHandle) {
    if let Some(w) = h.get_webview_window(LABEL) {
        tell(&STATE, &Live { app: h, window: &w }, false);
    }
    tracing::info!("the waiting prompt became the active window: it no longer waits");
}

/// The prompt became active outside a take of ours: it has had the
/// keyboard. Whether it was waiting for a click, which it no longer is.
/// Inside a take of ours it
/// changes nothing: that is the take's own activation, and Windows may
/// activate the prompt within its own thread even as it refuses the take,
/// so the take's result alone decides.
pub(crate) fn activated(s: &State) -> bool {
    if s.taking.load(Ordering::SeqCst) {
        return false;
    }
    s.took.store(true, Ordering::SeqCst);
    s.waiting.swap(false, Ordering::SeqCst)
}

/// Close the prompt (`why`), if open `open` is still the one showing,
/// posted to the main thread (see [`post`]).
pub fn close_later(app: &AppHandle, why: Why, open: u64) {
    post(app, move |h| close(h, why, open));
}

/// `close_on` on the real prompt, and its log line. Main thread.
fn close(app: &AppHandle, why: Why, open: u64) {
    let Some(w) = app.get_webview_window(LABEL) else {
        STATE.open.store(false, Ordering::SeqCst);
        STATE.back_to.store(0, Ordering::SeqCst);
        return;
    };
    log_closed(why, close_on(&STATE, &keyboard::Desk::of(app), &Live { app, window: &w }, why, open));
}

/// What a close did, for the log, which never carries what was typed.
fn log_closed(why: Why, closed: Closed) {
    let says = why.says();
    match closed {
        Closed::NotOpen => {
            if !matches!(why, Why::Blur | Why::Reload) {
                tracing::debug!("the prompt closed ({says}): it wasn't open");
            }
        }
        Closed::StillHasKeyboard if why == Why::Blur => {
            tracing::debug!("the prompt's page lost focus, but the prompt still has the keyboard");
        }
        Closed::StillHasKeyboard => {
            tracing::info!("the prompt stays: {says} would have closed it, but it is in front, in use, and no longer waits (its activation went unseen)");
        }
        Closed::Hidden => tracing::info!("the prompt closed ({says}): the keyboard was left where it was"),
        Closed::HandedTo(to) => tracing::info!("the prompt closed ({says}): the keyboard went back to {to:#x}"),
        Closed::Stranded(to) => {
            tracing::warn!("the prompt closed ({says}) but kept the keyboard: it went to the app in use, {to:#x}");
        }
    }
}

/// The prompt's page started loading (again): it isn't ready until it says
/// so, and an open prompt closes, since a page part-loaded takes neither
/// Enter nor Esc.
pub fn on_page_started(app: &AppHandle) {
    STATE.ready.store(false, Ordering::SeqCst);
    close_later(app, Why::Reload, opens());
}

/// The prompt window's events (the caller checks the label). Main thread.
pub fn on_window_event(app: &AppHandle, event: &WindowEvent) {
    match event {
        // Alt+F4. A declared window is hidden, never destroyed: it is
        // shown again for the next question.
        WindowEvent::CloseRequested { api, .. } => {
            api.prevent_close();
            close_later(app, Why::AltF4, opens());
        }
        WindowEvent::Focused(false) => close_later(app, Why::Blur, opens()),
        WindowEvent::Destroyed => {
            STATE.hwnd.store(0, Ordering::SeqCst);
            STATE.ready.store(false, Ordering::SeqCst);
            STATE.open.store(false, Ordering::SeqCst);
            STATE.back_to.store(0, Ordering::SeqCst);
            tracing::debug!("the prompt window is gone");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::sync::atomic::Ordering;

    use super::{
        activated, close_on, count_open, hotkey_request, open_on, pressed_into_on, prompt_rect, saw_press_on, settle_count,
        uncount_open, waiting_click_open_of, waiting_open_of, Closed, OpenKind, Opened, Opener, Place, PromptDesk, Refusal,
        State, WhenOpen, Why,
    };
    use crate::keyboard::Keyboard;
    use crate::windows_ext::Bounds;

    const PROMPT: isize = 5;
    /// A window the prompt owns (a popup of its page's).
    const POPUP: isize = 6;
    /// The tray's hidden window: ours, and never able to take the keyboard.
    const TRAY: isize = 9;
    /// The control window.
    const SETTINGS: isize = 30;
    const NOTEPAD: isize = 42;
    /// The app in use, which `next_app_window` answers.
    const BENEATH: isize = 77;

    fn name(h: isize) -> &'static str {
        match h {
            PROMPT => "PROMPT",
            POPUP => "POPUP",
            TRAY => "TRAY",
            SETTINGS => "SETTINGS",
            NOTEPAD => "NOTEPAD",
            BENEATH => "BENEATH",
            0 => "nobody",
            _ => "?",
        }
    }

    /// His window, bottom right of a 1920x1032 work area, and the prompt's
    /// place above him.
    const HIM: Bounds = Bounds { left: 1500, top: 400, right: 1840, bottom: 1040 };
    const WORK: Bounds = Bounds { left: 0, top: 0, right: 1920, bottom: 1032 };

    fn place() -> Place {
        Place { at: prompt_rect(Some(HIM), WORK, 96), anchor: Some(HIM), work: WORK }
    }

    fn rect(b: Bounds) -> String {
        super::rect(b)
    }

    /// A desktop whose foreground moves when a window takes it, and a
    /// prompt that leaves the foreground with `after_hide` when it hides
    /// while it has it (Windows' own choice; nobody, unless a test says).
    /// `watch`, when set, is the state whose counts a take of the prompt's
    /// records in `seen`; a granted take then activates the prompt inside
    /// it, as Windows does, and `woke` records whether that read as the end
    /// of a wait.
    struct Fake {
        fg: Cell<isize>,
        takeable: Vec<isize>,
        refuses: Vec<isize>,
        after_hide: isize,
        arranging: bool,
        place: Option<Place>,
        blocked: Option<String>,
        watch: Option<Rc<State>>,
        seen: RefCell<Vec<(u64, u64)>>,
        woke: RefCell<Vec<bool>>,
        /// With `watch`: a press into the prompt lands during each take of
        /// it, and what the hook's count gave is recorded.
        press_in_take: bool,
        pressed: RefCell<Vec<u64>>,
        /// With `watch`: Windows activates the prompt within its own thread
        /// even as it refuses the take (`woke` records it too).
        activates_refused: bool,
        calls: RefCell<Vec<String>>,
    }

    fn fake(fg: isize, takeable: &[isize]) -> Fake {
        Fake {
            fg: Cell::new(fg),
            takeable: takeable.to_vec(),
            refuses: vec![],
            after_hide: 0,
            arranging: false,
            place: Some(place()),
            blocked: None,
            watch: None,
            seen: RefCell::default(),
            woke: RefCell::default(),
            press_in_take: false,
            pressed: RefCell::default(),
            activates_refused: false,
            calls: RefCell::default(),
        }
    }

    impl Fake {
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }

    impl Keyboard for Fake {
        fn foreground(&self) -> isize {
            self.fg.get()
        }
        fn try_foreground(&self, hwnd: isize) -> bool {
            self.calls.borrow_mut().push(format!("foreground {}", name(hwnd)));
            if let (PROMPT, Some(s)) = (hwnd, &self.watch) {
                self.seen.borrow_mut().push((s.opens.load(Ordering::SeqCst), s.takes.load(Ordering::SeqCst)));
                if self.press_in_take {
                    self.pressed.borrow_mut().push(pressed_into_on(s));
                }
            }
            let ok = self.takeable.contains(&hwnd) && !self.refuses.contains(&hwnd);
            if ok {
                self.fg.set(hwnd);
            }
            if let (PROMPT, Some(s)) = (hwnd, &self.watch) {
                if ok || self.activates_refused {
                    self.woke.borrow_mut().push(activated(s));
                }
            }
            ok
        }
        fn can_take_foreground(&self, hwnd: isize) -> bool {
            self.takeable.contains(&hwnd)
        }
        fn owned_by_us(&self, hwnd: isize) -> bool {
            [PROMPT, POPUP, TRAY, SETTINGS].contains(&hwnd)
        }
        fn settings_window(&self) -> isize {
            SETTINGS
        }
        fn next_app_window(&self, skip: isize, also: isize) -> isize {
            assert_eq!((skip, also), (PROMPT, SETTINGS), "passes over the prompt, counts the control window");
            BENEATH
        }
    }

    impl PromptDesk for Fake {
        fn hide(&self) {
            self.calls.borrow_mut().push("hide".into());
            if [PROMPT, POPUP].contains(&self.fg.get()) {
                self.fg.set(self.after_hide);
            }
        }
        fn root_owner(&self, hwnd: isize) -> isize {
            if hwnd == POPUP {
                PROMPT
            } else {
                hwnd
            }
        }
        fn arranging(&self) -> bool {
            self.arranging
        }
        fn where_to(&self) -> Option<Place> {
            self.place
        }
        fn blocked(&self, _at: Bounds) -> Option<String> {
            self.blocked.clone()
        }
        fn place(&self, at: Bounds) {
            self.calls.borrow_mut().push(format!("place {}", rect(at)));
        }
        fn show(&self) {
            self.calls.borrow_mut().push("show".into());
        }
        fn tell_waiting(&self, on: bool) {
            self.calls.borrow_mut().push(format!("waiting {on}"));
        }
    }

    /// The prompt at rest, its page ready.
    fn ready() -> State {
        let s = State::new();
        s.hwnd.store(PROMPT, Ordering::SeqCst);
        s.ready.store(true, Ordering::SeqCst);
        s
    }

    /// The prompt as an open leaves it: open, its first open, having saved
    /// `back_to`.
    fn opened(back_to: isize) -> State {
        let s = State::new();
        s.hwnd.store(PROMPT, Ordering::SeqCst);
        s.open.store(true, Ordering::SeqCst);
        s.back_to.store(back_to, Ordering::SeqCst);
        s.opens.store(1, Ordering::SeqCst);
        s
    }

    fn is_closed(s: &State) -> bool {
        !s.open.load(Ordering::SeqCst) && s.back_to.load(Ordering::SeqCst) == 0
    }

    /// The keyboard goes back while the prompt still has it, then it hides.
    #[test]
    fn esc_hands_back_then_hides() {
        let s = opened(NOTEPAD);
        let d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"]);
        assert!(is_closed(&s));
    }

    /// Every way out but the owner going elsewhere hands the keyboard back.
    #[test]
    fn every_way_out_but_blur_hands_back() {
        for why in [Why::Sent, Why::Esc, Why::AltF4, Why::Reload] {
            let s = opened(NOTEPAD);
            let d = fake(PROMPT, &[NOTEPAD, BENEATH]);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::HandedTo(NOTEPAD), "{why:?}");
            assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"], "{why:?}");
        }
    }

    #[test]
    fn a_saved_window_gone_gives_way_to_the_app_beneath() {
        let s = opened(NOTEPAD);
        let d = fake(PROMPT, &[BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(BENEATH));
        assert_eq!(d.calls(), ["foreground BENEATH", "hide"]);
    }

    #[test]
    fn opened_from_settings_returns_to_settings() {
        let s = opened(SETTINGS);
        let d = fake(PROMPT, &[SETTINGS, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Sent, 1), Closed::HandedTo(SETTINGS));
        assert_eq!(d.calls(), ["foreground SETTINGS", "hide"]);
    }

    /// Nobody saved: the app in use.
    #[test]
    fn nobody_saved_gives_the_app_in_use() {
        let s = opened(0);
        let d = fake(PROMPT, &[BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(BENEATH));
        assert_eq!(d.calls(), ["foreground BENEATH", "hide"]);
    }

    /// A window the prompt owns in front is the prompt's own keyboard.
    #[test]
    fn a_popup_of_the_prompts_in_front_hands_back_too() {
        let s = opened(NOTEPAD);
        let d = fake(POPUP, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"]);
    }

    #[test]
    fn the_tray_window_in_front_hands_to_the_app_beneath() {
        let s = opened(NOTEPAD);
        let d = fake(TRAY, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Reload, 1), Closed::HandedTo(BENEATH));
        assert_eq!(d.calls(), ["foreground BENEATH", "hide"]);
    }

    #[test]
    fn closing_twice_touches_nothing_the_second_time() {
        let s = opened(NOTEPAD);
        let d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Sent, 1), Closed::HandedTo(NOTEPAD));
        for why in [Why::Sent, Why::Esc, Why::Blur, Why::AltF4] {
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::NotOpen, "{why:?}");
        }
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"]);
    }

    /// The keyboard is where the owner sent it: only hide.
    #[test]
    fn blur_to_another_app_only_hides() {
        let s = opened(SETTINGS);
        let d = fake(NOTEPAD, &[NOTEPAD, SETTINGS, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Blur, 1), Closed::Hidden);
        assert_eq!(d.calls(), ["hide"]);
        assert!(is_closed(&s));
    }

    /// The tray's menu opened over the prompt: it only hides, so the menu
    /// keeps the keyboard and stays open.
    #[test]
    fn a_blur_to_the_trays_menu_only_hides() {
        let s = opened(NOTEPAD);
        let d = fake(TRAY, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Blur, 1), Closed::Hidden);
        assert_eq!(d.calls(), ["hide"]);
        assert_eq!(d.fg.get(), TRAY);
    }

    /// The webview lost focus inside the prompt (to its own window, or a
    /// popup of its page's): nothing changes.
    #[test]
    fn a_spurious_blur_changes_nothing() {
        for fg in [PROMPT, POPUP] {
            let s = opened(NOTEPAD);
            let d = fake(fg, &[NOTEPAD, BENEATH]);
            assert_eq!(close_on(&s, &d, &d, Why::Blur, 1), Closed::StillHasKeyboard, "{}", name(fg));
            assert!(d.calls().is_empty(), "{}", name(fg));
            assert!(s.open.load(Ordering::SeqCst));
            assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD, "still saved for the real close");
        }
    }

    #[test]
    fn closing_while_someone_else_has_the_keyboard_only_hides() {
        let s = opened(NOTEPAD);
        let d = fake(SETTINGS, &[NOTEPAD, SETTINGS, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::Hidden);
        assert_eq!(d.calls(), ["hide"]);
        assert!(is_closed(&s));
    }

    /// Windows refused the hand-back and left the hidden prompt in front:
    /// the app in use is given the keyboard.
    #[test]
    fn a_hidden_prompt_never_keeps_the_keyboard() {
        let s = opened(NOTEPAD);
        let mut d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        d.refuses = vec![NOTEPAD];
        d.after_hide = PROMPT;
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::Stranded(BENEATH));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide", "foreground BENEATH"]);
        assert_eq!(d.fg.get(), BENEATH);
    }

    /// The hand-back refused, but Windows moved the keyboard on as the
    /// prompt hid: left there.
    #[test]
    fn a_refused_hand_back_is_left_to_windows() {
        let s = opened(NOTEPAD);
        let mut d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        d.refuses = vec![NOTEPAD];
        d.after_hide = NOTEPAD;
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::Hidden);
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"]);
    }

    #[test]
    fn nothing_when_never_opened() {
        let s = State::new();
        s.hwnd.store(PROMPT, Ordering::SeqCst);
        let d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        for why in [Why::Sent, Why::Esc, Why::AltF4, Why::Blur, Why::Reload] {
            assert_eq!(close_on(&s, &d, &d, why, 0), Closed::NotOpen, "{why:?}");
        }
        assert!(d.calls().is_empty());
    }

    /// The hotkey's open saves who had the keyboard, places the prompt,
    /// shows it, then takes the keyboard, counting the open (and the take)
    /// before it.
    #[test]
    fn opening_saves_who_had_the_keyboard_then_takes_it() {
        let s = Rc::new(ready());
        let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT, BENEATH]);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Shown(place()));
        assert_eq!(d.calls(), [format!("place {}", rect(place().at)), "show".into(), "foreground PROMPT".into()]);
        assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD);
        assert!(s.open.load(Ordering::SeqCst));
        assert_eq!(*d.seen.borrow(), [(1, 1)], "the open and the take counted before the take");
        assert_eq!((s.opens.load(Ordering::SeqCst), s.takes.load(Ordering::SeqCst)), (1, 1));
    }

    /// From the tray's window, or with nobody in front, the app beneath is
    /// saved, as Arrange does.
    #[test]
    fn opening_from_the_tray_window_or_nothing_saves_the_app_beneath() {
        for fg in [TRAY, 0] {
            let s = ready();
            let d = fake(fg, &[PROMPT, BENEATH]);
            assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Shown(place()), "{}", name(fg));
            assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH, "{}", name(fg));
            // Had, by the open's own word (no activation seen here).
            assert!(s.took.load(Ordering::SeqCst), "{}", name(fg));
        }
        // Opened from Settings, the keyboard goes back to Settings.
        let s = ready();
        let d = fake(SETTINGS, &[SETTINGS, PROMPT, BENEATH]);
        open_on(&s, &d, &d, Opener::Hotkey);
        assert_eq!(s.back_to.load(Ordering::SeqCst), SETTINGS);
    }

    /// Windows refused the keyboard: the prompt hides again at once, and
    /// the open is undone (but its number stays used, so no close asked in
    /// it can close a later one).
    #[test]
    fn a_refused_take_hides_it_again() {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert_eq!(d.calls().last().map(String::as_str), Some("hide"));
        assert_eq!(d.calls()[1..], ["show".to_owned(), "foreground PROMPT".into(), "hide".into()]);
        assert!(!s.open.load(Ordering::SeqCst));
        assert_eq!(s.back_to.load(Ordering::SeqCst), 0);
        assert_eq!((s.opens.load(Ordering::SeqCst), s.takes.load(Ordering::SeqCst)), (1, 0));
    }

    /// Nothing opens over the lock screen or a fullscreen app, and nothing
    /// is touched.
    #[test]
    fn nothing_opens_over_a_fullscreen_app_or_the_lock_screen() {
        for by in ["the lock screen", "Game (pid 1)", "presentation mode"] {
            let s = ready();
            let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
            d.blocked = Some(by.into());
            assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::Blocked(by.into())));
            assert!(d.calls().is_empty(), "{by}");
            assert!(!s.open.load(Ordering::SeqCst));
            assert_eq!((s.back_to.load(Ordering::SeqCst), s.opens.load(Ordering::SeqCst)), (0, 0));
        }
    }

    /// Nor while he is being arranged, before its page is ready, with no
    /// window, or with no monitor to stand on.
    #[test]
    fn nothing_opens_while_he_is_arranged_or_before_its_page_is_ready() {
        let s = ready();
        let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        d.arranging = true;
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::Arranging));
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        s.ready.store(false, Ordering::SeqCst);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotReady));
        let none = State::new();
        none.ready.store(true, Ordering::SeqCst);
        assert_eq!(open_on(&none, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NoWindow));
        let s = ready();
        let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        d.place = None;
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NoPlace));
        assert!(d.calls().is_empty());
        assert_eq!(s.opens.load(Ordering::SeqCst), 0);
    }

    /// Open, its key closes it, handing the keyboard back.
    #[test]
    fn the_key_again_closes_it_with_the_hand_back() {
        let s = opened(NOTEPAD);
        s.ready.store(true, Ordering::SeqCst);
        let d = fake(PROMPT, &[NOTEPAD, PROMPT, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Closed(Closed::HandedTo(NOTEPAD)));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide"]);
        assert!(!s.open.load(Ordering::SeqCst));
    }

    /// A pretend open stands off screen at its size, never asks for the
    /// keyboard, saves nobody, and isn't counted as a take; another pretend
    /// while it is open is refused.
    #[test]
    fn a_pretend_open_stands_off_screen_and_never_takes_the_keyboard() {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        assert_eq!(open_on(&s, &d, &d, Opener::Pretend), Opened::Shown(place()));
        let at = place().at;
        let (x, y) = crate::overlay_windows::PARK_AT;
        let off = Bounds { left: x, top: y, right: x + at.width(), bottom: y + at.height() };
        assert_eq!(d.calls(), [format!("place {}", rect(off)), "show".into()]);
        assert_eq!(s.back_to.load(Ordering::SeqCst), 0);
        assert_eq!((s.opens.load(Ordering::SeqCst), s.takes.load(Ordering::SeqCst)), (1, 0));
        assert_eq!(open_on(&s, &d, &d, Opener::Pretend), Opened::Refused(Refusal::AlreadyOpen));
    }

    /// Only a press of the prompt's own key opens it.
    #[test]
    fn only_presses_of_the_prompt_key_open_it() {
        assert!(hotkey_request(crate::hotkeys::PROMPT, true));
        assert!(!hotkey_request(crate::hotkeys::PROMPT, false), "its release");
        assert!(!hotkey_request(crate::hotkeys::PARK, true));
        assert!(!hotkey_request(crate::overlay_machine::HOTKEY_ARRANGE, true));
    }

    /// The size the prompt is placed at is the one tauri.conf.json declares.
    #[test]
    fn the_size_is_the_declared_one() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let w = conf["app"]["windows"].as_array().unwrap().iter().find(|w| w["label"] == super::LABEL).unwrap();
        assert_eq!((w["width"].as_f64(), w["height"].as_f64()), (Some(super::SIZE.0), Some(super::SIZE.1)));
    }

    /// The prompt opens inside `WM_HOTKEY`, under the global-shortcut
    /// plugin's lock: it must never call the plugin.
    #[test]
    fn the_prompt_never_calls_the_hotkey_plugin() {
        let all = include_str!("prompt.rs");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        assert!(!src.contains(concat!("global_", "shortcut")), "prompt.rs reaches the hotkey plugin");
    }

    /// The gate's lock is the cover watch's, checked with Windows when it
    /// says locked (`cover::lock_now`): never the cache alone.
    #[test]
    fn the_gate_checks_a_cached_lock() {
        let all = include_str!("prompt.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let at = src.find("fn blocked(&self, at: Bounds) -> Option<String> {").expect("the live desk's gate");
        let body = &src[at..at + src[at..].find("\n    }\n").unwrap()];
        assert!(body.contains("cover::covered_by(cover::lock_now(cover::locked(), windows_ext::session_locked), "));
        assert!(body.contains("cover::busy_now(cover::busy(), windows_ext::busy_now)"), "and a busy state too");
    }

    /// Where it stands: above him, centred on him, at his monitor's DPI;
    /// below him when there's no room above; shrunk and kept inside a small
    /// work area; a third of the way down with no anchor.
    #[test]
    fn prompt_rect_table() {
        let b = |l, t, r, bo| Bounds { left: l, top: t, right: r, bottom: bo };
        // 96 dpi, him at the bottom right: above him, a 12 px gap.
        let at = prompt_rect(Some(HIM), WORK, 96);
        // Centred on him it would run past the right edge: kept inside it.
        assert_eq!(at, b(1920 - 12 - 640, 268, 1920 - 12, 388));
        let mid = b(600, 500, 940, 1000);
        assert_eq!(prompt_rect(Some(mid), WORK, 96), b(450, 368, 1090, 488));
        // 144 and 192 dpi: the size and the gap scale.
        let big = b(0, 0, 3840, 2160);
        let him = b(1500, 1000, 2180, 2000);
        assert_eq!(prompt_rect(Some(him), big, 144), b(1360, 802, 2320, 982));
        assert_eq!(prompt_rect(Some(him), big, 192), b(1200, 736, 2480, 976));
        // Near the top: below him.
        let high = b(600, 40, 940, 400);
        assert_eq!(prompt_rect(Some(high), WORK, 96), b(450, 412, 1090, 532));
        // Exactly room above (its top on the work area's gap): above.
        let edge = b(600, 12 + 120 + 12, 940, 500);
        assert_eq!(prompt_rect(Some(edge), WORK, 96).top, 12);
        // One pixel less: below.
        let edge = b(600, 12 + 120 + 11, 940, 500);
        assert_eq!(prompt_rect(Some(edge), WORK, 96).top, 500 + 12);
        // No room above or below: kept inside, a gap from the bottom.
        let tall = b(600, 20, 940, 1030);
        assert_eq!(prompt_rect(Some(tall), WORK, 96), b(450, 1032 - 12 - 120, 1090, 1032 - 12));
        // No anchor: centred, its middle a third of the way down.
        assert_eq!(prompt_rect(None, WORK, 96), b(640, 344 - 60, 1280, 344 + 60));
        // A work area narrower than the prompt: shrunk to fit, inside it.
        let small = b(0, 0, 600, 400);
        assert_eq!(prompt_rect(None, small, 96), b(12, 133 - 60, 588, 133 + 60));
        // A monitor left of and above the primary.
        let left = b(-1920, -300, 0, 780);
        let there = prompt_rect(Some(b(-800, 100, -460, 740)), left, 96);
        assert_eq!(there, b(-950, -32, -310, 88));
        // A DPI below 48 is taken as 48; a work area too small for the gaps
        // still answers, never panicking.
        assert_eq!(prompt_rect(None, WORK, 10), prompt_rect(None, WORK, 48));
        let _ = prompt_rect(Some(HIM), b(0, 0, 10, 10), 96);
    }

    #[test]
    fn the_openers_table() {
        use OpenKind::{Holding, Pretend as Pretending, Waiting};
        use WhenOpen::{Close, Refuse, Take};
        for (how, on_screen, may_wait) in [
            (Opener::Hotkey, true, false),
            (Opener::Click, true, true),
            (Opener::Menu, true, false),
            (Opener::Pretend, false, false),
            (Opener::PretendWait, false, true),
        ] {
            assert_eq!((how.on_screen(), how.counts(), how.may_wait()), (on_screen, on_screen, may_wait), "{how:?}");
        }
        for (how, when) in [
            (Opener::Hotkey, [Close, Take, Close]),
            (Opener::Click, [Close, Close, Close]),
            (Opener::Menu, [Refuse, Take, Refuse]),
            (Opener::Pretend, [Refuse, Refuse, Refuse]),
            (Opener::PretendWait, [Refuse, Refuse, Refuse]),
        ] {
            assert_eq!([Holding, Waiting, Pretending].map(|k| how.when_open(k)), when, "{how:?}");
        }
        assert_eq!(
            [Opener::Hotkey, Opener::Click, Opener::Menu, Opener::Pretend, Opener::PretendWait].map(Opener::says),
            ["hotkey", "a double-click on him", "his menu", "pretend", "pretend double-click"]
        );
        // A gesture's close says so; the rest are its key's.
        assert_eq!(
            [Opener::Hotkey, Opener::Click, Opener::Menu, Opener::Pretend, Opener::PretendWait].map(Opener::closes_as),
            [Why::Toggle, Why::Gesture, Why::Toggle, Why::Toggle, Why::Toggle]
        );
        assert_eq!(
            [Why::Gesture, Why::Elsewhere, Why::Chose, Why::Unanswered, Why::HookOut].map(Why::says),
            ["a double-click on him again", "a click elsewhere", "another choice from his menu", "nobody clicked it", "the mouse hook went out"]
        );
        assert_eq!(super::WAIT_FOR_CLICK, std::time::Duration::from_secs(20));
    }

    fn takes(s: &State) -> u64 {
        s.takes.load(Ordering::SeqCst)
    }

    /// A gesture's open the keyboard is given to is the hotkey's open, the
    /// page told it may wait before it shows and told it doesn't once it has
    /// the keyboard.
    #[test]
    fn a_click_open_granted_is_the_hotkeys_open() {
        let s = Rc::new(ready());
        let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT, BENEATH]);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Shown(place()));
        let at = format!("place {}", rect(place().at));
        assert_eq!(d.calls(), [at, "waiting true".into(), "show".into(), "foreground PROMPT".into(), "waiting false".into()]);
        assert_eq!(*d.seen.borrow(), [(1, 1)], "counted before the take");
        assert_eq!(takes(&s), 1);
        assert!(s.took.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
        assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD);
    }

    /// Refused, it stays shown and waits, and isn't the owner's open (yet).
    #[test]
    fn a_click_open_refused_waits_shown() {
        let s = Rc::new(ready());
        let mut d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        let at = format!("place {}", rect(place().at));
        assert_eq!(d.calls(), [at, "waiting true".into(), "show".into(), "foreground PROMPT".into()], "never hidden");
        assert_eq!(*d.seen.borrow(), [(1, 1)], "counted for the take");
        assert_eq!(takes(&s), 0, "given back: not the owner's until clicked into");
        assert!(s.open.load(Ordering::SeqCst) && s.waiting.load(Ordering::SeqCst));
        assert!(!s.took.load(Ordering::SeqCst));
        assert_eq!(s.credit.load(Ordering::SeqCst), 0, "unclaimed");
    }

    /// The hotkey's open refused still hides at once: only a gesture waits.
    #[test]
    fn only_a_click_may_wait() {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert!(!d.calls().iter().any(|c| c.starts_with("waiting")));
        assert!(!s.open.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
    }

    fn waiting_click() -> (State, Fake) {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        d.calls.borrow_mut().clear();
        (s, d)
    }

    /// A click elsewhere closes it: hidden, the page told, nothing handed
    /// back (the owner's app still has the keyboard), nothing counted.
    #[test]
    fn a_waiting_prompt_closed_elsewhere_gives_nothing_back() {
        let (s, d) = waiting_click();
        assert_eq!(close_on(&s, &d, &d, Why::Elsewhere, 1), Closed::Hidden);
        assert_eq!(d.calls(), ["hide", "waiting false"]);
        assert!(is_closed(&s) && !s.waiting.load(Ordering::SeqCst));
        assert_eq!(takes(&s), 0);
        // Unanswered, another choice from his menu, or the hook going out,
        // the same.
        for why in [Why::Unanswered, Why::Chose, Why::HookOut] {
            let (s, d) = waiting_click();
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::Hidden, "{why:?}");
            assert_eq!(d.calls(), ["hide", "waiting false"], "{why:?}");
        }
    }

    /// Clicked into, it is an ordinary open: Esc hands the keyboard back to
    /// the app it came from.
    #[test]
    fn clicked_into_it_is_an_ordinary_open() {
        let (s, d) = waiting_click();
        d.fg.set(PROMPT);
        assert!(activated(&s), "it was waiting");
        assert!(!s.waiting.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst));
        assert!(!activated(&s), "waiting no more");
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide", "waiting false"]);
    }

    /// Once clicked into, or its key took the keyboard, a click elsewhere or
    /// none in time never closes it.
    #[test]
    fn elsewhere_and_unanswered_never_close_a_prompt_in_use() {
        for why in [Why::Elsewhere, Why::Chose, Why::Unanswered, Why::HookOut] {
            let (s, d) = waiting_click();
            activated(&s);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::NotOpen, "{why:?}");
            assert!(d.calls().is_empty() && s.open.load(Ordering::SeqCst));
            // Nor an open that never waited.
            let s = opened(NOTEPAD);
            let d = fake(PROMPT, &[NOTEPAD]);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::NotOpen, "{why:?}");
            assert!(d.calls().is_empty());
        }
    }

    /// An unanswered close from an earlier wait never closes a later one.
    #[test]
    fn an_old_unanswered_close_never_closes_a_later_open() {
        let (s, d) = waiting_click();
        assert_eq!(close_on(&s, &d, &d, Why::Elsewhere, 1), Closed::Hidden);
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::NotOpen, "the first wait's timer");
        assert!(s.open.load(Ordering::SeqCst) && s.waiting.load(Ordering::SeqCst));
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 2), Closed::Hidden);
    }

    /// Its key on a waiting prompt takes the keyboard into it, counted
    /// first; nothing closes.
    #[test]
    fn the_key_on_a_waiting_prompt_takes_the_keyboard() {
        let (s, mut d) = waiting_click();
        d.takeable.push(PROMPT);
        let s = Rc::new(s);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
        assert_eq!(d.calls(), ["foreground PROMPT", "waiting false"]);
        assert_eq!(*d.seen.borrow(), [(1, 1)], "counted before the take");
        assert_eq!(*d.woke.borrow(), [false], "the take's own activation is no click into it");
        assert_eq!(takes(&s), 1);
        assert!(s.open.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst));
        // And now it closes as any open the key took: its key again.
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Closed(Closed::HandedTo(NOTEPAD)));
        assert_eq!(takes(&s), 1, "the owner's open, kept");
    }

    #[test]
    fn the_key_on_a_waiting_prompt_refused_leaves_it_waiting() {
        let (s, d) = waiting_click();
        d.fg.set(BENEATH);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert_eq!(d.calls(), ["foreground PROMPT"], "nothing hidden, the page still waits");
        assert!(s.open.load(Ordering::SeqCst) && s.waiting.load(Ordering::SeqCst));
        assert_eq!(takes(&s), 0, "the count given back");
        assert_eq!(s.credit.load(Ordering::SeqCst), 0, "unclaimed");
        assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD, "its open's save kept");
    }

    /// The owner moved on during the wait (Alt+Tab, no click): its key's
    /// take saves who has the keyboard now, and the close hands it back
    /// there, never to the app they left.
    #[test]
    fn the_key_on_a_waiting_prompt_saves_who_has_the_keyboard_now() {
        let (s, mut d) = waiting_click();
        d.takeable.push(PROMPT);
        d.fg.set(BENEATH);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
        assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH);
        assert!(s.took.load(Ordering::SeqCst), "had, by the take's own word (no activation seen here)");
        assert!(!s.waiting.load(Ordering::SeqCst), "no longer waiting, by the take's own word");
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(BENEATH));
        assert_eq!(takes(&s), 1, "the owner's open, kept");
    }

    /// Clicked into while its activation went unseen (the watch didn't go
    /// in), it still reads as waiting; but it has the keyboard, so its close
    /// hands it back as any open's does.
    #[test]
    fn a_waiting_prompt_that_has_the_keyboard_after_all_hands_it_back() {
        let (s, d) = waiting_click();
        assert_eq!(pressed_into_on(&s), 1, "the hook's count of the press");
        d.fg.set(PROMPT);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!(d.calls(), ["foreground NOTEPAD", "hide", "waiting false"]);
        assert_eq!(takes(&s), 1, "it had the keyboard: the owner's, kept");
    }

    /// The tray icon's menu open over its hidden window while the prompt
    /// waits: the wait's close only hides, so the menu keeps the keyboard and
    /// stays open (a prompt that had the keyboard hands it to the app
    /// beneath; one still waiting never had it).
    #[test]
    fn a_waiting_prompt_closed_under_the_trays_menu_only_hides() {
        for why in [Why::Unanswered, Why::Elsewhere, Why::Toggle] {
            let (s, d) = waiting_click();
            d.fg.set(TRAY);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::Hidden, "{why:?}");
            assert_eq!(d.calls(), ["hide", "waiting false"], "{why:?}");
            assert_eq!(d.fg.get(), TRAY, "{why:?}");
        }
        // Its key on a pretend wait closes it the same way.
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Waiting(place()));
        d.fg.set(TRAY);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Closed(Closed::Hidden));
        assert_eq!(d.fg.get(), TRAY);
    }

    /// The gesture again on any open prompt closes it, with the hand-back.
    #[test]
    fn a_click_while_open_closes_it() {
        let (s, d) = waiting_click();
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Closed(Closed::Hidden));
        assert!(!s.open.load(Ordering::SeqCst));
        let s = opened(NOTEPAD);
        s.ready.store(true, Ordering::SeqCst);
        let d = fake(PROMPT, &[NOTEPAD, PROMPT]);
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Closed(Closed::HandedTo(NOTEPAD)));
    }

    /// A pretend wait stands off screen, never asks for the keyboard, never
    /// counts, and waits; its key closes it (never takes); another pretend
    /// is refused.
    #[test]
    fn a_pretend_wait_never_tries_the_take_and_never_counts() {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Waiting(place()));
        let at = place().at;
        let (x, y) = crate::overlay_windows::PARK_AT;
        let off = Bounds { left: x, top: y, right: x + at.width(), bottom: y + at.height() };
        assert_eq!(d.calls(), [format!("place {}", rect(off)), "waiting true".into(), "show".into()]);
        assert!(s.waiting.load(Ordering::SeqCst) && s.pretend.load(Ordering::SeqCst));
        assert_eq!((s.opens.load(Ordering::SeqCst), takes(&s)), (1, 0));
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Refused(Refusal::AlreadyOpen));
        assert_eq!(open_on(&s, &d, &d, Opener::Pretend), Opened::Refused(Refusal::AlreadyOpen));
        assert!(matches!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Closed(_)), "its key closes a pretend");
        assert!(!d.calls().iter().any(|c| c == "foreground PROMPT"));
    }

    /// A new open that waits is timed, once: the sleeper closes it unanswered
    /// after `WAIT_FOR_CLICK`, carrying its open's number. And Windows'
    /// activation of the prompt reaches `activated` from its window
    /// procedure, which only stores and posts.
    #[test]
    fn a_waiting_open_is_timed_and_its_activation_watched() {
        let all = include_str!("prompt.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let at = src.find("pub fn toggle(").unwrap();
        let toggle = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(toggle.contains("let waits = matches!(opened, Opened::Waiting(_));"));
        assert!(toggle.contains("if waits {\n        unanswered_after(app, opens(), WAIT_FOR_CLICK);\n    }"));
        let at = src.find("fn unanswered_after(").unwrap();
        let sleeper = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(sleeper.find("std::thread::sleep(wait);").unwrap() < sleeper.find("close_later(&h, Why::Unanswered, open);").unwrap());
        assert!(src.contains("windows_ext::watch_activation(hwnd, activation)"));
        let at = src.find("fn activation(active: bool)").unwrap();
        let act = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(
            act.contains("        if active {\n            if activated(&STATE) {\n                post(app, no_longer_waits);\n            }\n        } else {\n            close_later(app, Why::Blur, opens());\n        }"),
            "active: stores, then posts; inactive: the posted Blur"
        );
        for inline in ["tracing::", "tell(", ".emit", "Live {", ".lock(", "|h|", "try_foreground"] {
            assert!(!act.contains(inline), "the window procedure does {inline} inline");
        }
        let at = src.find("fn no_longer_waits(h: &AppHandle)").unwrap();
        let posted = &src[at..at + src[at..].find("\n}\n").unwrap()];
        assert!(posted.contains("tell(&STATE, &Live { app: h, window: &w }, false);"), "the page told it no longer waits");
    }

    /// Every open starts afresh: a first wait clicked into leaves nothing for
    /// a second wait that nobody clicks, whose close gives back nothing it
    /// never counted.
    #[test]
    fn two_waits_in_a_row_each_start_afresh() {
        let (s, d) = waiting_click();
        assert_eq!(pressed_into_on(&s), 1, "the hook's press into it, counted");
        d.fg.set(PROMPT);
        activated(&s);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!(takes(&s), 1, "clicked into: the owner's");
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        assert!(!s.took.load(Ordering::SeqCst), "the first open's took is gone");
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 2), Closed::Hidden);
        assert_eq!(takes(&s), 1, "the second, never the owner's, never counted");
        // Counted but never taken in (a click into it that didn't land): the
        // close gives the count back.
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        assert_eq!(pressed_into_on(&s), 3);
        assert_eq!(takes(&s), 2);
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 3), Closed::Hidden);
        assert_eq!(takes(&s), 1);
    }

    /// The hook's press into a waiting prompt counts it, once, before
    /// Windows activates it; its key after that doesn't count it again; the
    /// count stays, the keyboard having gone in.
    #[test]
    fn a_press_into_a_waiting_prompt_counts_it_once() {
        let (s, mut d) = waiting_click();
        assert_eq!(takes(&s), 0);
        assert_eq!(pressed_into_on(&s), 1);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 1));
        assert_eq!(pressed_into_on(&s), 0, "counted already");
        assert_eq!(takes(&s), 1);
        // Its key, before Windows' activation came: taken in, not counted again.
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
        assert_eq!(takes(&s), 1, "one open, one count");
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 0), "the owner's open, kept");
        // Not waiting, nothing is counted.
        assert_eq!(pressed_into_on(&s), 0);
        assert_eq!(takes(&s), 1);
        // Nor once Windows activated it unasked (no press of the owner's):
        // open, no longer waiting, uncounted, and it stays uncounted.
        let (s, _d) = waiting_click();
        activated(&s);
        assert_eq!(pressed_into_on(&s), 0);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
    }

    /// A press counted for an open that has ended (it raced the close) is
    /// taken back: by the press itself when it sees the open is over, or by
    /// the next open, which still counts its own.
    #[test]
    fn a_press_racing_its_close_is_taken_back() {
        // The press saw it waiting, but the close came before its claim.
        let s = ready();
        s.waiting.store(true, Ordering::SeqCst);
        s.opens.store(3, Ordering::SeqCst);
        assert_eq!(pressed_into_on(&s), 0, "the open is over");
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
        // A claim left over from open 1 (it landed after the close took the
        // claim out): the next open takes it back and counts its own.
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT, BENEATH]);
        s.opens.store(1, Ordering::SeqCst);
        s.takes.fetch_add(1, Ordering::SeqCst);
        s.credit.store(1, Ordering::SeqCst);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Shown(place()));
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 2), "the stale count back, its own made");
        // A pretend open, which counts nothing, takes it back too, at once.
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        s.opens.store(1, Ordering::SeqCst);
        s.takes.fetch_add(1, Ordering::SeqCst);
        s.credit.store(1, Ordering::SeqCst);
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Waiting(place()));
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
        // ... and one landing after an open began is taken back as that
        // open counts itself (its key's take into a wait).
        let (s, mut d) = waiting_click();
        s.takes.fetch_add(1, Ordering::SeqCst);
        s.credit.store(7, Ordering::SeqCst);
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 1));
        // A close takes a claim that isn't its open's back too.
        let (s, d) = waiting_click();
        s.takes.fetch_add(1, Ordering::SeqCst);
        s.credit.store(7, Ordering::SeqCst);
        s.took.store(true, Ordering::SeqCst);
        d.fg.set(PROMPT);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
    }

    /// A press into the prompt while a take of it is under way: the take's
    /// own count blocks the hook's claim; refused, the take gives its count
    /// back, and the press is counted again (the owner clicked into it).
    /// Granted, one open, one count.
    #[test]
    fn a_press_during_a_refused_take_is_still_counted() {
        // Its key's take into a waiting prompt.
        let (s, mut d) = waiting_click();
        let s = Rc::new(s);
        d.watch = Some(Rc::clone(&s));
        d.press_in_take = true;
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert_eq!(*d.pressed.borrow(), [0], "the hook's claim was blocked");
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 1), "counted again");
        assert!(s.waiting.load(Ordering::SeqCst), "still waiting");
        // A gesture's own take: it waits from the take on.
        let s = Rc::new(ready());
        let mut d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        d.watch = Some(Rc::clone(&s));
        d.press_in_take = true;
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        assert_eq!(*d.pressed.borrow(), [0]);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 1));
        // Granted: the take's count, once.
        let (s, mut d) = waiting_click();
        let s = Rc::new(s);
        d.watch = Some(Rc::clone(&s));
        d.press_in_take = true;
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 1));
        assert!(!s.waiting.load(Ordering::SeqCst));
        // No press: the refused take's count goes back.
        let (s, mut d) = waiting_click();
        let s = Rc::new(s);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
    }

    /// Activation inside a take of ours is the take's: it changes nothing
    /// (neither took nor waiting, no posted word); the take's result alone
    /// decides.
    #[test]
    fn an_activation_inside_a_take_is_the_takes() {
        let (s, _d) = waiting_click();
        s.taking.store(true, Ordering::SeqCst);
        assert!(!activated(&s));
        assert!(!s.took.load(Ordering::SeqCst) && s.waiting.load(Ordering::SeqCst), "nothing changed: the take decides");
        s.taking.store(false, Ordering::SeqCst);
        assert!(activated(&s), "outside a take: the owner's click");
        assert!(s.took.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
    }

    /// Windows may activate the prompt within its own thread even as it
    /// refuses the take: the gesture's open still waits, and its key's
    /// refused take leaves it waiting, so every way out of a wait still
    /// works.
    #[test]
    fn a_refused_takes_own_activation_leaves_it_waiting() {
        for why in [Why::Unanswered, Why::Elsewhere, Why::HookOut] {
            let s = Rc::new(ready());
            let mut d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
            d.watch = Some(Rc::clone(&s));
            d.activates_refused = true;
            assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
            assert_eq!(*d.woke.borrow(), [false]);
            assert!(s.waiting.load(Ordering::SeqCst) && !s.took.load(Ordering::SeqCst), "{why:?}");
            assert_eq!(waiting_open_of(&s), 1);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::Hidden, "{why:?}");
        }
        let (s, mut d) = waiting_click();
        let s = Rc::new(s);
        d.watch = Some(Rc::clone(&s));
        d.activates_refused = true;
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Refused(Refusal::NotGiven));
        assert!(s.waiting.load(Ordering::SeqCst) && !s.took.load(Ordering::SeqCst));
        assert_eq!(takes(&s), 0);
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::Hidden);
    }

    /// A waiting prompt found in front by a close meant for a wait is in
    /// use, its activation unseen: kept, had, no longer waiting, its page
    /// told; then it closes as any open that had the keyboard.
    #[test]
    fn a_waiting_prompt_in_front_is_in_use() {
        for why in [Why::Unanswered, Why::Elsewhere, Why::HookOut] {
            let (s, d) = waiting_click();
            d.fg.set(PROMPT);
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::StillHasKeyboard, "{why:?}");
            assert_eq!(d.calls(), ["waiting false"], "{why:?}");
            assert!(s.open.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
            assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::NotOpen, "no longer waiting");
            assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        }
        // A window it owns in front, the same.
        let (s, d) = waiting_click();
        d.fg.set(POPUP);
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::StillHasKeyboard);
    }

    /// Clicked into with no activation seen, it still reads as waiting; but
    /// in front, its key closes it (it doesn't take it again), and the
    /// keyboard goes back to the app the press was made from.
    #[test]
    fn its_key_closes_a_clicked_in_prompt_whose_activation_went_unseen() {
        for front in [PROMPT, POPUP] {
            let (s, mut d) = waiting_click();
            assert_eq!(pressed_into_on(&s), 1);
            d.fg.set(BENEATH);
            assert!(saw_press_on(&s, &d, &d, 1, BENEATH), "the press, made from BENEATH");
            assert!(s.waiting.load(Ordering::SeqCst), "not yet in front");
            d.fg.set(front);
            d.takeable.push(PROMPT);
            assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Closed(Closed::HandedTo(BENEATH)), "{}", name(front));
            assert_eq!(takes(&s), 1, "the owner's open, kept");
        }
        // Not in front: its key takes the keyboard into it, as before.
        let (s, mut d) = waiting_click();
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
    }

    /// The hook's report of a press into it, read once the click has made the
    /// prompt the one in front: in use, no longer waiting, its page told.
    #[test]
    fn a_press_into_it_found_in_front_ends_the_wait() {
        let (s, d) = waiting_click();
        d.fg.set(PROMPT);
        assert!(saw_press_on(&s, &d, &d, 1, NOTEPAD));
        assert!(!s.waiting.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst));
        assert_eq!(d.calls(), ["waiting false"]);
        assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD);
        // A press made while ours or nobody had the keyboard saves nothing,
        // but the prompt in front by now still ends the wait.
        for (front, at) in [(PROMPT, PROMPT), (POPUP, POPUP), (PROMPT, TRAY), (PROMPT, 0)] {
            let (s, d) = waiting_click();
            d.fg.set(front);
            assert!(!saw_press_on(&s, &d, &d, 1, at), "{}", name(at));
            assert!(!s.waiting.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst), "{}", name(at));
            assert_eq!(d.calls(), ["waiting false"], "{}", name(at));
            assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD, "{}", name(at));
        }
        // Another open's press touches nothing.
        let (s, d) = waiting_click();
        d.fg.set(PROMPT);
        assert!(!saw_press_on(&s, &d, &d, 2, NOTEPAD));
        assert!(s.waiting.load(Ordering::SeqCst));
    }

    /// A gesture's open the keyboard is given to stops waiting by its own
    /// word, whether or not Windows' activation is seen: a click elsewhere
    /// or the timer never closes it.
    #[test]
    fn a_granted_gesture_never_waits() {
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Shown(place()));
        assert!(!s.waiting.load(Ordering::SeqCst) && s.took.load(Ordering::SeqCst));
        for why in [Why::Unanswered, Why::Elsewhere, Why::HookOut] {
            assert_eq!(close_on(&s, &d, &d, why, 1), Closed::NotOpen, "{why:?}");
        }
    }

    /// The hook's owner thread, while the hook is out, closes a real open
    /// still waiting, never a pretend one.
    #[test]
    fn the_sweep_reads_only_a_real_wait() {
        let (s, d) = waiting_click();
        assert_eq!(waiting_click_open_of(&s), 1);
        assert_eq!(close_on(&s, &d, &d, Why::HookOut, 1), Closed::Hidden);
        assert_eq!(waiting_click_open_of(&s), 0);
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD]);
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Waiting(place()));
        assert_eq!((waiting_open_of(&s), waiting_click_open_of(&s)), (1, 0));
    }

    /// The orders that hold the count's word together across threads, which
    /// no single-threaded test can tell apart: `takes` up before a claim,
    /// the press marked before its claim, the take marked around the call
    /// and its count given back before a blocked press is counted again,
    /// and a close shut and its wait over before the count is settled.
    #[test]
    fn the_counts_orders_hold() {
        let all = include_str!("prompt.rs").replace("\r\n", "\n");
        let src = &all[..all.find("#[cfg(test)]").unwrap()];
        let body = |sig: &str| {
            let at = src.find(sig).unwrap_or_else(|| panic!("no {sig}"));
            &src[at..at + src[at..].find("\n}\n").unwrap()]
        };
        let order = |f: &str, parts: &[&str]| {
            let at: Vec<usize> = parts.iter().map(|p| f.find(p).unwrap_or_else(|| panic!("no {p}"))).collect();
            assert!(at.windows(2).all(|w| w[0] < w[1]), "{parts:?}");
        };
        let pressed = body("pub(crate) fn pressed_into_on(");
        order(pressed, &[
            "if !s.waiting.load(Ordering::SeqCst) {",
            "s.pressed.store(open, Ordering::SeqCst);",
            "s.takes.fetch_add(1, Ordering::SeqCst);",
            "s.credit.compare_exchange(0, open",
            "if !s.open.load(Ordering::SeqCst) || s.opens.load(Ordering::SeqCst) != open {\n        uncount_open(s, open);",
        ]);
        let count = body("fn count_open(");
        order(count, &["s.takes.fetch_add(1, Ordering::SeqCst);", "s.credit.compare_exchange(0, open"]);
        let uncount = body("fn uncount_open(");
        order(uncount, &["s.credit.compare_exchange(open, 0", "s.takes.fetch_sub(1, Ordering::SeqCst);"]);
        let take = body("fn take(");
        order(take, &[
            "s.taking.store(true, Ordering::SeqCst);",
            "k.try_foreground(ours)",
            "s.taking.store(false, Ordering::SeqCst);",
            "uncount_open(s, open);",
            "if s.pressed.load(Ordering::SeqCst) == open {\n            count_open(s, open);",
        ]);
        let close = body("pub(crate) fn close_on(");
        order(close, &["s.open.swap(false, Ordering::SeqCst)", "let waited = s.waiting.swap(false, Ordering::SeqCst);", "settle_count(s, open, s.took.load(Ordering::SeqCst) || holds(fg));"]);
        let open = body("pub(crate) fn open_on(");
        order(open, &[
            "settle_count(s, 0, false);",
            "let open = s.opens.fetch_add(1, Ordering::SeqCst) + 1;",
            "let counted_now = how.counts() && count_open(s, open);",
            "if how.may_wait() {\n                s.waiting.store(true, Ordering::SeqCst);\n            }\n            if take(s, k, ours, open, counted_now) {",
        ]);
        assert!(body("fn take_into(").contains("if take(s, k, ours, open, counted_now) {\n        s.waiting.store(false, Ordering::SeqCst);"), "waiting through the take");
        order(body("pub(crate) fn activated("), &["if s.taking.load(Ordering::SeqCst) {\n        return false;\n    }", "s.took.store(true, Ordering::SeqCst);", "s.waiting.swap(false, Ordering::SeqCst)"]);
    }

    /// Ask Fritz… opens it as its key does: the keyboard taken, counted
    /// first; refused, it hides again (it never waits). From the tray icon's
    /// menu (its hidden window in front), the app beneath is saved; from his
    /// right-click menu (mouse-only), the app the owner is in.
    #[test]
    fn a_menu_open_takes_the_keyboard_and_counts() {
        let s = Rc::new(ready());
        let mut d = fake(NOTEPAD, &[NOTEPAD, PROMPT, BENEATH]);
        d.watch = Some(Rc::clone(&s));
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Shown(place()));
        assert_eq!(d.calls(), [format!("place {}", rect(place().at)), "show".into(), "foreground PROMPT".into()]);
        assert_eq!(*d.seen.borrow(), [(1, 1)], "counted before the take");
        assert_eq!((takes(&s), s.back_to.load(Ordering::SeqCst)), (1, NOTEPAD));
        assert!(s.took.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
        // From the tray icon's menu.
        let s = ready();
        let d = fake(TRAY, &[NOTEPAD, PROMPT, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Shown(place()));
        assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH);
        // Refused: hidden again, the count given back, never waiting.
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Refused(Refusal::NotGiven));
        assert!(d.calls().ends_with(&["foreground PROMPT".into(), "hide".into()]));
        assert!(!s.open.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
        assert_eq!(takes(&s), 0);
    }

    /// On a prompt the owner is typing in, or a pretend one, Ask Fritz…
    /// changes nothing.
    #[test]
    fn the_menu_leaves_a_prompt_in_use_alone() {
        let s = opened(NOTEPAD);
        s.ready.store(true, Ordering::SeqCst);
        let d = fake(PROMPT, &[NOTEPAD, PROMPT]);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Refused(Refusal::AlreadyOpen));
        assert!(d.calls().is_empty() && s.open.load(Ordering::SeqCst));
        let s = ready();
        let d = fake(NOTEPAD, &[NOTEPAD, PROMPT]);
        assert_eq!(open_on(&s, &d, &d, Opener::Pretend), Opened::Shown(place()));
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Refused(Refusal::AlreadyOpen));
        // A waiting prompt the owner clicked into (in front): in use.
        let (s, mut d) = waiting_click();
        d.fg.set(PROMPT);
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Refused(Refusal::AlreadyOpen));
        assert!(s.open.load(Ordering::SeqCst) && !s.waiting.load(Ordering::SeqCst));
    }

    /// On a waiting prompt, Ask Fritz… takes the keyboard into it, counted
    /// once, and saves who has the keyboard now: from his right-click menu
    /// (mouse-only), the app the owner is in; from the tray icon's menu (its
    /// hidden window in front), the app in use, as a fresh open from it does.
    #[test]
    fn the_menu_takes_the_keyboard_into_a_waiting_prompt() {
        let (s, mut d) = waiting_click();
        d.takeable.push(PROMPT);
        d.fg.set(NOTEPAD);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Taken);
        assert_eq!(takes(&s), 1);
        assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(NOTEPAD));
        let (s, mut d) = waiting_click();
        d.takeable.push(PROMPT);
        d.fg.set(TRAY);
        assert_eq!(open_on(&s, &d, &d, Opener::Menu), Opened::Taken);
        assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH, "the app in use");
    }

    /// The count's word, step by step: claimed once, given back only by
    /// its holder, settled at a close by whether the open had the keyboard.
    #[test]
    fn the_counts_word() {
        let s = State::new();
        assert!(count_open(&s, 4));
        assert!(!count_open(&s, 4), "once");
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (1, 4));
        uncount_open(&s, 5);
        assert_eq!(takes(&s), 1, "only its holder gives it back");
        uncount_open(&s, 4);
        assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (0, 0));
        uncount_open(&s, 4);
        assert_eq!(takes(&s), 0, "never twice");
        for (claim, open, had, kept) in [(4, 4, true, 1), (4, 4, false, 0), (3, 4, true, 0), (0, 4, true, 0)] {
            let s = State::new();
            if claim != 0 {
                count_open(&s, claim);
            }
            settle_count(&s, open, had);
            assert_eq!((takes(&s), s.credit.load(Ordering::SeqCst)), (kept, 0), "{claim} {open} {had}");
        }
    }

    /// The press into it tells who had the keyboard then: that is who gets
    /// it back, if the open is still showing.
    #[test]
    fn a_press_into_it_saves_who_had_the_keyboard_then() {
        let (s, d) = waiting_click();
        assert!(saw_press_on(&s, &d, &d, 1, BENEATH));
        assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH);
        d.fg.set(PROMPT);
        activated(&s);
        assert_eq!(close_on(&s, &d, &d, Why::Esc, 1), Closed::HandedTo(BENEATH));
        // An open that is over, or another one, keeps its own save.
        let (s, d) = waiting_click();
        assert!(!saw_press_on(&s, &d, &d, 2, BENEATH));
        assert_eq!(close_on(&s, &d, &d, Why::Elsewhere, 1), Closed::Hidden);
        assert!(!saw_press_on(&s, &d, &d, 1, BENEATH));
        assert_eq!(s.back_to.load(Ordering::SeqCst), 0);
        // A window of ours in front at the press (the tray's menu, the
        // prompt itself), or nobody: the save from its open stands.
        for fg in [TRAY, PROMPT, POPUP, 0] {
            let (s, d) = waiting_click();
            assert!(!saw_press_on(&s, &d, &d, 1, fg), "{}", name(fg));
            assert_eq!(s.back_to.load(Ordering::SeqCst), NOTEPAD, "{}", name(fg));
        }
        // An app's window that can't take the keyboard back (gone, or
        // hidden since): the app in use, by the hand-back's rules.
        let (s, d) = waiting_click();
        assert!(saw_press_on(&s, &d, &d, 1, 55));
        assert_eq!(s.back_to.load(Ordering::SeqCst), BENEATH);
        // Settings is a window the keyboard goes back to.
        let (s, mut d) = waiting_click();
        d.takeable.push(SETTINGS);
        assert!(saw_press_on(&s, &d, &d, 1, SETTINGS));
        assert_eq!(s.back_to.load(Ordering::SeqCst), SETTINGS);
    }

    /// The hook reads which open waits: none, or the one showing.
    #[test]
    fn the_waiting_open_is_its_number() {
        let s = ready();
        assert_eq!(waiting_open_of(&s), 0);
        let (s, d) = waiting_click();
        assert_eq!(waiting_open_of(&s), 1);
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::Hidden);
        assert_eq!(waiting_open_of(&s), 0);
    }

    /// The double-click again closes the prompt, saying so.
    #[test]
    fn the_gesture_again_closes_it_as_the_gesture() {
        let src = include_str!("prompt.rs").replace("\r\n", "\n");
        let src = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(src.contains("WhenOpen::Close => Opened::Closed(close_on(s, k, d, how.closes_as(), s.opens.load(Ordering::SeqCst))),"));
        assert!(src.contains("Opened::Closed(c) => log_closed(how.closes_as(), c),"));
        assert!(!src.contains("log_closed(Why::Toggle"));
        // The hook's ways in post to the main thread.
        for (sig, call) in [
            ("pub fn toggle_later(", "post(app, move |h| toggle(h, how));"),
            ("pub fn saw_press_later(", "if saw_press_on(&STATE, &keyboard::Desk::of(h), &Live { app: h, window: &w }, open, fg) {"),
        ] {
            let at = src.find(sig).unwrap();
            let f = &src[at..at + src[at..].find("\n}\n").unwrap()];
            assert!(f.contains("post(app, move |h|") && f.contains(call), "{sig}");
        }
    }

    /// A pretend leaves nothing for a later open: its key takes the keyboard
    /// into a later real wait, rather than closing it as a pretend.
    #[test]
    fn a_pretend_leaves_nothing_for_a_later_open() {
        let s = ready();
        let mut d = fake(NOTEPAD, &[NOTEPAD, BENEATH]);
        assert_eq!(open_on(&s, &d, &d, Opener::PretendWait), Opened::Waiting(place()));
        assert_eq!(close_on(&s, &d, &d, Why::Unanswered, 1), Closed::Hidden);
        assert_eq!(open_on(&s, &d, &d, Opener::Click), Opened::Waiting(place()));
        assert!(!s.pretend.load(Ordering::SeqCst), "the pretend is over");
        d.takeable.push(PROMPT);
        assert_eq!(open_on(&s, &d, &d, Opener::Hotkey), Opened::Taken);
    }

    /// A close asked during an earlier open never closes a later one.
    #[test]
    fn a_close_from_an_earlier_open_is_ignored() {
        let s = opened(NOTEPAD);
        s.opens.store(2, Ordering::SeqCst);
        let d = fake(PROMPT, &[NOTEPAD, BENEATH]);
        assert_eq!(close_on(&s, &d, &d, Why::Sent, 1), Closed::NotOpen);
        assert!(d.calls().is_empty());
        assert!(s.open.load(Ordering::SeqCst));
        assert_eq!(close_on(&s, &d, &d, Why::Sent, 2), Closed::HandedTo(NOTEPAD));
    }
}
