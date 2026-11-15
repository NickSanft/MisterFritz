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
//! it hides again at once. The key again closes it.
//!
//! Every way out is one close, which runs once however many ask for it: a
//! question taken (Enter), Esc, Alt+F4, its key again, its page loading
//! again, or the owner going to another window.
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

use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize, WebviewWindow, WindowEvent};

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

/// Where the prompt stands: its HWND, whether its page is ready and whether
/// it is open, who had the keyboard before it took it, how many times it has
/// been opened (each close carries the open it was asked in), and how many
/// of those opens took the keyboard (the foreground log's fresh open).
pub(crate) struct State {
    hwnd: AtomicIsize,
    ready: AtomicBool,
    open: AtomicBool,
    back_to: AtomicIsize,
    opens: AtomicU64,
    takes: AtomicU64,
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
        }
    }
}

static STATE: State = State::new();
/// The app, for the deactivation watch (a plain `fn()` in a window procedure).
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
        }
    }
}

/// What a close did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Closed {
    /// It wasn't open, or that open is over: nothing was touched.
    NotOpen,
    /// A blur, while the prompt (or a window it owns) is still in front.
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
    /// `FRITZ_PRETEND_PROMPT`, in a debug build: off screen, the keyboard
    /// untouched, and never counted as the owner's open.
    Pretend,
}

/// Why the prompt stays hidden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refusal {
    NoWindow,
    NotReady,
    /// A pretend open while it is open already.
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
    if why == Why::Blur && holds(fg) {
        return Closed::StillHasKeyboard;
    }
    if !s.open.swap(false, Ordering::SeqCst) {
        return Closed::NotOpen;
    }
    let saved = s.back_to.swap(0, Ordering::SeqCst);
    let settings = k.settings_window();
    let target = if why == Why::Blur {
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

/// Open the prompt, or close it with its key again. The gates come first and
/// touch no window; then who has the keyboard is saved, the prompt placed
/// and shown, and (the hotkey's open only) the keyboard taken, the open
/// undone if Windows refuses. Main thread.
pub(crate) fn open_on(s: &State, k: &impl Keyboard, d: &impl PromptDesk, how: Opener) -> Opened {
    let ours = s.hwnd.load(Ordering::SeqCst);
    if ours == 0 {
        return Opened::Refused(Refusal::NoWindow);
    }
    if !s.ready.load(Ordering::SeqCst) {
        return Opened::Refused(Refusal::NotReady);
    }
    if s.open.load(Ordering::SeqCst) {
        return match how {
            Opener::Hotkey => Opened::Closed(close_on(s, k, d, Why::Toggle, s.opens.load(Ordering::SeqCst))),
            Opener::Pretend => Opened::Refused(Refusal::AlreadyOpen),
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
    let hotkey = how == Opener::Hotkey;
    let back_to = if hotkey { keyboard::return_target(k.foreground(), ours, k) } else { 0 };
    s.back_to.store(back_to, Ordering::SeqCst);
    d.place(if hotkey { place.at } else { off_screen(place.at) });
    s.open.store(true, Ordering::SeqCst);
    s.opens.fetch_add(1, Ordering::SeqCst);
    // Counted before the take, so the foreground log, polling on its own
    // thread, never sees the prompt in front without it.
    if hotkey {
        s.takes.fetch_add(1, Ordering::SeqCst);
    }
    d.show();
    if hotkey && !k.try_foreground(ours) {
        s.open.store(false, Ordering::SeqCst);
        s.back_to.store(0, Ordering::SeqCst);
        s.takes.fetch_sub(1, Ordering::SeqCst);
        d.hide();
        return Opened::Refused(Refusal::NotGiven);
    }
    Opened::Shown(place)
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

/// The prompt's HWND; 0 when there is none. For the foreground log.
pub fn hwnd() -> isize {
    STATE.hwnd.load(Ordering::SeqCst)
}

/// Whether a hotkey event opens (or closes) the prompt: its key's press.
pub fn hotkey_request(id: &str, pressed: bool) -> bool {
    pressed && id == crate::hotkeys::PROMPT
}

/// Whether a pretend open is asked for: in a debug build, with
/// `FRITZ_PRETEND_PROMPT` set (`debug` is `cfg!(debug_assertions)` at the
/// one caller, so a release build never reads it).
pub fn pretend_wanted(debug: bool, env: Option<OsString>) -> bool {
    debug && env.is_some_and(|v| !v.is_empty())
}

/// Open or close the prompt, and say what happened. The hotkey's runs inline
/// in `WM_HOTKEY`. Main thread.
pub fn toggle(app: &AppHandle, how: Opener) {
    let Some(w) = app.get_webview_window(LABEL) else {
        tracing::warn!("there is no prompt window");
        return;
    };
    match open_on(&STATE, &keyboard::Desk::of(app), &Live { app, window: &w }, how) {
        Opened::Shown(p) if how == Opener::Hotkey => tracing::info!("the prompt opened (hotkey) at {}", rect(p.at)),
        Opened::Shown(p) => tracing::info!(
            "the prompt opened (pretend: off screen, keyboard untouched); it would stand at {} in the work area {}, by his window at {}",
            rect(p.at),
            rect(p.work),
            p.anchor.map_or_else(|| "(none)".to_owned(), rect)
        ),
        Opened::Closed(c) => log_closed(Why::Toggle, c),
        Opened::Refused(r @ Refusal::NotGiven) => tracing::warn!("the prompt stays hidden: {}", r.says()),
        Opened::Refused(r) => tracing::info!("the prompt stays hidden: {}", r.says()),
    }
}

/// The prompt's page says it is ready (`prompt_ready`): the hotkey may open
/// it. With `pretend`, the first time, the pretend open, posted.
pub fn page_ready(app: &AppHandle, pretend: bool) {
    if !STATE.ready.swap(true, Ordering::SeqCst) {
        tracing::info!("the prompt's page is ready");
    }
    if pretend && !PRETENDED.swap(true, Ordering::SeqCst) {
        post(app, |h| toggle(h, Opener::Pretend));
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
/// Alt-Tab (as the avatar is), and watch for the owner going elsewhere.
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
    if !windows_ext::watch_deactivation(hwnd, deactivated) {
        tracing::warn!("couldn't watch the prompt for the owner going elsewhere");
    }
}

/// Windows says the prompt is no longer the active window: from inside its
/// window procedure, so only post.
fn deactivated() {
    logs::guard("prompt", || {
        if let Some(app) = APP.get() {
            close_later(app, Why::Blur, opens());
        }
    });
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
        Closed::StillHasKeyboard => tracing::debug!("the prompt's page lost focus, but the prompt still has the keyboard"),
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
        close_on, hotkey_request, open_on, prompt_rect, Closed, Opened, Opener, Place, PromptDesk, Refusal, State, Why,
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
    /// records in `seen`.
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
            }
            let ok = self.takeable.contains(&hwnd) && !self.refuses.contains(&hwnd);
            if ok {
                self.fg.set(hwnd);
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
