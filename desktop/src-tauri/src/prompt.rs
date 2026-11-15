//! The prompt ("Ask Fritz"): a small window, declared in tauri.conf.json and
//! hidden at rest, that takes the keyboard so the owner can ask him
//! something, and hands it back as it closes. Nothing opens it yet; this is
//! the window and every way out of it.
//!
//! Every way out is one close, which runs once however many ask for it: a
//! question taken (Enter), Esc, Alt+F4, its page loading again, or the owner
//! going to another window.
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
//! WebView2's own callbacks. Each carries the open it was asked in, so a
//! close left over from an earlier open never closes a later one.

use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::OnceLock;

use tauri::{AppHandle, Manager, WebviewWindow, WindowEvent};

use crate::keyboard::{self, Keyboard};
use crate::overlay_machine::should_return_focus;
use crate::{logs, windows_ext};

/// The window's label in tauri.conf.json.
pub const LABEL: &str = "prompt";

/// Where the prompt stands: its HWND, whether it is open, who had the
/// keyboard before it took it, and how many times it has been opened.
pub(crate) struct State {
    hwnd: AtomicIsize,
    open: AtomicBool,
    back_to: AtomicIsize,
    opens: AtomicU64,
}

impl State {
    const fn new() -> Self {
        Self {
            hwnd: AtomicIsize::new(0),
            open: AtomicBool::new(false),
            back_to: AtomicIsize::new(0),
            opens: AtomicU64::new(0),
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
}

impl Why {
    fn says(self) -> &'static str {
        match self {
            Self::Sent => "its question taken",
            Self::Esc => "Esc",
            Self::AltF4 => "Alt+F4",
            Self::Blur => "the owner went elsewhere",
            Self::Reload => "its page loaded again",
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

/// What closing does to the prompt itself. A trait so `close_on` is tested
/// without a window.
pub(crate) trait PromptWindow {
    fn hide(&self);
    /// The window at the top of `hwnd`'s owner chain.
    fn root_owner(&self, hwnd: isize) -> isize;
}

/// Close the prompt, if open `open` is still the one showing: the order the
/// module's doc gives, deciding by who has the keyboard now. Main thread.
pub(crate) fn close_on(s: &State, k: &impl Keyboard, w: &impl PromptWindow, why: Why, open: u64) -> Closed {
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

/// The prompt window, for `close_on`.
struct Live<'a>(&'a WebviewWindow);

impl PromptWindow for Live<'_> {
    fn hide(&self) {
        let _ = self.0.hide();
    }

    fn root_owner(&self, hwnd: isize) -> isize {
        windows_ext::root_owner(hwnd)
    }
}

/// How many times the prompt has been opened: what a close asked for now
/// carries, so it can't close a later open.
pub fn opens() -> u64 {
    STATE.opens.load(Ordering::SeqCst)
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

/// Close the prompt (`why`), if open `open` is still the one showing:
/// posted to the main thread through the async runtime from wherever it is
/// asked. Tauri runs a main-thread task inline when it is asked from the
/// main thread, so a close asked inside a window procedure or a WebView2
/// callback would otherwise run there.
pub fn close_later(app: &AppHandle, why: Why, open: u64) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let h = app.clone();
        let _ = app.run_on_main_thread(move || {
            logs::guard("prompt", || close(&h, why, open));
        });
    });
}

/// `close_on` on the real prompt, and its log line, which never carries what
/// was typed. Main thread.
fn close(app: &AppHandle, why: Why, open: u64) {
    let Some(w) = app.get_webview_window(LABEL) else {
        STATE.open.store(false, Ordering::SeqCst);
        STATE.back_to.store(0, Ordering::SeqCst);
        return;
    };
    let says = why.says();
    match close_on(&STATE, &keyboard::Desk::of(app), &Live(&w), why, open) {
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

/// The prompt's page started loading (again): an open prompt closes, since
/// a page part-loaded takes neither Enter nor Esc.
pub fn on_page_started(app: &AppHandle) {
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
    use std::sync::atomic::Ordering;

    use super::{close_on, Closed, PromptWindow, State, Why};
    use crate::keyboard::Keyboard;

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

    /// A desktop whose foreground moves when a window takes it, and a
    /// prompt that leaves the foreground with `after_hide` when it hides
    /// while it has it (Windows' own choice; nobody, unless a test says).
    struct Fake {
        fg: Cell<isize>,
        takeable: Vec<isize>,
        refuses: Vec<isize>,
        after_hide: isize,
        calls: RefCell<Vec<String>>,
    }

    fn fake(fg: isize, takeable: &[isize]) -> Fake {
        Fake { fg: Cell::new(fg), takeable: takeable.to_vec(), refuses: vec![], after_hide: 0, calls: RefCell::default() }
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

    impl PromptWindow for Fake {
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
