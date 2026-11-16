//! His menu: a right-click on his body shows the tray's own menu, where the
//! button was released, by mouse only.
//!
//! **On the tray's own window**, through a subclass asked by a registered
//! message the mouse hook's owner thread posts. tray-icon's hidden window
//! (`tray_icon_app`: never activated, click-through, out of Alt-Tab) already
//! owns that menu, and muda's subclass on it turns a chosen item into the
//! tray's own menu event. Shown from its window procedure, the menu runs
//! from the main thread's message loop, outside tao's event handler, so the
//! shell's main-thread work goes on while it is open. No menu object is
//! built: the tray's held one is shown, so no muda id is spent.
//!
//! **Mouse only, always** (the owner's call): the foreground is never asked
//! for, so the app in front keeps the keyboard, Esc and the arrows included,
//! and nothing needs handing back. A menu shown without the foreground
//! doesn't close itself on a click elsewhere: the hook reports such a press
//! and the menu is ended (`EndMenu`, on the menu's own thread, from here);
//! failing that, it ends after `MOUSE_ONLY_FOR`; and it ends as the hook
//! goes out. An end asked for one menu never closes a later one: each of his
//! menus has its number.
//!
//! **Rebuilds wait** while any menu of Fritz's is open (his, or the tray
//! icon's own): a rebuild swaps the menu its open copy and its queued click
//! belong to. `tray::refresh` checks on the main thread, just before it
//! would rebuild; a rebuild asked for meanwhile runs once the menu has
//! closed and no click for the tray's window is still queued.

use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU32, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use tauri::AppHandle;
use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{
    EndMenu, GetMenuItemCount, IsMenu, KillTimer, PeekMessageW, PostMessageW, RegisterWindowMessageW,
    SetTimer, HMENU, MSG, PM_NOREMOVE, WM_COMMAND, WM_ENTERMENULOOP, WM_EXITMENULOOP, WM_NCDESTROY,
    WM_TIMER,
};

use crate::windows_ext;

/// How long his menu stays up without a click.
pub const MOUSE_ONLY_FOR: Duration = Duration::from_secs(20);
/// The subclass id: "FRMN".
pub(crate) const SUBCLASS_ID: usize = 0x4652_4D4E;
/// The timer id: "FRTM" (tray-icon's own timer is 6008).
pub(crate) const TIMER_ID: usize = 0x4652_544D;

/// The tray's window, and the main thread that owns it.
static TRAY: AtomicIsize = AtomicIsize::new(0);
static MAIN_TID: AtomicU32 = AtomicU32::new(0);
/// His menu's number while it is up; 0 when it isn't.
static HIS: AtomicU32 = AtomicU32::new(0);
/// The last number given.
static NUMBERED: AtomicU32 = AtomicU32::new(0);
/// Why the menu up now was ended by Fritz (`Ended`), for the log.
static ENDED: AtomicU8 = AtomicU8::new(0);
/// Whether a menu of Fritz's is open, and a rebuild waits on it.
static REBUILDS: Rebuilds = Rebuilds::new();
static MSGS: OnceLock<Msgs> = OnceLock::new();
static APP: OnceLock<AppHandle> = OnceLock::new();

/// Registered messages, so none can be taken for tray-icon's own
/// (`WM_USER_*`, 6002-6010).
#[derive(Debug, Clone, Copy)]
struct Msgs {
    at: u32,
    end: u32,
    after: u32,
    pretend: u32,
}

/// The menu loop and the rebuild waiting on it: a menu of Fritz's is open
/// (his, or the tray icon's own: `WM_ENTERMENULOOP` to `WM_EXITMENULOOP`),
/// and a rebuild was asked for meanwhile. Atomics only: the hook's
/// procedure reads `up`.
pub(crate) struct Rebuilds {
    up: AtomicBool,
    due: AtomicBool,
}

/// What a closed menu's `after` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum After {
    /// A click for the tray's window is still queued: look again after it.
    Again,
    /// Rebuild now.
    Rebuild,
    Nothing,
}

impl Rebuilds {
    pub(crate) const fn new() -> Self {
        Self {
            up: AtomicBool::new(false),
            due: AtomicBool::new(false),
        }
    }

    /// A menu opened.
    pub(crate) fn enter(&self) {
        self.up.store(true, Ordering::SeqCst);
    }

    /// A menu closed: whether a rebuild waits (`after` is then posted).
    pub(crate) fn exit(&self) -> bool {
        self.up.store(false, Ordering::SeqCst);
        self.due.load(Ordering::SeqCst)
    }

    /// A rebuild is asked for: whether it may run now. While a menu is open
    /// it is marked due instead.
    pub(crate) fn ask(&self) -> bool {
        if self.up.load(Ordering::SeqCst) {
            self.due.store(true, Ordering::SeqCst);
            false
        } else {
            true
        }
    }

    /// After a menu closed: wait for a queued click, else the due rebuild,
    /// once.
    pub(crate) fn after(&self, queued: bool) -> After {
        if queued {
            After::Again
        } else if self.due.swap(false, Ordering::SeqCst) {
            After::Rebuild
        } else {
            After::Nothing
        }
    }

    pub(crate) fn up(&self) -> bool {
        self.up.load(Ordering::Relaxed)
    }
}

/// An end asked of the subclass: for his menu numbered so, or the timer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndAsk {
    For(u32, Ended),
    Time,
}

/// Whether an end closes the menu up now (`his`, 0 for none), and why: an
/// end for a menu only closes that one, never a later one; the timer closes
/// his menu if one is up.
pub(crate) fn end_now(ask: EndAsk, his: u32) -> Option<Ended> {
    match ask {
        EndAsk::For(n, why) => should_end(his, n).then_some(why),
        EndAsk::Time => (his != 0).then_some(Ended::Time),
    }
}

/// Why Fritz ended his menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ended {
    /// A press elsewhere, which the hook reported.
    Elsewhere = 1,
    /// `MOUSE_ONLY_FOR` with no click.
    Time = 2,
    /// The hook went out, for any of its gate's reasons (he was hidden or
    /// arranged, the switch turned off, the session locked...).
    HookOut = 3,
}

impl Ended {
    fn of(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Elsewhere),
            2 => Some(Self::Time),
            3 => Some(Self::HookOut),
            _ => None,
        }
    }

    fn words(self) -> &'static str {
        match self {
            Self::Elsewhere => "a click elsewhere",
            Self::Time => "20 s with no click",
            Self::HookOut => "the mouse hook going out",
        }
    }
}

/// The calls showing his menu makes, so the order is tested without a
/// window.
pub(crate) trait PopupDesk {
    /// The tray's held menu; 0 when there is none.
    fn menu(&self) -> isize;
    /// `TrackPopupMenu` with `TPM_RETURNCMD`: the chosen id, 0 for none.
    fn track(&self, menu: isize, at: (i32, i32), owner: isize) -> u32;
    /// The chosen id to the tray's window as `WM_COMMAND`, so muda
    /// dispatches it as a click on the tray icon's menu.
    fn send_command(&self, owner: isize, id: u32);
    /// The documented benign message after a popup.
    fn post_null(&self, owner: isize);
    fn arm_timer(&self, owner: isize);
    fn disarm_timer(&self, owner: isize);
}

/// What showing his menu came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shown {
    NoMenu,
    Chose(u32),
    Cancelled,
}

/// Show the menu at `at` on `owner`, by mouse: the timer armed first (a
/// menu without the foreground doesn't close itself), the menu tracked,
/// the timer disarmed, the benign message, then the chosen item sent.
/// The foreground is never asked for.
pub(crate) fn show_on(d: &impl PopupDesk, owner: isize, at: (i32, i32)) -> Shown {
    let menu = d.menu();
    if menu == 0 {
        return Shown::NoMenu;
    }
    d.arm_timer(owner);
    let id = d.track(menu, at, owner);
    d.disarm_timer(owner);
    d.post_null(owner);
    if id == 0 {
        Shown::Cancelled
    } else {
        d.send_command(owner, id);
        Shown::Chose(id)
    }
}

/// Whether an end asked for menu `asked` ends the menu up now (`up`, 0 for
/// none): only that menu, never a later one.
pub(crate) fn should_end(up: u32, asked: u32) -> bool {
    up != 0 && up == asked
}

/// The number after `n`, never 0 (0 is "none up").
pub(crate) fn next_number(n: u32) -> u32 {
    n.wrapping_add(1).max(1)
}

/// A screen point as a message's two parameters, and back: signed, as a
/// point left of or above the primary monitor is.
pub(crate) fn pack_at(at: (i32, i32)) -> (WPARAM, LPARAM) {
    (WPARAM(at.0 as isize as usize), LPARAM(at.1 as isize))
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn unpack_at(wp: WPARAM, lp: LPARAM) -> (i32, i32) {
    (wp.0 as isize as i32, lp.0 as i32)
}

/// His menu's number while it is up; 0 when it isn't. Atomics only: the
/// mouse hook's procedure reads it.
pub fn his_menu() -> u32 {
    HIS.load(Ordering::SeqCst)
}

/// A menu of Fritz's that isn't his is open (the tray icon's own): presses
/// are its, not his. Atomics only.
pub fn other_menu_up() -> bool {
    other_up(REBUILDS.up(), HIS.load(Ordering::Relaxed))
}

/// A menu is open (`menu_up`) and it isn't his (`his`, 0 when none is up).
pub(crate) fn other_up(menu_up: bool, his: u32) -> bool {
    menu_up && his == 0
}

/// Whether his menu may open now: no menu of Fritz's is open, his or the
/// tray icon's.
pub(crate) fn may_show(menu_up: bool, his: u32) -> bool {
    !menu_up && his == 0
}

/// `tray::rebuild`'s first question, on the main thread: whether it may
/// rebuild now, or (a menu of Fritz's open) waits to run once it closes.
pub fn may_rebuild() -> bool {
    REBUILDS.ask()
}

/// The thread that owns the tray's window and shows its menus.
pub fn main_thread() -> u32 {
    MAIN_TID.load(Ordering::SeqCst)
}

/// At setup, on the main thread, once the tray is built: the subclass on
/// the tray's window. A failure warns, and a right-click on him then does
/// nothing.
pub fn adopt(app: &AppHandle) -> bool {
    let _ = APP.set(app.clone());
    let tray = crate::tray::window(app);
    if tray == 0 {
        tracing::warn!("couldn't find the tray's window: a right-click on him opens nothing");
        return false;
    }
    // SAFETY: plain calls; the strings are static.
    let msgs = unsafe {
        Msgs {
            at: RegisterWindowMessageW(w!("Fritz.PopupAt")),
            end: RegisterWindowMessageW(w!("Fritz.PopupEnd")),
            after: RegisterWindowMessageW(w!("Fritz.PopupAfter")),
            pretend: RegisterWindowMessageW(w!("Fritz.PopupPretend")),
        }
    };
    if [msgs.at, msgs.end, msgs.after, msgs.pretend].contains(&0) {
        tracing::warn!("couldn't register his menu's messages: a right-click on him opens nothing");
        return false;
    }
    let _ = MSGS.set(msgs);
    TRAY.store(tray, Ordering::SeqCst);
    // SAFETY: this thread's id.
    MAIN_TID.store(unsafe { GetCurrentThreadId() }, Ordering::SeqCst);
    // SAFETY: the tray's window, made on this (the main) thread, and a
    // static procedure that takes itself off at WM_NCDESTROY.
    let set = unsafe { SetWindowSubclass(windows_ext::hwnd(tray), Some(popup_proc), SUBCLASS_ID, 0) }.as_bool();
    if !set {
        tracing::warn!("couldn't watch the tray's window: a right-click on him opens nothing");
    }
    set
}

/// Post one of his menu's messages to the tray's window. From any thread;
/// never waits.
fn post(which: fn(&Msgs) -> u32, wp: WPARAM, lp: LPARAM) -> bool {
    let (Some(m), tray) = (MSGS.get(), TRAY.load(Ordering::SeqCst)) else {
        return false;
    };
    if tray == 0 {
        return false;
    }
    // SAFETY: a window handle of ours; a stale one only makes the post fail.
    unsafe { PostMessageW(Some(windows_ext::hwnd(tray)), which(m), wp, lp) }.is_ok()
}

/// Show his menu at `at`, the release point (the hook's owner thread).
pub fn ask(at: (i32, i32)) -> bool {
    let (wp, lp) = pack_at(at);
    post(|m| m.at, wp, lp)
}

/// End his menu numbered `menu`, if it is still the one up.
pub fn end(menu: u32, why: Ended) -> bool {
    post(|m| m.end, WPARAM(menu as usize), LPARAM(why as isize))
}

/// Debug builds' pretend right-click: say what would open, show nothing.
pub fn pretend_ask(at: (i32, i32)) -> bool {
    let (wp, lp) = pack_at(at);
    post(|m| m.pretend, wp, lp)
}

/// The desk as it is: the tray's held menu, Win32's menu calls.
struct Live;

impl PopupDesk for Live {
    fn menu(&self) -> isize {
        crate::tray::hmenu()
    }
    fn track(&self, menu: isize, at: (i32, i32), owner: isize) -> u32 {
        windows_ext::track_menu(menu, owner, at)
    }
    fn send_command(&self, owner: isize, id: u32) {
        windows_ext::send_command(owner, id);
    }
    fn post_null(&self, owner: isize) {
        windows_ext::post_null(owner);
    }
    fn arm_timer(&self, owner: isize) {
        #[allow(clippy::cast_possible_truncation)]
        let ms = MOUSE_ONLY_FOR.as_millis() as u32;
        // SAFETY: the tray's window, on its own thread.
        let _ = unsafe { SetTimer(Some(windows_ext::hwnd(owner)), TIMER_ID, ms, None) };
    }
    fn disarm_timer(&self, owner: isize) {
        // SAFETY: the tray's window, on its own thread.
        let _ = unsafe { KillTimer(Some(windows_ext::hwnd(owner)), TIMER_ID) };
    }
}

/// The tray window's subclass: his menu's messages, the menu loop's edges
/// and the timer; everything passes on down the chain but our own.
unsafe extern "system" fn popup_proc(h: HWND, msg: u32, wp: WPARAM, lp: LPARAM, _id: usize, _data: usize) -> LRESULT {
    if msg == WM_NCDESTROY {
        // SAFETY: our own subclass on the window it was set on.
        let _ = unsafe { RemoveWindowSubclass(h, Some(popup_proc), SUBCLASS_ID) };
    } else if let Some(m) = MSGS.get() {
        let ours = [m.at, m.end, m.after, m.pretend].contains(&msg) || (msg == WM_TIMER && wp.0 == TIMER_ID);
        if ours || msg == WM_ENTERMENULOOP || msg == WM_EXITMENULOOP {
            crate::logs::guard("his menu", || on_message(h, msg, wp, lp, *m));
        }
        if ours {
            return LRESULT(0);
        }
    }
    // SAFETY: passes the message on down the subclass chain unchanged.
    unsafe { DefSubclassProc(h, msg, wp, lp) }
}

fn on_message(h: HWND, msg: u32, wp: WPARAM, lp: LPARAM, m: Msgs) {
    let tray = h.0 as isize;
    if msg == m.at {
        show(tray, unpack_at(wp, lp));
    } else if msg == m.end {
        #[allow(clippy::cast_possible_truncation)]
        let (asked, why) = (wp.0 as u32, Ended::of(lp.0 as u8));
        if let Some(why) = why.and_then(|w| end_now(EndAsk::For(asked, w), HIS.load(Ordering::SeqCst))) {
            close(why);
        }
    } else if msg == WM_TIMER {
        // SAFETY: the tray's window, on its own thread.
        let _ = unsafe { KillTimer(Some(h), TIMER_ID) };
        if let Some(why) = end_now(EndAsk::Time, HIS.load(Ordering::SeqCst)) {
            close(why);
        }
    } else if msg == WM_ENTERMENULOOP {
        REBUILDS.enter();
    } else if msg == WM_EXITMENULOOP {
        if REBUILDS.exit() {
            post(|m| m.after, WPARAM(0), LPARAM(0));
        }
    } else if msg == m.after {
        after(h);
    } else if msg == m.pretend {
        pretend(tray);
    }
}

/// End his menu, saying why: on the menu's own thread, as `EndMenu`
/// requires.
fn close(why: Ended) {
    ENDED.store(why as u8, Ordering::SeqCst);
    // SAFETY: called from the tray window's procedure, on the menu's thread.
    let _ = unsafe { EndMenu() };
}

/// Show his menu, unless a menu of Fritz's is already open, or a press (or
/// the hook going out) came between the right-click and now: a menu late
/// to open must not open over what the owner went on to do.
fn show(tray: isize, at: (i32, i32)) {
    if !may_show(REBUILDS.up(), HIS.load(Ordering::SeqCst)) {
        tracing::debug!("a right-click on him while a menu of Fritz's is open: nothing more opens");
        return;
    }
    // Up first, then the answer taken: a press (or the hook going out)
    // either clears the answer before it is taken, or sees this menu up and
    // ends it. Both sides store, then read.
    let n = next_number(NUMBERED.load(Ordering::SeqCst));
    NUMBERED.store(n, Ordering::SeqCst);
    ENDED.store(0, Ordering::SeqCst);
    HIS.store(n, Ordering::SeqCst);
    if !crate::hook::take_asked() {
        HIS.store(0, Ordering::SeqCst);
        tracing::info!("his menu wasn't opened: a click (or the hook going out) came first");
        return;
    }
    tracing::info!("his menu opened (a right-click on him), by mouse: a click elsewhere or 20 s closes it");
    let shown = show_on(&Live, tray, at);
    HIS.store(0, Ordering::SeqCst);
    let ended = Ended::of(ENDED.swap(0, Ordering::SeqCst));
    match (shown, ended) {
        (Shown::NoMenu, _) => tracing::warn!("his menu couldn't open: the tray has no menu"),
        (Shown::Chose(_), _) => tracing::info!("his menu closed: an item chosen"),
        (Shown::Cancelled, Some(why)) => tracing::info!("his menu closed: {}", why.words()),
        (Shown::Cancelled, None) => tracing::info!("his menu closed: nothing chosen"),
    }
}

/// After a menu closed with a rebuild due: once no click for the tray's
/// window is still queued, so the click meets the menu it came from. The
/// rebuild itself is posted to the event loop from the async runtime, so it
/// never runs inside this procedure.
fn after(h: HWND) {
    let mut msg = MSG::default();
    // SAFETY: a look at this thread's own queue, removing nothing.
    let queued = unsafe { PeekMessageW(&raw mut msg, Some(h), WM_COMMAND, WM_COMMAND, PM_NOREMOVE) }.as_bool();
    match REBUILDS.after(queued) {
        After::Again => {
            post(|m| m.after, WPARAM(0), LPARAM(0));
        }
        After::Rebuild => {
            if let Some(app) = APP.get() {
                let app = app.clone();
                tauri::async_runtime::spawn(async move { crate::tray::refresh(&app) });
            }
        }
        After::Nothing => {}
    }
}

/// What his menu would be, shown nothing (debug builds' pretend click).
fn pretend(tray: isize) {
    let menu = crate::tray::hmenu();
    let h = HMENU(menu as *mut core::ffi::c_void);
    // SAFETY: plain queries on a menu handle; a stale one answers false.
    let (is_menu, items) = unsafe { (IsMenu(h).as_bool(), GetMenuItemCount(Some(h))) };
    if is_menu {
        tracing::info!(
            "his menu would open where the button was released (pretend: nothing shown, the keyboard untouched): the tray's held menu, {items} items, on the tray's window {tray:#x}"
        );
    } else {
        tracing::warn!("pretend: his menu couldn't open: the tray has no menu");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::cell::RefCell;

    use super::*;

    const TRAY: isize = 9;

    /// A desk that logs each call, with the menu and the choice set.
    struct FakeDesk {
        menu: isize,
        choose: u32,
        log: RefCell<Vec<String>>,
    }

    impl FakeDesk {
        fn new(menu: isize, choose: u32) -> Self {
            Self { menu, choose, log: RefCell::default() }
        }
        fn log(&self) -> Vec<String> {
            self.log.borrow().clone()
        }
    }

    impl PopupDesk for FakeDesk {
        fn menu(&self) -> isize {
            self.menu
        }
        fn track(&self, menu: isize, at: (i32, i32), owner: isize) -> u32 {
            self.log.borrow_mut().push(format!("track {menu} at {},{} on {owner}", at.0, at.1));
            self.choose
        }
        fn send_command(&self, owner: isize, id: u32) {
            self.log.borrow_mut().push(format!("command {id} to {owner}"));
        }
        fn post_null(&self, owner: isize) {
            self.log.borrow_mut().push(format!("null to {owner}"));
        }
        fn arm_timer(&self, owner: isize) {
            self.log.borrow_mut().push(format!("timer on {owner}"));
        }
        fn disarm_timer(&self, owner: isize) {
            self.log.borrow_mut().push(format!("untimer on {owner}"));
        }
    }

    #[test]
    fn chosen_by_mouse() {
        let d = FakeDesk::new(77, 1003);
        assert_eq!(show_on(&d, TRAY, (-5, 300)), Shown::Chose(1003));
        assert_eq!(d.log(), ["timer on 9", "track 77 at -5,300 on 9", "untimer on 9", "null to 9", "command 1003 to 9"]);
    }

    #[test]
    fn cancelled_sends_nothing() {
        let d = FakeDesk::new(77, 0);
        assert_eq!(show_on(&d, TRAY, (10, 20)), Shown::Cancelled);
        assert_eq!(d.log(), ["timer on 9", "track 77 at 10,20 on 9", "untimer on 9", "null to 9"]);
    }

    #[test]
    fn no_menu_does_nothing() {
        let d = FakeDesk::new(0, 1003);
        assert_eq!(show_on(&d, TRAY, (10, 20)), Shown::NoMenu);
        assert!(d.log().is_empty());
    }

    #[test]
    fn an_end_closes_only_the_menu_it_was_asked_for() {
        assert!(should_end(3, 3));
        assert!(!should_end(4, 3), "a later menu");
        assert!(!should_end(0, 3), "none up");
        assert!(!should_end(0, 0));
        assert_eq!(next_number(0), 1);
        assert_eq!(next_number(41), 42);
        assert_eq!(next_number(u32::MAX), 1, "never 0, which is none up");
    }

    #[test]
    fn presses_pass_to_another_menu_of_ours_and_his_menu_opens_alone() {
        assert!(other_up(true, 0), "the tray icon's own menu");
        assert!(!other_up(true, 5), "his menu");
        assert!(!other_up(false, 0));
        assert!(may_show(false, 0));
        assert!(!may_show(true, 0), "the tray icon's menu is open");
        assert!(!may_show(false, 5), "his menu is open");
        assert!(!may_show(true, 5));
    }

    /// A rebuild asked for while a menu is open waits, and runs once it has
    /// closed and its click has gone; never twice, never lost.
    #[test]
    fn a_rebuild_waits_while_a_menu_is_open() {
        let r = Rebuilds::new();
        assert!(r.ask(), "no menu: now");
        assert!(!r.exit(), "nothing due");
        assert_eq!(r.after(false), After::Nothing);
        r.enter();
        assert!(r.up());
        assert!(!r.ask(), "a menu open: it waits");
        assert!(!r.ask());
        assert!(r.exit(), "closed: after is posted");
        assert!(!r.up());
        assert_eq!(r.after(true), After::Again, "a click still queued");
        assert_eq!(r.after(false), After::Rebuild);
        assert_eq!(r.after(false), After::Nothing, "once");
        assert!(r.ask(), "and the next asks run at once");
    }

    #[test]
    fn an_end_or_the_timer_closes_only_his_menu_up_now() {
        for (ask, his, want) in [
            (EndAsk::For(3, Ended::Elsewhere), 3, Some(Ended::Elsewhere)),
            (EndAsk::For(3, Ended::HookOut), 3, Some(Ended::HookOut)),
            (EndAsk::For(3, Ended::Elsewhere), 4, None),
            (EndAsk::For(3, Ended::Elsewhere), 0, None),
            (EndAsk::Time, 7, Some(Ended::Time)),
            (EndAsk::Time, 0, None),
        ] {
            assert_eq!(end_now(ask, his), want, "{ask:?} with {his} up");
        }
    }

    /// The subclass decides through the steps above, and his menu is
    /// numbered for the hook while it is up, once a press hasn't come first.
    #[test]
    fn the_subclass_goes_by_the_steps() {
        let me = source("popup");
        let at = me.find("fn on_message(").unwrap();
        let body = &me[at..at + me[at..].find("\n}\n").unwrap()];
        assert!(body.contains("end_now(EndAsk::For(asked, w), HIS.load(Ordering::SeqCst))"));
        assert!(body.contains("if let Some(why) = end_now(EndAsk::Time, HIS.load(Ordering::SeqCst)) {"));
        assert_eq!(body.matches("close(why);").count(), 2);
        let enter = body.find("msg == WM_ENTERMENULOOP {").unwrap();
        let exit = body.find("msg == WM_EXITMENULOOP {").unwrap();
        assert!(body[enter..exit].contains("REBUILDS.enter();"));
        assert!(body[exit..].contains("if REBUILDS.exit() {\n            post(|m| m.after"));
        assert!(body.contains("} else if msg == m.after {\n        after(h);"));
        let at = me.find("fn close(why: Ended)").unwrap();
        let close = &me[at..at + me[at..].find("\n}\n").unwrap()];
        assert!(close.find("ENDED.store(why as u8, Ordering::SeqCst);").unwrap() < close.find("EndMenu()").unwrap());
        let at = me.find("fn show(tray: isize").unwrap();
        let show = &me[at..at + me[at..].find("\n}\n").unwrap()];
        let order = [
            "if !may_show(REBUILDS.up(), HIS.load(Ordering::SeqCst)) {\n        tracing::debug!",
            "ENDED.store(0, Ordering::SeqCst);",
            "HIS.store(n, Ordering::SeqCst);",
            "if !crate::hook::take_asked() {\n        HIS.store(0, Ordering::SeqCst);\n        tracing::info!(\"his menu wasn't opened",
            "show_on(&Live, tray, at)",
            "    HIS.store(0, Ordering::SeqCst);\n    let ended = Ended::of(ENDED.swap(0, Ordering::SeqCst));",
        ]
        .map(|s| show.find(s).unwrap_or_else(|| panic!("{s}")));
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        assert_eq!(show.matches("        return;\n    }").count(), 2, "each guard returns");
        let at = me.find("pub fn other_menu_up(").unwrap();
        assert!(me[at..].starts_with("pub fn other_menu_up() -> bool {\n    other_up(REBUILDS.up(), HIS.load(Ordering::Relaxed))"));
        let at = me.find("pub fn his_menu(").unwrap();
        assert!(me[at..].starts_with("pub fn his_menu() -> u32 {\n    HIS.load(Ordering::SeqCst)"), "read as strictly as it is written");
        let at = me.find("pub fn may_rebuild(").unwrap();
        assert!(me[at..].starts_with("pub fn may_rebuild() -> bool {\n    REBUILDS.ask()\n}"));
    }

    #[test]
    fn a_point_crosses_a_message_whole() {
        for at in [(0, 0), (-10_000, 0), (3112, -1440), (i32::MIN, i32::MAX), (-1, -1)] {
            let (wp, lp) = pack_at(at);
            assert_eq!(unpack_at(wp, lp), at);
        }
    }

    #[test]
    fn the_ends_say_why() {
        for e in [Ended::Elsewhere, Ended::Time, Ended::HookOut] {
            assert_eq!(Ended::of(e as u8), Some(e));
        }
        assert_eq!(Ended::of(0), None);
        assert_eq!(Ended::Time.words(), "20 s with no click");
        assert_eq!(MOUSE_ONLY_FOR, Duration::from_secs(20));
    }

    /// Our subclass and timer ids are apart from each other's and from
    /// tray-icon's own messages and timer (6002-6010).
    #[test]
    fn the_ids_are_distinct() {
        let ours = [crate::windows_ext::PIN_ID, crate::windows_ext::DEACTIVATE_ID, SUBCLASS_ID, TIMER_ID];
        for (i, a) in ours.iter().enumerate() {
            for b in &ours[i + 1..] {
                assert_ne!(a, b);
            }
            assert!(!(6002..=6010).contains(a));
        }
    }

    fn source(name: &str) -> String {
        crate::keyboard::tests::SOURCES
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1
            .replace("\r\n", "\n")
    }

    /// His menu is the tray's, on the tray's window: never a menu built
    /// here, never one hosted on his window (muda's popup would take the
    /// foreground for it), and never the foreground asked for.
    #[test]
    fn his_menu_is_the_trays_by_mouse_on_the_trays_window() {
        let me = source("popup");
        let code = &me[..me.find("#[cfg(test)]").unwrap()];
        for never in [
            concat!("MenuBuil", "der"),
            concat!("MenuItemBuil", "der"),
            concat!(".te", "xt("),
            concat!("OVERLAY_", "LABEL"),
            concat!("popup_", "menu"),
            concat!("try_fore", "ground"),
            concat!("SetForeground", "Window"),
            concat!("hand_", "back"),
        ] {
            assert!(!code.contains(never), "popup.rs uses {never}");
        }
        for (name, src) in crate::keyboard::tests::SOURCES {
            for never in [concat!(".popup_", "menu("), concat!(".popup_", "menu_at(")] {
                assert!(!src.contains(never), "{name}.rs hosts a menu with {never}");
            }
        }
    }

    /// `TrackPopupMenu` is called in one place, with `TPM_RETURNCMD` (or a
    /// chosen item would act twice), from the subclass's desk alone.
    #[test]
    fn the_menu_is_tracked_once_and_returns_its_choice() {
        let mut calls = Vec::new();
        for (name, src) in crate::keyboard::tests::SOURCES {
            calls.extend(src.match_indices(concat!("TrackPopup", "Menu(")).map(|_| name));
        }
        assert_eq!(calls, ["windows_ext"]);
        let ext = source("windows_ext");
        let at = ext.find("pub fn track_menu(").unwrap();
        let track = &ext[at..at + ext[at..].find("\n}\n").unwrap()];
        assert!(track.contains("TPM_RETURNCMD") && track.contains("TPM_RIGHTBUTTON"), "{track}");
        let mut users = Vec::new();
        for (name, src) in crate::keyboard::tests::SOURCES {
            let code = src.split("#[cfg(test)]").next().unwrap();
            users.extend(code.match_indices("track_menu(").map(|_| name));
        }
        users.sort_unstable();
        assert_eq!(users, ["popup", "windows_ext"], "only the subclass's desk tracks it");
    }

    #[test]
    fn the_subclass_takes_itself_off_as_the_window_goes() {
        let me = source("popup");
        let at = me.find("unsafe extern \"system\" fn popup_proc(").unwrap();
        let body = &me[at..at + me[at..].find("\n}\n").unwrap()];
        let gone = body.find("WM_NCDESTROY").unwrap();
        assert!(body[gone..].contains("RemoveWindowSubclass(h, Some(popup_proc), SUBCLASS_ID)"));
        // Ours, and the menu loop's edges, reach on_message; only ours stop here.
        assert!(body.contains("let ours = [m.at, m.end, m.after, m.pretend].contains(&msg) || (msg == WM_TIMER && wp.0 == TIMER_ID);"));
        assert!(body.contains("if ours || msg == WM_ENTERMENULOOP || msg == WM_EXITMENULOOP {\n            crate::logs::guard(\"his menu\", || on_message(h, msg, wp, lp, *m));"));
        assert!(body.contains("if ours {\n            return LRESULT(0);"));
    }

    /// The rebuild after a menu waits while a click for the tray's window is
    /// still queued, and never runs inside the procedure.
    #[test]
    fn the_rebuild_after_a_menu_waits_for_its_click() {
        let me = source("popup");
        let at = me.find("fn after(h: HWND)").unwrap();
        let body = &me[at..at + me[at..].find("\n}\n").unwrap()];
        let peek = body.find("PeekMessageW(&raw mut msg, Some(h), WM_COMMAND, WM_COMMAND, PM_NOREMOVE)").unwrap();
        let step = body.find("match REBUILDS.after(queued) {").unwrap();
        let again = body.find("After::Again => {\n            post(|m| m.after").unwrap();
        let spawn = body.find("After::Rebuild => {").unwrap();
        assert!(peek < step && step < again && again < spawn);
        assert!(body[spawn..].contains("tauri::async_runtime::spawn(async move { crate::tray::refresh(&app) })"));
    }
}
