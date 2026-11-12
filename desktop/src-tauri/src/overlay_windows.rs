// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src-tauri/src/overlay_windows.rs at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.

//! The avatar window's shell — the glue that runs [`OverlayCore`]'s effects
//! on the real window: building, placing, showing, parking and closing it,
//! its click-through, the page's handshake and beats, the hotkey, the
//! watchdog and the lease. DivoraVoice's recorder hooks, capture protection
//! and charm serving are gone; parking, `focusable(false)`, Alt-Tab and
//! Lorraine's browser arguments are new.
//!
//! Threads and locks. `core` is a leaf: while holding it, code only
//! computes, emits, and (off the main thread) posts. Window calls run on the
//! main thread — posted as one closure, or inline when already there, with
//! `core` released first, because an inline setter can send a window event
//! straight back into [`on_window_event`], which takes `core`. Getters
//! (`hwnd`, `outer_position`, `inner_size`) run only on worker threads,
//! never under `core`: `open` calls them under the placement lock, which the
//! main thread never takes. Main-thread code reads the stored HWND (and
//! Win32's `GetWindowRect`, which is no round trip). The page's handshake and
//! beats are commands, never event listeners: tauri parks Rust event
//! listeners on lock contention (event/listener.rs:137-143, 196-213).
//!
//! Focus is taken with plain `SetForegroundWindow`, never tauri's or tao's
//! focus call, which injects an Alt key press when Windows refuses
//! (`the_overlay_never_calls_set_focus`). Locked, the window is not even
//! focusable (`WS_EX_NOACTIVATE`); only Arrange makes it so.

use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, WindowEvent,
};

use crate::overlay_machine::{
    hotkey_overlay_request, placement_settled, should_return_focus, Effect, HandshakeStage, Input,
    Mode, ModeRequest, OverlayCore, OverlayRect, OverlayStatus, Phase, Refusal, Setter, CLOSING,
    LEASE_TICK, NOT_OPEN, NO_READY_GLOBAL, OVERLAY_LABEL, OVERLAY_STATUS_EVENT, OVERLAY_TITLE,
    PRETEND_NO_READY_ENV, R1_RUNG, WATCHDOG,
};
use crate::windows_ext::{self, Bounds};

/// Lorraine's WebView2 arguments, byte-identical in every window this app
/// ever builds (and in tauri.conf.json's windows), or WebView2 forks a second
/// browser environment. They also stop Chromium treating an unfocused or
/// covered window as backgrounded, which would throttle the paced loop.
pub const BROWSER_ARGS: &str = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection,CalculateNativeWinOcclusion,UseEcoQoSForBackgroundProcess --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-background-timer-throttling";

/// Where a parked window stands: off every monitor, so it is never shown
/// again, re-activated or composed (the SnapShelf pattern), only moved back.
pub const PARK_AT: (i32, i32) = (-10_000, 0);

/// How long `open` waits for a closing window to go before it gives up.
const CLOSE_WAIT: Duration = Duration::from_secs(2);
const CLOSE_POLL: Duration = Duration::from_millis(50);

/// The avatar's state, shared by the commands, the window events, the
/// watchdog, the lease, the hotkeys, the tray and the cover watch.
pub struct OverlayShared {
    /// The machine. A leaf lock (see the module comment).
    core: Mutex<OverlayCore>,
    /// The live window's HWND as an integer; 0 when none.
    hwnd: AtomicIsize,
    /// The core is in Arrange, for the tray menu, which mustn't take `core`.
    arranging: AtomicBool,
    /// The window to give the keyboard back to when Arrange ends; 0 when
    /// none.
    return_to: AtomicIsize,
    /// The current park is the cover watch's, not the owner's Hide: the
    /// watch may undo it, and it is never saved.
    cover_parked: AtomicBool,
    /// The window is at the park spot because the shell moved it there: set
    /// before the move off screen, cleared after the move back, so a reader
    /// in between gets `home`. Not a guess from the window's coordinates,
    /// which a monitor far to the left of the primary can also have.
    window_parked: AtomicBool,
    /// Where Fritz stands when he isn't parked: set when the window is
    /// placed, and from the window itself just before it is parked. A leaf
    /// lock, never held across a window call.
    home: Mutex<Option<OverlayRect>>,
    /// Held while a park is decided and stepped, by the owner's Hide/Show and
    /// by the cover watch, so the watch's check and its act see the same
    /// state (an owner's Hide landing in between can't be taken for the
    /// watch's park). Taken before `core`, never after it.
    park_lock: Mutex<()>,
    /// Held by `open` from build to `Built`, so placements never interleave.
    placement: tauri::async_runtime::Mutex<()>,
}

impl OverlayShared {
    fn new() -> Self {
        Self {
            core: Mutex::new(OverlayCore::new()),
            hwnd: AtomicIsize::new(0),
            arranging: AtomicBool::new(false),
            return_to: AtomicIsize::new(0),
            cover_parked: AtomicBool::new(false),
            window_parked: AtomicBool::new(false),
            home: Mutex::new(None),
            park_lock: Mutex::new(()),
            placement: tauri::async_runtime::Mutex::new(()),
        }
    }

    fn core(&self) -> MutexGuard<'_, OverlayCore> {
        self.core.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn home(&self) -> MutexGuard<'_, Option<OverlayRect>> {
        self.home.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The thread the event loop runs on, set at setup.
static MAIN_THREAD: OnceLock<ThreadId> = OnceLock::new();

fn on_main_thread() -> bool {
    MAIN_THREAD.get() == Some(&thread::current().id())
}

fn shared(app: &AppHandle) -> Option<State<'_, OverlayShared>> {
    app.try_state::<OverlayShared>()
}

/// At setup, on the main thread.
pub fn init(app: &tauri::App) {
    let _ = MAIN_THREAD.set(thread::current().id());
    tracing::info!("avatar shell ready");
    app.manage(OverlayShared::new());
}

/// Build the window at `rect` (Windows' default spot when `None`) and show it
/// once its page is ready.
pub async fn open(app: AppHandle, rect: Option<OverlayRect>) -> Result<OverlayStatus, String> {
    let (status, effects) = match step_effects(&app, Input::Open { fit: false }) {
        Err(Refusal::Closing) => {
            // The last window is still going: wait for it, then try once more.
            let h = app.clone();
            let closed = tauri::async_runtime::spawn_blocking(move || {
                shared(&h).is_some_and(|s| wait_closed(&s, CLOSE_WAIT, CLOSE_POLL))
            })
            .await
            .unwrap_or(false);
            if !closed {
                return Err(CLOSING.to_owned());
            }
            step_effects(&app, Input::Open { fit: false }).map_err(|r| r.message().to_owned())?
        }
        other => other.map_err(|r| r.message().to_owned())?,
    };
    let build = effects.iter().find_map(|e| match *e {
        Effect::Build { instance } => Some(instance),
        _ => None,
    });
    match build {
        Some(instance) => build_window(&app, instance, rect).await,
        // Open, or still opening: nothing to build.
        None => Ok(status),
    }
}

/// Build window `instance` hidden, unfocusable and click-through, place it,
/// and step `Built`. Holds the placement lock throughout; never `core`
/// across a window call.
async fn build_window(
    app: &AppHandle,
    instance: u32,
    rect: Option<OverlayRect>,
) -> Result<OverlayStatus, String> {
    let Some(shared) = shared(app) else {
        return Err(NOT_OPEN.to_owned());
    };
    let placing = shared.placement.lock().await;
    let url = WebviewUrl::App(avatar_url(instance).into());
    let mut builder = WebviewWindowBuilder::new(app, OVERLAY_LABEL, url)
        .title(OVERLAY_TITLE)
        .inner_size(300.0, 460.0)
        .min_inner_size(120.0, 160.0)
        .transparent(true)
        .decorations(false)
        // An undecorated transparent window keeps a 1 px border unless the
        // shadow is off (tauri #8308).
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .resizable(true)
        .maximizable(false)
        .minimizable(false)
        // Hidden until the page is ready, and it never takes focus on open.
        .visible(false)
        .focused(false)
        // New relative to DivoraVoice (tauri >= 2.8): WS_EX_NOACTIVATE, so
        // nothing activates it while Locked. Arrange lifts it (`focus`).
        .focusable(false)
        .additional_browser_args(BROWSER_ARGS);
    let pretend = cfg!(debug_assertions) && std::env::var_os(PRETEND_NO_READY_ENV).is_some();
    if let Some(script) = no_ready_script(pretend) {
        tracing::warn!("{PRETEND_NO_READY_ENV} is set: the avatar page won't say it is ready");
        builder = builder.initialization_script(script);
    }
    adopt(&shared, 0);
    let window = match builder.build() {
        Ok(w) => w,
        Err(e) => {
            forget(&shared);
            let _ = step(app, Input::BuildFailed { instance });
            return Err(format!("Couldn't open Fritz's window: {e}"));
        }
    };
    // Click-through before anything can show it. Posted; the getter below
    // waits behind it.
    let _ = window.set_ignore_cursor_events(true);
    let hwnd = window_id(&window);
    shared.hwnd.store(hwnd, Ordering::SeqCst);
    let _ = app.run_on_main_thread(move || {
        if !crate::non_rude::mark_non_rude(hwnd) {
            tracing::warn!("couldn't mark the avatar window NonRudeHWND");
        }
        // Out of Alt-Tab for the window's life: tao rewrites the extended
        // style on every change, the subclass puts the bit back each time.
        if !windows_ext::pin_toolwindow(hwnd) {
            tracing::warn!("couldn't keep the avatar window out of Alt-Tab");
        }
    });
    if let Some(rect) = rect.and_then(OverlayRect::sane) {
        place_window(&window, rect);
        *shared.home() = Some(rect);
    }
    let status = step(app, Input::Built { instance }).map_err(|r| r.message().to_owned());
    drop(placing);
    status
}

/// `avatar.html?instance=<n>`: the page echoes `n` in its handshake and
/// beats, so a stale call from an earlier window never counts.
fn avatar_url(instance: u32) -> String {
    format!("avatar.html?instance={instance}")
}

/// The init script that makes the page skip its ready handshake, when
/// [`PRETEND_NO_READY_ENV`] is set in a debug build.
fn no_ready_script(pretend: bool) -> Option<String> {
    pretend.then(|| format!("window.{NO_READY_GLOBAL} = true;"))
}

/// The window's HWND as an integer. A main-thread round trip: worker threads
/// only, never under `core`.
fn window_id(window: &WebviewWindow) -> isize {
    window.hwnd().map_or(0, |h| h.0 as isize)
}

/// The cells for a new window: its HWND, nothing remembered yet.
fn adopt(shared: &OverlayShared, hwnd: isize) {
    shared.hwnd.store(hwnd, Ordering::SeqCst);
    shared.return_to.store(0, Ordering::SeqCst);
    shared.window_parked.store(false, Ordering::SeqCst);
}

/// The window is going (closing, being destroyed, or the app quitting):
/// remember where he stood (an Arrange may have moved him since `home` was
/// last set), and the shell's mark comes off.
fn going(shared: &OverlayShared, n: &impl Native) {
    let hwnd = shared.hwnd.load(Ordering::SeqCst);
    if !shared.window_parked.load(Ordering::SeqCst) {
        if let Some(b) = n.bounds(hwnd) {
            *shared.home() = Some(rect_of(b));
        }
    }
    n.unmark_non_rude(hwnd);
}

/// The window is gone: no HWND, nothing remembered. `home` stays: the next
/// window opens where this one stood.
fn forget(shared: &OverlayShared) {
    shared.hwnd.store(0, Ordering::SeqCst);
    shared.return_to.store(0, Ordering::SeqCst);
    shared.window_parked.store(false, Ordering::SeqCst);
}

/// Whether the core stepped back to Closed within `within`.
fn wait_closed(shared: &OverlayShared, within: Duration, poll: Duration) -> bool {
    let until = Instant::now() + within;
    loop {
        if shared.core().phase == Phase::Closed {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        thread::sleep(poll);
    }
}

/// Size, move, read back; at most three passes. Windows rescales a window
/// (`WM_DPICHANGED`) when most of it crosses onto a monitor with another
/// scale, on the resize as well as the move, so a second pass puts it right.
/// Worker threads only: the reads are main-thread round trips.
fn place_window(window: &WebviewWindow, rect: OverlayRect) {
    let size = PhysicalSize::new(rect.width, rect.height);
    let pos = PhysicalPosition::new(rect.x, rect.y);
    for _ in 0..3 {
        let _ = window.set_size(size);
        let _ = window.set_position(pos);
        let (Ok(p), Ok(s)) = (window.outer_position(), window.inner_size()) else {
            return;
        };
        if placement_settled(rect, (p.x, p.y), (s.width, s.height)) {
            return;
        }
    }
}

/// `set_avatar_mode`, the tray's Arrange / Lock and the hotkey.
pub fn set_mode(app: &AppHandle, req: ModeRequest) -> Result<OverlayStatus, String> {
    step(app, Input::Mode(req)).map_err(|r| r.message().to_owned())
}

/// The owner's Hide or Show (the tray, the hotkey, the saved state at
/// start). Whatever the cover watch did before, the park is the owner's now.
pub fn set_parked(app: &AppHandle, on: bool) -> OverlayStatus {
    let Some(s) = shared(app) else {
        return closed();
    };
    let _deciding = s.park_lock.lock().unwrap_or_else(PoisonError::into_inner);
    s.cover_parked.store(false, Ordering::SeqCst);
    park_step(app, on)
}

/// The cover watch's tick: whether a fullscreen app covers a monitor Fritz
/// stands on. `cover::decide` says what to do with that, against the state
/// read under the park lock, so the check and the act agree; it does
/// nothing while he is being arranged (a drag would be yanked off screen,
/// and the window is the owner's then). Some(on) when it parked or unparked
/// him. The flag is set before the park and cleared after the unpark, so the
/// saver never takes the watch's park for the owner's.
pub fn set_covered(app: &AppHandle, covered: bool) -> Option<bool> {
    let s = shared(app)?;
    let _deciding = s.park_lock.lock().unwrap_or_else(PoisonError::into_inner);
    let (open, parked, arranging) = {
        let c = s.core();
        (c.phase != Phase::Closed, c.parked, c.mode == Mode::Arrange)
    };
    if !open {
        return None;
    }
    let on = crate::cover::decide(
        covered,
        parked,
        s.cover_parked.load(Ordering::SeqCst),
        arranging,
    )?;
    if on {
        s.cover_parked.store(true, Ordering::SeqCst);
    }
    park_step(app, on);
    if !on {
        s.cover_parked.store(false, Ordering::SeqCst);
    }
    Some(on)
}

/// Hidden by the owner, which is what the state file keeps (a park by the
/// cover watch is not: he comes back when the fullscreen app goes).
pub fn owner_parked(app: &AppHandle) -> bool {
    let Some(s) = shared(app) else {
        return false;
    };
    let _deciding = s.park_lock.lock().unwrap_or_else(PoisonError::into_inner);
    let parked = s.core().parked;
    parked && !s.cover_parked.load(Ordering::SeqCst)
}

fn park_step(app: &AppHandle, on: bool) -> OverlayStatus {
    let status = step(app, Input::Park { on }).unwrap_or_else(|_| status(app));
    // The tray's items follow; posted, as the cover watch calls this from
    // its own thread.
    let h = app.clone();
    let _ = app.run_on_main_thread(move || crate::tray::refresh(&h));
    status
}

/// `rect` moved, if need be, so its middle is on some monitor's work area:
/// a saved spot on a monitor that has since been unplugged comes back to
/// the nearest one.
pub fn onto_screen(rect: OverlayRect) -> OverlayRect {
    rect_of(windows_ext::onto_screen(bounds_of(rect)))
}

/// `avatar_handshake`, from the page. A command rather than a listener:
/// tauri parks Rust event listeners on lock contention
/// (event/listener.rs:137-143, 196-213); keep this on commands.
pub fn handshake(app: &AppHandle, instance: u32, stage: HandshakeStage) -> OverlayStatus {
    let input = match stage {
        HandshakeStage::Ready => Input::Ready { instance },
        HandshakeStage::Painted => Input::Painted { instance },
    };
    step(app, input).unwrap_or_else(|_| status(app))
}

/// `avatar_beat`, the page's beat while arranging. It only steps: a beat
/// emits nothing. A command for the same reason as [`handshake`].
pub fn alive(app: &AppHandle, instance: u32, gen: u32) {
    let _ = step(app, Input::Alive { instance, gen });
}

/// A global shortcut's press or release (the main thread, inside
/// `WM_HOTKEY`, so the change and its focus run inline while the hotkey's
/// right to the foreground is fresh). Never registers or unregisters a
/// shortcut: the plugin may hold its store's lock.
pub fn hotkey(app: &AppHandle, id: &str, pressed: bool) {
    if let Some(req) = hotkey_overlay_request(id, pressed) {
        if let Err(e) = set_mode(app, req) {
            tracing::debug!(error = %e, "avatar hotkey refused");
        }
    }
}

/// `avatar_status`.
pub fn status(app: &AppHandle) -> OverlayStatus {
    shared(app).map_or_else(closed, |s| s.core().status())
}

/// The avatar window's events (the caller checks the label). Main thread.
pub fn on_window_event(app: &AppHandle, event: &WindowEvent) {
    let Some(shared) = shared(app) else {
        return;
    };
    match event {
        WindowEvent::CloseRequested { .. } => {
            going(&shared, &Desktop(None));
            let _ = step(app, Input::CloseRequested);
        }
        WindowEvent::Destroyed => {
            forget(&shared);
            let _ = step(app, Input::Destroyed);
        }
        _ => {}
    }
}

/// The avatar is in Arrange, for the tray menu.
pub fn arranging(app: &AppHandle) -> bool {
    shared(app).is_some_and(|s| s.arranging.load(Ordering::SeqCst))
}

/// The live avatar window's HWND; 0 when none. For the foreground log.
pub fn avatar_hwnd(app: &AppHandle) -> isize {
    shared(app).map_or(0, |s| s.hwnd.load(Ordering::SeqCst))
}

/// Where Fritz stands (or would stand, when parked), in physical pixels: the
/// cover watch tests that monitor, and the state file keeps it.
pub fn standing(app: &AppHandle) -> Option<OverlayRect> {
    let shared = shared(app)?;
    let parked = shared.core().parked;
    let on_screen = (!parked && !shared.window_parked.load(Ordering::SeqCst))
        .then(|| windows_ext::window_bounds(shared.hwnd.load(Ordering::SeqCst)))
        .flatten()
        .map(rect_of);
    let home = *shared.home();
    on_screen.or(home)
}

/// From the tray's Quit, before the app exits: the shell's mark comes off
/// (the window isn't closed through `CloseRequested` then).
pub fn before_exit(app: &AppHandle) {
    if let Some(shared) = shared(app) {
        going(&shared, &Desktop(None));
    }
}

fn closed() -> OverlayStatus {
    OverlayCore::new().status()
}

fn bounds_of(r: OverlayRect) -> Bounds {
    Bounds {
        left: r.x,
        top: r.y,
        right: r
            .x
            .saturating_add(i32::try_from(r.width).unwrap_or(i32::MAX)),
        bottom: r
            .y
            .saturating_add(i32::try_from(r.height).unwrap_or(i32::MAX)),
    }
}

fn rect_of(b: Bounds) -> OverlayRect {
    OverlayRect {
        x: b.left,
        y: b.top,
        width: u32::try_from(b.width()).unwrap_or(0),
        height: u32::try_from(b.height()).unwrap_or(0),
    }
}

/// Step the core with `input`, emit, and run the effects.
fn step(app: &AppHandle, input: Input) -> Result<OverlayStatus, Refusal> {
    step_effects(app, input).map(|(status, _)| status)
}

/// [`step`], also handing back the effects (`open` builds the window
/// itself).
fn step_effects(app: &AppHandle, input: Input) -> Result<(OverlayStatus, Vec<Effect>), Refusal> {
    let Some(shared) = shared(app) else {
        return Err(Refusal::NotOpen);
    };
    let on_main = on_main_thread();
    let mut core = shared.core();
    let effects = core.step(input)?;
    let status = core.status();
    shared
        .arranging
        .store(core.mode == Mode::Arrange, Ordering::SeqCst);
    if on_main {
        // The window calls run inline, and may come straight back here.
        drop(core);
        dispatch(app, &effects, true);
    } else {
        // Posting doesn't wait, and under `core` keeps the order across
        // threads.
        dispatch(app, &effects, false);
        drop(core);
    }
    Ok((status, effects))
}

/// Start the effects' threads, and run their window work on the main
/// thread: inline when already there, else posted as one closure (the event
/// loop runs closures in the order posted).
fn dispatch(app: &AppHandle, effects: &[Effect], on_main: bool) {
    for effect in effects {
        match *effect {
            Effect::ArmWatchdog { instance } => arm_watchdog(app, instance),
            Effect::ArmLease { instance, gen } => arm_lease(app, instance, gen),
            _ => {}
        }
    }
    let work = main_thread_effects(effects);
    if work.is_empty() {
        return;
    }
    let h = app.clone();
    let run = move || run_on_main(&h, &work);
    if on_main {
        run();
    } else if app.run_on_main_thread(run).is_err() {
        tracing::debug!("avatar effects not run: the app is closing");
    }
}

/// The effects that are main-thread work, in order: the window's, the
/// tray's, and the status event. `Build` is `open`'s and the two timers are
/// threads.
fn main_thread_effects(effects: &[Effect]) -> Vec<Effect> {
    effects
        .iter()
        .copied()
        .filter(|e| {
            !matches!(
                e,
                Effect::Build { .. } | Effect::ArmWatchdog { .. } | Effect::ArmLease { .. }
            )
        })
        .collect()
}

/// Main thread, `core` released.
///
/// `Emit` sends the status as it is when the effect runs, not as it was at
/// the step, and only ever from here. An emit runs inline on the main
/// thread but is only queued from any other, so emitting at the step let a
/// worker's older status (the cover watch, a command) reach the page after
/// a hotkey's newer one. From the main thread, in order, and level-triggered,
/// the last status the page sees is always the current one.
fn run_on_main(app: &AppHandle, effects: &[Effect]) {
    // A panic here would unwind into the event loop's callback and abort.
    crate::logs::guard("avatar effects", || run_effects(app, effects));
}

fn run_effects(app: &AppHandle, effects: &[Effect]) {
    let Some(shared) = shared(app) else {
        return;
    };
    let desktop = Desktop::of(app);
    for &effect in effects {
        match effect {
            Effect::RefreshTray => crate::tray::refresh(app),
            Effect::Emit => {
                let now = shared.core().status();
                let _ = app.emit(OVERLAY_STATUS_EVENT, &now);
            }
            _ => window_effect(&shared, effect, &desktop),
        }
    }
}

/// One window effect, on the main thread with `core` released.
fn window_effect(shared: &OverlayShared, effect: Effect, n: &impl Native) {
    match effect {
        Effect::ApplyWindow => {
            reconcile_on(shared, n);
        }
        Effect::Show => show(shared, n),
        Effect::Focus => focus(shared, n),
        Effect::ReturnFocus => return_focus(shared, n),
        Effect::Park { .. } => {
            // Level-triggered, like ApplyWindow: a closure posted from a
            // worker may run after a newer inline step, so the window goes
            // where the core says now, not where it said when this was
            // queued.
            let (on, live) = {
                let c = shared.core();
                (c.parked, c.live())
            };
            if live {
                park(shared, on, n);
            }
        }
        Effect::CloseWindow => n.close(),
        Effect::DestroyWindow => {
            going(shared, n);
            n.destroy();
        }
        Effect::Build { .. }
        | Effect::ArmWatchdog { .. }
        | Effect::ArmLease { .. }
        | Effect::Emit
        | Effect::RefreshTray => {}
    }
}

/// Apply the latest window state, if the window is live; whether it was.
/// A closure that runs late reads the state as it is then.
fn reconcile_on(shared: &OverlayShared, n: &impl Native) -> bool {
    let plan = shared.core().reconcile_plan(R1_RUNG);
    let Some(plan) = plan else {
        return false;
    };
    apply_setters(shared, &plan, n);
    true
}

/// Run `plan` in its order. tao skips the flags that didn't change, so a
/// repeat apply makes no Win32 call.
fn apply_setters(shared: &OverlayShared, plan: &[Setter], n: &impl Native) {
    let hwnd = shared.hwnd.load(Ordering::SeqCst);
    for setter in plan {
        match *setter {
            Setter::IgnoreCursor(on) => {
                n.set_ignore_cursor(on);
                // Locked is click-through and never activated: the two go
                // together. (Arrange made it focusable in `focus`, before it
                // took the foreground.)
                if on {
                    n.set_focusable(false);
                }
            }
            Setter::LayerOpaque => n.make_layered_opaque(hwnd),
            Setter::Resizable(on) => n.set_resizable(on),
        }
    }
}

/// Apply the window state, then show it, back to back. A window no longer
/// live isn't shown.
fn show(shared: &OverlayShared, n: &impl Native) {
    if reconcile_on(shared, n) {
        n.show();
    }
}

/// The window that should get the keyboard back after Arrange: `fg` if it
/// is someone else's usable window, else the app beneath in z-order. Arrange
/// from the tray finds the tray's own hidden window in the foreground (the
/// menu put it there), which can never take the keyboard back.
fn return_target(fg: isize, ours: isize, n: &impl Native) -> isize {
    if fg != 0 && fg != ours && !n.owned_by_us(fg) && n.can_take_foreground(fg) {
        fg
    } else {
        n.next_app_window(ours)
    }
}

/// Take the foreground for Arrange, remembering who to give it back to. The
/// window must be able to activate first.
fn focus(shared: &OverlayShared, n: &impl Native) {
    let ours = shared.hwnd.load(Ordering::SeqCst);
    let back = return_target(n.foreground(), ours, n);
    shared.return_to.store(back, Ordering::SeqCst);
    n.set_focusable(true);
    let ok = n.try_foreground(ours);
    tracing::info!(ok, "avatar focus");
}

/// On Lock: who gets the keyboard depends on who holds it now.
/// - Fritz: the window that had it before Arrange, if it can still take it,
///   else the app beneath.
/// - The tray's hidden window (a Lock from its menu): the app beneath, which
///   is the app in use before the menu opened — not the pre-Arrange window,
///   which the owner may have left since.
/// - Anyone else (someone else's app, or a visible window of ours): left
///   alone, as DivoraVoice's Lock from Settings is.
///
/// Locked Fritz never keeps the keyboard: with no app open, the desktop
/// takes it (`next_app_window`'s last resort).
fn return_focus(shared: &OverlayShared, n: &impl Native) {
    let ours = shared.hwnd.load(Ordering::SeqCst);
    let saved = shared.return_to.swap(0, Ordering::SeqCst);
    let fg = n.foreground();
    let target = if fg == ours {
        if saved != 0 && n.can_take_foreground(saved) {
            saved
        } else {
            n.next_app_window(ours)
        }
    } else if fg != 0 && n.owned_by_us(fg) && !n.can_take_foreground(fg) {
        n.next_app_window(ours)
    } else {
        return;
    };
    // `fg` is Fritz's, or stands in for him (the tray's window), here.
    if should_return_focus(ours, ours, target, n.can_take_foreground(target)) {
        let ok = n.try_foreground(target);
        tracing::debug!(ok, "avatar focus handed back");
    }
}

/// Park: remember where he stands (unless he's already off screen) and move
/// him off every monitor. Unpark: back to where he stood. A move, never a
/// hide or a show, so the window is never re-shown or re-activated.
fn park(shared: &OverlayShared, on: bool, n: &impl Native) {
    let hwnd = shared.hwnd.load(Ordering::SeqCst);
    if on {
        // Already off screen (parked twice): home stays where he stood.
        if !shared.window_parked.swap(true, Ordering::SeqCst) {
            if let Some(b) = n.bounds(hwnd) {
                *shared.home() = Some(rect_of(b));
            }
        }
        n.move_to(PARK_AT.0, PARK_AT.1);
    } else {
        let home = *shared.home();
        match home.map(|r| n.on_screen(r)) {
            Some(r) => {
                *shared.home() = Some(r);
                n.move_to(r.x, r.y);
                shared.window_parked.store(false, Ordering::SeqCst);
            }
            None => tracing::warn!("unparked with nowhere to stand; Fritz stays off screen"),
        }
    }
}

/// `Watchdog(instance)` once `WATCHDOG` has passed.
fn arm_watchdog(app: &AppHandle, instance: u32) {
    let h = app.clone();
    let spawned = thread::Builder::new()
        .name("fritz-avatar-watchdog".into())
        .spawn(move || {
            thread::sleep(WATCHDOG);
            let _ = step(&h, Input::Watchdog { instance });
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the avatar watchdog");
    }
}

/// A `LeaseTick` every `LEASE_TICK` while this Arrange lasts.
fn arm_lease(app: &AppHandle, instance: u32, gen: u32) {
    let h = app.clone();
    let spawned = thread::Builder::new()
        .name("fritz-avatar-lease".into())
        .spawn(move || loop {
            thread::sleep(LEASE_TICK);
            let _ = step(&h, Input::LeaseTick { instance, gen });
            if !shared(&h).is_some_and(|s| s.core().in_lease(instance, gen)) {
                break;
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the avatar lease");
    }
}

/// The calls the effects make on the window and the desktop. A trait so the
/// order the shell makes them in is tested without a window.
trait Native {
    fn set_ignore_cursor(&self, on: bool);
    fn set_focusable(&self, on: bool);
    fn make_layered_opaque(&self, hwnd: isize);
    fn set_resizable(&self, on: bool);
    fn show(&self);
    fn close(&self);
    fn destroy(&self);
    fn foreground(&self) -> isize;
    fn try_foreground(&self, hwnd: isize) -> bool;
    fn can_take_foreground(&self, hwnd: isize) -> bool;
    fn unmark_non_rude(&self, hwnd: isize);
    fn bounds(&self, hwnd: isize) -> Option<Bounds>;
    fn move_to(&self, x: i32, y: i32);
    /// A window of this process (the tray's hidden window, the avatar).
    fn owned_by_us(&self, hwnd: isize) -> bool;
    /// The app in use: the top ordinary app window that isn't `skip`
    /// (always-on-top ones only if there is no other), else the desktop.
    /// Where the keyboard goes when there's nobody saved to give it back to.
    fn next_app_window(&self, skip: isize) -> isize;
    /// `r`, moved onto a live monitor if its middle is on none.
    fn on_screen(&self, r: OverlayRect) -> OverlayRect;
}

/// The real avatar window, if it still exists, and the desktop.
struct Desktop(Option<WebviewWindow>);

impl Desktop {
    fn of(app: &AppHandle) -> Self {
        Self(app.get_webview_window(OVERLAY_LABEL))
    }
}

impl Native for Desktop {
    fn set_ignore_cursor(&self, on: bool) {
        if let Some(w) = &self.0 {
            let _ = w.set_ignore_cursor_events(on);
        }
    }

    fn set_focusable(&self, on: bool) {
        if let Some(w) = &self.0 {
            let _ = w.set_focusable(on);
        }
    }

    fn make_layered_opaque(&self, hwnd: isize) {
        if !windows_ext::make_layered_opaque(hwnd) {
            tracing::debug!("avatar layered attributes refused");
        }
    }

    fn set_resizable(&self, on: bool) {
        if let Some(w) = &self.0 {
            let _ = w.set_resizable(on);
        }
    }

    fn show(&self) {
        if let Some(w) = &self.0 {
            let _ = w.show();
        }
    }

    fn close(&self) {
        if let Some(w) = &self.0 {
            let _ = w.close();
        }
    }

    fn destroy(&self) {
        if let Some(w) = &self.0 {
            let _ = w.destroy();
        }
    }

    fn foreground(&self) -> isize {
        windows_ext::foreground_window()
    }

    fn try_foreground(&self, hwnd: isize) -> bool {
        windows_ext::try_foreground(hwnd)
    }

    fn can_take_foreground(&self, hwnd: isize) -> bool {
        windows_ext::can_take_foreground(hwnd)
    }

    fn unmark_non_rude(&self, hwnd: isize) {
        crate::non_rude::unmark_non_rude(hwnd);
    }

    fn bounds(&self, hwnd: isize) -> Option<Bounds> {
        windows_ext::window_bounds(hwnd)
    }

    fn move_to(&self, x: i32, y: i32) {
        if let Some(w) = &self.0 {
            let _ = w.set_position(PhysicalPosition::new(x, y));
        }
    }

    fn owned_by_us(&self, hwnd: isize) -> bool {
        windows_ext::owned_by_us(hwnd)
    }

    fn next_app_window(&self, skip: isize) -> isize {
        windows_ext::next_app_window(skip)
    }

    fn on_screen(&self, r: OverlayRect) -> OverlayRect {
        onto_screen(r)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use super::{
        adopt, apply_setters, avatar_url, forget, going, main_thread_effects, no_ready_script,
        park, return_focus, show, wait_closed, window_effect, Native, OverlayShared, BROWSER_ARGS,
        PARK_AT,
    };
    use crate::overlay_machine::{
        should_return_focus, Effect, OverlayRect, Phase, Setter, R1_RUNG,
    };
    use crate::windows_ext::Bounds;

    const OURS: isize = 7;
    const GAME: isize = 42;
    /// The tray's hidden window: ours, and never able to take the keyboard.
    const TRAY: isize = 9;
    /// The app beneath in z-order.
    const BENEATH: isize = 77;

    /// Records the shell's calls, in order.
    #[derive(Default)]
    struct Fake {
        calls: RefCell<Vec<String>>,
        foreground: isize,
        /// Windows that may take the foreground.
        takeable: Vec<isize>,
        /// Windows of our own process besides the avatar.
        ours: Vec<isize>,
        next_app: isize,
        bounds: Option<Bounds>,
        /// Where `on_screen` puts a rect; unchanged when None.
        screen: Option<OverlayRect>,
    }

    impl Fake {
        fn log(&self, call: String) {
            self.calls.borrow_mut().push(call);
        }

        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }

    impl Native for Fake {
        fn set_ignore_cursor(&self, on: bool) {
            self.log(format!("ignore {on}"));
        }
        fn set_focusable(&self, on: bool) {
            self.log(format!("focusable {on}"));
        }
        fn make_layered_opaque(&self, hwnd: isize) {
            self.log(format!("layered {hwnd}"));
        }
        fn set_resizable(&self, on: bool) {
            self.log(format!("resizable {on}"));
        }
        fn show(&self) {
            self.log("show".into());
        }
        fn close(&self) {
            self.log("close".into());
        }
        fn destroy(&self) {
            self.log("destroy".into());
        }
        fn foreground(&self) -> isize {
            self.foreground
        }
        fn try_foreground(&self, hwnd: isize) -> bool {
            self.log(format!("foreground {hwnd}"));
            true
        }
        fn can_take_foreground(&self, hwnd: isize) -> bool {
            self.takeable.contains(&hwnd)
        }
        fn unmark_non_rude(&self, hwnd: isize) {
            self.log(format!("unmark {hwnd}"));
        }
        fn bounds(&self, _hwnd: isize) -> Option<Bounds> {
            self.bounds
        }
        fn move_to(&self, x: i32, y: i32) {
            self.log(format!("move {x} {y}"));
        }
        fn owned_by_us(&self, hwnd: isize) -> bool {
            hwnd == OURS || self.ours.contains(&hwnd)
        }
        fn next_app_window(&self, _skip: isize) -> isize {
            self.next_app
        }
        fn on_screen(&self, r: OverlayRect) -> OverlayRect {
            self.screen.unwrap_or(r)
        }
    }

    fn with_window(phase: Phase) -> OverlayShared {
        let shared = OverlayShared::new();
        adopt(&shared, OURS);
        shared.core().phase = phase;
        shared
    }

    fn at(left: i32, top: i32) -> Option<Bounds> {
        Some(Bounds {
            left,
            top,
            right: left + 300,
            bottom: top + 460,
        })
    }

    /// A source scan, the honest limit without a window: tauri's and tao's
    /// focus call injects an Alt key press when Windows refuses, so the shell
    /// must never make it. The needles are built from halves so this test
    /// doesn't find itself, and are calls (a method or a path), so its own
    /// name doesn't either.
    #[test]
    fn the_overlay_never_calls_set_focus() {
        let src = include_str!("overlay_windows.rs");
        for needle in [concat!(".set_", "focus("), concat!("::set_", "focus(")] {
            assert!(
                !src.contains(needle),
                "the avatar takes the foreground with try_foreground, never {needle}"
            );
        }
    }

    /// The build flags the spike stands on, and their order: click-through
    /// before the HWND is stored (and so before anything can show it).
    #[test]
    fn the_window_is_built_as_the_spike_requires() {
        let src = include_str!("overlay_windows.rs");
        let body =
            &src[src.find("async fn build_window").unwrap()..src.find("fn avatar_url").unwrap()];
        for flag in [
            ".transparent(true)",
            ".decorations(false)",
            ".shadow(false)",
            ".always_on_top(true)",
            ".skip_taskbar(true)",
            ".visible(false)",
            ".focused(false)",
            ".focusable(false)",
            ".additional_browser_args(BROWSER_ARGS)",
        ] {
            assert!(body.contains(flag), "build_window lost {flag}");
        }
        let through = body.find(".set_ignore_cursor_events(true)").unwrap();
        let stored = body.find("shared.hwnd.store(hwnd").unwrap();
        assert!(through < stored);
    }

    /// Every window in tauri.conf.json carries the same browser arguments as
    /// the avatar, or WebView2 forks a second environment. (Spike 0 declares
    /// none; PR 2's control and prompt windows must pass this.)
    #[test]
    fn every_window_shares_the_browser_arguments() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        for w in conf["app"]["windows"].as_array().unwrap() {
            assert_eq!(
                w["additionalBrowserArgs"].as_str(),
                Some(BROWSER_ARGS),
                "{}",
                w["label"]
            );
        }
        assert!(BROWSER_ARGS.contains("--disable-background-timer-throttling"));
    }

    /// The window, tray and status work keeps its order on the main thread;
    /// the window is built by `open`, and the timers are threads.
    #[test]
    fn main_thread_effects_keep_order_and_drop_the_rest() {
        let effects = [
            Effect::Build { instance: 1 },
            Effect::ApplyWindow,
            Effect::Park { on: true },
            Effect::ArmWatchdog { instance: 1 },
            Effect::Emit,
            Effect::Show,
            Effect::Focus,
            Effect::ArmLease {
                instance: 1,
                gen: 2,
            },
            Effect::RefreshTray,
            Effect::ReturnFocus,
            Effect::CloseWindow,
            Effect::DestroyWindow,
        ];
        assert_eq!(
            main_thread_effects(&effects),
            vec![
                Effect::ApplyWindow,
                Effect::Park { on: true },
                Effect::Emit,
                Effect::Show,
                Effect::Focus,
                Effect::RefreshTray,
                Effect::ReturnFocus,
                Effect::CloseWindow,
                Effect::DestroyWindow,
            ]
        );
        assert_eq!(
            main_thread_effects(&[Effect::Emit, Effect::Build { instance: 3 }]),
            [Effect::Emit]
        );
    }

    /// Setters run in the plan's order, on the stored HWND, and Locked also
    /// makes the window unfocusable.
    #[test]
    fn setters_run_in_the_plan_order() {
        let shared = with_window(Phase::Starting { instance: 1 });
        let locked = [
            Setter::IgnoreCursor(true),
            Setter::LayerOpaque,
            Setter::Resizable(false),
        ];
        let fake = Fake::default();
        apply_setters(&shared, &locked, &fake);
        assert_eq!(
            fake.calls(),
            [
                "ignore true",
                "focusable false",
                "layered 7",
                "resizable false"
            ]
        );

        let arrange = [Setter::IgnoreCursor(false), Setter::Resizable(true)];
        let fake = Fake::default();
        apply_setters(&shared, &arrange, &fake);
        assert_eq!(fake.calls(), ["ignore false", "resizable true"]);
    }

    /// Show applies the state and shows only a live window; a window that
    /// closed before the posted Show ran is never shown.
    #[test]
    fn show_only_while_live() {
        for phase in [
            Phase::Closed,
            Phase::Creating {
                instance: 1,
                ready_early: true,
                cancel: false,
            },
            Phase::Closing { instance: 1 },
        ] {
            let shared = with_window(phase);
            let fake = Fake::default();
            show(&shared, &fake);
            assert_eq!(fake.calls(), Vec::<String>::new(), "{phase:?}");
        }
        let shared = with_window(Phase::Shown { instance: 1 });
        assert!(shared.core().reconcile_plan(R1_RUNG).is_some());
        let fake = Fake::default();
        show(&shared, &fake);
        let calls = fake.calls();
        assert_eq!(calls.last().map(String::as_str), Some("show"));
        assert!(
            calls.iter().position(|c| c == "ignore true") < calls.iter().position(|c| c == "show")
        );
    }

    /// Focus remembers who to give the keyboard back to: the foreground
    /// window when it is someone else's and usable, else the app beneath
    /// (Arrange from the tray finds the tray's own hidden window in front).
    /// It makes the window focusable before taking it.
    #[test]
    fn focus_remembers_who_gets_the_keyboard_back() {
        for (fg, back) in [(GAME, GAME), (OURS, BENEATH), (0, BENEATH), (TRAY, BENEATH)] {
            let shared = with_window(Phase::Painted { instance: 1 });
            let fake = Fake {
                foreground: fg,
                takeable: vec![GAME, BENEATH],
                ours: vec![TRAY],
                next_app: BENEATH,
                ..Fake::default()
            };
            window_effect(&shared, Effect::Focus, &fake);
            assert_eq!(shared.return_to.load(Ordering::SeqCst), back, "fg {fg}");
            assert_eq!(fake.calls(), ["focusable true", "foreground 7"], "fg {fg}");
        }
        // Someone else's window that can't take it back (minimised, gone).
        let shared = with_window(Phase::Painted { instance: 1 });
        let fake = Fake {
            foreground: GAME,
            next_app: BENEATH,
            ..Fake::default()
        };
        window_effect(&shared, Effect::Focus, &fake);
        assert_eq!(shared.return_to.load(Ordering::SeqCst), BENEATH);
    }

    /// With Fritz holding the keyboard, Lock gives it to the saved window if
    /// it can still take it, else to the app in use; and forgets the saved
    /// window either way.
    #[test]
    fn lock_gives_fritzs_keyboard_back() {
        for saved in [GAME, 0] {
            for takeable in [vec![], vec![GAME], vec![GAME, BENEATH], vec![BENEATH]] {
                let shared = with_window(Phase::Painted { instance: 1 });
                shared.return_to.store(saved, Ordering::SeqCst);
                let fake = Fake {
                    foreground: OURS,
                    takeable: takeable.clone(),
                    next_app: BENEATH,
                    ..Fake::default()
                };
                return_focus(&shared, &fake);
                let target = if saved != 0 && takeable.contains(&saved) {
                    saved
                } else {
                    BENEATH
                };
                let back = should_return_focus(OURS, OURS, target, takeable.contains(&target));
                let expected: Vec<String> = if back {
                    vec![format!("foreground {target}")]
                } else {
                    Vec::new()
                };
                let case = format!("saved {saved} takeable {takeable:?}");
                assert_eq!(fake.calls(), expected, "{case}");
                assert_eq!(shared.return_to.load(Ordering::SeqCst), 0, "{case}");
            }
        }
    }

    /// Lock from the tray finds the tray's hidden window in front, not
    /// Fritz: the app in use before the menu opened gets the keyboard — not
    /// the pre-Arrange window, which the owner may have left since.
    #[test]
    fn lock_from_the_tray_hands_on_to_the_app_in_use() {
        for saved in [GAME, TRAY, 0] {
            let shared = with_window(Phase::Painted { instance: 1 });
            shared.return_to.store(saved, Ordering::SeqCst);
            let fake = Fake {
                foreground: TRAY,
                takeable: vec![GAME, BENEATH],
                ours: vec![TRAY],
                next_app: BENEATH,
                ..Fake::default()
            };
            return_focus(&shared, &fake);
            assert_eq!(
                fake.calls(),
                [format!("foreground {BENEATH}")],
                "saved {saved}"
            );
        }
    }

    /// Someone else's window in front (Lock from Settings, the lease while
    /// the owner works elsewhere), or a visible window of ours: left alone.
    #[test]
    fn lock_leaves_someone_elses_keyboard_alone() {
        const PROMPT: isize = 11; // a visible window of ours (PR 2's prompt)
        for fg in [GAME, PROMPT, 0] {
            let shared = with_window(Phase::Painted { instance: 1 });
            shared.return_to.store(GAME, Ordering::SeqCst);
            let fake = Fake {
                foreground: fg,
                takeable: vec![GAME, BENEATH, PROMPT],
                ours: vec![PROMPT],
                next_app: BENEATH,
                ..Fake::default()
            };
            return_focus(&shared, &fake);
            assert_eq!(fake.calls(), Vec::<String>::new(), "fg {fg}");
            assert_eq!(shared.return_to.load(Ordering::SeqCst), 0, "fg {fg}");
        }
    }

    /// With no app open at all, the desktop takes the keyboard (that is what
    /// `next_app_window` returns then): Locked Fritz never keeps it.
    #[test]
    fn lock_over_an_empty_desktop_hands_it_to_the_desktop() {
        const DESKTOP: isize = 3;
        for fg in [OURS, TRAY] {
            let shared = with_window(Phase::Painted { instance: 1 });
            let fake = Fake {
                foreground: fg,
                takeable: vec![DESKTOP],
                ours: vec![TRAY],
                next_app: DESKTOP,
                ..Fake::default()
            };
            return_focus(&shared, &fake);
            assert_eq!(fake.calls(), [format!("foreground {DESKTOP}")], "fg {fg}");
        }
    }

    /// Parking remembers where he stood and moves him off every monitor;
    /// unparking puts him back there. Never a show or a hide.
    #[test]
    fn park_remembers_home_and_returns() {
        let shared = with_window(Phase::Painted { instance: 1 });
        let fake = Fake {
            bounds: at(1500, 600),
            ..Fake::default()
        };
        park(&shared, true, &fake);
        park(&shared, false, &fake);
        assert_eq!(
            fake.calls(),
            [
                format!("move {} {}", PARK_AT.0, PARK_AT.1),
                "move 1500 600".into()
            ]
        );
        let home = shared.home().unwrap();
        assert_eq!(
            (home.x, home.y, home.width, home.height),
            (1500, 600, 300, 460)
        );
    }

    /// Parked twice (the window already off screen): home stays where he
    /// last stood, not at the parking spot.
    #[test]
    fn park_twice_keeps_the_real_home() {
        let shared = with_window(Phase::Painted { instance: 1 });
        park(
            &shared,
            true,
            &Fake {
                bounds: at(200, 300),
                ..Fake::default()
            },
        );
        park(
            &shared,
            true,
            &Fake {
                bounds: at(PARK_AT.0, PARK_AT.1),
                ..Fake::default()
            },
        );
        let fake = Fake::default();
        park(&shared, false, &fake);
        assert_eq!(fake.calls(), ["move 200 300"]);
    }

    /// Unparked to a spot no monitor shows any more (undocked since): he
    /// comes back onto a live one, and that is home now.
    #[test]
    fn unpark_comes_back_onto_a_live_monitor() {
        let shared = with_window(Phase::Painted { instance: 1 });
        *shared.home() = Some(OverlayRect {
            x: 4000,
            y: 500,
            width: 300,
            height: 460,
        });
        let fixed = OverlayRect {
            x: 1500,
            y: 500,
            width: 300,
            height: 460,
        };
        let fake = Fake {
            screen: Some(fixed),
            ..Fake::default()
        };
        park(&shared, false, &fake);
        assert_eq!(fake.calls(), ["move 1500 500"]);
        assert_eq!(shared.home().map(|r| r.x), Some(1500));
    }

    /// A Park effect goes where the core says when it runs, not where it
    /// said when it was queued: a posted park that lands after a newer
    /// inline unpark doesn't send him off screen behind the core's back.
    #[test]
    fn a_stale_park_follows_the_core() {
        let shared = with_window(Phase::Painted { instance: 1 });
        *shared.home() = Some(OverlayRect {
            x: 100,
            y: 200,
            width: 300,
            height: 460,
        });
        shared.core().parked = false;
        let fake = Fake {
            bounds: at(100, 200),
            ..Fake::default()
        };
        window_effect(&shared, Effect::Park { on: true }, &fake);
        assert_eq!(fake.calls(), ["move 100 200"]);

        shared.core().parked = true;
        let fake = Fake {
            bounds: at(100, 200),
            ..Fake::default()
        };
        window_effect(&shared, Effect::Park { on: false }, &fake);
        assert_eq!(fake.calls(), [format!("move {} {}", PARK_AT.0, PARK_AT.1)]);

        // No live window: nothing to move.
        let shared = with_window(Phase::Closing { instance: 1 });
        shared.core().parked = true;
        let fake = Fake::default();
        window_effect(&shared, Effect::Park { on: true }, &fake);
        assert_eq!(fake.calls(), Vec::<String>::new());
    }

    /// With nowhere remembered, unparking leaves him where he is rather than
    /// guessing.
    #[test]
    fn unpark_without_home_stays_put() {
        let shared = with_window(Phase::Painted { instance: 1 });
        let fake = Fake::default();
        park(&shared, false, &fake);
        assert_eq!(fake.calls(), Vec::<String>::new());
    }

    /// A window going remembers where it stood, so a reopen after an
    /// Arrange drag and Alt+F4 comes back where he was dragged to; a parked
    /// one keeps the home it had.
    #[test]
    fn going_remembers_where_he_stood() {
        let shared = with_window(Phase::Painted { instance: 1 });
        *shared.home() = Some(OverlayRect {
            x: 1,
            y: 2,
            width: 300,
            height: 460,
        });
        going(
            &shared,
            &Fake {
                bounds: at(900, 400),
                ..Fake::default()
            },
        );
        assert_eq!(shared.home().map(|r| (r.x, r.y)), Some((900, 400)));

        // Parked, then going: the park spot is not where he stood.
        park(
            &shared,
            true,
            &Fake {
                bounds: at(900, 400),
                ..Fake::default()
            },
        );
        going(
            &shared,
            &Fake {
                bounds: at(PARK_AT.0, PARK_AT.1),
                ..Fake::default()
            },
        );
        assert_eq!(shared.home().map(|r| (r.x, r.y)), Some((900, 400)));
    }

    /// A monitor far to the left of the primary has coordinates like the
    /// park spot's. Dragged there, he stands there: parking, going and the
    /// next unpark all remember that spot.
    #[test]
    fn a_monitor_far_to_the_left_is_somewhere_to_stand() {
        let shared = with_window(Phase::Painted { instance: 1 });
        let far_left = Fake {
            bounds: at(-6500, 300),
            ..Fake::default()
        };
        going(&shared, &far_left);
        assert_eq!(shared.home().map(|r| (r.x, r.y)), Some((-6500, 300)));
        park(&shared, true, &far_left);
        let fake = Fake::default();
        park(&shared, false, &fake);
        assert_eq!(fake.calls(), ["move -6500 300"]);
        assert!(!shared.window_parked.load(Ordering::SeqCst));
    }

    /// Set before the move off screen and cleared after the move back, so
    /// whoever reads where he stands mid-move gets home, never the park spot.
    #[test]
    fn the_shell_knows_its_own_park() {
        let shared = with_window(Phase::Painted { instance: 1 });
        let fake = Fake {
            bounds: at(1500, 600),
            ..Fake::default()
        };
        assert!(!shared.window_parked.load(Ordering::SeqCst));
        park(&shared, true, &fake);
        assert!(shared.window_parked.load(Ordering::SeqCst));
        park(&shared, false, &fake);
        assert!(!shared.window_parked.load(Ordering::SeqCst));
        // A new window, or none, starts unparked.
        park(&shared, true, &fake);
        adopt(&shared, 7);
        assert!(!shared.window_parked.load(Ordering::SeqCst));
        park(&shared, true, &fake);
        forget(&shared);
        assert!(!shared.window_parked.load(Ordering::SeqCst));
    }

    /// Destroying takes the mark off first; closing goes through the window's
    /// own close.
    #[test]
    fn destroy_unmarks_first_and_close_closes() {
        let shared = with_window(Phase::Starting { instance: 1 });
        let fake = Fake::default();
        window_effect(&shared, Effect::DestroyWindow, &fake);
        assert_eq!(fake.calls(), ["unmark 7", "destroy"]);

        let fake = Fake::default();
        window_effect(&shared, Effect::CloseWindow, &fake);
        assert_eq!(fake.calls(), ["close"]);
    }

    /// A new window starts with clean cells; a going one loses its mark; a
    /// destroyed one leaves nothing behind but where he stood.
    #[test]
    fn the_cells_follow_the_window() {
        let shared = OverlayShared::new();
        shared.return_to.store(GAME, Ordering::SeqCst);
        *shared.home() = Some(OverlayRect {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        });
        adopt(&shared, OURS);
        assert_eq!(shared.hwnd.load(Ordering::SeqCst), OURS);
        assert_eq!(shared.return_to.load(Ordering::SeqCst), 0);

        let fake = Fake::default();
        going(&shared, &fake);
        assert_eq!(fake.calls(), ["unmark 7"]);

        shared.return_to.store(GAME, Ordering::SeqCst);
        forget(&shared);
        assert_eq!(shared.hwnd.load(Ordering::SeqCst), 0);
        assert_eq!(shared.return_to.load(Ordering::SeqCst), 0);
        assert!(shared.home().is_some());
    }

    /// `open` waits for a closing window to finish, but not forever.
    #[test]
    fn open_waits_for_closed_but_not_forever() {
        let poll = Duration::from_millis(5);
        let shared = OverlayShared::new();
        assert!(wait_closed(&shared, Duration::ZERO, poll));

        shared.core().phase = Phase::Closing { instance: 1 };
        assert!(!wait_closed(&shared, Duration::from_millis(30), poll));

        let shared = std::sync::Arc::new(shared);
        let s = std::sync::Arc::clone(&shared);
        let closer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            s.core().phase = Phase::Closed;
        });
        assert!(wait_closed(&shared, Duration::from_secs(5), poll));
        closer.join().unwrap();
    }

    /// The page reads its instance from the query, and the debug switch sets
    /// the global the page checks.
    #[test]
    fn the_page_url_and_the_debug_switch() {
        assert_eq!(avatar_url(3), "avatar.html?instance=3");
        assert_eq!(no_ready_script(false), None);
        assert_eq!(
            no_ready_script(true).as_deref(),
            Some("window.__FRITZ_AVATAR_NO_READY__ = true;")
        );
    }
}
