//! The study scene behind the icons (plan 14, PR 12): a wallpaper window on
//! each monitor, up to four, drawn in the desktop's wallpaper layer below
//! the icons, while `wallpaper.on` is (it is by default; Settings ›
//! Appearance turns it off and gives the owner's own wallpaper back).
//!
//! Plumbing from LorraineAtelier's `lib.rs` (the owner's own app) at
//! d62b7ec: its slots, `ensure_attached`, `reparent_controller` and the
//! 600 ms watchdog. Each slot's page lives in a hidden webview window of its
//! own, and its WebView2 controller is moved into a host window in the
//! layer (`host.rs`, `workerw.rs`). The watch puts right whatever goes
//! wrong, from what it sees rather than from flags: a host that died (an
//! Explorer restart) means its controller is dead too, so the window is
//! built anew; so does a monitor that moved, changed size or scale, a
//! homing that failed, a window someone else closed, and a resume from
//! sleep (Lorraine's lesson: a suspended controller can come back blank
//! under a live host). An unplugged monitor's slot is taken down.
//!
//! Changed on the way. The decision is one pure function ([`plan`]) of what
//! the watch sees. The watch is its own thread, and only window work runs on
//! the main thread: a webview window is asked for from the watch's thread
//! (`build` only posts its making to the event loop) and then its handle,
//! which waits until it is made or has failed; and asking Progman can wait a
//! second on a hung Explorer. Hosts are made, placed and destroyed on the
//! main thread (a window is its maker's), and a slot is left alone while its
//! homing is out there. A slot whose homing isn't heard from in about six
//! seconds is built anew. A host is where its monitor is, or it is put back
//! there: Lorraine placed one only when she made it, but the layer moves
//! when the desktop's top-left does (a monitor plugged in left of the
//! primary), and wry moves a controller's parent to the corner whenever its
//! own hidden window is resized. Windows are built only for the monitors
//! there are, when they are wanted, and labelled `wallpaper-<slot>-<n>`
//! (capabilities/wallpaper.json grants `wallpaper-*`), not declared in
//! tauri.conf.json: a declared window would be made at every start, wanted
//! or not. Lorraine's fullscreen pause, off by default there because hiding
//! the controller flashes, is a page pause here: a monitor covered by
//! another app's fullscreen window, or every monitor while the session is
//! locked, is told so (`wallpaper:covered`) and its page stops drawing;
//! nothing is hidden, so nothing flashes. No gift, buddy, casting or
//! hit-rect code.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;
use std::time::Duration;

use tauri::webview::Color;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use windows::Win32::Foundation::RECT;

use crate::settings::Theme;
use crate::windows_ext::{hwnd, Bounds};
use crate::workerw::{self, Found};

/// Every wallpaper window's label starts so (capabilities/wallpaper.json).
pub const LABEL_PREFIX: &str = "wallpaper-";
/// To one wallpaper window: `true` while its monitor is covered (another
/// app's fullscreen window, or the lock screen), `false` when it isn't.
pub const COVERED_EVENT: &str = "wallpaper:covered";
const PAGE: &str = "wallpaper.html";
const TITLE: &str = "Mister Fritz study scene";
/// Monitors drawn on, at most: Lorraine's four.
pub const MAX_SLOTS: usize = 4;
/// The watch's pace: Lorraine's 600 ms.
const TICK: Duration = Duration::from_millis(600);
/// Looks a homing may go unheard before the slot is built anew.
const UNHEARD: u32 = 10;
/// Looks a slot whose window couldn't be built waits before trying again,
/// doubled with each failure in a row, to at most [`MAX_COOLDOWN`]: each
/// failed window stays named in Tauri's own list for the app's life
/// (nothing can take out a window the event loop never made).
const COOLDOWN: u32 = 10;
/// About ten minutes.
const MAX_COOLDOWN: u32 = 1000;

/// How long to wait after `failures` failed builds in a row. Pure.
pub fn cooldown_after(failures: u32) -> u32 {
    COOLDOWN.saturating_mul(1u32 << failures.saturating_sub(1).min(16)).min(MAX_COOLDOWN)
}

/// A monitor as a slot is built for it: a change to either is a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub bounds: Bounds,
    pub dpi: u32,
}

/// What became of a slot's host, as the watch sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSeen {
    /// None was made, or it was taken down.
    None,
    /// It is there, a child of `parent`, covering `at` on the screen.
    Alive { parent: isize, at: Option<Bounds> },
    /// One was made and is gone: Explorer restarted.
    Dead,
}

/// A slot as [`plan`] reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub window: bool,
    /// The slot has a window label, but no such window any more: someone
    /// else closed it.
    pub gone: bool,
    pub built_for: Option<Geometry>,
    pub host: HostSeen,
    pub homed: bool,
    pub failed: bool,
    pub busy: bool,
}

/// Why a slot is built anew, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    HostDied,
    WindowGone,
    HomingFailed,
    MonitorChanged,
    Resumed,
    LayerChanged,
}

/// What the watch does to a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Nothing,
    /// Not wanted (off, or its monitor is gone): close its window, destroy
    /// its host.
    TearDown,
    /// Wanted with no window yet: build one.
    Build,
    /// Its window can't be trusted any more: tear it down and build anew.
    Rebuild(Why),
    /// Its window is ready: make it a host, if it has none, and move the
    /// controller in.
    Home,
    /// Homed, but its host isn't over its monitor any more: put it back.
    Place,
    /// Wanted, built, and nowhere to go yet (Explorer isn't up).
    Wait,
}

/// The one decision. `want` is the slot's monitor while the scene is on
/// (`None`: off, or no such monitor); `layer` is the wallpaper layer now,
/// if there is one; `forced` is a resume. A slot whose homing is out there
/// is left alone. Pure.
pub fn plan(v: View, want: Option<Geometry>, layer: Option<isize>, forced: bool) -> Step {
    if v.busy {
        return Step::Nothing;
    }
    let Some(g) = want else {
        let anything = v.window || v.host != HostSeen::None;
        return if anything { Step::TearDown } else { Step::Nothing };
    };
    if !v.window {
        return Step::Build;
    }
    if v.gone {
        return Step::Rebuild(Why::WindowGone);
    }
    if v.host == HostSeen::Dead {
        return Step::Rebuild(Why::HostDied);
    }
    if v.failed {
        return Step::Rebuild(Why::HomingFailed);
    }
    if v.built_for != Some(g) {
        return Step::Rebuild(Why::MonitorChanged);
    }
    if forced && v.homed {
        return Step::Rebuild(Why::Resumed);
    }
    if let (HostSeen::Alive { parent, .. }, Some(layer)) = (v.host, layer) {
        if parent != layer {
            return Step::Rebuild(Why::LayerChanged);
        }
    }
    if let (true, HostSeen::Alive { at, .. }) = (v.homed, v.host) {
        return if at == Some(g.bounds) || layer.is_none() { Step::Nothing } else { Step::Place };
    }
    if layer.is_some() {
        Step::Home
    } else {
        Step::Wait
    }
}

/// What the watch keeps of a slot.
#[derive(Debug, Clone)]
struct Slot {
    /// Counts builds, so a report from an older window is ignored.
    gen: u64,
    label: Option<String>,
    built_for: Option<Geometry>,
    host: isize,
    homed: bool,
    failed: bool,
    /// Looks since its homing went out; 0 when none is.
    busy: u32,
    /// Looks left before a failed build is tried again.
    cooldown: u32,
    /// Builds that failed in a row.
    failures: u32,
    /// What its page was last told.
    covered: bool,
    /// The layer is known to be missing, and that was said in the log.
    waiting_said: bool,
}

impl Slot {
    const EMPTY: Self = Self {
        gen: 0,
        label: None,
        built_for: None,
        host: 0,
        homed: false,
        failed: false,
        busy: 0,
        cooldown: 0,
        failures: 0,
        covered: false,
        waiting_said: false,
    };

    /// Taken down: nothing of it is left but how its builds have gone, and
    /// its count moves on, so a homing still on its way for what was taken
    /// down is ignored.
    fn clear(&mut self) {
        *self = Self { gen: self.gen.wrapping_add(1), failures: self.failures, ..Self::EMPTY };
    }
}

static SLOTS: Mutex<[Slot; MAX_SLOTS]> = Mutex::new([const { Slot::EMPTY }; MAX_SLOTS]);
/// A resume asked for every homed slot to be built anew.
static FORCE: AtomicBool = AtomicBool::new(false);
/// Quit has begun: the watch does nothing more.
static STOPPING: AtomicBool = AtomicBool::new(false);
static POKE: OnceLock<Sender<()>> = OnceLock::new();

fn slots() -> MutexGuard<'static, [Slot; MAX_SLOTS]> {
    SLOTS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A slot's window label for build `gen`. Pure.
pub fn label(slot: usize, gen: u64) -> String {
    format!("{LABEL_PREFIX}{slot}-{gen}")
}

/// Whether `label` is a wallpaper window's. Pure.
pub fn is_wallpaper(label: &str) -> bool {
    label.strip_prefix(LABEL_PREFIX).is_some_and(|rest| {
        let mut parts = rest.split('-');
        let slot = parts.next().and_then(|p| p.parse::<usize>().ok());
        let gen = parts.next().and_then(|p| p.parse::<u64>().ok());
        slot.is_some_and(|s| s < MAX_SLOTS) && gen.is_some() && parts.next().is_none()
    })
}

/// The mood's `--bg` (src/divora/tokens.css), which the scene is drawn on
/// and its hosts and webviews show before the page paints. Pure.
pub fn backdrop(theme: &Theme) -> (u8, u8, u8) {
    match (theme.theme.as_str(), theme.mood.as_str()) {
        ("light", "ink") => (0xef, 0xe9, 0xdf),
        ("light", "midnight") => (0xe1, 0xe4, 0xee),
        ("light", _) => (0xe6, 0xe2, 0xf1),
        (_, "ink") => (0x08, 0x07, 0x0b),
        (_, "midnight") => (0x04, 0x04, 0x06),
        _ => (0x07, 0x06, 0x0d),
    }
}

/// A monitor's size in the logical pixels a window is built with. Pure.
pub fn logical_size(g: Geometry) -> (f64, f64) {
    let scale = f64::from(g.dpi.max(48)) / 96.0;
    (
        (f64::from(g.bounds.width().max(1)) / scale).max(1.0),
        (f64::from(g.bounds.height().max(1)) / scale).max(1.0),
    )
}

/// What the watch sees of slot `s`; `gone`, whether its window was closed
/// by someone else (asked of the app outside the slots' lock).
fn view_of(s: &Slot, gone: bool) -> View {
    let host = if s.host == 0 {
        HostSeen::None
    } else if workerw::is_window(s.host) {
        HostSeen::Alive { parent: crate::host::parent_of(s.host), at: crate::windows_ext::window_bounds(s.host) }
    } else {
        HostSeen::Dead
    };
    View {
        window: s.label.is_some(),
        gone,
        built_for: s.built_for,
        host,
        homed: s.homed,
        failed: s.failed,
        busy: s.busy > 0,
    }
}

/// Start the watch: a look at once, then every 600 ms, or sooner when
/// poked (a setting changed, a resume).
pub fn start(app: AppHandle) {
    let (tx, rx) = mpsc::channel::<()>();
    if POKE.set(tx).is_err() {
        tracing::warn!("the study scene's watch is already running");
        return;
    }
    let spawned = thread::Builder::new()
        .name("fritz-wallpaper-watch".into())
        .spawn(move || loop {
            if STOPPING.load(Ordering::SeqCst) {
                break;
            }
            crate::logs::guard("wallpaper watch", || look(&app));
            match rx.recv_timeout(TICK) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the study scene's watch; no wallpaper this run");
    }
}

/// Look again now, not at the next tick.
pub fn poke() {
    if let Some(tx) = POKE.get() {
        let _ = tx.send(());
    }
}

/// From the power watch: the PC woke, so every homed slot is built anew.
pub fn resumed() {
    FORCE.store(true, Ordering::SeqCst);
    poke();
}

/// One look: every slot planned and stepped.
fn look(app: &AppHandle) {
    let (on, theme) = crate::settings::peek(|s| (s.wallpaper.on, s.theme.clone()));
    crate::host::set_backdrop(backdrop(&theme));
    let monitors = if on { workerw::monitors() } else { Vec::new() };
    // Progman is asked only while the scene is wanted.
    let layer = if on { workerw::layer() } else { None };
    let forced = FORCE.swap(false, Ordering::SeqCst);
    let locked = crate::cover::locked() == Some(true);
    for i in 0..MAX_SLOTS {
        let want = monitors.get(i).map(|m| Geometry { bounds: m.bounds, dpi: m.dpi });
        // Twice at most: a window built now is homed now, not a tick later.
        for _ in 0..2 {
            if STOPPING.load(Ordering::SeqCst) {
                return;
            }
            // Only this thread sets a slot's label, so it holds until the
            // lock is taken again below.
            let label = slots()[i].label.clone();
            let gone = label.as_deref().is_some_and(|l| app.get_webview_window(l).is_none());
            let step = {
                let mut all = slots();
                let s = &mut all[i];
                if s.busy > 0 {
                    s.busy += 1;
                    if s.busy > UNHEARD {
                        tracing::warn!(slot = i, "the study scene's homing went unheard");
                        s.busy = 0;
                        s.failed = true;
                    }
                }
                if s.cooldown > 0 {
                    s.cooldown -= 1;
                    Step::Nothing
                } else {
                    plan(view_of(s, gone), want, layer.map(Found::window), forced)
                }
            };
            let again = act(app, i, step, want, layer, &theme);
            if !again {
                break;
            }
        }
        tell_covered(app, i, want, locked);
    }
}

/// Take `step` for slot `i`; whether to plan the slot again at once.
fn act(app: &AppHandle, i: usize, step: Step, want: Option<Geometry>, layer: Option<Found>, theme: &Theme) -> bool {
    match step {
        Step::Nothing => false,
        Step::Wait => {
            let mut all = slots();
            if !all[i].waiting_said {
                all[i].waiting_said = true;
                tracing::info!(slot = i, "the study scene waits for the desktop's wallpaper layer");
            }
            false
        }
        Step::TearDown => {
            tear_down(app, i);
            tracing::info!(slot = i, "the study scene is taken down");
            false
        }
        Step::Build => want.is_some_and(|g| build(app, i, g, theme)),
        Step::Rebuild(why) => {
            tracing::info!(slot = i, ?why, "the study scene is built anew");
            tear_down(app, i);
            want.is_some_and(|g| build(app, i, g, theme))
        }
        Step::Home => {
            if let (Some(g), Some(layer)) = (want, layer) {
                home(app, i, g, layer);
            }
            false
        }
        Step::Place => {
            if let (Some(g), Some(layer)) = (want, layer) {
                place(app, i, g, layer);
            }
            false
        }
    }
}

/// Close the slot's window and destroy its host. Both are done on the main
/// thread, the host's first as it happens (the window's own destruction
/// waits behind it in the event loop): the controller survives losing its
/// host, as it does an Explorer restart, until its window goes.
fn tear_down(app: &AppHandle, i: usize) {
    let (label, host) = {
        let mut all = slots();
        let s = &mut all[i];
        let taken = (s.label.take(), s.host);
        s.clear();
        taken
    };
    if let Some(w) = label.as_deref().and_then(|l| app.get_webview_window(l)) {
        let _ = w.destroy();
    }
    if host != 0 {
        let _ = app.run_on_main_thread(move || crate::host::destroy(host));
    }
}

/// Build the slot's window for `g`, hidden; whether it was built.
fn build(app: &AppHandle, i: usize, g: Geometry, theme: &Theme) -> bool {
    let gen = {
        let mut all = slots();
        all[i].gen += 1;
        all[i].gen
    };
    let name = label(i, gen);
    let (w, h) = logical_size(g);
    let (r, gr, b) = backdrop(theme);
    let built = WebviewWindowBuilder::new(app, &name, WebviewUrl::App(PAGE.into()))
        .title(TITLE)
        .inner_size(w, h)
        .decorations(false)
        .shadow(false)
        .resizable(false)
        .maximizable(false)
        .minimizable(false)
        .skip_taskbar(true)
        // Never shown, never focused: only its controller is seen, in the
        // host.
        .visible(false)
        .focused(false)
        .focusable(false)
        .background_color(Color(r, gr, b, 255))
        .additional_browser_args(crate::overlay_windows::BROWSER_ARGS)
        .build()
        // Built from here, the window is only asked for: its handle waits
        // until the event loop has made it, and fails if it couldn't.
        .and_then(|w| w.hwnd().map(|_| w));
    let error = built.as_ref().err().map(ToString::to_string);
    let kept = {
        let mut all = slots();
        note_build(&mut all[i], gen, &name, g, error.is_none())
    };
    match (kept, built) {
        (Built::Kept, _) => true,
        (Built::Unwanted, Ok(w)) => {
            let _ = w.destroy();
            false
        }
        (_, _) => {
            tracing::warn!(slot = i, error = error.as_deref().unwrap_or("?"), "couldn't build the study scene's window");
            // Asked to go, in case the event loop made it after all; if it
            // never did, its name stays in Tauri's list (see COOLDOWN).
            if let Some(w) = app.get_webview_window(&name) {
                let _ = w.destroy();
            }
            false
        }
    }
}

/// What became of a build, for the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Built {
    /// Made, and the slot's window now.
    Kept,
    /// Made, but the slot was taken down meanwhile: destroy it.
    Unwanted,
    /// Not made: the slot waits, longer each time in a row.
    Failed,
}

/// Note build `gen` of the window `name` for `g` in slot `s`, made or not
/// (`made`). Pure but for the slot.
fn note_build(s: &mut Slot, gen: u64, name: &str, g: Geometry, made: bool) -> Built {
    if !made {
        s.failures = s.failures.saturating_add(1);
        s.cooldown = cooldown_after(s.failures);
        return Built::Failed;
    }
    if s.gen != gen {
        return Built::Unwanted;
    }
    s.label = Some(name.to_owned());
    s.built_for = Some(g);
    s.failures = 0;
    Built::Kept
}

/// Move the slot's controller into a host in `layer` (made first if it has
/// none), on the main thread. The slot is busy until that is heard.
fn home(app: &AppHandle, i: usize, g: Geometry, layer: Found) {
    let Some(origin) = workerw::client_origin(layer.window()) else {
        return;
    };
    let (label, gen, host) = {
        let mut all = slots();
        let s = &mut all[i];
        let Some(label) = s.label.clone() else {
            return;
        };
        s.busy = 1;
        s.waiting_said = false;
        (label, s.gen, s.host)
    };
    if matches!(layer, Found::Progman(_)) {
        tracing::info!(slot = i, "no WorkerW: the study scene goes in Progman itself");
    }
    let handle = app.clone();
    let posted = app.run_on_main_thread(move || {
        let done = crate::logs::guard("wallpaper homing", || home_on_main(&handle, i, gen, &label, host, layer, g, origin));
        if done.is_none() {
            heard(i, gen, false);
        }
    });
    if posted.is_err() {
        heard(i, gen, false);
    }
}

/// Main thread.
#[allow(clippy::too_many_arguments)]
fn home_on_main(app: &AppHandle, i: usize, gen: u64, label: &str, host: isize, layer: Found, g: Geometry, origin: (i32, i32)) {
    // The window first: no host is made over the owner's wallpaper for a
    // window that isn't there.
    let Some(window) = app.get_webview_window(label) else {
        heard(i, gen, false);
        return;
    };
    let host = if workerw::is_window(host) {
        // Kept, and put back where its monitor is.
        crate::host::place(host, g.bounds, origin);
        host
    } else {
        let Some(made) = crate::host::create(layer.window(), g.bounds, origin) else {
            tracing::warn!(slot = i, "couldn't make the study scene's host");
            heard(i, gen, false);
            return;
        };
        let (cw, ch) = crate::host::client_size(made);
        if (cw, ch) != (g.bounds.width(), g.bounds.height()) {
            tracing::warn!(slot = i, cw, ch, "the study scene's host isn't its monitor's size");
        }
        let mut all = slots();
        if all[i].gen != gen {
            drop(all);
            crate::host::destroy(made);
            return;
        }
        all[i].host = made;
        made
    };
    let (width, height) = (g.bounds.width(), g.bounds.height());
    let asked = window.with_webview(move |pw| {
        let controller = pw.controller();
        // SAFETY: the webview's own controller, on the main thread, moved
        // into a live window of this thread's; a host gone since only makes
        // the call fail.
        let moved = unsafe { controller.SetParentWindow(hwnd(host)) }.is_ok();
        if moved {
            // SAFETY: as above.
            unsafe {
                let _ = controller.SetBounds(RECT { left: 0, top: 0, right: width, bottom: height });
                let _ = controller.SetIsVisible(true);
            }
        }
        heard(i, gen, moved);
    });
    if asked.is_err() {
        heard(i, gen, false);
    }
}

/// Put slot `i`'s host back over its monitor, and its controller's bounds
/// with it (wry sets them to its hidden window's size), on the main thread.
fn place(app: &AppHandle, i: usize, g: Geometry, layer: Found) {
    let Some(origin) = workerw::client_origin(layer.window()) else {
        return;
    };
    let (label, host) = {
        let all = slots();
        match &all[i].label {
            Some(l) if all[i].host != 0 => (l.clone(), all[i].host),
            _ => return,
        }
    };
    tracing::info!(slot = i, "the study scene's host is put back over its monitor");
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        crate::logs::guard("wallpaper placing", || {
            crate::host::place(host, g.bounds, origin);
            if let Some(window) = handle.get_webview_window(&label) {
                let (width, height) = (g.bounds.width(), g.bounds.height());
                let _ = window.with_webview(move |pw| {
                    // SAFETY: the webview's own controller, on the main thread.
                    let _ = unsafe { pw.controller().SetBounds(RECT { left: 0, top: 0, right: width, bottom: height }) };
                });
            }
        });
    });
}

/// A homing's outcome, for build `gen` of slot `i`.
fn heard(i: usize, gen: u64, moved: bool) {
    let mut all = slots();
    let s = &mut all[i];
    if s.gen != gen {
        return;
    }
    s.busy = 0;
    if moved {
        s.homed = true;
        tracing::info!(slot = i, "the study scene is behind the icons");
    } else {
        s.failed = true;
        tracing::warn!(slot = i, "couldn't move the study scene behind the icons");
    }
}

/// Whether a monitor is covered: every one while the session is locked
/// (`fullscreen` isn't asked then), else when another app's fullscreen
/// window is over it.
pub fn covered_now(locked: bool, fullscreen: impl FnOnce() -> bool) -> bool {
    locked || fullscreen()
}

/// Whether a homed slot's monitor is covered now, told to its page when
/// that changes: every monitor while the session is locked, else one with
/// another app's fullscreen window at its centre.
fn tell_covered(app: &AppHandle, i: usize, want: Option<Geometry>, locked: bool) {
    let (label, was) = {
        let all = slots();
        let s = &all[i];
        match (&s.label, s.homed) {
            (Some(l), true) => (l.clone(), s.covered),
            _ => return,
        }
    };
    let now = covered_now(locked, || want.is_some_and(|g| crate::windows_ext::fullscreen_over(g.bounds)));
    if now == was {
        return;
    }
    {
        let mut all = slots();
        if all[i].label.as_deref() != Some(label.as_str()) {
            return;
        }
        all[i].covered = now;
    }
    tracing::debug!(slot = i, covered = now, "the study scene's monitor");
    let _ = app.emit_to(label.as_str(), COVERED_EVENT, now);
}

/// For `wallpaper_covered`: whether the window `label`'s monitor is covered
/// now (false for a window the watch doesn't know).
pub fn covered(label: &str) -> bool {
    slots().iter().any(|s| s.covered && s.label.as_deref() == Some(label))
}

/// From Quit, on the main thread: the watch stops and every host goes now,
/// so the owner's wallpaper is back before the app has gone.
pub fn before_exit() {
    STOPPING.store(true, Ordering::SeqCst);
    poke();
    let hosts: Vec<isize> = slots().iter_mut().map(|s| std::mem::take(&mut s.host)).collect();
    for h in hosts {
        crate::host::destroy(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Geometry = Geometry { bounds: Bounds { left: 0, top: 0, right: 1920, bottom: 1200 }, dpi: 96 };
    const LAYER: isize = 0x100;

    fn homed() -> View {
        View {
            window: true,
            gone: false,
            built_for: Some(G),
            host: HostSeen::Alive { parent: LAYER, at: Some(G.bounds) },
            homed: true,
            failed: false,
            busy: false,
        }
    }

    fn empty() -> View {
        View { window: false, gone: false, built_for: None, host: HostSeen::None, homed: false, failed: false, busy: false }
    }

    #[test]
    fn a_healthy_slot_is_left_alone() {
        assert_eq!(plan(homed(), Some(G), Some(LAYER), false), Step::Nothing);
        // Explorer gone for a moment with the host still alive: nothing yet.
        assert_eq!(plan(homed(), Some(G), None, false), Step::Nothing);
    }

    #[test]
    fn a_wanted_slot_is_built_then_homed() {
        assert_eq!(plan(empty(), Some(G), Some(LAYER), false), Step::Build);
        assert_eq!(plan(empty(), Some(G), None, false), Step::Build, "built while Explorer is away");
        let built = View { window: true, built_for: Some(G), ..empty() };
        assert_eq!(plan(built, Some(G), Some(LAYER), false), Step::Home);
        assert_eq!(plan(built, Some(G), None, false), Step::Wait);
        // A host made, the controller not yet in it (the homing failed to
        // be asked): homed again, into the same host.
        let hosted = View { host: HostSeen::Alive { parent: LAYER, at: Some(G.bounds) }, ..built };
        assert_eq!(plan(hosted, Some(G), Some(LAYER), false), Step::Home);
        // Not homed and moved: homed again, which puts it back too.
        let moved = View { host: HostSeen::Alive { parent: LAYER, at: None }, ..built };
        assert_eq!(plan(moved, Some(G), Some(LAYER), false), Step::Home);
        // Homed once, its host since taken: given one again.
        let hostless = View { host: HostSeen::None, ..homed() };
        assert_eq!(plan(hostless, Some(G), Some(LAYER), false), Step::Home);
        assert_eq!(plan(hostless, Some(G), None, false), Step::Wait);
    }

    #[test]
    fn an_unwanted_slot_is_taken_down() {
        assert_eq!(plan(homed(), None, Some(LAYER), false), Step::TearDown);
        assert_eq!(plan(View { window: false, ..homed() }, None, None, false), Step::TearDown, "a host alone");
        assert_eq!(plan(View { host: HostSeen::None, ..homed() }, None, None, false), Step::TearDown, "a window alone");
        assert_eq!(plan(empty(), None, Some(LAYER), false), Step::Nothing);
    }

    #[test]
    fn a_dead_host_is_a_rebuild() {
        let dead = View { host: HostSeen::Dead, ..homed() };
        assert_eq!(plan(dead, Some(G), Some(LAYER), false), Step::Rebuild(Why::HostDied));
        assert_eq!(plan(dead, Some(G), None, false), Step::Rebuild(Why::HostDied), "even before Explorer is back");
        assert_eq!(plan(View { homed: false, ..dead }, Some(G), None, false), Step::Rebuild(Why::HostDied));
        assert_eq!(plan(dead, None, None, false), Step::TearDown, "unless it isn't wanted");
    }

    #[test]
    fn a_failed_homing_is_a_rebuild() {
        let failed = View { failed: true, homed: false, ..homed() };
        assert_eq!(plan(failed, Some(G), Some(LAYER), false), Step::Rebuild(Why::HomingFailed));
    }

    #[test]
    fn a_changed_monitor_is_a_rebuild() {
        let moved = Geometry { bounds: Bounds { left: 1920, right: 3840, ..G.bounds }, ..G };
        let scaled = Geometry { dpi: 144, ..G };
        let resized = Geometry { bounds: Bounds { right: 2560, bottom: 1440, ..G.bounds }, ..G };
        for g in [moved, scaled, resized] {
            assert_eq!(plan(homed(), Some(g), Some(LAYER), false), Step::Rebuild(Why::MonitorChanged), "{g:?}");
        }
        let unhomed = View { homed: false, host: HostSeen::None, ..homed() };
        assert_eq!(plan(unhomed, Some(scaled), None, false), Step::Rebuild(Why::MonitorChanged));
    }

    #[test]
    fn a_resume_rebuilds_homed_slots_only() {
        assert_eq!(plan(homed(), Some(G), Some(LAYER), true), Step::Rebuild(Why::Resumed));
        let built = View { window: true, built_for: Some(G), ..empty() };
        assert_eq!(plan(built, Some(G), Some(LAYER), true), Step::Home);
        assert_eq!(plan(empty(), Some(G), Some(LAYER), true), Step::Build);
    }

    #[test]
    fn a_new_layer_is_a_rebuild() {
        assert_eq!(plan(homed(), Some(G), Some(LAYER + 1), false), Step::Rebuild(Why::LayerChanged));
    }

    /// The layer moved with the desktop's top-left, or wry moved the host
    /// to the layer's corner: put back, not rebuilt (nothing reloads).
    #[test]
    fn a_host_off_its_monitor_is_put_back() {
        let shifted = Bounds { left: -1920, right: 0, ..G.bounds };
        let elsewhere = View { host: HostSeen::Alive { parent: LAYER, at: Some(shifted) }, ..homed() };
        assert_eq!(plan(elsewhere, Some(G), Some(LAYER), false), Step::Place);
        let unknown = View { host: HostSeen::Alive { parent: LAYER, at: None }, ..homed() };
        assert_eq!(plan(unknown, Some(G), Some(LAYER), false), Step::Place);
        assert_eq!(plan(elsewhere, Some(G), None, false), Step::Nothing, "not while the layer is away");
        // A monitor that moved itself is a rebuild, as before.
        assert_eq!(
            plan(elsewhere, Some(Geometry { bounds: shifted, ..G }), Some(LAYER), false),
            Step::Rebuild(Why::MonitorChanged)
        );
    }

    #[test]
    fn a_window_someone_else_closed_is_a_rebuild() {
        let gone = View { gone: true, ..homed() };
        assert_eq!(plan(gone, Some(G), Some(LAYER), false), Step::Rebuild(Why::WindowGone));
        assert_eq!(plan(gone, None, None, false), Step::TearDown);
    }

    #[test]
    fn a_busy_slot_is_left_alone_whatever_else() {
        let busy = View { busy: true, ..homed() };
        assert_eq!(plan(busy, None, None, true), Step::Nothing);
        assert_eq!(plan(View { host: HostSeen::Dead, ..busy }, Some(G), Some(LAYER), false), Step::Nothing);
    }

    #[test]
    fn the_lock_screen_covers_every_monitor() {
        assert!(covered_now(true, || panic!("not asked while locked")));
        assert!(covered_now(false, || true));
        assert!(!covered_now(false, || false));
    }

    #[test]
    fn labels_are_the_capabilitys() {
        assert_eq!(label(2, 7), "wallpaper-2-7");
        assert!(is_wallpaper(&label(0, 1)));
        assert!(is_wallpaper(&label(MAX_SLOTS - 1, u64::MAX)));
        for other in ["avatar", "control", "prompt", "wallpaper-", "wallpaper-0", "wallpaper-4-1", "wallpaper-0-1-2", "wallpaper-x-1", ""] {
            assert!(!is_wallpaper(other), "{other:?}");
        }
        let cap: serde_json::Value = serde_json::from_str(include_str!("../capabilities/wallpaper.json")).unwrap();
        assert_eq!(cap["windows"], serde_json::json!(["wallpaper-*"]));
        assert!(label(0, 1).starts_with(cap["windows"][0].as_str().unwrap().trim_end_matches('*')));
    }

    /// The hosts' and webviews' colour is the page's own `--bg`, block by
    /// block, so nothing shows a colour the scene isn't.
    #[test]
    fn the_backdrop_is_each_moods_bg() {
        let tokens = include_str!("../../src/divora/tokens.css");
        let bg_of = |selector: &str| -> (u8, u8, u8) {
            let at = tokens.find(&format!("{selector} {{")).unwrap_or_else(|| panic!("no {selector}"));
            let block = &tokens[at..at + tokens[at..].find('}').unwrap()];
            let hex = block.split("--bg:").nth(1).unwrap().trim_start();
            let hex = &hex[1..7];
            let n = |k: usize| u8::from_str_radix(&hex[k..k + 2], 16).unwrap();
            (n(0), n(2), n(4))
        };
        for (theme, mood, selector) in [
            ("dark", "dusk", ":root"),
            ("dark", "ink", ":root[data-mood=\"ink\"]"),
            ("dark", "midnight", ":root[data-mood=\"midnight\"]"),
            ("light", "dusk", ":root[data-theme=\"light\"]"),
            ("light", "ink", ":root[data-theme=\"light\"][data-mood=\"ink\"]"),
            ("light", "midnight", ":root[data-theme=\"light\"][data-mood=\"midnight\"]"),
        ] {
            let t = Theme { theme: theme.into(), mood: mood.into(), ..Theme::default() };
            assert_eq!(backdrop(&t), bg_of(selector), "{theme} {mood}");
        }
    }

    #[test]
    fn a_window_is_built_its_monitors_logical_size() {
        assert_eq!(logical_size(G), (1920.0, 1200.0));
        assert_eq!(logical_size(Geometry { dpi: 144, ..G }), (1280.0, 800.0));
        let nothing = Geometry { bounds: Bounds { left: 0, top: 0, right: 0, bottom: 0 }, dpi: 0 };
        let (w, h) = logical_size(nothing);
        assert!(w >= 1.0 && h >= 1.0);
    }

    #[test]
    fn a_slot_taken_down_moves_its_count_on() {
        let mut s =
            Slot { gen: 4, label: Some("x".into()), host: 9, homed: true, covered: true, failures: 2, ..Slot::EMPTY };
        s.clear();
        assert_eq!(s.gen, 5, "a homing for what was taken down is ignored");
        assert_eq!(s.failures, 2, "how its builds have gone is kept");
        assert!(s.label.is_none() && s.host == 0 && !s.homed && !s.covered);
    }

    #[test]
    fn a_build_is_noted_in_its_slot() {
        let mut s = Slot { gen: 3, ..Slot::EMPTY };
        assert_eq!(note_build(&mut s, 3, "wallpaper-0-3", G, false), Built::Failed);
        assert_eq!((s.failures, s.cooldown), (1, COOLDOWN));
        assert_eq!(note_build(&mut s, 3, "wallpaper-0-3", G, false), Built::Failed);
        assert_eq!((s.failures, s.cooldown), (2, 2 * COOLDOWN));
        assert!(s.label.is_none());
        assert_eq!(note_build(&mut s, 3, "wallpaper-0-3", G, true), Built::Kept);
        assert_eq!((s.label.as_deref(), s.built_for, s.failures), (Some("wallpaper-0-3"), Some(G), 0));
        let mut gone = Slot { gen: 5, ..Slot::EMPTY };
        assert_eq!(note_build(&mut gone, 4, "wallpaper-0-4", G, true), Built::Unwanted);
        assert!(gone.label.is_none() && gone.built_for.is_none());
    }

    #[test]
    fn failed_builds_are_tried_ever_more_slowly() {
        assert_eq!(cooldown_after(1), COOLDOWN);
        assert_eq!(cooldown_after(2), 2 * COOLDOWN);
        assert_eq!(cooldown_after(3), 4 * COOLDOWN);
        assert_eq!(cooldown_after(7), 64 * COOLDOWN);
        assert_eq!(cooldown_after(8), MAX_COOLDOWN);
        assert_eq!(cooldown_after(u32::MAX), MAX_COOLDOWN, "no overflow");
        assert_eq!(cooldown_after(0), COOLDOWN);
    }

    /// Reports from an older build are ignored; this build's are heard.
    #[test]
    fn only_this_builds_homing_is_heard() {
        // Slot 3, which no other test touches.
        {
            let mut all = slots();
            all[3] = Slot { gen: 5, label: Some(label(3, 5)), busy: 2, ..Slot::EMPTY };
        }
        heard(3, 4, true);
        assert!(!slots()[3].homed && slots()[3].busy == 2, "an old build's report");
        heard(3, 5, true);
        assert!(slots()[3].homed && slots()[3].busy == 0);
        slots()[3].busy = 1;
        heard(3, 5, false);
        assert!(slots()[3].failed && slots()[3].busy == 0);
        assert!(!covered(&label(3, 5)));
        slots()[3].covered = true;
        assert!(covered(&label(3, 5)));
        assert!(!covered(&label(3, 4)));
        // Taken down, then a late report for the build it had: ignored.
        slots()[3].clear();
        heard(3, 5, true);
        assert!(!slots()[3].homed, "a cleared slot isn't homed by an old report");
        slots()[3].clear();
    }
}
