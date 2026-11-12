//! The display's power: when the monitor turns off (without a lock: the
//! screen timing out), Fritz stops drawing until it comes back. The lock
//! screen is the cover watch's (`cover.rs`, asked through WTS), not this.
//!
//! Plumbing from LorraineAtelier's `power.rs` (the owner's own app): a hidden
//! top-level window (a message-only one never gets `WM_POWERBROADCAST`) on a
//! thread of its own, registered for suspend/resume and for
//! `GUID_CONSOLE_DISPLAY_STATE`. Changed on the way: no session-change
//! handling (Lorraine merged lock into "display off"; here the cover watch
//! owns the lock, and a display waking to show the lock screen must not
//! unpause him); the event is sent only when the state changes; the state is
//! kept, so a page that starts late can ask for it; the window procedure
//! can't unwind into Windows (`logs::guard`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use tauri::{AppHandle, Emitter};
use windows::core::{w, GUID, PCWSTR};
use windows::Win32::Foundation::{HANDLE, HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Power::{
    RegisterPowerSettingNotification, RegisterSuspendResumeNotification, POWERBROADCAST_SETTING,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, RegisterClassW,
    TranslateMessage, DEVICE_NOTIFY_WINDOW_HANDLE, MSG, WM_POWERBROADCAST, WNDCLASSW,
    WS_EX_TOOLWINDOW, WS_POPUP,
};

/// The event every window gets: `true` when the display is on (or dimmed).
pub const DISPLAY_EVENT: &str = "power:display";

/// Sent on every resume (`PBT_APMRESUMESUSPEND` comes as well only when a
/// person woke it).
const PBT_APMRESUMEAUTOMATIC: u32 = 0x0012;
const PBT_POWERSETTINGCHANGE: u32 = 0x8013;
/// `Data[0]`: 0 off, 1 on, 2 dimmed.
const GUID_CONSOLE_DISPLAY_STATE: GUID = GUID::from_u128(0x6fe69556_704a_47a0_8f24_c28d936fda47);
const WATCH_CLASS: PCWSTR = w!("FritzPowerWatch");

/// What the watch saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seen {
    Display(bool),
    Resumed,
}

/// Where the watch's window procedure sends what it sees: set once, by
/// `start` (or a test's `watch`).
static SINK: OnceLock<Box<dyn Fn(Seen) + Send + Sync>> = OnceLock::new();

static DISPLAY_ON: AtomicBool = AtomicBool::new(true);

/// Whether the display is on now (dimmed counts as on: he is still seen).
pub fn display_on() -> bool {
    DISPLAY_ON.load(Ordering::SeqCst)
}

/// `GUID_CONSOLE_DISPLAY_STATE`'s value, as on (true) or off.
pub fn display_state(data0: u8) -> bool {
    data0 != 0
}

/// Record a display state; `true` if it changed.
fn note_display(on: bool) -> bool {
    DISPLAY_ON.swap(on, Ordering::SeqCst) != on
}

/// What a sighting means for the windows: the display state to announce,
/// when it changed. A resume, or a display state they already have, is
/// nothing to them.
pub fn news(seen: Seen) -> Option<bool> {
    match seen {
        Seen::Display(on) => note_display(on).then_some(on),
        Seen::Resumed => None,
    }
}

/// Start the watch for the app: display changes go to every window as
/// `power:display`, resumes to the logs.
pub fn start(app: AppHandle) {
    let started = watch(Box::new(move |seen| {
        if seen == Seen::Resumed {
            tracing::info!("resumed from sleep");
            crate::logs::startup("resumed from sleep");
        }
        if let Some(on) = news(seen) {
            tracing::info!(on, "the display turned {}", if on { "on" } else { "off" });
            let _ = app.emit(DISPLAY_EVENT, on);
        }
    }));
    if let Err(e) = started {
        tracing::warn!(error = %e, "couldn't start the power watch; Fritz draws while the display is off");
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg != WM_POWERBROADCAST {
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    crate::logs::guard("power watch", || {
        let seen = match u32::try_from(wparam.0).unwrap_or(0) {
            PBT_APMRESUMEAUTOMATIC => Some(Seen::Resumed),
            PBT_POWERSETTINGCHANGE if lparam.0 != 0 => {
                // SAFETY: for PBT_POWERSETTINGCHANGE, lparam points at a
                // POWERBROADCAST_SETTING Windows owns for this call.
                let setting = unsafe { &*(lparam.0 as *const POWERBROADCAST_SETTING) };
                (setting.PowerSetting == GUID_CONSOLE_DISPLAY_STATE)
                    .then(|| Seen::Display(display_state(setting.Data[0])))
            }
            _ => None,
        };
        if let (Some(seen), Some(sink)) = (seen, SINK.get()) {
            sink(seen);
        }
    });
    LRESULT(1) // TRUE: the broadcast is granted
}

/// The watch itself: a thread with a hidden window registered for the
/// notifications, sending what it sees to `sink`. Once per process.
/// Registering for the display state delivers its current value at once, so
/// the first `Display` arrives within moments of the start.
pub fn watch(sink: Box<dyn Fn(Seen) + Send + Sync>) -> Result<(), String> {
    if SINK.set(sink).is_err() {
        return Err("the power watch is already running".to_owned());
    }
    std::thread::Builder::new()
        .name("fritz-power-watch".into())
        .spawn(|| {
            // SAFETY: plain Win32 calls on this thread's own window; the
            // registrations live as long as the thread, which is the app's.
            unsafe {
                let hinst: HINSTANCE = GetModuleHandleW(None).map(HINSTANCE::from).unwrap_or_default();
                let wc = WNDCLASSW {
                    lpfnWndProc: Some(wndproc),
                    hInstance: hinst,
                    lpszClassName: WATCH_CLASS,
                    ..Default::default()
                };
                RegisterClassW(&wc);
                // Top-level, never shown, no size, off the taskbar: it exists
                // only to be told.
                let hwnd = match CreateWindowExW(
                    WS_EX_TOOLWINDOW,
                    WATCH_CLASS,
                    PCWSTR::null(),
                    WS_POPUP,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    Some(hinst),
                    None,
                ) {
                    Ok(h) if !h.0.is_null() => h,
                    _ => {
                        tracing::warn!("the power watch couldn't make its window");
                        return;
                    }
                };
                let _resume = RegisterSuspendResumeNotification(HANDLE(hwnd.0), DEVICE_NOTIFY_WINDOW_HANDLE);
                let _display = RegisterPowerSettingNotification(
                    HANDLE(hwnd.0),
                    &GUID_CONSOLE_DISPLAY_STATE,
                    DEVICE_NOTIFY_WINDOW_HANDLE,
                );
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{display_on, display_state, news, watch, Seen};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn dimmed_is_still_on() {
        assert!(!display_state(0), "off");
        assert!(display_state(1), "on");
        assert!(display_state(2), "dimmed: he is still seen");
    }

    /// The windows hear a display state only when it changes, and the state
    /// a late page asks for follows it.
    #[test]
    fn only_a_change_is_news() {
        assert_eq!(news(Seen::Display(true)), None, "on, as it starts");
        assert_eq!(news(Seen::Display(false)), Some(false));
        assert!(!display_on());
        assert_eq!(news(Seen::Display(false)), None);
        assert_eq!(news(Seen::Resumed), None);
        assert_eq!(news(Seen::Display(true)), Some(true));
        assert!(display_on());
    }

    /// The real registration: Windows answers it at once with the display's
    /// state now, on or off (a locked laptop may have it off).
    #[test]
    fn the_watch_hears_the_display_state_at_once() {
        let (tx, rx) = mpsc::channel();
        let tx = std::sync::Mutex::new(tx);
        watch(Box::new(move |seen| {
            let _ = tx.lock().unwrap().send(seen);
        }))
        .unwrap();
        let first = rx.recv_timeout(Duration::from_secs(5)).expect("no display state within 5 s");
        assert!(matches!(first, Seen::Display(_)), "first sighting was {first:?}");
        assert!(watch(Box::new(|_| {})).is_err(), "once per process");
    }
}
