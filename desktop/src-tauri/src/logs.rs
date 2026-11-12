//! The two logs that outlive a run, in `%APPDATA%\com.divora.fritz\`:
//! `startup.log`, a line for each milestone of a start and a stop, and
//! `crash.log`, every panic and every one `guard` caught. `logs\fritz.log`
//! (tracing) is the session's detailed log and starts afresh each run; these
//! two are what to read after a start that died before it could say why.
//!
//! Plumbing from LorraineAtelier (the owner's own app): `log_startup`,
//! `log_crash`, `guard` and the panic hook. Changed on the way: one capped
//! writer for both files, which keeps the previous file as `.old` instead of
//! deleting it; readable UTC times; and a panic hook that chains to the one
//! it replaces, so a debug build still prints its panics.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Past this size a log is moved to `<name>.old` (replacing the last one)
/// and started afresh.
const CAP_BYTES: u64 = 64 * 1024;

pub const STARTUP_LOG: &str = "startup.log";
pub const CRASH_LOG: &str = "crash.log";

/// One writer at a time, so two threads' lines never interleave.
static WRITING: Mutex<()> = Mutex::new(());

/// A line in `startup.log`.
pub fn startup(msg: &str) {
    append(&crate::settings::app_dir(), STARTUP_LOG, msg, CAP_BYTES);
}

/// A line in `crash.log`.
pub fn crash(msg: &str) {
    append(&crate::settings::app_dir(), CRASH_LOG, msg, CAP_BYTES);
}

/// Append `"<utc> <msg>"` to `dir/name`, moving the file to `name.old` first
/// when it has grown past `cap`. Every error is swallowed: a log that can't
/// be written must never be why Fritz stops.
pub fn append(dir: &Path, name: &str, msg: &str, cap: u64) {
    let _one = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let _ = fs::create_dir_all(dir);
    let path = dir.join(name);
    if fs::metadata(&path).is_ok_and(|m| m.len() > cap) {
        let _ = fs::rename(&path, dir.join(format!("{name}.old")));
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        // One line per call, whatever the message holds.
        let line = msg.replace(['\r', '\n'], " ");
        let _ = writeln!(f, "{} {line}", utc(SystemTime::now()));
    }
}

/// Run `f`, and if it panics, note it in `crash.log` and carry on: `None`.
///
/// For closures on the main thread (the shell's effects, the tray, hotkeys,
/// the power watch's window procedure), where a panic would otherwise unwind
/// into the windowing library's callback and abort the whole app. Needs
/// `panic = "unwind"` (Cargo.toml) and locks that tolerate poisoning.
pub fn guard<T>(name: &str, f: impl FnOnce() -> T) -> Option<T> {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Some(v),
        Err(_) => {
            crash(&format!("recovered from a panic in '{name}'"));
            None
        }
    }
}

/// Every panic goes to `crash.log` with its thread's name, then to whatever
/// hook was there before (the default prints it, with a backtrace when asked).
pub fn install_panic_hook() {
    let before = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let current = std::thread::current();
        let thread = current.name().unwrap_or("unnamed");
        crash(&format!("PANIC [{thread}]: {info}"));
        before(info);
    }));
}

/// `t` as `YYYY-MM-DD HH:MM:SS.mmmZ`, without a date crate.
pub fn utc(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Howard Hinnant's civil-from-days.
    let z = i64::try_from(days).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
        d.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::{append, guard, utc};
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fritz-logs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn utc_formats_known_instants() {
        assert_eq!(utc(UNIX_EPOCH), "1970-01-01 00:00:00.000Z");
        assert_eq!(
            utc(UNIX_EPOCH + Duration::from_millis(951_782_400_250)),
            "2000-02-29 00:00:00.250Z"
        );
        assert_eq!(
            utc(UNIX_EPOCH + Duration::from_secs(1_794_418_496)),
            "2026-11-11 17:34:56.000Z"
        );
    }

    #[test]
    fn lines_are_appended_with_a_time() {
        let d = dir("append");
        append(&d, "startup.log", "first", 1024);
        append(&d, "startup.log", "second", 1024);
        let text = fs::read_to_string(d.join("startup.log")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "{text}");
        assert!(lines[0].ends_with("Z first"), "{text}");
        assert!(lines[1].ends_with("Z second"), "{text}");
    }

    /// A message can't forge a second line.
    #[test]
    fn one_call_is_one_line() {
        let d = dir("oneline");
        append(&d, "crash.log", "PANIC: a\r\n2026-01-01 00:00:00.000Z forged", 1024);
        let text = fs::read_to_string(d.join("crash.log")).unwrap();
        assert_eq!(text.lines().count(), 1, "{text}");
    }

    /// Past the cap the file moves to `.old`, which the next roll replaces:
    /// the last run's lines survive one roll, and the logs stay bounded.
    #[test]
    fn a_full_log_rolls_to_old_and_keeps_going() {
        let d = dir("roll");
        let cap = 200;
        for i in 0..4 {
            append(&d, "startup.log", &format!("{i} {}", "x".repeat(90)), cap);
        }
        let now = fs::read_to_string(d.join("startup.log")).unwrap();
        let old = fs::read_to_string(d.join("startup.log.old")).unwrap();
        assert!(now.contains("3 xxx"), "{now}");
        assert!(old.contains("2 xxx") || old.contains("1 xxx"), "{old}");
        assert!(fs::metadata(d.join("startup.log")).unwrap().len() <= cap + 200);
    }

    /// The hook goes before whatever was there, never instead of it.
    #[test]
    fn the_panic_hook_still_calls_the_one_before_it() {
        use std::panic;
        use std::sync::atomic::{AtomicU32, Ordering};
        static CALLED: AtomicU32 = AtomicU32::new(0);
        let original = panic::take_hook();
        panic::set_hook(Box::new(|info| {
            if info.to_string().contains("hook-chain-test") {
                CALLED.fetch_add(1, Ordering::SeqCst);
            }
        }));
        super::install_panic_hook();
        let _ = panic::catch_unwind(|| panic!("hook-chain-test"));
        panic::set_hook(original);
        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn guard_returns_the_value_or_none_after_a_panic() {
        assert_eq!(guard("ok", || 7), Some(7));
        let caught = guard("boom", || -> u8 { panic!("test panic, caught by guard") });
        assert_eq!(caught, None);
    }
}
