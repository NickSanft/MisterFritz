//! Spike 0's instrument for its first row: `GetForegroundWindow` polled at
//! 10 Hz, and a line written whenever the foreground belongs to Fritz — our
//! own process, the avatar window or anything it owns, or a WebView2
//! process this app started. The pass mark is zero such lines in a working
//! session (plan 14 § Spike 0). A heartbeat every minute proves a quiet log
//! was listening, and the last few foreground changes go out with each hit so
//! it can be seen what Fritz took the keyboard from. The control window
//! (tray > Settings) is an ordinary window and takes the keyboard when it is
//! opened: its lines say SETTINGS, never HIT. The prompt takes it when the
//! owner opens it (its key, or a double-click on him): a PROMPT line, but
//! only for an open that took the keyboard, or a waiting prompt the hook
//! saw the owner press into or its key took the keyboard into
//! (`prompt::takes`); the prompt in front with no such open, or the
//! keyboard still with it after it closed, is a HIT.
//!
//! Off unless `settings.diagnostics.foregroundLog` is on (read at start): it
//! writes other apps' window titles to `diagnostics\foreground.log`, so it is
//! the owner's to switch on, and the file stays on this machine.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use crate::logs::utc;

use tauri::{AppHandle, Manager};
use windows::Win32::Foundation::{CloseHandle, HWND};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetAncestor, GetClassNameW, GetWindowTextW, GetWindowThreadProcessId, GA_ROOTOWNER,
};

const POLL: Duration = Duration::from_millis(100);
const HEARTBEAT_POLLS: u64 = 600;
const RECENT: usize = 5;
/// After Arrange ends, how long the avatar may keep the foreground while it
/// is handed back before that counts as a HIT.
const ARRANGE_GRACE: Duration = Duration::from_millis(1500);

/// Why a foreground window counts against Fritz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Someone else's window: the pass case.
    Elsewhere,
    /// The avatar window itself.
    Avatar,
    /// Owned by the avatar window (a popup it raised).
    OwnedByAvatar,
    /// The tray icon's hidden window or its menu: tray-icon makes its own
    /// window the foreground before showing the menu, as Windows requires.
    /// Not the avatar, and not this row's business.
    Tray,
    /// The control window (tray > Settings), which the owner opened: an
    /// ordinary window that takes the keyboard on purpose.
    Settings,
    /// The prompt (or a window it owns), while it shows.
    Prompt,
    /// Another window of this process.
    OurProcess,
    /// A WebView2 process descended from this one.
    OurWebView,
}

/// What the poller saw.
#[derive(Debug, Clone, Default)]
pub struct Seen {
    pub hwnd: isize,
    pub root_owner: isize,
    pub pid: u32,
    pub exe: String,
    pub class: String,
    pub title: String,
}

/// Judge one foreground window. `ancestors(pid)` walks the parent chain; it
/// is only asked about WebView2 processes, which are rare in the foreground.
pub fn classify(
    seen: &Seen,
    me: u32,
    avatar: isize,
    control: isize,
    prompt: isize,
    ancestors: impl FnOnce(u32) -> Vec<u32>,
) -> Verdict {
    if control != 0 && (seen.hwnd == control || seen.root_owner == control) {
        Verdict::Settings
    } else if prompt != 0 && (seen.hwnd == prompt || seen.root_owner == prompt) {
        Verdict::Prompt
    } else if avatar != 0 && seen.hwnd == avatar {
        Verdict::Avatar
    } else if avatar != 0 && seen.root_owner == avatar {
        Verdict::OwnedByAvatar
    } else if seen.pid == me && matches!(seen.class.as_str(), "tray_icon_app" | "#32768") {
        Verdict::Tray
    } else if seen.pid == me {
        Verdict::OurProcess
    } else if exe_name(&seen.exe).eq_ignore_ascii_case("msedgewebview2.exe")
        && ancestors(seen.pid).contains(&me)
    {
        Verdict::OurWebView
    } else {
        Verdict::Elsewhere
    }
}

/// The line a foreground change earns: none for someone else's window,
/// `ARRANGE` while the owner is arranging Fritz (Arrange takes the keyboard
/// on purpose) or just after, and `HIT` — a failure of the row — otherwise.
pub fn line_kind(verdict: Verdict, arranging_recently: bool) -> Option<&'static str> {
    match (verdict, arranging_recently) {
        (Verdict::Elsewhere, _) => None,
        (Verdict::Tray, _) => Some("TRAY"),
        (Verdict::Settings, _) => Some("SETTINGS"),
        (Verdict::Prompt, _) => Some("PROMPT"),
        (_, true) => Some("ARRANGE"),
        (_, false) => Some("HIT"),
    }
}

/// What one poll writes. On a change of foreground, [`line_kind`]; a change
/// to the prompt is PROMPT only with a `fresh_open` (an owner's open that
/// took the keyboard since the last PROMPT line) or continuing a PROMPT run,
/// and a HIT otherwise. With the same foreground as before, nothing, unless
/// Fritz still holds it once the reason is over: after Arrange (with its
/// grace), or after the prompt closed. That is a HIT, written once
/// ([`kept_after`]).
pub fn judge(
    changed: bool,
    verdict: Verdict,
    arranging_recently: bool,
    last_kind: Option<&'static str>,
    fresh_open: bool,
) -> Option<&'static str> {
    let elsewhere = matches!(verdict, Verdict::Elsewhere | Verdict::Tray | Verdict::Settings | Verdict::Prompt);
    if changed {
        if verdict == Verdict::Prompt && !fresh_open && last_kind != Some("PROMPT") {
            Some("HIT")
        } else {
            line_kind(verdict, arranging_recently)
        }
    } else if (last_kind == Some("ARRANGE") && !arranging_recently || last_kind == Some("PROMPT")) && !elsewhere {
        Some("HIT")
    } else {
        None
    }
}

/// The readings one poll judges by, taken so they agree: the foreground,
/// then who else is showing (`others`), then the foreground again. A close
/// (the keyboard handed back, then the prompt hidden) or an open (shown,
/// then the keyboard taken) landing between the readings moves the
/// foreground, and the poll is skipped (`None`); the next sees it settled.
/// Otherwise a prompt just closed could be judged hidden yet still in front.
pub fn settled<T>(fg: impl Fn() -> isize, others: impl FnOnce() -> T) -> Option<(isize, T)> {
    let before = fg();
    let seen = others();
    (fg() == before).then_some((before, seen))
}

/// What a HIT for a foreground that didn't change says it was kept after.
pub fn kept_after(last_kind: Option<&'static str>) -> &'static str {
    if last_kind == Some("PROMPT") {
        " kept after the prompt closed"
    } else {
        " kept after Arrange"
    }
}

fn exe_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// `pid`'s parent chain (nearest first), from one process snapshot.
pub fn ancestry(parents: &HashMap<u32, u32>, pid: u32) -> Vec<u32> {
    let mut chain = Vec::new();
    let mut at = pid;
    while let Some(&parent) = parents.get(&at) {
        if parent == 0 || chain.contains(&parent) || chain.len() >= 16 {
            break;
        }
        chain.push(parent);
        at = parent;
    }
    chain
}

fn wide(buf: &[u16], n: i32) -> String {
    String::from_utf16_lossy(&buf[..usize::try_from(n).unwrap_or(0).min(buf.len())])
}

fn look(h: HWND) -> Seen {
    let mut seen = Seen {
        hwnd: h.0 as isize,
        ..Seen::default()
    };
    // SAFETY: plain queries on a window handle that may already be gone,
    // which only makes them fail; buffers are sized.
    unsafe {
        seen.root_owner = GetAncestor(h, GA_ROOTOWNER).0 as isize;
        GetWindowThreadProcessId(h, Some(&raw mut seen.pid));
        let mut buf = [0u16; 256];
        let n = GetClassNameW(h, &mut buf);
        seen.class = wide(&buf, n);
        let mut buf = [0u16; 256];
        let n = GetWindowTextW(h, &mut buf);
        seen.title = wide(&buf, n);
        if let Ok(p) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, seen.pid) {
            let mut buf = [0u16; 1024];
            let mut len = buf.len() as u32;
            if QueryFullProcessImageNameW(
                p,
                PROCESS_NAME_WIN32,
                windows::core::PWSTR(buf.as_mut_ptr()),
                &raw mut len,
            )
            .is_ok()
            {
                seen.exe = String::from_utf16_lossy(&buf[..len as usize]);
            }
            let _ = CloseHandle(p);
        }
    }
    seen
}

fn parents() -> HashMap<u32, u32> {
    let mut map = HashMap::new();
    // SAFETY: a process snapshot we close; the entry is sized as the API asks.
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else {
            return map;
        };
        let mut e = PROCESSENTRY32W {
            dwSize: size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        if Process32FirstW(snap, &raw mut e).is_ok() {
            loop {
                map.insert(e.th32ProcessID, e.th32ParentProcessID);
                if Process32NextW(snap, &raw mut e).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);
    }
    map
}

fn describe(s: &Seen) -> String {
    format!(
        "hwnd=0x{:x} pid={} exe={} class={:?} title={:?}",
        s.hwnd,
        s.pid,
        exe_name(&s.exe),
        s.class,
        s.title.chars().take(80).collect::<String>()
    )
}

/// Start the poller. Writes to `path` (appending, so a session spans
/// restarts), and never stops until the process does.
pub fn start(app: AppHandle, path: PathBuf) {
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let spawned = thread::Builder::new()
        .name("fritz-foreground-log".into())
        .spawn(move || {
            let me = std::process::id();
            let write = |line: String| {
                if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = writeln!(f, "{} {line}", utc(SystemTime::now()));
                }
            };
            write(format!(
                "start pid={me} poll=100ms; a HIT line is a failure of Spike 0's first row \
                 (ARRANGE lines are Arrange taking the keyboard on purpose, TRAY lines the \
                 tray menu, SETTINGS lines the control window the owner opened, PROMPT \
                 lines the prompt the owner opened, by its key or a double-click on him)"
            ));
            let (mut polls, mut changes, mut hits, mut arranges, mut trays, mut settings, mut prompts) =
                (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
            // The prompt's takes at its last PROMPT line: a new take is a
            // fresh open.
            let mut seen_takes = crate::prompt::takes();
            // The control window's HWND, looked up once (a round trip to the
            // main thread); it is declared, so never made anew.
            let mut control_hwnd: isize = 0;
            let mut arranged_at: Option<Instant> = None;
            let mut last: isize = 0;
            // The kind of the last line written for the current foreground.
            let mut last_kind: Option<&'static str> = None;
            let mut recent: VecDeque<String> = VecDeque::with_capacity(RECENT);
            loop {
                thread::sleep(POLL);
                polls += 1;
                if control_hwnd == 0 {
                    control_hwnd = app
                        .get_webview_window(crate::CONTROL_LABEL)
                        .and_then(|w| w.hwnd().ok())
                        .map_or(0, |h| h.0 as isize);
                }
                let avatar = crate::overlay_windows::avatar_hwnd(&app);
                // The control window and the prompt count only while they can
                // be seen: either holding the keyboard hidden is a theft.
                let visible = |h: isize| Some(h).filter(|&h| crate::windows_ext::is_visible(h)).unwrap_or(0);
                let readings = settled(crate::windows_ext::foreground_window, || {
                    (visible(control_hwnd), visible(crate::prompt::hwnd()), crate::prompt::takes())
                });
                if crate::overlay_windows::arranging(&app) {
                    arranged_at = Some(Instant::now());
                }
                let arranging = arranged_at.is_some_and(|t| t.elapsed() < ARRANGE_GRACE);
                // The foreground moved while the readings were taken: the
                // next poll judges it.
                let Some((fg_now, (control, prompt, takes))) = readings else {
                    if polls % HEARTBEAT_POLLS == 0 {
                        write(format!(
                            "alive polls={polls} changes={changes} hits={hits} arranges={arranges} trays={trays} settings={settings} prompts={prompts} avatar=0x{avatar:x}"
                        ));
                    }
                    continue;
                };
                // A count given back since the last PROMPT line (an open
                // that never had the keyboard after all) never leaves the
                // mark above the count, or the owner's next open would read
                // as none.
                seen_takes = seen_takes.min(takes);
                let fg = HWND(fg_now as *mut core::ffi::c_void);
                let changed = fg_now != last;
                // Judged on every change, and again each poll while Fritz may
                // be keeping the foreground Arrange or the prompt gave him.
                if changed || (last_kind == Some("ARRANGE") && !arranging) || last_kind == Some("PROMPT") {
                    if changed {
                        last = fg_now;
                        changes += 1;
                    }
                    let seen = look(fg);
                    let verdict = classify(&seen, me, avatar, control, prompt, |pid| ancestry(&parents(), pid));
                    let kind = judge(changed, verdict, arranging, last_kind, takes > seen_takes);
                    let kept = if changed { "" } else { kept_after(last_kind) };
                    if changed || kind.is_some() {
                        last_kind = kind;
                    }
                    match kind {
                        None if changed => {
                            if recent.len() == RECENT {
                                recent.pop_front();
                            }
                            recent.push_back(format!(
                                "{} {}",
                                utc(SystemTime::now()),
                                describe(&seen)
                            ));
                        }
                        None => {}
                        Some(kind) => {
                            match kind {
                                "HIT" => hits += 1,
                                "TRAY" => trays += 1,
                                "ARRANGE" => arranges += 1,
                                "SETTINGS" => settings += 1,
                                "PROMPT" => {
                                    prompts += 1;
                                    seen_takes = takes;
                                }
                                _ => {}
                            }
                            write(format!(
                                "{kind} {verdict:?}{kept} avatar=0x{avatar:x} {}",
                                describe(&seen)
                            ));
                            for r in &recent {
                                write(format!("    before: {r}"));
                            }
                        }
                    }
                }
                if polls % HEARTBEAT_POLLS == 0 {
                    write(format!(
                        "alive polls={polls} changes={changes} hits={hits} arranges={arranges} trays={trays} settings={settings} prompts={prompts} avatar=0x{avatar:x}"
                    ));
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "couldn't start the foreground log");
    }
}

#[cfg(test)]
mod tests {
    use super::{ancestry, classify, judge, kept_after, line_kind, settled, Seen, Verdict};

    /// A count given back after a PROMPT line (an open that never had the
    /// keyboard after all) never leaves the mark above the count: the
    /// owner's next open still reads as fresh. Brought down each poll,
    /// before the judging.
    #[test]
    fn the_prompts_mark_never_stays_above_its_count() {
        let src = include_str!("foreground_log.rs").replace("\r\n", "\n");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        let min = code.find("seen_takes = seen_takes.min(takes);").expect("the mark brought down");
        assert!(min > code.find("let Some((fg_now, (control, prompt, takes))) = readings else {").unwrap());
        assert!(min < code.find("let kind = judge(changed, verdict, arranging, last_kind, takes > seen_takes);").unwrap());
    }
    use std::collections::HashMap;

    const ME: u32 = 100;
    const AVATAR: isize = 0x50;

    fn seen(hwnd: isize, root_owner: isize, pid: u32, exe: &str) -> Seen {
        Seen {
            hwnd,
            root_owner,
            pid,
            exe: exe.into(),
            ..Seen::default()
        }
    }

    /// The control window is opened by the owner and takes the keyboard on
    /// purpose: SETTINGS, never HIT, and Arrange handing the keyboard back
    /// to it (the window the owner was in) isn't Fritz keeping it.
    #[test]
    fn the_control_window_is_never_a_hit() {
        const CONTROL: isize = 0x60;
        let window = seen(CONTROL, CONTROL, ME, "");
        assert_eq!(classify(&window, ME, AVATAR, CONTROL, 0, never), Verdict::Settings);
        let dialog = seen(0x61, CONTROL, ME, "");
        assert_eq!(classify(&dialog, ME, AVATAR, CONTROL, 0, never), Verdict::Settings, "a window it owns");
        assert_eq!(line_kind(Verdict::Settings, false), Some("SETTINGS"));
        assert_eq!(line_kind(Verdict::Settings, true), Some("SETTINGS"));
        assert_eq!(judge(false, Verdict::Settings, false, Some("ARRANGE"), false), None);
        // Without its handle (the window not made yet), it is ours, as before.
        assert_eq!(classify(&window, ME, AVATAR, 0, 0, never), Verdict::OurProcess);
        // And the avatar is still the avatar.
        let avatar = seen(AVATAR, AVATAR, ME, "");
        assert_eq!(classify(&avatar, ME, AVATAR, CONTROL, 0, never), Verdict::Avatar);
    }

    fn never(_: u32) -> Vec<u32> {
        panic!("only a WebView2 process needs its ancestry")
    }

    #[test]
    fn someone_elses_window_is_the_pass_case() {
        let s = seen(0x99, 0x99, 555, r"C:\Windows\notepad.exe");
        assert_eq!(classify(&s, ME, AVATAR, 0, 0, never), Verdict::Elsewhere);
    }

    #[test]
    fn the_avatar_and_its_own_windows_are_hits() {
        assert_eq!(
            classify(&seen(AVATAR, AVATAR, ME, ""), ME, AVATAR, 0, 0, never),
            Verdict::Avatar
        );
        assert_eq!(
            classify(&seen(0x51, AVATAR, ME, ""), ME, AVATAR, 0, 0, never),
            Verdict::OwnedByAvatar
        );
        assert_eq!(
            classify(&seen(0x52, 0x52, ME, ""), ME, AVATAR, 0, 0, never),
            Verdict::OurProcess
        );
        // Before the window exists, our process still counts.
        assert_eq!(
            classify(&seen(0x52, 0x52, ME, ""), ME, 0, 0, 0, never),
            Verdict::OurProcess
        );
    }

    #[test]
    fn only_our_own_webview2_counts() {
        let wv = seen(
            0x60,
            0x60,
            300,
            r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application\msedgewebview2.exe",
        );
        assert_eq!(
            classify(&wv, ME, AVATAR, 0, 0, |_| vec![200, ME, 4]),
            Verdict::OurWebView
        );
        // Another app's WebView2 (Teams, Widgets) is not Fritz's.
        assert_eq!(
            classify(&wv, ME, AVATAR, 0, 0, |_| vec![200, 999, 4]),
            Verdict::Elsewhere
        );
    }

    /// Arrange takes the keyboard on purpose: that is logged, but isn't a
    /// failure of the row; the same moment outside Arrange is.
    #[test]
    fn arrange_is_not_a_hit() {
        assert_eq!(line_kind(Verdict::Elsewhere, false), None);
        assert_eq!(line_kind(Verdict::Elsewhere, true), None);
        assert_eq!(line_kind(Verdict::Avatar, true), Some("ARRANGE"));
        for v in [
            Verdict::Avatar,
            Verdict::OwnedByAvatar,
            Verdict::OurProcess,
            Verdict::OurWebView,
        ] {
            assert_eq!(line_kind(v, false), Some("HIT"), "{v:?}");
        }
    }

    /// The tray's hidden window (and its menu) take the foreground for every
    /// tray click: logged as TRAY, never a HIT.
    #[test]
    fn the_tray_is_not_a_hit() {
        for class in ["tray_icon_app", "#32768"] {
            let s = Seen {
                hwnd: 0x70,
                root_owner: 0x70,
                pid: ME,
                class: class.into(),
                ..Seen::default()
            };
            assert_eq!(classify(&s, ME, AVATAR, 0, 0, never), Verdict::Tray, "{class}");
            assert_eq!(line_kind(Verdict::Tray, false), Some("TRAY"));
        }
        // Someone else's menu is someone else's.
        let other = Seen {
            hwnd: 0x71,
            root_owner: 0x71,
            pid: 555,
            class: "#32768".into(),
            ..Seen::default()
        };
        assert_eq!(classify(&other, ME, AVATAR, 0, 0, never), Verdict::Elsewhere);
    }

    /// Fritz keeping the foreground once Arrange is over is a HIT, written
    /// once, even though the foreground window never changed.
    #[test]
    fn keeping_the_foreground_after_arrange_is_a_hit() {
        // Arrange takes it: ARRANGE.
        assert_eq!(judge(true, Verdict::Avatar, true, None, false), Some("ARRANGE"));
        // Still arranging, same window: nothing more.
        assert_eq!(judge(false, Verdict::Avatar, true, Some("ARRANGE"), false), None);
        // Arrange (and its grace) over, Fritz still in front: HIT.
        assert_eq!(
            judge(false, Verdict::Avatar, false, Some("ARRANGE"), false),
            Some("HIT")
        );
        // Once only.
        assert_eq!(judge(false, Verdict::Avatar, false, Some("HIT"), false), None);
        // Handed back in time: someone else's window is a change, and nothing.
        assert_eq!(
            judge(true, Verdict::Elsewhere, false, Some("ARRANGE"), false),
            None
        );
        // An unchanged foreground that was never ARRANGE is never re-judged.
        assert_eq!(judge(false, Verdict::Avatar, false, None, false), None);
    }

    const PROMPT: isize = 0x70;

    /// The prompt (or a window it owns) in front, while it shows, is the
    /// prompt; hidden, it is ours like any other window. Settings and the
    /// avatar are judged as before beside it.
    #[test]
    fn settings_and_avatar_are_judged_as_before_beside_a_prompt() {
        assert_eq!(classify(&seen(PROMPT, PROMPT, ME, ""), ME, AVATAR, 0x60, PROMPT, never), Verdict::Prompt);
        assert_eq!(classify(&seen(0x71, PROMPT, ME, ""), ME, AVATAR, 0x60, PROMPT, never), Verdict::Prompt, "its popup");
        assert_eq!(classify(&seen(PROMPT, PROMPT, ME, ""), ME, AVATAR, 0x60, 0, never), Verdict::OurProcess, "hidden");
        assert_eq!(classify(&seen(0x60, 0x60, ME, ""), ME, AVATAR, 0x60, PROMPT, never), Verdict::Settings);
        assert_eq!(classify(&seen(AVATAR, AVATAR, ME, ""), ME, AVATAR, 0x60, PROMPT, never), Verdict::Avatar);
        let notepad = seen(0x99, 0x99, 7, "notepad.exe");
        assert_eq!(classify(&notepad, ME, AVATAR, 0x60, PROMPT, never), Verdict::Elsewhere);
        assert_eq!(line_kind(Verdict::Prompt, false), Some("PROMPT"));
        assert_eq!(line_kind(Verdict::Prompt, true), Some("PROMPT"), "even just after Arrange");
    }

    /// The prompt coming to the front is PROMPT only for an owner's open
    /// that took the keyboard since its last line, or carrying on a PROMPT
    /// run (to a popup of its own); with no such open it is a HIT.
    #[test]
    fn a_prompt_line_needs_a_fresh_open() {
        assert_eq!(judge(true, Verdict::Prompt, false, None, true), Some("PROMPT"));
        assert_eq!(judge(true, Verdict::Prompt, false, Some("PROMPT"), false), Some("PROMPT"), "a run");
        assert_eq!(judge(true, Verdict::Prompt, false, None, false), Some("HIT"), "no open");
        assert_eq!(judge(true, Verdict::Prompt, false, Some("SETTINGS"), false), Some("HIT"));
        assert_eq!(judge(false, Verdict::Prompt, false, Some("PROMPT"), false), None, "still in front");
    }

    /// The prompt closed but its window still holds the keyboard: one HIT,
    /// "kept after the prompt closed"; the keyboard back with an app, or
    /// Settings, or the tray, is no line at all.
    #[test]
    fn the_prompt_keeping_the_keyboard_after_it_closed_is_a_hit_once() {
        assert_eq!(judge(false, Verdict::OurProcess, false, Some("PROMPT"), false), Some("HIT"));
        assert_eq!(kept_after(Some("PROMPT")), " kept after the prompt closed");
        assert_eq!(kept_after(Some("ARRANGE")), " kept after Arrange");
        // Written once: the HIT is now the last line.
        assert_eq!(judge(false, Verdict::OurProcess, false, Some("HIT"), false), None);
        for v in [Verdict::Elsewhere, Verdict::Tray, Verdict::Settings] {
            assert_eq!(judge(false, v, false, Some("PROMPT"), false), None, "{v:?}");
        }
    }

    /// A close landing between a poll's readings (the foreground read as
    /// the prompt, then the prompt seen hidden, the keyboard by then back
    /// with Notepad) skips the poll rather than judging the prompt hidden
    /// yet in front; readings that agree are judged.
    #[test]
    fn a_poll_judges_only_readings_that_agree() {
        let seq = std::cell::RefCell::new(vec![0x99_isize, PROMPT]);
        let fg = || seq.borrow_mut().pop().unwrap();
        assert_eq!(settled(fg, || "hidden"), None, "the prompt, then Notepad");
        let steady = std::cell::Cell::new(0);
        let fg = || {
            steady.set(steady.get() + 1);
            PROMPT
        };
        assert_eq!(settled(fg, || "shown"), Some((PROMPT, "shown")));
        assert_eq!(steady.get(), 2, "the foreground read on both sides");
        // Who is showing is read between the two foreground readings.
        let order = std::cell::RefCell::new(Vec::new());
        let fg = || {
            order.borrow_mut().push("fg");
            PROMPT
        };
        let _ = settled(fg, || order.borrow_mut().push("others"));
        assert_eq!(*order.borrow(), ["fg", "others", "fg"]);
    }

    /// The prompt handing the keyboard back: to an app, no line; to
    /// Settings, a SETTINGS line, as ever.
    #[test]
    fn handing_back_writes_no_line_or_settings() {
        assert_eq!(judge(true, Verdict::Elsewhere, false, Some("PROMPT"), false), None);
        assert_eq!(judge(true, Verdict::Settings, false, Some("PROMPT"), false), Some("SETTINGS"));
        // And Arrange's grace is unchanged beside it.
        assert_eq!(judge(false, Verdict::Prompt, false, Some("ARRANGE"), false), None, "Arrange to the prompt is the owner's");
    }

    #[test]
    fn ancestry_walks_up_and_stops_on_loops() {
        let parents: HashMap<u32, u32> = [(300, 200), (200, ME), (ME, 4), (4, 0)].into();
        assert_eq!(ancestry(&parents, 300), [200, ME, 4]);
        let looped: HashMap<u32, u32> = [(1, 2), (2, 1)].into();
        assert_eq!(ancestry(&looped, 1), [2, 1]);
        assert_eq!(ancestry(&parents, 12345), Vec::<u32>::new());
    }

}
