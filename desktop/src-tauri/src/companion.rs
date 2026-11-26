//! The brain's connection, as the shell sees it (plan 14 build § 4.2, PR 3).
//! The avatar's page holds the one session (its `CompanionClient`); the
//! shell keeps the token it connects with and what the page last said of
//! the link, for the control window.
//!
//! The token is the brain's `COMPANION_TOKEN`, in `companion.token` beside
//! settings.json: a file of its own, since every window hears the settings
//! on every change and settings.json gets read and shared when debugging.
//! The per-user `%APPDATA%` ACL is its protection. It is never logged, and
//! only the avatar's page is given it (`companion_config` in lib.rs).

use std::io;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

pub const TOKEN_FILE: &str = "companion.token";
/// The longest token kept, in characters. The brain's are 32 or more.
pub const TOKEN_MAX_CHARS: usize = 1024;
/// A token file larger than this isn't read: it isn't a token.
const TOKEN_FILE_MAX_BYTES: u64 = 4096;
/// The longest string kept from the page's report, in characters.
pub const REPORT_TEXT_MAX_CHARS: usize = 120;
/// The longest wait kept from the page's report, in seconds.
pub const RETRY_MAX_S: u32 = 3600;

/// A token as the owner pastes it: trimmed, and without the `COMPANION_TOKEN=`
/// and quotes of a line copied from the brain's `.env`. Anything else is
/// refused, never mended: the brain compares it exactly, nothing trimmed.
pub fn token_from(pasted: &str) -> Result<String, String> {
    let mut t = pasted.trim();
    if let Some(rest) = t.strip_prefix("COMPANION_TOKEN=") {
        t = rest.trim();
    }
    for q in ['"', '\''] {
        if t.len() >= 2 && t.starts_with(q) && t.ends_with(q) {
            t = &t[1..t.len() - 1];
            break;
        }
    }
    if t.is_empty() {
        return Err("The token is empty: paste COMPANION_TOKEN from the brain's .env.".to_owned());
    }
    if t.chars().any(char::is_whitespace) {
        return Err("The token has a space in it, which COMPANION_TOKEN never has.".to_owned());
    }
    if t.chars().any(char::is_control) {
        return Err("The token has a control character in it.".to_owned());
    }
    if t.chars().count() > TOKEN_MAX_CHARS {
        return Err(format!("The token is longer than {TOKEN_MAX_CHARS} characters."));
    }
    Ok(t.to_owned())
}

/// The token in `dir`, if there is one that reads as a token.
pub fn read_token(dir: &Path) -> Option<String> {
    let path = dir.join(TOKEN_FILE);
    let meta = std::fs::metadata(&path).ok()?;
    if !meta.is_file() || meta.len() > TOKEN_FILE_MAX_BYTES {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    token_from(text.trim_start_matches('\u{feff}')).ok()
}

/// Save `token` (already checked by `token_from`) in `dir`, atomically.
pub fn write_token(dir: &Path, token: &str) -> io::Result<()> {
    crate::settings::write_atomic(&dir.join(TOKEN_FILE), format!("{token}\n").as_bytes())
}

/// Whether the control window shows at start: when asked to
/// (`FRITZ_SHOW_SETTINGS`), and whenever there is no token yet, so the
/// first run opens where the brain is set up.
pub fn shows_control_at_start(token_present: bool, env: bool) -> bool {
    env || !token_present
}

/// Where the link stands, as the page's `CompanionClient` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Link {
    Unconfigured,
    Connecting,
    Online,
    Offline,
    Refused,
}

/// Why the link is offline or refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Reason {
    Token,
    Version,
    Hello,
    Full,
    Quiet,
    Unreachable,
}

impl Link {
    fn word(self) -> &'static str {
        match self {
            Self::Unconfigured => "unconfigured",
            Self::Connecting => "connecting",
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Refused => "refused",
        }
    }
}

impl Reason {
    fn word(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Version => "version",
            Self::Hello => "hello",
            Self::Full => "full",
            Self::Quiet => "quiet",
            Self::Unreachable => "unreachable",
        }
    }
}

/// The link, as the page reports it and the control window shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompanionStatus {
    pub link: Link,
    pub reason: Option<Reason>,
    /// Seconds until it tries again; `None` while it tries, or won't.
    pub retry_in: Option<u32>,
    /// The welcome's, while online: who the brain takes this desk to be.
    pub identity: Option<String>,
    pub display_name: Option<String>,
    pub thread_id: Option<String>,
    /// The welcome's `server_caps`, while online: what the brain can take
    /// (`audio-in`: it can hear). Only the protocol's caps are kept.
    #[serde(default)]
    pub server_caps: Vec<String>,
}

/// The protocol's caps (companion_protocol.py's `Cap`): anything else a
/// page reports is dropped.
pub const KNOWN_CAPS: [&str; 6] = ["text", "audio-in", "audio-out", "speak-local", "vrm", "widgets"];

impl CompanionStatus {
    /// Before the page has said anything: unconfigured without a token,
    /// else connecting.
    pub fn before_any_report(token_present: bool) -> Self {
        Self {
            link: if token_present { Link::Connecting } else { Link::Unconfigured },
            reason: None,
            retry_in: None,
            identity: None,
            display_name: None,
            thread_id: None,
            server_caps: Vec::new(),
        }
    }

    /// Online, to a brain that can hear (its welcome offered `audio-in`).
    pub fn hears(&self) -> bool {
        self.link == Link::Online && self.server_caps.iter().any(|c| c == "audio-in")
    }

    /// Online, to a brain that speaks a spoken turn's reply back (its
    /// welcome offered `speak-local`).
    pub fn speaks_back(&self) -> bool {
        self.link == Link::Online && self.server_caps.iter().any(|c| c == "speak-local")
    }

    /// What changed enough to log: the link, and why.
    pub fn headline(&self) -> (Link, Option<Reason>) {
        (self.link, self.reason)
    }

    /// One line for the log: never the token, which the page never reports.
    pub fn describe(&self) -> String {
        let mut line = format!("companion: {}", self.link.word());
        if let Some(r) = self.reason {
            line.push_str(&format!(": {}", r.word()));
        }
        if self.link == Link::Online {
            if let Some(id) = &self.identity {
                line.push_str(&format!(" as {id}"));
            }
            if let Some(thread) = &self.thread_id {
                line.push_str(&format!(" ({thread})"));
            }
        }
        line
    }
}

/// `s` cut to `REPORT_TEXT_MAX_CHARS` printable characters; empty is none.
fn printable(s: Option<String>) -> Option<String> {
    s.map(|s| crate::printable(&s, REPORT_TEXT_MAX_CHARS)).filter(|s| !s.trim().is_empty())
}

/// The page's report, checked: an unknown link or reason (or field) is
/// refused; its strings are cut to printable text of a sensible length, and
/// its wait to an hour.
pub fn status_from(report: serde_json::Value) -> Result<CompanionStatus, String> {
    let s: CompanionStatus =
        serde_json::from_value(report).map_err(|e| format!("that isn't a companion report: {e}"))?;
    let mut caps: Vec<String> = Vec::new();
    for c in &s.server_caps {
        if KNOWN_CAPS.contains(&c.as_str()) && !caps.contains(c) {
            caps.push(c.clone());
        }
    }
    Ok(CompanionStatus {
        retry_in: s.retry_in.map(|n| n.min(RETRY_MAX_S)),
        identity: printable(s.identity),
        display_name: printable(s.display_name),
        thread_id: printable(s.thread_id),
        server_caps: caps,
        ..s
    })
}

/// Why push-to-talk can't be heard: his brain can't hear yet.
pub const BRAIN_CANT_HEAR: &str = "His brain can't hear yet: type to him instead.";

/// Why push-to-talk can't be heard now, in the owner's words; `None` when it
/// can. As `say_refusal`: his window first (the page holds the session),
/// then the link, then whether the brain can hear at all.
pub fn ptt_refusal(last: &CompanionStatus, avatar_open: bool) -> Option<&'static str> {
    if !avatar_open {
        return Some("His window is closed: show him first.");
    }
    if last.link != Link::Online {
        return Some(NOT_CONNECTED);
    }
    if !last.hears() {
        return Some(BRAIN_CANT_HEAR);
    }
    None
}

/// The longest turn he can be asked, in characters once trimmed: the
/// protocol's `SAY_MAX_CHARS` (held to protocol.ts by a test).
pub const SAY_MAX_CHARS: usize = 8000;

/// Why a turn wasn't asked when the link isn't online.
pub const NOT_CONNECTED: &str = "He isn't connected to his brain just now.";

/// Why a turn can't be asked now, in words for whoever asked; `None` when
/// it can. Trimmed as the page trims (Unicode's White_Space, which is
/// `str::trim`'s). His window closed comes before the link: the page holds
/// the session, and the last report outlives a window closed under it.
pub fn say_refusal(text: &str, last: &CompanionStatus, avatar_open: bool) -> Option<&'static str> {
    let t = text.trim();
    if t.is_empty() {
        return Some("There's nothing to ask.");
    }
    if t.chars().count() > SAY_MAX_CHARS {
        return Some("That's more than he can be asked at once (8,000 characters at most).");
    }
    if !avatar_open {
        return Some("His window is closed: show him first.");
    }
    if last.link != Link::Online {
        return Some(NOT_CONNECTED);
    }
    None
}

/// The text `FRITZ_PRETEND_SAY` asks for, in a debug build only (`debug`
/// is `cfg!(debug_assertions)` at the one caller): a release build never
/// reads it.
pub fn pretend_say_text(debug: bool, env: Option<String>) -> Option<String> {
    if !debug {
        return None;
    }
    env.filter(|t| !t.trim().is_empty())
}

/// What the page last reported; `None` before its first report.
static LAST: Mutex<Option<CompanionStatus>> = Mutex::new(None);

/// Keep the page's report, returning the one it replaces.
pub fn keep(s: CompanionStatus) -> Option<CompanionStatus> {
    LAST.lock().unwrap_or_else(|e| e.into_inner()).replace(s)
}

/// Forget the page's last report: his window, and its session, are gone.
pub fn forget() {
    LAST.lock().unwrap_or_else(|e| e.into_inner()).take();
}

/// The link now: the page's last report, or what it will be before one.
pub fn current(token_present: bool) -> CompanionStatus {
    LAST.lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(|| CompanionStatus::before_any_report(token_present))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_pasted_token_is_trimmed_and_unwrapped_and_never_mended() {
        let t = "t0k-".to_owned() + &"x".repeat(40);
        for pasted in [
            t.clone(),
            format!("  {t}\r\n"),
            format!("COMPANION_TOKEN={t}"),
            format!("COMPANION_TOKEN=\"{t}\""),
            format!("'{t}'"),
            format!(" COMPANION_TOKEN= {t} "),
        ] {
            assert_eq!(token_from(&pasted).as_deref(), Ok(t.as_str()), "{pasted:?}");
        }
        for (bad, says) in [
            ("", "empty"),
            ("   ", "empty"),
            ("COMPANION_TOKEN=", "empty"),
            ("\"\"", "empty"),
            ("abc def", "space"),
            ("abc\u{0}def", "control"),
            ("abc\u{7f}def", "control"),
        ] {
            let e = token_from(bad).unwrap_err();
            assert!(e.contains(says), "{bad:?}: {e}");
        }
        assert!(token_from(&"x".repeat(TOKEN_MAX_CHARS)).is_ok());
        assert!(token_from(&"x".repeat(TOKEN_MAX_CHARS + 1)).unwrap_err().contains("longer"));
        // A lone quote is part of the token, not a wrapping.
        assert_eq!(token_from("\"abc").as_deref(), Ok("\"abc"));
    }

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fritz-token-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_token_is_written_and_read_back() {
        let d = dir("roundtrip");
        assert_eq!(read_token(&d), None, "no file");
        write_token(&d, "t0k-abc").unwrap();
        assert_eq!(std::fs::read_to_string(d.join(TOKEN_FILE)).unwrap(), "t0k-abc\n");
        assert_eq!(read_token(&d).as_deref(), Some("t0k-abc"));
        write_token(&d, "t0k-new").unwrap();
        assert_eq!(read_token(&d).as_deref(), Some("t0k-new"), "replaced");
        let left: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, [TOKEN_FILE], "nothing left behind");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_file_that_isnt_a_token_is_none() {
        let d = dir("odd");
        // A token, then 5 KiB of blank lines: a token once trimmed, but no
        // file a token is kept in.
        std::fs::write(d.join(TOKEN_FILE), format!("t0k-abc{}", "\n".repeat(5 * 1024))).unwrap();
        assert_eq!(read_token(&d), None, "5 KiB");
        std::fs::write(d.join(TOKEN_FILE), format!("t0k-abc{}", "\n".repeat(4000))).unwrap();
        assert_eq!(read_token(&d).as_deref(), Some("t0k-abc"), "under 4 KiB");
        std::fs::write(d.join(TOKEN_FILE), "  \n").unwrap();
        assert_eq!(read_token(&d), None, "blank");
        std::fs::write(d.join(TOKEN_FILE), "two words\n").unwrap();
        assert_eq!(read_token(&d), None, "a space");
        // A hand edit: the .env's line, with Notepad's byte order mark.
        std::fs::write(d.join(TOKEN_FILE), "\u{feff}COMPANION_TOKEN=t0k-hand\r\n").unwrap();
        assert_eq!(read_token(&d).as_deref(), Some("t0k-hand"));
        std::fs::remove_file(d.join(TOKEN_FILE)).unwrap();
        std::fs::create_dir(d.join(TOKEN_FILE)).unwrap();
        assert_eq!(read_token(&d), None, "a folder");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_control_window_shows_at_start_without_a_token_or_when_asked() {
        assert!(shows_control_at_start(false, false), "the first run");
        assert!(shows_control_at_start(false, true));
        assert!(shows_control_at_start(true, true), "FRITZ_SHOW_SETTINGS");
        assert!(!shows_control_at_start(true, false), "an ordinary start");
    }

    /// The settings every window hears never carry the token: it has no
    /// field there.
    #[test]
    fn the_token_is_never_in_the_settings() {
        let s = serde_json::to_string(&crate::settings::Settings::default()).unwrap().to_lowercase();
        assert!(!s.contains("token"), "{s}");
    }

    #[test]
    fn the_pages_report_is_checked() {
        let ok = json!({"link": "online", "reason": null, "retryIn": null, "identity": "desktop-nick-pc",
                        "displayName": "Nick", "threadId": "desktop-nick-pc#desktop", "serverCaps": ["text", "audio-in"]});
        let s = status_from(ok.clone()).unwrap();
        assert_eq!(s.link, Link::Online);
        assert_eq!(s.describe(), "companion: online as desktop-nick-pc (desktop-nick-pc#desktop)");
        assert_eq!(serde_json::to_value(&s).unwrap(), ok, "as the page sent it");
        for bad in [
            json!({"link": "sideways", "reason": null, "retryIn": null, "identity": null, "displayName": null, "threadId": null}),
            json!({"link": "offline", "reason": "sulking", "retryIn": 1, "identity": null, "displayName": null, "threadId": null}),
            json!({"link": "offline", "reason": "full", "retryIn": -1, "identity": null, "displayName": null, "threadId": null}),
            json!({"link": "offline", "reason": "full", "retryIn": 1, "identity": null, "displayName": null, "threadId": null, "token": "x"}),
            json!({"reason": "full"}),
            json!("online"),
        ] {
            assert!(status_from(bad.clone()).is_err(), "{bad}");
        }
        // A field left out is none, as null is.
        let bare = status_from(json!({"link": "offline"})).unwrap();
        assert_eq!(bare, CompanionStatus { link: Link::Offline, ..CompanionStatus::before_any_report(true) });
        let long = json!({"link": "online", "reason": null, "retryIn": 99999, "identity": format!("a\u{1b}[31m{}", "x".repeat(500)),
                          "displayName": "  ", "threadId": "t\r\nx"});
        let s = status_from(long).unwrap();
        let id = s.identity.unwrap();
        assert_eq!(id.chars().count(), REPORT_TEXT_MAX_CHARS);
        assert!(!id.contains('\u{1b}'), "{id}");
        assert_eq!(s.display_name, None, "blank is none");
        assert_eq!(s.thread_id.as_deref(), Some("tx"));
        assert_eq!(s.retry_in, Some(RETRY_MAX_S));
        // Only the protocol's caps, each once, in the order given.
        let caps = status_from(json!({"link": "online", "serverCaps": ["audio-in", "x", "text", "audio-in", "AUDIO-IN", ""]})).unwrap();
        assert_eq!(caps.server_caps, ["audio-in", "text"]);
        assert!(status_from(json!({"link": "online", "serverCaps": "audio-in"})).is_err(), "a list, or nothing");
        assert!(status_from(json!({"link": "online", "serverCaps": [1]})).is_err());
    }

    /// He hears only online, to a brain whose welcome offered `audio-in`;
    /// push-to-talk's refusal says which of those isn't so, his window first.
    #[test]
    fn push_to_talk_is_heard_only_by_a_brain_that_can_hear() {
        let hearing = CompanionStatus { link: Link::Online, server_caps: vec!["text".into(), "audio-in".into()],
                                        ..CompanionStatus::before_any_report(true) };
        assert!(hearing.hears());
        assert_eq!(ptt_refusal(&hearing, true), None);
        let deaf = CompanionStatus { server_caps: vec!["text".into()], ..hearing.clone() };
        assert!(!deaf.hears());
        assert_eq!(ptt_refusal(&deaf, true), Some(BRAIN_CANT_HEAR));
        for link in [Link::Unconfigured, Link::Connecting, Link::Offline, Link::Refused] {
            let s = CompanionStatus { link, ..hearing.clone() };
            assert!(!s.hears(), "{link:?}");
            assert_eq!(ptt_refusal(&s, true), Some(NOT_CONNECTED), "{link:?}");
        }
        assert!(ptt_refusal(&hearing, false).unwrap().contains("window is closed"));
        assert!(ptt_refusal(&CompanionStatus::before_any_report(true), false).unwrap().contains("window"));
    }

    /// He answers aloud only online, to a brain whose welcome offered
    /// `speak-local` (it sends a spoken turn's sentences, PR 10).
    #[test]
    fn a_brain_speaks_back_only_online_and_only_when_it_says_so() {
        let speaking = CompanionStatus { link: Link::Online, server_caps: vec!["text".into(), "speak-local".into()],
                                         ..CompanionStatus::before_any_report(true) };
        assert!(speaking.speaks_back());
        assert!(!speaking.hears(), "speaking isn't hearing");
        let quiet = CompanionStatus { server_caps: vec!["text".into(), "audio-in".into()], ..speaking.clone() };
        assert!(!quiet.speaks_back());
        for link in [Link::Unconfigured, Link::Connecting, Link::Offline, Link::Refused] {
            assert!(!CompanionStatus { link, ..speaking.clone() }.speaks_back(), "{link:?}");
        }
        // As the page reports it, the cap survives the filter.
        let reported = status_from(serde_json::json!({"link": "online", "serverCaps": ["speak-local"]})).unwrap();
        assert!(reported.speaks_back());
    }

    #[test]
    fn the_log_line_says_what_changed() {
        let refused = status_from(json!({"link": "refused", "reason": "token", "retryIn": null, "identity": null,
                                         "displayName": null, "threadId": null})).unwrap();
        assert_eq!(refused.describe(), "companion: refused: token");
        let offline = CompanionStatus { link: Link::Offline, reason: Some(Reason::Unreachable), retry_in: Some(4),
                                        ..CompanionStatus::before_any_report(true) };
        assert_eq!(offline.describe(), "companion: offline: unreachable");
        // A new wait is no news; a new reason is.
        assert_eq!(offline.headline(), CompanionStatus { retry_in: None, ..offline.clone() }.headline());
        assert_ne!(offline.headline(), CompanionStatus { reason: Some(Reason::Full), ..offline.clone() }.headline());
        assert_eq!(CompanionStatus::before_any_report(false).link, Link::Unconfigured);
        assert_eq!(CompanionStatus::before_any_report(true).link, Link::Connecting);
    }

    /// A turn is asked only with something to ask, his window open and the
    /// link online; each refusal says which.
    #[test]
    fn a_turn_is_asked_only_when_he_can_answer() {
        let online = CompanionStatus { link: Link::Online, ..CompanionStatus::before_any_report(true) };
        assert_eq!(say_refusal("good evening", &online, true), None);
        assert_eq!(say_refusal(&"x".repeat(SAY_MAX_CHARS), &online, true), None);
        // Counted in characters, as the protocol counts them, once trimmed.
        assert_eq!(say_refusal(&format!("  {}\u{85}", "😀".repeat(SAY_MAX_CHARS)), &online, true), None);
        assert!(say_refusal(&"x".repeat(SAY_MAX_CHARS + 1), &online, true).unwrap().contains("8,000"));
        for blank in ["", "   ", "\r\n\t", "\u{3000}\u{85}"] {
            assert_eq!(say_refusal(blank, &online, true), Some("There's nothing to ask."), "{blank:?}");
        }
        assert!(say_refusal("hi", &online, false).unwrap().contains("window is closed"));
        for link in [Link::Unconfigured, Link::Connecting, Link::Offline, Link::Refused] {
            let s = CompanionStatus { link, ..online.clone() };
            assert!(say_refusal("hi", &s, true).unwrap().contains("isn't connected"), "{link:?}");
        }
        // The window first: a report can outlive the window that made it.
        assert!(say_refusal("hi", &CompanionStatus::before_any_report(true), false).unwrap().contains("window"));
        assert_eq!(say_refusal("", &online, false), Some("There's nothing to ask."));
    }

    /// His window gone, its last report goes with it: an Ask then is
    /// refused until a new page says it is online. (The one test that
    /// touches the kept report.)
    #[test]
    fn a_closed_windows_report_is_forgotten() {
        let online = CompanionStatus { link: Link::Online, ..CompanionStatus::before_any_report(true) };
        keep(online.clone());
        assert_eq!(current(true), online);
        forget();
        assert_eq!(current(true), CompanionStatus::before_any_report(true));
        assert!(say_refusal("hi", &current(true), true).unwrap().contains("isn't connected"));
    }

    #[test]
    fn the_pretend_say_is_a_debug_builds_only() {
        assert_eq!(pretend_say_text(true, Some("good evening".into())).as_deref(), Some("good evening"));
        assert_eq!(pretend_say_text(false, Some("good evening".into())), None);
        assert_eq!(pretend_say_text(true, None), None);
        assert_eq!(pretend_say_text(true, Some("  ".into())), None);
    }
}
