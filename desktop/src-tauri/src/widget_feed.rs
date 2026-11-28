//! The brain's widgets, as the avatar's page hears them (plan 14 build
//! § 4.7): it holds the device's one session, subscribes to the widgets
//! and hands each `widget` frame here (`companion_widget`), and the shell
//! passes it on to the study scene's windows (`widget:seen`), keeping the
//! newest of each for a page that starts later (`widgets_now`). One
//! connection to the brain per device, as the plan chose, not one per
//! window.
//!
//! The study circle reads only whether each is fresh (its threads are lit
//! while it is); the payloads are kept for PR 13's widgets. Only the
//! avatar's page may hand one in; a name is a widget's (lower case, 32
//! characters at most), a payload a JSON object of 64 KiB at most, and no
//! more than eight are kept, the brain's own limit on a subscription.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use serde::Serialize;
use serde_json::Value;

/// To each study scene window, one widget as it comes.
pub const SEEN_EVENT: &str = "widget:seen";
/// The most kept: the brain's `WIDGETS_MAX`.
pub const MAX_WIDGETS: usize = 8;
/// The largest payload kept, as JSON.
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;

/// A widget as a page hears it: how long it holds (`ttl_s`, 0 until the
/// next), how long ago it came (`age_ms`), and what it says.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Seen {
    pub name: String,
    pub ttl_s: u32,
    pub age_ms: u64,
    pub payload: Value,
}

#[derive(Debug, Clone)]
struct Kept {
    ttl_s: u32,
    at: Instant,
    payload: Value,
}

static FEED: Mutex<BTreeMap<String, Kept>> = Mutex::new(BTreeMap::new());

fn feed() -> MutexGuard<'static, BTreeMap<String, Kept>> {
    FEED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether `name` is a widget's: a lower-case letter, then up to 31 more
/// letters, digits, `-` or `_`. Pure.
pub fn name_ok(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && name.len() <= 32
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Why a widget handed in isn't kept, if it isn't. Pure.
pub fn refusal(name: &str, payload: &Value, known: usize, is_new: bool) -> Option<&'static str> {
    if !name_ok(name) {
        return Some("that isn't a widget's name");
    }
    if !payload.is_object() {
        return Some("a widget's payload is an object");
    }
    if serde_json::to_vec(payload).map_or(usize::MAX, |b| b.len()) > MAX_PAYLOAD_BYTES {
        return Some("that widget's payload is too big");
    }
    if is_new && known >= MAX_WIDGETS {
        return Some("no more widgets are kept");
    }
    None
}

/// Keep widget `name` as heard now; what to pass on, or why not.
pub fn keep(name: &str, ttl_s: u32, payload: Value) -> Result<Seen, String> {
    let mut all = feed();
    if let Some(why) = refusal(name, &payload, all.len(), !all.contains_key(name)) {
        return Err(why.to_owned());
    }
    all.insert(name.to_owned(), Kept { ttl_s, at: Instant::now(), payload: payload.clone() });
    Ok(Seen { name: name.to_owned(), ttl_s, age_ms: 0, payload })
}

/// Every widget kept, with its age at `now`.
pub fn now_at(now: Instant) -> Vec<Seen> {
    feed()
        .iter()
        .map(|(name, k)| Seen {
            name: name.clone(),
            ttl_s: k.ttl_s,
            age_ms: u64::try_from(now.saturating_duration_since(k.at).as_millis()).unwrap_or(u64::MAX),
            payload: k.payload.clone(),
        })
        .collect()
}

/// Every widget kept, with its age now.
pub fn now() -> Vec<Seen> {
    now_at(Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn names_are_widgets() {
        for ok in ["clock", "weather", "desk", "a", "now-playing", "agenda_2", &"a".repeat(32)] {
            assert!(name_ok(ok), "{ok:?}");
        }
        for bad in ["", "Clock", "1clock", "-x", "we ather", "desk!", "ünï", &"a".repeat(33)] {
            assert!(!name_ok(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_widget_is_refused_for_what_it_is() {
        let p = json!({"time": "09:00"});
        assert_eq!(refusal("clock", &p, 0, true), None);
        assert!(refusal("Clock", &p, 0, true).is_some());
        assert!(refusal("clock", &json!([1]), 0, true).is_some(), "an array isn't a payload");
        assert!(refusal("clock", &json!("x"), 0, true).is_some());
        let big = json!({"s": "x".repeat(MAX_PAYLOAD_BYTES)});
        assert!(refusal("clock", &big, 0, true).is_some());
        let at_most = json!({"s": "x".repeat(MAX_PAYLOAD_BYTES - 8)});
        assert_eq!(serde_json::to_vec(&at_most).unwrap().len(), MAX_PAYLOAD_BYTES);
        assert_eq!(refusal("clock", &at_most, 0, true), None);
        assert!(refusal("ninth", &p, MAX_WIDGETS, true).is_some(), "no ninth");
        assert_eq!(refusal("clock", &p, MAX_WIDGETS, false), None, "an eighth is replaced");
    }

    #[test]
    fn the_newest_of_each_is_kept_with_its_age() {
        // Names no other test uses: the feed is the process's.
        let first = keep("t-feed-a", 120, json!({"n": 1})).unwrap();
        assert_eq!(first, Seen { name: "t-feed-a".into(), ttl_s: 120, age_ms: 0, payload: json!({"n": 1}) });
        keep("t-feed-a", 60, json!({"n": 2})).unwrap();
        keep("t-feed-b", 0, json!({})).unwrap();
        let later = Instant::now() + Duration::from_millis(1500);
        let all = now_at(later);
        let a = all.iter().find(|s| s.name == "t-feed-a").unwrap();
        assert_eq!((a.ttl_s, &a.payload), (60, &json!({"n": 2})), "the newest");
        assert!(a.age_ms >= 1500, "{}", a.age_ms);
        assert!(all.iter().any(|s| s.name == "t-feed-b"));
        assert!(keep("Bad", 1, json!({})).is_err());
        assert!(!now().iter().any(|s| s.name == "Bad"));
        let wire = serde_json::to_value(&first).unwrap();
        let mut keys: Vec<&str> = wire.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["ageMs", "name", "payload", "ttlS"]);
    }
}
