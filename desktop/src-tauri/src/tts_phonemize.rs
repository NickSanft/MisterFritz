// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// divora-core/src/tts/phonemize.rs at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.
//! Text → IPA phonemes via the bundled **espeak-ng** CLI, invoked as an
//! arm's-length subprocess.
//!
//! espeak-ng is **GPL-3.0**. We never link it: we spawn its command-line
//! binary and read its stdout. Under the FSF's "mere aggregation" guidance
//! this keeps `DivoraVoice`'s own code MIT — espeak-ng stays a separate GPL
//! component (its binary + `espeak-ng-data` ship alongside the app and are
//! disclosed in the README). All linking-free.
//!
//! The argument construction and IPA cleanup are pure and unit-tested; the
//! actual subprocess call is exercised on the desktop once the binary is
//! staged (see `MANUAL_TESTS.md`).
//!
//! Fritz (plan 14, PR 10): the text goes to espeak-ng on its stdin, as
//! UTF-8, not as an argument. An argument reaches a console program through
//! Windows' ANSI code page, which has no room for much of what a reply holds
//! (an em dash's neighbours, any letter outside Western Europe's). And no
//! thiserror: the errors say themselves, so no crate is added for them.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Locates the bundled espeak-ng binary (and its data dir) and turns text
/// into the continuous IPA string Kokoro's tokenizer indexes.
#[derive(Debug, Clone)]
pub struct Phonemizer {
    /// Path to the `espeak-ng` executable.
    bin: PathBuf,
    /// The directory *containing* `espeak-ng-data` (passed via `--path`).
    /// On a bundled/portable layout espeak-ng won't find its data otherwise,
    /// so this is required when present. `None` if no data dir is bundled.
    data: Option<PathBuf>,
}

/// Failures from the phonemizer. [`PhonemizeError::NotInstalled`] is the
/// graceful "espeak-ng isn't staged yet" path the synthesis gate relies on.
#[derive(Debug)]
pub enum PhonemizeError {
    NotInstalled(PathBuf),
    Spawn(std::io::Error),
    Exited { code: i32, stderr: String },
}

impl fmt::Display for PhonemizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled(p) => write!(f, "espeak-ng binary not found at {}", p.display()),
            Self::Spawn(e) => write!(f, "could not run espeak-ng: {e}"),
            Self::Exited { code, stderr } => write!(f, "espeak-ng exited with status {code}: {stderr}"),
        }
    }
}

impl std::error::Error for PhonemizeError {}

impl From<std::io::Error> for PhonemizeError {
    fn from(e: std::io::Error) -> Self {
        Self::Spawn(e)
    }
}

impl Phonemizer {
    /// Build a phonemizer for a bundled espeak-ng layout under `dir`:
    /// `dir/espeak-ng(.exe)` plus an optional `dir/espeak-ng-data`.
    #[must_use]
    pub fn from_dir(dir: &Path) -> Self {
        let bin = dir.join(espeak_bin_name());
        // espeak-ng's `--path` wants the directory *containing* `espeak-ng-data`
        // (this dir), not the data dir itself.
        let data = dir
            .join("espeak-ng-data")
            .is_dir()
            .then(|| dir.to_path_buf());
        Self { bin, data }
    }

    /// Whether the espeak-ng binary is physically present, so callers can gate
    /// synthesis behind a clear "voice not installed" rather than spawning a
    /// missing process.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.bin.exists()
    }

    /// Phonemize `text` in `lang` (e.g. `"en-us"`) to a continuous IPA string
    /// with stress marks — the form Kokoro's vocab maps to token ids.
    ///
    /// Returns [`PhonemizeError::NotInstalled`] without spawning anything when
    /// the binary is absent (so it can never hang waiting on a missing tool).
    ///
    /// v1.60.0 (D35): on Windows the child starts with no console window.
    /// `espeak-ng.exe` is a console program, and an installed Divora is a GUI
    /// process with no console of its own to share, so each spawn — one per
    /// chunk of a line — could open a window of its own, over a game, when a
    /// timer's line fires unattended.
    pub fn phonemize(&self, text: &str, lang: &str) -> Result<String, PhonemizeError> {
        if !self.is_available() {
            return Err(PhonemizeError::NotInstalled(self.bin.clone()));
        }
        let mut cmd = Command::new(&self.bin);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // CREATE_NO_WINDOW.
            cmd.creation_flags(0x0800_0000);
        }
        cmd.args(espeak_args(lang, self.data.as_deref()))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn()?;
        // All of it, then a newline, then closed, so espeak-ng sees the end
        // of its input. The newline is not optional: `--stdin` takes the
        // input's last byte for its line end and overwrites it, so without
        // one "Buy milk" is read "Buy mil" and a last "é" is cut in half. A
        // sentence is far below a pipe's buffer, so this can't wait on the
        // child reading.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes())?;
            stdin.write_all(b"\n")?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(PhonemizeError::Exited {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        Ok(clean_ipa(&String::from_utf8_lossy(&output.stdout)))
    }
}

/// The espeak-ng arguments for quiet IPA phonemization. `-q` suppresses audio;
/// `--ipa` emits IPA with stress marks (matching the `phonemizer` espeak
/// backend Kokoro trained on); `--path` points at the bundled data dir;
/// `--stdin` takes the text from stdin, all of it before speaking.
fn espeak_args(lang: &str, data: Option<&Path>) -> Vec<String> {
    let mut args = vec![
        "-q".to_string(),
        "--ipa".to_string(),
        "-v".to_string(),
        lang.to_string(),
    ];
    if let Some(dir) = data {
        args.push("--path".to_string());
        args.push(plain_path(dir));
    }
    args.push("--stdin".to_string());
    args
}

/// `dir` without Windows' verbatim `\\?\` prefix: `\\?\C:\x` → `C:\x`,
/// `\\?\UNC\host\share` → `\\host\share`, anything else as it is.
///
/// v1.60.0: a built app's resource dir comes from Tauri's canonicalized exe
/// path, so its `tts` dir starts `\\?\` (only a dev build, which uses the
/// source-staged assets, didn't). espeak-ng joins `/espeak-ng-data` onto
/// `--path` with a forward slash, which a verbatim path never treats as a
/// separator, so it missed its data, fell back to the dir itself and exited
/// on `phontab`: every Kokoro line failed outside a dev build (seen on this
/// version's release build; for an installed build, from the code: none was
/// tried).
fn plain_path(dir: &Path) -> String {
    let s = dir.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        // Only a drive path has a plain form (`\\?\Volume{…}\` has none).
        let b = rest.as_bytes();
        if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
            return rest.to_string();
        }
    }
    s.into_owned()
}

fn espeak_bin_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "espeak-ng.exe"
    } else {
        "espeak-ng"
    }
}

/// Normalize espeak-ng's IPA stdout into a single clean phoneme line: drop
/// language-switch markers like `(en)`, strip the tie bar espeak inserts in
/// affricates (Kokoro stores affricate components separately), and collapse
/// whitespace/newlines to single spaces.
fn clean_ipa(raw: &str) -> String {
    let mut stripped = String::with_capacity(raw.len());
    let mut in_paren = false;
    for ch in raw.chars() {
        match ch {
            '(' => in_paren = true,
            ')' => in_paren = false,
            _ if in_paren => {}
            '\u{0361}' | '\u{200d}' => {} // combining tie bar / zero-width joiner
            '\n' | '\r' | '\t' => stripped.push(' '),
            _ => stripped.push(ch),
        }
    }
    // Collapse runs of spaces and trim.
    let mut out = String::with_capacity(stripped.len());
    let mut prev_space = false;
    for ch in stripped.trim().chars() {
        if ch == ' ' {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_are_quiet_ipa_with_lang_and_the_text_on_stdin() {
        let args = espeak_args("en-us", None);
        assert_eq!(args, vec!["-q", "--ipa", "-v", "en-us", "--stdin"]);
    }

    #[test]
    fn args_put_the_path_before_stdin() {
        let args = espeak_args("en-us", Some(Path::new("C:/e/tts")));
        assert_eq!(args, vec!["-q", "--ipa", "-v", "en-us", "--path", "C:/e/tts", "--stdin"]);
    }

    /// v1.60.0: a built app's `tts` dir is `\\?\`-prefixed (Tauri
    /// canonicalizes the exe path), and espeak-ng can't find its data under
    /// one, so `--path` always goes out plain.
    #[test]
    fn args_pass_a_plain_path_never_a_verbatim_one() {
        let args = espeak_args("en-us", Some(Path::new(r"\\?\C:\Program Files\DivoraVoice\tts")));
        assert_eq!(
            args,
            vec!["-q", "--ipa", "-v", "en-us", "--path", r"C:\Program Files\DivoraVoice\tts", "--stdin"],
        );
        assert_eq!(plain_path(Path::new(r"\\?\UNC\host\share\tts")), r"\\host\share\tts");
        assert_eq!(plain_path(Path::new(r"C:\e\tts")), r"C:\e\tts");
        assert_eq!(plain_path(Path::new("/opt/divora/tts")), "/opt/divora/tts");
        // No plain form: left as it is.
        assert_eq!(plain_path(Path::new(r"\\?\Volume{1}\tts")), r"\\?\Volume{1}\tts");
    }

    /// The text is never an argument: whatever it holds (a dash, a quote,
    /// `--path`), it is only ever read from stdin, with a newline after it.
    /// Read from the source, as only the real espeak-ng can show it, and the
    /// real test isn't run in CI. Needles are split by `concat!`.
    #[test]
    fn the_text_is_never_an_argument() {
        let args = espeak_args("en-gb", Some(Path::new("C:/e/tts")));
        assert!(!args.iter().any(|a| a == "--"), "no text after a separator");
        assert_eq!(args.last().map(String::as_str), Some("--stdin"));
        let file = include_str!("tts_phonemize.rs").replace('\r', "");
        let prod = &file[..file.find(concat!("#[cfg(test)]\nmod ", "tests {")).unwrap()];
        let body = &prod[prod.find(concat!("pub fn ", "phonemize(")).unwrap()..];
        let body = &body[..body.find("\n    }\n").unwrap()];
        let at = |needle: &str| body.find(needle).unwrap_or_else(|| panic!("{needle}"));
        assert_eq!(body.matches(concat!(".arg", "s(")).count(), 1, "only espeak_args'");
        assert!(body.contains(concat!("cmd.args(", "espeak_args(lang, self.data.as_deref()))")));
        assert_eq!(body.matches(concat!(".arg", "(")).count(), 0);
        let text = at(concat!("stdin.write_all(", "text.as_bytes())?;"));
        let newline = at(concat!("stdin.write_all(", "b\"\\n\")?;"));
        let wait = at(concat!("child.wait_", "with_output()"));
        assert!(text < newline && newline < wait);
        assert_eq!(body.matches(concat!("text", ".as_bytes()")).count(), 1, "the text goes nowhere else");
    }

    /// The real espeak-ng, handed its dir the way a built app gets it: through
    /// `canonicalize`, which on Windows returns a `\\?\` path, and given text
    /// that the ANSI code page can't carry. Needs the fetched files
    /// (`scripts/fetch_voice.py`), so it's `#[ignore]`d: run with
    /// `FRITZ_TTS_TEST_DIR` set to that folder and `-- --ignored`.
    #[test]
    #[ignore = "needs the fetched espeak-ng (FRITZ_TTS_TEST_DIR)"]
    fn espeak_reads_its_data_from_a_verbatim_dir_and_utf8_from_stdin() {
        let dir = std::env::var_os("FRITZ_TTS_TEST_DIR").expect("FRITZ_TTS_TEST_DIR");
        let dir = Path::new(&dir).canonicalize().expect("the tts dir");
        #[cfg(windows)]
        assert!(dir.to_string_lossy().starts_with(r"\\?\"), "{dir:?}");
        let p = Phonemizer::from_dir(&dir);
        let plain = p.phonemize("I'm back.", "en-gb").expect("espeak-ng should find its data");
        assert_ne!(plain, "");
        // A curly apostrophe reads as the straight one, which it can't if
        // the code page turned it into something else.
        assert_eq!(p.phonemize("I\u{2019}m back.", "en-gb").unwrap(), plain);
        // Letters no Western code page has, not even by Windows' best fit
        // (which makes "Ł" an "L" and Greek "αβγ" "aß?"), are read as
        // letters: as an argument, these came as question marks, which are
        // nothing to say.
        for text in ["\u{416}\u{443}\u{43a}", "\u{4f60}\u{597d}"] {
            let ipa = p.phonemize(text, "en-gb").unwrap();
            assert!(!ipa.is_empty(), "{text}: {ipa:?}");
            assert_eq!(p.phonemize("???", "en-gb").unwrap(), "", "question marks are nothing to say");
        }
        // The last letter is read, with no stop after it, or a multi-byte
        // letter last (espeak-ng's --stdin overwrites the input's last byte).
        assert_eq!(p.phonemize("Buy milk", "en-gb").unwrap(), p.phonemize("Buy milk.", "en-gb").unwrap());
        assert_eq!(p.phonemize("Hi", "en-gb").unwrap(), p.phonemize("Hi.", "en-gb").unwrap());
        assert_eq!(p.phonemize("caf\u{e9}", "en-gb").unwrap(), p.phonemize("caf\u{e9}.", "en-gb").unwrap());
    }

    #[test]
    fn missing_binary_reports_not_installed_without_spawning() {
        let p = Phonemizer::from_dir(Path::new("Z:/divora/does-not-exist"));
        assert!(!p.is_available());
        match p.phonemize("hello", "en-us") {
            Err(PhonemizeError::NotInstalled(_)) => {}
            other => panic!("expected NotInstalled, got {other:?}"),
        }
    }

    #[test]
    fn clean_ipa_strips_markers_ties_and_collapses_space() {
        let raw = "  h\u{0361}ə(en)l\u{200d}ˈəʊ \n world  ";
        // tie bar + ZWJ removed, "(en)" removed, whitespace collapsed/trimmed.
        assert_eq!(clean_ipa(raw), "həlˈəʊ world");
    }

    #[test]
    fn clean_ipa_empty_is_empty() {
        assert_eq!(clean_ipa("   \n  "), "");
    }

    #[test]
    fn the_errors_say_what_went_wrong() {
        let e = PhonemizeError::Exited { code: 1, stderr: "no phontab".into() };
        assert_eq!(e.to_string(), "espeak-ng exited with status 1: no phontab");
        assert!(PhonemizeError::NotInstalled(PathBuf::from("x")).to_string().contains("not found"));
    }

    /// v1.60.0 (D35): `espeak-ng` starts with `CREATE_NO_WINDOW` on Windows,
    /// so a timer's line never opens a console over a game. Whether a window
    /// shows depends on how the app was started (a debug build shares its
    /// console), so this reads the source: the flag is set under
    /// `cfg(windows)` in `phonemize`, before the child is spawned. Needles
    /// are split by `concat!` so they can't match themselves.
    #[test]
    fn phonemize_never_opens_a_console() {
        let file = include_str!("tts_phonemize.rs").replace('\r', "");
        let tests_at = file
            .find(concat!("#[cfg(test)]\nmod ", "tests {"))
            .expect("the tests module");
        let prod = &file[..tests_at];
        let start = prod
            .find(concat!("pub fn ", "phonemize("))
            .expect("phonemize");
        let body = &prod[start..];
        let body = &body[..body.find("\n    }\n").expect("its end")];
        let at = |needle: &str| body.find(needle).unwrap_or_else(|| panic!("{needle}"));
        let cfg = at(concat!("#[cfg(", "windows)]"));
        let flag = at(concat!("cmd.creation_flags(", "0x0800_0000);"));
        let spawn = at(concat!("cmd.", "spawn()"));
        assert!(cfg < flag && flag < spawn);
        assert_eq!(prod.matches(concat!("creation_", "flags(")).count(), 1);
        assert_eq!(prod.matches(concat!(".spawn", "()")).count(), 1, "one child, and it is made here");
    }
}
