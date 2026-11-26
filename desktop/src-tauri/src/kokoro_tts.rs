// Lifted in part from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// divora-core/src/tts/kokoro.rs and divora-core/src/tts/mod.rs at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.
//! His voice (plan 14, PR 10): Kokoro on the CPU, from the files
//! `scripts/fetch_voice.py` puts in `%APPDATA%\com.divora.fritz\tts\`.
//!
//! Text → espeak-ng IPA (`tts_phonemize`) → Kokoro's token ids
//! (`tts_tokens`) → Kokoro's ONNX model with the voice's style vector → mono
//! f32 samples at 24 kHz. DivoraVoice's `StylePack` and `PRESET_VOICES` are
//! lifted as they are, and `infer` without ndarray; the rest is Fritz's own:
//!
//! - **Loaded once, then kept.** DivoraVoice loads the model (92 MB, an
//!   optimised graph) for every line; Fritz speaks a sentence at a time, so
//!   the session, the vocab and the voice pack stay loaded after the first
//!   (`ENGINE`, one sentence at a time).
//! - **ONNX Runtime from the data folder only**, by its full path
//!   (`ort::init_from`), never by name: Windows has an older copy of its own
//!   in System32 that `ort` refuses, and `ort` panics when a load by name
//!   fails. And only once Fritz has loaded it himself and asked its version
//!   (`runtime_usable`): `ort::init_from` hangs for good on a DLL it can't
//!   load or won't take. No `ort` call is made before that load has
//!   succeeded.
//! - **Four threads, none spinning** (the owner games on this PC).
//! - **A voice it doesn't know is George** (`bm_george`, the default), so a
//!   setting from elsewhere can't leave him silent.
//!
//! Nothing here logs what he says: only how long it was, and how long it
//! took.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::tts_phonemize::{PhonemizeError, Phonemizer};
use crate::tts_tokens;

/// Kokoro renders at a fixed 24 kHz.
pub const SAMPLE_RATE: u32 = 24_000;
/// The voice Settings start with, and the one an unknown voice falls back to.
pub const DEFAULT_VOICE: &str = "bm_george";
/// The longest text one call will speak, in characters. The brain sends a
/// sentence at a time, far below this; it only bounds the work one call
/// can ask for.
pub const MAX_TEXT_CHARS: usize = 2_000;
/// Kokoro's threads: four of the owner's 24 cores, so a reply never takes
/// the whole CPU from a game.
const THREADS: usize = 4;
/// The most a piece of text given to Kokoro holds, in characters, so its
/// phonemes stay under Kokoro's 510 as a rule (`render` halves one that
/// doesn't). A sentence from the brain is shorter, and is said whole.
const CHUNK_CHARS: usize = 300;

const MODEL_FILE: &str = "kokoro-v1.0.int8.onnx";
/// DivoraVoice's compact `DVTS` voice pack (its preset voices only).
const VOICES_FILE: &str = "voices-divora.bin";
const CONFIG_FILE: &str = "kokoro-config.json";
/// ONNX Runtime's CPU build, out of PyPI's onnxruntime 1.26.0 wheel.
const RUNTIME_FILE: &str = "onnxruntime.dll";
const ESPEAK_FILES: [&str; 3] = ["espeak-ng.exe", "libespeak-ng.dll", "espeak-ng-data/phontab"];

/// Kokoro's `style` vector dimension (the model's `style` input is `[1, 256]`).
const STYLE_DIM: usize = 256;

/// The curated preset voices: `(id, display name, lang)`. Display names are
/// DivoraVoice's; the ids are the voice pack's keys (DivoraVoice's
/// `PRESET_VOICES`, verbatim).
pub const PRESET_VOICES: &[(&str, &str, &str)] = &[
    ("af_heart", "Aria — warm (US)", "en-us"),
    ("af_bella", "Bella — bright (US)", "en-us"),
    ("af_aoede", "Aoede — clear (US)", "en-us"),
    ("am_michael", "Michael — deep (US)", "en-us"),
    ("am_puck", "Puck — lively (US)", "en-us"),
    ("bf_emma", "Emma — soft (UK)", "en-gb"),
    ("bm_george", "George — crisp (UK)", "en-gb"),
];

/// Why he can't speak a line.
#[derive(Debug)]
pub enum TtsError {
    /// These files aren't in the folder yet (fetch_voice.py puts them there).
    NotInstalled(Vec<&'static str>),
    /// Nothing in the text to say.
    EmptyText,
    /// Longer than `MAX_TEXT_CHARS`.
    TooLong(usize),
    Phonemize(PhonemizeError),
    /// ONNX Runtime or the model couldn't be loaded.
    Runtime(String),
    /// The model couldn't say it.
    Inference(String),
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled(missing) => write!(
                f,
                "his voice isn't installed (no {}): run desktop/scripts/fetch_voice.py",
                missing.join(", ")
            ),
            Self::EmptyText => write!(f, "nothing to say"),
            Self::TooLong(n) => write!(f, "too long to say at once ({n} characters, at most {MAX_TEXT_CHARS})"),
            Self::Phonemize(e) => write!(f, "couldn't read it aloud: {e}"),
            Self::Runtime(e) => write!(f, "his voice couldn't be loaded: {e}"),
            Self::Inference(e) => write!(f, "his voice failed: {e}"),
        }
    }
}

impl From<PhonemizeError> for TtsError {
    fn from(e: PhonemizeError) -> Self {
        Self::Phonemize(e)
    }
}

/// His voice's folder: `tts` in the data folder (a test's own, in a test).
#[must_use]
pub fn dir() -> PathBuf {
    crate::settings::app_dir().join("tts")
}

/// Where each of his voice's files is, in one folder.
#[derive(Debug, Clone)]
pub struct Assets {
    dir: PathBuf,
}

impl Assets {
    #[must_use]
    pub fn in_dir(dir: &Path) -> Self {
        Self { dir: dir.to_path_buf() }
    }

    /// The files that aren't there, by the name fetch_voice.py gives them.
    /// Files only: nothing is opened or loaded.
    #[must_use]
    pub fn missing(&self) -> Vec<&'static str> {
        [MODEL_FILE, VOICES_FILE, CONFIG_FILE, RUNTIME_FILE]
            .into_iter()
            .chain(ESPEAK_FILES)
            .filter(|name| !self.dir.join(name).is_file())
            .collect()
    }

    #[must_use]
    pub fn installed(&self) -> bool {
        self.missing().is_empty()
    }
}

/// Whether his voice's files are all in his folder.
#[must_use]
pub fn installed() -> bool {
    Assets::in_dir(&dir()).installed()
}

/// The voice for `id`, and the language espeak-ng reads it in: `id` if it is
/// one of the presets, else George.
#[must_use]
pub fn voice(id: &str) -> (&'static str, &'static str) {
    let (id, _, lang) = PRESET_VOICES
        .iter()
        .find(|&&(p, _, _)| p == id)
        .or_else(|| PRESET_VOICES.iter().find(|&&(p, _, _)| p == DEFAULT_VOICE))
        .copied()
        .expect("George is a preset");
    (id, lang)
}

/// A parsed Divora voice pack: voice id → `rows × dim` style matrix (one
/// style row per phoneme-token length, `0..rows`).
pub struct StylePack {
    rows: usize,
    dim: usize,
    voices: HashMap<String, Vec<f32>>,
}

impl StylePack {
    /// Parse the compact `DVTS` format: `"DVTS"` · u32 version · u32 voiceCount
    /// · u32 rows · u32 dim, then per voice: u32 nameLen · name (utf8) ·
    /// `rows·dim` f32 (little-endian). `None` on any structural mismatch.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 || &bytes[0..4] != b"DVTS" {
            return None;
        }
        let u32_at = |off: usize| -> Option<usize> {
            let s = bytes.get(off..off + 4)?;
            Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        };
        let _version = u32_at(4)?;
        let n = u32_at(8)?;
        let rows = u32_at(12)?;
        let dim = u32_at(16)?;
        if rows == 0 || dim == 0 {
            return None;
        }
        let count = rows.checked_mul(dim)?;
        let need = count.checked_mul(4)?;
        let mut off = 20usize;
        let mut voices = HashMap::with_capacity(n);
        for _ in 0..n {
            let name_len = u32_at(off)?;
            off += 4;
            let name = std::str::from_utf8(bytes.get(off..off + name_len)?)
                .ok()?
                .to_string();
            off += name_len;
            let raw = bytes.get(off..off + need)?;
            off += need;
            let (quads, _tail) = raw.as_chunks::<4>();
            let data = quads.iter().copied().map(f32::from_le_bytes).collect();
            voices.insert(name, data);
        }
        Some(Self { rows, dim, voices })
    }

    /// The style vector for `voice_id` at phoneme-token length `token_len`
    /// (clamped to the last row). `None` if the voice isn't in the pack.
    #[must_use]
    pub fn style(&self, voice_id: &str, token_len: usize) -> Option<&[f32]> {
        let v = self.voices.get(voice_id)?;
        let row = token_len.min(self.rows - 1);
        let start = row * self.dim;
        v.get(start..start + self.dim)
    }

    /// Whether the pack contains `voice_id`.
    #[must_use]
    pub fn has_voice(&self, voice_id: &str) -> bool {
        self.voices.contains_key(voice_id)
    }
}

/// Run one framed token sequence through Kokoro → 24 kHz f32 samples.
/// `tokens` is the padded `[0, …ids, 0]` sequence; `style` is a
/// [`STYLE_DIM`]-length vector; `speed` is `1.0` for normal rate. Tensors are
/// a shape and a vector each (DivoraVoice builds them with ndarray).
fn infer(session: &mut Session, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>, String> {
    if style.len() != STYLE_DIM {
        return Err(format!("a style of {} values, not {STYLE_DIM}", style.len()));
    }
    let e = |e: ort::Error| e.to_string();
    let tok = Tensor::from_array(([1usize, tokens.len()], tokens.to_vec())).map_err(e)?;
    let sty = Tensor::from_array(([1usize, style.len()], style.to_vec())).map_err(e)?;
    let spd = Tensor::from_array(([1usize], vec![speed])).map_err(e)?;
    let inputs = ort::inputs!["tokens" => tok, "style" => sty, "speed" => spd];
    let outputs = session.run(inputs).map_err(e)?;
    let (_, value) = outputs.iter().next().ok_or("the model gave no output")?;
    let (_, samples) = value.try_extract_tensor::<f32>().map_err(e)?;
    Ok(samples.to_vec())
}

/// Kokoro, loaded: kept from the first line he speaks.
struct Engine {
    session: Session,
    vocab: HashMap<char, i64>,
    pack: StylePack,
    phonemizer: Phonemizer,
}

/// The one engine, and the one sentence being spoken: a second waits.
static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
/// ONNX Runtime, opened once a run from the data folder (or why it
/// couldn't be). Never opened while its file is missing.
static RUNTIME: OnceLock<Result<(), String>> = OnceLock::new();
/// An unknown voice is said once a run, not every sentence.
static WARNED_VOICE: AtomicBool = AtomicBool::new(false);
/// How many times Kokoro was loaded, for the test that it is kept.
#[cfg(test)]
static LOADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The engine, waiting for the sentence before. A sentence that panicked
/// (in ONNX Runtime, say) leaves no half-used engine: it is loaded anew.
fn engine() -> MutexGuard<'static, Option<Engine>> {
    ENGINE.lock().unwrap_or_else(|poisoned| {
        ENGINE.clear_poison();
        let mut g = poisoned.into_inner();
        *g = None;
        g
    })
}

/// ONNX Runtime from `dll`, by its full path, once `runtime_usable` has
/// said `ort` can load it. Its first answer stands for the run: `ort` can't
/// load a second.
fn runtime(dll: &Path) -> Result<(), TtsError> {
    runtime_in(&RUNTIME, dll)
}

fn runtime_in(cell: &OnceLock<Result<(), String>>, dll: &Path) -> Result<(), TtsError> {
    cell.get_or_init(|| {
        let opened = (|| {
            let version = runtime_usable(dll)?;
            let env = ort::init_from(dll).map_err(|e| e.to_string())?;
            env.commit();
            Ok::<_, String>(version)
        })();
        // Said once, as it is kept for the run: a path and a version,
        // nothing he says.
        match &opened {
            Ok(version) => tracing::info!("voice: ONNX Runtime {version}"),
            Err(why) => tracing::warn!("voice: ONNX Runtime refused: {why}"),
        }
        opened.map(|_| ())
    })
    .clone()
    .map_err(TtsError::Runtime)
}

/// `dll`'s version, if it is an ONNX Runtime `ort` can use, asked here
/// before `ort` is: `ort::init_from` deadlocks on any failure to load or on
/// a version it refuses (it builds its error through the library it couldn't
/// load, re-entering its own once-lock), so a missing, broken or old DLL, or
/// a PC without the Visual C++ runtime, would hang his voice for good.
/// Loaded with the flags `ort`'s own load uses (libloading's, none), so what
/// loads here loads there, and kept loaded when it can be used, so `ort`'s
/// load is of the same library.
#[cfg(windows)]
fn runtime_usable(dll: &Path) -> Result<String, String> {
    use windows::core::{s, HSTRING};
    use windows::Win32::Foundation::FreeLibrary;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_FLAGS};
    // SAFETY: a library loaded by its full path; it is only asked its
    // version and its API's presence, through the entry point every ONNX
    // Runtime exports, with the signature ort-sys gives it.
    unsafe {
        // As libloading loads it for ort: with critical errors failed rather
        // than shown, so a damaged DLL is an error, not a "Bad Image" box
        // waiting for a click.
        let mut previous = 0u32;
        let guarded = SetThreadErrorMode(SEM_FAILCRITICALERRORS, &mut previous) != 0;
        let loaded = LoadLibraryExW(&HSTRING::from(dll.as_os_str()), None, LOAD_LIBRARY_FLAGS(0));
        if guarded {
            SetThreadErrorMode(previous, std::ptr::null_mut());
        }
        let module = loaded.map_err(|e| format!("couldn't load {}: {}", dll.display(), e.message()))?;
        let checked = (|| {
            let entry = GetProcAddress(module, s!("OrtGetApiBase")).ok_or("it isn't ONNX Runtime (no OrtGetApiBase)")?;
            let entry: unsafe extern "system" fn() -> *const ort::sys::OrtApiBase = std::mem::transmute(entry);
            let base = entry();
            if base.is_null() {
                return Err("it isn't ONNX Runtime (no API base)".to_string());
            }
            let version = std::ffi::CStr::from_ptr(((*base).GetVersionString)()).to_string_lossy().into_owned();
            let mut parts = version.split('.').map(|p| p.parse::<u32>().ok());
            let (major, minor) = (parts.next().flatten(), parts.next().flatten());
            if major != Some(1) || minor.is_none_or(|m| m < ort::MINOR_VERSION) {
                return Err(format!("ONNX Runtime {version} won't do: 1.{} or later is needed", ort::MINOR_VERSION));
            }
            if ((*base).GetApi)(ort::MINOR_VERSION).is_null() {
                return Err(format!("ONNX Runtime {version} has no API {}", ort::MINOR_VERSION));
            }
            Ok(version)
        })();
        if checked.is_err() {
            let _ = FreeLibrary(module);
        }
        checked
    }
}

/// SEM_FAILCRITICALERRORS, for SetThreadErrorMode.
#[cfg(windows)]
const SEM_FAILCRITICALERRORS: u32 = 1;

// kernel32's, declared as libloading declares it (the windows crate has it
// only behind a feature of its own).
#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn SetThreadErrorMode(new_mode: u32, old_mode: *mut u32) -> i32;
}

#[cfg(not(windows))]
fn runtime_usable(_dll: &Path) -> Result<String, String> {
    Err("his voice is Windows' only".into())
}

/// The voice pack in `bytes`, which must have George, the voice any other
/// falls back to.
fn pack_from(bytes: &[u8]) -> Result<StylePack, TtsError> {
    let pack = StylePack::parse(bytes).ok_or_else(|| TtsError::Runtime("invalid voice pack".into()))?;
    if !pack.has_voice(DEFAULT_VOICE) {
        return Err(TtsError::Runtime(format!("the voice pack has no {DEFAULT_VOICE}")));
    }
    Ok(pack)
}

/// `voice`'s style for `ids`, a sequence framed with a pad at each end:
/// Kokoro selects the style vector by the *unpadded* token length.
fn style_for<'a>(pack: &'a StylePack, voice: &str, ids: &[i64]) -> Option<&'a [f32]> {
    pack.style(voice, ids.len().saturating_sub(2))
}

/// Load Kokoro from `dir`: its files checked first, then ONNX Runtime, the
/// vocab, the voice pack and the model.
fn load(dir: &Path) -> Result<Engine, TtsError> {
    let assets = Assets::in_dir(dir);
    let missing = assets.missing();
    if !missing.is_empty() {
        return Err(TtsError::NotInstalled(missing));
    }
    runtime(&dir.join(RUNTIME_FILE))?;
    let config = std::fs::read_to_string(dir.join(CONFIG_FILE))
        .map_err(|e| TtsError::Runtime(format!("read kokoro config: {e}")))?;
    let vocab = tts_tokens::parse_vocab(&config).map_err(|e| TtsError::Runtime(format!("parse kokoro vocab: {e}")))?;
    let pack_bytes =
        std::fs::read(dir.join(VOICES_FILE)).map_err(|e| TtsError::Runtime(format!("read voice pack: {e}")))?;
    let pack = pack_from(&pack_bytes)?;
    let e = |e: ort::Error<_>| TtsError::Runtime(e.to_string());
    let session = Session::builder()
        .map_err(|e| TtsError::Runtime(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(e)?
        .with_intra_threads(THREADS)
        .map_err(e)?
        .with_inter_threads(1)
        .map_err(e)?
        .with_intra_op_spinning(false)
        .map_err(e)?
        .with_inter_op_spinning(false)
        .map_err(e)?
        .commit_from_file(dir.join(MODEL_FILE))
        .map_err(|e| TtsError::Runtime(e.to_string()))?;
    #[cfg(test)]
    LOADS.fetch_add(1, Ordering::SeqCst);
    Ok(Engine {
        session,
        vocab,
        pack,
        phonemizer: Phonemizer::from_dir(dir),
    })
}

/// The engine, loaded from `dir` if it isn't yet.
fn loaded<'a>(g: &'a mut MutexGuard<'static, Option<Engine>>, dir: &Path) -> Result<&'a mut Engine, TtsError> {
    if g.is_none() {
        let t = Instant::now();
        **g = Some(load(dir)?);
        tracing::info!(ms = t.elapsed().as_millis() as u64, "voice: Kokoro loaded");
    }
    Ok(g.as_mut().expect("just loaded"))
}

/// A warm on its way, so a run of presses starts one.
static WARMING: AtomicBool = AtomicBool::new(false);
/// How many warms were started, for the test that a run starts one.
#[cfg(test)]
static WARMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Load his voice now, if it isn't loaded, so the first sentence doesn't
/// wait for it.
pub fn warm() -> Result<(), TtsError> {
    loaded(&mut engine(), &dir()).map(|_| ())
}

/// `warm`, on a thread of its own, unless one is on its way: push-to-talk
/// asks it as the owner starts talking, which leaves the time he talks and
/// the brain thinks for Kokoro's load. A sentence asked meanwhile waits for
/// the load rather than loading it again.
pub fn warm_in_background() {
    if WARMING.swap(true, Ordering::SeqCst) {
        return;
    }
    let started = std::thread::Builder::new().name("voice-warm".into()).spawn(|| {
        #[cfg(test)]
        WARMS.fetch_add(1, Ordering::SeqCst);
        if let Err(e) = warm() {
            tracing::debug!("voice: not warmed: {e}");
        }
        WARMING.store(false, Ordering::SeqCst);
    });
    if let Err(e) = started {
        WARMING.store(false, Ordering::SeqCst);
        tracing::warn!("voice: couldn't warm: {e}");
    }
}

/// `text` in voice `voice_id` (George if it isn't one), as 24 kHz mono f32
/// samples: each of its sentences in turn, back to back.
pub fn synthesize(text: &str, voice_id: &str) -> Result<Vec<f32>, TtsError> {
    synthesize_in(&dir(), text, voice_id)
}

fn synthesize_in(dir: &Path, text: &str, voice_id: &str) -> Result<Vec<f32>, TtsError> {
    let chars = text.chars().count();
    if chars > MAX_TEXT_CHARS {
        return Err(TtsError::TooLong(chars));
    }
    let chunks = chunks(text);
    if chunks.is_empty() {
        return Err(TtsError::EmptyText);
    }
    let (id, lang) = voice(voice_id);
    if id != voice_id && !WARNED_VOICE.swap(true, Ordering::Relaxed) {
        tracing::warn!("voice: {:?} isn't one of his voices; {id} speaks", crate::printable(voice_id, 40));
    }
    let t = Instant::now();
    let mut g = engine();
    let engine = loaded(&mut g, dir)?;
    // The pack is DivoraVoice's seven; one it lacks is George's too.
    let id = if engine.pack.has_voice(id) { id } else { DEFAULT_VOICE };
    let mut samples: Vec<f32> = Vec::new();
    for chunk in chunks {
        render(engine, &chunk, lang, id, &mut samples)?;
    }
    if samples.is_empty() {
        return Err(TtsError::EmptyText);
    }
    tracing::info!(
        chars,
        ms = t.elapsed().as_millis() as u64,
        audio_ms = samples.len() as u64 * 1000 / u64::from(SAMPLE_RATE),
        "voice: spoke"
    );
    Ok(samples)
}

/// The text in the pieces Kokoro is given: the whole of it when it fits,
/// which a sentence from the brain does. DivoraVoice's `split_text` cuts at
/// every stop, and reads "3.50" as "3." and "50", or "e.g. here" as two
/// sentences; so it is only for a text too long to say at once, which is
/// then cut into sentences and words as DivoraVoice cuts it.
/// A piece with no letter or digit in it is nothing to say: espeak-ng reads
/// a lone "!" as "exclamation", a "*" as "asterisk".
fn chunks(text: &str) -> Vec<String> {
    let text = text.trim();
    let pieces = if text.chars().count() <= CHUNK_CHARS {
        vec![text.to_string()]
    } else {
        tts_tokens::split_text(text, CHUNK_CHARS)
    };
    pieces.into_iter().filter(|p| p.chars().any(char::is_alphanumeric)).collect()
}

/// `chunk` said into `out`. Kokoro takes at most 510 phonemes and drops the
/// rest, and a chunk's phonemes can outnumber its characters (numbers,
/// initials), so one that phonemizes longer is halved at the space nearest
/// its middle, each half said in turn; only a single word too long for
/// Kokoro (there is none in English) is cut as DivoraVoice cuts it.
fn render(engine: &mut Engine, chunk: &str, lang: &str, voice: &str, out: &mut Vec<f32>) -> Result<(), TtsError> {
    let phonemes = engine.phonemizer.phonemize(chunk, lang)?;
    if phonemes.chars().count() > tts_tokens::MAX_PHONEME_LENGTH {
        if let Some((first, rest)) = halves(chunk) {
            render(engine, first, lang, voice, out)?;
            return render(engine, rest, lang, voice, out);
        }
    }
    let ids = tts_tokens::phonemes_to_ids(&phonemes, &engine.vocab);
    // Nothing the model knows (a chunk of punctuation): nothing to say.
    if ids.len() <= 2 {
        return Ok(());
    }
    let style = style_for(&engine.pack, voice, &ids).ok_or_else(|| TtsError::Inference("style lookup failed".into()))?;
    out.extend(infer(&mut engine.session, &ids, style, 1.0).map_err(TtsError::Inference)?);
    Ok(())
}

/// `text` cut at the space nearest its middle (by characters), both halves
/// trimmed and neither empty; `None` when it has no such space.
fn halves(text: &str) -> Option<(&str, &str)> {
    let middle = text.chars().count() / 2;
    text.char_indices()
        .enumerate()
        .filter(|&(_, (_, c))| c.is_whitespace())
        .min_by_key(|&(n, _)| n.abs_diff(middle))
        .map(|(_, (at, c))| (text[..at].trim(), text[at + c.len_utf8()..].trim()))
        .filter(|(a, b)| !a.is_empty() && !b.is_empty())
}

/// Samples as the page takes them: f32, little-endian, back to back (a
/// Float32Array over the ArrayBuffer `ipc::Response` gives it).
#[must_use]
pub fn le_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic pack: 1 voice "x", rows=2, dim=3.
    fn synth_pack() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"DVTS");
        b.extend_from_slice(&1u32.to_le_bytes()); // version
        b.extend_from_slice(&1u32.to_le_bytes()); // voice count
        b.extend_from_slice(&2u32.to_le_bytes()); // rows
        b.extend_from_slice(&3u32.to_le_bytes()); // dim
        b.extend_from_slice(&1u32.to_le_bytes()); // name_len
        b.extend_from_slice(b"x");
        for f in [0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0] {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b
    }

    #[test]
    fn parse_indexes_rows_and_clamps() {
        let p = StylePack::parse(&synth_pack()).unwrap();
        assert!(p.has_voice("x"));
        assert_eq!(p.style("x", 0).unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(p.style("x", 1).unwrap(), &[10.0, 11.0, 12.0]);
        // Beyond the last row clamps to it.
        assert_eq!(p.style("x", 99).unwrap(), &[10.0, 11.0, 12.0]);
        assert!(p.style("missing", 0).is_none());
    }

    /// A pack of `voices`, `rows` rows of `dim` values, each value its row
    /// times ten plus its column.
    fn pack_of(voices: &[&str], rows: u32, dim: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"DVTS");
        for n in [1, voices.len() as u32, rows, dim] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        for v in voices {
            b.extend_from_slice(&(v.len() as u32).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
            for r in 0..rows {
                for c in 0..dim {
                    b.extend_from_slice(&((r * 10 + c) as f32).to_le_bytes());
                }
            }
        }
        b
    }

    #[test]
    fn a_pack_must_have_george() {
        let p = pack_from(&pack_of(&["af_heart", "bm_george"], 2, 3)).expect("George is in it");
        assert!(p.has_voice("af_heart") && p.has_voice("bm_george"));
        match pack_from(&pack_of(&["af_heart"], 2, 3)) {
            Err(TtsError::Runtime(e)) => assert_eq!(e, "the voice pack has no bm_george"),
            other => panic!("{:?}", other.map(|_| ())),
        }
        assert!(matches!(pack_from(b"DVTS"), Err(TtsError::Runtime(e)) if e == "invalid voice pack"));
    }

    #[test]
    fn the_style_is_picked_by_the_unpadded_length() {
        let p = pack_from(&pack_of(&["bm_george"], 4, 1)).unwrap();
        // One token between the pads: row 1.
        assert_eq!(style_for(&p, "bm_george", &[0, 5, 0]), Some(&[10.0][..]));
        assert_eq!(style_for(&p, "bm_george", &[0, 5, 6, 0]), Some(&[20.0][..]));
        // Just the pads: row 0; past the last row: the last.
        assert_eq!(style_for(&p, "bm_george", &[0, 0]), Some(&[0.0][..]));
        assert_eq!(style_for(&p, "bm_george", &[0, 1, 2, 3, 4, 5, 6, 0]), Some(&[30.0][..]));
        assert_eq!(style_for(&p, "bm_george", &[]), Some(&[0.0][..]));
        assert_eq!(style_for(&p, "af_heart", &[0, 5, 0]), None);
    }

    #[test]
    fn a_sentence_that_fits_is_said_whole() {
        for s in [
            "It costs 3.50 pounds, e.g. at the shop.",
            "Version 2.0 is out! Shall I install it?",
            "Well\u{2026} I suppose so.",
        ] {
            assert_eq!(chunks(s), [s], "{s}");
        }
        assert_eq!(chunks("  Good evening, sir.\n"), ["Good evening, sir."]);
        assert!(chunks(" \n ").is_empty());
        // Nothing with no letter or digit in it: espeak-ng would name it.
        for nothing in ["... !?", "!", "*", "#", "\u{2014}", "( )"] {
            assert!(chunks(nothing).is_empty(), "{nothing:?}");
        }
        for something in ["Hm...", "100%", "\u{3b1}!", "\u{4f60}\u{597d}"] {
            assert_eq!(chunks(something), [something], "{something:?}");
        }
        // A long text's pieces of punctuation alone are left out too.
        let long = format!("{} !!! {}", "word ".repeat(70), "word ".repeat(70));
        assert!(chunks(&long).iter().all(|p| p.chars().any(char::is_alphanumeric)));
        let fits = "a".repeat(CHUNK_CHARS);
        assert_eq!(chunks(&fits), [fits.as_str()]);
    }

    #[test]
    fn a_text_too_long_is_cut_into_sentences_and_words() {
        let sentence = "This is one of many sentences in a long reply, sir. ";
        let long = sentence.repeat(12);
        let pieces = chunks(&long);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|p| p.chars().count() <= CHUNK_CHARS), "{pieces:?}");
        let words = |s: &str| s.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(words(&pieces.join(" ")), words(&long), "nothing lost");
    }

    #[test]
    fn halves_cut_at_the_space_nearest_the_middle() {
        assert_eq!(halves("one two three four"), Some(("one two", "three four")));
        assert_eq!(halves("a b"), Some(("a", "b")));
        assert_eq!(halves("caf\u{e9} au lait, s'il vous pla\u{ee}t"), Some(("caf\u{e9} au lait,", "s'il vous pla\u{ee}t")));
        for none in ["word", "", " ", " x", "x "] {
            assert_eq!(halves(none), None, "{none:?}");
        }
    }

    #[test]
    fn parse_rejects_bad_input() {
        assert!(StylePack::parse(b"XXXXxxxxxxxxxxxxxxxx").is_none());
        assert!(StylePack::parse(b"DV").is_none());
        assert!(StylePack::parse(&[]).is_none());
        // A pack cut short, or with no rows, is no pack.
        let p = synth_pack();
        assert!(StylePack::parse(&p[..p.len() - 1]).is_none());
        let mut no_rows = p.clone();
        no_rows[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert!(StylePack::parse(&no_rows).is_none());
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fritz-tts-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn every_file_is_needed_and_named_as_the_fetcher_names_it() {
        let d = scratch("assets");
        let a = Assets::in_dir(&d);
        let all = [
            "kokoro-v1.0.int8.onnx",
            "voices-divora.bin",
            "kokoro-config.json",
            "onnxruntime.dll",
            "espeak-ng.exe",
            "libespeak-ng.dll",
            "espeak-ng-data/phontab",
        ];
        assert_eq!(a.missing(), all);
        std::fs::create_dir_all(d.join("espeak-ng-data")).unwrap();
        for (i, name) in all.iter().enumerate() {
            assert!(!a.installed(), "{name} and after are missing");
            std::fs::write(d.join(name), b"x").unwrap();
            assert_eq!(a.missing(), all[i + 1..]);
        }
        assert!(a.installed());
        // A folder by a file's name is not the file.
        std::fs::remove_file(d.join("onnxruntime.dll")).unwrap();
        std::fs::create_dir(d.join("onnxruntime.dll")).unwrap();
        assert_eq!(a.missing(), ["onnxruntime.dll"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The fetcher's list is the shell's: every file the shell needs is one
    /// fetch_voice.py fetches or unpacks.
    #[test]
    fn the_fetcher_fetches_every_file_the_shell_needs() {
        let fetcher = include_str!("../../scripts/fetch_voice.py");
        for name in [MODEL_FILE, VOICES_FILE, CONFIG_FILE, RUNTIME_FILE, "espeak-ng.exe", "libespeak-ng.dll"] {
            assert!(fetcher.contains(&format!("\"{name}\"")), "{name}");
        }
        assert!(fetcher.contains("\"espeak-ng-data.zip\"") && fetcher.contains("phontab"));
    }

    #[test]
    fn missing_files_say_so_before_anything_is_loaded() {
        let d = scratch("missing");
        match synthesize_in(&d, "Good evening, sir.", DEFAULT_VOICE) {
            Err(TtsError::NotInstalled(missing)) => assert_eq!(missing.len(), 7),
            other => panic!("expected NotInstalled, got {other:?}"),
        }
        assert!(RUNTIME.get().is_none(), "ONNX Runtime isn't opened for a folder without it");
        assert!(engine().is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A warm with no voice to load says so and is done, so the next press
    /// can warm again; a run of them starts one at a time.
    #[test]
    fn a_warm_without_his_voice_ends_and_can_come_again() {
        assert!(matches!(warm(), Err(TtsError::NotInstalled(_))), "a test's own folder has no voice");
        let ended = || {
            let deadline = Instant::now() + std::time::Duration::from_secs(10);
            while WARMING.load(Ordering::SeqCst) {
                assert!(Instant::now() < deadline, "the warm never ended");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        };
        // One on its way: a press now starts none.
        ended();
        let before = WARMS.load(Ordering::SeqCst);
        WARMING.store(true, Ordering::SeqCst);
        warm_in_background();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(WARMS.load(Ordering::SeqCst), before, "none while one is on its way");
        WARMING.store(false, Ordering::SeqCst);
        warm_in_background();
        ended();
        assert_eq!(WARMS.load(Ordering::SeqCst), before + 1);
        assert!(engine().is_none());
        assert!(RUNTIME.get().is_none(), "nothing to load: ONNX Runtime isn't opened");
    }

    #[test]
    fn text_is_checked_before_the_engine() {
        let d = scratch("text");
        assert!(matches!(synthesize_in(&d, "   \n ", DEFAULT_VOICE), Err(TtsError::EmptyText)));
        let long = "a".repeat(MAX_TEXT_CHARS + 1);
        assert!(matches!(synthesize_in(&d, &long, DEFAULT_VOICE), Err(TtsError::TooLong(n)) if n == MAX_TEXT_CHARS + 1));
        // At the limit, it goes on (to the missing files here).
        let most = "a".repeat(MAX_TEXT_CHARS);
        assert!(matches!(synthesize_in(&d, &most, DEFAULT_VOICE), Err(TtsError::NotInstalled(_))));
        // Characters, not bytes.
        let wide = "é".repeat(MAX_TEXT_CHARS);
        assert!(matches!(synthesize_in(&d, &wide, DEFAULT_VOICE), Err(TtsError::NotInstalled(_))));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_unknown_voice_is_george() {
        assert_eq!(voice("bm_george"), ("bm_george", "en-gb"));
        assert_eq!(voice("af_heart"), ("af_heart", "en-us"));
        assert_eq!(voice("bf_emma"), ("bf_emma", "en-gb"));
        for unknown in ["", "first", "BM_GEORGE", "bm_george ", "../x"] {
            assert_eq!(voice(unknown), ("bm_george", "en-gb"), "{unknown:?}");
        }
        assert!(PRESET_VOICES.iter().any(|&(id, _, _)| id == DEFAULT_VOICE));
    }

    /// Settings' default voice is his default here.
    #[test]
    fn settings_start_with_his_default_voice() {
        assert_eq!(crate::settings::Settings::default().audio.tts_voice, DEFAULT_VOICE);
    }

    #[test]
    fn samples_go_to_the_page_as_little_endian_f32() {
        let bytes = le_bytes(&[0.0, 1.0, -0.5]);
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[4..8], &1.0f32.to_le_bytes());
        assert_eq!(f32::from_le_bytes(bytes[8..12].try_into().unwrap()), -0.5);
        assert!(le_bytes(&[]).is_empty());
    }

    #[test]
    fn the_errors_say_what_to_do() {
        let e = TtsError::NotInstalled(vec!["onnxruntime.dll", "espeak-ng.exe"]);
        assert_eq!(
            e.to_string(),
            "his voice isn't installed (no onnxruntime.dll, espeak-ng.exe): run desktop/scripts/fetch_voice.py"
        );
        assert!(TtsError::TooLong(2001).to_string().contains("2001"));
    }

    /// ONNX Runtime is opened in one place only, by its full path from the
    /// data folder: never by name (which would find Windows' own, too old,
    /// and panic), and never before the files are checked.
    #[test]
    fn onnx_runtime_is_opened_only_by_its_full_path() {
        let all = include_str!("kokoro_tts.rs").replace('\r', "");
        let src = &all[..all.find(concat!("#[cfg(test)]\nmod ", "tests {")).unwrap()];
        assert_eq!(src.matches(concat!("ort::init", "_from(")).count(), 1);
        assert!(src.contains(concat!("runtime(&dir.join(", "RUNTIME_FILE))?;")));
        assert!(src.contains(concat!("let env = ort::init", "_from(dll)")), "the path it is given, nothing else");
        // And only once runtime_usable has said ort can load it: ort hangs on
        // a load that fails.
        let init = src.find(concat!("ort::init", "_from(dll)")).unwrap();
        let usable = src.find(concat!("let version = runtime", "_usable(dll)?;")).unwrap();
        assert!(usable < init);
        assert_eq!(src.matches(concat!("runtime", "_usable(dll)?")).count(), 1);
        assert!(!src.contains(concat!("ORT_DYLIB", "_PATH")), "no load by an environment variable");
        assert!(!src.contains(concat!("ort::", "init()")));
        // In load, the files come first, then the runtime, then the session.
        let load = &src[src.find(concat!("fn ", "load(")).unwrap()..];
        let files = load.find(concat!("assets.", "missing()")).unwrap();
        let rt = load.find(concat!("runtime(", "&dir")).unwrap();
        let session = load.find(concat!("Session::", "builder()")).unwrap();
        assert!(files < rt && rt < session);
        assert_eq!(src.matches(concat!("Session::", "builder()")).count(), 1);
    }

    /// `f`'s answer, and what it logged on this thread.
    fn logged<T>(f: impl FnOnce() -> T) -> (T, String) {
        use std::sync::Arc;
        #[derive(Clone)]
        struct Buf(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Buf(Arc::new(Mutex::new(Vec::new())));
        let w = buf.clone();
        let subscriber = tracing_subscriber::fmt().with_writer(move || w.clone()).finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8_lossy(&buf.0.lock().unwrap()).into_owned();
        (out, text)
    }

    /// A runtime `ort` couldn't load is refused at once and never handed to
    /// it (its init_from hangs on one). Each is tried on a thread with a
    /// cell of its own, so a hang fails this test rather than the run, and
    /// the run's own runtime is left alone.
    #[test]
    fn a_runtime_that_cant_be_used_is_refused_without_hanging() {
        let d = scratch("badrt");
        std::fs::write(d.join("fake.dll"), b"not a dll at all").unwrap();
        let mut cases = vec![
            (d.join("missing.dll"), "couldn't load"),
            (d.join("fake.dll"), "couldn't load"),
            (PathBuf::from(r"C:\Windows\System32\kernel32.dll"), "isn't ONNX Runtime"),
        ];
        // Windows' own, where it has one (1.17 on the owner's PC), is too old.
        let inbox = PathBuf::from(r"C:\Windows\System32\onnxruntime.dll");
        if inbox.is_file() {
            cases.push((inbox, "won't do: 1.24 or later is needed"));
        }
        for (dll, want) in cases {
            let (tx, rx) = std::sync::mpsc::channel();
            let p = dll.clone();
            std::thread::spawn(move || {
                // Twice on one cell, as every sentence asks: the same
                // answer, and said once.
                let cell = OnceLock::new();
                let _ = tx.send(logged(|| {
                    let first = runtime_in(&cell, &p).map_err(|e| e.to_string());
                    let again = runtime_in(&cell, &p).map_err(|e| e.to_string());
                    assert_eq!(first, again);
                    first
                }));
            });
            let (got, log) = rx
                .recv_timeout(std::time::Duration::from_secs(20))
                .unwrap_or_else(|_| panic!("{dll:?}: hung"));
            let err = got.expect_err("refused");
            assert!(err.contains(want), "{dll:?}: {err}");
            // And the log says why, once.
            assert_eq!(log.matches("voice: ONNX Runtime refused:").count(), 1, "{log}");
            assert!(log.contains(want), "{log}");
        }
        assert_eq!(ort::MINOR_VERSION, 24, "the api-24 feature");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Kokoro's session is built with four threads, none spinning (the
    /// owner games on this PC), and the graph optimised: read from the
    /// source, since only the real model can run it.
    #[test]
    fn kokoro_takes_four_threads_and_none_spin() {
        assert_eq!(THREADS, 4);
        let all = include_str!("kokoro_tts.rs").replace('\r', "");
        let src = &all[..all.find(concat!("#[cfg(test)]\nmod ", "tests {")).unwrap()];
        let build = &src[src.find(concat!("Session::", "builder()")).unwrap()..];
        let build = &build[..build.find(concat!(".commit_", "from_file(")).unwrap()];
        for want in [
            concat!(".with_optimization_level(", "GraphOptimizationLevel::Level3)"),
            concat!(".with_intra_threads(", "THREADS)"),
            concat!(".with_inter_threads(", "1)"),
            concat!(".with_intra_op_spinning(", "false)"),
            concat!(".with_inter_op_spinning(", "false)"),
        ] {
            assert_eq!(build.matches(want).count(), 1, "{want}");
        }
    }

    /// A sentence that panicked leaves the engine unloaded and the lock
    /// whole, so the next sentence loads it anew rather than failing.
    #[test]
    fn a_panic_leaves_no_poisoned_engine() {
        let _ = std::thread::spawn(|| {
            let _g = ENGINE.lock();
            panic!("a sentence panicked");
        })
        .join();
        assert!(engine().is_none());
        assert!(!ENGINE.is_poisoned(), "cleared, so it isn't reloaded at every sentence");
    }

    /// The real thing: the fetched files, ONNX Runtime and Kokoro, a sentence
    /// spoken. Needs the files (`scripts/fetch_voice.py`), so it's
    /// `#[ignore]`d: run with `FRITZ_TTS_TEST_DIR` set to that folder and
    /// `-- --ignored`.
    #[test]
    #[ignore = "needs the fetched voice (FRITZ_TTS_TEST_DIR)"]
    fn he_speaks() {
        let dir = PathBuf::from(std::env::var_os("FRITZ_TTS_TEST_DIR").expect("FRITZ_TTS_TEST_DIR"));
        assert_eq!(runtime_usable(&dir.join(RUNTIME_FILE)).as_deref(), Ok("1.26.0"), "the fetched one will do");
        let t = Instant::now();
        let first = synthesize_in(&dir, "Good evening, sir.", DEFAULT_VOICE).expect("he speaks");
        let cold = t.elapsed();
        let t = Instant::now();
        let second = synthesize_in(&dir, "It is half past nine \u{2014} shall I draw the curtains?", "first")
            .expect("in George's voice");
        let warm = t.elapsed();
        for s in [&first, &second] {
            assert!(s.len() > SAMPLE_RATE as usize / 2, "at least half a second: {}", s.len());
            assert!(s.iter().all(|x| x.is_finite() && x.abs() <= 4.0));
            let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
            assert!(rms > 0.005, "not silence: {rms}");
        }
        eprintln!(
            "cold {cold:?} for {:.2} s, warm {warm:?} for {:.2} s",
            first.len() as f32 / SAMPLE_RATE as f32,
            second.len() as f32 / SAMPLE_RATE as f32
        );
        // Timing can't show it (a longer second sentence can take longer
        // than a short first one with the load), so count the loads.
        assert_eq!(LOADS.load(Ordering::SeqCst), 1, "loaded once, then kept");
        // A sentence that panics leaves no engine, and the next loads anew.
        let _ = std::thread::spawn(|| {
            let _g = ENGINE.lock();
            panic!("a sentence panicked");
        })
        .join();
        assert!(engine().is_none(), "a panicked sentence leaves no engine behind");
        synthesize_in(&dir, "Very good, sir.", DEFAULT_VOICE).expect("loaded anew");
        assert_eq!(LOADS.load(Ordering::SeqCst), 2);
        for nothing in ["...", "... !?", "*"] {
            assert!(matches!(synthesize_in(&dir, nothing, DEFAULT_VOICE), Err(TtsError::EmptyText)), "{nothing}");
        }
        // A sentence whose phonemes outnumber Kokoro's 510 is said in full,
        // in halves, not cut short: as long as its two halves said apart.
        let numbers: Vec<String> = (1..=12).map(|i| (1_000_000 + i * 12_345).to_string()).collect();
        let all = synthesize_in(&dir, &numbers.join(" "), DEFAULT_VOICE).unwrap();
        let first = synthesize_in(&dir, &numbers[..6].join(" "), DEFAULT_VOICE).unwrap();
        let rest = synthesize_in(&dir, &numbers[6..].join(" "), DEFAULT_VOICE).unwrap();
        let (all, apart) = (all.len() as f32, (first.len() + rest.len()) as f32);
        assert!((all / apart - 1.0).abs() < 0.25, "{all} samples whole, {apart} apart");
    }
}
