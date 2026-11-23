//! His ears (plan 14, PR 8): what the microphone hears while push-to-talk is
//! held, as 16 kHz mono PCM16, which is what the protocol's `audio_end`
//! describes and the brain's Whisper reads.
//!
//! The microphone is opened through WASAPI, in shared mode, on a thread of
//! its own when push-to-talk is pressed, and closed when it is released, at
//! 30 seconds, or on an error: Windows shows it in use only while the key is
//! held. Windows converts whatever the device gives to 16 kHz mono PCM16
//! itself (`AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM`: the audio engine's own
//! resampler and channel matrix), so nothing here resamples; the owner chose
//! this over cpal and rubato on 2026-11-20, so that no crate beyond
//! `windows` is needed.
//!
//! `Ears` hands a clip to `on_done` and keeps it nowhere itself. A clip
//! for the brain waits in memory for his page to take it (`Handoff`: one
//! clip, up to `HANDOFF_FOR`, gone once taken); the one thing here that
//! writes a clip to disk is `save_clip`, the debug-only `FRITZ_SAVE_CLIPS`
//! saver. The capture loop
//! (`gather`) is generic over a `Source`, so its rules are tested with a
//! fake; `Mic` is the real one, which only a live run reaches.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Samples a second: the protocol's `audio_end.sample_rate` default.
pub const RATE: u32 = 16_000;
/// One frame: a mono 16-bit sample.
const FRAME_BYTES: usize = 2;
pub const BYTES_PER_SECOND: usize = RATE as usize * FRAME_BYTES;
/// The longest clip (plan 14, PR 8): it ends there, the key still held.
pub const MAX_SECONDS: usize = 30;
pub const MAX_BYTES: usize = BYTES_PER_SECOND * MAX_SECONDS;
/// Shorter than this is a slip of the key, not something said.
pub const MIN_MILLIS: u64 = 300;
/// How often the level is given while he listens: 20 times a second.
pub const LEVEL_EVERY: Duration = Duration::from_millis(50);
/// The longest wait for the microphone between looks at the stop flag, so a
/// release ends the clip within this.
const WAIT: Duration = Duration::from_millis(20);
/// A held key older than this, with nothing recording, is a release that
/// never came: a press starts anew.
pub const STALE: Duration = Duration::from_secs(MAX_SECONDS as u64 + 5);
/// The debug-only folder each clip is saved to, for a live check.
pub const SAVE_CLIPS_ENV: &str = "FRITZ_SAVE_CLIPS";

/// Why a clip ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    /// Let go before the microphone started: it heard nothing.
    NeverStarted,
    /// Push-to-talk was let go.
    Released,
    /// It reached `MAX_SECONDS`.
    Full,
}

/// A clip: PCM16 little-endian, 16 kHz, mono.
#[derive(Debug)]
pub struct Captured {
    pub pcm: Vec<u8>,
    pub ended: Ended,
    /// The loudest level given while it was gathered, 0 to 1.
    pub peak: f32,
}

impl Captured {
    /// No clip: let go before the microphone started.
    pub fn nothing() -> Self {
        Self { pcm: Vec::new(), ended: Ended::NeverStarted, peak: 0.0 }
    }

    pub fn millis(&self) -> u64 {
        (self.pcm.len() / FRAME_BYTES) as u64 * 1000 / u64::from(RATE)
    }

    /// Long enough to be something said, not a slip of the key.
    pub fn long_enough(&self) -> bool {
        self.millis() >= MIN_MILLIS
    }
}

/// What a source hands over: sound, as PCM16 little-endian bytes, or a run
/// of silent frames (WASAPI's SILENT flag: the buffer is to be read as
/// zeros, whatever it holds).
#[derive(Debug, Clone, Copy)]
pub enum Packet<'a> {
    Sound(&'a [u8]),
    Silence(usize),
}

/// The clip so far: whole frames only, and never more than `MAX_BYTES`.
#[derive(Debug, Default)]
pub struct Clip {
    pcm: Vec<u8>,
}

impl Clip {
    /// Take a packet, as much of it as fits; `true` once the clip is full.
    pub fn take(&mut self, p: Packet<'_>) -> bool {
        let room = MAX_BYTES - self.pcm.len();
        match p {
            Packet::Sound(data) => {
                let whole = data.len() - data.len() % FRAME_BYTES;
                self.pcm.extend_from_slice(&data[..whole.min(room)]);
            }
            Packet::Silence(frames) => {
                let n = frames.saturating_mul(FRAME_BYTES).min(room);
                self.pcm.resize(self.pcm.len() + n, 0);
            }
        }
        self.pcm.len() >= MAX_BYTES
    }

    pub fn bytes(&self) -> &[u8] {
        &self.pcm
    }
}

/// The RMS of PCM16 little-endian samples, 0 to 1; 0 for none.
pub fn rms(pcm: &[u8]) -> f32 {
    let mut sum = 0f64;
    let mut n = 0u32;
    for s in pcm.as_chunks::<FRAME_BYTES>().0 {
        let v = f64::from(i16::from_le_bytes(*s)) / 32768.0;
        sum += v * v;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    ((sum / f64::from(n)).sqrt() as f32).min(1.0)
}

/// Where the sound comes from: the microphone, or a test's fake.
pub(crate) trait Source {
    /// Start listening: nothing is heard before this. `gather` calls it
    /// once, and only while the key is still held.
    fn start(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Wait up to `timeout` for sound to be ready (it may not be).
    fn wait(&mut self, timeout: Duration) -> Result<(), String>;
    /// Hand every packet ready now to `take`, until it returns `false`.
    fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String>;
}

/// Gather a clip from `source` until `stop` is set or the clip is full,
/// giving the level of what came since the last one every `LEVEL_EVERY`.
/// A source let go of before this is never started, so nothing after the
/// release is heard; after that, the stop flag is looked at after each
/// drain, so what came before the release is in the clip.
pub(crate) fn gather(
    source: &mut impl Source,
    stop: &AtomicBool,
    now: &dyn Fn() -> Instant,
    level: &mut dyn FnMut(f32),
) -> Result<Captured, String> {
    if stop.load(Ordering::SeqCst) {
        return Ok(Captured::nothing());
    }
    source.start()?;
    let mut clip = Clip::default();
    let mut peak = 0f32;
    let mut since = 0;
    let mut last = now();
    let ended = loop {
        source.wait(WAIT)?;
        let mut full = false;
        source.drain(&mut |p| {
            full = clip.take(p);
            !full
        })?;
        if now().duration_since(last) >= LEVEL_EVERY {
            let l = rms(&clip.bytes()[since..]);
            peak = peak.max(l);
            level(l);
            since = clip.bytes().len();
            last = now();
        }
        if full {
            break Ended::Full;
        }
        if stop.load(Ordering::SeqCst) {
            break Ended::Released;
        }
    };
    peak = peak.max(rms(&clip.bytes()[since..]));
    Ok(Captured { pcm: clip.pcm, ended, peak })
}

/// The owner's words for a microphone that couldn't be used. `code` is the
/// HRESULT; `doing` says what was being done, for one Windows gave no
/// reason we know.
pub fn trouble(code: i32, doing: &str) -> String {
    match code as u32 {
        // E_NOTFOUND (HRESULT_FROM_WIN32(ERROR_NOT_FOUND)): no capture device.
        0x8007_0490 => "There's no microphone for him to listen with.".to_owned(),
        // E_ACCESSDENIED: the privacy switch for desktop apps is off.
        0x8007_0005 => "Windows isn't letting desktop apps use the microphone (Settings › Privacy & security › \
                        Microphone)."
            .to_owned(),
        // AUDCLNT_E_DEVICE_INVALIDATED: unplugged, or disabled, while open.
        0x8889_0004 => "The microphone went away while he was listening.".to_owned(),
        // AUDCLNT_E_SERVICE_NOT_RUNNING.
        0x8889_0010 => "Windows' audio service isn't running.".to_owned(),
        _ => format!("The microphone couldn't be used ({doing}: 0x{:08X}).", code as u32),
    }
}

/// A WAV file of a clip: the 44-byte header, then the samples.
pub fn wav(pcm: &[u8]) -> Vec<u8> {
    let data = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data)).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&RATE.to_le_bytes());
    out.extend_from_slice(&(BYTES_PER_SECOND as u32).to_le_bytes());
    out.extend_from_slice(&(FRAME_BYTES as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// The folder `FRITZ_SAVE_CLIPS` names, in a debug build only (`debug` is
/// `cfg!(debug_assertions)` at the one caller): a release build never reads
/// it, so nothing it hears is ever written to disk.
pub fn save_clips_dir(debug: bool, env: Option<OsString>) -> Option<PathBuf> {
    if !debug {
        return None;
    }
    env.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// What a push-to-talk edge does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edge {
    /// Let go: the clip, if any, ends.
    LetGo,
    /// Pressed, but nothing would hear him: the microphone stays closed,
    /// and this says why.
    StayClosed(&'static str),
    /// Pressed, in a debug build given FRITZ_SAVE_CLIPS: listen, and save
    /// the clip in this folder rather than send it.
    Save(PathBuf),
    /// Pressed, and the brain can hear: listen, and send the clip.
    Send,
}

/// What an edge does, given the folder clips are saved to (`None`: none,
/// which is every release build) and whether the brain would hear a clip
/// (`Err` says why not; `companion::ptt_refusal`). The debug folder comes
/// first, so a live check of his ears needs no brain; otherwise the
/// microphone is opened only for a brain that can hear.
pub fn on_edge(pressed: bool, save_to: Option<PathBuf>, heard: Result<(), &'static str>) -> Edge {
    if !pressed {
        return Edge::LetGo;
    }
    if let Some(dir) = save_to {
        return Edge::Save(dir);
    }
    match heard {
        Ok(()) => Edge::Send,
        Err(why) => Edge::StayClosed(why),
    }
}

/// How long a clip waits for his page to take it.
pub const HANDOFF_FOR: Duration = Duration::from_secs(10);

/// The one clip waiting for his page (`ptt_take`), with its number: a newer
/// clip replaces it, and one not taken within `HANDOFF_FOR` is dropped.
/// Taken, it is gone from here.
pub(crate) struct Handoff {
    waiting: Mutex<Option<(u64, Vec<u8>, Instant)>>,
    next: AtomicU64,
}

impl Handoff {
    pub(crate) const fn new() -> Self {
        Self { waiting: Mutex::new(None), next: AtomicU64::new(0) }
    }

    /// Hold `pcm` for the page, at `now`: its number, and whether a clip
    /// was waiting still (it is dropped).
    pub(crate) fn hold(&self, pcm: Vec<u8>, now: Instant) -> (u64, bool) {
        let id = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        let was = self.waiting.lock().unwrap_or_else(|e| e.into_inner()).replace((id, pcm, now));
        (id, was.is_some())
    }

    /// Clip `id`, if it is the one waiting and hasn't waited too long. Any
    /// clip waiting too long is dropped, whichever was asked for.
    pub(crate) fn take(&self, id: u64, now: Instant) -> Option<Vec<u8>> {
        let mut w = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
        if w.as_ref().is_some_and(|(_, _, at)| now.duration_since(*at) > HANDOFF_FOR) {
            *w = None;
        }
        if w.as_ref().is_some_and(|(held, _, _)| *held == id) {
            return w.take().map(|(_, pcm, _)| pcm);
        }
        None
    }

    /// Drop whatever waits: his window, and its page, are gone.
    pub(crate) fn clear(&self) {
        *self.waiting.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// The app's one hand-off.
static HANDOFF: Handoff = Handoff::new();

/// Hold a clip for his page: its number, and whether one was dropped.
pub fn hold_for_page(pcm: Vec<u8>) -> (u64, bool) {
    HANDOFF.hold(pcm, Instant::now())
}

/// The page takes clip `id`.
pub fn take_for_page(id: u64) -> Option<Vec<u8>> {
    HANDOFF.take(id, Instant::now())
}

/// His page is gone: nothing waits for it.
pub fn forget_for_page() {
    HANDOFF.clear();
}

/// A clip's file name in that folder.
pub fn clip_name(at: SystemTime) -> String {
    let ms = at.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis());
    format!("ptt-{ms}.wav")
}

/// Save a clip for a live check, and say what it was: its length, how it
/// ended and its loudest level, never anything of what was said.
pub fn save_clip(dir: &Path, result: Result<Captured, String>) {
    let clip = match result {
        Ok(clip) => clip,
        Err(e) => {
            tracing::warn!("push-to-talk: {e}");
            return;
        }
    };
    if clip.ended == Ended::NeverStarted {
        tracing::info!("push-to-talk: let go before the microphone started; nothing to save");
        return;
    }
    if clip.pcm.is_empty() {
        tracing::info!("push-to-talk: the microphone started, but nothing came from it before it was let go; nothing to save");
        return;
    }
    let name = clip_name(SystemTime::now());
    let saved = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(dir.join(&name), wav(&clip.pcm)));
    let what = describe(&clip);
    match saved {
        Ok(()) => tracing::info!("push-to-talk: {what}; saved as {name} ({SAVE_CLIPS_ENV})"),
        Err(e) => tracing::warn!("push-to-talk: {what}; couldn't be saved ({SAVE_CLIPS_ENV}): {e}"),
    }
}

/// A clip in a line for the log.
pub fn describe(clip: &Captured) -> String {
    let ended = match clip.ended {
        Ended::NeverStarted => "let go before the microphone started",
        Ended::Released => "let go",
        Ended::Full => "held to the 30-second limit",
    };
    let short = if clip.long_enough() { "" } else { ", too short to send" };
    format!(
        "{:.1} s at {} Hz ({ended}), loudest level {:.2}{short}",
        clip.millis() as f64 / 1000.0,
        RATE,
        clip.peak
    )
}

/// The ears' thread's work: let go before it ran, the device isn't touched
/// at all; else it is opened and a clip gathered from it.
pub(crate) fn listen<S: Source>(
    stop: &AtomicBool,
    open: impl FnOnce() -> Result<S, String>,
    level: &mut dyn FnMut(f32),
) -> Result<Captured, String> {
    if stop.load(Ordering::SeqCst) {
        return Ok(Captured::nothing());
    }
    open().and_then(|mut source| gather(&mut source, stop, &Instant::now, level))
}

/// What a press did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pressed {
    /// The microphone is opening.
    Started,
    /// Push-to-talk is already held: nothing changes.
    AlreadyHeld,
    /// Push-to-talk was held, by the look of it, since longer ago than any
    /// clip can last: the release never came, and this press starts anew.
    StartedAfterALostRelease,
    /// Couldn't start a thread to listen on.
    Failed(String),
    /// Push-to-talk is muted (the tray's Mute push-to-talk): nothing opens.
    Muted,
}

/// Push-to-talk held: since when, and the stop flag of its clip while that
/// is still being gathered (`None` once it ended by itself).
struct Held {
    gen: u64,
    at: Instant,
    stop: Option<Arc<AtomicBool>>,
}

/// Whether push-to-talk is held, and its clip. A held level, as plan 14's
/// build § 4.2 has it: a press while held changes nothing, so a clip ended
/// at 30 seconds isn't started again until the key is let go and pressed.
/// Muted, a press opens nothing; `muted` is only written under `held`'s
/// lock, so no press can start once muting has let go of the last.
pub(crate) struct Ears {
    held: Mutex<Option<Held>>,
    gen: AtomicU64,
    muted: AtomicBool,
}

impl Ears {
    pub(crate) const fn new() -> Self {
        Self { held: Mutex::new(None), gen: AtomicU64::new(0), muted: AtomicBool::new(false) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Held>> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Pressed at `now`: open a source with `open` on a thread of its own
    /// and gather a clip from it, which goes to `on_done` however it ends.
    /// The thread outlives the call, so only an `Ears` that lives for good
    /// presses: `EARS`, or a test's leaked one.
    pub(crate) fn press<S, O, L, D>(&'static self, now: Instant, open: O, mut on_level: L, on_done: D) -> Pressed
    where
        S: Source,
        O: FnOnce() -> Result<S, String> + Send + 'static,
        L: FnMut(f32) + Send + 'static,
        D: FnOnce(Result<Captured, String>) + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let gen = self.gen.fetch_add(1, Ordering::SeqCst) + 1;
        let after_lost = {
            let mut held = self.lock();
            if self.muted.load(Ordering::SeqCst) {
                return Pressed::Muted;
            }
            let lost = match held.as_ref() {
                Some(h) if h.stop.is_some() || now.duration_since(h.at) < STALE => return Pressed::AlreadyHeld,
                Some(_) => true,
                None => false,
            };
            *held = Some(Held { gen, at: now, stop: Some(stop.clone()) });
            lost
        };
        let ears = self;
        let spawned = thread::Builder::new().name("fritz-ears".into()).spawn(move || {
            let result = listen(&stop, open, &mut on_level);
            ears.ended(gen);
            on_done(result);
        });
        match spawned {
            Ok(_) => {
                if after_lost {
                    Pressed::StartedAfterALostRelease
                } else {
                    Pressed::Started
                }
            }
            Err(e) => {
                self.ended(gen);
                *self.lock() = None;
                Pressed::Failed(format!("couldn't start listening: {e}"))
            }
        }
    }

    /// Its clip ended (let go, full, or failed): no stop flag is held for
    /// it any longer. Push-to-talk itself stays held until it is let go.
    fn ended(&self, gen: u64) {
        if let Some(h) = self.lock().as_mut().filter(|h| h.gen == gen) {
            h.stop = None;
        }
    }

    /// Let go: its clip, if still being gathered, ends. `false` if it wasn't
    /// held.
    pub(crate) fn release(&self) -> bool {
        Self::let_go(self.lock().take())
    }

    fn let_go(held: Option<Held>) -> bool {
        match held {
            Some(h) => {
                if let Some(stop) = h.stop {
                    stop.store(true, Ordering::SeqCst);
                }
                true
            }
            None => false,
        }
    }

    /// Muted or unmuted. Muting lets go of push-to-talk if it is held (its
    /// clip ends, and goes on as a release's would); `true` if it was.
    pub(crate) fn set_muted(&self, on: bool) -> bool {
        let mut held = self.lock();
        self.muted.store(on, Ordering::SeqCst);
        if on {
            Self::let_go(held.take())
        } else {
            false
        }
    }

    /// Whether push-to-talk is muted.
    pub(crate) fn muted(&self) -> bool {
        self.muted.load(Ordering::SeqCst)
    }

    /// Whether push-to-talk is held.
    #[cfg(test)]
    pub(crate) fn held(&self) -> bool {
        self.lock().is_some()
    }
}

/// The app's one pair of ears.
static EARS: Ears = Ears::new();

/// Push-to-talk pressed: listen on `device` (`audio.inputDevice`'s name),
/// or the default microphone. `on_start` is called on the ears' thread just
/// before the device is opened (never, if let go before that thread ran),
/// so it comes before whatever `on_done` is given.
pub fn press(
    device: Option<String>,
    on_start: impl FnOnce() + Send + 'static,
    on_level: impl FnMut(f32) + Send + 'static,
    on_done: impl FnOnce(Result<Captured, String>) + Send + 'static,
) -> Pressed {
    EARS.press(Instant::now(), started_then(on_start, move || Mic::open(device.as_deref())), on_level, on_done)
}

/// `open`, told first: `on_start` is called, then the source opened, so
/// whoever is told hears it before anything the clip comes to.
pub(crate) fn started_then<S>(
    on_start: impl FnOnce() + Send + 'static,
    open: impl FnOnce() -> Result<S, String> + Send + 'static,
) -> impl FnOnce() -> Result<S, String> + Send + 'static {
    move || {
        on_start();
        open()
    }
}

/// Push-to-talk let go.
pub fn release() -> bool {
    EARS.release()
}

/// Push-to-talk muted (a press opens nothing, and one held is let go) or
/// unmuted; `true` if muting let go of a held one.
pub fn set_muted(on: bool) -> bool {
    EARS.set_muted(on)
}

/// Whether push-to-talk is muted.
pub fn muted() -> bool {
    EARS.muted()
}

/// The microphone, through WASAPI.
mod wasapi {
    use std::time::Duration;

    use windows::core::PCWSTR;
    use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Media::Audio::{
        eCapture, eConsole, IAudioCaptureClient, IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator,
        AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE_ACTIVE,
        WAVEFORMATEX, WAVE_FORMAT_PCM,
    };
    use windows::Win32::System::Com::StructuredStorage::{PropVariantClear, PropVariantToStringAlloc};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED, STGM_READ,
    };
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    use super::{trouble, Packet, Source, FRAME_BYTES, RATE};

    /// The engine's buffer: 200 ms, in 100-ns units.
    const BUFFER: i64 = 2_000_000;

    fn words(e: &windows::core::Error, doing: &str) -> String {
        trouble(e.code().0, doing)
    }

    /// COM on this thread, for as long as it lives.
    struct Com;

    impl Com {
        fn init() -> Result<Self, String> {
            // SAFETY: called once on the ears' own thread, undone by Drop.
            let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
            if hr.is_err() {
                return Err(trouble(hr.0, "starting COM"));
            }
            Ok(Self)
        }
    }

    impl Drop for Com {
        fn drop(&mut self) {
            // SAFETY: matches the CoInitializeEx that succeeded in `init`.
            unsafe { CoUninitialize() };
        }
    }

    /// An event handle, closed when dropped.
    struct Event(HANDLE);

    impl Drop for Event {
        fn drop(&mut self) {
            // SAFETY: the handle CreateEventW gave, closed once.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }

    /// The microphone, open (and, once `start` is called, started):
    /// dropped, it stops, and COM goes last (fields drop in the order they
    /// are declared).
    pub(crate) struct Mic {
        name: String,
        capture: IAudioCaptureClient,
        client: IAudioClient,
        event: Event,
        _com: Com,
    }

    /// A device's name as Windows shows it.
    fn friendly_name(device: &IMMDevice) -> Option<String> {
        // SAFETY: COM is initialised on this thread; the PROPVARIANT is
        // cleared and the string Windows allocated is freed, once each.
        unsafe {
            let store = device.OpenPropertyStore(STGM_READ).ok()?;
            let mut value = store.GetValue(&PKEY_Device_FriendlyName).ok()?;
            let text = PropVariantToStringAlloc(&value).ok();
            let _ = PropVariantClear(&mut value);
            let text = text?;
            let out = text.to_string().ok();
            CoTaskMemFree(Some(text.0 as *const _));
            out
        }
    }

    /// The active capture device named `wanted`, if there is one.
    fn named(enumerator: &IMMDeviceEnumerator, wanted: &str) -> Option<IMMDevice> {
        // SAFETY: COM is initialised on this thread.
        unsafe {
            let all = enumerator.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE).ok()?;
            for i in 0..all.GetCount().ok()? {
                let d = all.Item(i).ok()?;
                if friendly_name(&d).as_deref() == Some(wanted) {
                    return Some(d);
                }
            }
        }
        None
    }

    /// 16 kHz mono PCM16: what Windows converts the device's sound to.
    pub(crate) fn format() -> WAVEFORMATEX {
        WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: 1,
            nSamplesPerSec: RATE,
            nAvgBytesPerSec: RATE * FRAME_BYTES as u32,
            nBlockAlign: FRAME_BYTES as u16,
            wBitsPerSample: 16,
            cbSize: 0,
        }
    }

    /// The flags the stream is opened with: Windows converts to `format()`
    /// at its default quality, and sets the event when sound is ready.
    pub(crate) const FLAGS: u32 =
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;

    impl Mic {
        /// Open the device named `wanted` (or, if it isn't connected, the
        /// default), ready to start: nothing is heard until `start`, which
        /// `gather` calls only while the key is still held.
        pub(crate) fn open(wanted: Option<&str>) -> Result<Self, String> {
            let com = Com::init()?;
            // SAFETY: COM is initialised on this thread (`com`, which the
            // Mic keeps); every pointer passed lives across its call.
            unsafe {
                let enumerator: IMMDeviceEnumerator =
                    CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| words(&e, "listing devices"))?;
                let device = match wanted.and_then(|w| named(&enumerator, w)) {
                    Some(d) => d,
                    None => {
                        if let Some(w) = wanted {
                            tracing::warn!("push-to-talk: {w:?} (audio.inputDevice) isn't connected; using the default microphone");
                        }
                        enumerator.GetDefaultAudioEndpoint(eCapture, eConsole).map_err(|e| words(&e, "finding the microphone"))?
                    }
                };
                let name = friendly_name(&device).unwrap_or_else(|| "the default microphone".to_owned());
                let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(|e| words(&e, "opening the microphone"))?;
                let format = format();
                client
                    .Initialize(AUDCLNT_SHAREMODE_SHARED, FLAGS, BUFFER, 0, &format, None)
                    .map_err(|e| words(&e, "opening the microphone"))?;
                let event = Event(CreateEventW(None, false, false, PCWSTR::null()).map_err(|e| words(&e, "making an event"))?);
                client.SetEventHandle(event.0).map_err(|e| words(&e, "opening the microphone"))?;
                let capture: IAudioCaptureClient = client.GetService().map_err(|e| words(&e, "opening the microphone"))?;
                Ok(Self { name, capture, client, event, _com: com })
            }
        }
    }

    impl Drop for Mic {
        fn drop(&mut self) {
            // SAFETY: the client was initialised in `open`; stopping one
            // never started is S_FALSE, which is no error.
            let _ = unsafe { self.client.Stop() };
        }
    }

    impl Source for Mic {
        fn start(&mut self) -> Result<(), String> {
            // SAFETY: the client was initialised in `open`.
            unsafe { self.client.Start() }.map_err(|e| words(&e, "starting the microphone"))?;
            tracing::info!("push-to-talk: listening on {}", self.name);
            Ok(())
        }

        fn wait(&mut self, timeout: Duration) -> Result<(), String> {
            let ms = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
            // SAFETY: the event lives as long as the Mic.
            let _ = unsafe { WaitForSingleObject(self.event.0, ms) };
            Ok(())
        }

        fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
            // SAFETY: each buffer is read between its GetBuffer and its
            // ReleaseBuffer, and is `frames` whole frames of `format()`.
            unsafe {
                loop {
                    let ready = self.capture.GetNextPacketSize().map_err(|e| words(&e, "listening"))?;
                    if ready == 0 {
                        return Ok(());
                    }
                    let mut data = std::ptr::null_mut();
                    let mut frames = 0u32;
                    let mut flags = 0u32;
                    self.capture
                        .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                        .map_err(|e| words(&e, "listening"))?;
                    let n = frames as usize;
                    let silent = flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
                    let go = if silent || data.is_null() {
                        take(Packet::Silence(n))
                    } else {
                        take(Packet::Sound(std::slice::from_raw_parts(data, n * FRAME_BYTES)))
                    };
                    self.capture.ReleaseBuffer(frames).map_err(|e| words(&e, "listening"))?;
                    if !go {
                        return Ok(());
                    }
                }
            }
        }
    }
}

use wasapi::Mic;

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::sync::mpsc;

    use super::*;

    /// A source that hands over what the test queued, one batch per drain.
    struct Fake {
        batches: VecDeque<Vec<FakePacket>>,
        fail_at: Option<usize>,
        drains: usize,
    }

    #[derive(Clone)]
    enum FakePacket {
        Sound(Vec<u8>),
        Silence(usize),
    }

    impl Fake {
        fn new(batches: Vec<Vec<FakePacket>>) -> Self {
            Self { batches: batches.into(), fail_at: None, drains: 0 }
        }
    }

    impl Source for Fake {
        fn wait(&mut self, _timeout: Duration) -> Result<(), String> {
            Ok(())
        }

        fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
            self.drains += 1;
            if self.fail_at == Some(self.drains) {
                return Err("The microphone went away while he was listening.".to_owned());
            }
            for p in self.batches.pop_front().unwrap_or_default() {
                let go = match &p {
                    FakePacket::Sound(d) => take(Packet::Sound(d)),
                    FakePacket::Silence(n) => take(Packet::Silence(*n)),
                };
                if !go {
                    return Ok(());
                }
            }
            Ok(())
        }
    }

    fn sound(frames: usize, value: i16) -> FakePacket {
        FakePacket::Sound(value.to_le_bytes().repeat(frames))
    }

    /// A clock that moves on `step` at each look.
    fn ticking(step: Duration) -> impl Fn() -> Instant {
        let start = Instant::now();
        let n = Cell::new(0u32);
        move || {
            n.set(n.get() + 1);
            start + step * n.get()
        }
    }

    #[test]
    fn a_clip_takes_whole_frames_up_to_its_limit() {
        let mut c = Clip::default();
        assert!(!c.take(Packet::Sound(&[1, 2, 3])), "an odd byte is not a frame");
        assert_eq!(c.bytes(), &[1, 2]);
        assert!(!c.take(Packet::Silence(3)));
        assert_eq!(c.bytes(), &[1, 2, 0, 0, 0, 0, 0, 0]);
        let mut c = Clip::default();
        assert!(!c.take(Packet::Silence(MAX_BYTES / FRAME_BYTES - 1)));
        assert!(c.take(Packet::Sound(&[9; 64])), "full at the limit");
        assert_eq!(c.bytes().len(), MAX_BYTES);
        assert_eq!(&c.bytes()[MAX_BYTES - 2..], &[9, 9]);
        assert!(c.take(Packet::Silence(usize::MAX)), "nothing more fits, and nothing overflows");
        assert_eq!(c.bytes().len(), MAX_BYTES);
        assert_eq!(MAX_BYTES, 960_000, "30 s of 16 kHz PCM16 mono");
    }

    #[test]
    fn the_level_is_the_rms_of_the_samples() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&0i16.to_le_bytes().repeat(10)), 0.0);
        let half = rms(&16384i16.to_le_bytes().repeat(10));
        assert!((half - 0.5).abs() < 1e-6, "{half}");
        let full = rms(&i16::MIN.to_le_bytes().repeat(4));
        assert!((full - 1.0).abs() < 1e-6, "{full}");
        // A square wave of +/- half: RMS half, though it averages to zero.
        let mut wave = Vec::new();
        for i in 0..100 {
            wave.extend_from_slice(&(if i % 2 == 0 { 16384i16 } else { -16384 }).to_le_bytes());
        }
        assert!((rms(&wave) - 0.5).abs() < 1e-6);
        assert_eq!(rms(&[7]), 0.0, "half a frame is none");
    }

    #[test]
    fn a_clip_is_gathered_until_the_release_and_keeps_what_came_before_it() {
        let stop = AtomicBool::new(false);
        let mut fake = Fake::new(vec![vec![sound(160, 1000)], vec![FakePacket::Silence(160), sound(160, 2000)]]);
        let mut levels = Vec::new();
        let now = ticking(Duration::from_millis(10));
        // Let go during the second wait, after the first drain: what came in
        // during that wait came before the release, and still goes in.
        struct Stopping<'a>(&'a mut Fake, &'a AtomicBool);
        impl Source for Stopping<'_> {
            fn wait(&mut self, t: Duration) -> Result<(), String> {
                if self.0.drains == 1 {
                    self.1.store(true, Ordering::SeqCst);
                }
                self.0.wait(t)
            }
            fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
                self.0.drain(take)
            }
        }
        let clip = gather(&mut Stopping(&mut fake, &stop), &stop, &now, &mut |l| levels.push(l)).unwrap();
        assert_eq!(clip.ended, Ended::Released);
        assert_eq!(clip.pcm.len(), 480 * FRAME_BYTES);
        assert_eq!(&clip.pcm[..2], &1000i16.to_le_bytes());
        assert!(clip.pcm[320..640].iter().all(|b| *b == 0), "silence is zeros");
        assert_eq!(&clip.pcm[958..960], &2000i16.to_le_bytes());
        assert_eq!(fake.drains, 2, "it stops after the drain that saw the release");
        assert!(clip.peak > 0.0);
    }

    /// A source that says when it was started and drained.
    struct Watched {
        calls: Vec<&'static str>,
    }

    impl Source for Watched {
        fn start(&mut self) -> Result<(), String> {
            self.calls.push("start");
            Ok(())
        }
        fn wait(&mut self, _timeout: Duration) -> Result<(), String> {
            self.calls.push("wait");
            Ok(())
        }
        fn drain(&mut self, _take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
            self.calls.push("drain");
            Ok(())
        }
    }

    #[test]
    fn a_source_let_go_of_before_it_starts_is_never_started() {
        let stop = AtomicBool::new(true);
        let mut w = Watched { calls: Vec::new() };
        let clip = gather(&mut w, &stop, &Instant::now, &mut |_| {}).unwrap();
        assert!(w.calls.is_empty(), "{:?}", w.calls);
        assert_eq!((clip.pcm.len(), clip.ended), (0, Ended::NeverStarted));
    }

    #[test]
    fn a_source_is_started_once_before_anything_is_heard() {
        let stop = AtomicBool::new(false);
        let mut w = Watched { calls: Vec::new() };
        struct Twice<'a>(&'a mut Watched, &'a AtomicBool);
        impl Source for Twice<'_> {
            fn start(&mut self) -> Result<(), String> {
                self.0.start()
            }
            fn wait(&mut self, t: Duration) -> Result<(), String> {
                if self.0.calls.iter().filter(|c| **c == "drain").count() == 1 {
                    self.1.store(true, Ordering::SeqCst);
                }
                self.0.wait(t)
            }
            fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
                self.0.drain(take)
            }
        }
        gather(&mut Twice(&mut w, &stop), &stop, &Instant::now, &mut |_| {}).unwrap();
        assert_eq!(w.calls, ["start", "wait", "drain", "wait", "drain"]);
    }

    #[test]
    fn a_source_that_wont_start_fails_the_clip() {
        struct Refusing;
        impl Source for Refusing {
            fn start(&mut self) -> Result<(), String> {
                Err("Windows' audio service isn't running.".into())
            }
            fn wait(&mut self, _t: Duration) -> Result<(), String> {
                panic!("never waited on")
            }
            fn drain(&mut self, _take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
                panic!("never drained")
            }
        }
        let stop = AtomicBool::new(false);
        assert_eq!(gather(&mut Refusing, &stop, &Instant::now, &mut |_| {}).unwrap_err(), "Windows' audio service isn't running.");
    }

    #[test]
    fn a_clip_ends_full_at_thirty_seconds_with_the_key_still_held() {
        let stop = AtomicBool::new(false);
        let second = vec![sound(RATE as usize, 100)];
        let mut fake = Fake::new(vec![second; MAX_SECONDS + 3]);
        let clip = gather(&mut fake, &stop, &ticking(Duration::from_millis(1)), &mut |_| {}).unwrap();
        assert_eq!(clip.ended, Ended::Full);
        assert_eq!(clip.pcm.len(), MAX_BYTES);
        assert_eq!(clip.millis(), 30_000);
        assert_eq!(fake.drains, MAX_SECONDS, "it stops when full, not when told");
    }

    #[test]
    fn the_level_is_given_every_fifty_milliseconds_of_what_came_since() {
        let stop = AtomicBool::new(false);
        // Each look at the clock moves it 30 ms: once at the start, once a
        // round, and once more when a level is given (for `last`). So a
        // level falls due at rounds 2 and 4, each of the two rounds since:
        // a quarter then nothing, a half then nothing. Round 5 (nothing)
        // ends it, with the stop flag set.
        let mut batches = Vec::new();
        for v in [8192i16, 0, 16384, 0, 0] {
            batches.push(vec![sound(10, v)]);
        }
        let mut fake = Fake::new(batches);
        let drains = Cell::new(0);
        let mut levels = Vec::new();
        struct Counting<'a>(&'a mut Fake, &'a Cell<usize>, &'a AtomicBool);
        impl Source for Counting<'_> {
            fn wait(&mut self, t: Duration) -> Result<(), String> {
                self.0.wait(t)
            }
            fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
                self.1.set(self.1.get() + 1);
                if self.1.get() == 5 {
                    self.2.store(true, Ordering::SeqCst);
                }
                self.0.drain(take)
            }
        }
        let clip = gather(&mut Counting(&mut fake, &drains, &stop), &stop, &ticking(Duration::from_millis(30)), &mut |l| {
            levels.push(l)
        })
        .unwrap();
        // RMS over 20 samples, 10 of them at 0.25 (then 0.5) and 10 at 0.
        let (first, second) = ((0.0625f32 / 2.0).sqrt(), (0.25f32 / 2.0).sqrt());
        assert_eq!(levels.len(), 2, "{levels:?}");
        assert!((levels[0] - first).abs() < 1e-6 && (levels[1] - second).abs() < 1e-6, "{levels:?}");
        assert!((clip.peak - second).abs() < 1e-6, "the loudest stretch given: {}", clip.peak);
        assert_eq!(drains.get(), 5);
        assert_eq!(clip.ended, Ended::Released);
    }

    #[test]
    fn a_source_that_fails_fails_the_clip() {
        let stop = AtomicBool::new(false);
        let mut fake = Fake::new(vec![vec![sound(160, 1)]; 5]);
        fake.fail_at = Some(3);
        let err = gather(&mut fake, &stop, &ticking(Duration::from_millis(1)), &mut |_| {}).unwrap_err();
        assert_eq!(err, "The microphone went away while he was listening.");
    }

    #[test]
    fn a_clip_says_how_long_it_is_and_whether_it_is_worth_sending() {
        let c = |frames: usize| Captured { pcm: vec![0; frames * FRAME_BYTES], ended: Ended::Released, peak: 0.25 };
        assert_eq!(c(16_000).millis(), 1000);
        assert!(!c(4799).long_enough());
        assert!(c(4800).long_enough(), "300 ms");
        assert_eq!(describe(&c(24_000)), "1.5 s at 16000 Hz (let go), loudest level 0.25");
        assert_eq!(describe(&c(800)), "0.1 s at 16000 Hz (let go), loudest level 0.25, too short to send");
        let full = Captured { pcm: vec![0; MAX_BYTES], ended: Ended::Full, peak: 1.0 };
        assert_eq!(describe(&full), "30.0 s at 16000 Hz (held to the 30-second limit), loudest level 1.00");
    }

    #[test]
    fn a_wav_is_a_sixteen_khz_mono_pcm16_file() {
        let w = wav(&[1, 2, 3, 4]);
        assert_eq!(w.len(), 48);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(w[4..8].try_into().unwrap()), 40);
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(w[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(w[20..22].try_into().unwrap()), 1, "PCM");
        assert_eq!(u16::from_le_bytes(w[22..24].try_into().unwrap()), 1, "mono");
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(w[28..32].try_into().unwrap()), 32_000);
        assert_eq!(u16::from_le_bytes(w[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(w[34..36].try_into().unwrap()), 16);
        assert_eq!(&w[36..40], b"data");
        assert_eq!(u32::from_le_bytes(w[40..44].try_into().unwrap()), 4);
        assert_eq!(&w[44..], &[1, 2, 3, 4]);
    }

    #[test]
    fn the_stream_asks_windows_for_sixteen_khz_mono_pcm16_and_its_conversion() {
        let f = wasapi::format();
        let (tag, ch, rate, avg, align, bits, cb) =
            (f.wFormatTag, f.nChannels, f.nSamplesPerSec, f.nAvgBytesPerSec, f.nBlockAlign, f.wBitsPerSample, f.cbSize);
        assert_eq!((tag, ch, rate, avg, align, bits, cb), (1, 1, 16_000, 32_000, 2, 16, 0));
        use windows::Win32::Media::Audio::{
            AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        };
        for flag in [
            AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        ] {
            assert_ne!(wasapi::FLAGS & flag, 0, "{flag:#x}");
        }
    }

    #[test]
    fn trouble_is_said_in_the_owners_words() {
        assert_eq!(trouble(0x8007_0490_u32 as i32, "x"), "There's no microphone for him to listen with.");
        assert!(trouble(0x8007_0005_u32 as i32, "x").starts_with("Windows isn't letting desktop apps use the microphone"));
        assert_eq!(trouble(0x8889_0004_u32 as i32, "x"), "The microphone went away while he was listening.");
        assert_eq!(trouble(0x8889_0010_u32 as i32, "x"), "Windows' audio service isn't running.");
        assert_eq!(
            trouble(0x8000_4005_u32 as i32, "opening the microphone"),
            "The microphone couldn't be used (opening the microphone: 0x80004005)."
        );
    }

    #[test]
    fn clips_are_saved_only_in_a_debug_build_and_only_when_asked() {
        assert_eq!(save_clips_dir(false, Some("C:\\clips".into())), None);
        assert_eq!(save_clips_dir(true, None), None);
        assert_eq!(save_clips_dir(true, Some(OsString::new())), None);
        assert_eq!(save_clips_dir(true, Some("C:\\clips".into())), Some(PathBuf::from("C:\\clips")));
        assert_eq!(clip_name(UNIX_EPOCH + Duration::from_millis(1234)), "ptt-1234.wav");
    }

    #[test]
    fn the_microphone_opens_only_on_a_press_with_somewhere_to_put_the_clip() {
        let deaf = Err("His brain can't hear yet: type to him instead.");
        for heard in [Ok(()), deaf] {
            assert_eq!(on_edge(false, Some("C:\\clips".into()), heard), Edge::LetGo);
            assert_eq!(on_edge(false, None, heard), Edge::LetGo);
            // The debug folder first: a live check of his ears needs no brain.
            assert_eq!(on_edge(true, Some("C:\\clips".into()), heard), Edge::Save(PathBuf::from("C:\\clips")));
        }
        assert_eq!(on_edge(true, None, Ok(())), Edge::Send);
        assert_eq!(on_edge(true, None, deaf), Edge::StayClosed("His brain can't hear yet: type to him instead."));
    }

    /// Whoever is told he listens is told before the device opens, so
    /// before whatever the clip comes to; let go before the ears' thread
    /// ran, nobody is told and nothing opened.
    #[test]
    fn listening_is_told_before_the_device_opens() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (order.clone(), order.clone());
        let open = started_then(move || a.lock().unwrap().push("told"), move || {
            b.lock().unwrap().push("opened");
            Err::<Watched, _>("There's no microphone for him to listen with.".to_owned())
        });
        let stop = AtomicBool::new(false);
        let _ = listen(&stop, open, &mut |_| {});
        assert_eq!(*order.lock().unwrap(), ["told", "opened"]);
        let quiet = Arc::new(Mutex::new(Vec::new()));
        let q = quiet.clone();
        let open = started_then(move || q.lock().unwrap().push("told"), || -> Result<Watched, String> { panic!("opened") });
        let stop = AtomicBool::new(true);
        assert_eq!(listen(&stop, open, &mut |_| {}).unwrap().ended, Ended::NeverStarted);
        assert!(quiet.lock().unwrap().is_empty());
    }

    #[test]
    fn a_clip_waits_for_the_page_once_and_not_for_long() {
        let h = Handoff::new();
        let t = Instant::now();
        let (one, dropped) = h.hold(vec![1, 0], t);
        assert!(!dropped);
        assert_eq!(h.take(one + 1, t), None, "not the one waiting");
        assert_eq!(h.take(one, t), Some(vec![1, 0]));
        assert_eq!(h.take(one, t), None, "taken once");
        // A newer clip replaces one never taken.
        let (two, _) = h.hold(vec![2, 0], t);
        let (three, dropped) = h.hold(vec![3, 0], t);
        assert!(dropped && three > two);
        assert_eq!(h.take(two, t), None);
        assert_eq!(h.take(three, t), Some(vec![3, 0]));
        // Too long a wait: gone, whichever is asked for.
        let (four, _) = h.hold(vec![4, 0], t);
        assert_eq!(h.take(four, t + HANDOFF_FOR), Some(vec![4, 0]), "just in time");
        let (five, _) = h.hold(vec![5, 0], t);
        assert_eq!(h.take(999, t + HANDOFF_FOR + Duration::from_millis(1)), None);
        assert_eq!(h.take(five, t), None, "dropped by the late look");
        let (own, _) = h.hold(vec![7, 0], t);
        assert_eq!(h.take(own, t + HANDOFF_FOR + Duration::from_millis(1)), None, "its own id, too late");
        assert_eq!(h.take(own, t), None, "and gone");
        let (six, _) = h.hold(vec![6, 0], t);
        h.clear();
        assert_eq!(h.take(six, t), None, "his page gone");
    }

    #[test]
    fn a_saved_clip_is_a_wav_of_what_was_gathered() {
        let dir = std::env::temp_dir().join(format!("fritz-ears-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_clip(&dir, Ok(Captured { pcm: vec![5, 0, 6, 0], ended: Ended::Released, peak: 0.0 }));
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(files.len(), 1, "{files:?}");
        assert_eq!(std::fs::read(&files[0]).unwrap(), wav(&[5, 0, 6, 0]));
        save_clip(&dir, Err("nope".into()));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "a failure saves nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An `Ears` of the test's own, leaked so its thread may outlive the
    /// call, as `EARS` does.
    fn ears() -> &'static Ears {
        Box::leak(Box::new(Ears::new()))
    }

    /// A source fed through a channel, so a test decides when sound comes;
    /// it says when it first waits (started, and so past every check
    /// before gathering), so a test can let go only once it listens.
    struct Fed {
        fed: mpsc::Receiver<Vec<u8>>,
        there: Option<mpsc::Sender<()>>,
    }

    impl Fed {
        fn new(fed: mpsc::Receiver<Vec<u8>>) -> Self {
            Self { fed, there: None }
        }
    }

    impl Source for Fed {
        fn wait(&mut self, _timeout: Duration) -> Result<(), String> {
            if let Some(t) = self.there.take() {
                let _ = t.send(());
            }
            thread::sleep(Duration::from_millis(1));
            Ok(())
        }

        fn drain(&mut self, take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
            while let Ok(d) = self.fed.try_recv() {
                if !take(Packet::Sound(&d)) {
                    break;
                }
            }
            Ok(())
        }
    }

    type Clips = mpsc::Receiver<Result<Captured, String>>;

    fn done_channel() -> (impl FnOnce(Result<Captured, String>) + Send + 'static, Clips) {
        let (tx, rx) = mpsc::channel();
        (move |r| tx.send(r).unwrap(), rx)
    }

    const LONG: Duration = Duration::from_secs(10);

    #[test]
    fn a_press_listens_until_the_release_and_hands_on_the_clip() {
        let e = ears();
        let (feed, fed) = mpsc::channel();
        let (there_tx, there) = mpsc::channel();
        let (done, clips) = done_channel();
        // Queued before the press: the first drain after it is listening
        // takes it, however late its thread runs.
        feed.send(vec![1, 0, 2, 0]).unwrap();
        assert_eq!(e.press(Instant::now(), move || Ok(Fed { fed, there: Some(there_tx) }), |_| {}, done), Pressed::Started);
        assert!(e.held());
        there.recv_timeout(LONG).unwrap();
        assert!(e.release());
        assert!(!e.held());
        let clip = clips.recv_timeout(LONG).unwrap().unwrap();
        assert_eq!(clip.ended, Ended::Released);
        assert_eq!(clip.pcm, vec![1, 0, 2, 0]);
        assert!(!e.release(), "let go twice is once");
    }

    /// Muting lets go of a held press, its clip handed on as a release's
    /// would be; muted, a press opens nothing and nothing comes of it;
    /// unmuted, a press listens again.
    #[test]
    fn muted_ears_let_go_of_a_held_press_and_open_nothing() {
        let e = ears();
        let (feed, fed) = mpsc::channel();
        let (there_tx, there) = mpsc::channel();
        let (done, clips) = done_channel();
        feed.send(vec![3, 0, 4, 0]).unwrap();
        assert_eq!(e.press(Instant::now(), move || Ok(Fed { fed, there: Some(there_tx) }), |_| {}, done), Pressed::Started);
        there.recv_timeout(LONG).unwrap();
        assert!(e.set_muted(true), "it was held");
        assert!(e.muted() && !e.held());
        let clip = clips.recv_timeout(LONG).unwrap().unwrap();
        assert_eq!((clip.ended, clip.pcm), (Ended::Released, vec![3, 0, 4, 0]));
        let opened = Arc::new(AtomicBool::new(false));
        let o = opened.clone();
        let (muted_done, muted_clips) = done_channel();
        let open = move || {
            o.store(true, Ordering::SeqCst);
            Err::<Fed, _>("never".into())
        };
        assert_eq!(e.press(Instant::now(), open, |_| {}, muted_done), Pressed::Muted);
        assert!(!e.held() && !e.release(), "nothing is held");
        assert!(muted_clips.recv_timeout(Duration::from_millis(100)).is_err(), "nothing comes of it");
        assert!(!opened.load(Ordering::SeqCst), "nothing opened");
        assert!(!e.set_muted(true), "muted again, with nothing held");
        assert!(!e.set_muted(false) && !e.muted());
        let (_feed, fed) = mpsc::channel::<Vec<u8>>();
        let (done, clips) = done_channel();
        assert_eq!(e.press(Instant::now(), move || Ok(Fed::new(fed)), |_| {}, done), Pressed::Started);
        assert!(e.release());
        clips.recv_timeout(LONG).unwrap().unwrap();
    }

    #[test]
    fn a_second_press_while_held_changes_nothing() {
        let e = ears();
        let (_feed, fed) = mpsc::channel::<Vec<u8>>();
        let (done, clips) = done_channel();
        let t = Instant::now();
        assert_eq!(e.press(t, move || Ok(Fed::new(fed)), |_| {}, done), Pressed::Started);
        let opened = Arc::new(AtomicBool::new(false));
        let o = opened.clone();
        let (again, _) = done_channel();
        assert_eq!(
            e.press(t + Duration::from_secs(1), move || {
                o.store(true, Ordering::SeqCst);
                Err::<Fed, _>("never".into())
            }, |_| {}, again),
            Pressed::AlreadyHeld
        );
        assert!(e.release());
        clips.recv_timeout(LONG).unwrap().unwrap();
        thread::sleep(Duration::from_millis(20));
        assert!(!opened.load(Ordering::SeqCst), "the second press opened nothing");
    }

    #[test]
    fn a_full_clip_is_handed_on_while_held_and_not_started_again_until_let_go() {
        let e = ears();
        let (feed, fed) = mpsc::channel();
        let (done, clips) = done_channel();
        let t = Instant::now();
        assert_eq!(e.press(t, move || Ok(Fed::new(fed)), |_| {}, done), Pressed::Started);
        feed.send(vec![0; MAX_BYTES + 10]).unwrap();
        let clip = clips.recv_timeout(LONG).unwrap().unwrap();
        assert_eq!((clip.ended, clip.pcm.len()), (Ended::Full, MAX_BYTES));
        assert!(e.held(), "the key is still down");
        let (again, _) = done_channel();
        assert_eq!(
            e.press(t + Duration::from_secs(31), || Err::<Fed, _>("x".into()), |_| {}, again),
            Pressed::AlreadyHeld,
            "a press while still held, even after the clip ended"
        );
        assert!(e.release(), "let go after the clip ended");
        assert!(!e.held());
    }

    #[test]
    fn a_release_that_never_came_is_given_up_once_no_clip_could_still_be_going() {
        let e = ears();
        let (done, clips) = done_channel();
        let t = Instant::now();
        assert_eq!(e.press(t, || Err::<Fed, _>("There's no microphone for him to listen with.".into()), |_| {}, done), Pressed::Started);
        assert_eq!(clips.recv_timeout(LONG).unwrap().unwrap_err(), "There's no microphone for him to listen with.");
        thread::sleep(Duration::from_millis(20));
        let (again, _) = done_channel();
        assert_eq!(
            e.press(t + STALE - Duration::from_secs(1), || Err::<Fed, _>("x".into()), |_| {}, again),
            Pressed::AlreadyHeld,
            "not yet"
        );
        let (late, late_clips) = done_channel();
        assert_eq!(
            e.press(t + STALE, || Err::<Fed, _>("y".into()), |_| {}, late),
            Pressed::StartedAfterALostRelease
        );
        assert_eq!(late_clips.recv_timeout(LONG).unwrap().unwrap_err(), "y");
    }

    #[test]
    fn a_stale_hold_is_never_given_up_while_its_clip_is_still_being_gathered() {
        let e = ears();
        let (_feed, fed) = mpsc::channel::<Vec<u8>>();
        let (done, clips) = done_channel();
        let t = Instant::now();
        assert_eq!(e.press(t, move || Ok(Fed::new(fed)), |_| {}, done), Pressed::Started);
        let (again, _) = done_channel();
        assert_eq!(e.press(t + STALE * 2, || Err::<Fed, _>("x".into()), |_| {}, again), Pressed::AlreadyHeld);
        assert!(e.release());
        clips.recv_timeout(LONG).unwrap().unwrap();
    }

    /// A source that hears nothing and holds its clip's thread in `wait`
    /// until its gate is sent on or dropped, so a test decides when that
    /// thread ends; it says when it first gets there (started, and so past
    /// every check before gathering).
    struct Gated {
        gate: mpsc::Receiver<()>,
        there: Option<mpsc::Sender<()>>,
    }

    impl Source for Gated {
        fn wait(&mut self, _timeout: Duration) -> Result<(), String> {
            if let Some(t) = self.there.take() {
                let _ = t.send(());
            }
            let _ = self.gate.recv();
            Ok(())
        }
        fn drain(&mut self, _take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn an_old_clip_ending_never_touches_a_newer_hold() {
        let e = ears();
        let (gate1, shut1) = mpsc::channel::<()>();
        let (there_tx, there) = mpsc::channel::<()>();
        let (done1, clips1) = done_channel();
        let open1 = move || Ok(Gated { gate: shut1, there: Some(there_tx) });
        assert_eq!(e.press(Instant::now(), open1, |_| {}, done1), Pressed::Started);
        there.recv_timeout(LONG).unwrap();
        assert!(e.release());
        // Pressed again while the first clip's thread is held in its wait.
        let (feed2, fed2) = mpsc::channel::<Vec<u8>>();
        let (there2_tx, there2) = mpsc::channel();
        let (done2, clips2) = done_channel();
        feed2.send(vec![3, 0]).unwrap();
        assert_eq!(e.press(Instant::now(), move || Ok(Fed { fed: fed2, there: Some(there2_tx) }), |_| {}, done2), Pressed::Started);
        drop(gate1);
        clips1.recv_timeout(LONG).unwrap().unwrap();
        assert!(e.held(), "the first clip's end left the second hold alone");
        let gen_stop = e.lock().as_ref().map(|h| h.stop.is_some());
        assert_eq!(gen_stop, Some(true), "and its stop flag");
        there2.recv_timeout(LONG).unwrap();
        assert!(e.release());
        assert_eq!(clips2.recv_timeout(LONG).unwrap().unwrap().pcm, vec![3, 0]);
    }

    /// A tap shorter than the microphone takes to open: let go while it
    /// opens, it is never started, and the clip is nothing.
    #[test]
    fn let_go_while_the_microphone_opens_it_never_starts() {
        let e = ears();
        let (open_now, opening) = mpsc::channel::<()>();
        let started = Arc::new(AtomicBool::new(false));
        let s = started.clone();
        struct Starting(Arc<AtomicBool>);
        impl Source for Starting {
            fn start(&mut self) -> Result<(), String> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
            fn wait(&mut self, _t: Duration) -> Result<(), String> {
                Ok(())
            }
            fn drain(&mut self, _take: &mut dyn FnMut(Packet<'_>) -> bool) -> Result<(), String> {
                Ok(())
            }
        }
        let (done, clips) = done_channel();
        let (in_open_tx, in_open) = mpsc::channel::<()>();
        let open = move || {
            let _ = in_open_tx.send(()); // past listen's check, now opening
            let _ = opening.recv(); // still opening until the test says
            Ok(Starting(s))
        };
        assert_eq!(e.press(Instant::now(), open, |_| {}, done), Pressed::Started);
        in_open.recv_timeout(LONG).unwrap();
        assert!(e.release(), "let go while it opens");
        open_now.send(()).unwrap();
        let clip = clips.recv_timeout(LONG).unwrap().unwrap();
        assert_eq!((clip.pcm.len(), clip.ended), (0, Ended::NeverStarted));
        assert!(!started.load(Ordering::SeqCst), "never started");
    }

    #[test]
    fn let_go_before_listening_began_the_device_is_never_opened() {
        let stop = AtomicBool::new(true);
        let clip = listen(&stop, || -> Result<Watched, String> { panic!("opened") }, &mut |_| {}).unwrap();
        assert!(clip.pcm.is_empty());
        let stop = AtomicBool::new(false);
        let err = listen(&stop, || -> Result<Watched, String> { Err("There's no microphone for him to listen with.".into()) }, &mut |_| {});
        assert_eq!(err.unwrap_err(), "There's no microphone for him to listen with.");
    }

    #[test]
    fn nothing_is_saved_for_a_clip_never_started() {
        let dir = std::env::temp_dir().join(format!("fritz-ears-none-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_clip(&dir, Ok(Captured::nothing()));
        assert!(!dir.exists() || std::fs::read_dir(&dir).unwrap().count() == 0);
        // Started, but let go before anything came: nothing either.
        save_clip(&dir, Ok(Captured { pcm: Vec::new(), ended: Ended::Released, peak: 0.0 }));
        assert!(!dir.exists() || std::fs::read_dir(&dir).unwrap().count() == 0);
        assert_eq!(describe(&Captured::nothing()), "0.0 s at 16000 Hz (let go before the microphone started), loudest level 0.00, too short to send");
    }

    #[test]
    fn the_level_goes_to_its_callback_while_listening() {
        let e = ears();
        let (feed, fed) = mpsc::channel();
        let (done, clips) = done_channel();
        let (ltx, levels) = mpsc::channel();
        // Queued before the press: the first drain takes it, so the first
        // level given, whenever it falls due, is of it.
        feed.send(16384i16.to_le_bytes().repeat(80)).unwrap();
        assert_eq!(
            e.press(Instant::now(), move || Ok(Fed::new(fed)), move |l| { let _ = ltx.send(l); }, done),
            Pressed::Started
        );
        // Still held: the level comes while it listens, before the release.
        let first = levels.recv_timeout(LONG).unwrap();
        assert!((first - 0.5).abs() < 1e-3, "{first}");
        assert!(e.held());
        assert!(e.release());
        let clip = clips.recv_timeout(LONG).unwrap().unwrap();
        assert!((clip.peak - 0.5).abs() < 1e-3, "{}", clip.peak);
        drop(feed);
    }

    /// This module's code, without its tests.
    fn code() -> String {
        let src = include_str!("audio_in.rs").replace("\r\n", "\n");
        src[..src.find("#[cfg(test)]").unwrap()].to_owned()
    }

    #[test]
    fn nothing_here_keeps_or_logs_what_was_said() {
        let code = code();
        // The only write is the debug-only saver's.
        assert_eq!(code.matches("std::fs::write(").count(), 1);
        for line in code.lines().filter(|l| l.contains("tracing::")) {
            assert!(!line.contains("pcm"), "{line}");
        }
        assert!(!code.contains("static CLIPS"), "no clip is kept");
    }
}
