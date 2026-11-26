/**
 * His voice on his page (plan 14, PR 10): the brain's `speak` frames, a
 * sentence each, spoken by the shell (`tts_synthesize`, Kokoro) one at a
 * time and in order, and played back to back through Web Audio.
 *
 * - One turn's speech at a time: a frame for another turn begins it, and
 *   what was left of the last is dropped.
 * - A sentence is asked of the shell only once the one before has come
 *   back, so they come back in order and Kokoro speaks one at a time; each
 *   plays as soon as it comes, after what is already playing.
 * - `stop()` (a barge-in: push-to-talk, a typed question, the link lost)
 *   stops what plays and drops what waits; a sentence the shell is still
 *   speaking is dropped when it comes back, and the turn says nothing more.
 * - `busy` while anything is left to say or still playing: the bubble stays
 *   up while he speaks.
 * - A sentence that hasn't ended a little after it should have (the audio
 *   isn't playing: a device gone, or Chromium not letting it start) drops
 *   the rest of the turn and says so once, so the bubble isn't held up by a
 *   voice nobody hears.
 * - The audio context is made at the first sentence, so a desk he never
 *   speaks on opens no audio device, and suspended once he has been quiet
 *   a few seconds, so it doesn't keep the device's thread running.
 * - It doesn't ride the paced loop: Web Audio plays on its own, so he goes
 *   on talking while he is parked or the display is off.
 * - Nothing he says is logged.
 */
import type { Clock } from "../shared/clock";
import type { Speak } from "../shared/protocol";
import { TTS_SAMPLE_RATE, type Spoken } from "./voiceLink";

/** The part of an AudioBufferSourceNode used here. */
export interface SourceLike {
  buffer: unknown;
  onended: ((ev: Event) => unknown) | null;
  connect(to: unknown): unknown;
  start(when?: number): void;
  stop(): void;
}

/** The part of an AudioBuffer used here. */
export interface BufferLike {
  readonly duration: number;
  copyToChannel(source: Float32Array<ArrayBuffer>, channel: number): void;
}

/** The part of an AudioContext used here (a test passes its own). */
export interface OutLike {
  readonly currentTime: number;
  readonly state: string;
  readonly destination: unknown;
  resume(): Promise<void>;
  suspend(): Promise<void>;
  createBuffer(channels: number, length: number, rate: number): BufferLike;
  createBufferSource(): SourceLike;
}

export interface SpeechDeps {
  /** The audio context, made when he first speaks. */
  audio(): OutLike;
  /** One sentence spoken by the shell. */
  speak(text: string): Promise<Spoken>;
  clock: Clock;
}

/** How long he is quiet before the audio context is suspended. */
export const SUSPEND_AFTER_MS = 5_000;
/** How far ahead of now a sentence starts when nothing is playing, in
 * seconds, so its first samples aren't lost to the scheduling. */
export const LEAD_S = 0.05;

/** How long past a sentence's end, by the wall clock, it may still be
 * playing before the audio is taken not to be playing at all. */
export const STALL_MARGIN_MS = 2_000;

/** A frame's sequence number no later frame of a stopped turn can pass. */
const STOPPED = Number.MAX_SAFE_INTEGER;

export class Speech {
  private turnId: string | null = null;
  /** Bumped by every stop: a sentence asked before it is dropped. */
  private gen = 0;
  private waiting: string[] = [];
  /** The highest sequence number taken for this turn. */
  private lastSeq = -1;
  private asking = false;
  /** Each sentence scheduled and not yet ended, with its stall watch. */
  private playing = new Map<SourceLike, unknown>();
  /** When the last scheduled sentence ends, on the context's clock. */
  private endsAt = 0;
  private out: OutLike | null = null;
  private broken = false;
  private quietTimer: unknown = null;
  private warnedFor: string | null = null;
  /** What he has done, for a live check's eyes: counts, never words.
   * `started`: sentences set playing; `spoken`: played to their end. */
  readonly counts = { started: 0, spoken: 0, failed: 0, stopped: 0, stuck: 0 };

  constructor(private readonly deps: SpeechDeps) {}

  /** Anything left to say, or still playing. */
  get busy(): boolean {
    return this.asking || this.waiting.length > 0 || this.playing.size > 0;
  }

  /** The audio context's state, or "none" before he first speaks. */
  get audioState(): string {
    return this.out?.state ?? (this.broken ? "broken" : "none");
  }

  /** A `speak` frame from the brain. */
  hear(f: Speak): void {
    if (f.turn_id !== this.turnId) {
      this.stop();
      this.turnId = f.turn_id;
      this.lastSeq = -1;
    }
    // A repeat, or a turn stopped.
    if (!Number.isSafeInteger(f.seq) || f.seq <= this.lastSeq) return;
    this.lastSeq = f.seq;
    // An empty final: nothing more to say.
    if (f.text.trim() === "") return;
    this.waiting.push(f.text);
    this.next();
  }

  /** Stop speaking, at once: what plays stops, what waits is dropped, and
   * the turn says nothing more. */
  stop(): void {
    if (this.busy) this.counts.stopped += 1;
    this.halt();
  }

  /** Everything this turn has playing or waiting, dropped. */
  private halt(): void {
    this.gen += 1;
    this.waiting = [];
    this.asking = false;
    this.lastSeq = STOPPED;
    for (const [s, watch] of this.playing) {
      this.deps.clock.clearTimeout(watch);
      s.onended = null;
      try {
        s.stop();
      } catch {
        // Not started yet, or done: either way it won't play.
      }
    }
    this.playing.clear();
    this.endsAt = 0;
    this.quiet();
  }

  /** The next sentence, asked of the shell once the one before is back. */
  private next(): void {
    if (this.asking || this.waiting.length === 0) return;
    const text = this.waiting.shift()!;
    const gen = this.gen;
    this.asking = true;
    this.awake();
    this.deps
      .speak(text)
      .catch((e: unknown): Spoken => ({ why: e instanceof Error ? e.message : String(e) }))
      .then((spoken) => {
        if (gen !== this.gen) return;
        this.asking = false;
        if ("samples" in spoken) this.play(spoken.samples);
        else this.failed(spoken.why);
        this.next();
        this.quiet();
      });
  }

  /** `samples` played after whatever plays now. */
  private play(samples: Float32Array<ArrayBuffer>): void {
    const out = this.audio();
    if (out === null) {
      this.failed("there is no audio to play it on");
      return;
    }
    const buffer = out.createBuffer(1, samples.length, TTS_SAMPLE_RATE);
    buffer.copyToChannel(samples, 0);
    const src = out.createBufferSource();
    src.buffer = buffer;
    src.connect(out.destination);
    const at = Math.max(out.currentTime + LEAD_S, this.endsAt);
    const ends = at + buffer.duration;
    // By the wall clock: the context's own stands still when it isn't
    // playing, so it can't say so.
    const watch = this.deps.clock.setTimeout(() => this.stalled(), (ends - out.currentTime) * 1000 + STALL_MARGIN_MS);
    src.onended = () => {
      this.deps.clock.clearTimeout(watch);
      if (!this.playing.delete(src)) return;
      this.counts.spoken += 1;
      this.quiet();
    };
    this.playing.set(src, watch);
    src.start(at);
    this.endsAt = ends;
    this.counts.started += 1;
  }

  /** A sentence that should have ended hasn't: the audio isn't playing.
   * The turn goes, which says so once (a turn stalls once: it is gone). */
  private stalled(): void {
    this.counts.stuck += 1;
    console.warn(`avatar: his voice isn't playing (audio context ${this.audioState}); this turn goes unsaid`);
    this.halt();
  }

  private failed(why: string): void {
    this.counts.failed += 1;
    // Once a turn: a voice that isn't installed would say so every sentence.
    if (this.warnedFor === this.turnId) return;
    this.warnedFor = this.turnId;
    console.warn(`avatar: a sentence of his couldn't be spoken: ${why}`);
  }

  /** The audio context, made the first time; null if it can't be. */
  private audio(): OutLike | null {
    if (this.out === null && !this.broken) {
      try {
        this.out = this.deps.audio();
      } catch (e) {
        this.broken = true;
        console.warn("avatar: no audio context for his voice", e);
      }
    }
    if (this.out?.state === "suspended") {
      this.out.resume().catch((e: unknown) => console.warn("avatar: his voice's audio wouldn't resume", e));
    }
    return this.out;
  }

  /** About to speak: the context stays awake. */
  private awake(): void {
    if (this.quietTimer !== null) {
      this.deps.clock.clearTimeout(this.quietTimer);
      this.quietTimer = null;
    }
    if (this.out?.state === "suspended") this.audio();
  }

  /** Quiet now: the context is suspended unless he speaks again first. */
  private quiet(): void {
    if (this.busy || this.out === null || this.quietTimer !== null) return;
    this.quietTimer = this.deps.clock.setTimeout(() => {
      this.quietTimer = null;
      if (!this.busy && this.out?.state === "running") {
        this.out.suspend().catch((e: unknown) => console.warn("avatar: his voice's audio wouldn't suspend", e));
      }
    }, SUSPEND_AFTER_MS);
  }
}
