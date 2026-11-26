/**
 * His mouth, from his voice (plan 14, PR 10): how loud what he says is now,
 * read from the analyser his speech plays through (speech.ts), as how open
 * his mouth is (the VRM's `aa`, 0 to 1), eased so it follows his syllables
 * without chattering: quick to open, a little slower to close.
 *
 * - The level is the RMS of the analyser's last window (1024 samples, about
 *   21 ms at 48 kHz). Kokoro's speech measured about 0.06 at its middle,
 *   0.14 at its 90th percentile and 0.22 at its loudest, with its gaps under
 *   0.02; so the mouth opens past FLOOR at GAIN to the unit: a quarter open
 *   for most of a sentence, wide at its loudest, shut between words.
 * - ATTACK_MS and DECAY_MS are time constants, so a frame's step is the same
 *   at any frame rate.
 * - Shut while nothing is playing; it moves at motion 0 too, as he speaks.
 */

export const FLOOR = 0.02;
export const GAIN = 6;
export const ATTACK_MS = 80;
export const DECAY_MS = 150;
/** Under this it is shut: an ease never quite reaches 0. */
export const SHUT = 0.001;

/** The part of an AnalyserNode read here. */
export interface AnalyserLike {
  readonly fftSize: number;
  getFloatTimeDomainData(into: Float32Array<ArrayBuffer>): void;
}

/** How open his mouth is for a level `rms`, 0 to 1. */
export function openness(rms: number): number {
  if (!Number.isFinite(rms)) return 0;
  return Math.min(1, Math.max(0, (rms - FLOOR) * GAIN));
}

/** The root mean square of `samples` (0 for none, or any not finite). */
export function rmsOf(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const s of samples) sum += s * s;
  const rms = Math.sqrt(sum / samples.length);
  return Number.isFinite(rms) ? rms : 0;
}

/** `v` eased toward `target` over `dtMs`: by ATTACK_MS when opening, by
 * DECAY_MS when closing. */
export function ease(v: number, target: number, dtMs: number): number {
  const dt = Number.isFinite(dtMs) ? Math.max(0, dtMs) : 0;
  const tau = target > v ? ATTACK_MS : DECAY_MS;
  return v + (target - v) * (1 - Math.exp(-dt / tau));
}

export class LipSync {
  private v = 0;
  private window: Float32Array<ArrayBuffer> | null = null;

  /** How open his mouth is now. */
  get value(): number {
    return this.v;
  }

  /** His mouth this frame, `dtMs` after the last: from `analyser` while he
   * speaks, else closing. */
  sample(dtMs: number, analyser: AnalyserLike | null, speaking: boolean): number {
    let target = 0;
    if (speaking && analyser !== null) {
      if (this.window === null || this.window.length !== analyser.fftSize) this.window = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(this.window);
      target = openness(rmsOf(this.window));
    }
    this.v = ease(this.v, target, dtMs);
    if (this.v < SHUT) this.v = 0;
    return this.v;
  }
}
