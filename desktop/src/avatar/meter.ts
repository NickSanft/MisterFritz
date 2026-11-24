/**
 * The meter in his bubble while he listens: DivoraVoice's horizontal level
 * meter, fed the shell's levels (`ptt:level`, the RMS of the microphone 20
 * times a second). Speech is quiet as RMS goes (a few hundredths), so the
 * meter reads it in decibels, -60 dBFS to 0, as level meters do; its cap
 * holds the highest level of the last second.
 */

/** The quietest level the meter shows above empty, in dBFS. */
export const FLOOR_DB = -60;
/** How long the cap holds the highest level. */
export const PEAK_HOLD_MS = 1000;
/** A level not followed by another within this long has gone quiet (the
 * shell sends one every 50 ms while it listens). */
export const STALE_MS = 250;

/** An RMS, 0 to 1, as the meter shows it: 0 to 1 over FLOOR_DB to 0 dBFS. */
export function meterLevel(rms: number): number {
  if (!(rms > 0)) return 0;
  const db = 20 * Math.log10(Math.min(rms, 1));
  return Math.min(1, Math.max(0, (db - FLOOR_DB) / -FLOOR_DB));
}

/** What the meter shows: its level and its cap, each 0 to 1. */
export interface MeterView {
  level: number;
  peak: number;
}

/** The levels heard for the press he listens to now. */
export class LevelMeter {
  private press: number | null = null;
  private heard: { at: number; level: number }[] = [];

  /** Press `press`'s level `rms`, heard at `at` (ms): a newer press starts
   * afresh, and an older one's late level is dropped. */
  hear(press: number, rms: number, at: number): void {
    if (this.press !== null && press < this.press) return;
    if (press !== this.press) {
      this.press = press;
      this.heard = [];
    }
    this.heard.push({ at, level: meterLevel(rms) });
    this.heard = this.heard.filter((h) => at - h.at <= PEAK_HOLD_MS);
  }

  /** What it shows for press `press` at `now`: the latest level, unless it
   * has gone quiet, and the highest of the last second. Empty for a press
   * not heard from yet. */
  view(press: number, now: number): MeterView {
    if (press !== this.press) return { level: 0, peak: 0 };
    const recent = this.heard.filter((h) => now - h.at <= PEAK_HOLD_MS);
    const last = recent[recent.length - 1];
    return {
      level: last && now - last.at <= STALE_MS ? last.level : 0,
      peak: recent.reduce((m, h) => Math.max(m, h.level), 0),
    };
  }
}
