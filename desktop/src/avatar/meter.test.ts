/**
 * The meter in his bubble while he listens: the shell's RMS levels read in
 * decibels, the latest shown until it goes quiet, the cap holding the
 * highest of the last second, one press at a time.
 */
import { describe, expect, it } from "vitest";
import { FLOOR_DB, LevelMeter, PEAK_HOLD_MS, STALE_MS, meterLevel } from "./meter";

describe("the meter's scale", () => {
  it("reads RMS in decibels, -60 dBFS empty and 0 dBFS full", () => {
    expect(FLOOR_DB).toBe(-60);
    expect(meterLevel(1)).toBe(1);
    expect(meterLevel(0.001)).toBeCloseTo(0, 10);
    expect(meterLevel(0.0001)).toBe(0);
    expect(meterLevel(0.01)).toBeCloseTo(1 / 3, 6);
    expect(meterLevel(0.1)).toBeCloseTo(2 / 3, 6);
    // Speech as the live checks heard it (a few hundredths) reads near half.
    expect(meterLevel(0.03)).toBeGreaterThan(0.45);
    expect(meterLevel(0.03)).toBeLessThan(0.55);
  });

  it("takes nothing that isn't a level as empty, and never goes past full", () => {
    for (const bad of [0, -0.5, Number.NaN, Number.NEGATIVE_INFINITY]) {
      expect(meterLevel(bad), String(bad)).toBe(0);
    }
    expect(meterLevel(2)).toBe(1);
    expect(meterLevel(Number.POSITIVE_INFINITY)).toBe(1);
  });
});

describe("the levels it holds", () => {
  it("shows the latest level, and the highest of the last second as its cap", () => {
    const m = new LevelMeter();
    m.hear(1, 0.1, 0);
    m.hear(1, 0.01, 50);
    expect(m.view(1, 60)).toEqual({ level: meterLevel(0.01), peak: meterLevel(0.1) });
    // The cap lets go of a level once it is older than a second.
    m.hear(1, 0.01, PEAK_HOLD_MS + 10);
    expect(m.view(1, PEAK_HOLD_MS + 20)).toEqual({ level: meterLevel(0.01), peak: meterLevel(0.01) });
  });

  it("goes quiet when the levels stop coming, and empty once a second has passed", () => {
    const m = new LevelMeter();
    m.hear(1, 0.1, 0);
    expect(m.view(1, STALE_MS).level).toBe(meterLevel(0.1));
    expect(m.view(1, STALE_MS + 1)).toEqual({ level: 0, peak: meterLevel(0.1) });
    expect(m.view(1, PEAK_HOLD_MS + 1)).toEqual({ level: 0, peak: 0 });
  });

  it("is empty for a press it hasn't heard, starts afresh for a newer one, and drops an older one's late level", () => {
    const m = new LevelMeter();
    expect(m.view(1, 0)).toEqual({ level: 0, peak: 0 });
    m.hear(1, 0.5, 0);
    expect(m.view(2, 10)).toEqual({ level: 0, peak: 0 });
    m.hear(2, 0.01, 20);
    expect(m.view(2, 30)).toEqual({ level: meterLevel(0.01), peak: meterLevel(0.01) });
    m.hear(1, 1, 40);
    expect(m.view(2, 50)).toEqual({ level: meterLevel(0.01), peak: meterLevel(0.01) });
    expect(m.view(1, 50)).toEqual({ level: 0, peak: 0 });
  });
});
