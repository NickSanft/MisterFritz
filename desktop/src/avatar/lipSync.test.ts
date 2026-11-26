/**
 * His mouth from his voice: a level as how open, eased quick to open and
 * slower to close, shut when he is quiet, the same at any frame rate.
 */
import { describe, expect, it } from "vitest";
import { ATTACK_MS, DECAY_MS, FLOOR, GAIN, LipSync, SHUT, ease, openness, rmsOf, type AnalyserLike } from "./lipSync";

/** An analyser whose window holds `level` everywhere (so its RMS is |level|). */
function analyser(level: number, size = 1024): AnalyserLike & { level: number; reads: number } {
  return {
    fftSize: size,
    level,
    reads: 0,
    getFloatTimeDomainData(into: Float32Array) {
      this.reads += 1;
      expect(into.length).toBe(this.fftSize);
      into.fill(this.level);
    },
  };
}

describe("his mouth from his voice", () => {
  it("opens with the level: shut under the floor, wide at Kokoro's loudest", () => {
    expect([FLOOR, GAIN]).toEqual([0.02, 6]);
    expect(openness(0)).toBe(0);
    expect(openness(0.02)).toBe(0);
    // Kokoro's middle, its 90th percentile and its loudest, as measured.
    expect(openness(0.06)).toBeCloseTo(0.24);
    expect(openness(0.14)).toBeCloseTo(0.72);
    expect(openness(0.22)).toBe(1);
    expect(openness(5)).toBe(1);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -1]) expect(openness(bad)).toBe(0);
  });

  it("reads the level as the window's RMS", () => {
    expect(rmsOf(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(0.5);
    expect(rmsOf(new Float32Array([0.3, 0.4]))).toBeCloseTo(Math.sqrt(0.125));
    expect(rmsOf(new Float32Array(0))).toBe(0);
    expect(rmsOf(new Float32Array([Number.NaN, 1]))).toBe(0);
  });

  it("opens quicker than it closes, by its time constants", () => {
    expect(ease(0, 1, ATTACK_MS)).toBeCloseTo(1 - Math.exp(-1));
    expect(ease(1, 0, DECAY_MS)).toBeCloseTo(Math.exp(-1));
    expect(ATTACK_MS).toBeLessThan(DECAY_MS);
    expect(ease(0.4, 0.4, 100)).toBe(0.4);
    expect(ease(0.4, 1, 0)).toBe(0.4);
    expect(ease(0.4, 1, Number.NaN)).toBe(0.4);
    expect(ease(0.4, 1, -50)).toBe(0.4);
  });

  it("is the same at any frame rate", () => {
    const a = ease(ease(0, 1, 16), 1, 17);
    expect(a).toBeCloseTo(ease(0, 1, 33), 10);
  });

  it("follows his voice while he speaks, and shuts when he stops", () => {
    const lips = new LipSync();
    const loud = analyser(0.14);
    let v = 0;
    for (let i = 0; i < 30; i++) v = lips.sample(33, loud, true);
    expect(v).toBeCloseTo(openness(0.14), 3);
    expect(lips.value).toBe(v);
    expect(loud.reads).toBe(30);
    // Quiet between words: it closes, slower than it opened.
    loud.level = 0;
    const after = lips.sample(33, loud, true);
    expect(after).toBeLessThan(v);
    expect(after).toBeGreaterThan(v * 0.5);
    // He stops: it isn't read, and it settles shut.
    const reads = loud.reads;
    for (let i = 0; i < 60; i++) lips.sample(33, loud, false);
    expect(loud.reads).toBe(reads);
    expect(lips.value).toBe(0);
  });

  it("stays shut with nothing to read, and settles to exactly shut", () => {
    const lips = new LipSync();
    expect(lips.sample(33, null, true)).toBe(0);
    for (let i = 0; i < 10; i++) lips.sample(33, analyser(0.2), true);
    let v = lips.value;
    let steps = 0;
    while (v > 0 && steps < 1000) {
      v = lips.sample(33, null, true);
      steps += 1;
      if (v > 0) expect(v).toBeGreaterThanOrEqual(SHUT);
    }
    expect(v).toBe(0);
  });

  it("reads a window the analyser's size, whatever it is", () => {
    const lips = new LipSync();
    lips.sample(33, analyser(0.1, 512), true);
    lips.sample(33, analyser(0.1, 2048), true);
  });
});
