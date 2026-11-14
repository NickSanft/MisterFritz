import { describe, expect, it } from "vitest";
import { HEADROOM, frameSpan } from "./framing";

/** Where world height `y` lands on the view, from its top (0) to its bottom
 * (1), for a camera framed as `f`. */
function fromTop(f: { centerY: number; distance: number }, y: number, fov: number): number {
  const half = f.distance * Math.tan((fov * Math.PI) / 360);
  return (f.centerY + half - y) / (2 * half);
}

describe("frameSpan", () => {
  it("fits the span, with the margin, exactly in the view", () => {
    const fov = 25;
    const { centerY, distance } = frameSpan(0, 1.7, fov, 0.05);
    expect(centerY).toBeCloseTo(0.85);
    // The visible half-height at that distance is the span's half plus margin.
    const visibleHalf = distance * Math.tan((fov * Math.PI) / 360);
    expect(visibleHalf).toBeCloseTo((1.7 * 1.1) / 2);
  });

  it("stands further back for a taller model", () => {
    expect(frameSpan(0, 2, 25).distance).toBeGreaterThan(frameSpan(0, 1.5, 25).distance);
  });

  it("refuses an empty span, a nonsense lens or no room left for him", () => {
    expect(() => frameSpan(1, 1, 25)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, 0)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, Number.NaN)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, 25, 0.04, -0.1)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, 25, 0.04, 0.9)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, 25, 0.04, Number.NaN)).toThrow(RangeError);
  });

  it("with no headroom, frames him as before", () => {
    const f = frameSpan(0.1, 1.8, 22, 0.04, 0);
    expect(f.centerY).toBeCloseTo((0.1 + 1.8) / 2, 9);
    expect(f.distance).toBeCloseTo((1.7 * 1.08) / 2 / Math.tan((22 * Math.PI) / 360), 9);
  });

  /** The bubble's room: his crown sits below the headroom band, with the
   * margin under it, and his feet keep the margin from the bottom. */
  it("keeps the top share of the view clear over his head", () => {
    const fov = 22;
    const [bottom, top, margin] = [0, 1.7, 0.04];
    const f = frameSpan(bottom, top, fov, margin, HEADROOM);
    const crown = fromTop(f, top, fov);
    const feet = fromTop(f, bottom, fov);
    expect(crown).toBeGreaterThanOrEqual(HEADROOM);
    expect(crown).toBeCloseTo(HEADROOM + ((1 - HEADROOM) * margin) / (1 + 2 * margin), 9);
    expect(1 - feet).toBeCloseTo(((1 - HEADROOM) * margin) / (1 + 2 * margin), 9);
  });

  /** 640 × 0.72 is about PR 2's 460: he is the same size in the new window. */
  it("makes him in a 640-pixel window about the size he was in 460", () => {
    const fov = 22;
    const before = frameSpan(0, 1.7, fov);
    const after = frameSpan(0, 1.7, fov, 0.04, HEADROOM);
    // Pixels a metre of him takes: the window's height over the view's height.
    const perMetre = (px: number, f: { distance: number }) => px / (2 * f.distance * Math.tan((fov * Math.PI) / 360));
    expect(perMetre(640, after) / perMetre(460, before)).toBeCloseTo(1, 1);
  });
});
