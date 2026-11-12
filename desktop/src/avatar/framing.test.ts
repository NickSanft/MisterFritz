import { describe, expect, it } from "vitest";
import { frameSpan } from "./framing";

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

  it("refuses an empty span or a nonsense lens", () => {
    expect(() => frameSpan(1, 1, 25)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, 0)).toThrow(RangeError);
    expect(() => frameSpan(0, 1, Number.NaN)).toThrow(RangeError);
  });
});
