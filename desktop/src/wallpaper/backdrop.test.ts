import { describe, expect, it } from "vitest";
import { DRIFT_S, glowAt, paint, rampOf, reach, type Painter, type Ramp } from "./backdrop";

const style = (tokens: Record<string, string>) => ({ getPropertyValue: (n: string) => tokens[n] ?? "" });
const TOKENS: Record<string, string> = {
  "--bg": " #08070b",
  "--surface-0": "#0e0c10",
  "--surface-1": "#15131a ",
  "--surface-2": "#1c1922",
};

describe("the backdrop's colours", () => {
  it("are the page's own tokens: the third surface at a dark light's centre", () => {
    expect(rampOf(style(TOKENS), false)).toEqual({ bg: "#08070b", mid: "#15131a", core: "#1c1922" });
  });

  it("are the lightest surface at a light theme's centre", () => {
    expect(rampOf(style(TOKENS), true)).toEqual({ bg: "#08070b", mid: "#15131a", core: "#0e0c10" });
  });

  it("aren't there until every token they use is", () => {
    const without = (name: string) => Object.fromEntries(Object.entries(TOKENS).filter(([k]) => k !== name));
    for (const name of ["--bg", "--surface-1"]) {
      expect(rampOf(style(without(name)), false), name).toBeNull();
      expect(rampOf(style(without(name)), true), name).toBeNull();
    }
    expect(rampOf(style(without("--surface-2")), false)).toBeNull();
    expect(rampOf(style(without("--surface-2")), true)).not.toBeNull();
    expect(rampOf(style(without("--surface-0")), true)).toBeNull();
    expect(rampOf(style(without("--surface-0")), false)).not.toBeNull();
  });
});

describe("the light", () => {
  it("rests a little above the middle", () => {
    expect(glowAt(0)).toEqual({ x: 0.5, y: 0.42 });
    expect(glowAt(Number.NaN)).toEqual(glowAt(0));
    expect(glowAt(Number.POSITIVE_INFINITY)).toEqual(glowAt(0));
  });

  it("wanders slowly and not far, and comes back", () => {
    let maxStep = 0;
    let prev = glowAt(0);
    for (let t = 0.1; t < 2000; t += 0.1) {
      const g = glowAt(t);
      expect(g.x).toBeGreaterThanOrEqual(0.34 - 1e-9);
      expect(g.x).toBeLessThanOrEqual(0.66 + 1e-9);
      expect(g.y).toBeGreaterThanOrEqual(0.32 - 1e-9);
      expect(g.y).toBeLessThanOrEqual(0.52 + 1e-9);
      maxStep = Math.max(maxStep, Math.hypot(g.x - prev.x, g.y - prev.y));
      prev = g;
    }
    // A frame's step (a tenth of a second, at rich) is a tiny fraction of
    // the page: under two pixels on a 4K monitor, for a light whose edge is
    // a screen away.
    expect(maxStep * 3840).toBeLessThan(2);
    const back = glowAt(DRIFT_S.x * DRIFT_S.y);
    expect(back.x).toBeCloseTo(0.5, 6);
    expect(back.y).toBeCloseTo(0.42, 6);
    expect(glowAt(DRIFT_S.x / 4).x).toBeCloseTo(0.66, 6);
    expect(glowAt(DRIFT_S.y / 4).y).toBeCloseTo(0.52, 6);
  });

  it("reaches past every corner from anywhere it wanders", () => {
    for (const [w, h] of [[1920, 1080], [1080, 1920], [3840, 1600]]) {
      for (const [x, y] of [[0.34, 0.32], [0.66, 0.52], [0.34, 0.52], [0.66, 0.32]]) {
        const corners = [[0, 0], [w, 0], [0, h], [w, h]];
        const far = Math.max(...corners.map(([cx, cy]) => Math.hypot(cx - x * w, cy - y * h)));
        expect(reach(w, h), `${w}x${h} at ${x},${y}`).toBeGreaterThan(far);
      }
    }
  });
});

describe("painting", () => {
  it("covers the page in its colour, then lays the light over all of it", () => {
    const calls: unknown[][] = [];
    const stops: [number, string][] = [];
    const fills: unknown[] = [];
    const ctx: Painter = {
      fillStyle: "",
      fillRect(...a: number[]) {
        calls.push(["fillRect", ...a]);
        fills.push(this.fillStyle);
      },
      createRadialGradient(...a: number[]) {
        calls.push(["gradient", ...a]);
        return { addColorStop: (o: number, c: string) => void stops.push([o, c]) } as CanvasGradient;
      },
    };
    const ramp: Ramp = { bg: "#000001", mid: "#000002", core: "#000003" };
    paint(ctx, 1000, 500, ramp, { x: 0.5, y: 0.4 });
    expect(calls).toEqual([
      ["fillRect", 0, 0, 1000, 500],
      ["gradient", 500, 200, 0, 500, 200, reach(1000, 500)],
      ["fillRect", 0, 0, 1000, 500],
    ]);
    expect(fills[0]).toBe("#000001");
    expect(typeof fills[1]).toBe("object");
    expect(stops).toEqual([[0, "#000003"], [0.42, "#000002"], [1, "#000001"]]);
  });
});
