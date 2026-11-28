import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { WIDGET_NAMES } from "../shared/protocol";
import {
  GLYPH_PATHS, NODES, NODE_R, TICKS, drawCircle, inkOf, makeGlyphs, nodeAngles, nodeAt, openArc, radiusFor,
  type CircleFrame, type CirclePainter, type Ink,
} from "./circle";

const here = path.dirname(fileURLToPath(import.meta.url));
const TAU = Math.PI * 2;

describe("the circle's ink", () => {
  const style = (t: Record<string, string>) => ({ getPropertyValue: (n: string) => t[n] ?? "" });
  const TOKENS = { "--line": " a", "--line-strong": "b", "--line-glow": "c", "--indigo": "d ", "--text-dim": "e" };

  it("is the page's own tokens", () => {
    expect(inkOf(style(TOKENS))).toEqual({ line: "a", lineStrong: "b", glow: "c", accent: "d", dim: "e" });
  });

  it("isn't there until every token is", () => {
    for (const k of Object.keys(TOKENS)) {
      expect(inkOf(style(Object.fromEntries(Object.entries(TOKENS).filter(([n]) => n !== k)))), k).toBeNull();
    }
  });

  it("names tokens tokens.css has, in every mood", () => {
    const css = readFileSync(path.join(here, "..", "divora", "tokens.css"), "utf8");
    for (const t of ["--line:", "--line-strong:", "--line-glow:", "--indigo:", "--text-dim:"]) expect(css).toContain(t);
  });
});

describe("the ring", () => {
  it("is a little more than half his height, within bounds", () => {
    expect(radiusFor(640)).toBeCloseTo(396.8, 6);
    expect(radiusFor(10)).toBe(120);
    expect(radiusFor(100_000)).toBe(560);
    expect(radiusFor(Number.NaN)).toBe(120);
  });

  it("is open all the way round in the middle of a page", () => {
    expect(openArc(960, 540, 300, 1920, 1080)).toEqual([-Math.PI, Math.PI]);
  });

  it("opens up and to the left of him in the bottom right corner", () => {
    // Where he stands at first: near the bottom right of the screen.
    const arc = openArc(1750, 900, 400, 1920, 1080)!;
    expect(arc).not.toBeNull();
    const mid = (arc[0] + arc[1]) / 2;
    const dir = { x: Math.cos(mid), y: Math.sin(mid) };
    expect(dir.x).toBeLessThan(0);
    expect(dir.y).toBeLessThan(0);
    for (const a of nodeAngles(3, arc)) {
      const p = nodeAt({ x: 1750, y: 900, r: 400 }, a);
      expect(p.x).toBeGreaterThanOrEqual(24);
      expect(p.x).toBeLessThanOrEqual(1920 - 24);
      expect(p.y).toBeGreaterThanOrEqual(24);
      expect(p.y).toBeLessThanOrEqual(1080 - 24);
    }
  });

  it("finds the longest open stretch even across the angle where it wraps", () => {
    // Off the left edge: the open part runs through angle 0.
    const arc = openArc(100, 540, 300, 1920, 1080)!;
    expect(arc[0]).toBeLessThan(0 + TAU);
    expect(arc[1] - arc[0]).toBeGreaterThan(Math.PI / 2);
    const mid = (arc[0] + arc[1]) / 2;
    expect(Math.cos(mid)).toBeGreaterThan(0.9);
  });

  it("keeps the nodes a margin in from the page's edge", () => {
    // Its leftmost point is 10 px in: on the page, but too near the edge.
    expect(openArc(310, 540, 300, 1920, 1080)).not.toEqual([-Math.PI, Math.PI]);
    expect(openArc(330, 540, 300, 1920, 1080)).toEqual([-Math.PI, Math.PI]);
  });

  it("is nowhere when none of it is on the page", () => {
    expect(openArc(-5000, -5000, 200, 1920, 1080)).toBeNull();
  });
});

describe("the nodes", () => {
  it("are the brain's widgets", () => {
    expect([...NODES].sort()).toEqual([...WIDGET_NAMES].sort());
  });

  it("spread evenly, centred on the open stretch, at their gap when there is room", () => {
    const a = nodeAngles(3, [0, 2]);
    expect(a[1]).toBeCloseTo(1, 9);
    expect(a[1]! - a[0]!).toBeCloseTo(0.42, 9);
    expect(a[2]! - a[1]!).toBeCloseTo(0.42, 9);
  });

  it("squeeze into a short stretch", () => {
    const a = nodeAngles(3, [0, 0.3]);
    expect(a[0]).toBeCloseTo(0, 9);
    expect(a[2]).toBeCloseTo(0.3, 9);
  });

  it("gather above him and to his left on a whole ring", () => {
    const a = nodeAngles(3, [-Math.PI, Math.PI]);
    const mid = a[1]!;
    expect(Math.cos(mid)).toBeLessThan(0);
    expect(Math.sin(mid)).toBeLessThan(0);
    expect(nodeAngles(0, [0, 1])).toEqual([]);
    expect(nodeAngles(1, [0, 1])).toEqual([0.5]);
  });
});

/** A painter that records what is drawn, with the style it was drawn in. */
function recorder() {
  const ops: { op: string; alpha: number; stroke: unknown; fill: unknown; dash: number[]; width: number; cap: string; join: string }[] = [];
  const stack: { alpha: number; stroke: unknown; fill: unknown; width: number; cap: CanvasLineCap; join: CanvasLineJoin }[] = [];
  let dash: number[] = [];
  const c: CirclePainter & { ops: typeof ops } = {
    ops,
    globalAlpha: 1,
    lineCap: "butt",
    lineJoin: "miter",
    strokeStyle: "",
    fillStyle: "",
    lineWidth: 1,
    save() {
      stack.push({ alpha: this.globalAlpha, stroke: this.strokeStyle, fill: this.fillStyle, width: this.lineWidth,
                   cap: this.lineCap, join: this.lineJoin });
    },
    restore() {
      const s = stack.pop()!;
      Object.assign(this, { globalAlpha: s.alpha, strokeStyle: s.stroke, fillStyle: s.fill, lineWidth: s.width,
                            lineCap: s.cap, lineJoin: s.join });
    },
    beginPath() {},
    moveTo() {},
    lineTo() {},
    arc() {},
    stroke() {
      ops.push({ op: "stroke", alpha: this.globalAlpha, stroke: this.strokeStyle, fill: this.fillStyle, dash,
                 width: this.lineWidth, cap: this.lineCap, join: this.lineJoin });
    },
    fill() {
      ops.push({ op: "fill", alpha: this.globalAlpha, stroke: this.strokeStyle, fill: this.fillStyle, dash,
                 width: this.lineWidth, cap: this.lineCap, join: this.lineJoin });
    },
    setLineDash(d) {
      dash = d;
    },
    translate() {},
    scale() {},
    createRadialGradient() {
      return { addColorStop() {} } as unknown as CanvasGradient;
    },
  };
  return c;
}

const INK: Ink = { line: "line", lineStrong: "strong", glow: "glow", accent: "accent", dim: "dim" };
const frame = (more: Partial<CircleFrame> = {}): CircleFrame => ({
  x: 500, y: 400, r: 300, presence: 1, spin: 0, spark: 0.5,
  nodes: [
    { name: "clock", angle: -2, lit: true },
    { name: "weather", angle: -1.6, lit: false },
    { name: "desk", angle: -1.2, lit: true },
  ],
  ...more,
});

describe("drawing the circle", () => {
  it("draws nothing at all while it is gone", () => {
    const c = recorder();
    drawCircle(c, frame({ presence: 0 }), INK, () => null);
    drawCircle(c, frame({ presence: Number.NaN }), INK, () => null);
    drawCircle(c, frame({ r: 0 }), INK, () => null);
    expect(c.ops).toEqual([]);
  });

  it("lights a fresh widget's thread, and dashes a stale one's, faintly", () => {
    const c = recorder();
    drawCircle(c, frame(), INK, () => null);
    const threads = c.ops.filter((o) => o.op === "stroke"
      && ((o.stroke === "glow" && o.width === 1.25) || (o.stroke === "line" && o.dash.length)));
    expect(threads.map((o) => o.stroke)).toEqual(["glow", "line", "glow"]);
    expect(threads[1]!.dash).toEqual([2, 6]);
    expect(threads[0]!.dash).toEqual([]);
    // Faint: nothing drawn at full strength.
    expect(Math.max(...c.ops.map((o) => o.alpha))).toBeLessThan(1);
    // The lit nodes in their threads' light, a little under it; the stale one dim.
    const rings = c.ops.filter((o) => o.op === "stroke" && o.width === 1 && (o.stroke === "glow" || o.stroke === "dim")
      && !o.dash.length);
    expect(rings.map((o) => o.stroke)).toEqual(["glow", "dim", "glow"]);
    for (const r of rings) expect(r.alpha).toBeLessThan(threads[0]!.alpha);
    // A faint spark on each lit thread, none on the stale one.
    const sparks = c.ops.filter((o) => o.op === "fill" && o.fill === "accent");
    expect(sparks).toHaveLength(2);
    for (const s of sparks) expect(s.alpha).toBeLessThanOrEqual(0.3);
    // It leaves the context as it found it.
    expect(c.globalAlpha).toBe(1);
  });

  it("fades as a whole", () => {
    const full = recorder();
    drawCircle(full, frame(), INK, () => null);
    const half = recorder();
    drawCircle(half, frame({ presence: 0.5 }), INK, () => null);
    expect(half.ops.length).toBe(full.ops.length);
    half.ops.forEach((o, i) => expect(o.alpha).toBeCloseTo(full.ops[i]!.alpha / 2, 9));
  });

  it("has no spark when still", () => {
    const c = recorder();
    drawCircle(c, frame({ spark: null }), INK, () => null);
    expect(c.ops.filter((o) => o.op === "fill" && o.fill === "accent")).toHaveLength(0);
  });

  it("draws each node's sigil, filled where it has a fill", () => {
    const c = recorder();
    const made: string[] = [];
    const glyphs = makeGlyphs((d) => {
      made.push(d);
      return d as unknown as Path2D;
    });
    drawCircle(c, frame(), INK, glyphs);
    expect(made).toHaveLength(4); // three strokes, the desk's flame
    const sigilStrokes = c.ops.filter((o) => o.op === "stroke" && o.width === 1.5);
    expect(sigilStrokes).toHaveLength(3);
    for (const o of sigilStrokes) expect([o.cap, o.join]).toEqual(["round", "round"]);
    expect(c.ops.filter((o) => o.op === "fill" && o.width === 1.5)).toHaveLength(1);
    // Nothing of the sigils' style is left behind for the rest.
    expect([c.lineCap, c.lineJoin]).toEqual(["butt", "miter"]);
    expect(TICKS).toBeGreaterThan(24);
    expect(NODE_R).toBeGreaterThan(6);
  });
});

describe("the sigils", () => {
  it("are Sigil.tsx's own weather and desk, path for path", () => {
    const sigils = readFileSync(path.join(here, "..", "divora", "sigils", "Sigil.tsx"), "utf8");
    for (const d of [GLYPH_PATHS.weather.stroke[1]!, GLYPH_PATHS.weather.stroke[2]!, GLYPH_PATHS.desk.stroke[0]!,
                     GLYPH_PATHS.desk.fill[0]!]) {
      expect(sigils).toContain(`d="${d}"`);
    }
    // Its circle and rectangle, written as paths.
    expect(sigils).toContain('<circle cx="7.5" cy="7.5" r="2.6" />');
    expect(sigils).toContain('<rect x="13.5" y="8.5" width="3" height="5" rx="0.6" />');
    expect(GLYPH_PATHS.desk.stroke[1]).toContain("A0.6 0.6");
    expect(sigils).toContain('stroke-linecap="round" stroke-linejoin="round"');
  });
});
