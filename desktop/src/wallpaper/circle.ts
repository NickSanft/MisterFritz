/**
 * The study circle (plan 14, PR 12): the desk's spell circle, drawn behind
 * the icons around Fritz. His place is its core; a faint outer ring with
 * small ticks turns about it, very slowly; the brain's widgets (clock,
 * weather, his desk) sit on the ring as sigil-nodes, each on a thread from
 * the core that is lit while the widget's data is fresh. Faint, like a
 * watermark (the owner's call): the lit threads are the brightest thing in
 * it, and still soft. It fades out while he is hidden.
 *
 * Pure geometry and drawing on a 2D context, for the paced loop in
 * main.ts: nothing here keeps time or listens.
 */

/** The colours it is drawn in, from the page's own tokens. */
export interface Ink {
  /** The faintest line: the inner ring, the ticks, an unlit thread. */
  line: string;
  /** The outer ring. */
  lineStrong: string;
  /** A lit thread, and a lit node, a little under it. */
  glow: string;
  /** The light at the core, and a lit thread's spark. */
  accent: string;
  /** An unlit node. */
  dim: string;
}

/** The ink from the tokens as the page wears them now; null while one is
 * missing. */
export function inkOf(style: Pick<CSSStyleDeclaration, "getPropertyValue">): Ink | null {
  const read = (name: string): string => style.getPropertyValue(name).trim();
  const ink = {
    line: read("--line"),
    lineStrong: read("--line-strong"),
    glow: read("--line-glow"),
    accent: read("--indigo"),
    dim: read("--text-dim"),
  };
  return Object.values(ink).every((v) => v !== "") ? ink : null;
}

/** The widgets on the ring, in the order they sit along it. */
export const NODES = ["clock", "weather", "desk"] as const;
export type NodeName = (typeof NODES)[number];

/** The ring's radius for Fritz `size` tall (CSS pixels): a little more than
 * half his window's height, within bounds a small or huge window can't
 * break. */
export function radiusFor(size: number): number {
  const r = Number.isFinite(size) ? size * 0.62 : 0;
  return Math.min(560, Math.max(120, r));
}

/** How far in from the page's edge a node must sit to be seen whole. */
export const NODE_MARGIN = 24;

const TAU = Math.PI * 2;

/** The longest run of the ring, as [from, to] in radians (canvas angles,
 * y down, `to` > `from`), whose points are inside a `width` × `height`
 * page by `margin`: where the nodes can go. The whole ring, if all of it
 * is in; null if none of it is. Pure. */
export function openArc(
  cx: number, cy: number, r: number, width: number, height: number, margin = NODE_MARGIN,
): [number, number] | null {
  const STEPS = 360;
  const inside = (k: number): boolean => {
    const a = (k / STEPS) * TAU;
    const x = cx + r * Math.cos(a);
    const y = cy + r * Math.sin(a);
    return x >= margin && x <= width - margin && y >= margin && y <= height - margin;
  };
  const ins = Array.from({ length: STEPS }, (_, k) => inside(k));
  if (ins.every(Boolean)) return [-Math.PI, Math.PI];
  if (!ins.some(Boolean)) return null;
  // Start just after a point that is out, so a run never wraps unseen.
  const out = ins.indexOf(false);
  let best: [number, number] = [0, 0];
  let run = 0;
  let runStart = 0;
  for (let j = 1; j <= STEPS; j++) {
    const k = (out + j) % STEPS;
    if (ins[k]) {
      if (run === 0) runStart = out + j;
      run += 1;
      if (run > best[1] - best[0]) best = [runStart, runStart + run - 1];
    } else {
      run = 0;
    }
  }
  return [(best[0] / STEPS) * TAU, (best[1] / STEPS) * TAU];
}

/** The angles of `n` nodes on `arc`: spread evenly across it, no closer
 * than `gap` (radians) when it has the room, centred on it. On the whole
 * ring, they gather above him and to his left, where a monitor usually
 * has room. Pure. */
export function nodeAngles(n: number, arc: [number, number], gap = 0.42): number[] {
  if (n <= 0) return [];
  const [from, to] = arc;
  const whole = to - from >= TAU - 1e-9;
  const span = whole ? gap * (n - 1) : Math.min(gap * (n - 1), to - from);
  const mid = whole ? -Math.PI * 0.6 : (from + to) / 2;
  if (n === 1) return [mid];
  return Array.from({ length: n }, (_, i) => mid - span / 2 + (span * i) / (n - 1));
}

/** What a frame of the circle shows. */
export interface CircleFrame {
  /** The core, in CSS pixels. */
  x: number;
  y: number;
  r: number;
  /** 0 (gone) to 1 (shown). */
  presence: number;
  /** The ring's turn, in radians. */
  spin: number;
  /** How far along its thread a lit node's spark is, 0 to 1; null for
   * none (motion at functional). */
  spark: number | null;
  /** Each node's angle on the ring, and whether its thread is lit. */
  nodes: { name: NodeName; angle: number; lit: boolean }[];
}

/** The part of a 2D context the circle uses, so tests can watch it. */
export interface CirclePainter {
  globalAlpha: number;
  lineCap: CanvasLineCap;
  lineJoin: CanvasLineJoin;
  strokeStyle: string | CanvasGradient | CanvasPattern;
  fillStyle: string | CanvasGradient | CanvasPattern;
  lineWidth: number;
  save(): void;
  restore(): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, start: number, end: number): void;
  stroke(path?: Path2D): void;
  fill(path?: Path2D): void;
  setLineDash(segments: number[]): void;
  translate(x: number, y: number): void;
  scale(x: number, y: number): void;
  createRadialGradient(x0: number, y0: number, r0: number, x1: number, y1: number, r1: number): CanvasGradient;
}

/** A node's glyph as a path in its sigil's 24-unit box, or null where
 * paths can't be made (a test). */
export type Glyphs = (name: NodeName) => { stroke: Path2D; fill: Path2D | null } | null;

/** A node's radius, and its glyph's size within it. */
export const NODE_R = 11;
const GLYPH_SCALE = 0.62;
/** How many ticks ring the circle, and every how many is a long one. */
export const TICKS = 72;
const LONG_EVERY = 6;

/** Where a node at `angle` sits. Pure. */
export function nodeAt(f: Pick<CircleFrame, "x" | "y" | "r">, angle: number): { x: number; y: number } {
  return { x: f.x + f.r * Math.cos(angle), y: f.y + f.r * Math.sin(angle) };
}

/** Draw a frame of the circle. Nothing when it is all but gone. */
export function drawCircle(ctx: CirclePainter, f: CircleFrame, ink: Ink, glyphs: Glyphs): void {
  const p = Math.min(1, Math.max(0, Number.isFinite(f.presence) ? f.presence : 0));
  if (p < 0.005 || !(f.r > 0)) return;
  ctx.save();
  // The core: a soft light, the faintest wash of the accent.
  const core = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r * 0.55);
  core.addColorStop(0, ink.accent);
  core.addColorStop(1, "transparent");
  ctx.globalAlpha = 0.07 * p;
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(f.x, f.y, f.r * 0.55, 0, TAU);
  ctx.fill();

  // The rings, and the ticks between them, turning with the spin.
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.9 * p;
  ctx.strokeStyle = ink.lineStrong;
  ctx.beginPath();
  ctx.arc(f.x, f.y, f.r, 0, TAU);
  ctx.stroke();
  ctx.globalAlpha = 0.8 * p;
  ctx.strokeStyle = ink.line;
  ctx.beginPath();
  ctx.arc(f.x, f.y, f.r * 0.9, 0, TAU);
  ctx.stroke();
  ctx.beginPath();
  for (let k = 0; k < TICKS; k++) {
    const a = f.spin + (k / TICKS) * TAU;
    const inner = f.r * (k % LONG_EVERY === 0 ? 0.865 : 0.915);
    const c = Math.cos(a);
    const s = Math.sin(a);
    ctx.moveTo(f.x + inner * c, f.y + inner * s);
    ctx.lineTo(f.x + f.r * 0.985 * c, f.y + f.r * 0.985 * s);
  }
  ctx.stroke();

  // The threads, from just off the core to each node.
  for (const n of f.nodes) {
    const at = nodeAt(f, n.angle);
    const c = Math.cos(n.angle);
    const s = Math.sin(n.angle);
    const from = { x: f.x + f.r * 0.14 * c, y: f.y + f.r * 0.14 * s };
    const to = { x: at.x - NODE_R * c, y: at.y - NODE_R * s };
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    if (n.lit) {
      ctx.setLineDash([]);
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = ink.glow;
      ctx.globalAlpha = 0.85 * p;
    } else {
      ctx.setLineDash([2, 6]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = ink.line;
      ctx.globalAlpha = 0.9 * p;
    }
    ctx.stroke();
    if (n.lit && f.spark !== null) {
      const t = ((f.spark % 1) + 1) % 1;
      ctx.globalAlpha = 0.3 * Math.sin(Math.PI * t) * p;
      ctx.fillStyle = ink.accent;
      ctx.beginPath();
      ctx.arc(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, 1.6, 0, TAU);
      ctx.fill();
    }
  }

  // The nodes, each its widget's sigil in a small ring: a lit one in its
  // thread's own light, a little under it, so the threads stay the
  // brightest thing in the circle.
  ctx.setLineDash([]);
  for (const n of f.nodes) {
    const at = nodeAt(f, n.angle);
    const colour = n.lit ? ink.glow : ink.dim;
    const alpha = (n.lit ? 0.7 : 0.4) * p;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(at.x, at.y, NODE_R, 0, TAU);
    ctx.stroke();
    const g = glyphs(n.name);
    if (!g) continue;
    ctx.save();
    ctx.translate(at.x - 12 * GLYPH_SCALE, at.y - 12 * GLYPH_SCALE);
    ctx.scale(GLYPH_SCALE, GLYPH_SCALE);
    // As Sigil.tsx draws them: round ends and corners.
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = colour;
    ctx.stroke(g.stroke);
    if (g.fill) {
      ctx.fillStyle = colour;
      ctx.fill(g.fill);
    }
    ctx.restore();
  }
  ctx.restore();
}

/** The nodes' sigils as SVG path data, in the 24-unit box of
 * src/divora/sigils/Sigil.tsx: its own `weather` and `desk`, every path as
 * it is there (its circle and its rounded rectangle written as paths), and
 * a clock of the scene's own, since the set has none. Stroked as the set
 * strokes them, 1.5 wide with round ends and corners. */
export const GLYPH_PATHS: Readonly<Record<NodeName, { stroke: string[]; fill: string[] }>> = {
  clock: {
    stroke: ["M20 12 A8 8 0 1 1 4 12 A8 8 0 1 1 20 12 Z", "M12 7.5 V12 L15 14"],
    fill: [],
  },
  weather: {
    stroke: [
      "M10.1 7.5 A2.6 2.6 0 1 1 4.9 7.5 A2.6 2.6 0 1 1 10.1 7.5 Z",
      "M7.5 2.5 V3.4 M2.5 7.5 H3.4 M3.95 3.95 L4.6 4.6 M11.05 3.95 L10.4 4.6 M3.95 11.05 L4.6 10.4",
      "M9.5 19 H17.5 A3.5 3.5 0 0 0 17.5 12 A4.5 4.5 0 0 0 9.8 13.2 A2.9 2.9 0 0 0 9.5 19 Z",
    ],
    fill: [],
  },
  desk: {
    stroke: [
      "M3.5 13.5 H20.5 M6 13.5 V20 M18 13.5 V20",
      "M14.1 8.5 H15.9 A0.6 0.6 0 0 1 16.5 9.1 V12.9 A0.6 0.6 0 0 1 15.9 13.5 H14.1 A0.6 0.6 0 0 1 13.5 12.9 V9.1 A0.6 0.6 0 0 1 14.1 8.5 Z",
    ],
    fill: ["M15 3.5 C16.3 5 16.3 6.3 15 7 C13.7 6.3 13.7 5 15 3.5 Z"],
  },
};

/** The glyphs as Path2D, made once, in a page that has Path2D. */
export function makeGlyphs(make: (d: string) => Path2D = (d) => new Path2D(d)): Glyphs {
  const made = new Map<NodeName, { stroke: Path2D; fill: Path2D | null }>();
  for (const name of NODES) {
    const { stroke, fill } = GLYPH_PATHS[name];
    made.set(name, { stroke: make(stroke.join(" ")), fill: fill.length ? make(fill.join(" ")) : null });
  }
  return (name) => made.get(name) ?? null;
}
