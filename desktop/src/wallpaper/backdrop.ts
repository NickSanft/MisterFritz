/**
 * The study scene's backdrop (plan 14, PR 12): the owner's mood's
 * `--bg`/`--surface` ramp as one soft radial light that drifts very slowly
 * over the monitor. It is drawn on a canvas inside the paced loop; with
 * motion at functional it rests where it is, one still frame, drawn again
 * only when the page is resized, seen again or given new colours
 * (scene.ts).
 */

/** The colours the backdrop is drawn in, from the page's own tokens. */
export interface Ramp {
  /** The page's colour: the light's edge, and everything beyond it. */
  bg: string;
  /** Halfway out. */
  mid: string;
  /** The light's centre. */
  core: string;
}

/** The ramp from the tokens as the page wears them now: a light theme's
 * lightest surface at the centre, a dark one's third. Null while a token
 * is missing (the stylesheet hasn't applied). */
export function rampOf(style: Pick<CSSStyleDeclaration, "getPropertyValue">, light: boolean): Ramp | null {
  const read = (name: string): string => style.getPropertyValue(name).trim();
  const bg = read("--bg");
  const mid = read("--surface-1");
  const core = read(light ? "--surface-0" : "--surface-2");
  return bg && mid && core ? { bg, mid, core } : null;
}

/** How long the light takes to wander across and back, in seconds of
 * scene time, sideways and up and down: slow enough that no one sees it
 * move, only that it has. Unequal, so its path never repeats exactly. */
export const DRIFT_S = { x: 260, y: 370 } as const;

/** Where the light rests, and how far it wanders, as fractions of the
 * page: a little above the middle. */
const REST = { x: 0.5, y: 0.42 } as const;
const WANDER = { x: 0.16, y: 0.1 } as const;

export interface Glow {
  /** The centre, as fractions of the page's width and height. */
  x: number;
  y: number;
}

/** Where the light is `t` seconds of scene time in: at rest at 0. Pure. */
export function glowAt(t: number): Glow {
  const s = Number.isFinite(t) ? t : 0;
  return {
    x: REST.x + WANDER.x * Math.sin((2 * Math.PI * s) / DRIFT_S.x),
    y: REST.y + WANDER.y * Math.sin((2 * Math.PI * s) / DRIFT_S.y),
  };
}

/** The light's reach: past the far corners from anywhere it wanders. */
export function reach(width: number, height: number): number {
  return Math.hypot(width, height) * 0.85;
}

/** The part of a 2D context the backdrop uses, so tests can watch it. */
export interface Painter {
  fillStyle: string | CanvasGradient | CanvasPattern;
  fillRect(x: number, y: number, w: number, h: number): void;
  createRadialGradient(x0: number, y0: number, r0: number, x1: number, y1: number, r1: number): CanvasGradient;
}

/** Paint the backdrop over all of a `width` × `height` page (CSS pixels):
 * the page's colour, then the light at `glow`. */
export function paint(ctx: Painter, width: number, height: number, ramp: Ramp, glow: Glow): void {
  ctx.fillStyle = ramp.bg;
  ctx.fillRect(0, 0, width, height);
  const cx = glow.x * width;
  const cy = glow.y * height;
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, reach(width, height));
  g.addColorStop(0, ramp.core);
  g.addColorStop(0.42, ramp.mid);
  g.addColorStop(1, ramp.bg);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);
}
