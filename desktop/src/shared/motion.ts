/**
 * The shared motion spec (plan 14): how the speech bubble appears, holds and
 * goes, scaled by the theme's motion factor (theme.ts: 0, 0.6 or 1).
 *
 * - Movement (scale, blur) takes `base × motion`, so at functional motion
 *   there is none.
 * - An opacity change never takes less than CROSSFADE_MS, at any motion:
 *   with no motion the bubble still crossfades rather than popping.
 * - A finished reply holds long enough to read: HOLD_MS at least,
 *   READ_MS_PER_CHAR a character, never more than HOLD_MAX_MS (the plan's
 *   "fades 6 s after done", lengthened for long replies; READ_MS_PER_CHAR = 0
 *   would give the plan's literal 6 s).
 *
 * On the avatar's page these are driven by the paced loop, never by CSS.
 */

/** The materialise: from this scale and blur to none, over this long. */
export const MATERIALISE_MS = 400;
export const MATERIALISE_SCALE_FROM = 0.6;
export const MATERIALISE_BLUR_FROM_PX = 8;
/** The fade once the hold is over. */
export const FADE_MS = 600;
/** The shortest any opacity change takes. */
export const CROSSFADE_MS = 330;
/** How long a finished reply stays. */
export const HOLD_MS = 6000;
export const READ_MS_PER_CHAR = 50;
export const HOLD_MAX_MS = 30_000;

function unit(v: number): number {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}

/** How long a movement of `base` ms takes at `motion`: 0 at functional. */
export function transformMs(base: number, motion: number): number {
  return base * unit(motion);
}

/** How long an opacity change of `base` ms takes at `motion`: never under
 * CROSSFADE_MS. */
export function opacityMs(base: number, motion: number): number {
  return Math.max(CROSSFADE_MS, base * unit(motion));
}

/** How long a reply of `chars` characters holds once done. */
export function holdMs(chars: number): number {
  const c = Number.isFinite(chars) ? Math.max(0, chars) : 0;
  return Math.min(HOLD_MAX_MS, Math.max(HOLD_MS, c * READ_MS_PER_CHAR));
}

/** Fast, then settling: for the materialise. */
export function easeOutCubic(t: number): number {
  return 1 - (1 - unit(t)) ** 3;
}

/** Slow at both ends: for opacity, which may turn back halfway. */
export function smoothstep(t: number): number {
  const u = unit(t);
  return u * u * (3 - 2 * u);
}
