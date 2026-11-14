/** Where the camera stands so a model of any height fills the window. */
export interface Framing {
  /** World height the camera looks at. */
  centerY: number;
  /** Distance from the model along +Z (VRM 1.0 models face +Z). */
  distance: number;
}

/** The share of his height left free above and below him. */
export const MARGIN = 0.04;

/** The share of his window kept clear over his head, for the speech bubble:
 * he fills the rest. A 640-pixel window frames him as PR 2's 460 did. */
export const HEADROOM = 0.28;

/**
 * Frame the vertical span [bottom, top] in a perspective camera of `fovDeg`
 * (vertical), with `margin` of the span's height left free above and below,
 * and the top `headroom` share of the view left free above that. Any VRoid
 * export fits, whatever its height, because the span comes from the model's
 * own bounds.
 */
export function frameSpan(bottom: number, top: number, fovDeg: number, margin = MARGIN, headroom = 0): Framing {
  if (!(top > bottom)) throw new RangeError("frameSpan: top must be above bottom");
  if (!(fovDeg > 0 && fovDeg < 180)) throw new RangeError("frameSpan: fov out of range");
  if (!(headroom >= 0 && headroom < 0.9)) throw new RangeError("frameSpan: headroom out of range");
  const span = top - bottom;
  const view = (span * (1 + 2 * margin)) / (1 - headroom);
  const half = Math.tan((fovDeg * Math.PI) / 360);
  return { centerY: bottom - margin * span + view / 2, distance: view / 2 / half };
}
