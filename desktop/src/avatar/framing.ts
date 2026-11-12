/** Where the camera stands so a model of any height fills the window. */
export interface Framing {
  /** World height the camera looks at. */
  centerY: number;
  /** Distance from the model along +Z (VRM 1.0 models face +Z). */
  distance: number;
}

/**
 * Frame the vertical span [bottom, top] in a perspective camera of `fovDeg`
 * (vertical), with `margin` of the span's height left free above and below.
 * Any VRoid export fits, whatever its height, because the span comes from
 * the model's own bounds.
 */
export function frameSpan(bottom: number, top: number, fovDeg: number, margin = 0.04): Framing {
  if (!(top > bottom)) throw new RangeError("frameSpan: top must be above bottom");
  if (!(fovDeg > 0 && fovDeg < 180)) throw new RangeError("frameSpan: fov out of range");
  const height = (top - bottom) * (1 + 2 * margin);
  const half = Math.tan((fovDeg * Math.PI) / 360);
  return { centerY: (top + bottom) / 2, distance: height / 2 / half };
}
