/**
 * The parchment grain over the study scene, while the owner's grain switch
 * is on (Settings › Appearance): a still, very faint noise in the mood's
 * own blend and strength (tokens.css `--grain-blend`, `--grain-opacity`),
 * a third as strong as the control window's, one noise pixel to a device
 * pixel. It also hides the slight banding a big dark gradient shows. The
 * noise is the scene's own, made from a fixed seed, so it is the same
 * every time and never moves.
 */

/** How strong the scene's grain is, against the token's. */
export const GRAIN_SCALE = 1 / 3;
/** The noise tile's side, in device pixels. */
export const TILE = 128;

/** A small, fixed pseudo-random sequence (mulberry32): the same noise
 * every time. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A `size`² tile of grey noise around the middle grey, as RGBA, opaque:
 * blended in, a middle grey changes nothing, so it only speckles. */
export function noiseTile(size = TILE, seed = 0x5eed): Uint8ClampedArray<ArrayBuffer> {
  const next = rng(seed);
  const out = new Uint8ClampedArray(new ArrayBuffer(size * size * 4));
  for (let i = 0; i < size * size; i++) {
    // Two draws averaged: most near the middle, few far out.
    const v = Math.round(128 + ((next() + next()) / 2 - 0.5) * 96);
    out.set([v, v, v, 255], i * 4);
  }
  return out;
}

const BLENDS = new Set(["overlay", "multiply", "soft-light", "screen"]);

export interface Grain {
  blend: GlobalCompositeOperation;
  opacity: number;
}

/** The grain the mood wants, from its tokens; null when they say none or
 * aren't there. */
export function grainOf(style: Pick<CSSStyleDeclaration, "getPropertyValue">): Grain | null {
  const blend = style.getPropertyValue("--grain-blend").trim();
  const opacity = Number.parseFloat(style.getPropertyValue("--grain-opacity"));
  if (!BLENDS.has(blend) || !(opacity > 0) || opacity > 1) return null;
  return { blend: blend as GlobalCompositeOperation, opacity: opacity * GRAIN_SCALE };
}
