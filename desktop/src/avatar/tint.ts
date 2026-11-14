/**
 * Fritz takes the theme too (plan 14 § 7 Q10: a tint, nothing more): his
 * outlines drift toward the mood's quiet text colour, and a faint rim light
 * in the accent's colour. The colours are tokens.css's (`--text-lo` for the
 * mood and theme, `--indigo` for the accent), held here because the page
 * paints him in WebGL, not CSS; tint.test.ts checks every one against
 * tokens.css, so a sync of the tokens can't leave them behind.
 */
import type { Theme } from "../shared/settings";

export interface Tint {
  /** sRGB hex: what his outlines lean toward. */
  outline: string;
  /** sRGB hex: the rim light's colour. */
  rim: string;
  /** How much of the rim colour is added to the model's own rim. */
  rimStrength: number;
}

/** `--text-lo`, by theme then mood. */
export const OUTLINE: Readonly<Record<Theme["theme"], Record<Theme["mood"], string>>> = {
  dark: { ink: "#75694f", dusk: "#6E6590", midnight: "#5d6178" },
  light: { ink: "#837560", dusk: "#6f6690", midnight: "#6b7290" },
};

/** `--indigo`, by accent (the light theme keeps the accent's). */
export const RIM: Readonly<Record<Theme["accent"], string>> = {
  brand: "#6D5BF0",
  abyssal: "#4ABDE0",
  ember: "#F2A857",
};

/** A light page shows a rim more, so it gets less. */
export const RIM_STRENGTH: Readonly<Record<Theme["theme"], number>> = { dark: 0.25, light: 0.15 };

export function tintFor(t: Pick<Theme, "theme" | "mood" | "accent">): Tint {
  return { outline: OUTLINE[t.theme][t.mood], rim: RIM[t.accent], rimStrength: RIM_STRENGTH[t.theme] };
}
