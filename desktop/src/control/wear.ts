/**
 * The theme on the control window: the attributes every window takes
 * (src/divora/theme.ts), and the parchment grain, which is this window's
 * alone (plan 14: never on Fritz or anything drawn over the desktop).
 */
import { applyTheme } from "../divora/theme";
import type { Theme } from "../shared/settings";

export function wearTheme(doc: Pick<Document, "documentElement" | "body">, t: Theme, reduced: boolean): void {
  applyTheme(doc.documentElement, t, reduced);
  doc.body.classList.toggle("grain", t.grain);
}
