/**
 * The theme on the control window: the attributes every window takes
 * (src/divora/theme.ts), and the parchment grain, which among the windows
 * over the desktop is this one's alone (plan 14: never on Fritz; the study
 * scene behind the icons draws a grain of its own, src/wallpaper/grain.ts).
 */
import { applyTheme } from "../divora/theme";
import type { Theme } from "../shared/settings";

export function wearTheme(doc: Pick<Document, "documentElement" | "body">, t: Theme, reduced: boolean): void {
  applyTheme(doc.documentElement, t, reduced);
  doc.body.classList.toggle("grain", t.grain);
}
