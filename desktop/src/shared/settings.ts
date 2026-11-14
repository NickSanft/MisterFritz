/**
 * The settings every page reads, as the shell sends them (settings.rs, in
 * camelCase): the event that carries them on every change, the command that
 * asks for them, and the parts more than one page uses, read strictly.
 */

/** Every window hears the whole settings on every change. */
export const SETTINGS_EVENT = "settings:changed";
/** The settings now, for a page that starts after the last event. */
export const GET_SETTINGS_CMD = "get_settings";

export type ThemeName = "dark" | "light";
export type Mood = "ink" | "dusk" | "midnight";
export type Accent = "brand" | "abyssal" | "ember";

/** `settings.theme`. Plan 14 § 7 Q9's defaults are the shell's
 * (`Theme::default`), pinned to wire.fixture.json's `themeDefaults`. */
export interface Theme {
  theme: ThemeName;
  mood: Mood;
  accent: Accent;
  /** 0 to 1; snapped to a level (theme.ts `motionLevel`). */
  motion: number;
  /** Parchment grain, in the control window only. */
  grain: boolean;
  /** Keep his motion when Windows asks for less. */
  motionUser: boolean;
}

const THEMES: readonly ThemeName[] = ["dark", "light"];
const MOODS: readonly Mood[] = ["ink", "dusk", "midnight"];
const ACCENTS: readonly Accent[] = ["brand", "abyssal", "ember"];

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** `settings.theme` from the wire, or null when it isn't one: an unknown
 * name, a motion outside 0 to 1, a missing flag. The shell puts a bad value
 * back to its default at load and refuses one from a window, so null means
 * a shell and page out of step, not an owner's typo. */
export function parseTheme(v: unknown): Theme | null {
  const t = obj(v);
  if (!t) return null;
  if (!THEMES.includes(t.theme as ThemeName)) return null;
  if (!MOODS.includes(t.mood as Mood)) return null;
  if (!ACCENTS.includes(t.accent as Accent)) return null;
  if (typeof t.motion !== "number" || !(t.motion >= 0 && t.motion <= 1)) return null;
  if (typeof t.grain !== "boolean" || typeof t.motionUser !== "boolean") return null;
  return {
    theme: t.theme as ThemeName,
    mood: t.mood as Mood,
    accent: t.accent as Accent,
    motion: t.motion,
    grain: t.grain,
    motionUser: t.motionUser,
  };
}
