/**
 * The control window's look, from `settings.theme`: the attributes the
 * lifted DivoraVoice tokens (src/divora/tokens.css) key on. DivoraVoice's
 * default mood (violet, the plan's "dusk") and accent ("brand") are its
 * plain :root, so they set no attribute. PR 3's theme.ts takes this over,
 * with motion and the avatar's tint.
 */

export interface ThemeSettings {
  theme: string;
  mood: string;
  accent: string;
}

/** The attributes for `t`: null removes one. */
export function themeAttributes(t: ThemeSettings): Record<"data-theme" | "data-mood" | "data-accent", string | null> {
  return {
    "data-theme": t.theme === "light" ? "light" : null,
    "data-mood": t.mood === "ink" || t.mood === "midnight" ? t.mood : null,
    "data-accent": t.accent === "abyssal" || t.accent === "ember" ? t.accent : null,
  };
}

export function applyTheme(root: HTMLElement, t: ThemeSettings): void {
  for (const [name, value] of Object.entries(themeAttributes(t))) {
    if (value === null) root.removeAttribute(name);
    else root.setAttribute(name, value);
  }
}
