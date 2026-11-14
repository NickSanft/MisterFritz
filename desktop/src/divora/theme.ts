/**
 * The theme contract, on every window: the attributes the lifted DivoraVoice
 * tokens (tokens.css) key on, from `settings.theme`, and the motion level the
 * page's own animation (the paced loop) follows.
 *
 * Motion is a level, never an inline `--motion`: tokens.css maps
 * `data-motion` to `--motion` (0, 0.6, 1), and an inline value or the
 * attribute both outrank its reduced-motion rule, so Windows asking for less
 * motion is worked out here, in one place, for CSS and the paced loop alike:
 * asked, and the owner hasn't kept his motion (`motionUser`), it is
 * functional. The continuous `theme.motion` snaps to a level as DivoraVoice
 * does (src/App.tsx 168-177, src/overlay.tsx 111-124 at 62ed6a5).
 *
 * DivoraVoice's default mood (violet, the plan's "dusk") and accent ("brand")
 * are its plain :root, so they set no attribute.
 */
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { afterAsking } from "../shared/asking";
import { GET_SETTINGS_CMD, SETTINGS_EVENT, parseTheme, type Theme } from "../shared/settings";

export type MotionLevel = "functional" | "ambient" | "rich";

/** Each level's factor, as tokens.css sets `--motion` for it. */
export const MOTION_LEVELS: Readonly<Record<MotionLevel, number>> = { functional: 0, ambient: 0.6, rich: 1 };

/** The level a `theme.motion` (0 to 1) snaps to: none at all is
 * functional, anything below 0.8 ambient, the rest rich. */
export function motionLevel(m: number): MotionLevel {
  if (!(m > 0)) return "functional";
  return m < 0.8 ? "ambient" : "rich";
}

export interface Motion {
  level: MotionLevel;
  /** The paced loop's multiplier: 0 stops all but blinking and breathing. */
  factor: number;
}

/** The motion the windows show: functional while Windows asks for less,
 * unless the owner has kept his. */
export function effectiveMotion(t: Pick<Theme, "motion" | "motionUser">, reduced: boolean): Motion {
  const level = reduced && !t.motionUser ? "functional" : motionLevel(t.motion);
  return { level, factor: MOTION_LEVELS[level] };
}

type ThemeAttribute = "data-theme" | "data-mood" | "data-accent" | "data-motion" | "data-motion-user";

/** The attributes for `t`: null removes one. */
export function themeAttributes(t: Theme, reduced: boolean): Record<ThemeAttribute, string | null> {
  return {
    "data-theme": t.theme === "light" ? "light" : null,
    "data-mood": t.mood === "ink" || t.mood === "midnight" ? t.mood : null,
    "data-accent": t.accent === "abyssal" || t.accent === "ember" ? t.accent : null,
    "data-motion": effectiveMotion(t, reduced).level,
    "data-motion-user": t.motionUser ? "true" : null,
  };
}

/** Set `t`'s attributes on `root` (the page's <html>), removing stale ones.
 * Never an inline `--motion` (see above). */
export function applyTheme(root: Pick<Element, "setAttribute" | "removeAttribute">, t: Theme, reduced: boolean): void {
  for (const [name, value] of Object.entries(themeAttributes(t, reduced))) {
    if (value === null) root.removeAttribute(name);
    else root.setAttribute(name, value);
  }
}

export interface ReducedMotion {
  now(): boolean;
  /** Called on every change; returns the unsubscribe. */
  onChange(cb: () => void): () => void;
}

/** Whether Windows asks for less motion (WebView2 follows its "Animation
 * effects" switch), and when that changes. */
export function reducedMotion(win: Pick<Window, "matchMedia"> = window): ReducedMotion {
  const q = win.matchMedia("(prefers-reduced-motion: reduce)");
  return {
    now: () => q.matches,
    onChange(cb) {
      q.addEventListener("change", cb);
      return () => q.removeEventListener("change", cb);
    },
  };
}

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** The theme in a settings payload, or null. */
export function themeOf(settings: unknown): Theme | null {
  return parseTheme(obj(settings)?.theme);
}

export interface ThemeWatchDeps {
  listen(cb: (settings: unknown) => void): Promise<() => void>;
  ask(): Promise<unknown>;
  reduced: ReducedMotion;
}

const tauriDeps = (): ThemeWatchDeps => ({
  listen: (cb) => listen<unknown>(SETTINGS_EVENT, (e) => cb(e.payload)),
  ask: () => invoke<unknown>(GET_SETTINGS_CMD),
  reduced: reducedMotion(),
});

/** Keep `root` in the owner's theme: from every `settings:changed`, from the
 * settings asked for once listening (unless an event came first), and again
 * whenever Windows' reduced-motion switch changes. `onTheme` hears every
 * application, with the motion to follow. Resolves once listening, without
 * waiting for the answer, so a page can start without it. */
export async function watchTheme(
  root: Pick<Element, "setAttribute" | "removeAttribute">,
  onTheme: (t: Theme, motion: Motion) => void,
  deps: ThemeWatchDeps = tauriDeps(),
): Promise<() => void> {
  let current: Theme | null = null;
  let events = 0;
  const apply = () => {
    if (!current) return;
    const reduced = deps.reduced.now();
    applyTheme(root, current, reduced);
    onTheme(current, effectiveMotion(current, reduced));
  };
  const unlisten = await deps.listen((settings) => {
    const t = themeOf(settings);
    if (!t) return;
    events += 1;
    current = t;
    apply();
  });
  const offReduced = deps.reduced.onChange(apply);
  const before = events;
  void deps
    .ask()
    .then((settings) => {
      const t = themeOf(settings);
      if (!t) return;
      const next = afterAsking<Theme | null>(t, before, events, current);
      if (next !== current) {
        current = next;
        apply();
      }
    })
    .catch(() => {
      // No answer: the next settings event brings the theme.
    });
  return () => {
    unlisten();
    offReduced();
  };
}
