import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import { parseTheme, type Theme } from "../shared/settings";
import {
  MOTION_LEVELS, applyTheme, effectiveMotion, motionLevel, themeAttributes, watchTheme, type ReducedMotion,
} from "./theme";

const defaults = fixture.themeDefaults as Theme;

/** A root that records its attributes, and anything set inline. */
function fakeRoot() {
  const attrs = new Map<string, string>();
  const inline: string[] = [];
  return {
    attrs,
    inline,
    setAttribute: (n: string, v: string) => void attrs.set(n, v),
    removeAttribute: (n: string) => void attrs.delete(n),
    style: { setProperty: (n: string) => void inline.push(n) },
  };
}

describe("the theme contract", () => {
  it("reads the shell's defaults, and nothing it doesn't know", () => {
    expect(parseTheme(defaults)).toEqual(defaults);
    expect(parseTheme({ ...defaults, mood: "violet" })).toBeNull();
    expect(parseTheme({ ...defaults, accent: "teal" })).toBeNull();
    expect(parseTheme({ ...defaults, theme: "sepia" })).toBeNull();
    expect(parseTheme({ ...defaults, motion: 1.5 })).toBeNull();
    expect(parseTheme({ ...defaults, motion: "0.6" })).toBeNull();
    expect(parseTheme({ ...defaults, grain: undefined })).toBeNull();
    expect(parseTheme({ ...defaults, motionUser: 1 })).toBeNull();
    expect(parseTheme(null)).toBeNull();
  });

  it("snaps motion to the levels tokens.css knows", () => {
    expect(MOTION_LEVELS).toEqual(fixture.motionLevels);
    for (const [m, level] of fixture.motionSnap as [number, string][]) expect(motionLevel(m), String(m)).toBe(level);
  });

  it("sets every mood, accent and theme as the tokens key on them", () => {
    for (const theme of ["dark", "light"] as const) {
      for (const mood of ["ink", "dusk", "midnight"] as const) {
        for (const accent of ["brand", "abyssal", "ember"] as const) {
          const a = themeAttributes({ ...defaults, theme, mood, accent }, false);
          expect(a["data-theme"]).toBe(theme === "light" ? "light" : null);
          // Dusk and brand are DivoraVoice's plain :root.
          expect(a["data-mood"]).toBe(mood === "dusk" ? null : mood);
          expect(a["data-accent"]).toBe(accent === "brand" ? null : accent);
        }
      }
    }
  });

  it("goes functional when Windows asks for less motion, unless he keeps his", () => {
    const rich = { ...defaults, motion: 1 };
    expect(effectiveMotion(rich, false)).toEqual({ level: "rich", factor: 1 });
    expect(effectiveMotion(rich, true)).toEqual({ level: "functional", factor: 0 });
    expect(themeAttributes(rich, true)["data-motion"]).toBe("functional");
    expect(themeAttributes(rich, true)["data-motion-user"]).toBeNull();
    const kept = { ...rich, motionUser: true };
    expect(effectiveMotion(kept, true)).toEqual({ level: "rich", factor: 1 });
    expect(themeAttributes(kept, true)["data-motion"]).toBe("rich");
    expect(themeAttributes(kept, true)["data-motion-user"]).toBe("true");
    expect(effectiveMotion({ ...defaults, motion: 0 }, false)).toEqual({ level: "functional", factor: 0 });
  });

  it("removes what no longer applies, and never sets --motion inline", () => {
    const root = fakeRoot();
    applyTheme(root, { ...defaults, theme: "light", mood: "midnight", accent: "ember", motionUser: true }, false);
    expect(Object.fromEntries(root.attrs)).toEqual({
      "data-theme": "light", "data-mood": "midnight", "data-accent": "ember", "data-motion": "ambient",
      "data-motion-user": "true",
    });
    applyTheme(root, { ...defaults, mood: "dusk" }, false);
    expect(Object.fromEntries(root.attrs)).toEqual({ "data-motion": "ambient" });
    expect(root.inline).toEqual([]);
  });
});

describe("watching the theme", () => {
  function deps(answer: unknown) {
    let hear: (s: unknown) => void = () => {};
    let changed: () => void = () => {};
    let reduced = false;
    let release: (v: unknown) => void = () => {};
    const asked = new Promise((r) => (release = r));
    const r: ReducedMotion = { now: () => reduced, onChange: (cb) => ((changed = cb), () => {}) };
    return {
      deps: { listen: async (cb: (s: unknown) => void) => ((hear = cb), () => {}), ask: () => asked, reduced: r },
      hear: (s: unknown) => hear(s),
      answer: async () => {
        release(answer);
        await asked;
        await Promise.resolve();
      },
      setReduced: (on: boolean) => {
        reduced = on;
        changed();
      },
    };
  }

  it("takes the answer when nothing was heard while asking", async () => {
    const d = deps({ theme: { ...defaults, mood: "midnight" } });
    const root = fakeRoot();
    const seen: string[] = [];
    await watchTheme(root, (t) => seen.push(t.mood), d.deps);
    await d.answer();
    expect(seen).toEqual(["midnight"]);
    expect(root.attrs.get("data-mood")).toBe("midnight");
  });

  it("keeps an event heard while asking over the older answer", async () => {
    const d = deps({ theme: { ...defaults, mood: "midnight" } });
    const root = fakeRoot();
    const seen: string[] = [];
    await watchTheme(root, (t) => seen.push(t.mood), d.deps);
    d.hear({ theme: { ...defaults, mood: "dusk" } });
    await d.answer();
    expect(seen).toEqual(["dusk"]);
    expect(root.attrs.has("data-mood")).toBe(false);
  });

  it("follows Windows' reduced-motion switch, and ignores a theme it can't read", async () => {
    const d = deps({ theme: { ...defaults, motion: 1 } });
    const root = fakeRoot();
    const factors: number[] = [];
    await watchTheme(root, (_t, m) => factors.push(m.factor), d.deps);
    await d.answer();
    d.setReduced(true);
    d.hear({ theme: { ...defaults, mood: "violet" } });
    expect(factors).toEqual([1, 0]);
    expect(root.attrs.get("data-motion")).toBe("functional");
  });
});
