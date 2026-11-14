/**
 * His tint is tokens.css's colours, held in TS because he is drawn in WebGL:
 * every one is checked against the stylesheet, so syncing the tokens can't
 * leave him behind.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { OUTLINE, RIM, RIM_STRENGTH, tintFor } from "./tint";

const TOKENS = readFileSync(fileURLToPath(new URL("../divora/tokens.css", import.meta.url)), "utf8");

/** The declarations of the block whose selector is exactly `selector`. */
function block(selector: string): Record<string, string> {
  const at = TOKENS.indexOf(`${selector} {`);
  if (at < 0) throw new Error(`tokens.css has no ${selector} block`);
  const body = TOKENS.slice(TOKENS.indexOf("{", at) + 1, TOKENS.indexOf("}", at));
  return Object.fromEntries([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));
}

/** A token as the page would resolve it: the most specific block that sets it. */
function token(name: string, ...selectors: string[]): string {
  for (const s of selectors) {
    const v = block(s)[name];
    if (v !== undefined) return v;
  }
  throw new Error(`${name} isn't set by ${selectors.join(", ")}`);
}

const MOOD_SELECTOR = { ink: '[data-mood="ink"]', dusk: "", midnight: '[data-mood="midnight"]' } as const;

describe("his tint", () => {
  it("leans his outlines toward each mood's --text-lo", () => {
    for (const mood of ["ink", "dusk", "midnight"] as const) {
      const m = MOOD_SELECTOR[mood];
      expect(OUTLINE.dark[mood].toLowerCase(), `dark ${mood}`).toBe(token("--text-lo", `:root${m}`, ":root").toLowerCase());
      const light = m ? [`:root[data-theme="light"]${m}`, ':root[data-theme="light"]'] : [':root[data-theme="light"]'];
      expect(OUTLINE.light[mood].toLowerCase(), `light ${mood}`).toBe(token("--text-lo", ...light).toLowerCase());
    }
  });

  it("lights his rim in each accent's --indigo", () => {
    expect(RIM.brand.toLowerCase()).toBe(token("--indigo", ":root").toLowerCase());
    expect(RIM.abyssal.toLowerCase()).toBe(token("--indigo", ':root[data-accent="abyssal"]').toLowerCase());
    expect(RIM.ember.toLowerCase()).toBe(token("--indigo", ':root[data-accent="ember"]').toLowerCase());
  });

  it("picks by theme, mood and accent", () => {
    expect(tintFor({ theme: "light", mood: "midnight", accent: "ember" })).toEqual({
      outline: OUTLINE.light.midnight, rim: RIM.ember, rimStrength: RIM_STRENGTH.light,
    });
    expect(RIM_STRENGTH.light).toBeLessThan(RIM_STRENGTH.dark);
  });
});
