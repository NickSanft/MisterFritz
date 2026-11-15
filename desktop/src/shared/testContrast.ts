/**
 * The design tokens' colours, for tests: each mood and theme's value of a
 * token, read from src/divora/tokens.css, and WCAG's contrast between two.
 * Test-only (like testClock.ts): no page imports it.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const TOKENS = readFileSync(fileURLToPath(new URL("../divora/tokens.css", import.meta.url)), "utf8").replace(/\r\n/g, "\n");

export function block(selector: string): Record<string, string> {
  const at = TOKENS.indexOf(`${selector} {`);
  if (at < 0) throw new Error(`tokens.css has no ${selector} block`);
  const body = TOKENS.slice(TOKENS.indexOf("{", at) + 1, TOKENS.indexOf("}", at));
  return Object.fromEntries([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));
}

export function token(name: string, selectors: string[]): string {
  for (const s of selectors) {
    const v = block(s)[name];
    if (v !== undefined) return v;
  }
  throw new Error(`${name} isn't set by ${selectors.join(", ")}`);
}

export function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`not an opaque hex colour: ${hex}`);
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(m[1].slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

/** Each mood and theme's blocks, most specific first. */
export const COMBOS: Record<string, string[]> = {
  "dusk dark": [":root"],
  "ink dark": [':root[data-mood="ink"]', ":root"],
  "midnight dark": [':root[data-mood="midnight"]', ":root"],
  "dusk light": [':root[data-theme="light"]', ":root"],
  "ink light": [':root[data-theme="light"][data-mood="ink"]', ':root[data-theme="light"]', ':root[data-mood="ink"]', ":root"],
  "midnight light": [
    ':root[data-theme="light"][data-mood="midnight"]', ':root[data-theme="light"]', ':root[data-mood="midnight"]', ":root",
  ],
};
