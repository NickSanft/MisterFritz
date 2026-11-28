import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GRAIN_SCALE, TILE, grainOf, noiseTile, rng } from "./grain";

const here = path.dirname(fileURLToPath(import.meta.url));

describe("the grain's noise", () => {
  it("is the same every time, and differs by seed", () => {
    expect(noiseTile(16)).toEqual(noiseTile(16));
    expect(noiseTile(16, 1)).not.toEqual(noiseTile(16, 2));
    const r = rng(7);
    const xs = Array.from({ length: 1000 }, r);
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...xs)).toBeLessThan(1);
  });

  it("is opaque grey about the middle, so blended it only speckles", () => {
    const t = noiseTile();
    expect(t.length).toBe(TILE * TILE * 4);
    let sum = 0;
    let lo = 255;
    let hi = 0;
    for (let i = 0; i < t.length; i += 4) {
      expect(t[i]).toBe(t[i + 1]);
      expect(t[i]).toBe(t[i + 2]);
      expect(t[i + 3]).toBe(255);
      sum += t[i]!;
      lo = Math.min(lo, t[i]!);
      hi = Math.max(hi, t[i]!);
    }
    expect(sum / (TILE * TILE)).toBeGreaterThan(124);
    expect(sum / (TILE * TILE)).toBeLessThan(132);
    expect(lo).toBeGreaterThanOrEqual(80);
    expect(hi).toBeLessThanOrEqual(176);
  });
});

describe("the grain the mood wants", () => {
  const style = (t: Record<string, string | undefined>) => ({ getPropertyValue: (n: string) => t[n] ?? "" });

  it("is its own blend, a third as strong", () => {
    expect(grainOf(style({ "--grain-blend": " overlay", "--grain-opacity": "0.5" })))
      .toEqual({ blend: "overlay", opacity: 0.5 * GRAIN_SCALE });
    expect(GRAIN_SCALE).toBeCloseTo(1 / 3, 9);
    expect(grainOf(style({ "--grain-blend": "multiply", "--grain-opacity": "0.28" }))?.blend).toBe("multiply");
  });

  it("is none for a blend it doesn't know or no strength", () => {
    for (const t of [{ "--grain-blend": "difference", "--grain-opacity": "0.5" }, { "--grain-blend": "overlay" },
                     { "--grain-blend": "overlay", "--grain-opacity": "0" }, { "--grain-blend": "overlay", "--grain-opacity": "2" }]) {
      expect(grainOf(style(t)), JSON.stringify(t)).toBeNull();
    }
  });

  it("reads tokens tokens.css sets, dark and light", () => {
    const css = readFileSync(path.join(here, "..", "divora", "tokens.css"), "utf8");
    expect(css).toContain("--grain-blend: overlay;");
    expect(css).toContain("--grain-blend: multiply;");
    expect(css).toContain("--grain-opacity:");
  });
});
