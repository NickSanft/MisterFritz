/**
 * His face for his mood: the owner's mapping, eased in over IN_MS and out
 * over OUT_MS at any frame rate; and the page's use of it, read from
 * main.ts (which runs only in his window).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { FACES, Face, IN_MS, OUT_MS } from "./face";
import type { MoodName } from "../shared/protocol";

/** Each expression's weight now, by its first name. */
function weights(f: Face): Record<string, number> {
  return Object.fromEntries(f.faces().map((x) => [x.names[0], x.weight]));
}

function run(f: Face, ms: number, step = 1000 / 30): void {
  for (let t = 0; t < ms - 1e-9; t += step) f.tick(Math.min(step, ms - t));
}

describe("his face for his mood", () => {
  it("is the owner's mapping: subtle, his resting face for dry", () => {
    expect(FACES).toEqual({
      dry: null,
      pleased: { names: ["happy"], weight: 0.4 },
      alarmed: { names: ["surprised", "Surprised"], weight: 0.5 },
      weary: { names: ["sad"], weight: 0.4 },
    });
    expect([IN_MS, OUT_MS]).toEqual([300, 1000]);
  });

  it("starts at his resting face, every expression at nothing", () => {
    const f = new Face();
    expect(f.mood).toBeNull();
    expect(weights(f)).toEqual({ happy: 0, surprised: 0, sad: 0 });
  });

  it("eases a mood in over IN_MS, never past its weight", () => {
    const f = new Face();
    f.set("pleased");
    run(f, IN_MS / 2);
    expect(weights(f).happy).toBeCloseTo(0.2, 6);
    run(f, IN_MS / 2 - 1);
    expect(weights(f).happy).toBeLessThan(0.4);
    run(f, 1);
    expect(weights(f).happy).toBeCloseTo(0.4, 9);
    run(f, 5000);
    expect(weights(f).happy).toBe(0.4);
    expect(weights(f)).toMatchObject({ surprised: 0, sad: 0 });
  });

  it("and out over OUT_MS, to exactly nothing", () => {
    const f = new Face();
    f.set("weary");
    run(f, IN_MS);
    f.set(null);
    run(f, OUT_MS / 2);
    expect(weights(f).sad).toBeCloseTo(0.2, 6);
    run(f, OUT_MS / 2);
    expect(weights(f).sad).toBe(0);
  });

  it("is the same at any frame rate", () => {
    const a = new Face();
    const b = new Face();
    a.set("alarmed");
    b.set("alarmed");
    run(a, 200, 10);
    run(b, 200, 50);
    expect(weights(a).surprised).toBeCloseTo(weights(b).surprised, 9);
  });

  it("a new mood eases the old one out as it comes in", () => {
    const f = new Face();
    f.set("pleased");
    run(f, IN_MS);
    f.set("alarmed");
    run(f, IN_MS);
    const w = weights(f);
    expect(w.surprised).toBeCloseTo(0.5, 9);
    // 300 of the 1000 ms out: 0.7 of the way still in, eased.
    expect(w.happy).toBeCloseTo(0.4 * (0.7 * 0.7 * (3 - 2 * 0.7)), 6);
  });

  it("dry, or none, is his resting face: the others go, and it sets nothing of its own", () => {
    for (const rest of ["dry", null] as const) {
      const f = new Face();
      f.set("pleased");
      run(f, IN_MS);
      f.set(rest);
      expect(f.mood).toBe(rest);
      run(f, OUT_MS);
      expect(weights(f)).toEqual({ happy: 0, surprised: 0, sad: 0 });
      expect(f.faces()).toHaveLength(3);
    }
  });

  it("turns back from where it was, with no jump", () => {
    const f = new Face();
    f.set("pleased");
    run(f, IN_MS / 2);
    const half = weights(f).happy;
    f.set(null);
    f.tick(1);
    expect(weights(f).happy).toBeLessThan(half);
    expect(half - weights(f).happy).toBeLessThan(0.01);
  });

  it("takes no step for a bad frame time, and goes on as before after one", () => {
    for (const bad of [Number.NaN, -50, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0]) {
      const f = new Face();
      f.set("pleased");
      run(f, IN_MS / 2);
      const was = weights(f).happy;
      f.tick(bad);
      expect(weights(f).happy, String(bad)).toBe(was);
      run(f, IN_MS / 2);
      expect(weights(f).happy, `${bad}, then on`).toBeCloseTo(0.4, 9);
    }
  });

  it("shows nothing for a mood it doesn't know", () => {
    const f = new Face();
    f.set("smug" as MoodName);
    expect(f.mood).toBeNull();
    run(f, IN_MS);
    expect(weights(f)).toEqual({ happy: 0, surprised: 0, sad: 0 });
  });
});

describe("the page's use of his face", () => {
  const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8").replace(/\r\n/g, "\n");
  const tick = (() => {
    const at = main.indexOf("const loop = new PacedLoop((dt) => {");
    return main.slice(at, main.indexOf("\n}, FPS);", at));
  })();

  it("hears the brain's mood", () => {
    expect(main).toContain('for (const type of ["state", "token", "progress", "done", "error", "stt", "mood"] as const) {');
  });

  it("shows the shown turn's mood while its bubble stands, from the paced tick", () => {
    expect(tick).toContain("  face.set(bubbleClock.standing ? turn.mood : null);\n  face.tick(dt * 1000);\n");
    expect(main.split("face.set(").length - 1).toBe(1);
    expect(main.split("face.tick(").length - 1).toBe(1);
  });

  it("sets it on him before he is posed, whatever the motion", () => {
    const set = tick.indexOf("actor.setFace(face.faces());");
    expect(set).toBeGreaterThan(tick.indexOf("face.tick(dt * 1000);"));
    expect(set).toBeLessThan(tick.indexOf("actor?.update(dt);"));
    expect(tick).not.toContain("motionFactor");
  });
});
