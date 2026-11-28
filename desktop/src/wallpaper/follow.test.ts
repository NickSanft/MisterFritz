import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import { FADE_S, FOLLOW_S, FRITZ_CMD, FRITZ_EVENT, Follow, ease, parsePlace } from "./follow";

describe("where Fritz is, as the shell says", () => {
  it("is the shell's wire", () => {
    expect(FRITZ_EVENT).toBe(fixture.events.wallpaperFritz);
    expect(FRITZ_CMD).toBe(fixture.commands.wallpaperFritz.name);
  });

  it("is a place, null when he isn't on this monitor, or nothing", () => {
    expect(parsePlace({ x: 10, y: -20, size: 640, shown: true })).toEqual({ x: 10, y: -20, size: 640, shown: true });
    expect(parsePlace(null)).toBeNull();
    for (const bad of [undefined, [], 3, { x: 1.5, y: 0, size: 1, shown: true }, { x: 0, y: 0, size: 0, shown: true },
                       { x: 0, y: 0, size: 10, shown: "yes" }, { x: 0, y: 0, shown: true }]) {
      expect(parsePlace(bad), JSON.stringify(bad)).toBeUndefined();
    }
  });
});

describe("the circle following him", () => {
  it("comes in where he is, at the page's scale", () => {
    const f = new Follow();
    f.setScale(1.5);
    f.set({ x: 1500, y: 900, size: 960, shown: true });
    expect(f.step(0.1, false)).toBe(true);
    expect([f.x, f.y, f.size]).toEqual([1000, 600, 640]);
    expect(f.presence).toBeCloseTo(ease(0.1, FADE_S), 9);
  });

  it("eases after him as he moves, and settles", () => {
    const f = new Follow();
    f.set({ x: 100, y: 100, size: 600, shown: true });
    f.step(0, true);
    f.set({ x: 300, y: 100, size: 600, shown: true });
    f.step(0.1, false);
    expect(f.x).toBeCloseTo(100 + 200 * ease(0.1, FOLLOW_S), 9);
    let moving = true;
    for (let i = 0; i < 100 && moving; i++) moving = f.step(0.1, false);
    expect(moving).toBe(false);
    expect([f.x, f.presence]).toEqual([300, 1]);
  });

  it("fades out where he was when he is hidden, or goes to another monitor", () => {
    const f = new Follow();
    f.set({ x: 100, y: 200, size: 600, shown: true });
    f.step(0, true);
    f.set({ x: 100, y: 200, size: 600, shown: false });
    f.step(0.2, false);
    expect(f.presence).toBeCloseTo(1 - ease(0.2, FADE_S), 9);
    f.set(null);
    for (let i = 0; i < 100; i++) f.step(0.1, false);
    expect([f.x, f.y, f.presence]).toEqual([100, 200, 0]);
  });

  it("comes back where he is now, not sliding from where it faded", () => {
    const f = new Follow();
    f.set({ x: 100, y: 100, size: 600, shown: true });
    f.step(0, true);
    f.set(null);
    f.step(0, true);
    f.set({ x: 900, y: 100, size: 600, shown: true });
    f.step(0.1, false);
    expect(f.x).toBe(900);
  });

  it("is put there at once in a still scene", () => {
    const f = new Follow();
    f.set({ x: 50, y: 60, size: 400, shown: true });
    expect(f.step(0, true)).toBe(false);
    expect([f.x, f.y, f.size, f.presence]).toEqual([50, 60, 400, 1]);
    f.set({ x: 50, y: 60, size: 400, shown: false });
    expect(f.step(0, true)).toBe(false);
    expect(f.presence).toBe(0);
  });

  it("never moves in no time, and takes a bad scale as one", () => {
    expect(ease(0, 1)).toBe(0);
    expect(ease(-1, 1)).toBe(0);
    expect(ease(Number.NaN, 1)).toBe(0);
    const f = new Follow();
    f.setScale(Number.NaN);
    f.set({ x: 10, y: 10, size: 10, shown: true });
    f.step(0, true);
    expect(f.x).toBe(10);
  });
});
