import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import { COVERED_CMD, COVERED_EVENT, SCENE_FPS, SceneRun, drawing, type Loop } from "./scene";

class FakeLoop implements Loop {
  running = false;
  starts = 0;
  start(): void {
    if (!this.running) this.starts += 1;
    this.running = true;
  }
  stop(): void {
    this.running = false;
  }
}

function scene() {
  const loop = new FakeLoop();
  const painted: number[] = [];
  const run = new SceneRun((t) => void painted.push(t), loop);
  return { loop, painted, run };
}

describe("the study scene's wire", () => {
  it("is the shell's", () => {
    expect(COVERED_EVENT).toBe(fixture.events.wallpaperCovered);
    expect(COVERED_CMD).toBe(fixture.commands.wallpaperCovered.name);
    expect(fixture.commands.wallpaperCovered.args).toEqual([]);
  });

  it("is drawn slowly", () => {
    expect(SCENE_FPS).toBeGreaterThan(0);
    expect(SCENE_FPS).toBeLessThanOrEqual(15);
  });
});

describe("when the study scene draws", () => {
  it("runs while seen and moving, rests at functional, and stops when no one can see it", () => {
    expect(drawing(true, false, 0.6)).toBe("loop");
    expect(drawing(true, false, 1)).toBe("loop");
    expect(drawing(true, false, 0)).toBe("still");
    expect(drawing(false, false, 1)).toBe("none");
    expect(drawing(true, true, 1)).toBe("none");
    expect(drawing(false, true, 0)).toBe("none");
  });

  it("draws nothing before the first theme", () => {
    const { loop, painted, run } = scene();
    run.setDisplay(true);
    run.setCovered(false);
    run.redraw();
    expect(loop.running).toBe(false);
    expect(painted).toEqual([]);
    expect(run.now).toBe("none");
    run.setTheme(0.6);
    expect(loop.running).toBe(true);
    expect(painted).toEqual([0]);
  });

  it("stops while its monitor is covered and picks up where it was", () => {
    const { loop, painted, run } = scene();
    run.setTheme(1);
    run.tick(0.1);
    run.tick(0.1);
    run.setCovered(true);
    expect(loop.running).toBe(false);
    const before = painted.length;
    run.redraw();
    expect(painted.length, "nothing drawn while covered").toBe(before);
    run.setCovered(false);
    expect(loop.running).toBe(true);
    expect(loop.starts).toBe(2);
    run.tick(0.1);
    expect(run.time).toBeCloseTo(0.3, 9);
  });

  it("stops while the display is off, whatever else changes", () => {
    const { loop, run } = scene();
    run.setTheme(1);
    run.setDisplay(false);
    expect(loop.running).toBe(false);
    run.setCovered(false);
    run.setTheme(0.6);
    expect(loop.running, "no event restarts it while the display is off").toBe(false);
    run.setDisplay(true);
    expect(loop.running).toBe(true);
    run.setCovered(true);
    run.setDisplay(true);
    expect(loop.running, "the display back, the monitor still covered").toBe(false);
  });

  it("moves at the owner's motion: slower at ambient than rich", () => {
    const rich = scene();
    rich.run.setTheme(1);
    const ambient = scene();
    ambient.run.setTheme(0.6);
    for (let i = 0; i < 10; i++) {
      rich.run.tick(0.1);
      ambient.run.tick(0.1);
    }
    expect(rich.run.time).toBeCloseTo(1, 9);
    expect(ambient.run.time).toBeCloseTo(0.6, 9);
  });

  it("rests as one still frame at functional, drawn when it is seen again or resized", () => {
    const { loop, painted, run } = scene();
    run.setTheme(1);
    run.tick(0.5);
    run.setTheme(0);
    expect(loop.running).toBe(false);
    expect(painted.at(-1), "the frame it got to").toBeCloseTo(0.5, 9);
    run.tick(0.5); // a stray frame doesn't move it
    expect(run.time).toBeCloseTo(0.5, 9);
    const n = painted.length;
    run.redraw();
    expect(painted.length).toBe(n + 1);
    run.setCovered(true);
    run.redraw();
    expect(painted.length, "not while covered").toBe(n + 1);
    run.setCovered(false);
    expect(painted.length, "drawn again when uncovered").toBe(n + 2);
    expect(loop.running).toBe(false);
  });

  it("ignores a bad frame time and a bad factor", () => {
    const { run } = scene();
    run.setTheme(Number.NaN);
    expect(run.now).toBe("still");
    run.setTheme(1);
    run.tick(Number.NaN);
    run.tick(-1);
    run.tick(Number.POSITIVE_INFINITY);
    expect(run.time).toBe(0);
    run.setTheme(-3);
    expect(run.now).toBe("still");
  });
});
