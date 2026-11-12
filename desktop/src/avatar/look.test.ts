import { describe, expect, it } from "vitest";
import fixture from "./wire.fixture.json";
import {
  AHEAD, CURSOR_CMD, CURSOR_EVENT, MAX_PITCH, MAX_YAW, VIEW_DISTANCE, approach, gaze, parseCursor, type CursorFrame,
} from "./look";

const view = { width: 300, height: 460 };
/** His eyes, high in the middle of his window. */
const head = { x: 150, y: 80 };

/** A cursor at CSS pixels (x, y) on a page at 200 %. */
function at(x: number, y: number, scale = 2): CursorFrame {
  return { active: true, x: x * scale, y: y * scale, width: view.width * scale, height: view.height * scale, over: false };
}

describe("where he looks", () => {
  it("straight ahead at a cursor on his eyes", () => {
    const g = gaze(at(head.x, head.y), view, head);
    expect(g.yaw).toBeCloseTo(0, 6);
    expect(g.pitch).toBeCloseTo(0, 6);
  });

  it("right and up for a cursor right of and above his eyes", () => {
    const d = VIEW_DISTANCE * view.height;
    const g = gaze(at(head.x + d * Math.tan((20 * Math.PI) / 180), head.y - d * Math.tan((10 * Math.PI) / 180)), view, head);
    expect(g.yaw).toBeCloseTo(20, 6);
    expect(g.pitch).toBeCloseTo(10, 6);
    const left = gaze(at(head.x - 50, head.y + 50), view, head);
    expect(left.yaw).toBeLessThan(0);
    expect(left.pitch).toBeLessThan(0);
  });

  /** A cursor across the desktop is as far as eyes and head go. */
  it("no further than ±35° across and ±20° up and down", () => {
    expect(gaze(at(-5000, head.y), view, head).yaw).toBe(-MAX_YAW);
    expect(gaze(at(9000, head.y), view, head).yaw).toBe(MAX_YAW);
    expect(gaze(at(head.x, -5000), view, head).pitch).toBe(MAX_PITCH);
    expect(gaze(at(head.x, 9000), view, head).pitch).toBe(-MAX_PITCH);
  });

  /** The shell sends physical pixels; the page's scale doesn't change the
   * angle. */
  it("the same at any monitor's scale", () => {
    for (const scale of [1, 1.25, 2, 3]) {
      const g = gaze(at(head.x + 120, head.y - 40, scale), view, head);
      expect(g.yaw).toBeCloseTo(gaze(at(head.x + 120, head.y - 40, 1), view, head).yaw, 9);
      expect(g.pitch).toBeCloseTo(gaze(at(head.x + 120, head.y - 40, 1), view, head).pitch, 9);
    }
  });

  it("ahead when there is nothing to look at", () => {
    expect(gaze({ ...at(10, 10), active: false }, view, head)).toEqual(AHEAD);
    expect(gaze({ ...at(10, 10), width: 0 }, view, head)).toEqual(AHEAD);
    expect(gaze(at(10, 10), { width: 0, height: 0 }, head)).toEqual(AHEAD);
  });

  it("eases toward a new gaze, never past it", () => {
    let v = 0;
    const seen: number[] = [];
    for (let i = 0; i < 30; i++) {
      v = approach(v, 30, 1 / 30, 0.08);
      seen.push(v);
    }
    expect(seen.every((x, i) => x <= 30 && (i === 0 || x >= seen[i - 1]))).toBe(true);
    expect(v).toBeGreaterThan(29.9);
    // A stalled frame (half a second) still lands short, never past.
    expect(approach(0, 30, 0.5, 0.08)).toBeLessThanOrEqual(30);
    expect(approach(0, 30, 0.5, 0.08)).toBeGreaterThan(29);
    expect(approach(5, 30, 0, 0.08)).toBe(5);
    expect(approach(5, 30, -1, 0.08)).toBe(5);
    expect(approach(5, 30, Number.NaN, 0.08)).toBe(5);
  });

  it("reads the shell's payload strictly", () => {
    for (const sample of Object.values(fixture.cursorSamples)) expect(parseCursor(sample)).toEqual(sample);
    const ok = fixture.cursorSamples.over;
    expect(parseCursor(null)).toBeNull();
    expect(parseCursor({ ...ok, x: "1" })).toBeNull();
    expect(parseCursor({ ...ok, active: 1 })).toBeNull();
    expect(parseCursor({ ...ok, width: Number.POSITIVE_INFINITY })).toBeNull();
    const { over: _, ...missing } = ok;
    expect(parseCursor(missing)).toBeNull();
  });

  it("uses the event, command and keys the shell has", () => {
    expect(CURSOR_EVENT).toBe(fixture.events.cursor);
    expect(CURSOR_CMD).toBe(fixture.commands.cursorNow.name);
    expect(fixture.commands.cursorNow.args).toEqual([]);
    expect(Object.keys(fixture.cursorSamples.over).sort()).toEqual(fixture.cursorKeys);
  });
});
