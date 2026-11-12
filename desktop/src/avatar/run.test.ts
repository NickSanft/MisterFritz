import { describe, expect, it } from "vitest";
import fixture from "./wire.fixture.json";
import { POWER_CMD, POWER_EVENT, afterAsking, parseDisplay, shouldDraw } from "./run";
import type { PhaseName } from "./status";

describe("when the avatar draws", () => {
  const at = (phase: PhaseName, parked = false, mode: "locked" | "arrange" = "locked") => ({ phase, parked, mode });

  it("draws once the shell has shown him, with the display on", () => {
    for (const phase of ["starting", "shown", "painted"] as const) {
      expect(shouldDraw(at(phase), true)).toBe(true);
    }
  });

  it("never before the shell has said anything", () => {
    expect(shouldDraw(null, true)).toBe(false);
  });

  it("not while he is parked, or his window is going", () => {
    expect(shouldDraw(at("painted", true), true)).toBe(false);
    expect(shouldDraw(at("closing"), true)).toBe(false);
    expect(shouldDraw(at("closed"), true)).toBe(false);
  });

  /** The bug this replaces: any status event used to restart the loop,
   * whatever else had stopped it. */
  it("not while the display is off, whatever the status says", () => {
    for (const phase of ["starting", "shown", "painted", "closing", "closed"] as const) {
      expect(shouldDraw(at(phase), false)).toBe(false);
    }
  });

  /** Arrange's lease is beaten from the loop: with the display off (a
   * screen timeout while arranging), stopping it would end Arrange with a
   * false "stopped responding". */
  it("keeps drawing while he is being arranged, display or not", () => {
    expect(shouldDraw(at("painted", false, "arrange"), false)).toBe(true);
    expect(shouldDraw(at("painted", false, "arrange"), true)).toBe(true);
    expect(shouldDraw(at("closing", false, "arrange"), false)).toBe(false);
  });

  /** The start-up question's answer can arrive after an event that is
   * newer than it; it must not undo that event. */
  it("takes the start-up answer only when no event came while asking", () => {
    expect(afterAsking(false, 0, 0, true)).toBe(false);
    expect(afterAsking(true, 0, 0, false)).toBe(true);
    expect(afterAsking(true, 0, 1, false)).toBe(false);
    expect(afterAsking(false, 2, 3, true)).toBe(true);
    // The cursor's start-up question takes the same rule.
    const asked = { x: 1 };
    const heard = { x: 2 };
    expect(afterAsking(asked, 4, 4, null)).toBe(asked);
    expect(afterAsking<object | null>(asked, 4, 5, heard)).toBe(heard);
  });

  it("reads the display state strictly", () => {
    expect(parseDisplay(true)).toBe(true);
    expect(parseDisplay(false)).toBe(false);
    for (const junk of [0, 1, "true", null, undefined, {}]) expect(parseDisplay(junk)).toBeNull();
  });

  it("uses the event and command the shell has", () => {
    expect(POWER_EVENT).toBe(fixture.events.power);
    expect(POWER_CMD).toBe(fixture.commands.powerDisplay.name);
    expect(fixture.commands.powerDisplay.args).toEqual([]);
  });
});
