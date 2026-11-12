/**
 * The page's half of the wire contract. src-tauri/src/overlay_machine.rs
 * reads the same fixture, so a change on either side fails the other's tests.
 */
import { describe, expect, it } from "vitest";
import fixture from "./wire.fixture.json";
import {
  BEAT_CMD,
  HANDSHAKE_CMD,
  MODE_CMD,
  NO_READY_GLOBAL,
  STATUS_EVENT,
  VRM_CMD,
  beatArgs,
  handshakeArgs,
  modeArgs,
  vrmArgs,
  pageInstance,
  parseStatus,
} from "./status";

describe("the avatar's wire contract", () => {
  it("names what the shell names", () => {
    expect(STATUS_EVENT).toBe(fixture.events.status);
    expect(NO_READY_GLOBAL).toBe(fixture.debug.noReadyGlobal);
  });

  it("reads every status the shell sends, field for field", () => {
    for (const [name, sample] of Object.entries(fixture.statusSamples)) {
      const parsed = parseStatus(sample);
      expect(parsed, name).toEqual(sample);
      expect(Object.keys(parsed!).sort(), name).toEqual([...fixture.statusKeys].sort());
    }
  });

  it("knows every phase and mode", () => {
    const base = fixture.statusSamples.closed;
    for (const phase of fixture.phases) expect(parseStatus({ ...base, phase }), phase).not.toBeNull();
    for (const mode of fixture.modes) expect(parseStatus({ ...base, mode }), mode).not.toBeNull();
  });

  it("refuses anything that isn't a status", () => {
    const base = fixture.statusSamples.arranging;
    expect(parseStatus(null)).toBeNull();
    expect(parseStatus("painted")).toBeNull();
    expect(parseStatus({ ...base, phase: "hidden" })).toBeNull();
    expect(parseStatus({ ...base, mode: "drag" })).toBeNull();
    expect(parseStatus({ ...base, instance: -1 })).toBeNull();
    expect(parseStatus({ ...base, arrangeGen: 1.5 })).toBeNull();
    expect(parseStatus({ ...base, parked: "no" })).toBeNull();
    expect(parseStatus({ ...base, error: 3 })).toBeNull();
  });

  it("calls the commands the shell registers, with the arguments it takes", () => {
    const c = fixture.commands;
    expect([HANDSHAKE_CMD, BEAT_CMD, MODE_CMD, VRM_CMD]).toEqual([c.handshake.name, c.beat.name, c.mode.name, c.vrm.name]);
    expect(Object.keys(handshakeArgs(1, "ready"))).toEqual(c.handshake.args);
    expect(Object.keys(beatArgs(1, 2))).toEqual(c.beat.args);
    expect(Object.keys(modeArgs("locked"))).toEqual(c.mode.args);
    expect(Object.keys(vrmArgs())).toEqual(c.vrm.args);
    // The values the page sends are ones the shell parses.
    for (const stage of ["ready", "painted"] as const) expect(fixture.handshakeStages).toContain(handshakeArgs(1, stage).stage);
    expect(fixture.modeRequests).toContain(modeArgs("locked").mode);
  });

  it("reads its instance from the page URL", () => {
    expect(pageInstance("?instance=3")).toBe(3);
    expect(pageInstance("")).toBe(0);
    expect(pageInstance("?instance=-2")).toBe(0);
    expect(pageInstance("?instance=x")).toBe(0);
  });
});
