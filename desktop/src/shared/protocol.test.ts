/**
 * protocol.ts (generated from the brain's companion_protocol.py) against the
 * fixtures generated with it: the check takes every frame the server may
 * send and none of the broken ones, and the frames the client builds have
 * the keys and types the brain's models write. The brain's side of the same
 * fixtures is tests/test_gen_protocol_ts.py.
 */
import { describe, expect, it } from "vitest";
import fixtures from "./protocol.fixtures.json";
import {
  CLIENT_ID_PATTERN, SERVER_FRAME_TYPES, TURN_ID_PATTERN, WIDGET_NAMES, isServerFrame, isWidgetPayload,
  type WidgetPayloads,
} from "./protocol";
import { CAPS, audioEndFrame, cancelFrame, helloFrame, pingFrame, pongFrame, sayFrame } from "./companion";

/** Each key's JSON type, recursively: what a frame looks like on the wire. */
function shapeOf(v: unknown): unknown {
  if (v === null) return "null";
  // An array is the kinds of thing in it, whatever their number.
  if (Array.isArray(v)) return { arrayOf: [...new Set(v.map((x) => JSON.stringify(shapeOf(x))))].sort() };
  if (typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, shapeOf(x)]).sort());
  return typeof v;
}

describe("isServerFrame", () => {
  it("takes every frame the server may send", () => {
    for (const { why, frame } of fixtures.server.valid) expect(isServerFrame(frame), why).toBe(true);
  });

  it("refuses each broken one", () => {
    expect(fixtures.server.invalid.length).toBeGreaterThan(50);
    for (const { why, frame } of fixtures.server.invalid) expect(isServerFrame(frame), why).toBe(false);
  });

  it("knows every type there is a valid frame of, and no other", () => {
    const valid = new Set(fixtures.server.valid.map((e) => (e.frame as { type: string }).type));
    expect(new Set(SERVER_FRAME_TYPES)).toEqual(valid);
  });

  it("refuses what isn't an object", () => {
    for (const x of [undefined, 0, "", true, () => ({ type: "pong", t: 1 })]) expect(isServerFrame(x)).toBe(false);
  });
});

describe("isWidgetPayload", () => {
  const name = (n: string) => n as keyof WidgetPayloads;

  it("takes every payload the server may send, by its widget", () => {
    for (const { why, name: n, payload } of fixtures.widgets.valid) expect(isWidgetPayload(name(n), payload), why).toBe(true);
    expect(new Set(fixtures.widgets.valid.map((e) => e.name))).toEqual(new Set(WIDGET_NAMES));
  });

  it("refuses each broken one, down into the desk's exchange", () => {
    expect(fixtures.widgets.invalid.length).toBeGreaterThan(40);
    expect(fixtures.widgets.invalid.some((e) => e.why.startsWith("desk: last."))).toBe(true);
    for (const { why, name: n, payload } of fixtures.widgets.invalid) expect(isWidgetPayload(name(n), payload), why).toBe(false);
  });

  it("refuses a payload for a widget it doesn't know, and one meant for another", () => {
    const clock = fixtures.widgets.valid.find((e) => e.name === "clock")!.payload;
    expect(isWidgetPayload(name("agenda"), clock)).toBe(false);
    expect(isWidgetPayload("weather", clock)).toBe(false);
  });
});

describe("the frames the client sends", () => {
  const config = { url: "ws://127.0.0.1:8001/companion", token: "a-token", client: { id: "nick-pc", name: "Nick's PC" } };
  const id = "7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d";
  const valid = fixtures.client.valid;

  it("are shaped as the brain's models write them", () => {
    expect(shapeOf(helloFrame(config))).toEqual(shapeOf(valid.hello));
    expect(shapeOf(sayFrame(id, "Good evening"))).toEqual(shapeOf(valid.say));
    expect(shapeOf(cancelFrame(id))).toEqual(shapeOf(valid.cancel));
    expect(shapeOf(pingFrame(1))).toEqual(shapeOf(valid.ping));
    expect(shapeOf(pongFrame(1))).toEqual(shapeOf(valid.pong));
  });

  it("say what they are", () => {
    expect(helloFrame(config)).toEqual({
      type: "hello",
      protocol: 1,
      token: "a-token",
      client: { id: "nick-pc", name: "Nick's PC", caps: ["text", "audio-in", "vrm"] },
    });
    expect(CAPS).toEqual(["text", "audio-in", "vrm"]);
    // audio_end says what the shell's ears hand over: the fixture's own.
    expect(audioEndFrame(id)).toEqual({ ...fixtures.client.valid.audio_end, turn_id: id });
    expect(sayFrame(id, "hi")).toEqual({ type: "say", turn_id: id, text: "hi" });
    expect(pongFrame(42)).toEqual({ type: "pong", t: 42 });
  });

  it("use ids the brain accepts", () => {
    expect(TURN_ID_PATTERN.test(id)).toBe(true);
    expect(TURN_ID_PATTERN.test("a b")).toBe(false);
    expect(CLIENT_ID_PATTERN.test(config.client.id)).toBe(true);
    expect(CLIENT_ID_PATTERN.test("Nick-PC")).toBe(false);
  });
});
