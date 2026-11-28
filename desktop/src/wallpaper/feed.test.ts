import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import { COMPANION_STATUS_EVENT } from "../shared/link";
import { Freshness, NOW_CMD, SEEN_EVENT, STATUS_CMD, parseSeen } from "./feed";

describe("the widgets as the shell passes them on", () => {
  it("are the shell's wire", () => {
    expect(SEEN_EVENT).toBe(fixture.events.widgetSeen);
    expect(NOW_CMD).toBe(fixture.commands.widgetsNow.name);
    expect(STATUS_CMD).toBe(fixture.commands.companionStatus.name);
    expect(COMPANION_STATUS_EVENT).toBe(fixture.events.companionStatus);
  });

  it("are read strictly", () => {
    expect(parseSeen({ name: "clock", ttlS: 120, ageMs: 0, payload: {} })).toEqual({ name: "clock", ttlS: 120, ageMs: 0 });
    for (const bad of [null, [], { name: "", ttlS: 1, ageMs: 0 }, { name: "x", ttlS: -1, ageMs: 0 },
                       { name: "x", ttlS: 1.5, ageMs: 0 }, { name: "x", ttlS: 1 }, { ttlS: 1, ageMs: 0 }]) {
      expect(parseSeen(bad), JSON.stringify(bad)).toBeNull();
    }
  });
});

describe("which widgets are fresh", () => {
  it("holds each for its ttl, and none while the link is down", () => {
    const f = new Freshness();
    f.hear({ name: "clock", ttlS: 120, ageMs: 0 }, 1_000);
    expect(f.lit("clock", 1_000, true)).toBe(true);
    expect(f.lit("clock", 120_999, true)).toBe(true);
    expect(f.lit("clock", 121_000, true)).toBe(false);
    expect(f.lit("clock", 2_000, false)).toBe(false);
    expect(f.lit("weather", 2_000, true)).toBe(false);
  });

  it("holds one with no ttl until the next", () => {
    const f = new Freshness();
    f.hear({ name: "desk", ttlS: 0, ageMs: 0 }, 0);
    expect(f.lit("desk", 10 ** 12, true)).toBe(true);
    expect(f.untilStale(5)).toBeNull();
  });

  it("counts a late answer's age, and never lets it undo a newer event", () => {
    const f = new Freshness();
    f.hear({ name: "clock", ttlS: 120, ageMs: 0 }, 50_000);
    // widgets_now answers after: the clock it knew was heard 90 s ago.
    f.hear({ name: "clock", ttlS: 120, ageMs: 90_000 }, 60_000);
    expect(f.lit("clock", 169_000, true)).toBe(true);
    // A weather 119 s old holds for one more second.
    f.hear({ name: "weather", ttlS: 120, ageMs: 119_000 }, 60_000);
    expect(f.lit("weather", 60_999, true)).toBe(true);
    expect(f.lit("weather", 61_000, true)).toBe(false);
  });

  it("says when the soonest lit one goes stale", () => {
    const f = new Freshness();
    expect(f.untilStale(0)).toBeNull();
    f.hear({ name: "clock", ttlS: 120, ageMs: 0 }, 0);
    f.hear({ name: "weather", ttlS: 1800, ageMs: 0 }, 0);
    expect(f.untilStale(10_000)).toBe(110_000);
    expect(f.untilStale(200_000)).toBe(1_600_000);
    expect(f.untilStale(2_000_000)).toBeNull();
  });
});
