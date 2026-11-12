import { describe, expect, it } from "vitest";
import { MAX_DT, PacedLoop, type Clock } from "./PacedLoop";

/** A clock that only moves when the test says so, running due timers in
 * time order, as the browser would. */
class FakeClock implements Clock {
  t = 0;
  private seq = 0;
  private timers = new Map<number, { at: number; fn: () => void }>();
  delays: number[] = [];

  now(): number {
    return this.t;
  }
  setTimeout(fn: () => void, ms: number): unknown {
    const id = ++this.seq;
    this.delays.push(ms);
    this.timers.set(id, { at: this.t + ms, fn });
    return id;
  }
  clearTimeout(handle: unknown): void {
    this.timers.delete(handle as number);
  }
  get pending(): number {
    return this.timers.size;
  }
  /** Run everything due within the next `ms`, in order. */
  advance(ms: number): void {
    const end = this.t + ms;
    for (;;) {
      let due: [number, { at: number; fn: () => void }] | undefined;
      for (const entry of this.timers) {
        if (entry[1].at <= end && (!due || entry[1].at < due[1].at)) due = entry;
      }
      if (!due) break;
      this.timers.delete(due[0]);
      this.t = Math.max(this.t, due[1].at);
      due[1].fn();
    }
    this.t = Math.max(this.t, end); // a frame's cost may have run past the end
  }
}

function counting(clock: FakeClock, cost = 0) {
  const dts: number[] = [];
  const loop = new PacedLoop(
    (dt) => {
      dts.push(dt);
      clock.t += cost; // the frame's own cost
    },
    30,
    clock,
  );
  return { loop, dts };
}

describe("PacedLoop", () => {
  it("ticks 30 times a second", () => {
    const clock = new FakeClock();
    const { loop, dts } = counting(clock);
    loop.start();
    clock.advance(1000);
    expect(dts.length).toBeGreaterThanOrEqual(30);
    expect(dts.length).toBeLessThanOrEqual(31);
    expect(loop.running).toBe(true);
  });

  it("doesn't let the cost of a frame stretch the period", () => {
    const clock = new FakeClock();
    const { loop, dts } = counting(clock, 12);
    loop.start();
    clock.advance(3000);
    // A naive "draw, then wait a period" loop would manage 3000 / 45 ≈ 66.
    expect(dts.length).toBeGreaterThanOrEqual(89);
    expect(dts.length).toBeLessThanOrEqual(91);
  });

  it("resyncs after a stall instead of bursting to catch up", () => {
    const clock = new FakeClock();
    let stall = false;
    const dts: number[] = [];
    const loop = new PacedLoop(
      (dt) => {
        dts.push(dt);
        if (stall) {
          clock.t += 500;
          stall = false;
        }
      },
      30,
      clock,
    );
    loop.start();
    clock.advance(200);
    stall = true;
    clock.advance(40); // the stalled frame runs in here
    const before = dts.length;
    clock.advance(100);
    // A burst would run ~15 frames at once to make up the 500 ms.
    expect(dts.length - before).toBeLessThanOrEqual(4);
    // And the scene is told about at most MAX_DT, not the half second.
    expect(Math.max(...dts)).toBeLessThanOrEqual(MAX_DT);
  });

  it("stops at once, even with a tick already scheduled", () => {
    const clock = new FakeClock();
    const { loop, dts } = counting(clock);
    loop.start();
    clock.advance(100);
    const seen = dts.length;
    loop.stop();
    expect(loop.running).toBe(false);
    expect(clock.pending).toBe(0);
    clock.advance(1000);
    expect(dts.length).toBe(seen);
  });

  it("can be stopped from inside a tick", () => {
    const clock = new FakeClock();
    let n = 0;
    const loop: PacedLoop = new PacedLoop(
      () => {
        if (++n === 3) loop.stop();
      },
      30,
      clock,
    );
    loop.start();
    clock.advance(1000);
    expect(n).toBe(3);
    expect(loop.running).toBe(false);
  });

  it("never runs two loops: a restart supersedes the old one", () => {
    const clock = new FakeClock();
    const { loop, dts } = counting(clock);
    loop.start();
    loop.start(); // ignored while running
    clock.advance(50);
    loop.stop();
    loop.start();
    clock.advance(1000);
    expect(dts.length).toBeLessThanOrEqual(33);
  });

  it("keeps going when a frame throws", () => {
    const clock = new FakeClock();
    let n = 0;
    const loop = new PacedLoop(
      () => {
        n++;
        throw new Error("bad frame");
      },
      30,
      clock,
    );
    const quiet = console.error;
    console.error = () => {};
    try {
      loop.start();
      clock.advance(200);
    } finally {
      console.error = quiet;
    }
    expect(n).toBeGreaterThan(3);
    expect(loop.running).toBe(true);
  });

  it("changes rate from the next frame, within sane bounds", () => {
    const clock = new FakeClock();
    const { loop, dts } = counting(clock);
    loop.start();
    clock.advance(1000);
    loop.setFps(10);
    const before = dts.length;
    clock.advance(1000);
    expect(dts.length - before).toBeGreaterThanOrEqual(10);
    expect(dts.length - before).toBeLessThanOrEqual(12);
    loop.setFps(1000);
    expect(loop.rate).toBe(60);
    loop.setFps(Number.NaN);
    expect(loop.rate).toBe(30);
  });

  it("schedules only with setTimeout and never a negative delay", () => {
    const clock = new FakeClock();
    const { loop } = counting(clock, 50); // every frame overruns
    loop.start();
    clock.advance(1000);
    expect(clock.delays.every((d) => d >= 0)).toBe(true);
  });
});
