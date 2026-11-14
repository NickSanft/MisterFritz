/**
 * A clock for tests: it moves only when the test says so, and runs due
 * timers in time order, as the browser would. Never imported by a page.
 */
import type { Clock } from "./clock";

export class FakeClock implements Clock {
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
