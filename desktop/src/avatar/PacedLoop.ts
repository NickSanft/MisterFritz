/**
 * The avatar's frame loop, paced with setTimeout and never with
 * requestAnimationFrame.
 *
 * Lorraine's lesson (wallpaper.ts, confirmed with PresentMon): an active rAF
 * loop keeps Chromium's compositor presenting every vsync — 144 or 175 times a
 * second — even when the page only redraws at 30 Hz, and that was the
 * dominant idle GPU cost. A setTimeout loop damages the page only when it
 * draws, so the compositor presents at our rate. The loop steers toward an
 * absolute next-frame time, so the cost of a frame doesn't stretch the
 * period, and after a stall (more than two periods behind) it resyncs to now
 * instead of firing a catch-up burst that would fast-forward the scene.
 */

/** What the loop needs from the page; tests pass a fake. */
export interface Clock {
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export const browserClock: Clock = {
  now: () => performance.now(),
  setTimeout: (fn, ms) => window.setTimeout(fn, ms),
  clearTimeout: (handle) => window.clearTimeout(handle as number),
};

/** The longest step a frame is told about, in seconds: after a stall the
 * scene (and three-vrm's spring bones) move on by this much, not by the gap. */
export const MAX_DT = 0.1;

const MIN_FPS = 1;
const MAX_FPS = 60;

function clampFps(fps: number): number {
  return Number.isFinite(fps) ? Math.min(MAX_FPS, Math.max(MIN_FPS, fps)) : 30;
}

export class PacedLoop {
  private fps: number;
  private gen = 0;
  private timer: unknown = null;

  constructor(
    private readonly tick: (dt: number) => void,
    fps = 30,
    private readonly clock: Clock = browserClock,
  ) {
    this.fps = clampFps(fps);
  }

  get running(): boolean {
    return this.timer !== null;
  }

  get rate(): number {
    return this.fps;
  }

  /** Start ticking (the first tick on the next turn). A running loop is left
   * as it is. */
  start(): void {
    if (this.running) return;
    const gen = ++this.gen;
    let next = this.clock.now();
    let last = -1;
    const step = (): void => {
      if (gen !== this.gen) return;
      const now = this.clock.now();
      const dt = last < 0 ? 1 / this.fps : Math.min(MAX_DT, Math.max(0, (now - last) / 1000));
      last = now;
      try {
        this.tick(dt);
      } catch (e) {
        // One bad frame never stops the avatar.
        console.error("avatar frame failed", e);
      }
      if (gen !== this.gen) return; // the tick stopped the loop
      const period = 1000 / this.fps;
      next += period;
      const after = this.clock.now();
      if (after - next > 2 * period) next = after;
      this.timer = this.clock.setTimeout(step, Math.max(0, next - after));
    };
    this.timer = this.clock.setTimeout(step, 0);
  }

  /** Stop at once: no further tick runs, even one already scheduled. */
  stop(): void {
    this.gen++;
    if (this.timer !== null) this.clock.clearTimeout(this.timer);
    this.timer = null;
  }

  /** Change the rate; a running loop takes it from its next frame. */
  setFps(fps: number): void {
    this.fps = clampFps(fps);
  }
}
