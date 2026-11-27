/**
 * When the study scene draws: one decision, from everything that can stop
 * it, so no one event restarts the loop while another reason to stop still
 * holds (the display coming back while a fullscreen game still covers the
 * monitor, say).
 */

/** The shell's word to one study scene: `true` while its monitor is
 * covered (another app's fullscreen window, or the lock screen). */
export const COVERED_EVENT = "wallpaper:covered";
/** Whether it is covered now, for a page that starts after the last word. */
export const COVERED_CMD = "wallpaper_covered";

/** Frames a second: the backdrop moves too slowly to need more. */
export const SCENE_FPS = 10;

/** What the scene does: run the paced loop, rest as one still frame, or
 * nothing at all (no one could see it). */
export type Drawing = "loop" | "still" | "none";

/** The decision: nothing while the display is off or the monitor is
 * covered; a still frame at motion's functional level (`factor` 0); the
 * loop otherwise. Pure. */
export function drawing(displayOn: boolean, covered: boolean, factor: number): Drawing {
  if (!displayOn || covered) return "none";
  return factor > 0 ? "loop" : "still";
}

/** What the scene needs of its loop; tests pass a fake. */
export interface Loop {
  readonly running: boolean;
  start(): void;
  stop(): void;
}

/** The scene's clock and its loop. Scene time moves only while the loop
 * runs, at the owner's motion factor, so the backdrop picks up where it
 * stopped, and rests where it is at functional. Nothing is drawn before
 * the first theme (the page has no colours until then). */
export class SceneRun {
  private displayOn = true;
  private covered = false;
  private factor = 0;
  private themed = false;
  private t = 0;

  constructor(
    private readonly paint: (t: number) => void,
    private readonly loop: Loop,
  ) {}

  /** Scene time now, in seconds. */
  get time(): number {
    return this.t;
  }

  get now(): Drawing {
    return this.themed ? drawing(this.displayOn, this.covered, this.factor) : "none";
  }

  /** One of the loop's frames, `dt` seconds after the last. */
  tick(dt: number): void {
    if (Number.isFinite(dt) && dt > 0) this.t += dt * this.factor;
    this.paint(this.t);
  }

  setDisplay(on: boolean): void {
    this.displayOn = on;
    this.apply();
  }

  setCovered(covered: boolean): void {
    this.covered = covered;
    this.apply();
  }

  /** A theme, with the motion factor it brings (0, 0.6 or 1): its colours
   * are drawn at once, not at the next frame. */
  setTheme(factor: number): void {
    this.factor = Number.isFinite(factor) && factor > 0 ? factor : 0;
    this.themed = true;
    this.apply();
    if (this.now === "loop") this.paint(this.t);
  }

  /** Something the scene shows changed (a resize, new colours): drawn now,
   * unless nothing is being drawn. */
  redraw(): void {
    if (this.now !== "none") this.paint(this.t);
  }

  /** Start or stop the loop for the decision now; a still frame is drawn
   * each time the scene comes to rest or is seen again (a resize while it
   * was covered cleared the canvas). */
  private apply(): void {
    const now = this.now;
    if (now === "loop") {
      this.loop.start();
      return;
    }
    this.loop.stop();
    if (now === "still") this.paint(this.t);
  }
}
