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
  /** What the covered page keeps is out of date (his place changed while
   * the display was off): drawn once the display is back. */
  private stale = false;

  /** `paint(t, dt)`: draw the scene at scene time `t`, `dt` seconds of
   * real time after the last frame (0 for a frame drawn out of turn: a
   * resize, new colours, a still frame), for what eases in real time. */
  constructor(
    private readonly paint: (t: number, dt: number) => void,
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
    const real = Number.isFinite(dt) && dt > 0 ? dt : 0;
    this.t += real * this.factor;
    this.paint(this.t, real);
  }

  setDisplay(on: boolean): void {
    this.displayOn = on;
    this.apply();
    if (on && this.covered && this.stale && this.themed) {
      this.stale = false;
      this.paint(this.t, 0);
    }
  }

  setCovered(covered: boolean): void {
    const was = this.covered;
    this.covered = covered;
    this.apply();
    // Covered now: the frame kept is drawn as things are now, with
    // whatever was on its way (a fade, a still frame due) put in place.
    if (covered && !was) this.keep();
  }

  /** A theme, with the motion factor it brings (0, 0.6 or 1): its colours
   * are drawn at once, not at the next frame. */
  setTheme(factor: number): void {
    this.factor = Number.isFinite(factor) && factor > 0 ? factor : 0;
    this.themed = true;
    this.apply();
    if (this.now === "loop") this.paint(this.t, 0);
  }

  /** Something the scene shows changed (a resize, new colours): drawn now,
   * unless nothing is being drawn. */
  redraw(): void {
    if (this.now !== "none") this.paint(this.t, 0);
  }

  /** Something drawn changed out of turn (where Fritz is). While the
   * monitor is covered, nothing is drawn, but the frame its page keeps is
   * the one the owner sees the moment the cover goes, so it must be how
   * things are now (Fritz hidden by the fullscreen app, say): it is drawn
   * at the cover's coming and at each change under it, or, with the
   * display off (or no theme yet), once that is back. Otherwise what it
   * means: the loop will draw it, a still scene is to be drawn again, or
   * nothing is drawn. */
  placed(): "covered" | Drawing {
    if (!this.covered) return this.now;
    this.keep();
    return "covered";
  }

  /** Draw the frame a covered page keeps as things are now, or owe it
   * until the display (and a theme) is there. */
  private keep(): void {
    if (this.themed && this.displayOn) {
      this.stale = false;
      this.paint(this.t, 0);
    } else {
      this.stale = true;
    }
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
    if (now === "still") this.paint(this.t, 0);
  }
}
