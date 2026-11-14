/**
 * The speech bubble's arithmetic, kept pure: where it sits over his head,
 * how it appears, holds and goes (the shared motion spec), and the reply as
 * plain text. BubbleView.ts writes it into the page.
 *
 * The bubble's clock is moved only by the paced loop's tick, never by the
 * wall: parked, or with the display off, the loop stops, and so does the
 * bubble, so a reply that came while he was out of sight is still there to
 * read when he is back.
 */
import {
  FADE_MS, MATERIALISE_BLUR_FROM_PX, MATERIALISE_MS, MATERIALISE_SCALE_FROM, easeOutCubic, holdMs, opacityMs,
  smoothstep, transformMs,
} from "../shared/motion";

/** The widest the bubble grows, in CSS pixels. */
export const BUBBLE_MAX_WIDTH = 300;
/** Kept this far inside the window's edges. */
export const EDGE = 8;
/** Its tail's tip this far above the point over his head. */
export const GAP = 6;
/** How far the tail reaches below the panel (bubble.css: a 12-pixel square,
 * turned 45°, centred on the panel's bottom edge, reaches 8.5). */
export const TAIL_DEPTH = 8;
/** Below this much room over his head, it sits at the top of the window
 * instead, over him. */
export const MIN_BAND = 64;
/** The point the bubble points at: this far above his head bone at least,
 * in metres, and never lower than CROWN_CLEARANCE_M over the top of the
 * model (his hair). */
export const HEAD_CLEARANCE_M = 0.2;
export const CROWN_CLEARANCE_M = 0.02;

/** How far over his head bone the bubble points, for a model whose crown
 * is `crownAboveHead` metres over it. */
export function anchorAbove(crownAboveHead: number): number {
  return Number.isFinite(crownAboveHead) ? Math.max(HEAD_CLEARANCE_M, crownAboveHead + CROWN_CLEARANCE_M) : HEAD_CLEARANCE_M;
}
/** The share of the window the bubble may take when it sits over him. */
export const FALLBACK_HEIGHT = 0.4;
/** The tail keeps this far from the bubble's corners. */
const TAIL_INSET = 16;

export interface BubbleLayout {
  left: number;
  /** One of `top` and `bottom` is set: `bottom` (from the window's bottom
   * edge) over his head, `top` when it falls back to the top of the window. */
  top: number | null;
  bottom: number | null;
  maxWidth: number;
  maxHeight: number;
  /** Where the tail is, across the bubble. */
  tailX: number;
  fallback: boolean;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Where a bubble `measuredWidth` wide goes, in a window `view` big, for
 * his head at `anchor` (CSS pixels on the page). */
export function layout(
  anchor: { x: number; y: number },
  view: { width: number; height: number },
  measuredWidth: number,
): BubbleLayout {
  const maxWidth = Math.max(0, Math.min(BUBBLE_MAX_WIDTH, view.width - 2 * EDGE));
  const width = clamp(Number.isFinite(measuredWidth) ? measuredWidth : maxWidth, 0, maxWidth);
  const ax = Number.isFinite(anchor.x) ? anchor.x : view.width / 2;
  const ay = Number.isFinite(anchor.y) ? anchor.y : 0;
  const left = clamp(ax - width / 2, EDGE, Math.max(EDGE, view.width - EDGE - width));
  const tailX = width < 2 * TAIL_INSET ? width / 2 : clamp(ax - left, TAIL_INSET, width - TAIL_INSET);
  // Its tail's tip GAP above the point over his head (the panel the tail's
  // depth above that), never below the window's bottom edge; it may grow up
  // to the top edge.
  const tip = Math.min(ay - GAP - TAIL_DEPTH, view.height - EDGE);
  const band = tip - EDGE;
  if (band < MIN_BAND) {
    return {
      left, top: EDGE, bottom: null, maxWidth,
      maxHeight: Math.max(0, Math.round(view.height * FALLBACK_HEIGHT)), tailX, fallback: true,
    };
  }
  return { left, top: null, bottom: view.height - tip, maxWidth, maxHeight: band, tailX, fallback: false };
}

/** What the bubble looks like this frame. */
export interface Envelope {
  visible: boolean;
  opacity: number;
  scale: number;
  blurPx: number;
}

export const HIDDEN: Envelope = Object.freeze({ visible: false, opacity: 0, scale: 1, blurPx: 0 });

type Stage = "hidden" | "in" | "shown" | "hold" | "out";

/**
 * The bubble's appearing, holding and going, advanced only by `tick`:
 * - `show()`: a turn began. It materialises (scale 0.6 to 1 and blur 8 px
 *   to none over 400 ms × motion; opacity over at least 330 ms), or, when
 *   already up, stays up for the new turn.
 * - `end(chars)`: the turn is over. Once fully in, it holds `holdMs(chars)`
 *   (the hold paused while the cursor is over him), then fades, opacity
 *   only. The cursor coming back over him while it fades brings it back.
 * - `reset()`: gone at once.
 */
export class BubbleClock {
  private stage: Stage = "hidden";
  /** Opacity's progress, 0 to 1 (eased when drawn). */
  private o = 0;
  /** The materialise's progress, 0 to 1. */
  private p = 0;
  private ended = false;
  private holdFor = 0;
  private held = 0;

  get showing(): boolean {
    return this.stage !== "hidden";
  }

  show(): void {
    this.ended = false;
    this.held = 0;
    if (this.stage === "hidden") {
      this.o = 0;
      this.p = 0;
    }
    this.stage = "in";
  }

  end(chars: number): void {
    this.ended = true;
    this.holdFor = holdMs(chars);
    this.held = 0;
    if (this.stage === "shown") this.stage = "hold";
  }

  reset(): void {
    this.stage = "hidden";
    this.o = 0;
    this.p = 0;
    this.ended = false;
    this.held = 0;
  }

  tick(dtMs: number, hovered: boolean, motion: number): Envelope {
    const dt = Number.isFinite(dtMs) ? Math.max(0, dtMs) : 0;
    switch (this.stage) {
      case "hidden":
        return HIDDEN;
      case "in": {
        const moveMs = transformMs(MATERIALISE_MS, motion);
        this.o = Math.min(1, this.o + dt / opacityMs(MATERIALISE_MS, motion));
        this.p = moveMs === 0 ? 1 : Math.min(1, this.p + dt / moveMs);
        if (this.o >= 1 && this.p >= 1) this.stage = this.ended ? "hold" : "shown";
        break;
      }
      case "shown":
        break;
      case "hold":
        if (!hovered) this.held += dt;
        if (this.held >= this.holdFor) this.stage = "out";
        break;
      case "out":
        this.o = hovered
          ? Math.min(1, this.o + dt / opacityMs(MATERIALISE_MS, motion))
          : Math.max(0, this.o - dt / opacityMs(FADE_MS, motion));
        if (this.o <= 0) {
          this.reset();
          return HIDDEN;
        }
        break;
    }
    const e = easeOutCubic(this.p);
    return {
      visible: true,
      opacity: smoothstep(this.o),
      scale: MATERIALISE_SCALE_FROM + (1 - MATERIALISE_SCALE_FROM) * e,
      blurPx: MATERIALISE_BLUR_FROM_PX * (1 - e),
    };
  }
}

/**
 * The reply as the bubble shows it: Markdown's marks taken off (bold,
 * italic, code, headings, links, code fences), the words, emoji and line
 * breaks kept. The text is only ever set as text, never as markup.
 */
export function plainText(s: string): string {
  return s
    .replace(/^[ \t]*```[^\n]*$\n?/gm, "")
    .replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, "")
    .replace(/!?\[([^\]\n]*)\]\([^)\n]*\)/g, "$1")
    .replace(/\*\*(?=\S)([^*\n]*?\S)\*\*/g, "$1")
    .replace(/__(?=\S)([^_\n]*?\S)__/g, "$1")
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?![*\w])/g, "$1$2")
    .replace(/`([^`\n]+)`/g, "$1");
}

/** Characters as a reader counts them (code points, so an emoji is one). */
export function charCount(s: string): number {
  return [...s].length;
}
