/**
 * Where Fritz looks: the shell's `cursor` event, from his window's
 * top-left in physical pixels, turned into a gaze (degrees) from his head's
 * place on the page, clamped to how far eyes and head will turn (plan 14
 * build § 4.3: yaw ±35°, pitch ±20°), and eased so he glances rather than
 * snaps.
 */

import { invoke } from "@tauri-apps/api/core";

/** The shell's event, sent only on a real move (cursor.rs). */
export const CURSOR_EVENT = "cursor";
/** The cursor now, for a page that starts after the last event: a still
 * cursor sends none. */
export const CURSOR_CMD = "cursor_now";

/** How far he will look, in degrees. */
export const MAX_YAW = 35;
export const MAX_PITCH = 20;
/** How far in front of the screen his eyes are, in window heights: the
 * cursor this far to the side of his head is 45° (so past the clamp). */
export const VIEW_DISTANCE = 1.2;
/** The share of the turn his head takes; his eyes take the rest. The head's
 * is what shows on a model without eyes (the placeholder). */
export const HEAD_SHARE_YAW = 0.4;
export const HEAD_SHARE_PITCH = 0.3;
/** Seconds to cover two thirds of the way to a new gaze: eyes quick, the
 * head slower. */
export const EYE_TAU = 0.08;
export const HEAD_TAU = 0.3;

export interface CursorFrame {
  active: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  over: boolean;
}

/** Degrees: yaw to the right of the page is positive, pitch up is positive. */
export interface Gaze {
  yaw: number;
  pitch: number;
}

export const AHEAD: Gaze = Object.freeze({ yaw: 0, pitch: 0 });

const KEYS = ["active", "height", "over", "width", "x", "y"];

/** A cursor frame from the wire, or null when it isn't one. */
export function parseCursor(v: unknown): CursorFrame | null {
  if (typeof v !== "object" || v === null) return null;
  const o = v as Record<string, unknown>;
  if (!KEYS.every((k) => k in o)) return null;
  if (typeof o.active !== "boolean" || typeof o.over !== "boolean") return null;
  for (const k of ["x", "y", "width", "height"]) {
    if (typeof o[k] !== "number" || !Number.isFinite(o[k])) return null;
  }
  return o as unknown as CursorFrame;
}

/** The cursor from the shell now; null if it can't say. */
export async function fetchCursor(): Promise<CursorFrame | null> {
  try {
    return parseCursor(await invoke<unknown>(CURSOR_CMD));
  } catch {
    return null;
  }
}

function clamp(v: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, v));
}

const DEG = 180 / Math.PI;

/**
 * The gaze toward the cursor `c`, for a page `view` CSS pixels big, from his
 * head at `head` (CSS pixels on the page). Ahead when there is nothing to
 * look at.
 */
export function gaze(c: CursorFrame, view: { width: number; height: number }, head: { x: number; y: number }): Gaze {
  if (!c.active || c.width <= 0 || c.height <= 0 || view.width <= 0 || view.height <= 0) return AHEAD;
  // Physical to CSS pixels, by the window's own size: the page needn't know
  // which monitor's scale it is on.
  const x = (c.x * view.width) / c.width;
  const y = (c.y * view.height) / c.height;
  const d = VIEW_DISTANCE * view.height;
  return {
    yaw: clamp(Math.atan2(x - head.x, d) * DEG, MAX_YAW),
    pitch: clamp(Math.atan2(head.y - y, d) * DEG, MAX_PITCH),
  };
}

/** `current` moved toward `target` over `dt` seconds, with time constant
 * `tau`: never past it, and not at all for a step of no time. */
export function approach(current: number, target: number, dt: number, tau: number): number {
  if (!(dt > 0)) return current;
  return current + (target - current) * (1 - Math.exp(-dt / tau));
}
