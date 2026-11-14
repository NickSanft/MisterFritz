/**
 * When the avatar page draws: one decision, from everything that can stop
 * it, so no one event can restart the loop while another reason to stop
 * still holds (a status change while the display is off, say).
 */
import { invoke } from "@tauri-apps/api/core";
import type { AvatarStatus } from "./status";

/** The shell's display-power event: `true` while the display is on (or
 * dimmed), `false` while it is off. The lock screen is the cover watch's
 * business, and arrives as `parked`. */
export const POWER_EVENT = "power:display";
/** The display state now, for a page that starts after the last event. */
export const POWER_CMD = "power_display";

export function parseDisplay(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

/** The display state from the shell; on, if it can't say. */
export async function fetchDisplayOn(): Promise<boolean> {
  try {
    return parseDisplay(await invoke<unknown>(POWER_CMD)) ?? true;
  } catch {
    return true;
  }
}

/** Shared by every page now (src/shared/asking.ts). */
export { afterAsking } from "../shared/asking";

/** Whether the paced loop should be drawing. Not before the shell has said
 * anything, not while he is parked (hidden, or stepped aside for a
 * fullscreen app or the lock screen), not while his window is going, and
 * not while the display is off: the compositor would have nothing to show.
 *
 * Except in Arrange, which goes on whatever the display does: the loop is
 * what beats the shell's lease, and a lease that hears nothing ends Arrange
 * with "Fritz stopped responding". Arrange is short, and never parked. */
export function shouldDraw(
  status: Pick<AvatarStatus, "parked" | "phase" | "mode"> | null,
  displayOn: boolean,
): boolean {
  if (status === null || status.parked) return false;
  if (status.phase === "closing" || status.phase === "closed") return false;
  return displayOn || status.mode === "arrange";
}
