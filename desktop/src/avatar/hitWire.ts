/**
 * His shape's two commands (the shell's names and argument keys, which
 * src/avatar/wire.fixture.json holds both sides to): the page asks which
 * load of it this is, by the shell's count, and sends where his body is on
 * it (hitShape.ts), stamped with that number and its own count of sends.
 * The shell keeps only the latest send of the current load.
 */
import { invoke } from "@tauri-apps/api/core";
import type { HitSend } from "./hitShape";

export const HIT_RECTS_CMD = "set_hit_rects";
export const PAGE_CMD = "avatar_page";

export const hitArgs = (instance: number, page: number, s: HitSend) => ({
  instance,
  page,
  seq: s.seq,
  view: s.view,
  rects: s.rects,
});

/** Whether the shell kept the shape: false when it took it for an older
 * send, null when it refused it (it says why) or couldn't be reached. */
export async function sendHitRects(args: ReturnType<typeof hitArgs>): Promise<boolean | null> {
  try {
    return (await invoke<unknown>(HIT_RECTS_CMD, args)) === true;
  } catch (e) {
    console.warn(`avatar: ${HIT_RECTS_CMD} failed`, e);
    return null;
  }
}

/** A page number from the wire: a whole number from 1, else null. */
export function parsePage(v: unknown): number | null {
  return typeof v === "number" && Number.isInteger(v) && v >= 1 ? v : null;
}

/** Which load of this page the shell counts it as; null if it can't say. */
export async function fetchPage(): Promise<number | null> {
  try {
    return parsePage(await invoke<unknown>(PAGE_CMD));
  } catch (e) {
    console.warn(`avatar: ${PAGE_CMD} failed`, e);
    return null;
  }
}
