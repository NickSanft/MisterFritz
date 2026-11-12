// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/overlay/status.ts at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.

/**
 * The avatar window's side of the shell's handshake: the status the shell
 * emits on every change, and the commands the page answers with. Commands,
 * never events, for the page's half: tauri parks Rust event listeners on lock
 * contention.
 *
 * The page learns status only from `avatar:status` events, which the shell
 * emits under its lock in the order the state changed. A command's reply is
 * not applied: replies aren't ordered against events, so a stale one could
 * undo a newer park (DivoraVoice's page works the same way).
 */
import { invoke } from "@tauri-apps/api/core";

export const STATUS_EVENT = "avatar:status";
export const HANDSHAKE_CMD = "avatar_handshake";
export const BEAT_CMD = "avatar_beat";
export const MODE_CMD = "set_avatar_mode";
export const VRM_CMD = "avatar_vrm";
/** The control window's "Load it again": the page loads its model anew. */
export const RELOAD_EVENT = "avatar:reload";

/** Set by an init script when FRITZ_PRETEND_AVATAR_NO_READY is set in a debug
 * build: the page then skips its ready handshake, so the watchdog can be seen
 * to close it. */
export const NO_READY_GLOBAL = "__FRITZ_AVATAR_NO_READY__";

export type Mode = "locked" | "arrange";
export type PhaseName = "closed" | "starting" | "shown" | "painted" | "closing";
export type HandshakeStage = "ready" | "painted";
export type ModeRequest = "locked" | "arrange" | "toggle";

/** The shell's OverlayStatus, as it goes over the wire. */
export interface AvatarStatus {
  open: boolean;
  phase: PhaseName;
  instance: number;
  mode: Mode;
  arrangeGen: number;
  canArrange: boolean;
  fit: boolean;
  parked: boolean;
  error: string | null;
}

const PHASES: readonly string[] = ["closed", "starting", "shown", "painted", "closing"];
const MODES: readonly string[] = ["locked", "arrange"];

function count(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 0;
}

/** A status from the wire, or null when it isn't one. */
export function parseStatus(v: unknown): AvatarStatus | null {
  if (typeof v !== "object" || v === null) return null;
  const o = v as Record<string, unknown>;
  if (!count(o.instance) || !count(o.arrangeGen)) return null;
  if (typeof o.phase !== "string" || !PHASES.includes(o.phase)) return null;
  if (typeof o.mode !== "string" || !MODES.includes(o.mode)) return null;
  for (const flag of ["open", "canArrange", "fit", "parked"]) {
    if (typeof o[flag] !== "boolean") return null;
  }
  if (o.error !== null && typeof o.error !== "string") return null;
  return {
    open: o.open as boolean,
    phase: o.phase as PhaseName,
    instance: o.instance,
    mode: o.mode as Mode,
    arrangeGen: o.arrangeGen,
    canArrange: o.canArrange as boolean,
    fit: o.fit as boolean,
    parked: o.parked as boolean,
    error: (o.error as string | null) ?? null,
  };
}

/** This window's number, from `avatar.html?instance=<n>`; 0 without one. */
export function pageInstance(search: string): number {
  const n = Number(new URLSearchParams(search).get("instance"));
  return Number.isInteger(n) && n > 0 ? n : 0;
}

// The commands' arguments, built in one place so the wire-contract test
// checks exactly what is sent.
export const handshakeArgs = (instance: number, stage: HandshakeStage) => ({ instance, stage });
export const beatArgs = (instance: number, gen: number) => ({ instance, gen });
export const modeArgs = (mode: ModeRequest) => ({ mode });
export const vrmArgs = () => ({});

export async function sendHandshake(instance: number, stage: HandshakeStage): Promise<void> {
  try {
    await invoke(HANDSHAKE_CMD, handshakeArgs(instance, stage));
  } catch (e) {
    console.warn(`avatar: ${HANDSHAKE_CMD}(${stage}) failed`, e);
  }
}

export async function sendBeat(instance: number, gen: number): Promise<void> {
  try {
    await invoke(BEAT_CMD, beatArgs(instance, gen));
  } catch (e) {
    console.warn(`avatar: ${BEAT_CMD} failed`, e);
  }
}

export async function requestLock(): Promise<void> {
  try {
    await invoke(MODE_CMD, modeArgs("locked"));
  } catch (e) {
    console.warn(`avatar: ${MODE_CMD}(locked) failed`, e);
  }
}

/** The owner's VRoid export as raw bytes; rejects when there is none. */
export function fetchOwnersModel(): Promise<ArrayBuffer> {
  return invoke<ArrayBuffer>(VRM_CMD, vrmArgs());
}

/** After every load: what the page wears now, and why not the export when
 * there was one it couldn't use. The shell keeps it for the control window. */
export const LOADED_CMD = "avatar_loaded";
export type Wearing = "export" | "placeholder";
export const loadedArgs = (wearing: Wearing, error: string | null) => ({ wearing, error });
export async function reportLoaded(wearing: Wearing, error: string | null): Promise<void> {
  try {
    await invoke(LOADED_CMD, loadedArgs(wearing, error));
  } catch (e) {
    console.error("avatar: couldn't tell the shell which model loaded", e);
  }
}

/** What a failed `avatar_vrm` means: "none" is no file there (not an
 * error, he just wears the placeholder); anything else is why the file
 * couldn't be had (too large, unreadable). */
export function vrmFetchError(e: unknown): string | null {
  const text = typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
  return text === "none" ? null : text;
}
