/**
 * The control window's side of the shell: the commands it calls and what
 * comes back, read strictly. The names and argument keys are the shell's
 * (src/avatar/wire.fixture.json holds both sides to them).
 */
import { invoke } from "@tauri-apps/api/core";
import { parseStatus, type AvatarStatus } from "../avatar/status";
import type { Link, LinkStatus, Reason } from "../shared/companion";
import { SETTINGS_EVENT, parseBrain, parseTheme, type Brain, type Theme } from "../shared/settings";

export const CMD = {
  getSettings: "get_settings",
  setSetting: "set_setting",
  setParked: "set_parked",
  avatarStatus: "avatar_status",
  avatarModel: "avatar_model",
  openFolder: "open_folder",
  reloadModel: "reload_model",
  mode: "set_avatar_mode",
  controlOpening: "control_opening",
  getTokenPresent: "get_token_present",
  setToken: "set_token",
  companionStatus: "companion_status",
  companionConnect: "companion_connect",
  companionSay: "companion_say",
} as const;

export { SETTINGS_EVENT };
/** The avatar's page has loaded a model: ask `avatar_model` again. */
export const LOADED_EVENT = "avatar:loaded";
/** The avatar's page reported the brain's link (a `LinkStatus`). */
export const COMPANION_STATUS_EVENT = "companion:status";

export type Folder = "logs" | "avatar";

/** The window's sections, by their ids, in order: what `control_opening`
 * may answer (the shell's CONTROL_SECTIONS; the fixture holds both). */
export const SECTIONS = ["brain", "fritz", "appearance", "model", "diagnostics", "about"] as const;
export type SectionId = (typeof SECTIONS)[number];

export function parseSection(v: unknown): SectionId | null {
  return (SECTIONS as readonly unknown[]).includes(v) ? (v as SectionId) : null;
}

/** The settings the control window reads (the shell has more). */
export interface ControlSettings {
  brain: Brain;
  avatar: { lookatCursor: boolean; vrmPath: string | null };
  hotkeys: { arrange: string; park: string };
  diagnostics: { foregroundLog: boolean };
  theme: Theme;
}

export interface ModelInfo {
  from: "setting" | "folder";
  path: string;
  found: boolean;
  bytes: number | null;
  tooBig: boolean;
  /** What the avatar's page says it last loaded; null before its first. */
  wearing: "export" | "placeholder" | null;
  /** Why the export there now didn't load. */
  loadError: string | null;
  /** The file there isn't the one the page last loaded: what he wears is
   * from before, and loading again wears what's there. */
  changed: boolean;
}

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Settings from the wire, or null when they aren't settings. */
export function parseSettings(v: unknown): ControlSettings | null {
  const o = obj(v);
  const avatar = obj(o?.avatar);
  const hotkeys = obj(o?.hotkeys);
  const diagnostics = obj(o?.diagnostics);
  const theme = parseTheme(o?.theme);
  const brain = parseBrain(o?.brain);
  if (!brain || !avatar || !hotkeys || !diagnostics || !theme) return null;
  if (typeof avatar.lookatCursor !== "boolean") return null;
  if (avatar.vrmPath !== null && typeof avatar.vrmPath !== "string") return null;
  if (typeof hotkeys.arrange !== "string" || typeof hotkeys.park !== "string") return null;
  if (typeof diagnostics.foregroundLog !== "boolean") return null;
  return {
    brain,
    avatar: { lookatCursor: avatar.lookatCursor, vrmPath: avatar.vrmPath as string | null },
    hotkeys: { arrange: hotkeys.arrange, park: hotkeys.park },
    diagnostics: { foregroundLog: diagnostics.foregroundLog },
    theme,
  };
}

export function parseModelInfo(v: unknown): ModelInfo | null {
  const o = obj(v);
  if (!o) return null;
  if (o.from !== "setting" && o.from !== "folder") return null;
  if (typeof o.path !== "string" || typeof o.found !== "boolean" || typeof o.tooBig !== "boolean") return null;
  if (o.bytes !== null && (typeof o.bytes !== "number" || !(o.bytes >= 0))) return null;
  if (o.wearing !== null && o.wearing !== "export" && o.wearing !== "placeholder") return null;
  if (o.loadError !== null && typeof o.loadError !== "string") return null;
  if (typeof o.changed !== "boolean") return null;
  return {
    from: o.from,
    path: o.path,
    found: o.found,
    bytes: o.bytes as number | null,
    tooBig: o.tooBig,
    wearing: o.wearing as ModelInfo["wearing"],
    loadError: o.loadError as string | null,
    changed: o.changed,
  };
}

const LINKS: readonly Link[] = ["unconfigured", "connecting", "online", "offline", "refused"];
const REASONS: readonly Reason[] = ["token", "version", "hello", "full", "quiet", "unreachable", null];

/** The link as the avatar's page reported it, or null when it isn't one. */
export function parseCompanionStatus(v: unknown): LinkStatus | null {
  const o = obj(v);
  if (!o) return null;
  if (!LINKS.includes(o.link as Link) || !REASONS.includes(o.reason as Reason)) return null;
  if (o.retryIn !== null && !(typeof o.retryIn === "number" && Number.isInteger(o.retryIn) && o.retryIn >= 0)) return null;
  for (const k of ["identity", "displayName", "threadId"]) {
    if (o[k] !== null && typeof o[k] !== "string") return null;
  }
  return {
    link: o.link as Link,
    reason: o.reason as Reason,
    retryIn: o.retryIn as number | null,
    identity: o.identity as string | null,
    displayName: o.displayName as string | null,
    threadId: o.threadId as string | null,
  };
}

/** The argument objects, by the keys the shell's commands take. */
export const args = {
  setSetting: (path: string, value: unknown) => ({ path, value }),
  setParked: (on: boolean) => ({ on }),
  openFolder: (which: Folder) => ({ which }),
  mode: (mode: "arrange" | "locked") => ({ mode }),
  setToken: (token: string) => ({ token }),
  companionSay: (text: string) => ({ text }),
};

export async function getSettings(): Promise<ControlSettings | null> {
  return parseSettings(await invoke<unknown>(CMD.getSettings));
}

/** Change one setting; the settings after it, or the shell's refusal. */
export async function setSetting(path: string, value: unknown): Promise<ControlSettings | null> {
  return parseSettings(await invoke<unknown>(CMD.setSetting, args.setSetting(path, value)));
}

export async function setParked(on: boolean): Promise<AvatarStatus | null> {
  return parseStatus(await invoke<unknown>(CMD.setParked, args.setParked(on)));
}

export async function avatarStatus(): Promise<AvatarStatus | null> {
  return parseStatus(await invoke<unknown>(CMD.avatarStatus));
}

export async function avatarModel(): Promise<ModelInfo | null> {
  return parseModelInfo(await invoke<unknown>(CMD.avatarModel));
}

export async function openFolder(which: Folder): Promise<void> {
  await invoke(CMD.openFolder, args.openFolder(which));
}

export async function reloadModel(): Promise<void> {
  await invoke(CMD.reloadModel);
}

/** The section to open at (`FRITZ_SHOW_SETTINGS=appearance`), if any. */
export async function controlOpening(): Promise<SectionId | null> {
  return parseSection(await invoke<unknown>(CMD.controlOpening));
}

export async function arrange(): Promise<void> {
  await invoke(CMD.mode, args.mode("arrange"));
}

/** Whether there is a token (the window is never given the token itself). */
export async function getTokenPresent(): Promise<boolean> {
  return (await invoke<unknown>(CMD.getTokenPresent)) === true;
}

/** Save the token the owner pasted; the shell's refusal says why not. The
 * avatar's page connects with it at once. */
export async function setToken(token: string): Promise<void> {
  await invoke(CMD.setToken, args.setToken(token));
}

export async function companionStatus(): Promise<LinkStatus | null> {
  return parseCompanionStatus(await invoke<unknown>(CMD.companionStatus));
}

/** Have the avatar's page connect again now. */
export async function companionConnect(): Promise<void> {
  await invoke(CMD.companionConnect);
}

/** Ask the brain something, through his window: the reply shows in his
 * bubble. The shell's refusal says why not (nothing to ask, his window
 * closed, not connected). */
export async function companionSay(text: string): Promise<void> {
  await invoke(CMD.companionSay, args.companionSay(text));
}
