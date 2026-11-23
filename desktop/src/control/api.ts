/**
 * The control window's side of the shell: the commands it calls and what
 * comes back, read strictly. The names and argument keys are the shell's
 * (src/avatar/wire.fixture.json holds both sides to them).
 */
import { invoke } from "@tauri-apps/api/core";
import { parseStatus, type AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { parseCompanionStatus } from "../shared/link";
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
  setHotkey: "set_hotkey",
  hotkeysBound: "hotkeys_bound",
  mouseHookStatus: "mouse_hook_status",
} as const;

/** The actions a key may be set for, in the order Settings shows them
 * (the shell's hotkeys::ACTIONS; the fixture's hotkeyActions holds both). */
export const HOTKEY_ACTIONS = ["prompt", "arrange", "park", "ptt"] as const;
export type HotkeyAction = (typeof HOTKEY_ACTIONS)[number];

/** Each action's key by default (the shell's `Hotkeys::default()`). */
export const HOTKEY_DEFAULTS: Record<HotkeyAction, string> = {
  prompt: "Ctrl+Alt+F",
  arrange: "Ctrl+Alt+A",
  park: "Ctrl+Alt+H",
  ptt: "Ctrl+Alt+V",
};

/** Each action's key as Windows holds it now, or null when it has none. */
export type HotkeysBound = Record<HotkeyAction, string | null>;

export { SETTINGS_EVENT };
/** The avatar's page has loaded a model: ask `avatar_model` again. */
export const LOADED_EVENT = "avatar:loaded";
export { COMPANION_STATUS_EVENT } from "../shared/link";
/** The mouse hook went in or out, was put back or failed: its status. */
export const HOOK_STATUS_EVENT = "hook:status";

export type Folder = "logs" | "avatar";

/** The window's sections, by their ids, in order: what `control_opening`
 * may answer (the shell's CONTROL_SECTIONS; the fixture holds both). */
export const SECTIONS = ["brain", "fritz", "hotkeys", "appearance", "model", "diagnostics", "about"] as const;
export type SectionId = (typeof SECTIONS)[number];

export function parseSection(v: unknown): SectionId | null {
  return (SECTIONS as readonly unknown[]).includes(v) ? (v as SectionId) : null;
}

/** What `control_opening` answers for the first run (the shell's
 * CONTROL_WIZARD), which takes the sections' place while it is open. */
export const WIZARD = "wizard";
/** Where the window opens: a section, or the first run. */
export type Opening = SectionId | typeof WIZARD;

export function parseOpening(v: unknown): Opening | null {
  return v === WIZARD ? WIZARD : parseSection(v);
}

/** The settings the control window reads (the shell has more). */
export interface ControlSettings {
  brain: Brain;
  avatar: { lookatCursor: boolean; clicks: boolean; vrmPath: string | null };
  hotkeys: Record<HotkeyAction, string>;
  /** Push-to-talk muted, from the tray's Mute push-to-talk only. */
  audio: { muted: boolean };
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
  const audio = obj(o?.audio);
  const diagnostics = obj(o?.diagnostics);
  const theme = parseTheme(o?.theme);
  const brain = parseBrain(o?.brain);
  if (!brain || !avatar || !hotkeys || !audio || !diagnostics || !theme) return null;
  if (typeof avatar.lookatCursor !== "boolean" || typeof avatar.clicks !== "boolean") return null;
  if (avatar.vrmPath !== null && typeof avatar.vrmPath !== "string") return null;
  if (!HOTKEY_ACTIONS.every((a) => typeof hotkeys[a] === "string")) return null;
  if (typeof audio.muted !== "boolean") return null;
  if (typeof diagnostics.foregroundLog !== "boolean") return null;
  return {
    brain,
    avatar: { lookatCursor: avatar.lookatCursor, clicks: avatar.clicks, vrmPath: avatar.vrmPath as string | null },
    hotkeys: Object.fromEntries(HOTKEY_ACTIONS.map((a) => [a, hotkeys[a] as string])) as Record<HotkeyAction, string>,
    audio: { muted: audio.muted },
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

export { parseCompanionStatus } from "../shared/link";

/** The mouse hook, as `mouse_hook_status` reports it: in or out and why,
 * and its counts. Never a point. */
export interface HookStatus {
  /** The switch (Settings › Fritz › Clicks on him). */
  on: boolean;
  installed: boolean;
  /** Why it is out; null while it is in. */
  why: string | null;
  events: number;
  presses: number;
  onShape: number;
  covered: number;
  taken: number;
  gestures: number;
  slowestMicros: number;
  putBack: number;
  panicked: boolean;
}

const HOOK_COUNTS = ["events", "presses", "onShape", "covered", "taken", "gestures", "slowestMicros", "putBack"] as const;
/** The status's keys, sorted (the shell's; the fixture's hookStatusKeys). */
export const HOOK_KEYS = [...HOOK_COUNTS, "on", "installed", "why", "panicked"].sort();

/** The hook's status, or null when it isn't one: exactly its keys, each of
 * its kind, every count a whole number of at least 0. */
export function parseHookStatus(v: unknown): HookStatus | null {
  const o = obj(v);
  if (!o || Object.keys(o).sort().join() !== HOOK_KEYS.join()) return null;
  if (typeof o.on !== "boolean" || typeof o.installed !== "boolean" || typeof o.panicked !== "boolean") return null;
  if (o.why !== null && typeof o.why !== "string") return null;
  for (const k of HOOK_COUNTS) {
    const n = o[k];
    if (typeof n !== "number" || !Number.isSafeInteger(n) || n < 0) return null;
  }
  return o as unknown as HookStatus;
}

/** `hotkeys_bound`'s answer, or null when it isn't one: exactly the
 * actions, each a key or null. */
export function parseHotkeysBound(v: unknown): HotkeysBound | null {
  const o = obj(v);
  if (!o || Object.keys(o).sort().join() !== [...HOTKEY_ACTIONS].sort().join()) return null;
  for (const a of HOTKEY_ACTIONS) {
    if (o[a] !== null && typeof o[a] !== "string") return null;
  }
  return Object.fromEntries(HOTKEY_ACTIONS.map((a) => [a, o[a] as string | null])) as HotkeysBound;
}

/** The argument objects, by the keys the shell's commands take. */
export const args = {
  setSetting: (path: string, value: unknown) => ({ path, value }),
  setParked: (on: boolean) => ({ on }),
  openFolder: (which: Folder) => ({ which }),
  mode: (mode: "arrange" | "locked") => ({ mode }),
  setToken: (token: string) => ({ token }),
  companionSay: (text: string) => ({ text }),
  setHotkey: (action: HotkeyAction, accelerator: string) => ({ action, accelerator }),
};

/** Bind a key to an action and save it: the settings after it, or the
 * shell's refusal (the old key still working). */
export async function setHotkey(action: HotkeyAction, accelerator: string): Promise<ControlSettings | null> {
  return parseSettings(await invoke<unknown>(CMD.setHotkey, args.setHotkey(action, accelerator)));
}

/** Each action's key as Windows holds it now. */
export async function hotkeysBound(): Promise<HotkeysBound | null> {
  return parseHotkeysBound(await invoke<unknown>(CMD.hotkeysBound));
}

/** The mouse hook now. */
export async function mouseHookStatus(): Promise<HookStatus | null> {
  return parseHookStatus(await invoke<unknown>(CMD.mouseHookStatus));
}

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

/** Where to open (`FRITZ_SHOW_SETTINGS=appearance`, or the first run with
 * no token yet), if anywhere. */
export async function controlOpening(): Promise<Opening | null> {
  return parseOpening(await invoke<unknown>(CMD.controlOpening));
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
