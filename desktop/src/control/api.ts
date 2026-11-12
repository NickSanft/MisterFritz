/**
 * The control window's side of the shell: the commands it calls and what
 * comes back, read strictly. The names and argument keys are the shell's
 * (src/avatar/wire.fixture.json holds both sides to them).
 */
import { invoke } from "@tauri-apps/api/core";
import { parseStatus, type AvatarStatus } from "../avatar/status";

export const CMD = {
  getSettings: "get_settings",
  setSetting: "set_setting",
  setParked: "set_parked",
  avatarStatus: "avatar_status",
  avatarModel: "avatar_model",
  openFolder: "open_folder",
  reloadModel: "reload_model",
  mode: "set_avatar_mode",
} as const;

export const SETTINGS_EVENT = "settings:changed";
/** The avatar's page has loaded a model: ask `avatar_model` again. */
export const LOADED_EVENT = "avatar:loaded";

export type Folder = "logs" | "avatar";

/** The settings the control window reads (the shell has more). */
export interface ControlSettings {
  avatar: { lookatCursor: boolean; vrmPath: string | null };
  hotkeys: { arrange: string; park: string };
  diagnostics: { foregroundLog: boolean };
  theme: { theme: string; mood: string; accent: string };
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
  const theme = obj(o?.theme);
  if (!avatar || !hotkeys || !diagnostics || !theme) return null;
  if (typeof avatar.lookatCursor !== "boolean") return null;
  if (avatar.vrmPath !== null && typeof avatar.vrmPath !== "string") return null;
  if (typeof hotkeys.arrange !== "string" || typeof hotkeys.park !== "string") return null;
  if (typeof diagnostics.foregroundLog !== "boolean") return null;
  for (const k of ["theme", "mood", "accent"]) if (typeof theme[k] !== "string") return null;
  return {
    avatar: { lookatCursor: avatar.lookatCursor, vrmPath: avatar.vrmPath as string | null },
    hotkeys: { arrange: hotkeys.arrange, park: hotkeys.park },
    diagnostics: { foregroundLog: diagnostics.foregroundLog },
    theme: { theme: theme.theme as string, mood: theme.mood as string, accent: theme.accent as string },
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

/** The argument objects, by the keys the shell's commands take. */
export const args = {
  setSetting: (path: string, value: unknown) => ({ path, value }),
  setParked: (on: boolean) => ({ on }),
  openFolder: (which: Folder) => ({ which }),
  mode: (mode: "arrange" | "locked") => ({ mode }),
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

export async function arrange(): Promise<void> {
  await invoke(CMD.mode, args.mode("arrange"));
}
