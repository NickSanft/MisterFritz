/**
 * The avatar page's side of the brain's connection: it holds the device's
 * one session (plan 14 build § 4.3; the control window never opens a
 * socket), configured by the shell and reported to it.
 *
 * - The config (`companion_config`, which answers this window only) is the
 *   brain's URL and token and this desk's id and name. It is asked for at
 *   start, again when `brain.*` changes, and when the shell says the token
 *   was saved or Reconnect pressed (`companion:config`), which connects
 *   again even if nothing changed.
 * - Every change of the link goes to the shell (`companion_report`), which
 *   keeps it for the control window and tells every window.
 *
 * The token stays in this page and the shell: it is never logged or sent
 * to another window.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { CompanionClient, CompanionConfig, LinkStatus } from "../shared/companion";
import { SETTINGS_EVENT, parseBrain } from "../shared/settings";

export const CONFIG_CMD = "companion_config";
export const REPORT_CMD = "companion_report";
/** The token was saved, or Reconnect pressed: ask for the config again. */
export const CONFIG_EVENT = "companion:config";
/** The shell's bridge for a turn: `{text, id}` to say to the brain (from
 * Settings › Brain's Ask, or the prompt). The page takes it first
 * (`companion_take`): the shell says yes only while it still waits for it,
 * so a turn the shell has given up on (and said so) is never asked, and the
 * owner's "try again" never asks it twice. Then the page asks it, and
 * answers `companion_said` with whether it went on, so the window that
 * asked hears when the link had dropped. */
export const SAY_EVENT = "companion:say";
export const TAKE_CMD = "companion_take";
export const SAID_CMD = "companion_said";

/** A turn the shell hands over: what to say, and its number. */
export interface Say {
  text: string;
  id: number;
}

/** A `companion:say`, or null when it isn't one. */
export function parseSay(v: unknown): Say | null {
  const o = obj(v);
  if (!o || typeof o.text !== "string") return null;
  if (!(typeof o.id === "number" && Number.isSafeInteger(o.id) && o.id > 0)) return null;
  return { text: o.text, id: o.id };
}

/** What the page does with a `companion:say`: takes it, and only if the
 * shell still waits for it, asks it and answers with whether it went on. */
export async function onSay(
  payload: unknown,
  take: (id: number) => Promise<boolean>,
  ask: (text: string) => boolean,
  answer: (id: number, sent: boolean) => Promise<void>,
): Promise<"asked" | "late" | "bad"> {
  const say = parseSay(payload);
  if (say === null) return "bad";
  if (!(await take(say.id))) return "late";
  void answer(say.id, ask(say.text));
  return "asked";
}

export const takeArgs = (id: number) => ({ id });
export const saidArgs = (id: number, sent: boolean) => ({ id, sent });

/** Take turn `id` from the shell: whether it still waits for it. No answer
 * is a no: the page never asks a turn the shell may have given up. */
export async function takeSay(
  id: number,
  call: (cmd: string, args: Record<string, unknown>) => Promise<unknown> = invoke,
): Promise<boolean> {
  try {
    return (await call(TAKE_CMD, takeArgs(id))) === true;
  } catch (e) {
    console.warn(`avatar: ${TAKE_CMD} failed`, e);
    return false;
  }
}

/** Tell the shell whether turn `id` was sent on. */
export async function answerSay(
  id: number,
  sent: boolean,
  call: (cmd: string, args: Record<string, unknown>) => Promise<unknown> = invoke,
): Promise<void> {
  try {
    await call(SAID_CMD, saidArgs(id, sent));
  } catch (e) {
    console.warn(`avatar: ${SAID_CMD} failed`, e);
  }
}

/** What `companion_config` answers: the token is null when there is none. */
export interface ShellConfig {
  url: string;
  token: string | null;
  clientId: string;
  clientName: string;
}

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

export function parseShellConfig(v: unknown): ShellConfig | null {
  const o = obj(v);
  if (!o) return null;
  if (typeof o.url !== "string" || typeof o.clientId !== "string" || typeof o.clientName !== "string") return null;
  if (o.token !== null && typeof o.token !== "string") return null;
  return { url: o.url, token: o.token, clientId: o.clientId, clientName: o.clientName };
}

/** The client's config, or null (unconfigured) without a token. */
export function configFrom(s: ShellConfig): CompanionConfig | null {
  if (s.token === null || s.token === "") return null;
  return { url: s.url, token: s.token, client: { id: s.clientId, name: s.clientName } };
}

// The commands' arguments, built in one place so the wire test checks them.
export const configArgs = () => ({});
export const reportArgs = (s: LinkStatus) => ({
  report: {
    link: s.link,
    reason: s.reason,
    retryIn: s.retryIn,
    identity: s.identity,
    displayName: s.displayName,
    threadId: s.threadId,
  },
});

/** Send each report only once the one before it has been taken, and only
 * the newest of those that came meanwhile: two invokes in flight aren't
 * ordered, and the shell keeps the last it hears (connecting, then online a
 * moment later, must never end as connecting). */
export function oneAtATime(send: (s: LinkStatus) => Promise<void>): (s: LinkStatus) => void {
  let next: LinkStatus | null = null;
  let busy = false;
  const pump = async (): Promise<void> => {
    busy = true;
    while (next) {
      const s = next;
      next = null;
      try {
        await send(s);
      } catch {
        // The sender says why; the next report still goes.
      }
    }
    busy = false;
  };
  return (s) => {
    next = s;
    if (!busy) void pump();
  };
}

export interface LinkDeps {
  ask(): Promise<unknown>;
  report(s: LinkStatus): Promise<void>;
  onConfig(cb: () => void): Promise<unknown>;
  onSettings(cb: (settings: unknown) => void): Promise<unknown>;
}

const tauriDeps = (): LinkDeps => ({
  ask: () => invoke<unknown>(CONFIG_CMD, configArgs()),
  report: async (s) => {
    try {
      await invoke(REPORT_CMD, reportArgs(s));
    } catch (e) {
      console.warn(`avatar: ${REPORT_CMD} failed`, e);
    }
  },
  onConfig: (cb) => listen(CONFIG_EVENT, () => cb()),
  onSettings: (cb) => listen<unknown>(SETTINGS_EVENT, (e) => cb(e.payload)),
});

/** Connect `client` as the shell says, and keep the shell told. Resolves
 * once listening; the first config is asked for without waiting. */
export async function startCompanion(client: CompanionClient, deps: LinkDeps = tauriDeps()): Promise<void> {
  client.onLink(oneAtATime(deps.report));
  // The `brain` settings the client was last configured from, to tell a
  // change to them from a change to anything else.
  let brain: string | null = null;
  const brainKey = (b: { url: string; clientId: string; clientName: string }) =>
    JSON.stringify([b.url, b.clientId, b.clientName]);
  // Only the newest question's answer counts: answers aren't ordered.
  let asked = 0;
  const refresh = async (again: boolean): Promise<void> => {
    const mine = ++asked;
    let answer: unknown;
    try {
      answer = await deps.ask();
    } catch (e) {
      console.warn(`avatar: ${CONFIG_CMD} failed`, e);
      return;
    }
    if (mine !== asked) return;
    const shell = parseShellConfig(answer);
    if (!shell) return;
    brain = brainKey(shell);
    const changed = client.configure(configFrom(shell));
    if (again && !changed) client.reconnectNow();
  };
  await deps.onConfig(() => void refresh(true));
  // Every window hears every setting: only a change to `brain` matters here.
  await deps.onSettings((settings) => {
    const b = parseBrain(obj(settings)?.brain);
    if (!b || brainKey(b) === brain) return;
    brain = brainKey(b);
    void refresh(false);
  });
  void refresh(false);
}
