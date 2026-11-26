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
import { oneAtATime as newestOnly } from "../shared/oneAtATime";
import { voiceReady } from "./voiceLink";

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

// ── Push-to-talk (PR 8): what the shell's ears say, and the clip ─────────

/** `{press}`: the shell opened the microphone for the brain, for press
 * number `press`: he is listening. Its outcome (`ptt:nothing` or
 * `ptt:clip`) carries the same number. */
export const PTT_LISTENING_EVENT = "ptt:listening";
/** `{why}`: push-to-talk couldn't be heard, and the microphone stayed shut. */
export const PTT_REFUSED_EVENT = "ptt:refused";
/** `{press, why}`: a clip came to nothing (too short, or the microphone failed). */
export const PTT_NOTHING_EVENT = "ptt:nothing";
/** `{id, millis, press}`: a clip waits for the page, which takes it (`ptt_take`,
 * the raw PCM16 as an ArrayBuffer), sends it to the brain, and says whether
 * it went on (`ptt_sent`). */
export const PTT_CLIP_EVENT = "ptt:clip";
/** `{press, level}`, 20 times a second while he listens: the RMS of what
 * the microphone heard since the last, 0 to 1, for his bubble's meter. */
export const PTT_LEVEL_EVENT = "ptt:level";
export const PTT_TAKE_CMD = "ptt_take";
export const PTT_SENT_CMD = "ptt_sent";

/** A clip waiting for the page: its number, how long it is, and its press. */
export interface Clip {
  id: number;
  millis: number;
  press: number;
}

const whole = (n: unknown): n is number => typeof n === "number" && Number.isSafeInteger(n);

/** A `ptt:listening`'s press, or null when it isn't one. */
export function parseListening(v: unknown): number | null {
  const o = obj(v);
  return o && whole(o.press) && o.press > 0 ? o.press : null;
}

/** A `ptt:nothing`, or null when it isn't one. */
export function parseNothing(v: unknown): { press: number; why: string } | null {
  const why = parseWhy(v);
  const press = parseListening(v);
  return why === null || press === null ? null : { press, why };
}

/** A `ptt:level`, or null when it isn't one. */
export function parseLevel(v: unknown): { press: number; level: number } | null {
  const o = obj(v);
  const press = parseListening(v);
  if (!o || press === null || typeof o.level !== "number" || !(o.level >= 0 && o.level <= 1)) return null;
  return { press, level: o.level };
}

/** A `{why}`'s words, or null when it isn't one. */
export function parseWhy(v: unknown): string | null {
  const o = obj(v);
  return o && typeof o.why === "string" && o.why.trim() !== "" ? o.why : null;
}

/** A `ptt:clip`, or null when it isn't one. */
export function parseClip(v: unknown): Clip | null {
  const o = obj(v);
  if (!o) return null;
  const press = parseListening(v);
  if (!(whole(o.id) && o.id > 0 && whole(o.millis) && o.millis >= 0) || press === null) return null;
  return { id: o.id, millis: o.millis, press };
}

export const pttTakeArgs = (id: number) => ({ id });
export const pttSentArgs = (id: number, sent: boolean, why: string | null) => ({ id, sent, why });

/** Take clip `id` from the shell: its bytes, or null when it isn't waiting
 * any more (or the answer isn't bytes). */
export async function takeClip(
  id: number,
  call: (cmd: string, args: Record<string, unknown>) => Promise<unknown> = invoke,
): Promise<ArrayBuffer | null> {
  try {
    const got = await call(PTT_TAKE_CMD, pttTakeArgs(id));
    return got instanceof ArrayBuffer ? got : null;
  } catch (e) {
    console.warn(`avatar: ${PTT_TAKE_CMD} failed`, e);
    return null;
  }
}

/** Tell the shell whether clip `id` went on, and if not, why. */
export async function answerClip(
  id: number,
  sent: boolean,
  why: string | null,
  call: (cmd: string, args: Record<string, unknown>) => Promise<unknown> = invoke,
): Promise<void> {
  try {
    await call(PTT_SENT_CMD, pttSentArgs(id, sent, why));
  } catch (e) {
    console.warn(`avatar: ${PTT_SENT_CMD} failed`, e);
  }
}

/** What became of a clip the shell handed over. */
export type ClipOutcome =
  | { kind: "sent"; turnId: string }
  | { kind: "unsent"; why: string; press: number }
  | { kind: "bad" };

/** Why a clip can't be heard: the shell's words (companion.rs's
 * BRAIN_CANT_HEAR and NOT_CONNECTED; the wire fixture holds both to them). */
export const BRAIN_CANT_HEAR = "His brain can't hear yet: type to him instead.";
export const NOT_CONNECTED = "He isn't connected to his brain just now.";

/** A clip not sent because a newer press is listening (for the log: the
 * page shows the newer press). */
export const SUPERSEDED = "Spoken again before it was sent: only the newer clip goes to him.";

/** A clip couldn't be taken from the shell (it waited too long). */
export const CLIP_GONE = "What you said was lost on its way to his brain: say it again.";

/** What the page does with a `ptt:clip`: takes it, sends it (`send` gives
 * the turn's id, or why not), and answers the shell either way. */
export async function onClip(
  payload: unknown,
  take: (id: number) => Promise<ArrayBuffer | null>,
  send: (pcm: ArrayBuffer, press: number) => { turnId: string } | { why: string },
  answer: (id: number, sent: boolean, why: string | null) => Promise<void>,
): Promise<ClipOutcome> {
  const clip = parseClip(payload);
  if (clip === null) return { kind: "bad" };
  const pcm = await take(clip.id);
  if (pcm === null) {
    void answer(clip.id, false, CLIP_GONE);
    return { kind: "unsent", why: CLIP_GONE, press: clip.press };
  }
  const r = send(pcm, clip.press);
  if ("turnId" in r) {
    void answer(clip.id, true, null);
    return { kind: "sent", turnId: r.turnId };
  }
  void answer(clip.id, false, r.why);
  return { kind: "unsent", why: r.why, press: clip.press };
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
/** The client's config from the shell's, and whether his voice is
 * installed (the desk then offers to voice what he says). */
export function configFrom(s: ShellConfig, speaks = false): CompanionConfig | null {
  if (s.token === null || s.token === "") return null;
  return { url: s.url, token: s.token, client: { id: s.clientId, name: s.clientName }, speaks };
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
    serverCaps: [...s.serverCaps],
  },
});

/** Each report once the one before it has been taken, and only the newest
 * of those that came meanwhile (shared/oneAtATime.ts): connecting, then
 * online a moment later, must never end as connecting. */
export const oneAtATime = (send: (s: LinkStatus) => Promise<void>): ((s: LinkStatus) => void) => newestOnly(send);

export interface LinkDeps {
  ask(): Promise<unknown>;
  report(s: LinkStatus): Promise<void>;
  onConfig(cb: () => void): Promise<unknown>;
  onSettings(cb: (settings: unknown) => void): Promise<unknown>;
  /** Whether his voice is installed (asked once, at the start: files
   * fetched later count from the next start). Absent: it isn't. */
  voiceReady?(): Promise<boolean>;
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
  voiceReady: () => voiceReady(),
});

/** Connect `client` as the shell says, and keep the shell told. Resolves
 * once listening; the first config is asked for without waiting. */
export async function startCompanion(client: CompanionClient, deps: LinkDeps = tauriDeps()): Promise<void> {
  client.onLink(oneAtATime(deps.report));
  // Asked at once, and waited for only by the first config.
  const speaks = deps.voiceReady ? deps.voiceReady().catch(() => false) : Promise.resolve(false);
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
    const voice = await speaks;
    if (mine !== asked) return;
    const shell = parseShellConfig(answer);
    if (!shell) return;
    brain = brainKey(shell);
    const changed = client.configure(configFrom(shell, voice));
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
