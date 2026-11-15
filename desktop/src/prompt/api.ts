/**
 * The prompt's side of the shell: the commands it calls and what comes
 * back, read strictly. The names and argument keys are the shell's
 * (src/avatar/wire.fixture.json holds both sides to them).
 */
import { invoke } from "@tauri-apps/api/core";
import { parseStatus, type AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { parseCompanionStatus } from "../shared/link";

export const CMD = {
  say: "prompt_say",
  close: "prompt_close",
  ready: "prompt_ready",
  companionStatus: "companion_status",
  avatarStatus: "avatar_status",
} as const;

/** The argument objects, by the keys the shell's commands take. */
export const args = {
  say: (text: string) => ({ text }),
};

/** The shell's refusal as words. */
function words(e: unknown): string {
  if (typeof e === "string") return e;
  return e instanceof Error ? e.message : String(e);
}

/** Ask him `text`: null once it is taken (the shell then closes the
 * prompt), else why not, in the shell's words. */
export async function say(text: string): Promise<string | null> {
  try {
    await invoke(CMD.say, args.say(text));
    return null;
  } catch (e) {
    return words(e);
  }
}

/** The page has mounted and listens: the shell may open the prompt now. */
export async function ready(): Promise<void> {
  await invoke(CMD.ready);
}

/** Esc: the shell hands the keyboard back and hides the prompt. */
export async function close(): Promise<void> {
  await invoke(CMD.close);
}

export async function companionStatus(): Promise<LinkStatus | null> {
  return parseCompanionStatus(await invoke<unknown>(CMD.companionStatus));
}

export async function avatarStatus(): Promise<AvatarStatus | null> {
  return parseStatus(await invoke<unknown>(CMD.avatarStatus));
}
