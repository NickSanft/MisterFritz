/**
 * The brain's link as every window reads it: the avatar's page reports it,
 * the shell keeps it and tells every window (`companion:status`), and each
 * page reads it strictly and says it in the same words.
 */
import type { AvatarStatus } from "../avatar/status";
import type { Link, LinkStatus, Reason } from "./companion";

/** The avatar's page reported the brain's link (a `LinkStatus`). */
export const COMPANION_STATUS_EVENT = "companion:status";

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

export const LINKS: readonly Link[] = ["unconfigured", "connecting", "online", "offline", "refused"];
export const REASONS: readonly Reason[] = ["token", "version", "hello", "full", "quiet", "unreachable", null];

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

export interface ConnectionLines {
  title: string;
  detail: string;
  /** `bad` when the brain refused him (he won't try again by himself),
   * `warn` while he keeps trying. */
  tone: "ok" | "warn" | "bad" | "quiet";
}

/** Whether he connects, from what the avatar's page last reported. `url`
 * is the brain's address in the settings; `tokenPresent` is null until the
 * shell has said. */
export function connectionLines(
  s: LinkStatus | null,
  avatar: AvatarStatus | null,
  tokenPresent: boolean | null,
  url: string,
): ConnectionLines {
  if (tokenPresent === false) {
    return {
      title: "He needs his brain's token",
      detail: "Paste COMPANION_TOKEN from the brain's .env, then press Connect.",
      tone: "warn",
    };
  }
  if (tokenPresent === null || !s) return { title: "Asking how he's connected…", detail: "", tone: "quiet" };
  // His page holds the connection: with his window closed there is none.
  if (avatar && (!avatar.open || avatar.phase === "closed" || avatar.phase === "closing")) {
    return { title: "His window is closed", detail: "Show him and he'll connect to his brain.", tone: "quiet" };
  }
  const wait = s.retryIn === null ? "Trying again now." : `Trying again in ${s.retryIn} s.`;
  switch (s.link) {
    case "unconfigured":
    case "connecting":
      return { title: "Connecting…", detail: `To his brain at ${url}.`, tone: "quiet" };
    case "online":
      return { title: "Connected", detail: `To his brain at ${url}.`, tone: "ok" };
    case "offline":
      if (s.reason === "full") {
        return {
          title: "His brain has all the desks it allows",
          detail: `COMPANION_MAX_CLIENTS desks are connected to it already. ${wait}`,
          tone: "warn",
        };
      }
      if (s.reason === "quiet") {
        return { title: "His brain stopped answering", detail: `At ${url}. ${wait}`, tone: "warn" };
      }
      // A socket that never opened can't say whether the brain is down or
      // its /companion route is off.
      return {
        title: "Can't reach his brain",
        detail: `Nothing answers at ${url}: is the brain running, with COMPANION_ENABLED on? ${wait}`,
        tone: "warn",
      };
    case "refused":
      if (s.reason === "token") {
        return {
          title: "His brain refused the token",
          detail: "Paste COMPANION_TOKEN from the brain's .env again, then Connect.",
          tone: "bad",
        };
      }
      if (s.reason === "version") {
        return {
          title: "His brain speaks another version",
          detail: "Fritz and his brain need updating to the same release. Reconnect once they are.",
          tone: "bad",
        };
      }
      return {
        title: "His brain couldn't read his hello",
        detail: "The brain's log says why. Reconnect once it's mended.",
        tone: "bad",
      };
  }
}
