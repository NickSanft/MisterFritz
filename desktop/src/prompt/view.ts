/**
 * What the prompt says about whether he can be asked now: pure, so it is
 * tested without a window. The words are the control window's
 * (`connectionLines`), so both windows say the same; the dot's colour only
 * repeats them.
 */
import type { AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { connectionLines } from "../shared/link";

export interface PromptStatus {
  /** `ok`: he can be asked now; `wait`: not yet; `bad`: not until
   * something changes. */
  tone: "ok" | "wait" | "bad";
  text: string;
  /** The gradient rule under the panel shows only while he can be asked. */
  rule: boolean;
}

/** The shell's own words when his window is closed (companion.rs's
 * `say_refusal`). */
export const WINDOW_CLOSED = "His window is closed: show him first.";
export const HIDDEN = "He's hidden: his answer waits until he's shown.";

export function status(link: LinkStatus | null, avatar: AvatarStatus | null): PromptStatus {
  if (avatar && (!avatar.open || avatar.phase === "closed" || avatar.phase === "closing")) {
    return { tone: "bad", text: WINDOW_CLOSED, rule: false };
  }
  // With no token there is nothing to connect with, so the link says
  // unconfigured: the shell's first answer, and the page's report.
  const tokenPresent = link === null ? null : link.link !== "unconfigured";
  const { title } = connectionLines(link, avatar, tokenPresent, "");
  if (link === null || link.link === "connecting") return { tone: "wait", text: title, rule: false };
  if (link.link === "online") return { tone: "ok", text: avatar?.parked ? HIDDEN : title, rule: true };
  return { tone: "bad", text: title, rule: false };
}
