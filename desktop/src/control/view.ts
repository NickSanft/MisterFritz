/**
 * What the control window says, from what the shell reports: pure, so it is
 * tested without a window.
 */
import type { AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import type { ModelInfo } from "./api";

export interface Standing {
  /** One line on where he is. */
  line: string;
  /** Whether the button offers to show him (else to hide him). */
  offerShow: boolean;
  canArrange: boolean;
}

export function standing(s: AvatarStatus | null): Standing {
  if (!s) return { line: "Waiting for Fritz…", offerShow: false, canArrange: false };
  if (!s.open || s.phase === "closed" || s.phase === "closing") {
    return { line: "His window is closed. Show him to open it again.", offerShow: true, canArrange: false };
  }
  if (s.parked) {
    return {
      line: "He's out of sight: hidden, or stepped aside for a fullscreen app or the lock screen.",
      offerShow: true,
      canArrange: false,
    };
  }
  if (s.mode === "arrange") {
    return { line: "You're arranging him: drag him, then press Esc.", offerShow: false, canArrange: false };
  }
  return { line: "He's on your desktop.", offerShow: false, canArrange: s.canArrange };
}

/** A size in megabytes, to one decimal place. */
export function megabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export interface ModelLines {
  title: string;
  detail: string;
  /** `warn` when he wears the placeholder though a model was meant. */
  tone: "ok" | "warn" | "quiet";
}

export function modelLines(m: ModelInfo | null): ModelLines {
  if (!m) return { title: "Looking for his model…", detail: "", tone: "quiet" };
  if (m.found && m.tooBig) {
    const size = m.bytes === null ? "over 256 MB" : megabytes(m.bytes);
    return {
      title: "Your model is too large to load",
      detail: `${m.path} is ${size}, more than he can load${m.changed ? "" : ", so he wears the placeholder"}.`,
      tone: "warn",
    };
  }
  // The file there isn't the one he loaded last: what he wears is from
  // before, and nothing is known yet of the new one.
  if (m.changed && m.found) {
    return {
      title: "A new model is here",
      detail: `${m.path}${m.bytes === null ? "" : ` (${megabytes(m.bytes)})`}. Load it again to wear it.`,
      tone: "quiet",
    };
  }
  if (m.changed && m.wearing === "export") {
    return {
      title: "Your model isn't there any more",
      detail: `Nothing is at ${m.path} now. He keeps wearing it until it's loaded again, then the placeholder.`,
      tone: "warn",
    };
  }
  if (m.found && m.wearing === "placeholder") {
    const why = (m.loadError ?? "it couldn't be used").replace(/[\s.]+$/, "");
    return {
      title: "Your model didn't load",
      detail: `${m.path}: ${why}. He wears the placeholder.`,
      tone: "warn",
    };
  }
  if (m.found) {
    return {
      title: "Your model",
      detail: m.bytes === null ? m.path : `${m.path} (${megabytes(m.bytes)})`,
      tone: "ok",
    };
  }
  if (m.from === "setting") {
    return {
      title: "No model where avatar.vrmPath points",
      detail: `There is nothing at ${m.path}, so he wears the placeholder.`,
      tone: "warn",
    };
  }
  return {
    title: "The placeholder",
    detail: `Put your VRoid export at ${m.path}, then load it.`,
    tone: "quiet",
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

/** Who the brain takes this desk to be, while online; else null. */
export function identityLine(s: LinkStatus | null): string | null {
  if (!s || s.link !== "online" || !s.identity) return null;
  const who = s.displayName ? `${s.identity}, talking with ${s.displayName}` : s.identity;
  return s.threadId ? `${who}, in the thread ${s.threadId}.` : `${who}.`;
}

/** Whether the brain's address is this PC (a loopback host). */
export function onThisPc(url: string): boolean {
  let host: string;
  try {
    host = new URL(url).hostname.toLowerCase();
  } catch {
    return false;
  }
  if (host.startsWith("[") && host.endsWith("]")) host = host.slice(1, -1);
  return host === "localhost" || host.endsWith(".localhost") || host === "::1" || /^127(\.\d{1,3}){3}$/.test(host);
}
