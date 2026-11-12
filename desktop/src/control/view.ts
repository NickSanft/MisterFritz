/**
 * What the control window says, from what the shell reports: pure, so it is
 * tested without a window.
 */
import type { AvatarStatus } from "../avatar/status";
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
