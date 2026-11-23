/**
 * What the control window says, from what the shell reports: pure, so it is
 * tested without a window.
 */
import type { AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import type { HookStatus, HotkeyAction, ModelInfo } from "./api";

/** What Settings › Fritz › Clicks on him says the switch does. */
export const CLICKS_DESC =
  "Double-click him to ask him something; right-click him for his menu. A click on his body is his and never reaches the window behind; around him, clicks go through. While he's on screen, Fritz watches the mouse buttons across the desktop to tell a click on him from one beside him. He keeps no record of where you click.";

/** Settings › Diagnostics › The mouse hook: in or out and why, how quick
 * it is, how often it went quiet and was put back (Windows dropped it, or
 * the mouse was over an app run as administrator, which Windows never
 * shows it), and whether it failed. */
export function hookLine(s: HookStatus | null): string {
  if (!s) return "Not known.";
  const slowest = `${(s.slowestMicros / 1000).toFixed(2)} ms`;
  const head = s.installed
    ? s.events > 0
      ? `In while he's on screen; the slowest answer took ${slowest}.`
      : "In while he's on screen."
    : s.why
      ? `Out: ${s.why}.`
      : "Out.";
  const back = s.putBack > 0
    ? ` It went quiet ${s.putBack === 1 ? "once" : `${s.putBack} times`} while the mouse moved, and Fritz put it back${s.putBack === 1 ? "" : " each time"}.`
    : "";
  const failed = !s.panicked
    ? ""
    : s.installed
      ? " It failed, and lets every click through until it is put in again: turn Clicks on him off and on."
      : " It failed, and lets every click through until it goes in again.";
  return head + back + failed;
}

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

export { connectionLines, type ConnectionLines } from "../shared/link";

/** What each of his keys does. */
export const HOTKEY_DOES: Record<HotkeyAction, string> = {
  prompt: "Opens the prompt, to ask him something from any app.",
  arrange: "Lets you move and resize him; Esc locks him there.",
  park: "Hides him, or brings him back.",
  ptt: "Hold it while you speak, and he hears you, once his brain can hear.",
};

export interface HotkeyLine {
  text: string;
  /** The shell's refusal of the key just tried, said as an alert. */
  alert: boolean;
}

/** Text naming keys, with the Windows key called Win, as its keycap says
 * (the settings and the shell's words call it Super). */
export function winNamed(text: string): string {
  return text.replace(/\b(?:Super|Cmd|Command)(?=\+)/gi, "Win");
}

/** Push-to-talk's row while it is muted: its key is let go of. */
export const PTT_MUTED_LINE =
  "Muted from the tray: its key is left to other apps until you unmute it there (Mute push-to-talk).";

/**
 * A hotkey row's line: the shell's refusal of the key just tried, as an
 * alert, followed by whether the key set is the one working; else that
 * alone, if it isn't (the default it fell back to, or none at all); else
 * what the key does. `active` is the key the shell holds for it, null for
 * none, undefined until it has said. Push-to-talk's, while `muted`, says
 * so instead (the shell's refusal alone, as the alert, if a key was tried).
 */
export function hotkeyLine(
  action: HotkeyAction,
  setting: string,
  active: string | null | undefined,
  refusal: string | null,
  muted = false,
): HotkeyLine {
  if (action === "ptt" && muted) {
    return refusal ? { text: winNamed(refusal), alert: true } : { text: PTT_MUTED_LINE, alert: false };
  }
  const status =
    active === null
      ? `Not working: he couldn't have ${setting} (another app may hold it). Pick another.`
      : active !== undefined && active.trim() !== setting.trim()
        ? `Working on ${active}: ${setting} couldn't be used. Pick another, or press Default.`
        : null;
  if (refusal) return { text: winNamed(status ? `${refusal} ${status}` : refusal), alert: true };
  if (status) return { text: winNamed(status), alert: false };
  return { text: HOTKEY_DOES[action], alert: false };
}

/** "Or press Ctrl+Alt+H.", for a row whose action has a key; "" if none. */
export function orPress(active: string | null | undefined): string {
  return active ? winNamed(`Or press ${active}.`) : "";
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
