/**
 * What a key does in the prompt. Enter (without Shift) and Esc are armed on
 * the key's way down, with the browser's own action prevented, and act on
 * its way up. So the closing key's release never reaches the owner's app
 * once the prompt has handed the keyboard back, and a release with nothing
 * armed does nothing: the hotkey that opened the prompt is still being let
 * go as it shows.
 *
 * Shift+Enter is a new line. Nothing is armed while an IME is composing:
 * its Enter is the composition's. ↑ in an empty field brings back the last
 * question sent. Tab is swallowed, so the keyboard never leaves the field,
 * and so are the browser's own keys (reload, print, find...: the avatar
 * page's list). Chromium acts on those by the key's place (Windows' virtual
 * key), which on a Latin layout follows the letter e.key gives; on a
 * Cyrillic, Greek or Hebrew layout it doesn't (Ctrl and the R key is
 * reload, though it types к), so there they are matched by e.code.
 */
import { isBrowserKey, type Key } from "../avatar/keys";

/** The parts of a KeyboardEvent this needs. */
export interface PromptKey extends Key {
  code: string;
  isComposing: boolean;
  keyCode: number;
}

/** The places of the avatar page's Ctrl letters (reload, print, find,
 * save, devtools, zoom...). */
const ACCELERATOR_PLACES = new Set([
  "KeyR", "KeyP", "KeyF", "KeyS", "KeyG", "KeyU", "KeyJ", "KeyI", "KeyO", "KeyN", "KeyW", "KeyH", "Equal", "Minus", "Digit0",
]);

/** A browser key by its place, on a layout whose letters aren't Latin. A
 * Latin key is the avatar page's list's to judge: there the place follows
 * the letter (Ctrl+C on Dvorak is copy, wherever its key sits). With Alt
 * held it is no browser key, and may be AltGr typing a letter (Polish ś is
 * AltGr and the S key). */
function browserKeyByPlace(k: PromptKey): boolean {
  if (!(k.ctrlKey || k.metaKey) || k.altKey) return false;
  if (/^[\x20-\x7e]$/.test(k.key)) return false;
  return ACCELERATOR_PLACES.has(k.code);
}

/** A key on its way down that acts on its way up. */
export type Armed = "send" | "close" | null;

export interface KeyState {
  armed: Armed;
  /** Whether the field is empty. */
  empty: boolean;
}

export interface KeyOutcome {
  action: "send" | "close" | "recall" | "none";
  /** Whether the browser's own action for the key is prevented. */
  prevent: boolean;
  /** What is armed after the key. */
  armed: Armed;
}

function composing(k: PromptKey): boolean {
  // 229: a key the IME took, in browsers that don't set isComposing.
  return k.isComposing || k.keyCode === 229;
}

function modified(k: PromptKey): boolean {
  return k.ctrlKey || k.altKey || k.metaKey;
}

export function promptKey(type: "keydown" | "keyup", k: PromptKey, s: KeyState): KeyOutcome {
  const none = (prevent: boolean): KeyOutcome => ({ action: "none", prevent, armed: s.armed });
  if (composing(k)) return none(false);
  if (type === "keyup") {
    if (k.key === "Enter" && s.armed === "send") return { action: "send", prevent: true, armed: null };
    if (k.key === "Escape" && s.armed === "close") return { action: "close", prevent: true, armed: null };
    return none(false);
  }
  if (k.key === "Enter") {
    if (k.shiftKey && !modified(k)) return none(false);
    if (modified(k)) return none(true);
    return { action: "none", prevent: true, armed: "send" };
  }
  if (k.key === "Escape") return { action: "none", prevent: true, armed: "close" };
  if (k.key === "ArrowUp" && s.empty && !modified(k) && !k.shiftKey) {
    return { action: "recall", prevent: true, armed: s.armed };
  }
  if (k.key === "Tab") return none(true);
  return none(isBrowserKey(k) || browserKeyByPlace(k));
}
