/**
 * What the avatar page does with the keyboard and the right button. WebView2
 * keeps its browser context menu and accelerator keys on (tauri 2.11 doesn't
 * expose the settings), so in Arrange — when the window takes clicks and the
 * keyboard — a right-click would offer Print and Save as, F5 would reload the
 * page mid-Arrange, and Ctrl+P would open a print preview from a transparent
 * always-on-top window. So while arranging every key is swallowed but Esc
 * (which locks), Alt+F4 and Tab, as DivoraVoice's overlay editor does: that
 * holds on any keyboard layout (Chromium fires its accelerators from the
 * key's position, while e.key carries the layout's character). Locked, the
 * window never has the keyboard; the browser's own keys are refused anyway.
 */

/** The parts of a KeyboardEvent this needs. */
export interface Key {
  key: string;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  metaKey: boolean;
}

const CTRL_LETTERS = new Set(["r", "p", "f", "s", "g", "u", "j", "i", "o", "n", "w", "h", "+", "-", "=", "0"]);
const FUNCTION_KEYS = new Set(["F1", "F3", "F5", "F6", "F7", "F11", "F12", "BrowserBack", "BrowserForward", "BrowserRefresh"]);

/** A key the browser would act on (reload, print, find, devtools, zoom,
 * history...), so the page must swallow it. */
export function isBrowserKey(k: Key): boolean {
  if (FUNCTION_KEYS.has(k.key)) return true;
  if (k.altKey && (k.key === "ArrowLeft" || k.key === "ArrowRight" || k.key === "Home")) return true;
  if ((k.ctrlKey || k.metaKey) && CTRL_LETTERS.has(k.key.toLowerCase())) return true;
  return false;
}

/** What a key does on the avatar page. */
export function keyAction(k: Key, arranging: boolean): "lock" | "swallow" | "ignore" {
  if (arranging) {
    if (k.key === "Escape") return "lock";
    if (k.key === "Tab" || (k.key === "F4" && k.altKey && !k.ctrlKey && !k.metaKey)) return "ignore";
    return "swallow";
  }
  return isBrowserKey(k) ? "swallow" : "ignore";
}
