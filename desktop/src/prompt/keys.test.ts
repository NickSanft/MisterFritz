import { describe, expect, it } from "vitest";
import { promptKey, type Armed, type PromptKey } from "./keys";

const key = (k: string, mods: Partial<PromptKey> = {}): PromptKey => ({
  key: k,
  code: "",
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  metaKey: false,
  isComposing: false,
  keyCode: 0,
  ...mods,
});
const down = (k: PromptKey, armed: Armed = null, empty = false) => promptKey("keydown", k, { armed, empty });
const up = (k: PromptKey, armed: Armed = null, empty = false) => promptKey("keyup", k, { armed, empty });

describe("the prompt's keys", () => {
  it("arm Enter on its way down, preventing the new line, and send on its way up", () => {
    const d = down(key("Enter"));
    expect(d).toEqual({ action: "none", prevent: true, armed: "send" });
    expect(up(key("Enter"), d.armed)).toEqual({ action: "send", prevent: true, armed: null });
  });

  it("do nothing on a release with nothing armed (the hotkey that opened it)", () => {
    for (const k of [key("Enter"), key("Escape"), key("f", { ctrlKey: true, altKey: true }), key("Alt")]) {
      expect(up(k), k.key).toEqual({ action: "none", prevent: false, armed: null });
    }
  });

  it("send only on Enter's own release", () => {
    expect(up(key("Shift"), "send")).toEqual({ action: "none", prevent: false, armed: "send" });
    expect(up(key("Escape"), "send").action).toBe("none");
    expect(up(key("Enter"), "close").action).toBe("none");
  });

  it("make a new line with Shift+Enter, and never send it", () => {
    const d = down(key("Enter", { shiftKey: true }));
    expect(d).toEqual({ action: "none", prevent: false, armed: null });
    expect(up(key("Enter", { shiftKey: true }), d.armed).action).toBe("none");
  });

  it("neither send nor break the line with Ctrl, Alt or Win held", () => {
    for (const m of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }, { ctrlKey: true, shiftKey: true }]) {
      expect(down(key("Enter", m)), JSON.stringify(m)).toEqual({ action: "none", prevent: true, armed: null });
    }
  });

  it("leave an IME's Enter to the IME", () => {
    for (const k of [key("Enter", { isComposing: true }), key("Enter", { keyCode: 229 }), key("Process", { keyCode: 229 })]) {
      expect(down(k), JSON.stringify(k)).toEqual({ action: "none", prevent: false, armed: null });
      expect(up(k, "send").action, JSON.stringify(k)).toBe("none");
    }
  });

  it("close on Esc's way up", () => {
    const d = down(key("Escape"));
    expect(d).toEqual({ action: "none", prevent: true, armed: "close" });
    expect(up(key("Escape"), d.armed)).toEqual({ action: "close", prevent: true, armed: null });
  });

  it("bring back the last question with ↑, only in an empty field", () => {
    expect(down(key("ArrowUp"), null, true)).toEqual({ action: "recall", prevent: true, armed: null });
    expect(down(key("ArrowUp"), null, false)).toEqual({ action: "none", prevent: false, armed: null });
    expect(down(key("ArrowUp", { shiftKey: true }), null, true).action).toBe("none");
    expect(down(key("ArrowUp", { ctrlKey: true }), null, true).action).toBe("none");
  });

  it("swallow Tab and the browser's own keys, and leave editing to the field", () => {
    for (const k of [key("Tab"), key("Tab", { shiftKey: true }), key("F5"), key("F7"), key("F12"), key("r", { ctrlKey: true }), key("p", { ctrlKey: true }), key("f", { ctrlKey: true })]) {
      expect(down(k), JSON.stringify(k)).toEqual({ action: "none", prevent: true, armed: null });
    }
    for (const k of [key("a", { ctrlKey: true }), key("c", { ctrlKey: true }), key("v", { ctrlKey: true }), key("x", { ctrlKey: true }), key("z", { ctrlKey: true }), key("h"), key("Backspace"), key("ArrowLeft")]) {
      expect(down(k), JSON.stringify(k)).toEqual({ action: "none", prevent: false, armed: null });
    }
  });

  /** Chromium reloads on Ctrl and the R key wherever the layout puts к. */
  it("swallow the browser's keys by their place on a layout that isn't Latin", () => {
    for (const [k, code] of [["к", "KeyR"], ["з", "KeyP"], ["а", "KeyF"], ["π", "KeyP"], ["ר", "KeyR"], ["Ш", "KeyI"]]) {
      expect(down(key(k, { ctrlKey: true, code })), k).toEqual({ action: "none", prevent: true, armed: null });
    }
    // Copy, paste, cut, undo and select all stay the field's, on any layout.
    for (const [k, code] of [["с", "KeyC"], ["м", "KeyV"], ["ч", "KeyX"], ["я", "KeyZ"], ["ф", "KeyA"]]) {
      expect(down(key(k, { ctrlKey: true, code })), k).toEqual({ action: "none", prevent: false, armed: null });
    }
    // Typing к without Ctrl is typing.
    expect(down(key("к", { code: "KeyR" })).prevent).toBe(false);
    // A Latin letter is judged as it types: Dvorak's c sits on the I key.
    expect(down(key("c", { ctrlKey: true, code: "KeyI" })).prevent).toBe(false);
    // AltGr letters are typing: Polish ś, ó and ń, Romanian ș and î.
    for (const [k, code] of [["ś", "KeyS"], ["ó", "KeyO"], ["ń", "KeyN"], ["ș", "KeyS"], ["î", "KeyI"]]) {
      expect(down(key(k, { ctrlKey: true, altKey: true, code })).prevent, k).toBe(false);
    }
  });

  it("keep what is armed through other keys", () => {
    expect(down(key("a"), "send").armed).toBe("send");
    expect(down(key("Shift"), "close").armed).toBe("close");
  });
});
