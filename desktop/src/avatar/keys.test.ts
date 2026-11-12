import { describe, expect, it } from "vitest";
import { isBrowserKey, keyAction, type Key } from "./keys";

const key = (k: string, mods: Partial<Key> = {}): Key => ({
  key: k,
  ctrlKey: false,
  shiftKey: false,
  altKey: false,
  metaKey: false,
  ...mods,
});

describe("the avatar page's keys", () => {
  it("swallows the browser's own keys in every mode", () => {
    for (const k of [key("F5"), key("F12"), key("r", { ctrlKey: true }), key("P", { ctrlKey: true, shiftKey: true }),
      key("f", { ctrlKey: true }), key("ArrowLeft", { altKey: true }), key("=", { ctrlKey: true })]) {
      expect(isBrowserKey(k), JSON.stringify(k)).toBe(true);
      expect(keyAction(k, false)).toBe("swallow");
      expect(keyAction(k, true)).toBe("swallow");
    }
  });

  it("lets ordinary keys alone while locked", () => {
    for (const k of [key("a"), key("Enter"), key("ArrowLeft"), key("c", { ctrlKey: true }), key("v", { ctrlKey: true })]) {
      expect(isBrowserKey(k), JSON.stringify(k)).toBe(false);
      expect(keyAction(k, false)).toBe("ignore");
    }
  });

  it("swallows every key while arranging but Esc, Tab and Alt+F4, on any layout", () => {
    for (const k of [key("a"), key(" ", { altKey: true }), key("Alt", { altKey: true }), key("F10"),
      key("к", { ctrlKey: true }), key("з", { ctrlKey: true }), key("Enter")]) {
      expect(keyAction(k, true), JSON.stringify(k)).toBe("swallow");
    }
    expect(keyAction(key("Tab"), true)).toBe("ignore");
    expect(keyAction(key("F4", { altKey: true }), true)).toBe("ignore");
    expect(keyAction(key("F4", { altKey: true, ctrlKey: true }), true)).toBe("swallow");
  });

  it("locks on Esc, but only while arranging", () => {
    expect(keyAction(key("Escape"), true)).toBe("lock");
    expect(keyAction(key("Escape"), false)).toBe("ignore");
  });
});
