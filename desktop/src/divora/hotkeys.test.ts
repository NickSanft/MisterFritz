// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/data/hotkeys.test.ts at 62ed6a5: lines 1-61, the three converters'
// tests, unchanged below this header but for the import, which no longer
// names HOTKEY_ACTIONS or readHotkeyBindings (not lifted). See
// desktop/LICENSES/DivoraVoiceMod.txt.

import { describe, expect, it } from "vitest";
import {
  acceleratorToKeys,
  keysToAccelerator,
  normalizeAccelerator,
} from "./hotkeys";

describe("keysToAccelerator", () => {
  it("joins chips, turning the Windows key into the name the parser accepts", () => {
    expect(keysToAccelerator(["Ctrl", "Win", "P"])).toBe("Ctrl+Super+P");
    expect(keysToAccelerator(["Win", "F9"])).toBe("Super+F9");
    expect(keysToAccelerator(["meta", "1"])).toBe("Super+1");
    expect(keysToAccelerator(["Ctrl", "Shift", "P"])).toBe("Ctrl+Shift+P");
  });

  it("empty or blank chips mean unbound", () => {
    expect(keysToAccelerator([])).toBe("");
    expect(keysToAccelerator([" ", ""])).toBe("");
  });

  it("does not touch a key merely containing 'win'", () => {
    // Only whole tokens are aliases; nothing here is the Windows key.
    expect(keysToAccelerator(["Ctrl", "W"])).toBe("Ctrl+W");
  });
});

describe("acceleratorToKeys", () => {
  it("shows Super (and its aliases) as Win", () => {
    expect(acceleratorToKeys("Ctrl+Super+P")).toEqual(["Ctrl", "Win", "P"]);
    expect(acceleratorToKeys("Cmd+K")).toEqual(["Win", "K"]);
    expect(acceleratorToKeys("Command+K")).toEqual(["Win", "K"]);
    expect(acceleratorToKeys("")).toEqual([]);
    expect(acceleratorToKeys("Space")).toEqual(["Space"]);
  });

  it("round-trips with keysToAccelerator", () => {
    for (const acc of ["Ctrl+Super+P", "Space", "Alt+Shift+F12", "Super+1"]) {
      expect(keysToAccelerator(acceleratorToKeys(acc))).toBe(acc);
    }
  });
});

describe("normalizeAccelerator", () => {
  it("fixes an accelerator saved with Win, and is idempotent", () => {
    expect(normalizeAccelerator("Ctrl+Win+M")).toBe("Ctrl+Super+M");
    expect(normalizeAccelerator(normalizeAccelerator("Ctrl+Win+M"))).toBe("Ctrl+Super+M");
    expect(normalizeAccelerator("Ctrl+Shift+P")).toBe("Ctrl+Shift+P");
    expect(normalizeAccelerator("")).toBe("");
  });

  it("never collapses a '+' key into a bare modifier", () => {
    // Shift+= types "+": the chips ["Shift", "+"] join to "Shift++". Dropping
    // the empty tokens would store "Shift", and the in-app push-to-modulate
    // listener matches the last token, so plain Shift would trigger it.
    expect(normalizeAccelerator("Shift++")).toBe("Shift++");
    expect(normalizeAccelerator("+")).toBe("+");
    expect(normalizeAccelerator("Win++")).toBe("Super++");
  });
});
