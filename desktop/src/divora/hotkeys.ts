// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/data/hotkeys.ts at 62ed6a5: lines 1-15 and 45-82 (the three
// converters), unchanged below this header. Its import, its actions and
// the reading of saved bindings (lines 17-43 and 84-96) are DivoraVoice's
// own. See desktop/LICENSES/DivoraVoiceMod.txt.

// v1.51.4: global-hotkey accelerators and the persisted bindings.
//
// Two defects lived here, both silent:
//
// 1. The Windows key never worked. HotkeyCapture labels it "Win", and the
//    store joined the chips into the accelerator as-is ("Ctrl+Win+P"). The
//    parser behind `register_global_shortcut` (global-hotkey 0.8.0,
//    `hotkey.rs`) knows the modifier only as SUPER / CMD / COMMAND, so "Win"
//    was taken for a key name, rejected, and the registration failed with a
//    console warning nobody saw.
// 2. The push-to-modulate / panic / monitor bindings were never saved, so
//    they were gone after every restart — although the docs said otherwise.
//
// Chips keep saying "Win" (what Windows users call the key); the
// accelerator handed to the backend says "Super". Pure, so it unit-tests.

/** Names for the Windows key the parser rejects, mapped to one it accepts. */
const WINDOWS_KEY_ALIASES = new Set(["WIN", "WINDOWS", "META"]);
/** Names the parser accepts for it, shown to the user as "Win". */
const SUPER_NAMES = new Set(["SUPER", "CMD", "COMMAND"]);

/** Chips (`["Ctrl", "Win", "P"]`) → an accelerator the backend can parse
 *  (`"Ctrl+Super+P"`). Empty chips → `""` (unbound). */
export function keysToAccelerator(keys: readonly string[]): string {
  return keys
    .map((k) => k.trim())
    .filter(Boolean)
    .map((k) => (WINDOWS_KEY_ALIASES.has(k.toUpperCase()) ? "Super" : k))
    .join("+");
}

/** An accelerator → chips for display. "Super" (and its aliases) shows as
 *  "Win", which is what the key is called on Windows. */
export function acceleratorToKeys(accelerator: string): string[] {
  if (!accelerator) return [];
  return accelerator
    .split("+")
    .map((k) => k.trim())
    .filter(Boolean)
    .map((k) => (SUPER_NAMES.has(k.toUpperCase()) ? "Win" : k));
}

/** An accelerator that may contain "Win" (written by an older build, or
 *  typed by hand) → one the backend can parse. Only the Windows key's name
 *  changes: every other token is kept exactly, including the empty ones a
 *  "+" key leaves ("Shift++"), so a binding can never collapse into a bare
 *  modifier that the in-app push-to-modulate listener would fire on.
 *  Idempotent. */
export function normalizeAccelerator(accelerator: string): string {
  return accelerator
    .split("+")
    .map((t) => (WINDOWS_KEY_ALIASES.has(t.trim().toUpperCase()) ? "Super" : t))
    .join("+");
}
