/**
 * The prompt's wire and colours: the commands it calls, by the names and
 * keys the shell has (wire.fixture.json), and text that is AA on its panel
 * in every mood and theme.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import { COMBOS, contrast, token } from "../shared/testContrast";
import { COMPANION_STATUS_EVENT } from "../shared/link";
import { STATUS_EVENT } from "../avatar/status";
import { CMD, WAITING_EVENT, args, parseWaiting } from "./api";

const CSS = readFileSync(fileURLToPath(new URL("./prompt.css", import.meta.url)), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
const PROMPT_TSX = readFileSync(fileURLToPath(new URL("./Prompt.tsx", import.meta.url)), "utf8");
const MAIN_TSX = readFileSync(fileURLToPath(new URL("./main.tsx", import.meta.url)), "utf8");

describe("the prompt's wire", () => {
  it("calls the commands the shell registers, with the arguments it takes", () => {
    const c = fixture.commands;
    expect(CMD.say).toBe(c.promptSay.name);
    expect(Object.keys(args.say("good evening"))).toEqual(c.promptSay.args);
    expect(CMD.close).toBe(c.promptClose.name);
    expect(c.promptClose.args).toEqual([]);
    expect(CMD.ready).toBe(c.promptReady.name);
    expect(c.promptReady.args).toEqual([]);
    expect(CMD.companionStatus).toBe(c.companionStatus.name);
    expect(CMD.avatarStatus).toBe(c.status.name);
  });

  /** The page says it is ready only once it has mounted and listens, so
   * the shell never opens an empty or deaf prompt. */
  it("says it is ready once mounted and listening", () => {
    const listening = MAIN_TSX.lastIndexOf("await listen<unknown>(");
    const mounted = MAIN_TSX.indexOf("render(");
    const ready = MAIN_TSX.indexOf("ready().catch(");
    expect(listening).toBeGreaterThan(0);
    expect(mounted).toBeGreaterThan(listening);
    expect(ready).toBeGreaterThan(mounted);
    // The shell's word on waiting among what it listens to first.
    const waits = MAIN_TSX.indexOf("await listen<unknown>(WAITING_EVENT,");
    expect(waits).toBeGreaterThan(0);
    expect(waits).toBeLessThan(mounted);
    expect(MAIN_TSX).toMatch(/waiting=\{waiting\}/);
  });

  it("hears the events every window hears", () => {
    expect(COMPANION_STATUS_EVENT).toBe(fixture.events.companionStatus);
    expect(STATUS_EVENT).toBe(fixture.events.status);
  });

  it("hears the shell's word on waiting, as the shell says it", () => {
    expect(WAITING_EVENT).toBe(fixture.events.promptWaiting);
    expect(parseWaiting(true)).toBe(true);
    expect(parseWaiting(false)).toBe(false);
    for (const bad of [null, "true", 1, {}]) expect(parseWaiting(bad)).toBeNull();
  });
});

describe("the prompt's colours", () => {
  it("are AA (4.5:1) on its panel in every mood and theme", () => {
    expect(CSS).toMatch(/\.prompt \{[^}]*background:\s*var\(--surface-1\)/);
    for (const [name, selectors] of Object.entries(COMBOS)) {
      const panel = token("--surface-1", selectors);
      for (const text of ["--text-hi", "--text-mid"]) {
        expect(contrast(token(text, selectors), panel), `${text} on ${name}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("never use the dimmer text tokens", () => {
    expect(CSS).not.toMatch(/--text-lo\b/);
    expect(CSS).not.toMatch(/--text-dim\b/);
    for (const used of CSS.matchAll(/color:\s*var\((--text-[\w-]+)\)/g)) {
      expect(["--text-hi", "--text-mid"], used[0]).toContain(used[1]);
    }
  });

  /** The dot only repeats the words, so it is hidden from a reader; the
   * line itself is a live status. */
  it("carry meaning in words, never the dot's colour alone", () => {
    expect(PROMPT_TSX).toMatch(/class=\{`prompt-dot \$\{props\.status\(\)\.tone\}`\} aria-hidden="true"/);
    expect(PROMPT_TSX).toMatch(/const line = \(\) => lineFor\(props\.status\(\), refusal\(\), asking\(\), face\(\)\);/);
    expect(PROMPT_TSX).toMatch(/role="status">\s*\{line\(\)\.text\}/);
  });

  it("show the line asking for a click in the brightest text", () => {
    expect(CSS).toMatch(/\.prompt-waiting \{\s*color: var\(--text-hi\);/);
  });

  it("show where the panel ends, in high contrast too", () => {
    expect(CSS).toMatch(/@media \(forced-colors: active\)[\s\S]*\.prompt \{\s*border-color: CanvasText;/);
  });
});
