/**
 * The bubble's arithmetic: the motion spec, its appearing, holding and
 * going (moved only by the tick), where it sits, its text, and the colours
 * it may use (AA on its own panel, in every mood and theme).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  CROSSFADE_MS, FADE_MS, HOLD_MAX_MS, HOLD_MS, MATERIALISE_BLUR_FROM_PX, MATERIALISE_MS, MATERIALISE_SCALE_FROM,
  easeOutCubic, holdMs, opacityMs, smoothstep, transformMs,
} from "../shared/motion";
import {
  BUBBLE_MAX_WIDTH, BubbleClock, CROWN_CLEARANCE_M, EDGE, FALLBACK_HEIGHT, GAP, HEAD_CLEARANCE_M, HIDDEN, MIN_BAND, TAIL_DEPTH,
  anchorAbove, charCount, layout, plainText, type Envelope,
} from "./bubble";

const FRAME = 1000 / 30;

/** Tick `clock` for `ms` in frames, returning every envelope. */
function frames(clock: BubbleClock, ms: number, motion: number, hovered = false): Envelope[] {
  const out: Envelope[] = [];
  for (let t = 0; t < ms - 1e-9; t += FRAME) out.push(clock.tick(Math.min(FRAME, ms - t), hovered, motion));
  return out;
}

/** The ms from the first frame until `pred` holds. */
function until(clock: BubbleClock, motion: number, pred: (e: Envelope) => boolean, hovered = false, limit = 60_000): number {
  let t = 0;
  for (; t < limit; t += FRAME) {
    if (pred(clock.tick(FRAME, hovered, motion))) return t + FRAME;
  }
  return Number.POSITIVE_INFINITY;
}

describe("the motion spec", () => {
  it("moves for 400 ms × motion, and none at functional", () => {
    expect(transformMs(MATERIALISE_MS, 1)).toBe(400);
    expect(transformMs(MATERIALISE_MS, 0.6)).toBeCloseTo(240);
    expect(transformMs(MATERIALISE_MS, 0)).toBe(0);
    expect(transformMs(MATERIALISE_MS, Number.NaN)).toBe(0);
  });

  it("never changes opacity in under 330 ms", () => {
    expect(CROSSFADE_MS).toBe(330);
    for (const m of [0, 0.2, 0.6, 1, -1, Number.NaN]) expect(opacityMs(MATERIALISE_MS, m)).toBeGreaterThanOrEqual(330);
    expect(opacityMs(FADE_MS, 1)).toBe(600);
  });

  it("holds 6 s, longer for a long reply, never past 30 s", () => {
    expect(holdMs(0)).toBe(HOLD_MS);
    expect(holdMs(40)).toBe(6000);
    expect(holdMs(200)).toBe(10_000);
    expect(holdMs(100_000)).toBe(HOLD_MAX_MS);
    expect(holdMs(Number.NaN)).toBe(HOLD_MS);
  });

  it("eases from 0 to 1, never past", () => {
    for (const ease of [easeOutCubic, smoothstep]) {
      expect(ease(0)).toBe(0);
      expect(ease(1)).toBe(1);
      expect(ease(-1)).toBe(0);
      expect(ease(2)).toBe(1);
      let last = -1;
      for (let t = 0; t <= 1; t += 0.05) {
        expect(ease(t)).toBeGreaterThanOrEqual(last);
        last = ease(t);
      }
    }
  });
});

describe("the bubble's clock", () => {
  it("is hidden until shown, and moves only when ticked", () => {
    const c = new BubbleClock();
    expect(c.tick(FRAME, false, 1)).toEqual(HIDDEN);
    c.show();
    expect(c.showing).toBe(true);
    // No tick, no time: however long the wait, the next frame is the first.
    const first = c.tick(FRAME, false, 1);
    expect(first.opacity).toBeLessThan(0.1);
    expect(first.scale).toBeLessThan(0.7);
  });

  it("at full motion, materialises from 0.6 and 8 px over 400 ms", () => {
    const c = new BubbleClock();
    c.show();
    const seen = frames(c, MATERIALISE_MS + FRAME, 1);
    expect(seen[0].scale).toBeGreaterThanOrEqual(MATERIALISE_SCALE_FROM);
    expect(seen[0].scale).toBeLessThan(0.75);
    expect(seen[0].blurPx).toBeGreaterThan(5);
    expect(seen[0].blurPx).toBeLessThanOrEqual(MATERIALISE_BLUR_FROM_PX);
    const at = Math.ceil(MATERIALISE_MS / FRAME) - 1;
    expect(seen[at].scale).toBeCloseTo(1, 6);
    expect(seen[at].blurPx).toBeCloseTo(0, 6);
    expect(seen[at].opacity).toBeCloseTo(1, 6);
    expect(seen[at - 2].scale).toBeLessThan(1);
  });

  it("with no motion, only crossfades, over at least 330 ms", () => {
    const c = new BubbleClock();
    c.show();
    const seen = frames(c, 500, 0);
    for (const e of seen) {
      expect(e.scale).toBe(1);
      expect(e.blurPx).toBe(0);
    }
    const full = seen.findIndex((e) => e.opacity >= 1);
    expect((full + 1) * FRAME).toBeGreaterThanOrEqual(CROSSFADE_MS);
    expect(seen[0].opacity).toBeLessThan(0.1);
  });

  it("at ambient motion, moves for 240 ms and fades in over 330", () => {
    const c = new BubbleClock();
    c.show();
    const seen = frames(c, 400, 0.6);
    const settled = seen.findIndex((e) => e.scale >= 1);
    expect((settled + 1) * FRAME).toBeGreaterThanOrEqual(240);
    expect((settled + 1) * FRAME).toBeLessThan(240 + FRAME + 1);
    const full = seen.findIndex((e) => e.opacity >= 1);
    expect((full + 1) * FRAME).toBeGreaterThanOrEqual(330);
    expect((full + 1) * FRAME).toBeLessThan(330 + FRAME + 1);
  });

  it("only ever grows while it materialises", () => {
    for (const m of [0, 0.6, 1]) {
      const c = new BubbleClock();
      c.show();
      const seen = frames(c, 600, m);
      for (let i = 1; i < seen.length; i++) {
        expect(seen[i].opacity).toBeGreaterThanOrEqual(seen[i - 1].opacity);
        expect(seen[i].scale).toBeGreaterThanOrEqual(seen[i - 1].scale);
        expect(seen[i].blurPx).toBeLessThanOrEqual(seen[i - 1].blurPx);
      }
    }
  });

  it("stays while the turn runs, holds once it ends, then fades away", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 60_000, 1);
    expect(c.tick(FRAME, false, 1).opacity, "a turn that runs a minute stays up").toBe(1);
    c.end(10);
    const held = until(c, 1, (e) => e.opacity < 1);
    expect(held).toBeGreaterThanOrEqual(HOLD_MS);
    expect(held).toBeLessThan(HOLD_MS + 3 * FRAME);
    const fade = until(c, 1, (e) => !e.visible);
    expect(fade).toBeGreaterThanOrEqual(FADE_MS - 2 * FRAME);
    expect(fade).toBeLessThan(FADE_MS + 2 * FRAME);
    expect(c.showing).toBe(false);
  });

  it("fades with opacity only", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 500, 1);
    c.end(0);
    frames(c, HOLD_MS, 1);
    for (const e of frames(c, FADE_MS - 2 * FRAME, 1)) {
      expect(e.scale).toBe(1);
      expect(e.blurPx).toBe(0);
    }
  });

  it("holds a long reply longer", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 500, 1);
    c.end(400);
    expect(until(c, 1, (e) => e.opacity < 1)).toBeGreaterThanOrEqual(holdMs(400));
  });

  it("an ended turn still materialises in full before its hold", () => {
    const c = new BubbleClock();
    c.show();
    c.end(0);
    const up = until(c, 1, (e) => e.opacity >= 1);
    expect(up).toBeGreaterThanOrEqual(MATERIALISE_MS - FRAME);
    expect(until(c, 1, (e) => e.opacity < 1)).toBeGreaterThanOrEqual(HOLD_MS);
  });

  it("holds while the cursor is over him, and comes back while it fades", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 500, 1);
    c.end(0);
    frames(c, HOLD_MS - 1000, 1);
    // Hovered for a minute: no time passes on the hold.
    for (const e of frames(c, 60_000, 1, true)) expect(e.opacity).toBe(1);
    // The second it had left, and no less.
    const left = until(c, 1, (e) => e.opacity < 1);
    expect(left).toBeGreaterThanOrEqual(1000 - FRAME);
    expect(left).toBeLessThan(1000 + 3 * FRAME);
    frames(c, 200, 1);
    const fading = c.tick(FRAME, false, 1).opacity;
    expect(fading).toBeLessThan(1);
    const back = frames(c, 600, 1, true);
    expect(back.at(-1)!.opacity).toBe(1);
    expect(back[0].opacity, "eased back, not snapped").toBeLessThan(1);
  });

  it("stays up for a new turn asked while it shows, and comes back if it was fading", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 500, 1);
    c.end(0);
    frames(c, HOLD_MS + 200, 1);
    const fading = c.tick(FRAME, false, 1);
    expect(fading.opacity).toBeLessThan(1);
    c.show();
    const next = frames(c, 600, 1);
    expect(next[0].opacity).toBeGreaterThan(0.3);
    expect(next[0].scale, "no second materialise").toBe(1);
    expect(next.at(-1)!.opacity).toBe(1);
    frames(c, 60_000, 1);
    expect(c.tick(FRAME, false, 1).opacity, "held until the new turn ends").toBe(1);
  });

  it("reset hides it at once", () => {
    const c = new BubbleClock();
    c.show();
    frames(c, 500, 1);
    c.reset();
    expect(c.tick(FRAME, false, 1)).toEqual(HIDDEN);
  });

  it("takes a bad frame time as none", () => {
    const c = new BubbleClock();
    c.show();
    for (const dt of [Number.NaN, -50, Number.NEGATIVE_INFINITY]) expect(c.tick(dt, false, 1).opacity).toBe(0);
  });
});

describe("where the bubble sits", () => {
  const view = { width: 340, height: 640 };

  it("over his head: its tail GAP above the point, centred on it", () => {
    const at = layout({ x: 170, y: 200 }, view, 200);
    expect(at.fallback).toBe(false);
    expect(at.top).toBeNull();
    // The panel's bottom edge the tail's depth above the tip.
    expect(at.bottom).toBe(640 - (200 - GAP - TAIL_DEPTH));
    expect(at.left).toBe(70);
    expect(at.tailX).toBe(100);
    expect(at.maxHeight).toBe(200 - GAP - TAIL_DEPTH - EDGE);
    expect(at.maxWidth).toBe(Math.min(BUBBLE_MAX_WIDTH, view.width - 2 * EDGE));
  });

  it("kept inside both edges, its tail still pointing at him", () => {
    const left = layout({ x: 20, y: 200 }, view, 200);
    expect(left.left).toBe(EDGE);
    expect(left.tailX).toBe(16);
    const right = layout({ x: 330, y: 200 }, view, 200);
    expect(right.left).toBe(340 - EDGE - 200);
    expect(right.tailX).toBe(200 - 16);
    const narrow = layout({ x: 170, y: 200 }, view, 20);
    expect(narrow.tailX).toBe(10);
  });

  it("is never wider than 300, or than the window allows", () => {
    expect(layout({ x: 170, y: 200 }, view, 9999).maxWidth).toBe(300);
    const small = layout({ x: 60, y: 200 }, { width: 120, height: 640 }, 9999);
    expect(small.maxWidth).toBe(120 - 2 * EDGE);
    expect(small.left).toBe(EDGE);
    expect(layout({ x: 0, y: 200 }, { width: 10, height: 640 }, 50).maxWidth).toBe(0);
  });

  it("falls back to the top of the window, over him, when there's no room over his head", () => {
    const low = layout({ x: 170, y: MIN_BAND + GAP + TAIL_DEPTH + EDGE - 1 }, view, 200);
    expect(low.fallback).toBe(true);
    expect(low.top).toBe(EDGE);
    expect(low.bottom).toBeNull();
    expect(low.maxHeight).toBe(Math.round(640 * FALLBACK_HEIGHT));
    expect(layout({ x: 170, y: MIN_BAND + GAP + TAIL_DEPTH + EDGE }, view, 200).fallback).toBe(false);
  });

  it("never sits below the window's bottom edge", () => {
    const deep = layout({ x: 170, y: 9000 }, view, 200);
    expect(deep.bottom).toBe(EDGE);
    expect(deep.maxHeight).toBe(640 - 2 * EDGE);
  });

  /** It points over his hair, however tall: never below the model's own
   * top. The tail in bubble.css reaches TAIL_DEPTH below the panel. */
  it("points over the top of the model, and draws its tail to that depth", () => {
    expect(anchorAbove(0.1)).toBe(HEAD_CLEARANCE_M);
    expect(anchorAbove(0.234)).toBeCloseTo(0.234 + CROWN_CLEARANCE_M, 9);
    expect(anchorAbove(Number.NaN)).toBe(HEAD_CLEARANCE_M);
    const css = readFileSync(fileURLToPath(new URL("./bubble.css", import.meta.url)), "utf8");
    const after = /\.bubble::after \{([^}]*)\}/.exec(css)![1];
    const size = Number(/width:\s*(\d+)px/.exec(after)![1]);
    const below = -Number(/bottom:\s*(-?\d+)px/.exec(after)![1]) - 1; // from the border's outer edge
    expect(after).toMatch(/rotate\(45deg\)/);
    // A square turned 45° reaches half its diagonal past its centre.
    expect(TAIL_DEPTH).toBeCloseTo(below - size / 2 + (size * Math.SQRT2) / 2, 0);
  });

  it("copes with a head it can't place", () => {
    const at = layout({ x: Number.NaN, y: Number.NaN }, view, Number.NaN);
    expect(at.fallback).toBe(true);
    expect(Number.isFinite(at.left)).toBe(true);
  });
});

describe("the reply as plain text", () => {
  it("takes Markdown's marks off and keeps the words, emoji and lines", () => {
    expect(plainText("**Good** evening, *sir*.")).toBe("Good evening, sir.");
    expect(plainText("__Very__ well.")).toBe("Very well.");
    expect(plainText("Run `ollama ps` now.")).toBe("Run ollama ps now.");
    expect(plainText("## Today\nRain, then sun ☀️")).toBe("Today\nRain, then sun ☀️");
    expect(plainText("See [the forecast](https://example.com/x).")).toBe("See the forecast.");
    expect(plainText("```python\nprint(1)\n```\nDone.")).toBe("print(1)\nDone.");
    expect(plainText("Line one\n\nLine two")).toBe("Line one\n\nLine two");
  });

  it("leaves what only looks like a mark", () => {
    expect(plainText("2 * 3 * 4")).toBe("2 * 3 * 4");
    expect(plainText("snake_case_name")).toBe("snake_case_name");
    expect(plainText("#hashtag and C#")).toBe("#hashtag and C#");
    expect(plainText("a**b")).toBe("a**b");
    expect(plainText("**unclosed")).toBe("**unclosed");
  });

  it("counts characters as a reader does", () => {
    expect(charCount("abc")).toBe(3);
    expect(charCount("😀😀")).toBe(2);
    expect(charCount("")).toBe(0);
  });
});

// ── Its colours: AA on its own panel ────────────────────────────────────────

const TOKENS = readFileSync(fileURLToPath(new URL("../divora/tokens.css", import.meta.url)), "utf8");
const CSS = readFileSync(fileURLToPath(new URL("./bubble.css", import.meta.url)), "utf8");
const VIEW_TS = readFileSync(fileURLToPath(new URL("./BubbleView.ts", import.meta.url)), "utf8");

function block(selector: string): Record<string, string> {
  const at = TOKENS.indexOf(`${selector} {`);
  if (at < 0) throw new Error(`tokens.css has no ${selector} block`);
  const body = TOKENS.slice(TOKENS.indexOf("{", at) + 1, TOKENS.indexOf("}", at));
  return Object.fromEntries([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));
}

function token(name: string, selectors: string[]): string {
  for (const s of selectors) {
    const v = block(s)[name];
    if (v !== undefined) return v;
  }
  throw new Error(`${name} isn't set by ${selectors.join(", ")}`);
}

function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`not an opaque hex colour: ${hex}`);
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(m[1].slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

/** Each mood and theme's blocks, most specific first. */
const COMBOS: Record<string, string[]> = {
  "dusk dark": [":root"],
  "ink dark": [':root[data-mood="ink"]', ":root"],
  "midnight dark": [':root[data-mood="midnight"]', ":root"],
  "dusk light": [':root[data-theme="light"]', ":root"],
  "ink light": [':root[data-theme="light"][data-mood="ink"]', ':root[data-theme="light"]', ':root[data-mood="ink"]', ":root"],
  "midnight light": [
    ':root[data-theme="light"][data-mood="midnight"]', ':root[data-theme="light"]', ':root[data-mood="midnight"]', ":root",
  ],
};

describe("the bubble's colours", () => {
  it("are AA (4.5:1) on its opaque panel in every mood and theme", () => {
    for (const [name, selectors] of Object.entries(COMBOS)) {
      const panel = token("--surface-2", selectors);
      for (const text of ["--text-hi", "--text-mid"]) {
        const ratio = contrast(token(text, selectors), panel);
        expect(ratio, `${text} on ${name}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("never uses the dimmer text tokens, which fail there", () => {
    const lo = Math.min(...Object.values(COMBOS).map((s) => contrast(token("--text-lo", s), token("--surface-2", s))));
    expect(lo, "why --text-lo is kept out").toBeLessThan(4.5);
    const rules = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
    expect(rules).not.toMatch(/--text-lo\b/);
    expect(rules).not.toMatch(/--text-dim\b/);
    expect(rules).toMatch(/background:\s*var\(--surface-2\)/);
  });

  it("are set as text only, never as markup", () => {
    const code = VIEW_TS.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
    expect(code).not.toMatch(/innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
    expect(code).toMatch(/textContent/);
  });

  /** One clipped column holds all it says, so nothing spills past the
   * panel; the top fade never takes more than a third of what shows. */
  it("keep everything it says inside the panel", () => {
    const rules = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
    expect(rules).toMatch(/\.bubble-clip \{[^}]*overflow:\s*hidden/);
    expect(rules).toMatch(/\.bubble-clip \{[^}]*min-height:\s*0/);
    expect(rules).toMatch(/\.bubble-clip \{[^}]*justify-content:\s*flex-end/);
    expect(rules).toMatch(/\.bubble-content \{[^}]*flex:\s*none/);
    expect(rules).toMatch(/\.bubble-clip\.clipped \{[^}]*#000 min\(2\.2em, 33%\)/);
    expect(VIEW_TS).toMatch(/"bubble-content", clip\)/);
    for (const row of ["bubble-progress", "bubble-text", "bubble-apology", "bubble-note"]) {
      expect(VIEW_TS, row).toMatch(new RegExp(`"${row}", content\\)`));
    }
  });

  it("take no input", () => {
    expect(CSS).toMatch(/\.bubble \{[^}]*pointer-events:\s*none/);
    expect(VIEW_TS).not.toMatch(/tabIndex|tabindex|focus\(|addEventListener/);
  });
});
