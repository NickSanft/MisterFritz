/**
 * The build's gates (scripts/): what they forbid, what they let through, and
 * that they look at everything the avatar page loads, wherever it lives.
 */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
// @ts-expect-error: plain ESM scripts, no types
import { blankComments, codeSpecifiers, reached } from "../scripts/reach.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { violations, targets } from "../scripts/check-paced.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { checkBuilt, cssViolations, inlineCss } from "../scripts/check-no-css-animation.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { PINNED, TOKENS, bodyOf, compare, pinnedOf, rewrite, tokenBlock } from "../scripts/sync-divora-tokens.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const rel = (files: string[]) => files.map((f) => path.relative(root, f).replace(/\\/g, "/")).sort();

describe("what the avatar page loads", () => {
  const r = reached(path.join(root, "avatar.html"), { root });

  it("is followed out of src/avatar, into the shared modules and their stylesheets", () => {
    expect(r.missing).toEqual([]);
    const code = rel(r.code);
    expect(code).toContain("src/avatar/main.ts");
    expect(code).toContain("src/divora/theme.ts");
    expect(code).toContain("src/shared/asking.ts");
    expect(rel(r.css)).toEqual(["src/divora/fonts.css", "src/divora/tokens.css"]);
    expect(r.packages).toContain("three");
  });

  it("never reaches the control window's stylesheets or tests", () => {
    expect(rel(r.css)).not.toContain("src/divora/components.css");
    expect(rel(r.code).some((f: string) => f.includes(".test."))).toBe(false);
    expect(rel(targets()).some((f: string) => f.includes("control/"))).toBe(false);
  });

  it("follows every way a module names another", () => {
    expect(codeSpecifiers([
      'import a from "./a";',
      'import "./side.css";',
      'import type { B } from "../b";',
      'export { c } from "./c";',
      'const d = await import("./d");',
      "import {\n  e,\n  f,\n} from './ef';",
      '// import "./commented-out";',
    ].join("\n")).sort()).toEqual(["../b", "./a", "./c", "./d", "./ef", "./side.css"]);
  });

  it("isn't fooled by a comment marker in a line comment or a string", () => {
    // A regex took the `/*` here for a block comment, and lost every module
    // named after it until the next `*/`.
    const text = [
      'import "./a"; // see src/**/*.ts',
      'import "./b";',
      'const glob = "/*";',
      'import "./c";',
      "/** a real comment */",
      'import "./d";',
    ].join("\n");
    expect(codeSpecifiers(text).sort()).toEqual(["./a", "./b", "./c", "./d"]);
    const blanked = blankComments('el.go(); // requestAnimationFrame\nconst s = "// not a comment";');
    expect(blanked).not.toContain("requestAnimationFrame");
    expect(blanked).toContain('"// not a comment"');
    expect(blanked.split("\n")).toHaveLength(2);
  });

  it("follows .tsx, and names an import that isn't there", () => {
    const files: Record<string, string> = {
      "/p/page.html": '<script type="module" src="/src/main.ts"></script>',
      "/p/src/main.ts": 'import "./View";\nimport "./gone";',
      "/p/src/View.tsx": 'import "./look.css";',
      "/p/src/look.css": "@import './more.css';",
      "/p/src/more.css": "",
    };
    const norm = (p: string) => p.replace(/\\/g, "/").replace(/^[A-Za-z]:/, "");
    const got = reached("/p/page.html", {
      root: "/p",
      read: (p: string) => files[norm(p)],
      exists: (p: string) => norm(p) in files,
    });
    expect(got.code.map(norm).sort()).toEqual(["/p/src/View.tsx", "/p/src/main.ts"]);
    expect(got.css.map(norm).sort()).toEqual(["/p/src/look.css", "/p/src/more.css"]);
    expect(got.missing).toEqual([`${path.join("src", "main.ts")}: ./gone`]);
  });
});

describe("the paced-loop gate", () => {
  it("forbids every way to animate outside the loop, from code too", () => {
    for (const bad of [
      "requestAnimationFrame(tick)",
      "renderer.setAnimationLoop(draw)",
      "el.animate([{ opacity: 0 }], 300)",
      "el.style.transition = 'opacity 1s'",
      "el.style.animationName = 'x'",
      "const s = { transitionDuration: '1s' }",
      "el.style.setProperty('transition', 'all 1s')",
      "el.scrollIntoView({ behavior: 'smooth' })",
      "el.style.cssText = 'transition: opacity 1s'",
      '<div style={{ "transition-duration": "1s" }} />',
      "el.style['animation-name'] = 'x'",
      "el.style.cssText = '-webkit-transition: opacity 1s'",
    ]) {
      expect(violations(bad), bad).not.toEqual([]);
    }
  });

  it("finds what a `/*` in a line comment once hid", () => {
    expect(violations("x(); // src/**/*.ts\nrequestAnimationFrame(tick);\n/** ok */")).toHaveLength(1);
  });

  it("reads the page itself as HTML", () => {
    expect(violations("<style>/* transition: x */ .a { color: red }</style>", true)).toEqual([]);
    expect(violations("<style>.a { transition: x 1s }</style>", true)).toHaveLength(1);
  });

  it("lets the loop's own writes, and comments, through", () => {
    for (const fine of [
      "el.style.opacity = String(a)",
      "el.style.transform = `scale(${s})`",
      "// no requestAnimationFrame here",
      "/* transition: none */",
      "scrollIntoView({ block: 'start' })",
    ]) {
      expect(violations(fine), fine).toEqual([]);
    }
  });
});

describe("the CSS-motion gate", () => {
  it("forbids keyframes, animation, transition and smooth scrolling", () => {
    for (const bad of [
      ".a { -webkit-transition: opacity 1s }",
      "@keyframes spin { to { transform: rotate(1turn) } }",
      ".a { animation: spin 1s }",
      ".a { animation-name: spin }",
      ".a { transition: opacity 1s }",
      ".a { transition-duration: 1s }",
      "html { scroll-behavior: smooth }",
      ".a{opacity:0;transition:opacity 1s}",
    ]) {
      expect(cssViolations(bad), bad).toHaveLength(1);
    }
  });

  it("lets custom properties named after them, and comments, through", () => {
    expect(cssViolations(":root { --transition-fast: 120ms; --animation-x: 1; }")).toEqual([]);
    expect(cssViolations("/* transition: opacity 1s */\n.a { opacity: 1 }")).toEqual([]);
  });

  it("reads a page's own <style> and style= attributes", () => {
    const css = inlineCss('<style>.a { transition: x 1s }</style><div style="animation: y 1s"></div>');
    expect(cssViolations(css)).toHaveLength(2);
  });

  it("removes a built page it refuses, so no exe can embed it, and keeps one it passes", () => {
    const make = (css: string) => {
      const dist = mkdtempSync(path.join(os.tmpdir(), "fritz-dist-"));
      mkdirSync(path.join(dist, "assets"));
      writeFileSync(path.join(dist, "avatar.html"), '<link rel="stylesheet" href="/assets/a.css">');
      writeFileSync(path.join(dist, "assets", "a.css"), css);
      return dist;
    };
    const bad = make(".a{opacity:1;transition:opacity 1s}");
    const refused = checkBuilt(bad);
    expect(refused.found).toHaveLength(1);
    expect(refused.removed).toBe(true);
    expect(existsSync(bad)).toBe(false);
    const good = make(":root{--transition-x:1s}.a{opacity:1}");
    expect(checkBuilt(good)).toMatchObject({ found: [], removed: false });
    expect(existsSync(path.join(good, "avatar.html"))).toBe(true);
  });
});

describe("the token sync", () => {
  const tokens = readFileSync(TOKENS, "utf8");

  it("keeps the pin in tokens.css's header, the one place", () => {
    expect(PINNED).toMatch(/^[0-9a-f]{40}$/);
    expect(PINNED).toBe(pinnedOf(tokens));
    expect(() => pinnedOf("/* no commit here */\n:root {}")).toThrow(/names no commit/);
  });

  it("finds the block by its markers and tells a changed token", () => {
    const body = bodyOf(tokens);
    const styles = `/* above */\n@tailwind base;\n\n${body}\n\n.after { color: red }\n`;
    expect(tokenBlock(styles)).toBe(body);
    expect(compare(tokens, styles)).toEqual([]);
    const changed = styles.replace("--indigo: #6D5BF0;", "--indigo: #000000;");
    const diff = compare(tokens, changed);
    expect(diff).toHaveLength(2);
    expect(diff[1]).toContain("#000000");
    const sha = "abcdef0123456789abcdef0123456789abcdef01";
    const written = rewrite(tokens, changed, sha);
    expect(pinnedOf(written)).toBe(sha);
    expect(written).toContain("lines 4-");
    expect(compare(written, changed)).toEqual([]);
    expect(() => rewrite(tokens, changed, "abcdef0")).toThrow(/full commit/);
  });

  it("refuses a styles.css with no token block", () => {
    expect(() => tokenBlock(".a { color: red }")).toThrow(/no token block/);
  });
});
