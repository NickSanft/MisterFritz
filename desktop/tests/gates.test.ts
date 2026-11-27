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
import { PAGES as PACED_PAGES, violations, targets } from "../scripts/check-paced.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { PAGES, checkBuilt, cssViolations, inlineCss, sourcePass } from "../scripts/check-no-css-animation.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { PINNED, TOKENS, bodyOf, compare, pinnedOf, rewrite, tokenBlock } from "../scripts/sync-divora-tokens.mjs";
// @ts-expect-error: plain ESM scripts, no types
import { QUIET_PAGES, inputViolations } from "../scripts/check-no-input.mjs";

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
    expect(code).toContain("src/shared/motion.ts");
    expect(code).toContain("src/avatar/BubbleView.ts");
    expect(rel(r.css)).toEqual(["src/avatar/bubble.css", "src/divora/fonts.css", "src/divora/tokens.css"]);
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

  it("knows where a template's substitution ends, and a regular expression from a division", () => {
    // After a `${...}`, the template's own text isn't code, and a comment
    // after the template is still one.
    const afterSub = blankComments("const q = `(a: ${x}dppx)`;\n// requestAnimationFrame\nrequestAnimationFrame(f);");
    expect(afterSub.split("\n")[1].trim()).toBe("");
    expect(afterSub).toContain("requestAnimationFrame(f);");
    // `/*` in a template's text after a substitution hides nothing.
    const hiding = blankComments("const s = `a ${b} /* c`;\nrequestAnimationFrame(f);\nconst t = `*/`;");
    expect(hiding).toContain("requestAnimationFrame(f);");
    // Nested, and braces inside a substitution.
    const nested = blankComments("const n = `${ `${a}` + { k: 1 }.k } // x`;\nrequestAnimationFrame(f); // y");
    expect(nested).toContain("// x`");
    expect(nested).toContain("requestAnimationFrame(f);");
    expect(nested).not.toContain("// y");
    // A brace closed inside a substitution doesn't end it: the comment after
    // it is still code's.
    const braced = blankComments("const m = `x${ { a: 1 }.a /* gone */ }y`;");
    expect(braced).not.toContain("gone");
    expect(braced).toContain("}y`;");
    // A regular expression with slashes in it is no comment; a division is
    // still a division.
    const re = blankComments("const r = /\\/\\//; requestAnimationFrame(f);\nconst h = a / b; // gone");
    expect(re).toContain("requestAnimationFrame(f);");
    expect(re).not.toContain("gone");
    expect(re).toContain("a / b;");
    // A division after a non-null `!`, a property named for a keyword, or a
    // contextual keyword is still one; a regular expression after `return`
    // or `await` is still one.
    for (const line of ["const h = canvas.width! / 2;", "const h = get()! / 2;", "const t = e.type / 2;",
                        "const d = m.default / 2;", "const r = x.return / 2;", "const n = number / 2;",
                        "const o = of / 2;", "const u = undefined / 2;",
                        // Unspaced, where only the operand tells it from a regex.
                        "const h = canvas.width!/2;", "const t = e.type/2;", "const n = number/2;",
                        "const d = m.default/2;"]) {
      const out = blankComments(`${line} // gone`);
      expect(out, line).not.toContain("gone");
      expect(out, line).toContain(line);
    }
    expect(blankComments(String.raw`function f(s) { return /\/\//.test(s) && requestAnimationFrame(f); }`))
      .toContain("requestAnimationFrame(f);");
    // JSX: a tag whose last word is a keyword closes with `/>`, and a slash
    // in its text is no regular expression.
    for (const jsx of ["const v = <div data-new />; // gone", "const v = <Tab default />; /* gone\n gone */",
                       "const v = <var />; // gone", "const v = <Tab new default />; // gone",
                       "const v = <li>Path: / {/* gone */}</li>;"]) {
      expect(blankComments(jsx), jsx).not.toContain("gone");
    }
    expect(blankComments('const v = <Tab default />, s = "a/b", t = "//", g = requestAnimationFrame(f);'))
      .toContain("requestAnimationFrame(f);");
    expect(blankComments(String.raw`function f(s) { return />\/*/.test(s) && requestAnimationFrame(f); /* x */ }`))
      .toContain("requestAnimationFrame(f);");
    // A regular expression that starts with a space, in a module of either
    // kind; and JSX text that only looks like a comment is text.
    for (const jsx of [true, false]) {
      expect(blankComments('const a = s.split(/ +/); // gone', jsx)).not.toContain("gone");
      expect(blankComments('const id = name.replace(/ /g, "-"); /* gone */', jsx)).not.toContain("gone");
    }
    expect(blankComments("const v = <li>// kept, it is text</li>;")).toContain("// kept, it is text");
    // A .ts module's `<T>` cast is read as TypeScript, not JSX.
    expect(blankComments("const n = <number>x; // gone\nrequestAnimationFrame(f);", false)).not.toContain("gone");
    // Comments after the last statement, a JSDoc among them, and a file
    // that is only a doc comment.
    for (const jsx of [true, false]) {
      expect(blankComments("export const a = 1;\n// requestAnimationFrame was here\n/** end */\n", jsx))
        .not.toContain("requestAnimation");
      expect(blankComments("/** requestAnimationFrame */", jsx)).not.toContain("requestAnimation");
      expect(inputViolations("x;\n/**\n * @typedef {{ onclick: () => void }} H\n */\n", false, jsx)).toEqual([]);
    }
    // Blanking keeps every line where it was, so a gate names the right one.
    const lines = blankComments("/* one\n   two */\r\nrequestAnimationFrame(f); // three\n").split("\n");
    expect(lines).toHaveLength(4);
    expect(lines[2]).toBe("requestAnimationFrame(f);         ");
    expect(lines[1].endsWith("\r")).toBe(true);
    expect(violations("/* one\n   two */\nrequestAnimationFrame(f);")[0]).toMatch(/^3: /);
    // Read as a division, this one's `/*` would swallow the code after it.
    expect(blankComments(String.raw`async function f() { await /\/*/.exec(s); requestAnimationFrame(f); /* x */ }`))
      .toContain("requestAnimationFrame(f);");
    expect(violations("const h = w! / 2; const g = \"src/**/*.ts\"; requestAnimationFrame(f);")).toHaveLength(1);
    expect(violations("const q = `${x}`;\n// requestAnimationFrame here\nrequestAnimationFrame(f);")).toHaveLength(1);
    expect(inputViolations("const q = `${x}`;\n// el.onclick = go\nel.onclick = go;")).toHaveLength(1);
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

describe("what the prompt page loads", () => {
  const r = reached(path.join(root, "prompt.html"), { root });

  it("is read by both gates", () => {
    expect(PAGES).toEqual(["avatar.html", "prompt.html", "wallpaper.html"]);
    expect(PACED_PAGES).toBe(PAGES);
    expect(sourcePass("prompt.html").found).toEqual([]);
    const files = rel(targets(path.join(root, "prompt.html")));
    expect(files).toContain("prompt.html");
    expect(files).toContain("src/prompt/main.tsx");
    expect(files).toContain("src/prompt/Prompt.tsx");
    for (const f of targets(path.join(root, "prompt.html"))) {
      expect(violations(readFileSync(f, "utf8"), f.endsWith(".html"), /\.[jt]sx$/.test(f)), f).toEqual([]);
    }
  });

  it("loads only the tokens, the fonts and its own stylesheet", () => {
    expect(r.missing).toEqual([]);
    expect(rel(r.css)).toEqual(["src/divora/fonts.css", "src/divora/tokens.css", "src/prompt/prompt.css"]);
  });

  it("never reaches the control window, its stylesheets, or tests", () => {
    const code = rel(r.code);
    expect(code.some((f: string) => f.startsWith("src/control/"))).toBe(false);
    expect(rel(r.css)).not.toContain("src/divora/components.css");
    expect(rel(r.css)).not.toContain("src/control/control.css");
    expect(code.some((f: string) => f.includes(".test.") || /\/test[A-Z]/.test(f))).toBe(false);
  });

  it("is refused, and removed, when its built page would animate", () => {
    const dist = mkdtempSync(path.join(os.tmpdir(), "fritz-dist-"));
    mkdirSync(path.join(dist, "assets"));
    writeFileSync(path.join(dist, "prompt.html"), '<link rel="stylesheet" href="/assets/p.css">');
    writeFileSync(path.join(dist, "assets", "p.css"), ".prompt{background:#000;transition:opacity 1s}");
    const refused = checkBuilt(dist, "prompt.html");
    expect(refused.found).toHaveLength(1);
    expect(refused.found[0]).toContain("transition");
    expect(refused.removed).toBe(true);
    expect(existsSync(dist)).toBe(false);
  });
});

describe("what the study scene loads", () => {
  const page = path.join(root, "wallpaper.html");
  const r = reached(page, { root });

  it("is read by all three gates, and passes them", () => {
    expect(PAGES).toContain("wallpaper.html");
    expect(QUIET_PAGES).toEqual(["wallpaper.html"]);
    expect(r.missing).toEqual([]);
    expect(sourcePass("wallpaper.html").found).toEqual([]);
    const files = targets(page);
    expect(rel(files)).toEqual(expect.arrayContaining([
      "wallpaper.html", "src/wallpaper/main.ts", "src/wallpaper/scene.ts", "src/wallpaper/backdrop.ts",
      "src/avatar/PacedLoop.ts", "src/divora/theme.ts",
    ]));
    for (const f of files) {
      const text = readFileSync(f, "utf8");
      expect(violations(text, f.endsWith(".html"), /\.[jt]sx$/.test(f)), f).toEqual([]);
      expect(inputViolations(text, f.endsWith(".html"), /\.[jt]sx$/.test(f)), f).toEqual([]);
    }
  });

  it("loads only the tokens, and none of the avatar's or the control window's look", () => {
    expect(rel(r.css)).toEqual(["src/divora/tokens.css"]);
    const code = rel(r.code);
    expect(code.some((f: string) => f.startsWith("src/control/"))).toBe(false);
    expect(code).not.toContain("src/avatar/main.ts");
    expect(code).not.toContain("src/avatar/VrmActor.ts");
    expect(code.some((f: string) => f.includes(".test.") || /\/test[A-Z]/.test(f))).toBe(false);
    expect(r.packages).not.toContain("three");
  });
});

describe("the no-input gate", () => {
  it("forbids every way to take a click, a key or a touch", () => {
    for (const bad of [
      '<canvas onclick="go()"></canvas>',
      "<div onClick={go} />",
      "<div on:click={go} />",
      "el.onpointerdown = go;",
      "el.onmousemove = go;",
      "const props = { onMouseDown: go };",
      'window.addEventListener("click", go);',
      "window.addEventListener('pointermove', go);",
      "window.addEventListener(`mouseup`, go);",
      'el.addEventListener("dblclick", go);',
      'el.addEventListener("contextmenu", go);',
      'el.addEventListener("wheel", go, { passive: true });',
      'el.addEventListener("touchstart", go);',
      'el.addEventListener("dragover", go);',
      'document.addEventListener("keydown", go);',
      "el.onkeyup = go;",
    ]) {
      expect(inputViolations(bad, bad.startsWith("<")), bad).not.toEqual([]);
    }
  });

  it("lets resize, timers, media queries, Tauri's events and comments through", () => {
    for (const fine of [
      'window.addEventListener("resize", fit);',
      'q.addEventListener("change", cb);',
      "setTimeout(step, 100);",
      "await listen(POWER_EVENT, (e) => run.setDisplay(e.payload));",
      "const onTheme = (t) => t;",
      "// el.addEventListener('click', go) once lived here",
      "/* onclick */",
      "const label = 'Clicks on him';",
    ]) {
      expect(inputViolations(fine), fine).toEqual([]);
    }
    expect(inputViolations("<!-- <div onclick=\"x\"> --><canvas></canvas>", true)).toEqual([]);
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
