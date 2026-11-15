// Build-time gate (plan 14 build § 4.3): nothing the avatar page or the
// prompt loads may animate in CSS. A @keyframes, an animation, a transition
// or a smooth scroll keeps the compositor presenting every vsync (see
// check-paced.mjs); the avatar animates only through the paced loop, writing
// its styles from the tick, and the prompt not at all. Two passes, for each
// page in PAGES:
// - the source: every stylesheet the page reaches through its imports
//   (scripts/reach.mjs), and the page's own <style> and style= attributes;
// - with --dist, after `vite build`: the built page's linked stylesheets
//   and inline styles, which is everything it ships, packages too.
// The control window (control.html) isn't checked: it animates as
// DivoraVoice does, at the owner's motion level. Node ESM only.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { reached } from "./reach.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** The pages drawn over the owner's desktop, whose webviews never sleep. */
export const PAGES = ["avatar.html", "prompt.html"];

export const FORBIDDEN = [
  [/@keyframes\b/, "a CSS @keyframes animation"],
  [/(^|[;{\s"'])(-[a-z]+-)?animation(-[a-z-]+)?\s*:/, "a CSS animation property"],
  [/(^|[;{\s"'])(-[a-z]+-)?transition(-[a-z-]+)?\s*:/, "a CSS transition"],
  [/scroll-behavior\s*:\s*smooth/, "a smooth scroll"],
];

/** Each line of a stylesheet (comments aside) that animates. A custom
 * property named after one (`--transition-x:`) doesn't. */
export function cssViolations(css) {
  const out = [];
  css
    .replace(/\/\*[\s\S]*?\*\//g, (c) => c.replace(/[^\n]/g, " "))
    .split("\n")
    .forEach((line, i) => {
      for (const [re, what] of FORBIDDEN) {
        if (re.test(line)) out.push(`${i + 1}: ${what}\n    ${line.trim()}`);
      }
    });
  return out;
}

/** The CSS an HTML page carries itself: <style> blocks and style= values. */
export function inlineCss(html) {
  const t = html.replace(/<!--[\s\S]*?-->/g, "");
  const blocks = [...t.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]);
  const attrs = [...t.matchAll(/\bstyle=(["'])([\s\S]*?)\1/g)].map((m) => `x{${m[2]}}`);
  return [...blocks, ...attrs].join("\n");
}

export function sourcePass(name) {
  const entry = path.join(root, name);
  const r = reached(entry, { root });
  if (r.missing.length) throw new Error(`${name} imports what isn't there:\n  ${r.missing.join("\n  ")}`);
  const sheets = [[entry, inlineCss(fs.readFileSync(entry, "utf8"))], ...r.css.map((f) => [f, fs.readFileSync(f, "utf8")])];
  return check(sheets);
}

/** A built page (`dist/<name>`): its own styles and its linked stylesheets,
 * packages' included. A refused build is removed, `dist` and all: vite build
 * has already replaced the last good one, and what is left must not be
 * embedded by the cargo build that follows. */
export function checkBuilt(dist = path.join(root, "dist"), name = "avatar.html") {
  const page = path.join(dist, name);
  if (!fs.existsSync(page)) throw new Error(`${page} isn't there: run vite build first`);
  const html = fs.readFileSync(page, "utf8");
  const linked = [...html.matchAll(/<link\b[^>]*\brel=["']stylesheet["'][^>]*\bhref=["']([^"']+)["']/g)].map((m) => m[1]);
  const sheets = [[page, inlineCss(html)], ...linked.map((href) => {
    const f = path.join(dist, href.replace(/^\//, ""));
    return [f, fs.readFileSync(f, "utf8").replace(/}/g, "}\n")];
  })];
  const result = check(sheets);
  if (result.found.length) fs.rmSync(dist, { recursive: true, force: true });
  return { ...result, removed: result.found.length > 0 };
}

function check(sheets) {
  const found = [];
  for (const [file, css] of sheets) {
    for (const v of cssViolations(css)) found.push(`${path.relative(root, file)}:${v}`);
  }
  return { found, count: sheets.length };
}

function main() {
  const dist = process.argv.includes("--dist");
  for (const name of PAGES) {
    const result = dist ? checkBuilt(undefined, name) : sourcePass(name);
    if (result.found.length) {
      console.error(`\nNothing ${name} loads may animate in CSS${dist ? " (the built page)" : ""}:\n`);
      for (const v of result.found) console.error(v);
      if (result.removed) console.error("\ndist/ removed, so no exe can embed this page: build again once it's fixed.");
      process.exit(1);
    }
    console.log(`[ok] ${result.count} ${dist ? "built" : "source"} stylesheets ${name} loads don't animate.`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
