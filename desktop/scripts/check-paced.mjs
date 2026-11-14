// Build-time gate for the avatar page: it animates only through the paced
// setTimeout loop (src/avatar/PacedLoop.ts). requestAnimationFrame, three.js's
// setAnimationLoop (rAF underneath), the Web Animations API and CSS
// animations or transitions, set in a stylesheet or from code, all keep
// Chromium's compositor presenting every vsync, which is the idle GPU cost
// Spike 0 measures (Lorraine's lesson). It checks our own modules the page
// loads (scripts/reach.mjs follows avatar.html's imports, into src/divora and
// src/shared too) and avatar.html itself. Packages are listed, not read:
// three.js has rAF paths of its own (setAnimationLoop, which this forbids us
// to call). Stylesheets are check-no-css-animation.mjs's. Node ESM only, so
// it runs the same in PowerShell, cmd and CI.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blankComments, reached } from "./reach.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const FORBIDDEN = [
  [/\brequestAnimationFrame\b/, "requestAnimationFrame"],
  [/\bsetAnimationLoop\b/, "setAnimationLoop (rAF underneath)"],
  [/\.animate\s*\(/, "the Web Animations API"],
  [/@keyframes\b/, "a CSS @keyframes animation"],
  // In a stylesheet string, or a style object's key: quoted or not, prefixed or not.
  [/(^|[;{\s"'`])(-[a-z]+-)?animation(-[a-z-]+)?["'`]?\s*:/, "a CSS animation property"],
  [/(^|[;{\s"'`])(-[a-z]+-)?transition(-[a-z-]+)?["'`]?\s*:/, "a CSS transition"],
  [/\.style\.(transition|animation)\w*\s*=/, "a transition or animation set from code"],
  [/\b(transition|animation)[A-Z]\w*\s*:/, "a transition or animation style property"],
  // A property named in a string: setProperty, style["..."], a Solid style key.
  [/["'`](-[a-z]+-)?(transition|animation)(-[a-z-]+)?["'`]/, "a transition or animation named in a string"],
  [/\bbehavior["'`]?\s*:\s*["'`]smooth/, "a smooth scroll"],
];

/** The page itself, its HTML comments and its <style>'s CSS comments blanked. */
function blankHtmlComments(text) {
  const blank = (c) => c.replace(/[^\n]/g, " ");
  return text.replace(/<!--[\s\S]*?-->/g, blank).replace(/\/\*[\s\S]*?\*\//g, blank);
}

/** Each line of `text` (comments aside) that breaks a rule; `html` reads it
 * as the page rather than as a module. */
export function violations(text, html = false) {
  const out = [];
  (html ? blankHtmlComments(text) : blankComments(text))
    .split("\n")
    .forEach((line, i) => {
      for (const [re, what] of FORBIDDEN) {
        if (re.test(line)) out.push(`${i + 1}: ${what}\n    ${line.trim()}`);
      }
    });
  return out;
}

/** The files the gate reads: the page and every module it loads, but tests. */
export function targets(entry = path.join(root, "avatar.html")) {
  const r = reached(entry, { root });
  if (r.missing.length) throw new Error(`avatar.html imports what isn't there:\n  ${r.missing.join("\n  ")}`);
  return [entry, ...r.code.filter((f) => /\.(ts|tsx|js|mjs)$/.test(f) && !/\.test\.tsx?$/.test(f))];
}

function main() {
  const files = targets();
  const found = [];
  for (const file of files) {
    const text = fs.readFileSync(file, "utf8");
    for (const v of violations(text, file.endsWith(".html"))) found.push(`${path.relative(root, file)}:${v}`);
  }
  if (found.length) {
    console.error("\nThe avatar page must animate only through PacedLoop:\n");
    for (const v of found) console.error(v);
    process.exit(1);
  }
  console.log(`[ok] ${files.length} files the avatar page loads animate only through the paced loop.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
