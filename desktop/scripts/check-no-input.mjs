// Build-time gate for the study scene (plan 14, PR 12): the wallpaper
// windows live behind the desktop's icons in the shell's wallpaper layer
// and never get a click, a key or the focus, so an input handler in what
// wallpaper.html loads is dead code, and a sign that something was meant to
// be clicked that never can be.
//
// From LorraineAtelier's scripts/check-no-input.mjs (the owner's own app) at
// d62b7ec: line by line, case-blind, for click, pointer and mouse handlers,
// as attributes, framework props or addEventListener, never resize or a
// timer. Changed on the way: it reads every module the page loads
// (scripts/reach.mjs, as the paced-loop gate does), with comments blanked,
// rather than two named files; it also forbids double-click, context-menu,
// wheel, touch, drag and key handlers; a missing file is an error, not a
// warning. Node ESM only.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { blankComments } from "./reach.mjs";
import { targets } from "./check-paced.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** The pages that take no input. */
export const QUIET_PAGES = ["wallpaper.html"];

const EVENTS = "click|dblclick|auxclick|contextmenu|wheel|pointer\\w*|mouse\\w*|touch\\w*|drag\\w*|drop|key(?:down|up|press)";

export const FORBIDDEN = [
  // onclick=, onClick={...}, el.onpointerdown = ..., onMouseMove: ...
  [new RegExp(`\\bon(?:${EVENTS})\\b`, "i"), "an input handler"],
  // Solid's on:click and its kin.
  [new RegExp(`\\bon:(?:${EVENTS})\\b`, "i"), "an input handler (on:)"],
  [new RegExp(`addEventListener\\s*\\(\\s*["'\`](?:${EVENTS})`, "i"), "an input listener"],
];

/** The page itself, its HTML comments and its <style>'s CSS comments blanked. */
function blankHtmlComments(text) {
  const blank = (c) => c.replace(/[^\n]/g, " ");
  return text.replace(/<!--[\s\S]*?-->/g, blank).replace(/\/\*[\s\S]*?\*\//g, blank);
}

/** Each line of `text` (comments aside) that takes input; `html` reads it
 * as a page rather than a module, `jsx` as a .tsx or .jsx one. */
export function inputViolations(text, html = false, jsx = true) {
  const out = [];
  (html ? blankHtmlComments(text) : blankComments(text, jsx))
    .split("\n")
    .forEach((line, i) => {
      for (const [re, what] of FORBIDDEN) {
        if (re.test(line)) out.push(`${i + 1}: ${what}\n    ${line.trim()}`);
      }
    });
  return out;
}

function main() {
  for (const name of QUIET_PAGES) {
    const files = targets(path.join(root, name));
    const found = [];
    for (const file of files) {
      const text = fs.readFileSync(file, "utf8");
      for (const v of inputViolations(text, file.endsWith(".html"), /\.[jt]sx$/.test(file))) found.push(`${path.relative(root, file)}:${v}`);
    }
    if (found.length) {
      console.error(`\n${name} lives behind the desktop's icons and gets no input, so it may take none:\n`);
      for (const v of found) console.error(v);
      process.exit(1);
    }
    console.log(`[ok] none of the ${files.length} files ${name} loads takes input.`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
