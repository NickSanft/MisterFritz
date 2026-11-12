// Build-time gate for the avatar page: it animates only through the paced
// setTimeout loop (src/avatar/PacedLoop.ts). requestAnimationFrame, three.js's
// setAnimationLoop (rAF underneath), the Web Animations API and CSS
// animations or transitions all keep Chromium's compositor presenting every
// vsync, which is the idle GPU cost Spike 0 measures (Lorraine's lesson).
// Node ESM only, so it runs the same in PowerShell, cmd and CI.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function files(dir, exts) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...files(p, exts));
    else if (exts.some((e) => entry.name.endsWith(e)) && !/\.test\.ts$/.test(entry.name)) out.push(p);
  }
  return out;
}

// Comments may name what they forbid; only code counts.
function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/<!--[\s\S]*?-->/g, "").replace(/(^|[^:"'`])\/\/.*$/gm, "$1");
}

const forbidden = [
  [/\brequestAnimationFrame\b/, "requestAnimationFrame"],
  [/\bsetAnimationLoop\b/, "setAnimationLoop (rAF underneath)"],
  [/\.animate\s*\(/, "the Web Animations API"],
  [/@keyframes\b/, "a CSS @keyframes animation"],
  [/(^|[;{\s"'])animation(-[a-z-]+)?\s*:/, "a CSS animation property"],
  [/(^|[;{\s"'])transition(-[a-z-]+)?\s*:/, "a CSS transition"],
];

const targets = [...files(path.join(root, "src", "avatar"), [".ts", ".css"]), path.join(root, "avatar.html")];
const violations = [];
for (const file of targets) {
  const lines = stripComments(fs.readFileSync(file, "utf8")).split("\n");
  lines.forEach((line, i) => {
    for (const [re, what] of forbidden) {
      if (re.test(line)) violations.push(`${path.relative(root, file)}:${i + 1}: ${what}\n    ${line.trim()}`);
    }
  });
}

if (violations.length) {
  console.error("\nThe avatar page must animate only through PacedLoop:\n");
  for (const v of violations) console.error(v);
  process.exit(1);
}
console.log(`[ok] ${targets.length} avatar files animate only through the paced loop.`);
