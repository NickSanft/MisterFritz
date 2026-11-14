// Keeps src/divora/tokens.css in step with DivoraVoice's design tokens (plan
// 14 build § 4.3): the block of github.com/NickSanft/DivoraVoiceMod's
// src/styles.css from the first `:root {` to the end of the
// `:root[data-motion="rich"]` block. The commit it was lifted at is named in
// tokens.css's own header, the one place the pin is kept.
//
//   node scripts/sync-divora-tokens.mjs                      compare with the pinned commit
//   node scripts/sync-divora-tokens.mjs --sha <sha>          compare with another commit
//   node scripts/sync-divora-tokens.mjs --sha <sha> --write  take that commit's tokens, and pin it
//   ... --from <styles.css> --sha <sha>                      a local copy of that commit's file
//
// Exit 0 when they match, 1 with the differing lines. It reads the owner's
// repository through the GitHub CLI (`gh api`), so it is not part of
// `pnpm build`, which needs no network. Node ESM only.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const TOKENS = path.join(root, "src", "divora", "tokens.css");

function lines(text) {
  return text.replace(/\r\n/g, "\n").split("\n");
}

/** tokens.css's header comment. */
export function headerOf(tokensCss) {
  const m = lines(tokensCss).join("\n").match(/^\/\*[\s\S]*?\*\//);
  return m ? m[0] : "";
}

/** The DivoraVoiceMod commit tokens.css's header names. */
export function pinnedOf(tokensCss) {
  const m = headerOf(tokensCss).match(/\bat ([0-9a-f]{40})\b/);
  if (!m) throw new Error("tokens.css's header names no commit (`at <40-character sha>`)");
  return m[1];
}

/** The commit tokens.css is pinned to now. */
export const PINNED = pinnedOf(fs.readFileSync(TOKENS, "utf8"));

/** The token block of DivoraVoice's styles.css: from the first `:root {` to
 * the closing brace of `:root[data-motion="rich"]`, by markers rather than
 * line numbers, so an edit above it upstream doesn't move what is compared.
 * `first` and `last` are its 1-based lines there. */
export function tokenRange(styles) {
  const l = lines(styles);
  const start = l.findIndex((x) => x.trimEnd() === ":root {");
  const rich = l.findIndex((x) => x.trimEnd() === ':root[data-motion="rich"] {');
  if (start < 0 || rich < start) throw new Error("styles.css has no token block (no `:root {` before `:root[data-motion=\"rich\"] {`)");
  const end = l.findIndex((x, i) => i > rich && x.trimEnd() === "}");
  if (end < 0) throw new Error("the rich motion block never closes");
  return { text: l.slice(start, end + 1).join("\n"), first: start + 1, last: end + 1 };
}

export function tokenBlock(styles) {
  return tokenRange(styles).text;
}

/** tokens.css without its header comment: what is compared. */
export function bodyOf(tokensCss) {
  const text = lines(tokensCss).join("\n");
  const m = text.match(/^\/\*[\s\S]*?\*\/\n+/);
  return (m ? text.slice(m[0].length) : text).trimEnd();
}

/** The lines that differ, as `-ours` / `+theirs`, or [] when they match. */
export function compare(tokensCss, styles) {
  const ours = lines(bodyOf(tokensCss));
  const theirs = lines(tokenBlock(styles));
  const out = [];
  for (let i = 0; i < Math.max(ours.length, theirs.length); i++) {
    if (ours[i] !== theirs[i]) {
      if (ours[i] !== undefined) out.push(`${i + 1}: -${ours[i]}`);
      if (theirs[i] !== undefined) out.push(`${i + 1}: +${theirs[i]}`);
    }
  }
  return out;
}

/** tokens.css rewritten from `styles` at `sha`: its header kept, naming the
 * new commit and the block's lines there. */
export function rewrite(tokensCss, styles, sha) {
  if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error(`${sha} isn't a full commit sha`);
  const { text, first, last } = tokenRange(styles);
  const header = headerOf(tokensCss)
    .replace(/\bat [0-9a-f]{40}\b/, `at ${sha}`)
    .replace(/\blines \d+-\d+\b/, `lines ${first}-${last}`);
  return `${header}\n\n${text}\n`;
}

function arg(name) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function main() {
  const from = arg("--from");
  const write = process.argv.includes("--write");
  const given = arg("--sha");
  if (write && from && !given) {
    console.error("--write with --from needs --sha: the commit that file is from, to pin.");
    process.exit(2);
  }
  const sha = given ?? PINNED;
  const styles = from
    ? fs.readFileSync(from, "utf8")
    : execFileSync("gh", ["api", "-H", "Accept: application/vnd.github.raw",
      `repos/NickSanft/DivoraVoiceMod/contents/src/styles.css?ref=${sha}`], { encoding: "utf8" });
  const tokens = fs.readFileSync(TOKENS, "utf8");
  const diff = compare(tokens, styles);
  const at = from ? `${from} (${sha.slice(0, 7)})` : sha.slice(0, 7);
  if (write) {
    const full = from ? sha : execFileSync("gh", ["api", `repos/NickSanft/DivoraVoiceMod/commits/${sha}`, "--jq", ".sha"], { encoding: "utf8" }).trim();
    fs.writeFileSync(TOKENS, rewrite(tokens, styles, full));
    console.log(`[written] tokens.css holds DivoraVoiceMod's tokens at ${full.slice(0, 7)}, pinned there (${diff.length} lines changed).`);
    return;
  }
  if (!diff.length) {
    console.log(`[ok] tokens.css matches DivoraVoiceMod's tokens at ${at}.`);
    return;
  }
  console.error(`tokens.css differs from DivoraVoiceMod's tokens at ${at}:\n`);
  for (const d of diff) console.error(d);
  process.exit(1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
