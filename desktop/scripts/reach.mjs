// What a page loads: from its HTML's module scripts and stylesheets, every
// file reached through import, export-from, dynamic import and CSS @import,
// so the build gates check exactly what the page runs, wherever it lives
// (src/avatar, src/divora, src/shared...). Package imports are listed, not
// followed: a package's CSS reaches the page through the bundle, which
// check-no-css-animation.mjs --dist reads. Modules are read with the
// TypeScript compiler's own scanner, which tells a comment from a string.
// Node ESM only.
import fs from "node:fs";
import path from "node:path";
import ts from "typescript";

const CODE = [".ts", ".tsx", ".js", ".mjs"];
const TRY = [".ts", ".tsx", ".js", ".mjs", ".css", "/index.ts", "/index.tsx"];

/** `text` (TypeScript or JavaScript, and JSX when `jsx`) with every
 * comment blanked, character for character, so line numbers hold. The
 * comments are the TypeScript parser's own: the trivia before each token
 * it reads, so a `/*` in a string, a template or a regular expression, a
 * slash that divides or starts a regular expression, and text inside JSX
 * are all told apart as the compiler tells them, which no regex and no
 * bare scanner can. (A bare scanner lost its place after a template's
 * `${...}`.) JSDoc is a comment like any other. */
export function blankComments(text, jsx = true) {
  const kind = jsx ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const source = ts.createSourceFile(jsx ? "x.tsx" : "x.ts", text, ts.ScriptTarget.Latest, false, kind);
  const out = text.split("");
  const blank = (pos, end) => {
    for (let i = pos; i < end; i++) {
      if (out[i] !== "\n" && out[i] !== "\r") out[i] = " ";
    }
  };
  // Reads trivia only, where a comment can't be mistaken for anything else.
  const trivia = ts.createScanner(ts.ScriptTarget.Latest, false, ts.LanguageVariant.Standard);
  const visit = (node) => {
    // JSDoc hangs off the node it documents; its text is the trivia of the
    // token that follows it, blanked there.
    if (node.kind >= ts.SyntaxKind.FirstJSDocNode && node.kind <= ts.SyntaxKind.LastJSDocNode) return;
    // The end of the file is a leaf like any token, though it hands back the
    // JSDoc found after the last statement as its children.
    const children = node.kind === ts.SyntaxKind.EndOfFileToken ? [] : node.getChildren(source);
    if (children.length) {
      for (const child of children) visit(child);
      return;
    }
    // A token: what lies between its full start and its start is trivia,
    // whitespace and comments only (JSX text has none: it is all token).
    const start = node.getStart(source, false);
    if (start <= node.pos) return;
    trivia.setText(text, node.pos, start - node.pos);
    for (let k = trivia.scan(); k !== ts.SyntaxKind.EndOfFileToken; k = trivia.scan()) {
      if (k === ts.SyntaxKind.SingleLineCommentTrivia || k === ts.SyntaxKind.MultiLineCommentTrivia) {
        blank(trivia.getTokenStart(), trivia.getTokenEnd());
      }
    }
  };
  visit(source);
  return out.join("");
}

/** The specifiers a module names: static, side-effect and type imports,
 * export-from and dynamic import(), as the TypeScript compiler reads them. */
export function codeSpecifiers(text) {
  return ts.preProcessFile(text, true, true).importedFiles.map((f) => f.fileName);
}

/** The files a stylesheet @imports. */
export function cssSpecifiers(text) {
  const t = text.replace(/\/\*[\s\S]*?\*\//g, "");
  return [...t.matchAll(/@import\s+(?:url\(\s*)?["']?([^"')\s;]+)["']?\s*\)?/g)].map((m) => m[1]);
}

/** The module scripts and stylesheets an HTML page loads. */
export function htmlEntries(html) {
  const t = html.replace(/<!--[\s\S]*?-->/g, "");
  const scripts = [...t.matchAll(/<script\b[^>]*\btype=["']module["'][^>]*\bsrc=["']([^"']+)["']/g)].map((m) => m[1]);
  const styles = [...t.matchAll(/<link\b[^>]*\brel=["']stylesheet["'][^>]*\bhref=["']([^"']+)["']/g)].map((m) => m[1]);
  return [...scripts, ...styles];
}

function isBare(spec) {
  return !spec.startsWith(".") && !spec.startsWith("/");
}

/** `spec` from `fromFile`, as a file on disk, or null. "/x" is the project
 * root's (as Vite serves it). */
function resolve(spec, fromFile, root, exists) {
  const clean = spec.split(/[?#]/)[0];
  const base = clean.startsWith("/") ? path.join(root, clean) : path.resolve(path.dirname(fromFile), clean);
  if (path.extname(base) && exists(base)) return base;
  for (const ext of TRY) {
    if (exists(base + ext)) return base + ext;
  }
  return null;
}

/**
 * Everything `entryHtml` (a path) reaches: `code` (.ts, .tsx, .js, .mjs) and
 * `css` files, absolute; `packages`, the bare specifiers seen; `missing`,
 * relative imports that resolve to nothing (a gate should fail on those, or
 * it would pass by not looking).
 */
export function reached(entryHtml, opts = {}) {
  const root = opts.root ?? path.dirname(entryHtml);
  const read = opts.read ?? ((p) => fs.readFileSync(p, "utf8"));
  const exists = opts.exists ?? ((p) => fs.existsSync(p) && fs.statSync(p).isFile());
  const code = new Set();
  const css = new Set();
  const packages = new Set();
  const missing = [];
  const queue = [];
  const visit = (spec, from) => {
    if (isBare(spec)) {
      packages.add(spec);
      return;
    }
    const file = resolve(spec, from, root, exists);
    if (!file) {
      missing.push(`${path.relative(root, from)}: ${spec}`);
      return;
    }
    if (code.has(file) || css.has(file)) return;
    const ext = path.extname(file);
    if (ext === ".css") css.add(file);
    else if (CODE.includes(ext)) code.add(file);
    else return; // JSON, a model, an image: nothing to follow
    queue.push(file);
  };
  for (const spec of htmlEntries(read(entryHtml))) visit(spec, entryHtml);
  while (queue.length) {
    const file = queue.shift();
    const text = read(file);
    const specs = path.extname(file) === ".css" ? cssSpecifiers(text) : codeSpecifiers(text);
    for (const spec of specs) visit(spec, file);
  }
  return { code: [...code], css: [...css], packages: [...packages], missing };
}
