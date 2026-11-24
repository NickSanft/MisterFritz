/**
 * docs/LIFTED.md lists every file here that came from another app, and says
 * "Each file also says so at its top": each listed file's opening comment
 * names where it came from (DivoraVoiceMod, or LorraineAtelier).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LIFTED = readFileSync(path.join(root, "docs", "LIFTED.md"), "utf8");

/** `a{1,2}b` as `a1b` and `a2b`. */
const expand = (name: string): string[] => {
  const m = /^(.*)\{([^}]+)\}(.*)$/.exec(name);
  return m ? m[2].split(",").map((part) => `${m[1]}${part}${m[3]}`) : [name];
};

/** Each table row's files, by the section (the app) it is under. A bare
 * name is beside the row's previous file; a name with neither a folder nor
 * an extension is a thing, not a file (`BROWSER_ARGS`). Names that are
 * files but aren't found are `missing`. */
function listed(): { rows: { app: string; file: string }[]; missing: string[] } {
  const rows: { app: string; file: string }[] = [];
  const missing: string[] = [];
  let app = "";
  for (const line of LIFTED.split(/\r?\n/)) {
    if (line.startsWith("## ")) app = line.slice(3);
    if (!line.startsWith("| `")) continue;
    let dir = "";
    for (const [, written] of line.split("|")[1].matchAll(/`([^`]+)`/g)) {
      for (const name of expand(written)) {
        if (!name.includes("/") && !name.includes(".")) continue;
        const file = name.includes("/") ? name : path.posix.join(dir, name);
        if (!existsSync(path.join(root, file))) {
          missing.push(file);
          continue;
        }
        dir = path.posix.dirname(file);
        rows.push({ app, file });
      }
    }
  }
  return { rows, missing };
}

/** A file's opening comment: its first lines that are comments or blank
 * (a JSON file's, its leading "about"). */
function opening(text: string): string {
  const lines: string[] = [];
  for (const l of text.split(/\r?\n/)) {
    const s = l.trim();
    if (s === "" || s === "{" || /^(\/\/|\/\*|\*|#|"about")/.test(s)) lines.push(l);
    else break;
  }
  return lines.join("\n");
}

describe("what is lifted", () => {
  const { rows, missing } = listed();

  it("is found: every file the record names, from both apps", () => {
    expect(missing, "named in LIFTED.md, but not here").toEqual([]);
    expect(rows.length).toBeGreaterThan(30);
    expect(rows.some((r) => r.app.startsWith("DivoraVoiceMod"))).toBe(true);
    expect(rows.some((r) => r.app.startsWith("LorraineAtelier"))).toBe(true);
    const files = rows.map((r) => r.file);
    for (const f of ["src/avatar/bubble.css", "src/divora/sigils/Sigil.test.tsx", "src/divora/hotkeys.test.ts",
                     "src/divora/components/Toggle.test.tsx", "src/divora/fonts/space-mono-latin-700-normal.woff2"]) {
      expect(files, f).toContain(f);
    }
  });

  it("says so at its top, naming the app it came from", () => {
    for (const { app, file } of rows) {
      if (!app.startsWith("DivoraVoiceMod") && !app.startsWith("LorraineAtelier")) continue;
      const name = app.startsWith("DivoraVoiceMod") ? "DivoraVoiceMod" : "Lorraine";
      const top = opening(readFileSync(path.join(root, file), "utf8"));
      expect(top, `${file} doesn't name ${name} at its top`).toContain(name);
    }
  });
});
