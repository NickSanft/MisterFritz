// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Sigil.test.tsx at 62ed6a5. See desktop/LICENSES/DivoraVoiceMod.txt.
// The last describe, Fritz's own glyphs, is the body's.

import { describe, expect, it } from "vitest";
import { render } from "@solidjs/testing-library";
import { Sigil, SIGIL_NAMES } from "./Sigil";

describe("Sigil", () => {
  it("renders a SVG of the requested size", () => {
    const { container } = render(() => <Sigil name="mixer" size={32} />);
    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(svg!.getAttribute("width")).toBe("32");
    expect(svg!.getAttribute("height")).toBe("32");
  });

  it("defaults to size 20", () => {
    const { container } = render(() => <Sigil name="clean" />);
    expect(container.querySelector("svg")!.getAttribute("width")).toBe("20");
  });

  it("includes every effect name in the catalog", () => {
    for (const name of ["pitch", "formant", "reverb", "eq", "robot", "distortion", "echo", "gate"] as const) {
      expect(SIGIL_NAMES).toContain(name);
    }
  });

  it("includes the recorders' sigils (v1.52.0)", () => {
    for (const name of ["record", "screen"] as const) {
      expect(SIGIL_NAMES).toContain(name);
    }
  });

  it("includes Replay's sigil, and it draws (v1.54.0)", () => {
    expect(SIGIL_NAMES).toContain("replay");
    const { container } = render(() => <Sigil name="replay" size={13} />);
    const svg = container.querySelector("svg")!;
    expect(svg.getAttribute("width")).toBe("13");
    expect(svg.querySelectorAll("path, circle").length).toBeGreaterThan(0);
  });

  it("includes the Charms sigils, and they draw (v1.55.0)", () => {
    for (const name of ["image", "unlock", "expand"] as const) {
      expect(SIGIL_NAMES).toContain(name);
      const { container } = render(() => <Sigil name={name} size={16} />);
      const svg = container.querySelector("svg")!;
      expect(svg.getAttribute("width")).toBe("16");
      expect(svg.querySelectorAll("path, circle, rect").length).toBeGreaterThan(0);
    }
  });

  it("includes Hourglass's sigils, and they draw (v1.59.0)", () => {
    for (const name of ["hourglass", "pause"] as const) {
      expect(SIGIL_NAMES).toContain(name);
      const { container } = render(() => <Sigil name={name} size={15} />);
      const svg = container.querySelector("svg")!;
      expect(svg.getAttribute("width")).toBe("15");
      expect(svg.querySelectorAll("path, rect").length).toBeGreaterThan(0);
    }
    // The hourglass is stroked; pause is two filled bars, as play and stop
    // are filled.
    const glass = render(() => <Sigil name="hourglass" />).container.querySelector("svg > g")!;
    expect(glass.getAttribute("fill")).toBe("none");
    expect(glass.getAttribute("stroke")).toBe("currentColor");
    const bars = render(() => <Sigil name="pause" />).container.querySelectorAll("svg rect");
    expect(bars).toHaveLength(2);
    for (const bar of bars) expect(bar.closest("[fill]")?.getAttribute("fill")).toBe("currentColor");
  });

  it("includes the timer actions' sigils, and they draw (v1.60.0)", () => {
    for (const name of ["endNow", "shuffle", "undo"] as const) {
      expect(SIGIL_NAMES).toContain(name);
      const { container } = render(() => <Sigil name={name} size={15} />);
      const svg = container.querySelector("svg")!;
      expect(svg.getAttribute("width")).toBe("15");
      expect(svg.querySelectorAll("path, rect").length).toBeGreaterThan(0);
    }
    // End now is filled, as play is; the other two are stroked, as the
    // hourglass is.
    const end = render(() => <Sigil name="endNow" />).container.querySelector("svg > g")!;
    expect(end.getAttribute("fill")).toBe("currentColor");
    for (const name of ["shuffle", "undo"] as const) {
      const g = render(() => <Sigil name={name} />).container.querySelector("svg > g")!;
      expect(g.getAttribute("fill")).toBe("none");
      expect(g.getAttribute("stroke")).toBe("currentColor");
    }
  });

  it("includes Talk to OBS's sigil, and it draws (v1.61.0)", () => {
    expect(SIGIL_NAMES).toContain("link");
    const { container } = render(() => <Sigil name="link" size={16} />);
    const svg = container.querySelector("svg")!;
    expect(svg.getAttribute("width")).toBe("16");
    expect(svg.getAttribute("viewBox")).toBe("0 0 24 24");
    // Two chain links, stroked 1.5 in the text's colour, as the hourglass is.
    expect(svg.querySelectorAll("rect")).toHaveLength(2);
    const g = svg.querySelector("svg > g")!;
    expect(g.getAttribute("fill")).toBe("none");
    expect(g.getAttribute("stroke")).toBe("currentColor");
    expect(g.getAttribute("stroke-width")).toBe("1.5");
  });

  it("includes every nav name in the catalog", () => {
    for (const name of ["mixer", "soundboard", "presets", "settings"] as const) {
      expect(SIGIL_NAMES).toContain(name);
    }
  });

  it("renders independent DOM trees when the same name is used twice (v0.7.1 regression)", () => {
    // In v0.7.0 the SIGILS map stored pre-evaluated JSX elements, so
    // two <Sigil> components with the same name SHARED a single DOM
    // subtree — Solid would move the node to the latest mount point
    // and the earlier site would blank out. The fix converts each
    // SIGILS entry to a factory (`() => JSX.Element`) so every use
    // creates its own subtree.
    const { container } = render(() => (
      <div>
        <Sigil name="presets" size={20} />
        <Sigil name="presets" size={20} />
      </div>
    ));
    const svgs = container.querySelectorAll("svg");
    expect(svgs).toHaveLength(2);
    // Both SVGs must contain their own children (the icon's <g> or
    // <path>). A shared node would leave one of them empty.
    for (const svg of svgs) {
      expect(svg.children.length).toBeGreaterThan(0);
    }
  });

  it("renders independent DOM trees for the mixer sigil too (covers the Wizard pillar + Sidebar collision)", () => {
    const { container } = render(() => (
      <div>
        <Sigil name="mixer" size={20} />
        <Sigil name="mixer" size={20} />
      </div>
    ));
    const svgs = container.querySelectorAll("svg");
    expect(svgs).toHaveLength(2);
    for (const svg of svgs) {
      expect(svg.children.length).toBeGreaterThan(0);
    }
  });

  it("renders different glyphs side-by-side without bleed", () => {
    const { container } = render(() => (
      <div>
        <Sigil name="mixer" size={20} />
        <Sigil name="settings" size={20} />
        <Sigil name="presets" size={20} />
        <Sigil name="soundboard" size={20} />
      </div>
    ));
    expect(container.querySelectorAll("svg")).toHaveLength(4);
    for (const svg of container.querySelectorAll("svg")) {
      expect(svg.children.length).toBeGreaterThan(0);
    }
  });
});

describe("Fritz's own sigils", () => {
  const FRITZ = ["butler", "desk", "calendar", "inbox", "weather", "listening", "thinking", "speaking"] as const;

  it("are in the catalog, and GitHub's mark isn't", () => {
    for (const name of FRITZ) expect(SIGIL_NAMES).toContain(name);
    expect(SIGIL_NAMES as string[]).not.toContain("github");
  });

  it("draw on the same 24x24 grid, in the text's colour and nothing else, stroked as the others are", () => {
    for (const name of FRITZ) {
      const { container } = render(() => <Sigil name={name} size={19} />);
      const svg = container.querySelector("svg")!;
      expect(svg.getAttribute("viewBox"), name).toBe("0 0 24 24");
      expect(svg.getAttribute("width"), name).toBe("19");
      const marks = svg.querySelectorAll("path, circle, rect");
      expect(marks.length, name).toBeGreaterThan(0);
      for (const el of svg.querySelectorAll("*")) {
        for (const attr of ["fill", "stroke"]) {
          const v = el.getAttribute(attr);
          if (v !== null) expect(["none", "currentColor"], `${name} ${attr}`).toContain(v);
        }
        expect(el.getAttribute("style"), name).toBeNull();
      }
      const g = svg.querySelector("svg > g")!;
      expect(g.getAttribute("fill"), name).toBe("none");
      expect(g.getAttribute("stroke"), name).toBe("currentColor");
      expect(g.getAttribute("stroke-width"), name).toBe("1.5");
      expect(g.getAttribute("stroke-linecap"), name).toBe("round");
    }
  });

  it("stay inside the grid", () => {
    const inside = (n: number) => n >= 0 && n <= 24;
    for (const name of FRITZ) {
      const { container } = render(() => <Sigil name={name} />);
      for (const el of container.querySelectorAll("path")) {
        const d = el.getAttribute("d") ?? "";
        for (const [, cmd, rest] of d.matchAll(/([A-Za-z])([^A-Za-z]*)/g)) {
          const nums = (rest!.match(/-?\d*\.?\d+/g) ?? []).map(Number);
          if (cmd === "A") {
            // An arc's radii and flags aren't points: only where it ends is.
            for (let i = 0; i + 6 < nums.length; i += 7) {
              expect(inside(nums[i + 5]!) && inside(nums[i + 6]!), `${name}: ${d}`).toBe(true);
            }
          } else if (cmd === cmd!.toUpperCase()) {
            // Every absolute coordinate is on the grid.
            for (const n of nums) expect(inside(n), `${name}: ${d}`).toBe(true);
          } else {
            // A relative step never crosses more than the grid.
            for (const n of nums) expect(Math.abs(n), `${name}: ${d}`).toBeLessThanOrEqual(24);
          }
        }
      }
      for (const el of container.querySelectorAll("circle")) {
        const [cx, cy, r] = ["cx", "cy", "r"].map((a) => Number(el.getAttribute(a)));
        expect(inside(cx! - r!) && inside(cx! + r!) && inside(cy! - r!) && inside(cy! + r!), name).toBe(true);
      }
      for (const el of container.querySelectorAll("rect")) {
        const [x, y, w, h] = ["x", "y", "width", "height"].map((a) => Number(el.getAttribute(a)));
        expect(inside(x!) && inside(y!) && inside(x! + w!) && inside(y! + h!), name).toBe(true);
      }
    }
  });
});
