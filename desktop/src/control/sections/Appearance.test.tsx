import { describe, expect, it, vi } from "vitest";
import { createSignal } from "solid-js";
import { fireEvent, render, within } from "@solidjs/testing-library";
import fixture from "../../avatar/wire.fixture.json";
import { parseTheme, type Theme } from "../../shared/settings";
import { wearTheme } from "../wear";
import { Appearance, motionLine } from "./Appearance";

const DEFAULTS: Theme = parseTheme(fixture.themeDefaults)!;
const theme = (more: Partial<Theme> = {}): Theme => ({ ...DEFAULTS, ...more });

function mount(t: Theme | null, lessMotion = false) {
  const set = vi.fn();
  const view = render(() => <Appearance theme={t} lessMotion={lessMotion} set={set} />);
  return { view, set };
}

/** The checked option of the radio group named `name`. */
function picked(view: ReturnType<typeof render>, name: string): string | null {
  const group = view.getByRole("radiogroup", { name });
  const on = within(group).getAllByRole("radio").filter((r) => r.getAttribute("aria-checked") === "true");
  expect(on.length, name).toBeLessThanOrEqual(1);
  return on[0]?.textContent ?? null;
}

describe("Settings › Appearance", () => {
  it("is the section FRITZ_SHOW_SETTINGS=appearance opens, under its sigil", () => {
    const { view } = mount(DEFAULTS);
    const section = view.container.querySelector("section#appearance");
    expect(section).not.toBeNull();
    expect(fixture.controlSections).toContain(section!.id);
    expect(section!.querySelector("svg.section-sigil")).not.toBeNull();
  });

  it("shows where the theme stands, the defaults first", () => {
    const { view } = mount(DEFAULTS);
    expect(picked(view, "Theme")).toBe("Dark");
    expect(picked(view, "Colour mood")).toBe("Ink + candle");
    expect(picked(view, "Accent")).toBe("Brand");
    expect(picked(view, "Motion")).toBe("Ambient");
    expect(view.getByRole("switch", { name: "Parchment grain" })).toHaveAttribute("aria-checked", "true");
    expect(view.getByRole("switch", { name: "Keep his motion when Windows asks for less" }))
      .toHaveAttribute("aria-checked", "false");
    view.unmount();
    const other = mount(theme({ theme: "light", mood: "midnight", accent: "ember", motion: 1, grain: false, motionUser: true }));
    expect(picked(other.view, "Theme")).toBe("Light");
    expect(picked(other.view, "Colour mood")).toBe("Midnight");
    expect(picked(other.view, "Accent")).toBe("Ember");
    expect(picked(other.view, "Motion")).toBe("Rich");
    expect(other.view.getByRole("switch", { name: "Parchment grain" })).toHaveAttribute("aria-checked", "false");
    expect(other.view.getByRole("switch", { name: "Keep his motion when Windows asks for less" }))
      .toHaveAttribute("aria-checked", "true");
  });

  it("shows motion at the level the windows snap it to", () => {
    const label = { functional: "Functional", ambient: "Ambient", rich: "Rich" } as const;
    for (const [m, level] of fixture.motionSnap as [number, keyof typeof label][]) {
      const { view } = mount(theme({ motion: m }));
      expect(picked(view, "Motion"), String(m)).toBe(label[level]);
      view.unmount();
    }
  });

  it("sets what each pick names, motion as its level's own value", () => {
    const { view, set } = mount(theme({ motion: 0 }));
    fireEvent.click(view.getByRole("radio", { name: "Light" }));
    fireEvent.click(view.getByRole("radio", { name: "Midnight" }));
    fireEvent.click(view.getByRole("radio", { name: "Dusk violet" }));
    fireEvent.click(view.getByRole("radio", { name: "Ember" }));
    fireEvent.click(view.getByRole("radio", { name: "Abyssal" }));
    fireEvent.click(view.getByRole("radio", { name: "Ambient" }));
    fireEvent.click(view.getByRole("radio", { name: "Rich" }));
    fireEvent.click(view.getByRole("radio", { name: "Functional" }));
    fireEvent.click(view.getByRole("switch", { name: "Keep his motion when Windows asks for less" }));
    fireEvent.click(view.getByRole("switch", { name: "Parchment grain" }));
    expect(set.mock.calls).toEqual([
      ["theme.theme", "light"],
      ["theme.mood", "midnight"],
      ["theme.mood", "dusk"],
      ["theme.accent", "ember"],
      ["theme.accent", "abyssal"],
      ["theme.motion", fixture.motionLevels.ambient],
      ["theme.motion", fixture.motionLevels.rich],
      ["theme.motion", fixture.motionLevels.functional],
      ["theme.motionUser", true],
      ["theme.grain", false],
    ]);
  });

  it("shows what the settings say, not what was clicked", () => {
    const set = vi.fn();
    const [t, setT] = createSignal<Theme>(DEFAULTS);
    const view = render(() => <Appearance theme={t()} lessMotion={false} set={set} />);
    fireEvent.click(view.getByRole("radio", { name: "Midnight" }));
    expect(set).toHaveBeenCalledWith("theme.mood", "midnight");
    expect(picked(view, "Colour mood")).toBe("Ink + candle");
    // The shell's settings:changed, from here or the tray.
    setT(theme({ mood: "midnight" }));
    expect(picked(view, "Colour mood")).toBe("Midnight");
  });

  it("waits for the settings before it changes anything", () => {
    const { view, set } = mount(null);
    for (const r of view.getAllByRole("radio")) {
      expect(r).toBeDisabled();
      fireEvent.click(r);
    }
    for (const s of view.getAllByRole("switch")) {
      expect(s).toBeDisabled();
      fireEvent.click(s);
    }
    expect(set).not.toHaveBeenCalled();
  });

  it("says when Windows has him keeping still, and when keeping his motion lifts it", () => {
    const motionRow = (v: ReturnType<typeof render>) =>
      v.getByRole("radiogroup", { name: "Motion" }).closest(".row")!.textContent!;
    const held = mount(theme({ motion: 1 }), true);
    expect(motionRow(held.view)).toContain("Windows is asking for less motion, so");
    expect(held.view.container.textContent).toContain("Windows is asking for less motion right now.");
    held.view.unmount();
    const kept = mount(theme({ motion: 1, motionUser: true }), true);
    expect(motionRow(kept.view)).toContain("He sways freely.");
    kept.view.unmount();
    const free = mount(theme({ motion: 1 }), false);
    expect(motionRow(free.view)).toContain("He sways freely.");
    expect(free.view.container.textContent).not.toContain("right now");
  });

  it("says what each level does", () => {
    expect(motionLine(theme({ motion: 0 }), false)).toContain("only blinks and breathes");
    expect(motionLine(theme({ motion: 0.6 }), false)).toBe("He sways a little.");
    expect(motionLine(theme({ motion: 1 }), false)).toBe("He sways freely.");
    expect(motionLine(theme({ motion: 0.6 }), true)).toContain("Windows is asking");
    expect(motionLine(theme({ motion: 0.6, motionUser: true }), true)).toBe("He sways a little.");
    expect(motionLine(null, true)).toBe("He sways a little.");
  });
});

describe("the theme on the control window", () => {
  it("puts the grain on this window only while the setting is on", () => {
    wearTheme(document, theme({ grain: true }), false);
    expect(document.body.classList.contains("grain")).toBe(true);
    expect(document.documentElement.getAttribute("data-mood")).toBe("ink");
    wearTheme(document, theme({ grain: false, mood: "midnight" }), false);
    expect(document.body.classList.contains("grain")).toBe(false);
    expect(document.documentElement.getAttribute("data-mood")).toBe("midnight");
  });

  it("takes the shared attributes, reduced motion included", () => {
    wearTheme(document, theme({ motion: 1 }), true);
    expect(document.documentElement.getAttribute("data-motion")).toBe("functional");
    wearTheme(document, theme({ motion: 1, motionUser: true }), true);
    expect(document.documentElement.getAttribute("data-motion")).toBe("rich");
    expect(document.documentElement.style.getPropertyValue("--motion")).toBe("");
  });
});
