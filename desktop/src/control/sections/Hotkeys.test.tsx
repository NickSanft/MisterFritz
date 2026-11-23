import { describe, expect, it, vi } from "vitest";
import { createSignal } from "solid-js";
import { fireEvent, render } from "@solidjs/testing-library";
import fixture from "../../avatar/wire.fixture.json";
import { HOTKEY_DEFAULTS, type HotkeyAction, type HotkeysBound } from "../api";
import { HOTKEY_DOES } from "../view";
import { Hotkeys, HOTKEY_TITLES, type HotkeysProps } from "./Hotkeys";

const DEFAULTS = { ...HOTKEY_DEFAULTS };
const BOUND: HotkeysBound = { ...HOTKEY_DEFAULTS };

/** The section over settings the test can change, as the shell's answers
 * and settings:changed would. */
function mount(more: Partial<HotkeysProps> = {}, keys: Record<HotkeyAction, string> | null = DEFAULTS) {
  const [k, setKeys] = createSignal(keys);
  const [b, setBound] = createSignal<HotkeysBound | null>(more.bound === undefined ? BOUND : more.bound);
  const set = vi.fn(more.set ?? (async (_a: HotkeyAction, _k: string): Promise<string | null> => null));
  const view = render(() => <Hotkeys keys={k()} bound={b()} set={set} />);
  const capture = (action: HotkeyAction) =>
    view.getByRole("button", { name: new RegExp(`^${HOTKEY_TITLES[action]} hotkey:`) });
  const reset = (action: HotkeyAction) =>
    view.getByRole("button", { name: `${HOTKEY_TITLES[action]}: Default (${HOTKEY_DEFAULTS[action]})` }) as HTMLButtonElement;
  return { view, set, setKeys, setBound, capture, reset };
}

/** A key pressed anywhere in the window, as the capture listens for it. */
const press = (key: string, more: KeyboardEventInit = {}) =>
  window.dispatchEvent(new KeyboardEvent("keydown", { key, cancelable: true, bubbles: true, ...more }));

const settle = () => new Promise((r) => setTimeout(r, 0));

describe("Settings › Hotkeys", () => {
  it("is a section the shell may open at, under its sigil, after Fritz", () => {
    const { view } = mount();
    const section = view.container.querySelector("section#hotkeys");
    expect(section).not.toBeNull();
    expect(section!.querySelector("svg.section-sigil")).not.toBeNull();
    expect(fixture.controlSections.indexOf("hotkeys")).toBe(fixture.controlSections.indexOf("fritz") + 1);
    expect(view.getByText(/A key Fritz or another app already uses does its job rather than being captured/))
      .toBeInTheDocument();
  });

  it("has a row for each of his four keys, push-to-talk's among them", () => {
    const { view, capture } = mount();
    for (const [action, title] of Object.entries(HOTKEY_TITLES) as [HotkeyAction, string][]) {
      expect(view.getByText(title)).toBeInTheDocument();
      expect(capture(action)).toHaveAccessibleName(`${title} hotkey: ${HOTKEY_DEFAULTS[action].split("+").join(" ")}`);
      expect(view.getByText(HOTKEY_DOES[action])).toBeInTheDocument();
    }
    expect(view.getAllByRole("button", { name: / hotkey: / })).toHaveLength(4);
    expect(capture("ptt")).toHaveAccessibleName("Talk to him hotkey: Ctrl Alt V");
  });

  it("a captured key is bound to its row's action", async () => {
    const { set, capture } = mount();
    fireEvent.click(capture("prompt"), { detail: 1 });
    expect(capture("prompt")).toHaveAccessibleName("Ask Fritz hotkey: press a key, Esc to cancel");
    press("k", { ctrlKey: true, altKey: true });
    await settle();
    expect(set).toHaveBeenCalledTimes(1);
    expect(set).toHaveBeenCalledWith("prompt", "Ctrl+Alt+K");
  });

  it("the Windows key is saved as Super and shown as Win", async () => {
    const { set, capture, setKeys } = mount();
    fireEvent.click(capture("park"), { detail: 1 });
    press("p", { ctrlKey: true, metaKey: true });
    await settle();
    expect(set).toHaveBeenCalledWith("park", "Ctrl+Super+P");
    // The settings that come back hold it as the shell does.
    setKeys({ ...DEFAULTS, park: "Ctrl+Super+P" });
    expect(capture("park")).toHaveAccessibleName("Hide or show him hotkey: Ctrl Win P");
    expect(capture("park")).toHaveTextContent("CtrlWinP");
  });

  it("a refused key is said on its row, which keeps the old key", async () => {
    const why = "Another app already uses Ctrl+Alt+K: pick another.";
    const set = vi.fn(async () => why);
    const { view, capture } = mount({ set });
    fireEvent.click(capture("arrange"), { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    await settle();
    expect(set).toHaveBeenCalledWith("arrange", "Ctrl+Alt+K");
    const alert = view.getByRole("alert");
    expect(alert).toHaveTextContent(why);
    expect(view.getAllByRole("alert")).toHaveLength(1);
    // On its own row: the row Arrange him, its line replaced by the words.
    expect(alert.closest(".row")).toHaveTextContent("Arrange him");
    expect(view.queryByText(HOTKEY_DOES.arrange)).toBeNull();
    expect(capture("arrange")).toHaveAccessibleName("Arrange him hotkey: Ctrl Alt A");
    // The next key tried clears it.
    set.mockResolvedValueOnce(null as unknown as string);
    fireEvent.click(capture("arrange"), { detail: 1 });
    press("j", { ctrlKey: true, altKey: true });
    await settle();
    expect(view.queryByRole("alert")).toBeNull();
    expect(view.getByText(HOTKEY_DOES.arrange)).toBeInTheDocument();
  });

  it("only the latest key tried on a row is answered there", async () => {
    // The first try is answered last, with a refusal; the second, at once.
    let answerFirst: (why: string | null) => void = () => {};
    const set = vi.fn<HotkeysProps["set"]>()
      .mockImplementationOnce(() => new Promise((r) => { answerFirst = r; }))
      .mockImplementationOnce(async () => null);
    const { view, capture } = mount({ set });
    fireEvent.click(capture("prompt"), { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    fireEvent.click(capture("prompt"), { detail: 1 });
    press("j", { ctrlKey: true, altKey: true });
    await settle();
    answerFirst("Another app already uses Ctrl+Alt+K: pick another.");
    await settle();
    expect(set.mock.calls).toEqual([["prompt", "Ctrl+Alt+K"], ["prompt", "Ctrl+Alt+J"]]);
    expect(view.queryByRole("alert")).toBeNull();
  });

  it("a refusal goes once what is held is read again: it may no longer be true", async () => {
    const set = vi.fn<HotkeysProps["set"]>(async () => "Ctrl+Alt+A is already Fritz's key to arrange him.");
    const { view, capture, setBound } = mount({ set });
    fireEvent.click(capture("prompt"), { detail: 1 });
    press("a", { ctrlKey: true, altKey: true });
    await settle();
    expect(view.getByRole("alert")).toHaveTextContent("already Fritz's key to arrange him");
    // Arrange moved elsewhere, or the window came forward: asked again.
    setBound({ ...BOUND, arrange: "Ctrl+Alt+J" });
    expect(view.queryByRole("alert")).toBeNull();
    expect(view.getByText(HOTKEY_DOES.prompt)).toBeInTheDocument();
  });

  it("the same refusal again is said again, as a new alert", async () => {
    let answer: (why: string | null) => void = () => {};
    const set = vi.fn<HotkeysProps["set"]>(() => new Promise((r) => { answer = r; }));
    const { view, capture } = mount({ set });
    const why = "Ctrl or Alt by itself is what apps use for their own keys.";
    fireEvent.click(capture("park"), { detail: 1 });
    press("s", { ctrlKey: true });
    answer(why);
    await settle();
    const first = view.getByRole("alert");
    fireEvent.click(capture("park"), { detail: 1 });
    press("f", { altKey: true });
    await settle();
    expect(view.queryByRole("alert"), "gone while the shell decides").toBeNull();
    answer(why);
    await settle();
    const second = view.getByRole("alert");
    expect(second).toHaveTextContent(why);
    expect(second).not.toBe(first);
  });

  it("a row tried waits for what is held before saying its key isn't working", async () => {
    // The new setting arrives (settings:changed) before what is held.
    let answer: (why: string | null) => void = () => {};
    const set = vi.fn<HotkeysProps["set"]>(() => new Promise((r) => { answer = r; }));
    const { view, capture, setKeys, setBound } = mount({ set });
    fireEvent.click(capture("prompt"), { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    setKeys({ ...DEFAULTS, prompt: "Ctrl+Alt+K" });
    await settle();
    expect(view.queryByText(/^Working on/)).toBeNull();
    expect(view.getByText(HOTKEY_DOES.prompt)).toBeInTheDocument();
    expect(view.getByRole("button", { name: "Ask Fritz: Default (Ctrl+Alt+F)" })).toBeDisabled();
    setBound({ ...BOUND, prompt: "Ctrl+Alt+K" });
    answer(null);
    await settle();
    expect(view.queryByText(/^Working on/)).toBeNull();
    expect(view.getByRole("button", { name: "Ask Fritz: Default (Ctrl+Alt+F)" })).toBeEnabled();
  });

  it("Default from the keyboard keeps the focus in its row", async () => {
    const state = { setKeys: (_: Record<HotkeyAction, string>) => {} };
    const set = vi.fn<HotkeysProps["set"]>(async () => {
      state.setKeys({ ...DEFAULTS });
      return null;
    });
    const m = mount({ set }, { ...DEFAULTS, arrange: "Ctrl+Alt+J" });
    state.setKeys = m.setKeys;
    m.reset("arrange").focus();
    fireEvent.click(m.reset("arrange"), { detail: 0 });
    await settle();
    await settle();
    expect(m.reset("arrange")).toBeDisabled();
    expect(document.activeElement).toBe(m.capture("arrange"));
  });

  it("a held Enter acts once in a row: its repeats click nothing", () => {
    const { capture, reset } = mount({}, { ...DEFAULTS, prompt: "Ctrl+Alt+K" });
    const field = capture("prompt");
    field.focus();
    // fireEvent answers false when the default was prevented.
    expect(fireEvent.keyDown(field, { key: "Enter", repeat: true })).toBe(false);
    expect(fireEvent.keyDown(reset("prompt"), { key: "Enter", repeat: true })).toBe(false);
    expect(fireEvent.keyDown(field, { key: "Enter" })).toBe(true);
    expect(fireEvent.keyDown(field, { key: " ", repeat: true })).toBe(true);
  });

  it("Default sends the default, and waits while the key is it", async () => {
    const { set, reset, setKeys } = mount({}, { ...DEFAULTS, prompt: "Ctrl+Alt+K" });
    expect(reset("prompt").disabled).toBe(false);
    expect(reset("arrange").disabled).toBe(true);
    expect(reset("park").disabled).toBe(true);
    fireEvent.click(reset("prompt"));
    await settle();
    expect(set.mock.calls).toEqual([["prompt", "Ctrl+Alt+F"]]);
    setKeys({ ...DEFAULTS });
    expect(reset("prompt").disabled).toBe(true);
  });

  it("says when the key set isn't the one working", () => {
    const { view } = mount({ bound: { ...BOUND, prompt: null, park: "Ctrl+Alt+H" } }, { ...DEFAULTS, park: "Ctrl+Alt+Nope" });
    expect(view.getByText(/^Not working: he couldn't have Ctrl\+Alt\+F when he started/)).toBeInTheDocument();
    expect(view.getByText(/^Working on Ctrl\+Alt\+H: Ctrl\+Alt\+Nope couldn't be used/)).toBeInTheDocument();
    expect(view.getByText(HOTKEY_DOES.arrange)).toBeInTheDocument();
  });

  it("waits for the settings, and for the shell to say what it holds", () => {
    const { view, capture, reset, setKeys, setBound } = mount({ bound: null }, null);
    expect(capture("prompt")).toHaveAccessibleName("Ask Fritz hotkey: not set");
    expect(reset("prompt").disabled).toBe(true);
    expect(view.queryByText(HOTKEY_DOES.prompt)).toBeNull();
    setKeys({ ...DEFAULTS });
    // Until the shell says, the key set is taken to be the one working.
    expect(view.getByText(HOTKEY_DOES.prompt)).toBeInTheDocument();
    setBound({ ...BOUND, prompt: null });
    expect(view.getByText(/^Not working/)).toBeInTheDocument();
  });
});
