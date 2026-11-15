// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/HotkeyCapture.test.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// v1.59.0: HotkeyCapture holds its window `keydown` listener in the
// component and removes it however a capture ends. Before, a field removed
// while armed, or clicked again to cancel, left the listener behind: the
// next key anywhere was swallowed (prevented) and handed to a field nobody
// could see. Hourglass puts up to 32 of these on Settings, each named for
// its timer, beside a Delete.

import { fireEvent, render } from "@solidjs/testing-library";
import { createSignal, Show } from "solid-js";
import { describe, expect, it, vi } from "vitest";
import { HotkeyCapture } from "./HotkeyCapture";

/** A key pressed anywhere in the window; true when something prevented it. */
const press = (key: string, more: KeyboardEventInit = {}): boolean => {
  const e = new KeyboardEvent("keydown", { key, cancelable: true, bubbles: true, ...more });
  window.dispatchEvent(e);
  return e.defaultPrevented;
};

/** A mouse click (`detail` 1) or one from the keyboard (`detail` 0). */
const click = (el: Element, detail: number): void => {
  fireEvent.click(el, { detail });
};

const named = (state: "set" | "unset" | "capturing"): string =>
  state === "set"
    ? "BRB start or pause hotkey: Ctrl Alt F13"
    : state === "unset"
      ? "BRB start or pause hotkey: not set"
      : "BRB start or pause hotkey: press a key, Esc to cancel";

describe("HotkeyCapture (v1.59.0)", () => {
  it("captures the next key once, then lets keys go by", () => {
    const onChange = vi.fn();
    const view = render(() => <HotkeyCapture value={null} onChange={onChange} />);
    click(view.getByRole("button"), 1);
    expect(press("Shift")).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
    expect(press("F9", { ctrlKey: true, altKey: true })).toBe(true);
    expect(onChange.mock.calls).toEqual([[["Ctrl", "Alt", "F9"]]]);
    expect(press("F10")).toBe(false);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("unmounted while armed: a later key calls nothing and isn't prevented", () => {
    const onChange = vi.fn();
    const [shown, setShown] = createSignal(true);
    const view = render(() => (
      <Show when={shown()}>
        <HotkeyCapture value={null} onChange={onChange} />
      </Show>
    ));
    click(view.getByRole("button"), 1);
    setShown(false);
    expect(view.queryByRole("button")).toBeNull();
    expect(press("F9")).toBe(false);
    expect(press(" ")).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("the whole tree unmounted while armed: the same", () => {
    const onChange = vi.fn();
    const view = render(() => <HotkeyCapture value={null} onChange={onChange} />);
    click(view.getByRole("button"), 1);
    view.unmount();
    expect(press("F9")).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("the cancel click: a later key calls nothing and isn't prevented", () => {
    const onChange = vi.fn();
    const view = render(() => <HotkeyCapture value={null} onChange={onChange} />);
    const button = view.getByRole("button");
    click(button, 1);
    expect(button).toHaveTextContent("Press a key… (Esc to cancel)");
    click(button, 1);
    expect(button).toHaveTextContent("Click to set");
    expect(press("F9")).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
    // Armed again, it captures once.
    click(button, 1);
    expect(press("F9")).toBe(true);
    expect(onChange.mock.calls).toEqual([[["F9"]]]);
  });

  it("Escape: a later key calls nothing and isn't prevented", () => {
    const onChange = vi.fn();
    const view = render(() => <HotkeyCapture value={["F9"]} onChange={onChange} />);
    click(view.getByRole("button"), 1);
    expect(press("Escape")).toBe(true);
    expect(view.getByRole("button")).toHaveTextContent("F9");
    expect(press("F10")).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("ariaLabel names it in each state", () => {
    const [value, setValue] = createSignal<string[] | null>(null);
    const view = render(() => <HotkeyCapture value={value()} onChange={setValue} ariaLabel={named} />);
    expect(view.getByRole("button", { name: "BRB start or pause hotkey: not set" })).toBeInTheDocument();
    click(view.getByRole("button"), 1);
    expect(view.getByRole("button", { name: "BRB start or pause hotkey: press a key, Esc to cancel" })).toBeInTheDocument();
    press("F13", { ctrlKey: true, altKey: true });
    expect(value()).toEqual(["Ctrl", "Alt", "F13"]);
    expect(view.getByRole("button", { name: "BRB start or pause hotkey: Ctrl Alt F13" })).toBeInTheDocument();
    expect(view.queryByRole("button", { name: "Click to set" })).toBeNull();
  });

  it("without ariaLabel the name is its content, as before", () => {
    const unset = render(() => <HotkeyCapture value={null} onChange={() => {}} />);
    const button = unset.getByRole("button", { name: "Click to set" });
    expect(button).not.toHaveAttribute("aria-label");
    unset.unmount();
    const set = render(() => <HotkeyCapture value={["Ctrl", "M"]} onChange={() => {}} />);
    expect(set.getByRole("button", { name: /^Ctrl\s*M$/ })).not.toHaveAttribute("aria-label");
  });

  it("armed by a pointer, it gives the focus back once the key lands; from the keyboard it keeps it", () => {
    const onChange = vi.fn();
    const view = render(() => <HotkeyCapture value={null} onChange={onChange} />);
    const button = view.getByRole("button");

    button.focus();
    click(button, 1);
    expect(document.activeElement).toBe(button);
    press("F9");
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(document.body);

    // Tab, then Space or Enter: the keyboard keeps working the field.
    button.focus();
    click(button, 0);
    press("F10");
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(button);
  });

  it("the focus leaving the armed field cancels it: a key typed in a text field is the field's", () => {
    const onChange = vi.fn();
    const view = render(() => (
      <>
        <HotkeyCapture value={null} onChange={onChange} ariaLabel={named} />
        <input aria-label="Length" />
      </>
    ));
    const button = view.getByRole("button");
    const field = view.getByRole("textbox", { name: "Length" });
    // Armed by a mouse click, which in WebView2 focuses the button.
    button.focus();
    click(button, 1);
    expect(button).toHaveAccessibleName("BRB start or pause hotkey: press a key, Esc to cancel");
    // Then a click into Length, and "1" typed there.
    field.focus();
    expect(button).toHaveAccessibleName("BRB start or pause hotkey: not set");
    const e = new KeyboardEvent("keydown", { key: "1", cancelable: true, bubbles: true });
    field.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(false);
    expect(press(" ")).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("arming a second capture cancels the first: one key binds one field", () => {
    const first = vi.fn();
    const second = vi.fn();
    const view = render(() => (
      <>
        <HotkeyCapture value={null} onChange={first} />
        <HotkeyCapture value={null} onChange={second} />
      </>
    ));
    const [a, b] = view.getAllByRole("button");
    a!.focus();
    click(a!, 1);
    b!.focus();
    click(b!, 1);
    expect(a).toHaveTextContent("Click to set");
    expect(b).toHaveTextContent("Press a key… (Esc to cancel)");
    expect(press("F9")).toBe(true);
    expect(first).not.toHaveBeenCalled();
    expect(second.mock.calls).toEqual([[["F9"]]]);
  });

  it("a pointer-armed capture cancelled by Escape gives the focus back too", () => {
    const view = render(() => <HotkeyCapture value={null} onChange={() => {}} />);
    const button = view.getByRole("button");
    button.focus();
    click(button, 1);
    press("Escape");
    expect(document.activeElement).toBe(document.body);
  });
});
