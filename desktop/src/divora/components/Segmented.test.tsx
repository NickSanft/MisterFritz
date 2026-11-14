// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Segmented.test.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

import { describe, expect, it, vi } from "vitest";
import { createSignal } from "solid-js";
import { fireEvent, render, within } from "@solidjs/testing-library";
import { Segmented } from "./Segmented";

describe("Segmented", () => {
  it("picks an option", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Segmented options={["A", "B"]} value="A" onChange={onChange} />
    ));
    fireEvent.click(getByRole("tab", { name: "B" }));
    expect(onChange).toHaveBeenCalledWith("B");
  });

  // v1.54.0
  it("a disabled group changes nothing", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Segmented options={["A", "B"]} value="A" onChange={onChange} disabled />
    ));
    const b = getByRole("tab", { name: "B" });
    expect(b).toBeDisabled();
    fireEvent.click(b);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("one option can be disabled, with a title saying why", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Segmented<string>
        options={[
          { value: "60", label: "60 s" },
          {
            value: "120",
            label: "120 s",
            disabled: true,
            title: "Needs up to 578 MB; this PC allows 536 MB.",
          },
        ]}
        value="60"
        onChange={onChange}
      />
    ));
    const over = getByRole("tab", { name: "120 s" });
    expect(over).toBeDisabled();
    expect(over).toHaveAttribute("title", "Needs up to 578 MB; this PC allows 536 MB.");
    fireEvent.click(over);
    expect(onChange).not.toHaveBeenCalled();

    const ok = getByRole("tab", { name: "60 s" });
    expect(ok).toBeEnabled();
    expect(ok).not.toHaveAttribute("title");
    fireEvent.click(ok);
    expect(onChange).toHaveBeenCalledWith("60");
  });

  // v1.58.0: Push to modulate's Hold to apply / Hold to bypass. A focused
  // option takes Space, the in-app push-to-modulate key, so a pointer click
  // hands the focus back to the page; a keyboard click (detail 0) keeps it.
  it("blurOnPointer: a pointer click leaves the focus on the page, the keyboard keeps it", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Segmented options={["A", "B"]} value="A" onChange={onChange} blurOnPointer />
    ));
    const b = getByRole("tab", { name: "B" });
    b.focus();
    fireEvent.click(b, { detail: 1 });
    expect(onChange).toHaveBeenCalledWith("B");
    expect(document.activeElement).toBe(document.body);

    b.focus();
    fireEvent.click(b, { detail: 0 });
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(b);
  });

  it("without blurOnPointer a pointer click keeps the focus, as before", () => {
    const { getByRole } = render(() => (
      <Segmented options={["A", "B"]} value="A" onChange={vi.fn()} />
    ));
    const b = getByRole("tab", { name: "B" });
    b.focus();
    fireEvent.click(b, { detail: 1 });
    expect(document.activeElement).toBe(b);
  });

  // v1.61.0: Follow OBS's Off / Streaming / Recording is a choice of one,
  // named for its timer, so it's a radio group; every other host is the tab
  // list it was.
  describe("with ariaLabel (v1.61.0)", () => {
    const FOLLOW = [
      { value: "off", label: "Off" },
      { value: "stream", label: "Streaming" },
      { value: "record", label: "Recording" },
    ];
    function mountFollow(more: { disabledRecord?: boolean } = {}) {
      const onChange = vi.fn();
      const view = render(() => {
        const [v, setV] = createSignal("off");
        return (
          <Segmented<string>
            options={FOLLOW.map((o) => (o.value === "record" && more.disabledRecord ? { ...o, disabled: true } : o))}
            value={v()}
            onChange={(next) => {
              onChange(next);
              setV(next);
            }}
            ariaLabel="Follow OBS for Uptime"
          />
        );
      });
      return { view, onChange };
    }

    it("is a radio group by its name, each option a radio with aria-checked, no tab roles", () => {
      const { view, onChange } = mountFollow();
      const group = view.getByRole("radiogroup", { name: "Follow OBS for Uptime" });
      const radios = within(group).getAllByRole("radio");
      expect(radios.map((r) => r.textContent)).toEqual(["Off", "Streaming", "Recording"]);
      expect(radios.map((r) => r.getAttribute("aria-checked"))).toEqual(["true", "false", "false"]);
      for (const r of radios) expect(r).not.toHaveAttribute("aria-selected");
      expect(view.queryByRole("tablist")).toBeNull();
      expect(view.queryByRole("tab")).toBeNull();
      fireEvent.click(view.getByRole("radio", { name: "Recording" }));
      expect(onChange.mock.calls).toEqual([["record"]]);
      expect(radios.map((r) => r.getAttribute("aria-checked"))).toEqual(["false", "false", "true"]);
    });

    it("Tab reaches the checked option only; the arrows move round the ends, pick, and skip a disabled option", () => {
      const { view, onChange } = mountFollow({ disabledRecord: true });
      const [off, stream, record] = view.getAllByRole("radio") as [HTMLElement, HTMLElement, HTMLElement];
      expect([off, stream, record].map((r) => r.getAttribute("tabindex"))).toEqual(["0", "-1", "-1"]);
      off.focus();
      fireEvent.keyDown(off, { key: "ArrowRight" });
      expect(onChange.mock.calls).toEqual([["stream"]]);
      expect(document.activeElement).toBe(stream);
      expect([off, stream, record].map((r) => r.getAttribute("tabindex"))).toEqual(["-1", "0", "-1"]);
      // Recording can't be picked: Down goes round to Off.
      fireEvent.keyDown(stream, { key: "ArrowDown" });
      expect(onChange.mock.calls).toEqual([["stream"], ["off"]]);
      expect(document.activeElement).toBe(off);
      // Left and Up go back, round the start.
      fireEvent.keyDown(off, { key: "ArrowLeft" });
      expect(onChange.mock.calls.at(-1)).toEqual(["stream"]);
      fireEvent.keyDown(stream, { key: "ArrowUp" });
      expect(onChange.mock.calls.at(-1)).toEqual(["off"]);
      // Other keys change nothing.
      fireEvent.keyDown(off, { key: "Home" });
      fireEvent.keyDown(off, { key: "a" });
      expect(onChange).toHaveBeenCalledTimes(4);
    });

    it("without ariaLabel, arrows do nothing and every option is a tab in the tab order, as before", () => {
      const onChange = vi.fn();
      const view = render(() => <Segmented options={["A", "B"]} value="A" onChange={onChange} />);
      const tabs = view.getAllByRole("tab");
      expect(view.getByRole("tablist")).not.toHaveAttribute("aria-label");
      for (const t of tabs) {
        expect(t).not.toHaveAttribute("tabindex");
        expect(t).not.toHaveAttribute("aria-checked");
      }
      expect(tabs.map((t) => t.getAttribute("aria-selected"))).toEqual(["true", "false"]);
      fireEvent.keyDown(tabs[0]!, { key: "ArrowRight" });
      expect(onChange).not.toHaveBeenCalled();
    });
  });
});
