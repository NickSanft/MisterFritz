// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Toggle.test.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@solidjs/testing-library";
import { Toggle } from "./Toggle";

describe("Toggle", () => {
  it("is a switch that asks for the other state", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => <Toggle on={false} onChange={onChange} ariaLabel="X" />);
    const sw = getByRole("switch", { name: "X" });
    expect(sw).toHaveAttribute("aria-checked", "false");
    fireEvent.click(sw);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  // v1.54.0
  it("disabled blocks the change, and can say why", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Toggle
        on
        onChange={onChange}
        ariaLabel="X"
        disabled
        title="Replay is on — turn it off to change these."
      />
    ));
    const sw = getByRole("switch", { name: "X" });
    expect(sw).toBeDisabled();
    expect(sw).toHaveAttribute("title", "Replay is on — turn it off to change these.");
    fireEvent.click(sw);
    expect(onChange).not.toHaveBeenCalled();
  });

  // v1.58.0: Push to modulate's switch. A focused switch takes Space, the
  // in-app push-to-modulate key, so a pointer click hands the focus back to
  // the page; a keyboard click (detail 0) keeps it.
  it("blurOnPointer: a pointer click leaves the focus on the page, the keyboard keeps it", () => {
    const onChange = vi.fn();
    const { getByRole } = render(() => (
      <Toggle on={false} onChange={onChange} ariaLabel="X" blurOnPointer />
    ));
    const sw = getByRole("switch", { name: "X" });
    sw.focus();
    fireEvent.click(sw, { detail: 1 });
    expect(onChange).toHaveBeenCalledWith(true);
    expect(document.activeElement).toBe(document.body);

    sw.focus();
    fireEvent.click(sw, { detail: 0 });
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(sw);
  });

  it("without blurOnPointer a pointer click keeps the focus, as before", () => {
    const { getByRole } = render(() => <Toggle on={false} onChange={vi.fn()} ariaLabel="X" />);
    const sw = getByRole("switch", { name: "X" });
    sw.focus();
    fireEvent.click(sw, { detail: 1 });
    expect(document.activeElement).toBe(sw);
  });

  it("ariaDescribedBy names the line that says what it does", () => {
    const { getByRole } = render(() => (
      <>
        <Toggle on onChange={vi.fn()} ariaLabel="X" ariaDescribedBy="x-line" />
        <div id="x-line">Off · your effects are always on.</div>
      </>
    ));
    expect(getByRole("switch", { name: "X" })).toHaveAccessibleDescription(
      "Off · your effects are always on.",
    );
  });
});
