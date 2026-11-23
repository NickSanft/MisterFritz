import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@solidjs/testing-library";
import fixture from "../avatar/wire.fixture.json";
import { parseStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { HOTKEY_DEFAULTS } from "./api";
import { Wizard, type WizardProps } from "./Wizard";

const settle = () => new Promise((r) => setTimeout(r, 0));

function mount(more: Partial<WizardProps> = {}) {
  const props: WizardProps = {
    brain: {
      url: "ws://127.0.0.1:8001/companion",
      link: fixture.companionStatusSamples.online as LinkStatus,
      avatar: parseStatus(fixture.statusSamples.arranging),
      tokenPresent: true,
      promptKey: "Ctrl+Alt+F",
      setUrl: vi.fn(async () => true),
      saveToken: vi.fn(async () => true),
      reconnect: vi.fn(),
      ask: vi.fn(async () => true),
    },
    model: { model: null, openFolder: vi.fn(), reload: vi.fn() },
    hotkeys: { keys: { ...HOTKEY_DEFAULTS }, bound: { ...HOTKEY_DEFAULTS }, muted: false, set: vi.fn(async () => null) },
    link: fixture.companionStatusSamples.online as LinkStatus,
    modelInfo: null,
    bound: { ...HOTKEY_DEFAULTS },
    tokenPresent: true,
    skip: vi.fn(),
    summon: vi.fn(async () => {}),
    ...more,
  };
  const view = render(() => <Wizard {...props} />);
  const button = (name: string) => view.getByRole("button", { name }) as HTMLButtonElement;
  const heading = () => view.container.querySelector("h2.wizard-title") as HTMLHeadingElement;
  const current = () => view.container.querySelector("[aria-current=step]")?.textContent;
  return { view, props, button, heading, current };
}

afterEach(() => vi.restoreAllMocks());

describe("the first run", () => {
  it("opens at his brain, with its rows, the heading focused", () => {
    const { view, heading, current } = mount();
    expect(view.container.querySelector("[data-view=wizard]")).not.toBeNull();
    expect(heading()).toHaveTextContent("Connect his brain");
    expect(document.activeElement).toBe(heading());
    expect(current()).toContain("Connect his brain");
    expect(view.getByLabelText("His token")).toBeInTheDocument();
    expect(view.getByText("Step 1 of 4: Connect his brain")).toHaveAttribute("aria-live", "polite");
    expect(view.queryByRole("button", { name: "Back" })).toBeNull();
  });

  it("walks forward and back, each step's heading taking the focus and the line saying where it is", () => {
    const { view, button, heading, current } = mount();
    // A press puts the focus on the button first, as a real one does.
    const pressed = (name: string) => {
      button(name).focus();
      expect(document.activeElement).toBe(button(name));
      fireEvent.click(button(name));
    };
    pressed("Continue");
    expect(heading()).toHaveTextContent("Choose his look");
    expect(document.activeElement).toBe(heading());
    expect(view.getByText("Step 2 of 4: Choose his look")).toBeInTheDocument();
    expect(view.getByRole("button", { name: "Open" })).toBeInTheDocument();
    fireEvent.click(button("Continue"));
    expect(heading()).toHaveTextContent("Learn his keys");
    expect(view.getByRole("button", { name: "Ask Fritz hotkey: Ctrl Alt F" })).toBeInTheDocument();
    expect(current()).toContain("Learn his keys");
    // The steps before are marked done.
    expect(view.container.querySelectorAll(".wizard-step.is-done")).toHaveLength(2);
    pressed("Back");
    expect(heading()).toHaveTextContent("Choose his look");
    expect(document.activeElement).toBe(heading());
    fireEvent.click(button("Continue"));
    fireEvent.click(button("Continue"));
    expect(heading()).toHaveTextContent("Summon him");
    expect(view.getByText("He's connected to his brain.")).toBeInTheDocument();
    expect(view.getByText("Press Ctrl+Alt+F in any app to ask him something.")).toBeInTheDocument();
    expect(view.queryByRole("button", { name: "Continue" })).toBeNull();
    expect(button("Summon Fritz")).toBeInTheDocument();
  });

  it("Summon Fritz, on the last step only, summons him once", async () => {
    let done: () => void = () => {};
    const summon = vi.fn(() => new Promise<void>((r) => { done = r; }));
    const { button, props } = mount({ summon });
    for (let i = 0; i < 3; i += 1) fireEvent.click(button("Continue"));
    expect(props.summon).not.toHaveBeenCalled();
    fireEvent.click(button("Summon Fritz"));
    expect(button("Summon Fritz")).toBeDisabled();
    fireEvent.click(button("Summon Fritz"));
    done();
    await settle();
    expect(summon).toHaveBeenCalledTimes(1);
    expect(button("Summon Fritz")).toBeEnabled();
  });

  it("Skip setup leaves it, from any step", () => {
    const { button, props } = mount();
    fireEvent.click(button("Continue"));
    fireEvent.click(button("Skip setup"));
    expect(props.skip).toHaveBeenCalledTimes(1);
  });

  it("Esc does nothing here: Brain's field still has its own Esc", () => {
    const { view, props, heading } = mount();
    fireEvent.keyDown(heading(), { key: "Escape" });
    fireEvent.keyDown(window, { key: "Escape" });
    const address = view.getByLabelText("His brain's address") as HTMLInputElement;
    fireEvent.input(address, { target: { value: "ws://elsewhere/companion" } });
    fireEvent.keyDown(address, { key: "Escape" });
    expect(address.value).toBe("ws://127.0.0.1:8001/companion");
    expect(props.skip).not.toHaveBeenCalled();
    expect(heading()).toHaveTextContent("Connect his brain");
  });

  it("keeps nothing: the browser's storage is never written", () => {
    const write = vi.spyOn(Storage.prototype, "setItem");
    const { button, view } = mount();
    for (let i = 0; i < 3; i += 1) fireEvent.click(button("Continue"));
    fireEvent.click(button("Skip setup"));
    view.unmount();
    expect(write).not.toHaveBeenCalled();
  });
});
