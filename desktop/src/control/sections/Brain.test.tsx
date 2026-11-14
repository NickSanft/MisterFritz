import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@solidjs/testing-library";
import fixture from "../../avatar/wire.fixture.json";
import { parseStatus, type AvatarStatus } from "../../avatar/status";
import type { LinkStatus } from "../../shared/companion";
import { Brain, type BrainProps } from "./Brain";

const URL = "ws://127.0.0.1:8001/companion";
const SHOWN: AvatarStatus = { ...parseStatus(fixture.statusSamples.arranging)!, mode: "locked" };
const link = (name: keyof typeof fixture.companionStatusSamples): LinkStatus =>
  fixture.companionStatusSamples[name] as LinkStatus;

function mount(more: Partial<BrainProps> = {}) {
  const props: BrainProps = {
    url: URL,
    link: link("online"),
    avatar: SHOWN,
    tokenPresent: true,
    setUrl: vi.fn(async () => true),
    saveToken: vi.fn(async () => true),
    reconnect: vi.fn(),
    ...more,
  };
  const view = render(() => <Brain {...props} />);
  const address = () => view.getByLabelText("His brain's address") as HTMLInputElement;
  const token = () => view.getByLabelText("His token") as HTMLInputElement;
  const button = () => view.getByRole("button") as HTMLButtonElement;
  return { view, props, address, token, button };
}

const settle = () => new Promise((r) => setTimeout(r, 0));

describe("Settings › Brain", () => {
  it("is the first section, which the first run opens at, under its sigil", () => {
    const { view } = mount();
    const section = view.container.querySelector("section#brain");
    expect(section).not.toBeNull();
    expect(fixture.controlSections[0]).toBe("brain");
    expect(section!.querySelector("svg.section-sigil")).not.toBeNull();
  });

  it("shows the link as the avatar's page reported it, and who he is to the brain", () => {
    const { view } = mount();
    expect(view.getByText("Connected")).toBeInTheDocument();
    expect(view.getByText(`To his brain at ${URL}.`)).toBeInTheDocument();
    expect(view.getByText("Who he is to it")).toBeInTheDocument();
    expect(view.getByText(/desktop-nick-pc, talking with Nick/)).toBeInTheDocument();
    expect(view.container.querySelector(".dot.tone-ok")).not.toBeNull();
  });

  it("says why he isn't connected, and hides who he is", () => {
    const { view } = mount({ link: link("refused") });
    expect(view.getByText("His brain refused the token")).toBeInTheDocument();
    expect(view.container.querySelector(".dot.tone-bad")).not.toBeNull();
    expect(view.queryByText("Who he is to it")).toBeNull();
  });

  it("says when the brain is on this PC", () => {
    const here = mount();
    expect(here.view.getByText("On-device")).toBeInTheDocument();
    expect(here.view.getByText("On this PC.")).toBeInTheDocument();
    here.view.unmount();
    const away = mount({ url: "ws://10.0.0.5:8001/companion" });
    expect(away.view.queryByText("On-device")).toBeNull();
    expect(away.view.getByText("On another machine.")).toBeInTheDocument();
  });

  it("saves the address on Enter, and only a changed one", async () => {
    const { props, address } = mount();
    expect(address().value).toBe(URL);
    fireEvent.keyDown(address(), { key: "Enter" });
    expect(props.setUrl).not.toHaveBeenCalled();
    fireEvent.input(address(), { target: { value: "ws://10.0.0.5:8001/companion" } });
    fireEvent.keyDown(address(), { key: "Enter" });
    await settle();
    expect(props.setUrl).toHaveBeenCalledWith("ws://10.0.0.5:8001/companion");
  });

  it("doesn't save the address it already has, typed again", async () => {
    const { props, address } = mount();
    fireEvent.input(address(), { target: { value: URL } });
    fireEvent.blur(address());
    await settle();
    expect(props.setUrl).not.toHaveBeenCalled();
  });

  it("saves the address on leaving the field; Escape puts it back", async () => {
    const { props, address } = mount();
    fireEvent.input(address(), { target: { value: "ws://elsewhere/companion" } });
    fireEvent.blur(address());
    await settle();
    expect(props.setUrl).toHaveBeenCalledWith("ws://elsewhere/companion");
    const other = mount();
    fireEvent.input(other.address(), { target: { value: "ws://typo" } });
    fireEvent.keyDown(other.address(), { key: "Escape" });
    expect(other.address().value).toBe(URL);
    fireEvent.blur(other.address());
    expect(other.props.setUrl).not.toHaveBeenCalled();
  });

  it("keeps an address the shell refused, to be mended", async () => {
    const { address } = mount({ setUrl: vi.fn(async () => false) });
    fireEvent.input(address(), { target: { value: "http://wrong" } });
    fireEvent.keyDown(address(), { key: "Enter" });
    await settle();
    expect(address().value).toBe("http://wrong");
  });

  it("saves a pasted token with Connect, then clears the field", async () => {
    const { props, token, button } = mount({ tokenPresent: false, link: link("unconfigured") });
    expect(token().type).toBe("password");
    expect(button()).toHaveTextContent("Connect");
    expect(button()).toBeDisabled();
    fireEvent.input(token(), { target: { value: "t0k-pasted" } });
    expect(button()).toBeEnabled();
    fireEvent.click(button());
    await settle();
    expect(props.saveToken).toHaveBeenCalledWith("t0k-pasted");
    expect(token().value).toBe("");
    expect(props.reconnect).not.toHaveBeenCalled();
  });

  it("clears the field even when the token is refused", async () => {
    const { token, button } = mount({ saveToken: vi.fn(async () => false) });
    fireEvent.input(token(), { target: { value: "a bad token" } });
    fireEvent.click(button());
    await settle();
    expect(token().value).toBe("");
  });

  it("saves with Enter in the token field too", async () => {
    const { props, token } = mount();
    fireEvent.input(token(), { target: { value: "t0k-new" } });
    fireEvent.keyDown(token(), { key: "Enter" });
    await settle();
    expect(props.saveToken).toHaveBeenCalledWith("t0k-new");
  });

  it("offers Reconnect with a token saved and the field empty", () => {
    const { props, button, token } = mount({ link: link("refused") });
    expect(button()).toHaveTextContent("Reconnect");
    fireEvent.click(button());
    expect(props.reconnect).toHaveBeenCalledOnce();
    expect(props.saveToken).not.toHaveBeenCalled();
    fireEvent.input(token(), { target: { value: "   " } });
    expect(button(), "blank is no token").toHaveTextContent("Reconnect");
  });

  it("waits for the settings before the address can be changed", () => {
    const { address, view } = mount({ url: null, tokenPresent: null, link: null });
    expect(address()).toBeDisabled();
    expect(view.getByText("Asking how he's connected…")).toBeInTheDocument();
    expect(view.queryByText("On-device")).toBeNull();
  });
});
