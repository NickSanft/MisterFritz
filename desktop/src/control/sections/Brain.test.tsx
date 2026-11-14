import { describe, expect, it, vi } from "vitest";
import { createSignal } from "solid-js";
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
    ask: vi.fn(async () => true),
    ...more,
  };
  const view = render(() => <Brain {...props} />);
  const address = () => view.getByLabelText("His brain's address") as HTMLInputElement;
  const token = () => view.getByLabelText("His token") as HTMLInputElement;
  const button = () => view.getByRole("button", { name: /connect/i }) as HTMLButtonElement;
  const question = () => view.getByLabelText("Ask him something") as HTMLInputElement;
  const askButton = () => view.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
  return { view, props, address, token, button, question, askButton };
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

  it("asks him something while he's connected, then says where the answer is", async () => {
    const { props, question, askButton, view } = mount();
    expect(askButton()).toBeDisabled();
    fireEvent.input(question(), { target: { value: "good evening" } });
    expect(askButton()).toBeEnabled();
    fireEvent.click(askButton());
    await settle();
    expect(props.ask).toHaveBeenCalledWith("good evening");
    expect(question().value).toBe("");
    expect(view.getByText("He's answering on your desktop.")).toBeInTheDocument();
    // Typing again is a new question.
    fireEvent.input(question(), { target: { value: "and the weather?" } });
    expect(view.queryByText("He's answering on your desktop.")).toBeNull();
    fireEvent.keyDown(question(), { key: "Enter" });
    await settle();
    expect(props.ask).toHaveBeenLastCalledWith("and the weather?");
  });

  it("keeps a question the shell refused", async () => {
    const { question, askButton, view } = mount({ ask: vi.fn(async () => false) });
    fireEvent.input(question(), { target: { value: "good evening" } });
    fireEvent.click(askButton());
    await settle();
    expect(question().value).toBe("good evening");
    expect(view.queryByText("He's answering on your desktop.")).toBeNull();
  });

  it("asks nothing blank, and nothing until he's connected", async () => {
    const blank = mount();
    fireEvent.input(blank.question(), { target: { value: "   " } });
    expect(blank.askButton()).toBeDisabled();
    fireEvent.keyDown(blank.question(), { key: "Enter" });
    await settle();
    expect(blank.props.ask).not.toHaveBeenCalled();
    blank.view.unmount();
    for (const name of ["offline", "refused", "connecting", "unconfigured"] as const) {
      const away = mount({ link: link(name) });
      expect(away.question(), name).toBeDisabled();
      expect(away.askButton(), name).toBeDisabled();
      expect(away.view.getByText("Once he's connected, you can try him from here.")).toBeInTheDocument();
      away.view.unmount();
    }
    const unknown = mount({ link: null });
    expect(unknown.question()).toBeDisabled();
  });

  /** A question typed while connected, then the link drops: it isn't sent
   * into a session that isn't there. */
  it("won't ask once the link has gone, even with a question typed", async () => {
    const [now, setNow] = createSignal<LinkStatus>(link("online"));
    const ask = vi.fn(async () => true);
    const props: BrainProps = {
      url: URL, avatar: SHOWN, tokenPresent: true, setUrl: vi.fn(async () => true), saveToken: vi.fn(async () => true),
      reconnect: vi.fn(), ask, get link() { return now(); },
    };
    const view = render(() => <Brain {...props} />);
    const question = view.getByLabelText("Ask him something") as HTMLInputElement;
    const askButton = view.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
    fireEvent.input(question, { target: { value: "good evening" } });
    expect(askButton).toBeEnabled();
    setNow(link("offline"));
    expect(askButton).toBeDisabled();
    expect(question).toBeDisabled();
    fireEvent.keyDown(question, { key: "Enter" });
    fireEvent.click(askButton);
    await settle();
    expect(ask).not.toHaveBeenCalled();
    setNow(link("online"));
    expect(question.value, "kept for when he's back").toBe("good evening");
    fireEvent.keyDown(question, { key: "Enter" });
    await settle();
    expect(ask).toHaveBeenCalledWith("good evening");
  });

  it("asks once at a time", async () => {
    let answer: (ok: boolean) => void = () => {};
    const ask = vi.fn(() => new Promise<boolean>((r) => (answer = r)));
    const { question, askButton } = mount({ ask });
    fireEvent.input(question(), { target: { value: "good evening" } });
    fireEvent.click(askButton());
    fireEvent.keyDown(question(), { key: "Enter" });
    expect(askButton()).toBeDisabled();
    answer(true);
    await settle();
    expect(ask).toHaveBeenCalledOnce();
  });

  it("waits for the settings before the address can be changed", () => {
    const { address, view } = mount({ url: null, tokenPresent: null, link: null });
    expect(address()).toBeDisabled();
    expect(view.getByText("Asking how he's connected…")).toBeInTheDocument();
    expect(view.queryByText("On-device")).toBeNull();
  });
});
