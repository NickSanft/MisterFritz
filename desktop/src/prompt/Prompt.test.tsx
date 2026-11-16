import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { ASKING, Prompt, type PromptProps } from "./Prompt";
import { WAITING_LINE, WAITING_PLACEHOLDER, WINDOW_CLOSED, type PromptStatus } from "./view";

const ONLINE: PromptStatus = { tone: "ok", text: "Connected", rule: true };
const OFFLINE: PromptStatus = { tone: "bad", text: "Can't reach his brain", rule: false };

/** The prompt with a shell that answers `say` with `answers` in turn, and
 * window focus and blur the test fires. */
function mount(answers: (string | null)[] = [null], status: PromptStatus = ONLINE) {
  const [st, setSt] = createSignal(status);
  const [waiting, setWaiting] = createSignal(false);
  const focus: (() => void)[] = [];
  const blur: (() => void)[] = [];
  const say = vi.fn(async (_text: string) => answers.shift() ?? null);
  const close = vi.fn(async () => {});
  const props: PromptProps = {
    status: st,
    say,
    close,
    onWindowFocus: (cb) => {
      focus.push(cb);
      return () => {};
    },
    onWindowBlur: (cb) => {
      blur.push(cb);
      return () => {};
    },
    waiting,
  };
  const r = render(() => <Prompt {...props} />);
  const field = r.getByRole("textbox", { name: "Ask Fritz" }) as HTMLTextAreaElement;
  const type = (text: string) => fireEvent.input(field, { target: { value: text } });
  const press = (key: string, mods: Record<string, unknown> = {}) => {
    fireEvent.keyDown(field, { key, ...mods });
    fireEvent.keyUp(field, { key, ...mods });
  };
  const settle = () => new Promise((r) => setTimeout(r, 0));
  return {
    ...r, field, type, press, settle, say, close, setSt, setWaiting,
    focus: () => focus.forEach((f) => f()),
    blur: () => blur.forEach((f) => f()),
  };
}

describe("the prompt waiting for a click", () => {
  /** Mounted, then shown without the keyboard, as a refused gesture shows it. */
  function waitingPrompt(status: PromptStatus = ONLINE) {
    const p = mount([null], status);
    p.blur();
    p.setWaiting(true);
    return p;
  }

  it("asks for a click, as a status, while it hasn't the keyboard", () => {
    const p = waitingPrompt();
    expect(p.field).toHaveAttribute("placeholder", WAITING_PLACEHOLDER);
    const line = p.getByRole("status");
    expect(line).toHaveTextContent(WAITING_LINE);
    expect(line).toHaveClass("prompt-waiting");
  });

  it("waits from the start, in a page loaded hidden", () => {
    const hf = vi.spyOn(document, "hasFocus").mockReturnValue(false);
    const p = mount([null]);
    hf.mockRestore();
    // No blur: the window never had the keyboard.
    p.setWaiting(true);
    expect(p.field).toHaveAttribute("placeholder", WAITING_PLACEHOLDER);
    expect(p.getByRole("status")).toHaveTextContent(WAITING_LINE);
  });

  it("is ready to type in a page that mounted with the keyboard", () => {
    const hf = vi.spyOn(document, "hasFocus").mockReturnValue(true);
    const p = mount([null]);
    hf.mockRestore();
    p.setWaiting(true);
    expect(p.field).toHaveAttribute("placeholder", "Ask Fritz…");
    expect(p.getByRole("status")).toHaveTextContent("Connected");
  });

  it("is the ordinary prompt once clicked into, the draft kept", () => {
    const p = waitingPrompt();
    p.type("good evening");
    p.field.blur();
    p.focus();
    expect(p.field).toHaveAttribute("placeholder", "Ask Fritz…");
    expect(p.getByRole("status")).toHaveTextContent("Connected");
    expect(p.getByRole("status")).not.toHaveClass("prompt-waiting");
    expect(document.activeElement).toBe(p.field);
    expect(p.field.value).toBe("good evening");
  });

  it("asks again at a later wait, after an open that had the keyboard", () => {
    const p = mount([null]);
    p.focus();
    expect(p.field).toHaveAttribute("placeholder", "Ask Fritz…");
    // Closed by its blur; the next gesture's take refused: it waits.
    p.blur();
    p.setWaiting(true);
    expect(p.field).toHaveAttribute("placeholder", WAITING_PLACEHOLDER);
    expect(p.getByRole("status")).toHaveTextContent(WAITING_LINE);
  });

  it("stops asking when the shell says it no longer waits", () => {
    const p = waitingPrompt();
    p.setWaiting(false);
    expect(p.field).toHaveAttribute("placeholder", "Ask Fritz…");
    expect(p.getByRole("status")).toHaveTextContent("Connected");
  });

  it("says why he can't be asked, rather than ask for a click", () => {
    const p = waitingPrompt({ tone: "bad", text: WINDOW_CLOSED, rule: false });
    expect(p.getByRole("status")).toHaveTextContent(WINDOW_CLOSED);
    expect(p.field).toHaveAttribute("placeholder", WAITING_PLACEHOLDER);
  });

  /** A wait is a new open: what the open before was refused is no part of
   * it, so the line asks for a click; the draft stays. */
  it("leaves the open before's refusal out of a wait", async () => {
    const p = mount(["There's nothing to ask."]);
    p.type("good evening");
    p.press("Enter");
    await p.settle();
    expect(p.getByRole("alert")).toHaveTextContent("There's nothing to ask.");
    p.blur();
    p.setWaiting(true);
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent(WAITING_LINE);
    expect(p.field.value).toBe("good evening");
  });

  /** A question still out from the open before is let go at a wait: the
   * line asks for a click, not "Asking him…", and its late refusal says
   * nothing here. */
  it("lets a question from the open before go at a wait", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("the first");
    p.press("Enter");
    p.press("Escape");
    p.blur();
    p.setWaiting(true);
    expect(p.getByRole("status")).toHaveTextContent(WAITING_LINE);
    answer("His window didn't answer just now: try again.");
    await p.settle();
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent(WAITING_LINE);
  });

  /** A gesture given the keyboard: told it waits only after the window's
   * focus began the open, the question just asked is kept. */
  it("keeps a question asked in an open begun by the window's focus", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.focus();
    p.type("good evening");
    p.press("Enter");
    p.setWaiting(true);
    p.setWaiting(false);
    expect(p.getByRole("status")).toHaveTextContent(ASKING);
    answer("His window didn't answer just now: try again.");
    await p.settle();
    expect(p.getByRole("alert")).toHaveTextContent("try again");
  });
});

describe("the prompt", () => {
  it("takes the keyboard into its field as it mounts", () => {
    const p = mount();
    expect(document.activeElement).toBe(p.field);
    expect(p.field).toHaveAttribute("autocomplete", "off");
    expect(p.field).toHaveAttribute("placeholder", "Ask Fritz…");
  });

  it("asks on Enter's release, then clears, and ↑ brings the question back", async () => {
    const p = mount([null]);
    p.type("good evening");
    fireEvent.keyDown(p.field, { key: "Enter" });
    expect(p.say).not.toHaveBeenCalled();
    fireEvent.keyUp(p.field, { key: "Enter" });
    await p.settle();
    expect(p.say).toHaveBeenCalledWith("good evening");
    expect(p.field.value).toBe("");
    p.press("ArrowUp");
    expect(p.field.value).toBe("good evening");
  });

  it("keeps the draft and says why, when the shell refuses", async () => {
    const p = mount(["He isn't connected to his brain just now."]);
    p.type("good evening");
    p.press("Enter");
    await p.settle();
    expect(p.field.value).toBe("good evening");
    expect(p.getByRole("alert")).toHaveTextContent("He isn't connected to his brain just now.");
    p.press("ArrowUp");
    expect(p.field.value, "↑ never replaces a draft").toBe("good evening");
  });

  it("forgets the refusal once the owner types", async () => {
    const p = mount(["There's nothing to ask."]);
    p.press("Enter");
    await p.settle();
    expect(p.queryByRole("alert")).not.toBeNull();
    p.type("x");
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent("Connected");
  });

  it("closes on Esc, keeping the draft", () => {
    const p = mount();
    p.type("half a thought");
    p.press("Escape");
    expect(p.close).toHaveBeenCalledOnce();
    expect(p.field.value).toBe("half a thought");
  });

  it("asks one question at a time", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("one");
    p.press("Enter");
    p.press("Enter");
    expect(p.say).toHaveBeenCalledOnce();
    answer(null);
    await p.settle();
    p.type("two");
    p.press("Enter");
    expect(p.say).toHaveBeenCalledTimes(2);
  });

  it("makes a new line with Shift+Enter, never a question", async () => {
    const p = mount();
    p.type("a");
    p.press("Enter", { shiftKey: true });
    await p.settle();
    expect(p.say).not.toHaveBeenCalled();
  });

  it("takes the keyboard into its field each time the window gets it, forgetting a refusal", async () => {
    const p = mount(["There's nothing to ask."]);
    p.press("Enter");
    await p.settle();
    p.field.blur();
    p.focus();
    expect(document.activeElement).toBe(p.field);
    expect(p.queryByRole("alert")).toBeNull();
  });

  it("forgets an armed key when the window loses or gets the keyboard", async () => {
    const p = mount();
    p.type("hello");
    fireEvent.keyDown(p.field, { key: "Enter" });
    p.blur();
    fireEvent.keyUp(p.field, { key: "Enter" });
    fireEvent.keyDown(p.field, { key: "Escape" });
    p.focus();
    fireEvent.keyUp(p.field, { key: "Escape" });
    await p.settle();
    expect(p.say).not.toHaveBeenCalled();
    expect(p.close).not.toHaveBeenCalled();
  });

  it("shows the rule only while he can be asked, and says the link in words", () => {
    const p = mount();
    const rule = p.container.querySelector(".prompt-rule") as HTMLElement;
    expect(rule.style.visibility).toBe("visible");
    p.setSt(OFFLINE);
    expect(rule.style.visibility).toBe("hidden");
    expect(p.getByRole("status")).toHaveTextContent("Can't reach his brain");
    expect(p.container.querySelector(".prompt-dot")).toHaveClass("bad");
    expect(p.container.querySelector(".prompt-dot")).toHaveAttribute("aria-hidden", "true");
  });

  /** A refusal is of its moment: once the link or his window changes, the
   * line says how things stand now, and the dot and the words agree. */
  it("forgets a refusal once the link changes", async () => {
    const p = mount(["He isn't connected to his brain just now."], OFFLINE);
    p.type("are you there");
    p.press("Enter");
    await p.settle();
    expect(p.getByRole("alert")).toHaveTextContent("He isn't connected");
    // A report that changes nothing the line says keeps the refusal up.
    p.setSt({ ...OFFLINE });
    expect(p.getByRole("alert")).toHaveTextContent("He isn't connected");
    p.setSt(ONLINE);
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent("Connected");
    expect(p.field.value).toBe("are you there");
  });

  /** What the owner types while a question is out is theirs: the
   * question's being taken doesn't wipe it. */
  it("keeps what was typed while asking", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("first");
    p.press("Enter");
    p.type("first, and then");
    answer(null);
    await p.settle();
    expect(p.field.value).toBe("first, and then");
  });

  /** A question from the open before that is refused says nothing in this
   * open: the line still says how things stand, the new draft untouched. */
  it("keeps the open before's refusal out of this one", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("the first");
    p.press("Enter");
    p.press("Escape");
    p.focus();
    p.type("the second");
    answer("His window didn't answer just now: try again.");
    await p.settle();
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent("Connected");
    expect(p.field.value).toBe("the second");
  });

  /** A question from the open before that is taken leaves the field if it
   * is still what's there, so Enter can't ask it twice. */
  it("clears a question taken in the open before, if it is still in the field", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("what's on tomorrow");
    p.press("Enter");
    p.press("Escape");
    p.focus();
    answer(null);
    await p.settle();
    expect(p.field.value).toBe("");
    p.press("ArrowUp");
    expect(p.field.value).toBe("what's on tomorrow");
  });

  /** While the shell decides, the line says so; a retry takes the old
   * refusal down first, so the same words refused again are said again. */
  it("says it is asking, and says a refusal afresh each time", async () => {
    let answer!: (v: string | null) => void;
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => (answer = r)));
    p.type("hello");
    p.press("Enter");
    expect(p.getByRole("status")).toHaveTextContent(ASKING);
    answer("His window didn't answer just now: try again.");
    await p.settle();
    const first = p.getByRole("alert");
    p.press("Enter");
    expect(p.queryByRole("alert")).toBeNull();
    expect(p.getByRole("status")).toHaveTextContent(ASKING);
    answer("His window didn't answer just now: try again.");
    await p.settle();
    expect(p.getByRole("alert")).not.toBe(first);
    expect(p.getByRole("alert")).toHaveTextContent("try again");
  });

  /** A question still out when the prompt was closed and opened again is
   * let go: its answer touches nothing in the new open, which can ask its
   * own at once. */
  it("lets a question from the open before go", async () => {
    const answers: ((v: string | null) => void)[] = [];
    const p = mount();
    p.say.mockImplementation(() => new Promise((r) => answers.push(r)));
    p.type("the first");
    p.press("Enter");
    p.focus();
    p.type("the second");
    answers[0](null);
    await p.settle();
    expect(p.field.value).toBe("the second");
    expect(p.getByRole("status")).toHaveTextContent("Connected");
    p.press("Enter");
    expect(p.say).toHaveBeenCalledTimes(2);
    expect(p.say).toHaveBeenLastCalledWith("the second");
    answers[1]("There's nothing to ask.");
    await p.settle();
    expect(p.getByRole("alert")).toHaveTextContent("There's nothing to ask.");
  });

  /** A click on the line, the border or a corner leaves the keyboard in
   * the field, where Enter and Esc work; a click in the field is the field's. */
  it("keeps the keyboard in its field through a click elsewhere", () => {
    const p = mount();
    const line = p.getByRole("status");
    expect(fireEvent.mouseDown(line)).toBe(false);
    expect(fireEvent.mouseDown(document.body)).toBe(false);
    expect(fireEvent.mouseDown(p.field)).toBe(true);
    expect(document.activeElement).toBe(p.field);
    p.press("Escape");
    expect(p.close).toHaveBeenCalledOnce();
  });

  it("says it is local", () => {
    const p = mount();
    expect(p.container.querySelector(".prompt-local")).toHaveTextContent("LOCAL · NO ACCOUNT");
  });
});
