// @vitest-environment jsdom
/**
 * The bubble in a page: what it shows of a turn, written as text only, and
 * written (and measured) only when something changed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EDGE, GAP, HIDDEN, type Envelope } from "./bubble";
import { LISTENING, NOTHING_TO_SAY, SIGN_SIZE, THINKING, bubbleText, createBubble, signAt } from "./BubbleView";
import { INITIAL, type TurnView } from "./state";

const SHOWN: Envelope = { visible: true, opacity: 1, scale: 1, blurPx: 0 };
const VIEW = { width: 340, height: 640 };
const HEAD = { x: 170, y: 200 };
const turn = (more: Partial<TurnView> = {}): TurnView => ({
  ...INITIAL, turnId: "t1", phase: "thinking", ended: false, revision: 1, ...more,
});

let widthReads = 0;
beforeEach(() => {
  document.body.innerHTML = "";
  widthReads = 0;
  // jsdom lays nothing out: a bubble is 200 px wide, and its text fits.
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(function (this: HTMLElement) {
    if (this.classList.contains("bubble")) widthReads++;
    return 200;
  });
});
afterEach(() => vi.restoreAllMocks());

function mount() {
  const b = createBubble(document);
  const q = (sel: string) => b.element.querySelector(sel) as HTMLElement;
  return { b, root: b.element, q };
}

describe("what the bubble shows of a turn", () => {
  it("says he is listening while push-to-talk is held", () => {
    expect(bubbleText({ ...INITIAL, turnId: "ptt-1", phase: "listening", ended: false }).body).toBe(LISTENING);
    expect(bubbleText({ ...INITIAL, turnId: "ptt-1", phase: "transcribing", ended: false }).body).toBe(THINKING);
  });

  it("an ellipsis while he thinks, the reply as it comes", () => {
    expect(bubbleText(turn()).body).toBe(THINKING);
    expect(bubbleText(turn({ reply: "**Good** evening" })).body).toBe("Good evening");
  });

  /** Never an empty bubble: a turn that ended with nothing to show says so
   * in his words; one with an apology or a note shows those alone. */
  it("a line of his own for a reply with nothing in it", () => {
    expect(bubbleText(turn({ ended: true })).body).toBe(NOTHING_TO_SAY);
    expect(bubbleText(turn({ ended: true, reply: "```\n```" })).body).toBe(NOTHING_TO_SAY);
    expect(bubbleText(turn({ ended: true, reply: "  \n " })).body).toBe(NOTHING_TO_SAY);
    expect(bubbleText(turn({ ended: true, apology: "Forgive me, sir." })).body).toBe("");
    expect(bubbleText(turn({ ended: true, interrupted: true })).body).toBe("");
    expect(bubbleText(turn({ ended: true, pictures: 1 })).body).toBe("");
    expect(bubbleText(turn({ ended: false, reply: " " })).body).toBe(THINKING);
  });

  it("the progress while the turn runs; the apology; a note for what it can't show", () => {
    expect(bubbleText(turn({ progress: "🌐 Looking" })).progress).toBe("🌐 Looking");
    expect(bubbleText(turn({ progress: "🌐 Looking", ended: true })).progress).toBeNull();
    expect(bubbleText(turn({ apology: "Forgive me, sir." })).apology).toBe("Forgive me, sir.");
    expect(bubbleText(turn()).note).toBeNull();
    expect(bubbleText(turn({ interrupted: true })).note).toBe("Interrupted");
    expect(bubbleText(turn({ pictures: 1 })).note).toBe("1 picture came with this; the bubble can't show it yet");
    expect(bubbleText(turn({ pictures: 3, interrupted: true })).note).toBe(
      "Interrupted · 3 pictures came with this; the bubble can't show them yet",
    );
  });
});

describe("the bubble in the page", () => {
  it("is hidden until its envelope says otherwise, and takes no input", () => {
    const { root } = mount();
    expect(root.parentElement).toBe(document.body);
    expect(root.style.display).toBe("none");
    expect(root.querySelector("input, button, [tabindex], a")).toBeNull();
  });

  it("writes the turn as text, never markup", () => {
    const { b, root, q } = mount();
    b.paint(turn({ reply: '<img src=x onerror="alert(1)"> <b>hi</b>', progress: "<i>p</i>" }), SHOWN, HEAD, VIEW);
    expect(root.style.display).toBe("flex");
    expect(q(".bubble-text").textContent).toBe('<img src=x onerror="alert(1)"> <b>hi</b>');
    expect(root.querySelector("img, b, i")).toBeNull();
    expect(q(".bubble-progress").style.display).toBe("flex");
    expect(q(".bubble-progress-text").textContent).toBe("<i>p</i>");
    expect(q(".bubble-apology").style.display).toBe("none");
    expect(q(".bubble-note").style.display).toBe("none");
  });

  it("shows the apology and the note when there are some, and hides an empty reply", () => {
    const { b, q } = mount();
    b.paint(turn({ ended: true, apology: "I lost my connection, sir.", interrupted: true }), SHOWN, HEAD, VIEW);
    expect(q(".bubble-text").style.display).toBe("none");
    // All of it in the one clipped column, inside the panel.
    expect(q(".bubble-clip .bubble-content .bubble-apology")).not.toBeNull();
    expect(q(".bubble-clip .bubble-content .bubble-note")).not.toBeNull();
    expect(q(".bubble-apology").style.display).toBe("flex");
    expect(q(".bubble-apology-text").textContent).toBe("I lost my connection, sir.");
    expect(q(".bubble-note").style.display).toBe("block");
    expect(q(".bubble-note").textContent).toBe("Interrupted");
  });

  it("sits over his head and draws its envelope", () => {
    const { b, root } = mount();
    b.paint(turn({ reply: "Hi" }), { visible: true, opacity: 0.5, scale: 0.8, blurPx: 3.2 }, HEAD, VIEW);
    expect(root.style.left).toBe("70px");
    expect(root.style.bottom).toBe(`${640 - 186}px`);
    expect(root.style.top).toBe("auto");
    expect(root.style.maxWidth).toBe("300px");
    expect(root.style.maxHeight).toBe("178px");
    expect(root.style.getPropertyValue("--tail-x")).toBe("100px");
    expect(root.style.opacity).toBe("0.5");
    expect(root.style.transform).toBe("scale(0.8)");
    expect(root.style.filter).toBe("blur(3.2px)");
    expect(root.classList.contains("no-tail")).toBe(false);
    b.paint(turn({ reply: "Hi" }), SHOWN, HEAD, VIEW);
    expect(root.style.transform).toBe("none");
    expect(root.style.filter).toBe("none");
  });

  it("over him at the top of the window, with no tail, when there's no room over his head", () => {
    const { b, root } = mount();
    b.paint(turn(), SHOWN, { x: 170, y: 40 }, VIEW);
    expect(root.style.top).toBe("8px");
    expect(root.style.bottom).toBe("auto");
    expect(root.classList.contains("no-tail")).toBe(true);
  });

  it("writes nothing when nothing changed, and measures only a new revision", () => {
    const { b, root } = mount();
    const v = turn({ reply: "Good evening" });
    b.paint(v, SHOWN, HEAD, VIEW);
    expect(widthReads).toBe(1);
    const writes = vi.spyOn(root.style, "setProperty");
    for (let i = 0; i < 30; i++) b.paint(v, SHOWN, HEAD, VIEW);
    expect(writes).not.toHaveBeenCalled();
    expect(widthReads).toBe(1);
    // A new revision is written and measured once.
    const text = root.querySelector(".bubble-text")!;
    b.paint({ ...v, reply: "Good evening, sir.", revision: 2 }, SHOWN, HEAD, VIEW);
    expect(text.textContent).toBe("Good evening, sir.");
    expect(widthReads).toBe(2);
    // A reply changed without a revision isn't looked at.
    b.paint({ ...v, reply: "something else", revision: 2 }, SHOWN, HEAD, VIEW);
    expect(text.textContent).toBe("Good evening, sir.");
    // He moved: only where it sits is written.
    writes.mockClear();
    b.paint({ ...v, revision: 2 }, SHOWN, { x: 171, y: 201 }, VIEW);
    expect(writes.mock.calls.map((c) => c[0]).sort()).toEqual(["bottom", "left", "max-height"]);
  });

  it("is measured again, and its clipping looked at again, when the window's width changes", () => {
    const { b, q } = mount();
    let contentHeight = 40;
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("bubble-content") ? contentHeight : 0;
    });
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("bubble-clip") ? 100 : 0;
    });
    b.paint(turn(), SHOWN, HEAD, VIEW);
    expect(q(".bubble-clip").classList.contains("clipped")).toBe(false);
    // Narrower, the same text wraps to more lines than fit.
    contentHeight = 160;
    b.paint(turn(), SHOWN, HEAD, { width: 200, height: 640 });
    expect(widthReads).toBe(2);
    expect(q(".bubble-clip").classList.contains("clipped")).toBe(true);
    // Nothing changed: not looked at again.
    contentHeight = 40;
    b.paint(turn(), SHOWN, HEAD, { width: 200, height: 640 });
    expect(q(".bubble-clip").classList.contains("clipped")).toBe(true);
  });

  it("goes when its envelope does, and comes back as it was", () => {
    const { b, root, q } = mount();
    b.paint(turn({ reply: "Hi" }), SHOWN, HEAD, VIEW);
    b.paint(turn({ reply: "Hi" }), HIDDEN, HEAD, VIEW);
    expect(root.style.display).toBe("none");
    b.paint(turn({ reply: "Hi" }), SHOWN, HEAD, VIEW);
    expect(root.style.display).toBe("flex");
    expect(q(".bubble-text").textContent).toBe("Hi");
  });
});

describe("the muted sign", () => {
  it("shows on its own over his head while no bubble shows, and in the bubble's corner while one does", () => {
    const { b, root, q } = mount();
    const corner = q(".bubble-muted");
    b.paint(turn(), HIDDEN, HEAD, VIEW, true);
    expect(root.style.display).toBe("none");
    expect(b.sign.style.display).toBe("flex");
    expect(b.sign.parentElement).toBe(document.body);
    expect(signAt(HEAD, VIEW)).toEqual({ left: HEAD.x - SIGN_SIZE / 2, top: HEAD.y - GAP - SIGN_SIZE });
    expect([b.sign.style.left, b.sign.style.top]).toEqual([`${HEAD.x - SIGN_SIZE / 2}px`, `${HEAD.y - GAP - SIGN_SIZE}px`]);
    b.paint(turn(), SHOWN, HEAD, VIEW, true);
    expect(b.sign.style.display).toBe("none");
    expect(corner.style.display).toBe("flex");
    expect(corner.parentElement).toBe(root);
    b.paint(turn(), SHOWN, HEAD, VIEW, false);
    expect(corner.style.display).toBe("none");
    b.paint(turn(), HIDDEN, HEAD, VIEW, false);
    expect(b.sign.style.display).toBe("none");
    // Unmuted is the default: a page that never says so shows none.
    b.paint(turn(), HIDDEN, HEAD, VIEW);
    expect(b.sign.style.display).toBe("none");
  });

  it("follows his head on its own, kept inside the window as the bubble is", () => {
    const { b } = mount();
    b.paint(turn(), HIDDEN, HEAD, VIEW, true);
    b.paint(turn(), HIDDEN, { x: 100, y: 300 }, VIEW, true);
    expect([b.sign.style.left, b.sign.style.top]).toEqual([`${100 - SIGN_SIZE / 2}px`, `${300 - GAP - SIGN_SIZE}px`]);
    expect(signAt({ x: 0, y: 0 }, VIEW)).toEqual({ left: EDGE, top: EDGE });
    expect(signAt({ x: 9999, y: 9999 }, VIEW)).toEqual({ left: VIEW.width - SIGN_SIZE - EDGE, top: VIEW.height - SIGN_SIZE - EDGE });
    expect(signAt({ x: Number.NaN, y: Number.NaN }, VIEW)).toEqual({ left: VIEW.width / 2 - SIGN_SIZE / 2, top: EDGE });
  });

  it("is a picture only: no text, nothing a reader is told, no input", () => {
    const { b, q } = mount();
    for (const sign of [b.sign, q(".bubble-muted")]) {
      expect(sign.textContent).toBe("");
      expect(sign.querySelector("svg")?.getAttribute("aria-hidden")).toBe("true");
      expect(sign.querySelector("button, a, input, [tabindex]")).toBeNull();
    }
  });
});

describe("the meter while he listens", () => {
  const listening = (revision = 1) => turn({ turnId: "ptt-1", phase: "listening", revision });
  const level = (q: (s: string) => HTMLElement) => q(".hmeter-level").style.width;

  it("shows under Listening…, its level and cap moved at every paint, the bubble not measured again", () => {
    const { b, q } = mount();
    b.paint(listening(), SHOWN, HEAD, VIEW, false, { level: 0.5, peak: 0.75 });
    expect(q(".bubble-text").textContent).toBe(LISTENING);
    expect(q(".bubble-meter").style.display).toBe("block");
    expect(q(".bubble-meter").classList.contains("hmeter")).toBe(true);
    expect(level(q)).toBe("50%");
    expect(q(".hmeter-peak").style.display).toBe("block");
    expect(q(".hmeter-peak").style.left).toBe("calc(75% - 1px)");
    const reads = widthReads;
    b.paint(listening(), SHOWN, HEAD, VIEW, false, { level: 0.123, peak: 0.75 });
    expect(level(q)).toBe("12.3%");
    expect(widthReads, "the meter's moves aren't measured").toBe(reads);
  });

  it("is empty with no level yet, and keeps its cap hidden at 2% or less, as DivoraVoice's", () => {
    const { b, q } = mount();
    b.paint(listening(), SHOWN, HEAD, VIEW, false, null);
    expect(level(q)).toBe("0%");
    expect(q(".hmeter-peak").style.display).toBe("none");
    b.paint(listening(), SHOWN, HEAD, VIEW, false, { level: 0.01, peak: 0.02 });
    expect(q(".hmeter-peak").style.display).toBe("none");
    b.paint(listening(), SHOWN, HEAD, VIEW, false, { level: Number.NaN, peak: 7 });
    expect(level(q)).toBe("0%");
    expect(q(".hmeter-peak").style.left).toBe("calc(100% - 1px)");
  });

  it("goes with the listening: hidden for any other turn, and once it ends", () => {
    const { b, q } = mount();
    b.paint(turn({ reply: "Hi" }), SHOWN, HEAD, VIEW, false, { level: 0.5, peak: 0.5 });
    expect(q(".bubble-meter").style.display).toBe("none");
    b.paint(listening(2), SHOWN, HEAD, VIEW, false, { level: 0.5, peak: 0.5 });
    expect(q(".bubble-meter").style.display).toBe("block");
    b.paint({ ...listening(3), phase: "transcribing" }, SHOWN, HEAD, VIEW, false, { level: 0.5, peak: 0.5 });
    expect(q(".bubble-meter").style.display).toBe("none");
    b.paint({ ...listening(4), ended: true }, SHOWN, HEAD, VIEW, false, { level: 0.9, peak: 0.9 });
    expect(q(".bubble-meter").style.display).toBe("none");
    expect(level(q), "nothing moved once it ended").toBe("50%");
  });
});
