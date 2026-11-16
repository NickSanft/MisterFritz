import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import type { AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { connectionLines, parseCompanionStatus } from "../shared/link";
import { ASKING, HIDDEN, WAITING_LINE, WAITING_PLACEHOLDER, WINDOW_CLOSED, face, lineFor, status, type PromptStatus } from "./view";

const shown: AvatarStatus = {
  open: true,
  phase: "painted",
  instance: 1,
  mode: "locked",
  arrangeGen: 0,
  canArrange: true,
  fit: false,
  parked: false,
  error: null,
};
const online: LinkStatus = { link: "online", reason: null, retryIn: null, identity: "nick-pc", displayName: null, threadId: null };
const samples = Object.entries(fixture.companionStatusSamples).map(([name, s]) => [name, parseCompanionStatus(s)!] as const);

describe("the prompt waiting for a click", () => {
  it("waits only while it hasn't the keyboard", () => {
    expect(face(true, false)).toBe("waiting");
    expect(face(true, true)).toBe("typing");
    expect(face(false, false)).toBe("typing");
    expect(face(false, true)).toBe("typing");
  });

  it("says a refusal first, then the question asked, then the ask for a click, then the status", () => {
    const ok: PromptStatus = { tone: "ok", text: "Connected", rule: true };
    const wait: PromptStatus = { tone: "wait", text: "Connecting…", rule: false };
    const bad: PromptStatus = { tone: "bad", text: WINDOW_CLOSED, rule: false };
    expect(lineFor(ok, "Not connected to his brain.", true, "waiting")).toEqual({ text: "Not connected to his brain.", alert: true, waiting: false });
    expect(lineFor(ok, null, true, "waiting")).toEqual({ text: ASKING, alert: false, waiting: false });
    expect(lineFor(ok, null, false, "waiting")).toEqual({ text: WAITING_LINE, alert: false, waiting: true });
    expect(lineFor(wait, null, false, "waiting").text).toBe(WAITING_LINE);
    // He can't be asked: the status says why, not the ask for a click.
    expect(lineFor(bad, null, false, "waiting")).toEqual({ text: WINDOW_CLOSED, alert: false, waiting: false });
    expect(lineFor(ok, null, false, "typing")).toEqual({ text: "Connected", alert: false, waiting: false });
  });

  it("asks in words", () => {
    expect(WAITING_PLACEHOLDER).toBe("Click here to type…");
    expect(WAITING_LINE).toBe("Click in the box to ask him; anywhere else closes it.");
  });
});

describe("what the prompt says", () => {
  it("says every link in words, the control window's own", () => {
    expect(samples.length).toBeGreaterThan(4);
    for (const [name, s] of samples) {
      const got = status(s, shown);
      expect(got.text, name).not.toBe("");
      const tokenPresent = s.link !== "unconfigured";
      expect(got.text, name).toBe(connectionLines(s, shown, tokenPresent, "").title);
    }
  });

  it("can be asked only while online, the rule under it saying so", () => {
    for (const [name, s] of samples) {
      const got = status(s, shown);
      const tone = s.link === "online" ? "ok" : s.link === "connecting" ? "wait" : "bad";
      expect(got.tone, name).toBe(tone);
      expect(got.rule, name).toBe(s.link === "online");
    }
  });

  it("says when nothing is heard yet, and waits", () => {
    expect(status(null, null)).toEqual({ tone: "wait", text: "Asking how he's connected…", rule: false });
    expect(status(null, shown).tone).toBe("wait");
  });

  it("says he needs his token, with none", () => {
    const none: LinkStatus = { ...online, link: "unconfigured", identity: null };
    expect(status(none, shown)).toEqual({ tone: "bad", text: "He needs his brain's token", rule: false });
  });

  it("says his window is closed in the shell's words, whatever the link", () => {
    for (const phase of ["closed", "closing"] as const) {
      expect(status(online, { ...shown, phase })).toEqual({ tone: "bad", text: WINDOW_CLOSED, rule: false });
    }
    expect(status(online, { ...shown, open: false }).text).toBe(WINDOW_CLOSED);
    expect(status(null, { ...shown, open: false }).text).toBe(WINDOW_CLOSED);
    expect(WINDOW_CLOSED).toBe("His window is closed: show him first.");
  });

  it("says his answer waits while he is hidden", () => {
    expect(status(online, { ...shown, parked: true })).toEqual({ tone: "ok", text: HIDDEN, rule: true });
    expect(status(online, shown)).toEqual({ tone: "ok", text: "Connected", rule: true });
    expect(status(online, null).text).toBe("Connected");
  });
});
