import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import type { AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { connectionLines, parseCompanionStatus } from "../shared/link";
import { HIDDEN, WINDOW_CLOSED, status } from "./view";

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
