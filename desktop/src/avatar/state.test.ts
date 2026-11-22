/**
 * The turn the bubble shows: the brain's frames in the orders
 * companion_api.py sends them (a turn's tokens, then one done or error, then
 * idle; a barge-in's cancelled turn ending after the new one is asked).
 */
import { describe, expect, it } from "vitest";
import type { ServerFrame } from "../shared/protocol";
import {
  INITIAL,
  INTERRUPTED_MARK,
  LOST_APOLOGY,
  NOTE_PREFIX,
  PRESS_PREFIX,
  apologyText,
  cancelFirst,
  isLocal,
  pressOf,
  reduce,
  type TurnEvent,
  type TurnView,
} from "./state";

const asked = (turnId: string): TurnEvent => ({ kind: "asked", turnId });
const f = (frame: ServerFrame): TurnEvent => ({ kind: "frame", frame });
const state = (turn_id: string, s: "thinking" | "idle" | "speaking" = "thinking") => f({ type: "state", turn_id, state: s });
const token = (turn_id: string, delta: string, restart = false) => f({ type: "token", turn_id, delta, restart });
const progress = (turn_id: string, text: string) => f({ type: "progress", turn_id, text });
const done = (turn_id: string, text: string, more: { images?: string[]; interrupted?: boolean } = {}) =>
  f({ type: "done", turn_id, text, images: more.images ?? [], interrupted: more.interrupted ?? false });
const error = (turn_id: string | null, text: string, code: "busy" | "turn_failed" | "bad_frame" = "turn_failed") =>
  f({ type: "error", turn_id, code, text });

function run(...events: TurnEvent[]): TurnView {
  return events.reduce(reduce, INITIAL);
}

describe("the turn the bubble shows", () => {
  it("starts with nothing, and asking shows the new turn thinking", () => {
    expect(INITIAL.turnId).toBeNull();
    expect(INITIAL.ended).toBe(true);
    const v = run(asked("t1"));
    expect(v).toMatchObject({ turnId: "t1", phase: "thinking", reply: "", progress: null, apology: null, ended: false });
    expect(cancelFirst(INITIAL)).toBeNull();
    expect(cancelFirst(v)).toBe("t1");
  });

  it("streams tokens, and a restart replaces what came before", () => {
    let v = run(asked("t1"), state("t1"), token("t1", "Good "), token("t1", "eve"));
    expect(v.reply).toBe("Good eve");
    v = reduce(v, token("t1", "Let me look.", true));
    expect(v.reply).toBe("Let me look.");
    v = reduce(v, token("t1", " Ah."));
    expect(v.reply).toBe("Let me look. Ah.");
  });

  it("ends on done's text, which supersedes the tokens, and goes idle", () => {
    const v = run(asked("t1"), token("t1", "Good evenin"), progress("t1", "🌐 Looking"), done("t1", "Good evening, sir."), state("t1", "idle"));
    expect(v).toMatchObject({ reply: "Good evening, sir.", progress: null, phase: "idle", ended: true, interrupted: false });
    expect(cancelFirst(v)).toBeNull();
  });

  it("takes the interrupted mark off, and says so", () => {
    let v = run(asked("t1"), token("t1", "Well, sir"), done("t1", `Well, sir${INTERRUPTED_MARK}`, { interrupted: true }));
    expect(v.reply).toBe("Well, sir");
    expect(v.interrupted).toBe(true);
    // Cut off before a word: the mark alone, as the brain trims it.
    v = run(asked("t2"), done("t2", "[interrupted]", { interrupted: true }));
    expect(v.reply).toBe("");
    expect(v.interrupted).toBe(true);
    // A reply that merely ends so, not interrupted, is left as it is.
    v = run(asked("t3"), done("t3", `Quote: ${INTERRUPTED_MARK}`));
    expect(v.reply).toBe(`Quote: ${INTERRUPTED_MARK}`);
  });

  it("counts the pictures and keeps none of them", () => {
    const big = `data:image/png;base64,${"A".repeat(10_000)}`;
    const v = run(asked("t1"), done("t1", "Here.", { images: [big, big] }));
    expect(v.pictures).toBe(2);
    expect(JSON.stringify(v)).not.toContain("base64");
  });

  it("ends a failed turn with his apology, its backticks taken off, the reply so far kept", () => {
    const v = run(asked("t1"), token("t1", "Let me see"), progress("t1", "🔎"),
      error("t1", "I'm afraid something went wrong, sir. (ref `ab12cd34`)"), state("t1", "idle"));
    expect(v).toMatchObject({
      reply: "Let me see", apology: "I'm afraid something went wrong, sir. (ref ab12cd34)",
      progress: null, phase: "idle", ended: true,
    });
    expect(apologyText("`a` and `b`")).toBe("a and b");
  });

  it("ends a turn the brain was too busy for (an error, and no idle after it)", () => {
    const v = run(asked("t1"), error("t1", "One moment, sir.", "busy"));
    expect(v).toMatchObject({ apology: "One moment, sir.", phase: "idle", ended: true });
  });

  it("doesn't show an error for no turn", () => {
    const before = run(asked("t1"), token("t1", "Hm"));
    expect(reduce(before, error(null, "I keep no widgets at this desk as yet, sir.", "bad_frame"))).toBe(before);
  });

  it("ignores every other turn's frames", () => {
    const before = run(asked("t1"), token("t1", "Mine"));
    for (const e of [token("t0", "late"), progress("t0", "x"), done("t0", "old"), error("t0", "old"), state("t0", "idle")]) {
      expect(reduce(before, e)).toBe(before);
    }
    // Frames with no turn at all.
    expect(reduce(before, f({ type: "ping", t: 1 }))).toBe(before);
    expect(reduce(before, f({ type: "welcome", protocol: 1, identity: "d", display_name: "D", server_caps: [], thread_id: "x" }))).toBe(before);
  });

  /** The page cancels the shown turn and asks anew; the brain ends the old
   * one (its last tokens, done and idle) before the new one starts. */
  it("after a barge-in, shows only the new turn", () => {
    let v = run(asked("t1"), state("t1"), token("t1", "A long answer"));
    expect(cancelFirst(v)).toBe("t1");
    v = reduce(v, asked("t2"));
    expect(v).toMatchObject({ turnId: "t2", reply: "", phase: "thinking", ended: false });
    for (const late of [token("t1", " that goes on"), done("t1", "A long answer that goes on [interrupted]", { interrupted: true }), state("t1", "idle")]) {
      v = reduce(v, late);
    }
    expect(v).toMatchObject({ turnId: "t2", reply: "", phase: "thinking", ended: false });
    v = [state("t2"), token("t2", "Yes, sir?"), done("t2", "Yes, sir?"), state("t2", "idle")].reduce(reduce, v);
    expect(v).toMatchObject({ turnId: "t2", reply: "Yes, sir?", phase: "idle", ended: true });
  });

  /** A say still waiting behind a cancelled turn, cancelled itself: the
   * brain ends it empty and interrupted. */
  it("ends a cancelled waiting turn empty and interrupted", () => {
    const v = run(asked("t2"), done("t2", "", { interrupted: true }), state("t2", "idle"));
    expect(v).toMatchObject({ reply: "", interrupted: true, phase: "idle", ended: true });
  });

  it("shows only the latest progress, once, and clears it at the end", () => {
    let v = run(asked("t1"), progress("t1", "🌐 Making enquiries further afield."));
    expect(v.progress).toBe("🌐 Making enquiries further afield.");
    const same = reduce(v, progress("t1", "🌐 Making enquiries further afield."));
    expect(same, "a repeat changes nothing").toBe(v);
    v = reduce(v, progress("t1", "📅 Checking the calendar."));
    expect(v.progress).toBe("📅 Checking the calendar.");
    expect(reduce(v, done("t1", "Done.")).progress).toBeNull();
  });

  it("follows the turn's phase", () => {
    let v = run(asked("t1"));
    expect(v.phase).toBe("thinking");
    v = reduce(v, state("t1", "speaking"));
    expect(v.phase).toBe("speaking");
    v = reduce(v, done("t1", "x"));
    expect(v.phase).toBe("idle");
    expect(reduce(run(asked("t2")), error("t2", "no")).phase).toBe("idle");
  });

  it("apologises for a link lost mid-turn, and only mid-turn", () => {
    const v = run(asked("t1"), token("t1", "Let me"), { kind: "lost" });
    expect(v).toMatchObject({ reply: "Let me", apology: LOST_APOLOGY, phase: "idle", ended: true });
    const over = run(asked("t1"), done("t1", "Done."));
    expect(reduce(over, { kind: "lost" })).toBe(over);
    expect(reduce(INITIAL, { kind: "lost" })).toBe(INITIAL);
    expect(LOST_APOLOGY).toMatch(/sir/);
  });

  it("takes nothing more for a turn once it has ended, but its phase", () => {
    const v = run(asked("t1"), done("t1", "Done."));
    for (const e of [token("t1", "more"), progress("t1", "x"), error("t1", "late"), done("t1", "again")]) {
      expect(reduce(v, e)).toBe(v);
    }
  });

  /** The page writes and measures only on a new revision. */
  it("moves the revision on a change the bubble shows, and only then", () => {
    let v = run(asked("t1"));
    let r = v.revision;
    v = reduce(v, state("t1", "thinking"));
    expect(v.revision, "a phase isn't shown in the bubble").toBe(r);
    v = reduce(v, state("t1", "speaking"));
    expect(v.revision).toBe(r);
    v = reduce(v, token("t1", "Hi"));
    expect(v.revision).toBe(++r);
    v = reduce(v, token("t1", ""));
    expect(v.revision, "an empty token changes nothing").toBe(r);
    v = reduce(v, progress("t1", "x"));
    expect(v.revision).toBe(++r);
    v = reduce(v, done("t1", "Hi"));
    expect(v.revision, "the progress went, and it ended").toBe(++r);
    v = reduce(v, asked("t2"));
    expect(v.revision).toBe(++r);
  });
});

describe("push-to-talk's turns", () => {
  const listening: TurnEvent = { kind: "listening", press: 1 };
  const unheard = (why: string, press?: number): TurnEvent => ({ kind: "unheard", why, press });

  it("listening is a turn of its own, under its press's local id no frame carries", () => {
    const v = run(listening);
    expect(v).toMatchObject({ turnId: `${PRESS_PREFIX}1`, phase: "listening", reply: "", apology: null, ended: false });
    expect(isLocal(v.turnId)).toBe(true);
    expect(pressOf(v.turnId)).toBe(1);
    expect(isLocal("7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d")).toBe(false);
    expect(pressOf("7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d")).toBeNull();
    expect(isLocal(`${NOTE_PREFIX}3`)).toBe(true);
    expect(pressOf(`${NOTE_PREFIX}3`)).toBeNull();
    expect(isLocal(null)).toBe(false);
    // A brain frame for any turn leaves it be.
    expect(reduce(v, token("7f3c2a10-4b5e-4f6a-9c8d-0e1f2a3b4c5d", "hi"))).toBe(v);
    // The next press is a new turn, so the bubble comes up for it.
    const again = reduce(v, { kind: "listening", press: 2 });
    expect(again.turnId).toBe(`${PRESS_PREFIX}2`);
    expect(again.revision).toBeGreaterThan(v.revision);
  });

  it("another press's outcome never ends the press listening now", () => {
    const v = run(listening, { kind: "listening", press: 2 });
    expect(reduce(v, unheard("That was too short for him to hear.", 1))).toBe(v);
    const own = reduce(v, unheard("That was too short for him to hear.", 2));
    expect(own).toMatchObject({ turnId: `${PRESS_PREFIX}2`, apology: "That was too short for him to hear.", ended: true });
  });

  it("a link lost while listening leaves the turn to its clip", () => {
    const v = run(listening);
    expect(reduce(v, { kind: "lost" })).toBe(v);
    const after = run(listening, { kind: "lost" }, unheard("He isn't connected to his brain just now.", 1));
    expect(after).toMatchObject({
      turnId: v.turnId, phase: "idle", apology: "He isn't connected to his brain just now.", ended: true,
    });
  });

  it("the clip sent, the brain's turn takes its place", () => {
    const v = run(listening, asked("t1"), state("t1"), token("t1", "Good evening."));
    expect(v).toMatchObject({ turnId: "t1", phase: "thinking", reply: "Good evening.", ended: false });
  });

  it("a refusal, a newer press's, never ends the press listening: its notice until its own outcome", () => {
    const v = run(listening);
    const refused = reduce(v, unheard("He isn't connected to his brain just now."));
    expect(refused).toMatchObject({ turnId: v.turnId, phase: "listening", progress: "He isn't connected to his brain just now.", ended: false });
    const own = reduce(refused, unheard("He isn't connected to his brain just now.", 1));
    expect(own).toMatchObject({ turnId: v.turnId, apology: "He isn't connected to his brain just now.", progress: null, ended: true });
  });

  it("unheard ends a listening turn with why", () => {
    const v = run(listening, unheard("That was too short for him to hear.", 1));
    expect(v).toMatchObject({ phase: "idle", apology: "That was too short for him to hear.", ended: true });
    expect(isLocal(v.turnId)).toBe(true);
  });

  it("unheard with nothing showing shows why on its own", () => {
    const v = run(unheard("His brain can't hear yet: type to him instead."));
    expect(v).toMatchObject({ phase: "idle", apology: "His brain can't hear yet: type to him instead.", ended: true });
    expect(isLocal(v.turnId)).toBe(true);
    // After a turn that ended, too: a new local turn, so the bubble comes up.
    const after = run(asked("t1"), done("t1", "Quite."), unheard("x"));
    expect(after).toMatchObject({ apology: "x", ended: true, reply: "" });
    expect(after.turnId).not.toBe("t1");
  });

  it("unheard over a reply the brain is still giving is its notice, the reply kept", () => {
    const giving = run(asked("t1"), token("t1", "Well, sir"));
    const v = reduce(giving, unheard("His brain can't hear yet: type to him instead."));
    expect(v).toMatchObject({
      turnId: "t1", reply: "Well, sir", progress: "His brain can't hear yet: type to him instead.", ended: false,
    });
    expect(v.revision).toBeGreaterThan(giving.revision);
    expect(reduce(v, unheard("His brain can't hear yet: type to him instead."))).toBe(v);
    expect(reduce(v, done("t1", "Well, sir."))).toMatchObject({ progress: null, ended: true, reply: "Well, sir." });
  });

  it("listening is a barge-in: the reply running is the one to cancel", () => {
    const giving = run(asked("t1"), token("t1", "Well"));
    expect(cancelFirst(giving)).toBe("t1");
    const v = reduce(giving, listening);
    // A listening turn is the page's own: nothing to cancel at the brain.
    expect(cancelFirst(v)).toBeNull();
    expect(reduce(v, done("t1", "Well, sir."))).toBe(v);
  });
});
