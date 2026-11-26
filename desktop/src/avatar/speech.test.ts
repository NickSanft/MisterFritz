/**
 * His voice on his page: the brain's sentences spoken one at a time and in
 * order, played back to back, stopped at once by a barge-in, and the audio
 * context made only when he speaks and suspended when he is quiet. A fake
 * context and a fake shell stand in for Web Audio and Kokoro.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeClock } from "../shared/testClock";
import type { Speak } from "../shared/protocol";
import type { Spoken } from "./voiceLink";
import { LEAD_S, STALL_MARGIN_MS, SUSPEND_AFTER_MS, Speech, type BufferLike, type OutLike, type SourceLike } from "./speech";

class FakeSource implements SourceLike {
  buffer: unknown = null;
  onended: ((ev: Event) => unknown) | null = null;
  connected: unknown = null;
  at: number | null = null;
  stopped = false;
  connect(to: unknown) {
    this.connected = to;
    return to;
  }
  start(when = 0) {
    this.at = when;
  }
  stop() {
    if (this.at === null) throw new Error("not started");
    this.stopped = true;
  }
  /** It has played to its end. */
  end() {
    this.onended?.(new Event("ended"));
  }
}

class FakeOut implements OutLike {
  currentTime = 0;
  state = "running";
  destination = { the: "speakers" };
  sources: FakeSource[] = [];
  resumes = 0;
  suspends = 0;
  async resume() {
    this.resumes += 1;
    this.state = "running";
  }
  async suspend() {
    this.suspends += 1;
    this.state = "suspended";
  }
  createBuffer(channels: number, length: number, rate: number): BufferLike {
    expect([channels, rate]).toEqual([1, 24_000]);
    let copied = 0;
    return {
      duration: length / rate,
      copyToChannel: (s: Float32Array, ch: number) => {
        expect(ch).toBe(0);
        copied += s.length;
      },
    };
  }
  createBufferSource() {
    const s = new FakeSource();
    this.sources.push(s);
    return s;
  }
}

/** The shell, answering each sentence when the test says. */
class FakeShell {
  asked: string[] = [];
  private answers: ((s: Spoken) => void)[] = [];
  speak = (text: string) => {
    this.asked.push(text);
    return new Promise<Spoken>((r) => this.answers.push(r));
  };
  /** The next sentence back, `seconds` long (or why not). */
  async answer(seconds: number | string) {
    const r = this.answers.shift();
    if (!r) throw new Error("nothing asked");
    r(typeof seconds === "string" ? { why: seconds } : { samples: new Float32Array(Math.round(seconds * 24_000)) });
    await flush();
  }
  get waiting(): number {
    return this.answers.length;
  }
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const speak = (turn_id: string, seq: number, text: string, final = false): Speak => ({ type: "speak", turn_id, seq, text, final });

describe("his voice on his page", () => {
  let clock: FakeClock;
  let out: FakeOut;
  let shell: FakeShell;
  let made: number;
  let speech: Speech;
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    clock = new FakeClock();
    out = new FakeOut();
    shell = new FakeShell();
    made = 0;
    speech = new Speech({ audio: () => (made++, out), speak: shell.speak, clock });
    warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("speaks the sentences one at a time, in order, back to back", async () => {
    speech.hear(speak("t1", 0, "Good evening, sir."));
    speech.hear(speak("t1", 1, "It is half past nine."));
    expect(shell.asked).toEqual(["Good evening, sir."]);
    expect(speech.busy).toBe(true);
    out.currentTime = 1;
    await shell.answer(2);
    expect(shell.asked).toEqual(["Good evening, sir.", "It is half past nine."]);
    const [first] = out.sources;
    expect(first.at).toBeCloseTo(1 + LEAD_S);
    expect(first.connected).toBe(out.destination);
    out.currentTime = 1.5;
    await shell.answer(3);
    // After the first, not now.
    expect(out.sources[1].at).toBeCloseTo(1 + LEAD_S + 2);
    expect(speech.counts.started).toBe(2);
    expect(speech.counts.spoken, "spoken once played to the end").toBe(0);
    first.end();
    expect(speech.busy).toBe(true);
    out.sources[1].end();
    expect(speech.counts.spoken).toBe(2);
    expect(speech.busy).toBe(false);
  });

  it("is busy from the moment a sentence is asked of the shell", async () => {
    speech.hear(speak("t1", 0, "Only this."));
    expect(speech.busy, "asked, nothing waiting or playing yet").toBe(true);
    await shell.answer(1);
    expect(speech.busy).toBe(true);
    out.sources[0].end();
    expect(speech.busy).toBe(false);
  });

  it("starts at once when what played before has ended", async () => {
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    out.sources[0].end();
    out.currentTime = 10;
    speech.hear(speak("t1", 1, "Two."));
    await shell.answer(1);
    expect(out.sources[1].at).toBeCloseTo(10 + LEAD_S);
  });

  it("makes the audio context only when he first has something to play", async () => {
    speech.hear(speak("t1", 0, "", true));
    expect(made).toBe(0);
    speech.hear(speak("t2", 0, "Hello."));
    expect(made).toBe(0);
    expect(speech.audioState).toBe("none");
    await shell.answer(1);
    expect(made).toBe(1);
    speech.hear(speak("t2", 1, "Again."));
    await shell.answer(1);
    expect(made).toBe(1);
    expect(speech.audioState).toBe("running");
  });

  it("stops at once: what plays, what waits, and what the shell is still speaking", async () => {
    speech.hear(speak("t1", 0, "One."));
    speech.hear(speak("t1", 1, "Two."));
    speech.hear(speak("t1", 2, "Three."));
    await shell.answer(1);
    expect(shell.asked).toEqual(["One.", "Two."]);
    speech.stop();
    expect(out.sources[0].stopped).toBe(true);
    expect(speech.busy).toBe(false);
    expect(speech.counts.stopped).toBe(1);
    // "Two." comes back after the stop: it isn't played, and "Three." isn't asked.
    await shell.answer(1);
    expect(out.sources).toHaveLength(1);
    expect(shell.asked).toEqual(["One.", "Two."]);
    // The stopped turn says nothing more.
    speech.hear(speak("t1", 3, "Four."));
    expect(shell.asked).toEqual(["One.", "Two."]);
    expect(speech.busy).toBe(false);
  });

  it("after a stop, the next turn's first sentence starts at once", async () => {
    speech.hear(speak("t1", 0, "A long one."));
    await shell.answer(30);
    speech.stop();
    out.currentTime = 2;
    speech.hear(speak("t2", 0, "Yes, sir?"));
    await shell.answer(1);
    expect(out.sources[1].at, "not after the 30 s that were stopped").toBeCloseTo(2 + LEAD_S);
  });

  it("a sentence that never plays drops the turn a little after its end, and says so once", async () => {
    // The context's clock stands still: nothing is playing.
    speech.hear(speak("t1", 0, "One."));
    speech.hear(speak("t1", 1, "Two."));
    speech.hear(speak("t1", 2, "Three."));
    await shell.answer(1);
    await shell.answer(1);
    expect(out.sources).toHaveLength(2);
    // The second waits on the first, so its watch runs to the first's end too.
    clock.advance(1_000 + LEAD_S * 1000 + STALL_MARGIN_MS - 1);
    expect(speech.busy).toBe(true);
    clock.advance(1);
    expect(speech.busy).toBe(false);
    expect(speech.counts).toEqual({ started: 2, spoken: 0, failed: 0, stopped: 0, stuck: 1 });
    expect(out.sources.every((s) => s.stopped)).toBe(true);
    // "Three." was being asked: dropped when it comes back, and the turn says no more.
    await shell.answer(1);
    expect(out.sources).toHaveLength(2);
    speech.hear(speak("t1", 3, "Four."));
    expect(shell.waiting).toBe(0);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("isn't playing");
    expect(String(warn.mock.calls[0][0])).not.toMatch(/One|Two|Three/);
    // A sentence that ends in time is no stall, and stops its watch.
    speech.hear(speak("t2", 0, "Fine."));
    await shell.answer(1);
    out.sources[2].end();
    clock.advance(60_000);
    expect(speech.counts.stuck).toBe(1);
    expect(speech.counts.spoken).toBe(1);
  });

  it("a stopped sentence isn't spoken, and its watch is stopped with it", async () => {
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    speech.stop();
    out.sources[0].end();
    clock.advance(60_000);
    expect(speech.counts).toMatchObject({ started: 1, spoken: 0, stopped: 1, stuck: 0 });
  });

  it("a stop with nothing to stop is no stop", () => {
    speech.stop();
    expect(speech.counts.stopped).toBe(0);
  });

  it("a sentence not yet started is stopped too", async () => {
    out.currentTime = 0;
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(5);
    speech.hear(speak("t1", 1, "Two."));
    await shell.answer(1);
    expect(out.sources[1].at).toBeCloseTo(LEAD_S + 5, 5);
    speech.stop();
    expect(out.sources.every((s) => s.stopped)).toBe(true);
    // Their ends, if they still come, change nothing.
    out.sources[0].end();
    expect(speech.busy).toBe(false);
  });

  it("a frame for another turn begins it, and the last's is dropped", async () => {
    speech.hear(speak("t1", 0, "One."));
    speech.hear(speak("t1", 1, "Two."));
    await shell.answer(1);
    speech.hear(speak("t2", 0, "Hello."));
    expect(out.sources[0].stopped).toBe(true);
    // t1's "Two." was being spoken: it comes back dropped, then t2's is asked.
    expect(shell.asked).toEqual(["One.", "Two.", "Hello."]);
    await shell.answer(1);
    await shell.answer(1);
    expect(out.sources).toHaveLength(2);
    expect(speech.counts.started).toBe(2);
  });

  it("takes each sentence once, and an empty final says nothing", async () => {
    speech.hear(speak("t1", 0, "One."));
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    speech.hear(speak("t1", 1, "  ", true));
    speech.hear(speak("t1", 1, "One."));
    expect(shell.asked).toEqual(["One."]);
    expect(shell.waiting).toBe(0);
    speech.hear(speak("t1", 1.5 as number, "Odd."));
    expect(shell.asked).toEqual(["One."]);
  });

  it("goes on past a sentence the shell couldn't speak, and says so once a turn", async () => {
    speech.hear(speak("t1", 0, "One."));
    speech.hear(speak("t1", 1, "Two."));
    speech.hear(speak("t1", 2, "Three."));
    await shell.answer("his voice isn't installed");
    await shell.answer("his voice isn't installed");
    await shell.answer(1);
    expect(out.sources).toHaveLength(1);
    expect(speech.counts).toEqual({ started: 1, spoken: 0, failed: 2, stopped: 0, stuck: 0 });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("his voice isn't installed");
    expect(String(warn.mock.calls[0][0])).not.toContain("One.");
    speech.hear(speak("t2", 0, "Four."));
    await shell.answer("again");
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it("goes on past a shell that throws", async () => {
    const throwing = new Speech({
      audio: () => out,
      speak: vi.fn(async (t: string) => {
        if (t === "One.") throw new Error("boom");
        return { samples: new Float32Array(24) } as Spoken;
      }),
      clock,
    });
    throwing.hear(speak("t1", 0, "One."));
    throwing.hear(speak("t1", 1, "Two."));
    await flush();
    await flush();
    expect(throwing.counts).toEqual({ started: 1, spoken: 0, failed: 1, stopped: 0, stuck: 0 });
  });

  it("without an audio context, it fails each sentence once and never retries the context", async () => {
    let tries = 0;
    const broken = new Speech({
      audio: () => {
        tries += 1;
        throw new Error("no audio device");
      },
      speak: shell.speak,
      clock,
    });
    broken.hear(speak("t1", 0, "One."));
    broken.hear(speak("t1", 1, "Two."));
    await shell.answer(1);
    await shell.answer(1);
    expect(tries).toBe(1);
    expect(broken.counts.failed).toBe(2);
    expect(broken.audioState).toBe("broken");
    expect(broken.busy).toBe(false);
  });

  it("suspends the context once he has been quiet a while, and wakes it to speak", async () => {
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    clock.advance(2_000);
    expect(out.suspends, "not while he speaks").toBe(0);
    out.sources[0].end();
    clock.advance(SUSPEND_AFTER_MS - 1);
    expect(out.suspends).toBe(0);
    clock.advance(1);
    await flush();
    expect(out.suspends).toBe(1);
    expect(out.state).toBe("suspended");
    // Speaking again wakes it before the sentence is even back.
    speech.hear(speak("t1", 1, "Two."));
    await flush();
    expect(out.resumes).toBe(1);
    expect(out.state).toBe("running");
    await shell.answer(1);
    expect(out.sources[1].at).not.toBeNull();
  });

  it("isn't suspended while he speaks again before the wait is out", async () => {
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    out.sources[0].end();
    expect(clock.pending, "the wait begins").toBe(1);
    clock.advance(SUSPEND_AFTER_MS / 2);
    speech.hear(speak("t1", 1, "Two."));
    expect(clock.pending, "and is called off as he speaks again").toBe(0);
    clock.advance(SUSPEND_AFTER_MS);
    expect(out.suspends).toBe(0);
    await shell.answer(1);
    clock.advance(2_000);
    expect(out.suspends).toBe(0);
    out.sources[1].end();
    clock.advance(SUSPEND_AFTER_MS);
    await flush();
    expect(out.suspends).toBe(1);
  });

  it("a context made suspended is resumed to play", async () => {
    out.state = "suspended";
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    await flush();
    expect(out.resumes).toBe(1);
    expect(out.sources[0].at).not.toBeNull();
  });

  it("a stop suspends it in time too", async () => {
    speech.hear(speak("t1", 0, "One."));
    await shell.answer(1);
    speech.stop();
    clock.advance(SUSPEND_AFTER_MS);
    await flush();
    expect(out.suspends).toBe(1);
  });
});

/** The page's own wiring, read from its source (main.ts runs only in his
 * window): his sentences for the shown turn only; stopped by push-to-talk,
 * a typed question and the link lost; the bubble held while he speaks. */
describe("the page's use of his voice", () => {
  const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8").replace(/\r\n/g, "\n");
  const body = (start: string, end = "\n}\n") => {
    const at = main.indexOf(start);
    expect(at, start).toBeGreaterThanOrEqual(0);
    return main.slice(at, main.indexOf(end, at));
  };

  it("speaks the shown turn's sentences only", () => {
    expect(main).toContain('companion.on("speak", (frame) => {\n  if (frame.turn_id === turn.turnId) speech.hear(frame);\n});');
    expect(main.split("speech.hear(").length - 1).toBe(1);
  });

  it("stops him for push-to-talk, a typed question and a lost link, first", () => {
    expect(body("function ask(text: string): boolean {").split("\n")[1].trim()).toBe("speech.stop();");
    const listening = body("void listen<unknown>(PTT_LISTENING_EVENT, (e) => {", "\n});");
    expect(listening).toContain("if (press === null) return;\n  speech.stop();\n  const old = cancelFirst(turn);");
    expect(body("companion.onLink((s) => {", "\n});")).toContain('if (wasOnline && s.link !== "online") {\n    speech.stop();');
    expect(main.split("speech.stop()").length - 1).toBe(3);
  });

  it("holds the bubble while he speaks", () => {
    expect(body("function paintBubble(dt: number): void {")).toContain("bubbleClock.tick(dt * 1000, hovered || speech.busy, motionFactor)");
  });

  it("shows what the brain heard", () => {
    expect(main).toContain('for (const type of ["state", "token", "progress", "done", "error", "stt"] as const) {');
  });
});
