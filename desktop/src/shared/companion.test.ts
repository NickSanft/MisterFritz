/**
 * The companion client against a fake socket and a fake clock: the hello,
 * the welcome, what is refused and what is tried again (and when), the
 * heartbeat, turns, and frames it must ignore.
 */
import { describe, expect, it } from "vitest";
import fixtures from "./protocol.fixtures.json";
import { FakeClock } from "./testClock";
import { CLOSE, SAY_MAX_CHARS, TURN_ID_PATTERN } from "./protocol";
import {
  AUDIO_MAX_BYTES,
  BACKOFF_S,
  CompanionClient,
  PING_EVERY_MS,
  SILENCE_MS,
  SPEAKS,
  WELCOME_TIMEOUT_MS,
  backoffDelay,
  closeReason,
  sayable,
  type CompanionConfig,
  type LinkStatus,
  type SocketLike,
} from "./companion";

class FakeSocket implements SocketLike {
  sent: Record<string, unknown>[] = [];
  closed: number | undefined | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(readonly url: string) {}
  /** Binary frames are recorded as their size, and the bytes kept. */
  binary: ArrayBuffer[] = [];
  send(data: string | ArrayBuffer): void {
    if (typeof data === "string") {
      this.sent.push(JSON.parse(data) as Record<string, unknown>);
    } else {
      this.binary.push(data);
      this.sent.push({ binary: data.byteLength });
    }
  }
  close(code?: number): void {
    this.closed = code;
  }
  // What the brain (or the network) does:
  open(): void {
    this.onopen?.();
  }
  frame(f: unknown): void {
    this.onmessage?.({ data: typeof f === "string" ? f : JSON.stringify(f) });
  }
  raw(data: unknown): void {
    this.onmessage?.({ data });
  }
  shut(code: number): void {
    this.onerror?.();
    this.onclose?.({ code });
  }
}

const CONFIG: CompanionConfig = {
  url: "ws://127.0.0.1:8001/companion",
  token: "t0k-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  client: { id: "nick-pc", name: "Nick's PC" },
};
const WALL = 1_763_112_000_000;

function sample<T extends string>(type: T): Record<string, unknown> & { type: T } {
  const found = fixtures.server.valid.find((e) => (e.frame as { type: string }).type === type);
  if (!found) throw new Error(`no ${type} in the fixtures`);
  return { ...(found.frame as Record<string, unknown>), type };
}
const WELCOME = sample("welcome");

function harness(opts: { throwOn?: number } = {}) {
  const clock = new FakeClock();
  const sockets: FakeSocket[] = [];
  const opened: number[] = [];
  let ids = 0;
  const client = new CompanionClient({
    socket: (url) => {
      opened.push(clock.t);
      if (opts.throwOn === opened.length) throw new SyntaxError("bad url");
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    },
    clock,
    uuid: () => `7f3c2a10-0000-4000-8000-${String(++ids).padStart(12, "0")}`,
    wallNow: () => WALL + clock.t + 0.75,
  });
  const links: LinkStatus[] = [];
  client.onLink((s) => links.push(s));
  const last = () => sockets[sockets.length - 1]!;
  /** Configure, open, welcome: online. */
  const online = () => {
    client.configure(CONFIG);
    last().open();
    last().frame(WELCOME);
    expect(client.status.link).toBe("online");
    return last();
  };
  return { clock, sockets, opened, client, links, last, online };
}

describe("connecting", () => {
  it("sends the hello first, exactly", () => {
    const h = harness();
    expect(h.client.configure(CONFIG)).toBe(true);
    expect(h.sockets).toHaveLength(1);
    expect(h.last().url).toBe(CONFIG.url);
    expect(h.client.status.link).toBe("connecting");
    expect(h.last().sent).toEqual([]);
    h.last().open();
    expect(h.last().sent).toEqual([
      { type: "hello", protocol: 1, token: CONFIG.token, client: { id: "nick-pc", name: "Nick's PC", caps: ["text", "audio-in", "vrm"] } },
    ]);
  });

  it("offers to voice what he says only with his voice installed", () => {
    const h = harness();
    h.client.configure({ ...CONFIG, speaks: true });
    h.last().open();
    expect(h.last().sent[0]).toMatchObject({ client: { caps: ["text", "audio-in", "vrm", "speak-local"] } });
    expect(SPEAKS).toBe("speak-local");
    // A change to it is a new config: connected anew.
    expect(h.client.configure({ ...CONFIG, speaks: false })).toBe(true);
    h.last().open();
    expect(h.last().sent[0]).toMatchObject({ client: { caps: ["text", "audio-in", "vrm"] } });
    expect(h.client.configure({ ...CONFIG, speaks: false })).toBe(false);
    expect(h.client.configure(CONFIG), "absent is false").toBe(false);
  });

  it("is online only once welcomed, and says who it is to the brain", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.last().open();
    expect(h.client.status.link).toBe("connecting");
    h.last().frame(WELCOME);
    expect(h.client.status).toEqual({
      link: "online",
      reason: null,
      retryIn: null,
      identity: WELCOME.identity,
      displayName: WELCOME.display_name,
      threadId: WELCOME.thread_id,
      serverCaps: WELCOME.server_caps,
    });
    expect(h.links.map((s) => s.link)).toEqual(["connecting", "online"]);
  });

  it("ignores everything before the welcome", () => {
    const h = harness();
    const heard: unknown[] = [];
    h.client.on("token", (f) => heard.push(f));
    h.client.on("ping", (f) => heard.push(f));
    h.client.configure(CONFIG);
    h.last().open();
    h.last().frame(sample("token"));
    h.last().frame({ type: "ping", t: 5 });
    expect(heard).toEqual([]);
    expect(h.last().sent.map((f) => f.type), "no pong before the welcome").toEqual(["hello"]);
    expect(h.client.status.link).toBe("connecting");
    expect(h.client.say("hello")).toBeNull();
  });

  it("refuses a welcome in another protocol version, for good", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.last().open();
    h.last().frame({ ...WELCOME, protocol: 2 });
    expect(h.client.status).toMatchObject({ link: "refused", reason: "version" });
    h.clock.advance(10 * 60_000);
    expect(h.sockets).toHaveLength(1);
  });
});

describe("refusals and retries", () => {
  it("never retries a refusal", () => {
    for (const [code, reason] of [[CLOSE.badToken, "token"], [CLOSE.unsupported, "version"], [CLOSE.badFrame, "hello"]] as const) {
      const h = harness();
      h.client.configure(CONFIG);
      h.last().open();
      h.last().shut(code);
      expect(h.client.status, String(code)).toMatchObject({ link: "refused", reason, retryIn: null });
      h.clock.advance(10 * 60_000);
      expect(h.sockets, String(code)).toHaveLength(1);
      expect(closeReason(code)).toEqual({ reason, retry: false });
    }
  });

  it("tries everything else again at 1, 2, 4, 8, 16, then every 30 seconds", () => {
    for (const [code, reason] of [
      [CLOSE.noHello, "quiet"],
      [CLOSE.goneQuiet, "quiet"],
      [CLOSE.full, "full"],
      [1006, "unreachable"],
      [1009, "unreachable"],
    ] as const) {
      const h = harness();
      h.client.configure(CONFIG);
      for (const wait of [1, 2, 4, 8, 16, 30, 30]) {
        h.last().open();
        h.last().shut(code);
        expect(h.client.status, `${code}, then ${wait} s`).toMatchObject({ link: "offline", reason, retryIn: wait });
        h.clock.advance(wait * 1000);
      }
      // Each socket is shut as it opens, so the gaps between them are the waits.
      const gaps = h.opened.slice(1).map((t, i) => (t - h.opened[i]!) / 1000);
      expect(gaps, String(code)).toEqual([1, 2, 4, 8, 16, 30, 30]);
      expect(closeReason(code)).toEqual({ reason, retry: true });
    }
  });

  it("waits exactly the backoff", () => {
    const h = harness();
    h.client.configure(CONFIG);
    for (const wait of [1, 2, 4, 8, 16, 30, 30]) {
      const before = h.sockets.length;
      h.last().shut(1006);
      h.clock.advance(wait * 1000 - 1);
      expect(h.sockets.length, `${wait} s`).toBe(before);
      h.clock.advance(1);
      expect(h.sockets.length, `${wait} s`).toBe(before + 1);
    }
    expect([0, 1, 2, 5, 6, 99].map(backoffDelay)).toEqual([1, 2, 4, 30, 30, 30]);
    expect(BACKOFF_S).toEqual([1, 2, 4, 8, 16, 30]);
  });

  it("starts the count again after a welcome", () => {
    const h = harness();
    h.client.configure(CONFIG);
    for (const wait of [1, 2, 4, 8]) {
      h.last().shut(1006);
      h.clock.advance(wait * 1000);
    }
    expect(h.sockets).toHaveLength(5);
    h.last().open();
    h.last().frame(WELCOME);
    h.last().shut(1006);
    expect(h.client.status).toMatchObject({ link: "offline", reason: "unreachable", retryIn: 1 });
  });

  it("keeps showing what went wrong while it tries again, without flickering", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.last().shut(1006);
    h.clock.advance(1000);
    h.last().shut(1006);
    expect(h.links.map((s) => [s.link, s.reason, s.retryIn])).toEqual([
      ["connecting", null, null],
      ["offline", "unreachable", 1],
      ["offline", "unreachable", null],
      ["offline", "unreachable", 2],
    ]);
  });

  it("drops a socket with no welcome in time: quiet if it opened, unreachable if not", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.clock.advance(WELCOME_TIMEOUT_MS - 1);
    expect(h.last().closed).toBeNull();
    h.clock.advance(1);
    expect(h.last().closed).toBe(1000);
    expect(h.client.status).toMatchObject({ link: "offline", reason: "unreachable", retryIn: 1 });
    h.clock.advance(1000);
    h.last().open();
    h.clock.advance(WELCOME_TIMEOUT_MS);
    expect(h.client.status).toMatchObject({ link: "offline", reason: "quiet", retryIn: 2 });
  });

  it("tries again when the browser won't open the URL", () => {
    const h = harness({ throwOn: 1 });
    h.client.configure(CONFIG);
    expect(h.client.status).toMatchObject({ link: "offline", reason: "unreachable", retryIn: 1 });
    h.clock.advance(1000);
    expect(h.sockets).toHaveLength(1);
  });

  it("ignores a socket it has let go", () => {
    const h = harness();
    const old = h.online();
    h.client.configure({ ...CONFIG, token: "another" });
    expect(old.closed).toBe(1000);
    old.shut(1006);
    old.frame(WELCOME);
    expect(h.client.status.link).toBe("connecting");
    h.clock.advance(5_000);
    expect(h.sockets, "nothing scheduled by the old socket").toHaveLength(2);
  });
});

describe("the heartbeat", () => {
  it("answers the brain's ping with its own t", () => {
    const h = harness();
    const s = h.online();
    s.frame({ type: "ping", t: 1234567 });
    expect(s.sent.at(-1)).toEqual({ type: "pong", t: 1234567 });
  });

  it("pings every 20 seconds, with a whole number", () => {
    const h = harness();
    const s = h.online();
    for (let i = 1; i <= 3; i++) {
      h.clock.advance(PING_EVERY_MS - 1);
      expect(s.sent.filter((f) => f.type === "ping"), `ping ${i}`).toHaveLength(i - 1);
      h.clock.advance(1);
      const ping = s.sent.at(-1)!;
      expect(ping.type).toBe("ping");
      expect(Number.isInteger(ping.t)).toBe(true);
      expect(ping.t).toBe(Math.trunc(WALL + h.clock.t + 0.75));
      s.frame({ type: "pong", t: ping.t }); // the brain answers
    }
  });

  it("drops a brain that has said nothing for two pings and more, and connects again", () => {
    const h = harness();
    const s = h.online();
    expect(SILENCE_MS).toBe(45_000);
    h.clock.advance(2 * PING_EVERY_MS);
    expect(s.closed, "40 s of silence is not yet too long").toBeNull();
    h.clock.advance(PING_EVERY_MS);
    expect(s.closed).toBe(1000);
    expect(h.client.status).toMatchObject({ link: "offline", reason: "quiet", retryIn: 1 });
    h.clock.advance(1000);
    expect(h.sockets).toHaveLength(2);
  });

  it("counts any frame as a sign of life", () => {
    const h = harness();
    const s = h.online();
    h.clock.advance(30_000);
    s.raw("not even JSON");
    h.clock.advance(30_000); // 60 s since the welcome, 30 since that
    expect(s.closed).toBeNull();
    h.clock.advance(20_000); // 50 s of silence
    expect(s.closed).toBe(1000);
  });
});

describe("turns", () => {
  it("say sends the trimmed text under a fresh turn id", () => {
    const h = harness();
    const s = h.online();
    const id = h.client.say("  What's the weather?  ");
    expect(id).not.toBeNull();
    expect(TURN_ID_PATTERN.test(id!)).toBe(true);
    expect(s.sent.at(-1)).toEqual({ type: "say", turn_id: id, text: "What's the weather?" });
    expect(h.client.say("again")).not.toBe(id);
  });

  it("say refuses what the brain would", () => {
    const h = harness();
    const s = h.online();
    const before = s.sent.length;
    for (const text of ["", "   \n\t", "x".repeat(SAY_MAX_CHARS + 1)]) expect(h.client.say(text)).toBeNull();
    expect(s.sent).toHaveLength(before);
    // Characters, as the brain counts them: 8000 emoji are 16000 UTF-16 units.
    expect(sayable("😀".repeat(SAY_MAX_CHARS))).not.toBeNull();
    expect(sayable("😀".repeat(SAY_MAX_CHARS + 1))).toBeNull();
    expect(h.client.say("x".repeat(SAY_MAX_CHARS))).not.toBeNull();
    // Trimmed as the brain trims: Unicode's White_Space, not JS's trim.
    expect(sayable("\u0085")).toBeNull();
    expect(sayable("x".repeat(SAY_MAX_CHARS) + "\u0085")).toBe("x".repeat(SAY_MAX_CHARS));
    expect(sayable("\ufeffhi")).toBe("\ufeffhi");
    expect(sayable("\u3000hi\u00a0")).toBe("hi");
  });

  it("say is null when not online", () => {
    const h = harness();
    expect(h.client.say("hello")).toBeNull();
    h.client.configure(CONFIG);
    h.last().open();
    expect(h.client.say("hello")).toBeNull();
    h.last().frame(WELCOME);
    h.last().shut(1006);
    expect(h.client.say("hello")).toBeNull();
  });

  it("cancel sends its frame, for a well-formed id, while online", () => {
    const h = harness();
    const s = h.online();
    const id = h.client.say("Tell me a long story")!;
    h.client.cancel(id);
    expect(s.sent.at(-1)).toEqual({ type: "cancel", turn_id: id });
    const before = s.sent.length;
    h.client.cancel("not an id!");
    expect(s.sent).toHaveLength(before);
  });

  it("hands each frame to its listeners", () => {
    const h = harness();
    const tokens: string[] = [];
    const dones: string[] = [];
    h.client.on("token", (f) => tokens.push(f.delta));
    h.client.on("done", (f) => dones.push(f.text));
    const s = h.online();
    s.frame({ ...sample("token"), delta: "Good" });
    s.frame({ ...sample("token"), delta: " evening" });
    s.frame({ ...sample("done"), text: "Good evening, sir." });
    expect(tokens).toEqual(["Good", " evening"]);
    expect(dones).toEqual(["Good evening, sir."]);
  });

  it("ignores junk, unknown types and malformed frames", () => {
    const h = harness();
    const heard: unknown[] = [];
    h.client.on("token", (f) => heard.push(f));
    const s = h.online();
    s.raw("{ not json");
    s.raw(new ArrayBuffer(8));
    s.frame({ type: "nonsense" });
    s.frame({ type: "constructor" });
    s.frame({ type: "token", turn_id: "x" }); // no delta
    s.frame([sample("token")]);
    expect(heard).toEqual([]);
    expect(h.client.status.link).toBe("online");
  });

  it("keeps going when a listener throws", () => {
    const h = harness();
    const heard: string[] = [];
    h.client.on("token", () => {
      throw new Error("a broken listener");
    });
    h.client.on("token", (f) => heard.push(f.delta));
    const s = h.online();
    const quiet = console.error;
    console.error = () => {};
    try {
      s.frame(sample("token"));
    } finally {
      console.error = quiet;
    }
    expect(heard).toHaveLength(1);
  });
});

describe("configuring", () => {
  it("null is unconfigured: closed, with nothing tried again", () => {
    const h = harness();
    const s = h.online();
    expect(h.client.configure(null)).toBe(true);
    expect(s.closed).toBe(1000);
    expect(h.client.status).toEqual({
      link: "unconfigured", reason: null, retryIn: null, identity: null, displayName: null, threadId: null, serverCaps: [],
    });
    h.clock.advance(10 * 60_000);
    expect(h.sockets).toHaveLength(1);
    expect(h.client.configure(null)).toBe(false);
  });

  it("the same config changes nothing; a changed one connects now", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.last().shut(1006);
    h.clock.advance(1000);
    h.last().shut(1006); // two failures: the next wait would be 4 s
    expect(h.client.configure({ ...CONFIG, client: { ...CONFIG.client } })).toBe(false);
    expect(h.sockets).toHaveLength(2);
    expect(h.client.configure({ ...CONFIG, url: "ws://10.0.0.5:8001/companion" })).toBe(true);
    expect(h.sockets).toHaveLength(3);
    expect(h.last().url).toBe("ws://10.0.0.5:8001/companion");
    expect(h.client.status.link).toBe("connecting");
    h.last().shut(1006);
    expect(h.client.status.retryIn).toBe(1);
    for (const changed of [{ ...CONFIG, token: "x" }, { ...CONFIG, client: { id: "other", name: "Nick's PC" } },
      { ...CONFIG, client: { id: "nick-pc", name: "Desk" } }]) {
      expect(h.client.configure(changed)).toBe(true);
    }
  });

  it("reconnectNow tries a refused link again at once", () => {
    const h = harness();
    h.client.configure(CONFIG);
    h.last().open();
    h.last().shut(CLOSE.badToken);
    expect(h.client.status.link).toBe("refused");
    h.client.reconnectNow();
    expect(h.sockets).toHaveLength(2);
    expect(h.client.status).toMatchObject({ link: "connecting", reason: null });
    // And an online one, when the owner asks.
    h.last().open();
    h.last().frame(WELCOME);
    const was = h.last();
    h.client.reconnectNow();
    expect(was.closed).toBe(1000);
    expect(h.sockets).toHaveLength(3);
  });

  it("reconnectNow starts the count again", () => {
    const h = harness();
    h.client.configure(CONFIG);
    for (const wait of [1, 2, 4]) {
      h.last().shut(1006);
      h.clock.advance(wait * 1000);
    }
    h.client.reconnectNow();
    expect(h.client.status.link, "the first try of a new count").toBe("connecting");
    h.last().shut(1006);
    expect(h.client.status.retryIn).toBe(1);
  });

  it("reconnectNow does nothing without a config", () => {
    const h = harness();
    h.client.reconnectNow();
    expect(h.sockets).toHaveLength(0);
  });

  it("listeners unsubscribe", () => {
    const h = harness();
    const links: string[] = [];
    const tokens: string[] = [];
    const offLink = h.client.onLink((s) => links.push(s.link));
    const offToken = h.client.on("token", (f) => tokens.push(f.delta));
    const s = h.online();
    s.frame(sample("token"));
    offLink();
    offToken();
    s.frame(sample("token"));
    s.shut(1006);
    expect(links).toEqual(["connecting", "online"]);
    expect(tokens).toHaveLength(1);
  });

  it("after close, nothing connects again", () => {
    const h = harness();
    const s = h.online();
    h.client.close();
    expect(s.closed).toBe(1000);
    h.clock.advance(10 * 60_000);
    expect(h.client.configure({ ...CONFIG, token: "new" })).toBe(false);
    h.client.reconnectNow();
    expect(h.sockets).toHaveLength(1);
    expect(h.clock.pending).toBe(0);
  });
});

describe("sending a clip", () => {
  const HEARING = { ...WELCOME, server_caps: ["text", "audio-in"] };
  /** Online to a brain whose welcome says `caps`. */
  const welcomed = (h: ReturnType<typeof harness>, caps: string[]) => {
    h.client.configure(CONFIG);
    h.last().open();
    h.last().frame({ ...WELCOME, server_caps: caps });
    h.last().sent.length = 0;
    return h.last();
  };

  it("goes to a brain that can hear: one binary frame, then its audio_end", () => {
    const h = harness();
    const s = welcomed(h, HEARING.server_caps);
    expect(h.client.hears()).toBe(true);
    expect(h.client.status.serverCaps).toEqual(["text", "audio-in"]);
    const pcm = new ArrayBuffer(4800);
    const id = h.client.sendAudio(pcm);
    expect(id).not.toBeNull();
    expect(TURN_ID_PATTERN.test(id!)).toBe(true);
    expect(s.sent).toEqual([
      { binary: 4800 },
      { type: "audio_end", turn_id: id, sample_rate: 16000, channels: 1, format: "pcm16" },
    ]);
    expect(s.binary[0]).toBe(pcm);
    // The longest a clip may be, and no more.
    expect(h.client.sendAudio(new ArrayBuffer(AUDIO_MAX_BYTES))).not.toBeNull();
    expect(AUDIO_MAX_BYTES).toBe(30 * 16000 * 2);
  });

  it("never goes to a brain that can't hear, nor anywhere offline", () => {
    const h = harness();
    expect(h.client.sendAudio(new ArrayBuffer(4))).toBeNull();
    const s = welcomed(h, ["text"]);
    expect(h.client.hears()).toBe(false);
    expect(h.client.sendAudio(new ArrayBuffer(4))).toBeNull();
    expect(s.sent, "nothing sent at all").toEqual([]);
    s.shut(1006);
    expect(h.client.hears()).toBe(false);
    expect(h.client.status.serverCaps, "gone with the link").toEqual([]);
  });

  it("a client closed for good hears nothing, whatever its last status said", () => {
    const h = harness();
    welcomed(h, HEARING.server_caps);
    h.client.close();
    expect(h.client.status.serverCaps).toEqual(["text", "audio-in"]);
    expect(h.client.hears()).toBe(false);
    expect(h.client.sendAudio(new ArrayBuffer(4))).toBeNull();
  });

  it("isn't a clip unless it is whole samples, and no longer than 30 s", () => {
    const h = harness();
    const s = welcomed(h, HEARING.server_caps);
    for (const n of [0, 3, AUDIO_MAX_BYTES + 2]) {
      expect(h.client.sendAudio(new ArrayBuffer(n)), String(n)).toBeNull();
    }
    expect(s.sent).toEqual([]);
  });

  it("a socket that won't take the clip sends no audio_end", () => {
    const h = harness();
    const s = welcomed(h, HEARING.server_caps);
    const frames: unknown[] = [];
    s.send = (data) => {
      if (typeof data !== "string") throw new Error("closing");
      frames.push(JSON.parse(data));
    };
    expect(h.client.sendAudio(new ArrayBuffer(4))).toBeNull();
    expect(frames, "no audio_end after a clip that didn't go").toEqual([]);
  });
});
