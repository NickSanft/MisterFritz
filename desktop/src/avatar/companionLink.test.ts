/**
 * The avatar page's link to the shell for the brain's connection: what it
 * asks for and when, how it configures the client, and what it reports,
 * by the names and keys the shell has (wire.fixture.json).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import fixture from "./wire.fixture.json";
import type { CompanionClient, CompanionConfig, LinkStatus } from "../shared/companion";
import {
  CONFIG_CMD,
  CONFIG_EVENT,
  REPORT_CMD,
  SAID_CMD,
  SAY_EVENT,
  TAKE_CMD,
  answerSay,
  answerClip,
  BRAIN_CANT_HEAR,
  CLIP_GONE,
  NOT_CONNECTED,
  PTT_CLIP_EVENT,
  PTT_LEVEL_EVENT,
  PTT_LISTENING_EVENT,
  PTT_NOTHING_EVENT,
  PTT_REFUSED_EVENT,
  PTT_SENT_CMD,
  PTT_TAKE_CMD,
  onClip,
  parseClip,
  parseLevel,
  parseListening,
  parseNothing,
  parseWhy,
  pttSentArgs,
  pttTakeArgs,
  takeClip,
  onSay,
  configArgs,
  configFrom,
  oneAtATime,
  parseSay,
  saidArgs,
  takeArgs,
  takeSay,
  parseShellConfig,
  reportArgs,
  startCompanion,
  type LinkDeps,
  type ShellConfig,
} from "./companionLink";

const SHELL: ShellConfig = {
  url: "ws://127.0.0.1:8001/companion",
  token: "t0k-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  clientId: "nick-pc",
  clientName: "NICK-PC",
};
const BRAIN = { url: SHELL.url, clientId: SHELL.clientId, clientName: SHELL.clientName };

function settingsWith(brain: unknown, rest: Record<string, unknown> = {}) {
  return { version: 1, brain, avatar: { lookatCursor: true }, ...rest };
}

/** A client that records what it was told; `changes` says whether a
 * configure changes anything. */
function fakeClient() {
  let link: ((s: LinkStatus) => void) | null = null;
  let current: CompanionConfig | null | undefined;
  const client = {
    configured: [] as (CompanionConfig | null)[],
    reconnects: 0,
    configure(c: CompanionConfig | null) {
      this.configured.push(c);
      const changed = JSON.stringify(c) !== JSON.stringify(current);
      current = c;
      return changed;
    },
    reconnectNow() {
      this.reconnects += 1;
    },
    onLink(cb: (s: LinkStatus) => void) {
      link = cb;
      return () => (link = null);
    },
    tell(s: LinkStatus) {
      link?.(s);
    },
  };
  return client;
}

/** The shell, faked: each ask waits until the test answers it. */
function fakeShell() {
  const asks: { resolve: (v: unknown) => void; reject: (e: unknown) => void }[] = [];
  const reports: LinkStatus[] = [];
  let onConfig: () => void = () => {};
  let onSettings: (s: unknown) => void = () => {};
  const deps: LinkDeps = {
    ask: () => new Promise((resolve, reject) => asks.push({ resolve, reject })),
    report: async (s) => void reports.push(s),
    onConfig: async (cb) => (onConfig = cb),
    onSettings: async (cb) => (onSettings = cb),
  };
  return {
    deps,
    asks,
    reports,
    config: () => onConfig(),
    settings: (s: unknown) => onSettings(s),
  };
}

const settle = () => new Promise((r) => setTimeout(r, 0));

async function started() {
  const client = fakeClient();
  const shell = fakeShell();
  await startCompanion(client as unknown as CompanionClient, shell.deps);
  return { client, shell };
}

describe("the companion link", () => {
  it("asks for the config at start and connects with it", async () => {
    const { client, shell } = await started();
    expect(shell.asks).toHaveLength(1);
    shell.asks[0]!.resolve(SHELL);
    await settle();
    expect(client.configured).toEqual([{ url: SHELL.url, token: SHELL.token, client: { id: "nick-pc", name: "NICK-PC" }, speaks: false }]);
    expect(client.reconnects).toBe(0);
  });

  it("offers to voice what he says only when his voice is installed, asked once", async () => {
    const cases: [(() => Promise<boolean>) | undefined, boolean][] = [
      [async () => true, true],
      [async () => false, false],
      [async () => Promise.reject(new Error("gone")), false],
      [undefined, false],
    ];
    for (const [ready, speaks] of cases) {
      const client = fakeClient();
      const shell = fakeShell();
      let asked = 0;
      if (ready) {
        shell.deps.voiceReady = () => {
          asked += 1;
          return ready();
        };
      }
      await startCompanion(client as unknown as CompanionClient, shell.deps);
      shell.asks[0]!.resolve(SHELL);
      await settle();
      expect(client.configured.at(-1)?.speaks, String(ready)).toBe(speaks);
      shell.config();
      shell.asks[1]!.resolve(SHELL);
      await settle();
      expect(client.configured.at(-1)?.speaks).toBe(speaks);
      if (ready) expect(asked, "once, at the start").toBe(1);
    }
  });

  it("the first config waits for the answer on his voice", async () => {
    const client = fakeClient();
    const shell = fakeShell();
    let answer: (v: boolean) => void = () => {};
    shell.deps.voiceReady = () => new Promise<boolean>((r) => (answer = r));
    await startCompanion(client as unknown as CompanionClient, shell.deps);
    shell.asks[0]!.resolve(SHELL);
    await settle();
    expect(client.configured).toEqual([]);
    answer(true);
    await settle();
    expect(client.configured.map((c) => c?.speaks)).toEqual([true]);
  });

  it("without a token, the client is unconfigured", async () => {
    const { client, shell } = await started();
    shell.asks[0]!.resolve({ ...SHELL, token: null });
    await settle();
    expect(client.configured).toEqual([null]);
  });

  it("the token saved, or Reconnect: asks again, and connects again even with nothing changed", async () => {
    const { client, shell } = await started();
    shell.asks[0]!.resolve(SHELL);
    await settle();
    shell.config();
    shell.asks[1]!.resolve(SHELL);
    await settle();
    expect(client.configured).toHaveLength(2);
    expect(client.reconnects).toBe(1);
    shell.config();
    shell.asks[2]!.resolve({ ...SHELL, token: "a-new-token" });
    await settle();
    expect(client.reconnects, "a changed config connects anew by itself").toBe(1);
  });

  it("asks again when brain.* changes, and only then", async () => {
    const { client, shell } = await started();
    shell.asks[0]!.resolve(SHELL);
    await settle();
    shell.settings(settingsWith(BRAIN, { theme: { mood: "dusk" } }));
    shell.settings(settingsWith(BRAIN));
    shell.settings(settingsWith({ url: 3 }));
    shell.settings(null);
    expect(shell.asks, "nothing about the brain changed").toHaveLength(1);
    for (const changed of [{ ...BRAIN, url: "ws://10.0.0.5:8001/companion" }, { ...BRAIN, clientId: "desk" },
      { ...BRAIN, clientName: "Desk" }]) {
      shell.settings(settingsWith(changed));
    }
    expect(shell.asks).toHaveLength(4);
    shell.asks[3]!.resolve({ ...SHELL, clientName: "Desk" });
    await settle();
    expect(client.configured.at(-1)?.client.name).toBe("Desk");
    expect(client.reconnects).toBe(0);
  });

  it("a brain change heard before the first answer is still asked about", async () => {
    const { client, shell } = await started();
    shell.settings(settingsWith({ ...BRAIN, url: "ws://10.0.0.5:8001/companion" }));
    expect(shell.asks).toHaveLength(2);
    shell.asks[1]!.resolve({ ...SHELL, url: "ws://10.0.0.5:8001/companion" });
    shell.asks[0]!.resolve(SHELL); // the older answer, late
    await settle();
    expect(client.configured.map((c) => c?.url)).toEqual(["ws://10.0.0.5:8001/companion"]);
  });

  it("leaves the client as it is when the shell's answer is no config", async () => {
    const { client, shell } = await started();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    shell.asks[0]!.reject("only the avatar's window may ask for that");
    shell.config();
    shell.asks[1]!.resolve({ url: SHELL.url });
    await settle();
    expect(client.configured).toEqual([]);
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("reports every change of the link, one at a time", async () => {
    const client = fakeClient();
    const shell = fakeShell();
    const sent: LinkStatus[] = [];
    const taken: (() => void)[] = [];
    shell.deps.report = (s) => {
      sent.push(s);
      return new Promise<void>((r) => taken.push(r));
    };
    await startCompanion(client as unknown as CompanionClient, shell.deps);
    const connecting: LinkStatus = {
      link: "connecting", reason: null, retryIn: null, identity: null, displayName: null, threadId: null, serverCaps: [],
    };
    const online: LinkStatus = {
      link: "online", reason: null, retryIn: null, identity: "desktop-nick-pc", displayName: "Nick", threadId: "desktop-nick-pc#desktop",
      serverCaps: ["text", "audio-in"],
    };
    client.tell(connecting);
    client.tell(online);
    expect(sent, "online waits until connecting has been taken").toEqual([connecting]);
    taken[0]!();
    await settle();
    expect(sent).toEqual([connecting, online]);
  });
});

describe("reports, one at a time", () => {
  const status = (link: LinkStatus["link"]): LinkStatus => ({
    link, reason: null, retryIn: null, identity: null, displayName: null, threadId: null, serverCaps: [],
  });

  it("waits for each to be taken, and sends only the newest of those that came meanwhile", async () => {
    const sent: string[] = [];
    const taken: (() => void)[] = [];
    const report = oneAtATime((s) => {
      sent.push(s.link);
      return new Promise<void>((r) => taken.push(r));
    });
    report(status("connecting"));
    report(status("offline"));
    report(status("online"));
    expect(sent).toEqual(["connecting"]);
    taken[0]!();
    await settle();
    expect(sent, "offline was overtaken while it waited").toEqual(["connecting", "online"]);
    taken[1]!();
    await settle();
    report(status("offline"));
    expect(sent).toEqual(["connecting", "online", "offline"]);
  });

  it("keeps going after one fails", async () => {
    const sent: string[] = [];
    const report = oneAtATime(async (s) => {
      sent.push(s.link);
      if (s.link === "connecting") throw new Error("the shell is busy");
    });
    report(status("connecting"));
    report(status("online"));
    await settle();
    expect(sent).toEqual(["connecting", "online"]);
  });
});

describe("the companion link's wire", () => {
  it("calls the commands the shell registers, with the arguments it takes", () => {
    const c = fixture.commands;
    expect(CONFIG_CMD).toBe(c.companionConfig.name);
    expect(Object.keys(configArgs())).toEqual(c.companionConfig.args);
    expect(REPORT_CMD).toBe(c.companionReport.name);
    const status: LinkStatus = {
      link: "offline", reason: "unreachable", retryIn: 4, identity: null, displayName: null, threadId: null, serverCaps: [],
    };
    expect(Object.keys(reportArgs(status))).toEqual(c.companionReport.args);
    expect(Object.keys(reportArgs(status).report).sort()).toEqual(fixture.companionStatusKeys);
    // What the brain can take goes to the shell as the welcome said it.
    expect(reportArgs({ ...status, link: "online", serverCaps: ["text", "audio-in"] }).report.serverCaps).toEqual(["text", "audio-in"]);
    expect(CONFIG_EVENT).toBe(fixture.events.companionConfig);
    expect(SAY_EVENT).toBe(fixture.events.companionSay);
    expect(SAID_CMD).toBe(c.companionSaid.name);
    expect(Object.keys(saidArgs(3, true))).toEqual(c.companionSaid.args);
    expect(TAKE_CMD).toBe(c.companionTake.name);
    expect(Object.keys(takeArgs(3))).toEqual(c.companionTake.args);
    // Push-to-talk's, PR 8.
    expect(PTT_LISTENING_EVENT).toBe(fixture.events.pttListening);
    expect(PTT_REFUSED_EVENT).toBe(fixture.events.pttRefused);
    expect(PTT_NOTHING_EVENT).toBe(fixture.events.pttNothing);
    expect(PTT_CLIP_EVENT).toBe(fixture.events.pttClip);
    expect(PTT_LEVEL_EVENT).toBe(fixture.events.pttLevel);
    expect(PTT_TAKE_CMD).toBe(c.pttTake.name);
    expect(Object.keys(pttTakeArgs(3))).toEqual(c.pttTake.args);
    expect(PTT_SENT_CMD).toBe(c.pttSent.name);
    expect(Object.keys(pttSentArgs(3, false, "why"))).toEqual(c.pttSent.args);
    expect(BRAIN_CANT_HEAR).toBe(fixture.pttWords.brainCantHear);
    expect(NOT_CONNECTED).toBe(fixture.pttWords.notConnected);
  });

  it("reads push-to-talk's payloads strictly", () => {
    expect(Object.keys({ why: "x" })).toEqual(fixture.pttWhyKeys);
    expect(parseWhy({ why: "Too short." })).toBe("Too short.");
    for (const bad of [null, "why", {}, { why: "" }, { why: "   " }, { why: 3 }, ["why"]]) {
      expect(parseWhy(bad), JSON.stringify(bad)).toBeNull();
    }
    const clip = { id: 4, millis: 2400, press: 2 };
    expect(Object.keys(clip).sort()).toEqual(fixture.pttClipKeys);
    expect(parseClip(clip)).toEqual(clip);
    expect(parseClip({ id: 1, millis: 0, press: 1 })).toEqual({ id: 1, millis: 0, press: 1 });
    for (const bad of [null, {}, { id: 0, millis: 1, press: 1 }, { id: -1, millis: 1, press: 1 }, { id: 1.5, millis: 1, press: 1 },
                       { id: "1", millis: 1, press: 1 }, { id: 1, press: 1 }, { id: 1, millis: -1, press: 1 },
                       { id: 1, millis: 2.5, press: 1 }, { id: 2 ** 60, millis: 1, press: 1 }, { id: 1, millis: 1 },
                       { id: 1, millis: 1, press: 0 }]) {
      expect(parseClip(bad), JSON.stringify(bad)).toBeNull();
    }
    const listening = { press: 3 };
    expect(Object.keys(listening)).toEqual(fixture.pttListeningKeys);
    expect(parseListening(listening)).toBe(3);
    for (const bad of [null, {}, { press: 0 }, { press: -1 }, { press: 1.5 }, { press: "3" }]) {
      expect(parseListening(bad), JSON.stringify(bad)).toBeNull();
    }
    const nothing = { press: 3, why: "That was too short for him to hear." };
    expect(Object.keys(nothing).sort()).toEqual(fixture.pttNothingKeys);
    expect(parseNothing(nothing)).toEqual(nothing);
    for (const bad of [{ why: "x" }, { press: 3 }, { press: 3, why: "" }, { press: 0, why: "x" }]) {
      expect(parseNothing(bad), JSON.stringify(bad)).toBeNull();
    }
    const level = { press: 3, level: 0.0312 };
    expect(Object.keys(level).sort()).toEqual(fixture.pttLevelKeys);
    expect(parseLevel(level)).toEqual(level);
    expect(parseLevel({ press: 3, level: 0 })).toEqual({ press: 3, level: 0 });
    expect(parseLevel({ press: 3, level: 1 })).toEqual({ press: 3, level: 1 });
    for (const bad of [null, { press: 3 }, { level: 0.5 }, { press: 0, level: 0.5 }, { press: 3, level: -0.1 },
                       { press: 3, level: 1.01 }, { press: 3, level: "0.5" }, { press: 3, level: null }]) {
      expect(parseLevel(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  /** A clip is taken, sent, and the shell told either way; one that can't
   * be taken, or sent, says why. */
  it("takes a clip, sends it and says so", async () => {
    const answers: [number, boolean, string | null][] = [];
    const answer = async (id: number, sent: boolean, why: string | null) => void answers.push([id, sent, why]);
    const pcm = new ArrayBuffer(4);
    let given: ArrayBuffer | null = null;
    let givenPress = 0;
    const sent = await onClip({ id: 5, millis: 2400, press: 9 }, async (id) => (id === 5 ? pcm : null), (b, press) => {
      given = b;
      givenPress = press;
      return { turnId: "t-1" };
    }, answer);
    expect(sent).toEqual({ kind: "sent", turnId: "t-1" });
    expect(given).toBe(pcm);
    expect(givenPress).toBe(9);
    expect(answers).toEqual([[5, true, null]]);
    const refused = await onClip({ id: 6, millis: 2400, press: 9 }, async () => pcm, () => ({ why: BRAIN_CANT_HEAR }), answer);
    expect(refused).toEqual({ kind: "unsent", why: BRAIN_CANT_HEAR, press: 9 });
    expect(answers[1]).toEqual([6, false, BRAIN_CANT_HEAR]);
    let asked = false;
    const gone = await onClip({ id: 7, millis: 2400, press: 10 }, async () => null, () => {
      asked = true;
      return { turnId: "never" };
    }, answer);
    expect(gone).toEqual({ kind: "unsent", why: CLIP_GONE, press: 10 });
    expect(asked, "nothing sent for a clip not taken").toBe(false);
    expect(answers[2]).toEqual([7, false, CLIP_GONE]);
    expect(await onClip({ id: 0 }, async () => pcm, () => ({ turnId: "x" }), answer)).toEqual({ kind: "bad" });
    expect(answers).toHaveLength(3);
  });

  it("takes a clip's bytes, and nothing else, from the shell", async () => {
    const buf = new ArrayBuffer(2);
    const calls: [string, Record<string, unknown>][] = [];
    expect(await takeClip(3, async (cmd, args) => (calls.push([cmd, args]), buf))).toBe(buf);
    expect(calls).toEqual([[PTT_TAKE_CMD, { id: 3 }]]);
    expect(await takeClip(3, async () => [1, 2])).toBeNull();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await takeClip(3, async () => { throw new Error("That clip isn't waiting any more."); })).toBeNull();
    expect(warn).toHaveBeenCalledOnce();
    await answerClip(3, false, "why", async () => { throw new Error("gone"); });
    expect(warn).toHaveBeenCalledTimes(2);
    expect(warn.mock.calls[1]![0]).toBe(`avatar: ${PTT_SENT_CMD} failed`);
    warn.mockRestore();
    const said: [string, Record<string, unknown>][] = [];
    await answerClip(4, false, "Too short.", async (cmd, args) => void said.push([cmd, args]));
    await answerClip(5, true, null, async (cmd, args) => void said.push([cmd, args]));
    expect(said).toEqual([[PTT_SENT_CMD, { id: 4, sent: false, why: "Too short." }], [PTT_SENT_CMD, { id: 5, sent: true, why: null }]]);
  });

  /** The shell's bridge for a turn: `{text, id}`, and nothing else is one. */
  it("reads a say strictly", () => {
    const ok = { text: "good evening", id: 7 };
    expect(Object.keys(ok).sort()).toEqual(fixture.sayKeys);
    expect(parseSay(ok)).toEqual(ok);
    expect(parseSay({ ...ok, text: "" })).toEqual({ ...ok, text: "" });
    for (const bad of [
      null, "good evening", ["good evening"], {}, { ...ok, text: 1 }, { words: "hi", id: 1 },
      { text: "hi" }, { ...ok, id: 0 }, { ...ok, id: -1 }, { ...ok, id: 1.5 }, { ...ok, id: "1" },
    ]) {
      expect(parseSay(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  /** A say the shell still waits for is taken, asked, and answered with
   * whether it went on; one the shell has given up is neither asked nor
   * answered; a bad one is ignored. */
  it("asks a say only once the shell lets it take it, and answers what asking did", async () => {
    const say = { text: "good evening", id: 7 };
    for (const went of [true, false]) {
      const take = vi.fn(async () => true);
      const ask = vi.fn(() => went);
      const answer = vi.fn(async () => {});
      expect(await onSay(say, take, ask, answer)).toBe("asked");
      expect(take).toHaveBeenCalledWith(7);
      expect(ask).toHaveBeenCalledWith("good evening");
      expect(answer).toHaveBeenCalledWith(7, went);
      expect(take.mock.invocationCallOrder[0]).toBeLessThan(ask.mock.invocationCallOrder[0]);
    }
    const ask = vi.fn(() => true);
    const answer = vi.fn(async () => {});
    expect(await onSay(say, async () => false, ask, answer)).toBe("late");
    const take = vi.fn(async () => true);
    expect(await onSay({ text: "hi" }, take, ask, answer)).toBe("bad");
    expect(take).not.toHaveBeenCalled();
    expect(ask).not.toHaveBeenCalled();
    expect(answer).not.toHaveBeenCalled();
  });

  /** The avatar page hands every say to onSay with the shell's take and
   * the answer, so the handshake above is the one it runs. */
  it("is what the avatar page runs for every say", () => {
    const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8");
    expect(main).toMatch(/listen<unknown>\(SAY_EVENT, \(e\) => \{\s*void onSay\(e\.payload, takeSay, ask, answerSay\)/);
  });

  /** The shell's yes is `true` and nothing else; no answer is a no. */
  it("takes a say only on the shell's yes", async () => {
    expect(await takeSay(7, async () => true)).toBe(true);
    for (const no of [false, null, "true", 1]) {
      expect(await takeSay(7, async () => no), JSON.stringify(no)).toBe(false);
    }
    const call = vi.fn(async () => true);
    await takeSay(9, call);
    expect(call).toHaveBeenCalledWith("companion_take", { id: 9 });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await takeSay(7, async () => { throw new Error("gone"); })).toBe(false);
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  /** The page tells the shell whether each turn went on, and a failed
   * answer is only a warning (the shell's wait ends by itself). */
  it("answers each say", async () => {
    const call = vi.fn(async () => undefined);
    await answerSay(7, false, call);
    expect(call).toHaveBeenCalledWith("companion_said", { id: 7, sent: false });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await answerSay(8, true, async () => {
      throw new Error("gone");
    });
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it("reads the config strictly", () => {
    expect(Object.keys(SHELL).sort()).toEqual(fixture.companionConfigKeys);
    expect(parseShellConfig(SHELL)).toEqual(SHELL);
    expect(parseShellConfig({ ...SHELL, token: null })).toEqual({ ...SHELL, token: null });
    expect(parseShellConfig({ ...SHELL, token: 3 })).toBeNull();
    expect(parseShellConfig({ ...SHELL, url: null })).toBeNull();
    const { clientName: _, ...noName } = SHELL;
    expect(parseShellConfig(noName)).toBeNull();
    expect(parseShellConfig([SHELL])).toBeNull();
    expect(configFrom({ ...SHELL, token: "" })).toBeNull();
  });

  it("every report the page can make is one the shell keeps", () => {
    for (const [name, sample] of Object.entries(fixture.companionStatusSamples)) {
      expect(Object.keys(sample).sort(), name).toEqual(fixture.companionStatusKeys);
    }
  });
});
