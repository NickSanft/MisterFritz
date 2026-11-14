/**
 * The avatar page's link to the shell for the brain's connection: what it
 * asks for and when, how it configures the client, and what it reports,
 * by the names and keys the shell has (wire.fixture.json).
 */
import { describe, expect, it, vi } from "vitest";
import fixture from "./wire.fixture.json";
import type { CompanionClient, CompanionConfig, LinkStatus } from "../shared/companion";
import {
  CONFIG_CMD,
  CONFIG_EVENT,
  REPORT_CMD,
  SAY_EVENT,
  configArgs,
  configFrom,
  oneAtATime,
  parseSay,
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
    expect(client.configured).toEqual([{ url: SHELL.url, token: SHELL.token, client: { id: "nick-pc", name: "NICK-PC" } }]);
    expect(client.reconnects).toBe(0);
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
    const connecting: LinkStatus = { link: "connecting", reason: null, retryIn: null, identity: null, displayName: null, threadId: null };
    const online: LinkStatus = {
      link: "online", reason: null, retryIn: null, identity: "desktop-nick-pc", displayName: "Nick", threadId: "desktop-nick-pc#desktop",
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
    link, reason: null, retryIn: null, identity: null, displayName: null, threadId: null,
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
    const status: LinkStatus = { link: "offline", reason: "unreachable", retryIn: 4, identity: null, displayName: null, threadId: null };
    expect(Object.keys(reportArgs(status))).toEqual(c.companionReport.args);
    expect(Object.keys(reportArgs(status).report).sort()).toEqual(fixture.companionStatusKeys);
    expect(CONFIG_EVENT).toBe(fixture.events.companionConfig);
    expect(SAY_EVENT).toBe(fixture.events.companionSay);
  });

  /** The shell's bridge for a turn: `{text}`, and nothing else is one. */
  it("reads a say strictly", () => {
    expect(parseSay({ text: "good evening" })).toBe("good evening");
    expect(parseSay({ text: "" })).toBe("");
    for (const bad of [null, "good evening", ["good evening"], {}, { text: 1 }, { words: "hi" }]) {
      expect(parseSay(bad), JSON.stringify(bad)).toBeNull();
    }
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
