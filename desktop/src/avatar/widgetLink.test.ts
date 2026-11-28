import { describe, expect, it } from "vitest";
import fixture from "./wire.fixture.json";
import protocolFixtures from "../shared/protocol.fixtures.json";
import { CompanionClient, subscribeFrame, type SocketLike } from "../shared/companion";
import { WIDGET_NAMES } from "../shared/protocol";
import { WANTED, WIDGET_CMD, linkWidgets, type Handed } from "./widgetLink";

class FakeSocket implements SocketLike {
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  send(data: string | ArrayBuffer): void {
    if (typeof data === "string") this.sent.push(JSON.parse(data));
  }
  close(): void {}
  open(): void {
    this.onopen?.();
  }
  frame(f: unknown): void {
    this.onmessage?.({ data: JSON.stringify(f) });
  }
}

const sample = (type: string) => {
  const found = protocolFixtures.server.valid.find((e) => (e.frame as { type: string }).type === type);
  if (!found) throw new Error(`no ${type}`);
  return found.frame as Record<string, unknown>;
};

function harness() {
  const sockets: FakeSocket[] = [];
  const client = new CompanionClient({
    socket: () => {
      const s = new FakeSocket();
      sockets.push(s);
      return s;
    },
    clock: { now: () => 0, setTimeout: () => 0, clearTimeout: () => {} },
    uuid: () => "7f3c2a10-0000-4000-8000-000000000001",
    wallNow: () => 0,
  });
  const handed: Handed[] = [];
  const off = linkWidgets(client, { hand: async (w) => void handed.push(w) });
  const connect = (caps: string[]) => {
    client.configure({ url: "ws://127.0.0.1:8001/companion", token: "t".repeat(43), client: { id: "desk", name: "Desk" } });
    const s = sockets[sockets.length - 1]!;
    s.open();
    s.frame({ ...sample("welcome"), server_caps: caps });
    return s;
  };
  return { client, connect, handed, off, sockets };
}

describe("the widgets, for the study scene", () => {
  it("are every widget the brain feeds, asked for by name", () => {
    expect([...WANTED]).toEqual([...WIDGET_NAMES]);
    expect([...WANTED].sort()).toEqual(["clock", "desk", "weather"]);
    expect(WIDGET_CMD).toBe(fixture.commands.companionWidget.name);
    expect(fixture.commands.companionWidget.args).toEqual(["name", "ttlS", "payload"]);
    const valid = protocolFixtures.client.valid.subscribe as { type: string; widgets: string[] };
    expect(subscribeFrame(valid.widgets)).toEqual(valid);
  });

  it("are asked for on every welcome whose brain offers them", () => {
    const h = harness();
    const s = h.connect(["text", "widgets"]);
    expect(s.sent.filter((f) => (f as { type: string }).type === "subscribe"))
      .toEqual([{ type: "subscribe", widgets: [...WIDGET_NAMES] }]);
    // Connected again: a new session asks again.
    h.client.reconnectNow();
    const again = h.sockets[h.sockets.length - 1]!;
    again.open();
    again.frame({ ...sample("welcome"), server_caps: ["text", "widgets"] });
    expect(again.sent.filter((f) => (f as { type: string }).type === "subscribe")).toHaveLength(1);
  });

  it("aren't asked while the link is down", () => {
    const h = harness();
    const s = h.connect(["text", "widgets"]);
    s.onclose?.({ code: 1006 });
    expect(h.client.status.link).not.toBe("online");
    const sent = s.sent.length;
    expect(h.client.subscribe(WANTED)).toBe(false);
    expect(s.sent).toHaveLength(sent);
  });

  it("aren't asked of a brain that doesn't offer them", () => {
    const h = harness();
    const s = h.connect(["text"]);
    expect(s.sent.some((f) => (f as { type: string }).type === "subscribe")).toBe(false);
    expect(h.client.subscribe(WANTED)).toBe(false);
  });

  it("are handed to the shell as they come, and no longer once unlinked", () => {
    const h = harness();
    const s = h.connect(["text", "widgets"]);
    s.frame({ type: "widget", name: "clock", payload: { time: "09:00" }, ttl_s: 120 });
    s.frame({ type: "widget", name: "desk", payload: {}, ttl_s: 0 });
    expect(h.handed).toEqual([
      { name: "clock", ttlS: 120, payload: { time: "09:00" } },
      { name: "desk", ttlS: 0, payload: {} },
    ]);
    h.off();
    s.frame({ type: "widget", name: "clock", payload: {}, ttl_s: 120 });
    expect(h.handed).toHaveLength(2);
  });
});
