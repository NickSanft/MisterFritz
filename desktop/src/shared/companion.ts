/**
 * The brain's companion client: this device's one session with the brain's
 * /companion socket (plan 14 build § 2.2), held by the avatar page. The
 * frames are protocol.ts's, generated from the brain's companion_protocol.py.
 *
 * - The first frame is the hello. The link is online only once the welcome
 *   comes; anything before it is ignored.
 * - A close the client can't mend by trying again is a refusal, and it never
 *   retries one: a wrong token (4401), a protocol version the brain doesn't
 *   speak (4426), a hello it couldn't read (4400). Anything else is tried
 *   again after 1, 2, 4, 8, 16, then every 30 seconds: the brain full
 *   (4409), gone quiet (4000, 4408, or no welcome in time), or not there at
 *   all (1006, and the rest). A welcome starts the count again.
 * - The heartbeat: a ping from the brain is answered with a pong carrying
 *   its `t`; the client pings every 20 seconds; and once nothing at all has
 *   come for two of those and a little more, it drops the socket and
 *   connects again, without waiting for the socket to say it closed.
 * - A frame that isn't JSON, or isn't a frame the server sends
 *   (`isServerFrame`), is ignored.
 *
 * Retrying while offline isn't animation: it runs on the clock, not the
 * paced loop. Audio (`sendAudio`, PR 8) and widgets (`subscribe`, PR 13)
 * come with the PRs that use them; the brain answers both bad_frame today.
 */
import { browserClock, type Clock } from "./clock";
import {
  CLOSE,
  PROTOCOL_VERSION,
  SAY_MAX_CHARS,
  TURN_ID_PATTERN,
  isServerFrame,
  type Cancel,
  type Cap,
  type Hello,
  type Ping,
  type Pong,
  type Say,
  type ServerFrame,
} from "./protocol";

/** Where the link stands. `unconfigured`: there is no token to connect
 * with. `offline`: it will try again (`retryIn` seconds, or now when null).
 * `refused`: it won't, until it is configured anew or told to. */
export type Link = "unconfigured" | "connecting" | "online" | "offline" | "refused";
/** Why it is offline or refused. */
export type Reason = "token" | "version" | "hello" | "full" | "quiet" | "unreachable" | null;

export interface CompanionConfig {
  url: string;
  token: string;
  client: { id: string; name: string };
}

export interface LinkStatus {
  link: Link;
  reason: Reason;
  retryIn: number | null;
  /** The welcome's: who the brain takes this desk to be, while online. */
  identity: string | null;
  displayName: string | null;
  threadId: string | null;
}

/** What the client needs of a WebSocket; tests pass a fake. */
export interface SocketLike {
  send(data: string): void;
  close(code?: number): void;
  onopen: (() => void) | null;
  onmessage: ((e: { data: unknown }) => void) | null;
  onclose: ((e: { code: number }) => void) | null;
  onerror: (() => void) | null;
}

export interface ClientDeps {
  socket(url: string): SocketLike;
  clock: Clock;
  uuid(): string;
  /** The time of day in milliseconds, for a ping's `t`. */
  wallNow(): number;
}

/** What this desk can do, told to the brain in its hello. */
export const CAPS: Cap[] = ["text", "vrm"];
/** Seconds between tries, then the last for ever. */
export const BACKOFF_S = [1, 2, 4, 8, 16, 30] as const;
export const WELCOME_TIMEOUT_MS = 10_000;
export const PING_EVERY_MS = 20_000;
/** How long the brain may say nothing at all before the socket is dropped:
 * two of its pings missed, and a little more. */
export const SILENCE_MS = 2 * PING_EVERY_MS + 5_000;

/** The wait before try number `attempt` (0 is the first retry), in seconds. */
export function backoffDelay(attempt: number): number {
  const i = Math.max(0, Math.min(BACKOFF_S.length - 1, Math.floor(attempt)));
  return BACKOFF_S[i]!;
}

/** What a close code means, and whether trying again could help. */
export function closeReason(code: number): { reason: Exclude<Reason, null>; retry: boolean } {
  switch (code) {
    case CLOSE.badToken:
      return { reason: "token", retry: false };
    case CLOSE.unsupported:
      return { reason: "version", retry: false };
    case CLOSE.badFrame:
      return { reason: "hello", retry: false };
    case CLOSE.full:
      return { reason: "full", retry: true };
    case CLOSE.goneQuiet:
    case CLOSE.noHello:
      return { reason: "quiet", retry: true };
    default:
      return { reason: "unreachable", retry: true };
  }
}

// The frames the client sends, built in one place so the tests hold each to
// the generated fixtures.
export const helloFrame = (c: CompanionConfig): Hello => ({
  type: "hello",
  protocol: PROTOCOL_VERSION,
  token: c.token,
  client: { id: c.client.id, name: c.client.name, caps: [...CAPS] },
});
export const sayFrame = (turnId: string, text: string): Say => ({ type: "say", turn_id: turnId, text });
export const cancelFrame = (turnId: string): Cancel => ({ type: "cancel", turn_id: turnId });
export const pingFrame = (t: number): Ping => ({ type: "ping", t });
export const pongFrame = (t: number): Pong => ({ type: "pong", t });

/** `text` as a `say` may carry it: trimmed, and 1 to SAY_MAX_CHARS
 * characters (code points, as the brain counts them); else null. Trimmed of
 * Unicode's White_Space, as the brain's `strip_whitespace` trims, not JS's
 * `trim`: the two differ (U+0085 is space to the brain; U+FEFF is not). */
export function sayable(text: string): string | null {
  const t = text.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "");
  if (t === "") return null;
  let n = 0;
  for (const _ of t) if (++n > SAY_MAX_CHARS) return null;
  return t;
}

const OFFLINE: LinkStatus = {
  link: "unconfigured",
  reason: null,
  retryIn: null,
  identity: null,
  displayName: null,
  threadId: null,
};

function sameConfig(a: CompanionConfig | null, b: CompanionConfig | null): boolean {
  if (a === null || b === null) return a === b;
  return a.url === b.url && a.token === b.token && a.client.id === b.client.id && a.client.name === b.client.name;
}

function sameStatus(a: LinkStatus, b: LinkStatus): boolean {
  return (
    a.link === b.link &&
    a.reason === b.reason &&
    a.retryIn === b.retryIn &&
    a.identity === b.identity &&
    a.displayName === b.displayName &&
    a.threadId === b.threadId
  );
}

const browserDeps = (): ClientDeps => ({
  socket: (url) => new WebSocket(url) as unknown as SocketLike,
  clock: browserClock,
  uuid: () => crypto.randomUUID(),
  wallNow: () => Date.now(),
});

type Listener = (f: ServerFrame) => void;

export class CompanionClient {
  private readonly deps: ClientDeps;
  private config: CompanionConfig | null = null;
  private current: LinkStatus = OFFLINE;
  private sock: SocketLike | null = null;
  /** Bumped for each socket: a callback from an older one is ignored. */
  private gen = 0;
  /** Retries since the last welcome. */
  private attempt = 0;
  private opened = false;
  private welcomed = false;
  private lastFrameAt = 0;
  private retryTimer: unknown = null;
  private welcomeTimer: unknown = null;
  private pingTimer: unknown = null;
  private closed = false;
  private readonly frameListeners = new Map<string, Set<Listener>>();
  private readonly linkListeners = new Set<(s: LinkStatus) => void>();

  constructor(deps: Partial<ClientDeps> = {}) {
    this.deps = { ...browserDeps(), ...deps };
  }

  get status(): LinkStatus {
    return { ...this.current };
  }

  /** Where to connect, and as whom; null when there is no token. Null is
   * unconfigured: closed, with no retries. The same config again changes
   * nothing; a changed one connects anew at once, the count started again.
   * Returns whether it changed anything. */
  configure(c: CompanionConfig | null): boolean {
    if (this.closed || sameConfig(this.config, c)) return false;
    this.config = c ? { url: c.url, token: c.token, client: { ...c.client } } : null;
    if (!this.config) {
      this.drop();
      this.setStatus(OFFLINE);
      return true;
    }
    this.attempt = 0;
    this.connect();
    return true;
  }

  /** Connect again now, whatever the link: the owner's Reconnect, or a
   * token saved again. Nothing without a config. */
  reconnectNow(): void {
    if (this.closed || !this.config) return;
    this.attempt = 0;
    this.connect();
  }

  /** For good: the socket closed, and nothing connects again. */
  close(): void {
    this.drop();
    this.closed = true;
    this.frameListeners.clear();
    this.linkListeners.clear();
  }

  /** Ask him something: the turn's id, or null when the link isn't online
   * or the text isn't something a `say` carries (empty once trimmed, or
   * longer than SAY_MAX_CHARS). */
  say(text: string): string | null {
    const t = sayable(text);
    if (t === null || !this.online()) return null;
    const turnId = this.deps.uuid();
    if (!TURN_ID_PATTERN.test(turnId)) return null;
    return this.send(sayFrame(turnId, t)) ? turnId : null;
  }

  /** Stop a turn (the brain ends it with `done`, interrupted). */
  cancel(turnId: string): void {
    if (this.online() && TURN_ID_PATTERN.test(turnId)) this.send(cancelFrame(turnId));
  }

  /** Hear every frame of one type, from the welcome on. Returns the
   * unsubscribe. */
  on<K extends ServerFrame["type"]>(type: K, cb: (f: Extract<ServerFrame, { type: K }>) => void): () => void {
    let set = this.frameListeners.get(type);
    if (!set) {
      set = new Set();
      this.frameListeners.set(type, set);
    }
    const listener = cb as Listener;
    set.add(listener);
    return () => void this.frameListeners.get(type)?.delete(listener);
  }

  /** Hear every change of the link. Returns the unsubscribe. */
  onLink(cb: (s: LinkStatus) => void): () => void {
    this.linkListeners.add(cb);
    return () => void this.linkListeners.delete(cb);
  }

  // ── The socket ──────────────────────────────────────────────────────────

  private online(): boolean {
    return this.welcomed && this.sock !== null;
  }

  private connect(): void {
    this.drop();
    const config = this.config;
    if (this.closed || !config) return;
    const gen = ++this.gen;
    // The first try after a configure or a welcome says it is connecting;
    // a retry keeps what went wrong on show (with no wait left), so the
    // link doesn't flicker on every try.
    if (this.attempt === 0) this.setStatus({ ...OFFLINE, link: "connecting" });
    else this.setStatus({ ...this.current, retryIn: null });
    let sock: SocketLike;
    try {
      sock = this.deps.socket(config.url);
    } catch {
      // A URL the browser won't open: nothing to wait for.
      this.lost(gen, 1006);
      return;
    }
    this.sock = sock;
    this.opened = false;
    this.welcomed = false;
    sock.onopen = () => {
      if (gen !== this.gen) return;
      this.opened = true;
      this.send(helloFrame(config));
    };
    sock.onmessage = (e) => {
      if (gen === this.gen) this.received(e.data);
    };
    sock.onclose = (e) => {
      if (gen === this.gen) this.lost(gen, e.code);
    };
    sock.onerror = () => {
      // onclose follows, with the code.
    };
    this.welcomeTimer = this.deps.clock.setTimeout(() => {
      if (gen !== this.gen || this.welcomed) return;
      // Opened, but no welcome: the brain has gone quiet. Never opened: it
      // isn't there (a host that drops the connection silently).
      this.lost(gen, this.opened ? CLOSE.goneQuiet : 1006);
    }, WELCOME_TIMEOUT_MS);
  }

  private received(data: unknown): void {
    this.lastFrameAt = this.deps.clock.now();
    if (typeof data !== "string") return;
    let frame: unknown;
    try {
      frame = JSON.parse(data);
    } catch {
      return;
    }
    if (!isServerFrame(frame)) return;
    if (!this.welcomed) {
      if (frame.type === "welcome") this.welcome(frame);
      return;
    }
    if (frame.type === "ping") this.send(pongFrame(frame.t));
    this.emit(frame);
  }

  private welcome(w: Extract<ServerFrame, { type: "welcome" }>): void {
    if (w.protocol !== PROTOCOL_VERSION) {
      this.lost(this.gen, CLOSE.unsupported);
      return;
    }
    this.welcomed = true;
    this.attempt = 0;
    this.clear("welcomeTimer");
    this.lastFrameAt = this.deps.clock.now();
    this.schedulePing(this.gen);
    this.setStatus({
      link: "online",
      reason: null,
      retryIn: null,
      identity: w.identity,
      displayName: w.display_name,
      threadId: w.thread_id,
    });
    this.emit(w);
  }

  private schedulePing(gen: number): void {
    this.pingTimer = this.deps.clock.setTimeout(() => {
      if (gen !== this.gen) return;
      if (this.deps.clock.now() - this.lastFrameAt > SILENCE_MS) {
        this.lost(gen, CLOSE.goneQuiet);
        return;
      }
      this.send(pingFrame(Math.trunc(this.deps.wallNow())));
      this.schedulePing(gen);
    }, PING_EVERY_MS);
  }

  /** The socket is gone (closed, or dropped by the client): refused for
   * good, or tried again after the backoff. */
  private lost(gen: number, code: number): void {
    if (gen !== this.gen) return;
    this.drop();
    const { reason, retry } = closeReason(code);
    if (!retry) {
      this.setStatus({ ...OFFLINE, link: "refused", reason });
      return;
    }
    const wait = backoffDelay(this.attempt);
    this.attempt += 1;
    this.setStatus({ ...OFFLINE, link: "offline", reason, retryIn: wait });
    this.retryTimer = this.deps.clock.setTimeout(() => this.connect(), wait * 1000);
  }

  /** Close the socket (if any) and stop every timer; its callbacks are
   * ignored from now on. */
  private drop(): void {
    this.gen += 1;
    this.clear("retryTimer");
    this.clear("welcomeTimer");
    this.clear("pingTimer");
    const sock = this.sock;
    this.sock = null;
    this.welcomed = false;
    this.opened = false;
    if (sock) {
      sock.onopen = sock.onmessage = sock.onclose = sock.onerror = null;
      try {
        sock.close(1000);
      } catch {
        // Already closing.
      }
    }
  }

  private clear(timer: "retryTimer" | "welcomeTimer" | "pingTimer"): void {
    if (this[timer] !== null) this.deps.clock.clearTimeout(this[timer]);
    this[timer] = null;
  }

  private send(frame: object): boolean {
    if (!this.sock) return false;
    try {
      this.sock.send(JSON.stringify(frame));
      return true;
    } catch {
      return false;
    }
  }

  private emit(frame: ServerFrame): void {
    for (const cb of [...(this.frameListeners.get(frame.type) ?? [])]) {
      try {
        cb(frame);
      } catch (e) {
        console.error(`companion: a ${frame.type} listener failed`, e);
      }
    }
  }

  private setStatus(next: LinkStatus): void {
    if (sameStatus(this.current, next)) return;
    this.current = { ...next };
    for (const cb of [...this.linkListeners]) {
      try {
        cb(this.status);
      } catch (e) {
        console.error("companion: a link listener failed", e);
      }
    }
  }
}
