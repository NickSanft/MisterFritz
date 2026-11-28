/**
 * Which of the brain's widgets are fresh, for the study circle's threads.
 * The shell passes on each widget the avatar's page hears (`widget:seen`)
 * and keeps the newest of each for a page that starts later
 * (`widgets_now`, each with its age); a widget holds for its `ttlS`
 * seconds (0: until the next), and none is fresh while the brain's link
 * isn't online.
 */

export const SEEN_EVENT = "widget:seen";
export const NOW_CMD = "widgets_now";
/** The brain's link now, for a page that starts after its last word. */
export const STATUS_CMD = "companion_status";

/** A widget as the shell passes it on (its payload is PR 13's). */
export interface Seen {
  name: string;
  ttlS: number;
  ageMs: number;
}

function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

const whole = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

/** A `widget:seen`, or an item of `widgets_now`; null when it isn't one. */
export function parseSeen(v: unknown): Seen | null {
  const o = obj(v);
  if (!o || typeof o.name !== "string" || o.name === "" || !whole(o.ttlS) || !whole(o.ageMs)) return null;
  return { name: o.name, ttlS: o.ttlS, ageMs: o.ageMs };
}

/** What was heard of each widget, by the page's clock (milliseconds). */
export class Freshness {
  private heard = new Map<string, { ttlS: number; at: number }>();

  /** `s` heard at `now`: kept unless what is kept already came later (a
   * late answer to `widgets_now` never undoes a newer event). */
  hear(s: Seen, now: number): void {
    const at = now - s.ageMs;
    const was = this.heard.get(s.name);
    if (!was || at >= was.at) this.heard.set(s.name, { ttlS: s.ttlS, at });
  }

  /** Whether widget `name` is fresh at `now`, the link `online`. */
  lit(name: string, now: number, online: boolean): boolean {
    if (!online) return false;
    const h = this.heard.get(name);
    if (!h) return false;
    return h.ttlS === 0 || now - h.at < h.ttlS * 1000;
  }

  /** How long until a widget fresh at `now` stops being, in milliseconds:
   * the soonest; null when none will (none fresh with a ttl). */
  untilStale(now: number): number | null {
    let soonest: number | null = null;
    for (const { ttlS, at } of this.heard.values()) {
      if (ttlS === 0) continue;
      const left = at + ttlS * 1000 - now;
      if (left > 0 && (soonest === null || left < soonest)) soonest = left;
    }
    return soonest;
  }
}
