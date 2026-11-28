/**
 * Where the study circle is, following Fritz: the shell tells this page
 * where he is on its monitor (`wallpaper:fritz`: his core in physical
 * pixels from the monitor's corner, his window's height, whether he is
 * shown), or `null` when he isn't on it. The circle eases after him (a
 * quarter second) and fades in and out (a little longer) in real time; a
 * still scene, or a first sight of him, puts it there at once. Hidden, or
 * gone to another monitor, it fades where it was.
 */

export const FRITZ_EVENT = "wallpaper:fritz";
export const FRITZ_CMD = "wallpaper_fritz";

/** Seconds for the circle to come most of the way after him, and to fade. */
export const FOLLOW_S = 0.25;
export const FADE_S = 0.4;

/** Fritz on this page's monitor, as the shell says it. */
export interface Place {
  x: number;
  y: number;
  size: number;
  shown: boolean;
}

const int = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v);

/** A `wallpaper:fritz` payload: a place, null (not on this monitor), or
 * undefined when it is neither. */
export function parsePlace(v: unknown): Place | null | undefined {
  if (v === null) return null;
  if (typeof v !== "object" || Array.isArray(v)) return undefined;
  const o = v as Record<string, unknown>;
  if (!int(o.x) || !int(o.y) || !int(o.size) || o.size <= 0 || typeof o.shown !== "boolean") return undefined;
  return { x: o.x, y: o.y, size: o.size, shown: o.shown };
}

export class Follow {
  /** The circle now, in CSS pixels, and how much of it shows (0 to 1). */
  x = 0;
  y = 0;
  size = 0;
  presence = 0;
  private place: Place | null = null;
  private dpr = 1;
  private placed = false;

  /** Where the shell says he is now. */
  set(place: Place | null): void {
    this.place = place;
  }

  /** The page's device pixels per CSS pixel, for the shell's physical ones. */
  setScale(dpr: number): void {
    this.dpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  }

  /** Step `dt` seconds of real time towards where he is; `snap` puts it
   * there at once. Whether it is still on its way. */
  step(dt: number, snap: boolean): boolean {
    const p = this.place;
    const want = p && p.shown ? 1 : 0;
    if (p) {
      const tx = p.x / this.dpr;
      const ty = p.y / this.dpr;
      const ts = p.size / this.dpr;
      // Not seen yet, or faded right out: it comes in where he is now.
      if (snap || !this.placed || this.presence < 0.01) {
        this.x = tx;
        this.y = ty;
        this.size = ts;
        this.placed = true;
      } else {
        const k = ease(dt, FOLLOW_S);
        this.x += (tx - this.x) * k;
        this.y += (ty - this.y) * k;
        this.size += (ts - this.size) * k;
      }
    }
    this.presence = snap ? want : this.presence + (want - this.presence) * ease(dt, FADE_S);
    if (Math.abs(want - this.presence) < 0.002) this.presence = want;
    const away = p ? Math.hypot(p.x / this.dpr - this.x, p.y / this.dpr - this.y) : 0;
    if (away < 0.25 && p) {
      this.x = p.x / this.dpr;
      this.y = p.y / this.dpr;
      this.size = p.size / this.dpr;
    }
    return this.presence !== want || away >= 0.25;
  }
}

/** The share of the way covered in `dt` seconds, `tau` being the time to
 * come about two-thirds of it. Pure. */
export function ease(dt: number, tau: number): number {
  if (!(dt > 0)) return 0;
  return 1 - Math.exp(-dt / tau);
}
