/**
 * His face for his mood (plan 14 PR 11): the brain tells the page a reply's
 * mood (the `mood` frame, from the tag the model began it with), and his
 * model's expression for it eases in while the reply shows.
 *
 * The owner's mapping, subtle: dry is his resting face (no expression);
 * pleased is `happy` at 0.4 (a VRoid 0.x export's Joy), alarmed `surprised`
 * at 0.5 (the owner's export has it as a custom expression, `Surprised`),
 * weary `sad` at 0.4 (its Sorrow). An expression eases in over IN_MS and
 * out over OUT_MS, whatever the motion setting (his mouth moves under it,
 * unless the model's own expression is made to hold the mouth still, which
 * a VRM 1.0 model may be; a binary one shows nothing at these weights); a
 * new mood eases the old one out as it eases in.
 *
 * Kept pure: `set` says which mood to show (or none), `tick` moves it on by
 * a frame's time, `faces` is what to set on the model this frame. The page
 * ticks it from the paced loop, so it waits while he is parked.
 */
import { smoothstep } from "../shared/motion";
import { MOOD_NAME_VALUES, type MoodName } from "../shared/protocol";

/** An expression for a mood: the first of `names` the model has, at `weight`. */
export interface FaceFor {
  readonly names: readonly string[];
  readonly weight: number;
}

/** His face for each mood; null is his resting face. */
export const FACES: Readonly<Record<MoodName, FaceFor | null>> = Object.freeze({
  dry: null,
  pleased: Object.freeze({ names: Object.freeze(["happy"]), weight: 0.4 }),
  alarmed: Object.freeze({ names: Object.freeze(["surprised", "Surprised"]), weight: 0.5 }),
  weary: Object.freeze({ names: Object.freeze(["sad"]), weight: 0.4 }),
});

/** How long an expression takes to come in, and to go, in milliseconds. */
export const IN_MS = 300;
export const OUT_MS = 1000;
/** Closer than this to either end is at it. */
const SETTLE = 1e-9;

export class Face {
  private want: MoodName | null = null;
  /** Each mood's way in, 0 (not shown) to 1 (wholly shown), eased when set. */
  private readonly p = new Map<MoodName, number>(MOOD_NAME_VALUES.map((m) => [m, 0]));

  /** The mood to show, or none (his resting face). */
  set(mood: MoodName | null): void {
    this.want = mood !== null && this.p.has(mood) ? mood : null;
  }

  get mood(): MoodName | null {
    return this.want;
  }

  /** Move on by `dtMs` milliseconds: the mood shown comes in, every other goes. */
  tick(dtMs: number): void {
    const dt = Number.isFinite(dtMs) && dtMs > 0 ? dtMs : 0;
    for (const [m, p] of this.p) {
      const q = m === this.want ? p + dt / IN_MS : p - dt / OUT_MS;
      // Settles exactly: frame times that sum to the duration may leave a
      // rounding's worth either side.
      this.p.set(m, q >= 1 - SETTLE ? 1 : q <= SETTLE ? 0 : q);
    }
  }

  /** Every mood's expression and its weight now (0 when it isn't shown), so
   * one that has gone is set back to nothing. His resting face sets none. */
  faces(): FaceFor[] {
    const out: FaceFor[] = [];
    for (const [m, p] of this.p) {
      const face = FACES[m];
      if (face) out.push({ names: face.names, weight: face.weight * smoothstep(p) });
    }
    return out;
  }
}
