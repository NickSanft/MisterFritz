/**
 * Where Fritz's body is on his page: the shape the shell keeps so a click
 * on him can be told from one beside him (src-tauri/src/hit_shape.rs).
 *
 * One box round him would make the empty corners beside his head his too,
 * so the shape is a short stack of boxes. Once per model, standing at rest,
 * his triangles are cut into horizontal bands, each as wide as he is at
 * that height, and neighbouring bands are merged until at most `MAX_RECTS`
 * are left. Whenever his page is fitted (a load, a resize, a new scale), each
 * band is seen through the camera into CSS pixels, with a little padding,
 * since his idle sways and breathes him a few pixels either way.
 *
 * The page offers its shape at every fit; only a move of `MIN_MOVE` pixels
 * or more is sent, one send at a time and the newest only, each numbered.
 * While the page draws, the shape is sent again every `KEEP_MS`, which is
 * how the shell knows the page is still there: a shape it hasn't heard for
 * three of those is gone.
 *
 * Pure but for three's maths: nothing here touches the DOM or the shell.
 */
import * as THREE from "three";
import { oneAtATime } from "../shared/oneAtATime";

/** The bands his height is first cut into. */
export const BANDS = 12;
/** The most boxes a shape has (the shell's MAX_RECTS). */
export const MAX_RECTS = 6;
/** The padding round each box, as a share of the view's shorter side. */
export const PAD = 0.03;
/** An edge moved by this many CSS pixels is a new shape. */
export const MIN_MOVE = 2;
/** How often the shape is sent again while the page draws, in ms. */
export const KEEP_MS = 5000;
/** The most triangles measured: past it, every nth is (a model that large
 * is dense, so what a stride skips lies beside what it keeps). */
export const SAMPLE_CAP = 200_000;

/** A slice of him at rest, in world metres: its height, and how far he
 * reaches across and in depth within it. */
export interface Band {
  y0: number;
  y1: number;
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

/** A box of his shape on the page, in whole CSS pixels. */
export interface HitRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface View {
  width: number;
  height: number;
}

/** One send: the page's count of them, and the shape as the view saw it. */
export interface HitSend {
  seq: number;
  view: View;
  rects: HitRect[];
}

/** Every nth triangle, so no more than `cap` of `total` are taken. */
export function strideFor(total: number, cap: number): number {
  return Math.max(1, Math.ceil(total / Math.max(1, cap)));
}

/** The triangles of `root`'s visible meshes as they are posed now, in the
 * world: nine numbers each (three corners' x, y, z), at most about `cap` of
 * them, every nth past that. Each corner is read as three's precise
 * bounding box reads a vertex: with its morphs and skinning, then the
 * mesh's place in the world. Triangles, not vertices alone: a long flat
 * face (a low-poly shin) is his all the way along it, though it has no
 * vertex between its ends. */
export function sampleTriangles(root: THREE.Object3D, cap = SAMPLE_CAP): Float32Array {
  root.updateMatrixWorld(true);
  const meshes: { mesh: THREE.Mesh; triangles: number }[] = [];
  let total = 0;
  root.traverseVisible((o) => {
    const mesh = o as THREE.Mesh;
    const position = mesh.isMesh ? mesh.geometry?.getAttribute("position") : undefined;
    if (!position) return;
    const triangles = Math.floor((mesh.geometry.index?.count ?? position.count) / 3);
    meshes.push({ mesh, triangles });
    total += triangles;
  });
  const stride = strideFor(total, cap);
  const out: number[] = [];
  const v = new THREE.Vector3();
  for (const { mesh, triangles } of meshes) {
    const index = mesh.geometry.index;
    const count = mesh.geometry.getAttribute("position").count;
    // Each vertex is posed once, however many triangles share it.
    const posed = new Float32Array(count * 3);
    const done = new Uint8Array(count);
    for (let t = 0; t < triangles; t += stride) {
      for (let corner = 0; corner < 3; corner++) {
        const i = index ? index.getX(t * 3 + corner) : t * 3 + corner;
        if (!done[i]) {
          mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
          posed[i * 3] = v.x;
          posed[i * 3 + 1] = v.y;
          posed[i * 3 + 2] = v.z;
          done[i] = 1;
        }
        out.push(posed[i * 3], posed[i * 3 + 1], posed[i * 3 + 2]);
      }
    }
  }
  return new Float32Array(out);
}

/** Cut `triangles` (nine numbers each) into `bands` equal slices of their
 * height, each as wide and deep as the triangles are within it: a triangle
 * is cut at a slice's edges, so a slanting limb widens a slice only as far
 * as it reaches there. A slice nothing passes through is left out. With
 * `span` (his height, as his bounds have it), the slices are of that, and
 * the lowest and highest reach its ends: a stride may have skipped his very
 * crown or soles, and they are his too. */
export function bandsOf(triangles: Float32Array, bands = BANDS, span?: { bottom: number; top: number }): Band[] {
  const n = Math.floor(triangles.length / 9);
  if (n === 0 || bands < 1) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n * 3; i++) {
    const y = triangles[i * 3 + 1];
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  if (!(hi >= lo)) return []; // a model with a corner that isn't a number
  if (span && span.top > span.bottom) {
    lo = Math.min(lo, span.bottom);
    hi = Math.max(hi, span.top);
  }
  const height = hi - lo;
  const count = height > 0 ? bands : 1;
  const slots: (Band | null)[] = new Array<Band | null>(count).fill(null);
  const slotOf = (y: number): number => (height > 0 ? Math.min(count - 1, Math.max(0, Math.floor(((y - lo) / height) * count))) : 0);
  const reach = (at: number, x: number, z: number): void => {
    if (Number.isNaN(x) || Number.isNaN(z)) return;
    const b = slots[at];
    if (!b) {
      slots[at] = { y0: lo + (at * height) / count, y1: lo + ((at + 1) * height) / count, x0: x, x1: x, z0: z, z1: z };
      return;
    }
    if (x < b.x0) b.x0 = x;
    if (x > b.x1) b.x1 = x;
    if (z < b.z0) b.z0 = z;
    if (z > b.z1) b.z1 = z;
  };
  for (let t = 0; t < n; t++) {
    const o = t * 9;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let c = 0; c < 3; c++) {
      const y = triangles[o + c * 3 + 1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    if (!(yMax >= yMin)) continue;
    const first = slotOf(yMin);
    const last = slotOf(yMax);
    for (let at = first; at <= last; at++) {
      const y0 = lo + (at * height) / count;
      const y1 = lo + ((at + 1) * height) / count;
      for (let c = 0; c < 3; c++) {
        const px = triangles[o + c * 3];
        const py = triangles[o + c * 3 + 1];
        const pz = triangles[o + c * 3 + 2];
        // A corner inside the slice (or the only slice there is).
        if (first === last || (py >= y0 && py <= y1)) reach(at, px, pz);
        // Where the edge to the next corner crosses the slice's edges.
        const d = ((c + 1) % 3) * 3;
        const qx = triangles[o + d];
        const qy = triangles[o + d + 1];
        const qz = triangles[o + d + 2];
        for (const y of [y0, y1]) {
          if ((py - y) * (qy - y) < 0) {
            const k = (y - py) / (qy - py);
            reach(at, px + (qx - px) * k, pz + (qz - pz) * k);
          }
        }
      }
    }
  }
  const out = slots.filter((b): b is Band => b !== null);
  if (out.length > 0) {
    out[0].y0 = lo;
    out[out.length - 1].y1 = hi;
  }
  return out;
}

const area = (b: Band): number => (b.x1 - b.x0) * (b.y1 - b.y0);

function union(a: Band, b: Band): Band {
  return {
    y0: Math.min(a.y0, b.y0),
    y1: Math.max(a.y1, b.y1),
    x0: Math.min(a.x0, b.x0),
    x1: Math.max(a.x1, b.x1),
    z0: Math.min(a.z0, b.z0),
    z1: Math.max(a.z1, b.z1),
  };
}

/** Merge neighbouring bands (lowest first) until `max` are left: each time,
 * the pair whose union adds the least empty area, so bands about as wide as
 * each other go together and his outline is kept where it changes most. */
export function mergeBands(bands: Band[], max = MAX_RECTS): Band[] {
  const out = bands.map((b) => ({ ...b }));
  while (out.length > Math.max(1, max)) {
    let best = 0;
    let least = Infinity;
    for (let i = 0; i + 1 < out.length; i++) {
      const added = area(union(out[i], out[i + 1])) - area(out[i]) - area(out[i + 1]);
      if (added < least) {
        least = added;
        best = i;
      }
    }
    out.splice(best, 2, union(out[best], out[best + 1]));
  }
  return out;
}

/** His shape at rest: `root`'s triangles as they are posed now, in bands
 * over `span`, merged to at most `MAX_RECTS`. */
export function restBands(root: THREE.Object3D, span: { bottom: number; top: number }, cap = SAMPLE_CAP): Band[] {
  return mergeBands(bandsOf(sampleTriangles(root, cap), BANDS, span));
}

/** The bands as `camera` sees them on a page `view` big: each band's eight
 * corners projected, the box round them padded by `pad` of the view's
 * shorter side, its edges rounded outward and kept inside the view. A band
 * with a corner at or behind the camera, or under a pixel either way, is
 * left out. The camera's matrices must be current. */
export function toView(bands: readonly Band[], camera: THREE.Camera, view: View, pad = PAD): HitRect[] {
  const near = (camera as Partial<THREE.PerspectiveCamera>).near ?? 0;
  const margin = pad * Math.min(view.width, view.height);
  const v = new THREE.Vector3();
  const out: HitRect[] = [];
  for (const b of bands) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    let seen = true;
    for (const x of [b.x0, b.x1]) {
      for (const y of [b.y0, b.y1]) {
        for (const z of [b.z0, b.z1]) {
          v.set(x, y, z).applyMatrix4(camera.matrixWorldInverse);
          // The camera looks down -Z: at or behind it, a corner has no place on the page.
          if (!(v.z < -near)) seen = false;
          v.applyMatrix4(camera.projectionMatrix);
          const px = ((v.x + 1) / 2) * view.width;
          const py = ((1 - v.y) / 2) * view.height;
          if (px < x0) x0 = px;
          if (px > x1) x1 = px;
          if (py < y0) y0 = py;
          if (py > y1) y1 = py;
        }
      }
    }
    if (!seen) continue;
    const left = Math.max(0, Math.floor(x0 - margin));
    const top = Math.max(0, Math.floor(y0 - margin));
    const right = Math.min(view.width, Math.ceil(x1 + margin));
    const bottom = Math.min(view.height, Math.ceil(y1 + margin));
    if (!(right - left >= 1) || !(bottom - top >= 1)) continue;
    out.push({ x: left, y: top, width: right - left, height: bottom - top });
  }
  return out;
}

/** The shape to offer: his, while he is drawn; none when he isn't (no
 * model yet, or the WebGL context lost, when nothing of him shows and a
 * click on where he was must go to what is there instead). */
export function shapeNow(drawn: boolean, rects: () => HitRect[]): HitRect[] {
  return drawn ? rects() : [];
}

/** Whether `b` is a new shape after `a`: the first, another view, another
 * number of boxes, or any edge moved by `min` pixels or more. */
export function moved(
  a: { view: View; rects: readonly HitRect[] } | null,
  b: { view: View; rects: readonly HitRect[] },
  min = MIN_MOVE,
): boolean {
  if (!a) return true;
  if (a.view.width !== b.view.width || a.view.height !== b.view.height) return true;
  if (a.rects.length !== b.rects.length) return true;
  return a.rects.some((r, i) => {
    const s = b.rects[i];
    return (
      Math.abs(r.x - s.x) >= min ||
      Math.abs(r.y - s.y) >= min ||
      Math.abs(r.x + r.width - (s.x + s.width)) >= min ||
      Math.abs(r.y + r.height - (s.y + s.height)) >= min
    );
  });
}

/** Sends his shape as it changes, and again every `KEEP_MS` while asked to
 * (the page's tick asks): one send at a time, the newest only. */
export class HitPublisher {
  private seq = 0;
  private last: { view: View; rects: HitRect[] } | null = null;
  private sentAt = -Infinity;
  private readonly push: (s: HitSend) => void;

  /** `send` takes one send to the shell; `now` is the page's clock, in ms. */
  constructor(
    send: (s: HitSend) => Promise<unknown>,
    private readonly now: () => number,
  ) {
    this.push = oneAtATime(send);
  }

  /** The shape now (from every fit): sent if it `moved`. */
  offer(rects: HitRect[], view: View): void {
    const next = { view: { ...view }, rects: rects.map((r) => ({ ...r })) };
    if (!moved(this.last, next)) return;
    this.last = next;
    this.go();
  }

  /** From the page's tick: the last shape again, once `KEEP_MS` has passed
   * since anything was sent. */
  keep(): void {
    if (this.last && this.now() - this.sentAt >= KEEP_MS) this.go();
  }

  /** The page starts drawing again (shown after a park, the display back
   * on): the next `keep` sends at once, since the shell may have counted
   * the shape as gone meanwhile. */
  resume(): void {
    this.sentAt = -Infinity;
  }

  private go(): void {
    if (!this.last) return;
    this.sentAt = this.now();
    this.seq += 1;
    this.push({ seq: this.seq, view: this.last.view, rects: this.last.rects });
  }
}

/** The page's sender: each send stamped with the number of this load of
 * the page, which the shell counts and is asked for once (`ask`; null when
 * it couldn't be had, and it is asked again with the next send). A send the
 * shell took for an older page's (`send` answers false) has it asked again
 * too: the page then goes by the shell's count, whatever went wrong. */
export function stamped(
  ask: () => Promise<number | null>,
  send: (page: number, s: HitSend) => Promise<boolean | null>,
): (s: HitSend) => Promise<void> {
  let page: Promise<number | null> | null = null;
  return async (s) => {
    page ??= ask();
    const n = await page;
    if (n === null) {
      page = null;
      return;
    }
    if ((await send(n, s)) === false) page = null;
  };
}
