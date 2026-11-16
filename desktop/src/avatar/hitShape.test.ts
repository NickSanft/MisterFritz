import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import fixture from "./wire.fixture.json";
import {
  BANDS, HitPublisher, KEEP_MS, MAX_RECTS, MIN_MOVE, PAD, bandsOf, mergeBands, moved, restBands, sampleTriangles, shapeNow,
  stamped, strideFor, toView, type Band, type HitRect, type HitSend,
} from "./hitShape";
import { HIT_RECTS_CMD, PAGE_CMD, hitArgs, parsePage } from "./hitWire";
import { HEADROOM, MARGIN, frameSpan } from "./framing";

const VIEW = { width: 340, height: 640 };

/** A camera framed as the page's `fit()` frames a figure of this height. */
function framed(bottom: number, top: number, view = VIEW): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(22, view.width / view.height, 0.05, 20);
  const f = frameSpan(bottom, top, 22, MARGIN, HEADROOM);
  camera.position.set(0, f.centerY, f.distance);
  camera.lookAt(0, f.centerY, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}

/** Points, as triangles with all three corners in one place. */
function points(...xyz: number[]): Float32Array {
  const out: number[] = [];
  for (let i = 0; i + 2 < xyz.length; i += 3) for (let c = 0; c < 3; c++) out.push(xyz[i], xyz[i + 1], xyz[i + 2]);
  return new Float32Array(out);
}

/** A flat upright face (two triangles) from (x0, y0) to (x1, y1), z deep. */
function face(x0: number, y0: number, x1: number, y1: number, z = 0): number[] {
  return [x0, y0, z, x1, y0, z, x1, y1, z, x0, y0, z, x1, y1, z, x0, y1, z];
}

/** A figure of long flat faces, with no vertex between a part's ends: two
 * legs, a wide chest and arms, a narrow head. 1.7 m tall. */
function figure(): Float32Array {
  return new Float32Array([
    ...face(-0.12, 0, -0.04, 0.8), ...face(0.04, 0, 0.12, 0.8, 0.05),
    ...face(-0.3, 0.8, 0.3, 1.4, 0.1), ...face(-0.3, 0.8, 0.3, 1.4, -0.1),
    ...face(-0.1, 1.4, 0.1, 1.7, 0.08), ...face(-0.1, 1.4, 0.1, 1.7, -0.08),
  ]);
}

const band = (y0: number, y1: number, x: number): Band => ({ y0, y1, x0: -x, x1: x, z0: -0.1, z1: 0.1 });
const rect = (x: number, y: number, width: number, height: number): HitRect => ({ x, y, width, height });

describe("his shape's bands", () => {
  it("cuts a figure into slices, each as wide as he is there, with no gap along a long face", () => {
    const b = bandsOf(figure());
    // His shins are single faces from foot to hip: every slice of them is his.
    expect(b).toHaveLength(BANDS);
    expect(b[0].y0).toBeCloseTo(0, 6);
    expect(b[b.length - 1].y1).toBeCloseTo(1.7, 6);
    // The slices tile his height with no gap.
    for (let i = 1; i < b.length; i++) expect(b[i].y0).toBeCloseTo(b[i - 1].y1, 6);
    const reach = (s: Band) => [s.x0, s.x1, s.z0, s.z1].map((v) => +v.toFixed(3));
    // Legs (the slices wholly below his hips), chest, head.
    for (const s of b.filter((s) => s.y1 < 0.8)) expect(reach(s)).toEqual([-0.12, 0.12, 0, 0.05]);
    expect(b.filter((s) => s.y1 < 0.8).length).toBeGreaterThanOrEqual(4);
    for (const s of b.filter((s) => s.y0 > 0.8 && s.y1 < 1.4)) expect(reach(s)).toEqual([-0.3, 0.3, -0.1, 0.1]);
    for (const s of b.filter((s) => s.y0 > 1.4)) expect(reach(s)).toEqual([-0.1, 0.1, -0.08, 0.08]);
    // A slice his hips cross holds both his legs and his chest.
    const hips = b.find((s) => s.y0 < 0.8 && s.y1 > 0.8)!;
    expect(reach(hips)).toEqual([-0.3, 0.3, -0.1, 0.1]);
  });

  it("cuts a slanting face at each slice's edges, so it widens a slice only as far as it reaches there", () => {
    // A line from (0, 0, 0) up and out to (1, 1, 2), as a triangle with two corners together.
    const b = bandsOf(new Float32Array([0, 0, 0, 1, 1, 2, 1, 1, 2]), 4);
    expect(b.map((s) => [s.x0, s.x1, s.z0, s.z1])).toEqual([
      [0, 0.25, 0, 0.5], [0.25, 0.5, 0.5, 1], [0.5, 0.75, 1, 1.5], [0.75, 1, 1.5, 2],
    ]);
    // A real triangle: an arm from his shoulder out to his hand.
    const arm = bandsOf(new Float32Array([0.2, 1.4, 0, 0.3, 1.4, 0, 0.5, 0.8, 0]), 6);
    expect(arm).toHaveLength(6);
    expect(arm[arm.length - 1].x0).toBeCloseTo(0.2, 6); // at his shoulder
    expect(arm[arm.length - 1].x1).toBeLessThan(0.35);
    expect(arm[0].x1).toBeCloseTo(0.5, 6); // at his hand
    expect(arm[0].x0).toBeGreaterThan(0.44);
    for (let i = 1; i < arm.length; i++) expect(arm[i].x1).toBeLessThan(arm[i - 1].x1);
  });

  it("leaves out a slice nothing passes through, and makes one of a single height", () => {
    const gap = bandsOf(points(0, 0, 0, 0.1, 0.01, 0, 0, 1.2, 0), 12);
    expect(gap).toHaveLength(2);
    expect(gap[0].y1).toBeLessThan(gap[1].y0);
    expect(bandsOf(points(1, 2, 3))).toEqual([{ y0: 2, y1: 2, x0: 1, x1: 1, z0: 3, z1: 3 }]);
    expect(bandsOf(points(-1, 2, 3, 1, 2, 5))).toEqual([{ y0: 2, y1: 2, x0: -1, x1: 1, z0: 3, z1: 5 }]);
    expect(bandsOf(new Float32Array())).toEqual([]);
    expect(bandsOf(points(1, 2, 3), 0)).toEqual([]);
  });

  it("is never spoiled by a corner that isn't a number", () => {
    expect(bandsOf(points(NaN, NaN, NaN))).toEqual([]);
    // A triangle with a bad corner, alone in the middle slice and met first:
    // nothing of it that isn't a number may start that slice.
    const bad = [NaN, 0.5, 0, 0.2, 0.5, NaN, 0.3, NaN, 0];
    const b = bandsOf(new Float32Array([...bad, ...points(0, 0, 0, 1, 1, 0)]), 3);
    expect(b).toHaveLength(3);
    for (const s of b) for (const v of Object.values(s)) expect(Number.isNaN(v)).toBe(false);
    expect([b[1].x0, b[1].x1, b[1].z0, b[1].z1].map((v) => +v.toFixed(3))).toEqual([0.3, 0.3, 0, 0]);
    expect(Math.max(...b.map((s) => s.x1))).toBe(1);
  });

  it("reaches his whole height when told it, whatever a stride skipped", () => {
    // The triangles stop short of his crown and soles.
    const pts = points(0, 0.3, 0, 0.1, 0.9, 0, -0.1, 1.5, 0);
    const b = bandsOf(pts, 12, { bottom: 0, top: 1.7 });
    expect(b[0].y0).toBe(0);
    expect(b[b.length - 1].y1).toBe(1.7);
    // A span they already pass changes nothing.
    const inside = bandsOf(pts, 12, { bottom: 0.5, top: 1 });
    expect(inside[0].y0).toBeCloseTo(0.3, 6);
    expect(inside[inside.length - 1].y1).toBeCloseTo(1.5, 6);
  });

  it("merges neighbours until six are left, keeping his outline", () => {
    const b = bandsOf(figure());
    const m = mergeBands(b);
    expect(m).toHaveLength(MAX_RECTS);
    expect(m[0].y0).toBe(b[0].y0);
    expect(m[m.length - 1].y1).toBe(b[b.length - 1].y1);
    expect(Math.min(...m.map((s) => s.x0))).toBeCloseTo(-0.3, 5);
    expect(Math.max(...m.map((s) => s.x1))).toBeCloseTo(0.3, 5);
    for (let i = 1; i < m.length; i++) expect(m[i].y0).toBeCloseTo(m[i - 1].y1, 6);
    // Bands as wide as each other go together first: his head stays its own
    // width, and is never merged into his chest while his legs are apart.
    const three = mergeBands([band(0, 1, 0.3), band(1, 2, 0.1), band(2, 3, 0.1), band(3, 4, 0.3)], 3);
    expect(three.map((s) => [s.y0, s.y1, s.x1])).toEqual([[0, 1, 0.3], [1, 3, 0.1], [3, 4, 0.3]]);
    expect(mergeBands(b, 1)).toHaveLength(1);
    expect(mergeBands(b, 0)).toHaveLength(1);
    expect(mergeBands(b.slice(0, 3))).toHaveLength(3);
    expect(mergeBands([])).toEqual([]);
    // The bands given are left as they were.
    expect(b).toHaveLength(BANDS);
  });
});

describe("his shape on the page", () => {
  it("lies inside the view, under the headroom, and round him", () => {
    const camera = framed(0, 1.7);
    const rects = toView(mergeBands(bandsOf(figure())), camera, VIEW);
    expect(rects).toHaveLength(MAX_RECTS);
    const pad = PAD * VIEW.width;
    for (const r of rects) {
      for (const v of [r.x, r.y, r.width, r.height]) expect(Number.isInteger(v)).toBe(true);
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(VIEW.height * HEADROOM - pad - 1);
      expect(r.x + r.width).toBeLessThanOrEqual(VIEW.width);
      expect(r.y + r.height).toBeLessThanOrEqual(VIEW.height);
    }
    // Lowest first in the world is lowest on the page; the page's y runs down.
    expect(rects[0].y).toBeGreaterThan(rects[rects.length - 1].y);
    // His chest is wider than his head, and he stands in the middle.
    const widest = Math.max(...rects.map((r) => r.width));
    expect(widest).toBeGreaterThan(rects[rects.length - 1].width);
    for (const r of rects) expect(Math.abs(r.x + r.width / 2 - VIEW.width / 2)).toBeLessThan(3);
    // Well under the window: the corners beside his head are not his.
    const area = rects.reduce((a, r) => a + r.width * r.height, 0);
    expect(area).toBeLessThan(VIEW.width * VIEW.height * 0.5);
  });

  it("pads each box, and rounds it outward", () => {
    const camera = framed(0, 1.7);
    const one = [band(0.5, 1, 0.1)];
    const [bare] = toView(one, camera, VIEW, 0);
    const [padded] = toView(one, camera, VIEW, 0.03);
    const pad = 0.03 * VIEW.width;
    expect(padded.x).toBeLessThanOrEqual(bare.x - Math.floor(pad));
    expect(padded.x + padded.width).toBeGreaterThanOrEqual(bare.x + bare.width + Math.floor(pad));
    expect(padded.height).toBeGreaterThanOrEqual(bare.height + 2 * Math.floor(pad));
    // The pad he is published with is the default.
    expect(toView(one, camera, VIEW)).toEqual([padded]);
    // Outward on every edge: a box seen at 100.4 to 199.6 is kept as 100 to 200.
    const xs: number[] = [];
    const ys: number[] = [];
    for (const x of [-0.1, 0.1]) for (const y of [0.5, 1]) for (const z of [-0.1, 0.1]) {
      const p = new THREE.Vector3(x, y, z).project(camera);
      xs.push(((p.x + 1) / 2) * VIEW.width);
      ys.push(((1 - p.y) / 2) * VIEW.height);
    }
    const left = Math.floor(Math.min(...xs));
    const top = Math.floor(Math.min(...ys));
    expect(bare).toEqual({
      x: left,
      y: top,
      width: Math.ceil(Math.max(...xs)) - left,
      height: Math.ceil(Math.max(...ys)) - top,
    });
    // A share of the view's shorter side: 3% of 400, not of 600.
    const wide = { width: 600, height: 400 };
    const cam = framed(0, 1.7, wide);
    expect(toView(one, cam, wide, 0)[0].x - toView(one, cam, wide, 0.03)[0].x).toBe(12);
  });

  it("keeps a box inside the view, and drops one behind the camera or under a pixel", () => {
    const camera = framed(0, 1.7);
    // Far wider than the view: clamped to it.
    const [wide] = toView([band(0.5, 1, 5)], camera, VIEW);
    expect([wide.x, wide.width]).toEqual([0, VIEW.width]);
    // Behind the camera (it stands on +Z, looking down -Z).
    const behind: Band = { y0: 0.5, y1: 1, x0: -0.1, x1: 0.1, z0: camera.position.z + 1, z1: camera.position.z + 2 };
    expect(toView([behind], camera, VIEW)).toEqual([]);
    // One corner behind it is enough: such a box has no honest place on the page.
    expect(toView([{ ...behind, z0: 0 }], camera, VIEW)).toEqual([]);
    // Wholly off the page to the right: nothing is left of it inside.
    const off: Band = { y0: 0.5, y1: 1, x0: 50, x1: 51, z0: 0, z1: 0 };
    expect(toView([off], camera, VIEW, 0)).toEqual([]);
    expect(toView([], camera, VIEW)).toEqual([]);
  });

  it("is none while he isn't drawn", () => {
    const rects = vi.fn(() => [rect(1, 2, 3, 4)]);
    expect(shapeNow(true, rects)).toEqual([rect(1, 2, 3, 4)]);
    expect(shapeNow(false, rects)).toEqual([]);
    expect(rects).toHaveBeenCalledTimes(1);
  });
});

describe("a model's triangles", () => {
  /** Two meshes: a tall box of 12 triangles, and hidden ones far off. */
  function scene(indexed = true): THREE.Group {
    const g = new THREE.Group();
    const box = new THREE.BoxGeometry(0.4, 1.6, 0.2);
    const body = new THREE.Mesh(indexed ? box : box.toNonIndexed());
    body.position.set(0, 0.8, 0);
    const hidden = new THREE.Mesh(new THREE.BoxGeometry(9, 9, 9));
    hidden.visible = false;
    const parent = new THREE.Group();
    parent.visible = false;
    parent.add(new THREE.Mesh(new THREE.BoxGeometry(9, 9, 9)));
    g.add(body, hidden, parent, new THREE.Object3D());
    return g;
  }
  const column = (t: Float32Array, c: number) => [...t].filter((_, i) => i % 3 === c);

  it("are read where they stand in the world, the visible meshes only", () => {
    for (const indexed of [true, false]) {
      const t = sampleTriangles(scene(indexed));
      expect(t.length, String(indexed)).toBe(12 * 9);
      expect(Math.min(...column(t, 1))).toBeCloseTo(0, 6);
      expect(Math.max(...column(t, 1))).toBeCloseTo(1.6, 6);
      expect(Math.max(...column(t, 0))).toBeCloseTo(0.2, 6);
      expect(Math.min(...column(t, 2))).toBeCloseTo(-0.1, 6);
    }
    // Every corner of the box is a corner of some triangle.
    const t = sampleTriangles(scene());
    const corners = new Set<string>();
    for (let i = 0; i < t.length; i += 3) corners.add([t[i], t[i + 1], t[i + 2]].map((v) => v.toFixed(2)).join());
    expect(corners.size).toBe(8);
  });

  it("are every nth past the cap, and his shape still reaches his whole height", () => {
    expect(strideFor(100, 200)).toBe(1);
    expect(strideFor(200, 200)).toBe(1);
    expect(strideFor(201, 200)).toBe(2);
    expect(strideFor(1000, 200)).toBe(5);
    expect(strideFor(0, 200)).toBe(1);
    expect(strideFor(10, 0)).toBe(10);
    const s = scene();
    // A cap of 5 of 12: every third triangle.
    expect(sampleTriangles(s, 5).length).toBe(4 * 9);
    const full = restBands(s, { bottom: 0, top: 1.6 });
    const sparse = restBands(s, { bottom: 0, top: 1.6 }, 2);
    expect(sparse[0].y0).toBe(full[0].y0);
    expect(sparse[sparse.length - 1].y1).toBe(full[full.length - 1].y1);
    // The whole box, at every height: its faces have no vertex between its ends.
    expect(full.length).toBeGreaterThanOrEqual(1);
    expect(full.length).toBeLessThanOrEqual(MAX_RECTS);
    for (const b of full) expect([b.x0, b.x1].map((v) => +v.toFixed(3))).toEqual([-0.2, 0.2]);
    for (let i = 1; i < full.length; i++) expect(full[i].y0).toBeCloseTo(full[i - 1].y1, 6);
  });

  /** One mesh of four triangles, whose second and fourth reach his crown
   * and soles: a stride of two reads only the first and third. */
  function spiky(): THREE.Mesh {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
      -0.1, 0.6, 0, 0.1, 0.6, 0, 0, 1.0, 0, // mid
      -0.1, 1.0, 0, 0.1, 1.0, 0, 0, 1.6, 0, // up to his crown
      -0.1, 0.7, 0, 0.1, 0.7, 0, 0, 0.9, 0, // mid
      -0.1, 0.6, 0, 0.1, 0.6, 0, 0, 0.0, 0, // down to his soles
    ]), 3));
    return new THREE.Mesh(g);
  }

  it("stop short of his crown and soles past the cap, and his shape doesn't", () => {
    const s = new THREE.Group().add(spiky());
    const t = sampleTriangles(s, 2);
    expect(t.length).toBe(2 * 9);
    expect(Math.min(...column(t, 1))).toBeCloseTo(0.6, 6);
    expect(Math.max(...column(t, 1))).toBeCloseTo(1.0, 6);
    const sparse = restBands(s, { bottom: 0, top: 1.6 }, 2);
    expect(sparse[0].y0).toBe(0);
    expect(sparse[sparse.length - 1].y1).toBe(1.6);
    // The cap is over all his meshes together, not each alone.
    expect(sampleTriangles(new THREE.Group().add(spiky(), spiky()), 4).length).toBe(4 * 9);
  });

  it("are his actor's, read once over his bounds", () => {
    const actor = readFileSync(fileURLToPath(new URL("./VrmActor.ts", import.meta.url)), "utf8");
    expect(actor).toContain("this.rest = restBands(vrm.scene, this.bounds());");
  });
});

describe("a new shape", () => {
  const a = { view: VIEW, rects: [rect(100, 200, 140, 400)] };
  it("is the first, another view, another count, or an edge moved two pixels", () => {
    expect(moved(null, a)).toBe(true);
    expect(moved(a, { view: VIEW, rects: [rect(100, 200, 140, 400)] })).toBe(false);
    expect(moved(a, { view: VIEW, rects: [rect(101.9, 200, 140, 400)] })).toBe(false);
    expect(moved(a, { view: VIEW, rects: [rect(102, 200, 138, 400)] })).toBe(true); // its left edge
    expect(moved(a, { view: VIEW, rects: [rect(100, 198, 140, 402)] })).toBe(true); // its top
    expect(moved(a, { view: VIEW, rects: [rect(100, 200, 142, 400)] })).toBe(true); // its right
    expect(moved(a, { view: VIEW, rects: [rect(100, 200, 140, 398)] })).toBe(true); // its bottom
    expect(moved(a, { view: VIEW, rects: [rect(101, 201, 139, 399)] })).toBe(false); // every edge by one
    expect(moved(a, { view: { width: 341, height: 640 }, rects: a.rects })).toBe(true);
    expect(moved(a, { view: { width: 340, height: 641 }, rects: a.rects })).toBe(true);
    expect(moved(a, { view: VIEW, rects: [] })).toBe(true);
    expect(moved({ view: VIEW, rects: [] }, { view: VIEW, rects: [] })).toBe(false);
    expect(MIN_MOVE).toBe(fixture.hitShape.minMove);
  });
});

/** A publisher over sends the test settles by hand, and a clock it moves. */
function publisher() {
  const sent: HitSend[] = [];
  const settle: (() => void)[] = [];
  let t = 0;
  const hits = new HitPublisher((s) => {
    sent.push(s);
    return new Promise<void>((r) => settle.push(r));
  }, () => t);
  const flush = async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  };
  return { hits, sent, settle, flush, at: (ms: number) => { t = ms; } };
}

describe("the page's sends of his shape", () => {
  const body = [rect(100, 200, 140, 400)];

  it("go when the shape moves, one at a time, the newest only", async () => {
    const { hits, sent, settle, flush } = publisher();
    hits.offer(body, VIEW);
    await flush();
    expect(sent).toEqual([{ seq: 1, view: VIEW, rects: body }]);
    hits.offer([rect(101, 200, 140, 400)], VIEW); // a pixel: nothing
    hits.offer([rect(103, 200, 140, 400)], VIEW);
    hits.offer([rect(106, 200, 140, 400)], VIEW);
    await flush();
    expect(sent).toHaveLength(1); // the first is still on its way
    settle[0]();
    await flush();
    expect(sent).toHaveLength(2);
    expect(sent[1].rects).toEqual([rect(106, 200, 140, 400)]);
    expect(sent[1].seq).toBeGreaterThan(sent[0].seq);
    settle[1]();
    await flush();
    expect(sent).toHaveLength(2);
  });

  it("number every send higher than the last, whatever was overtaken", async () => {
    const { hits, sent, settle, flush } = publisher();
    for (let i = 0; i < 4; i++) {
      hits.offer([rect(100 + i * 10, 200, 140, 400)], VIEW);
      await flush();
      settle[i]?.();
      await flush();
    }
    const seqs = sent.map((s) => s.seq);
    expect(seqs).toEqual([...seqs].sort((a, b) => a - b));
    expect(new Set(seqs).size).toBe(seqs.length);
    expect(seqs[0]).toBe(1);
  });

  it("send what was offered, not what the caller does with it afterwards", async () => {
    const { hits, sent, flush } = publisher();
    const mine = [rect(100, 200, 140, 400)];
    const view = { ...VIEW };
    hits.offer(mine, view);
    mine[0].x = 999;
    view.width = 1;
    await flush();
    expect(sent[0]).toEqual({ seq: 1, view: VIEW, rects: body });
  });

  it("send the shape again every five seconds while the tick asks, and at once after a pause", async () => {
    const { hits, sent, settle, flush, at } = publisher();
    hits.keep(); // nothing to keep yet
    await flush();
    expect(sent).toHaveLength(0);
    hits.offer(body, VIEW);
    await flush();
    settle[0]();
    at(KEEP_MS - 1);
    hits.keep();
    await flush();
    expect(sent).toHaveLength(1);
    at(KEEP_MS);
    hits.keep();
    await flush();
    expect(sent).toHaveLength(2);
    expect(sent[1]).toEqual({ seq: 2, view: VIEW, rects: body });
    settle[1]();
    // A send for a moved shape counts as the keep-alive too.
    at(KEEP_MS + 3000);
    hits.offer([rect(110, 200, 140, 400)], VIEW);
    await flush();
    settle[2]();
    at(2 * KEEP_MS + 2999);
    hits.keep();
    await flush();
    expect(sent).toHaveLength(3);
    // Drawing again after a park: the next tick sends at once.
    hits.resume();
    hits.keep();
    await flush();
    expect(sent).toHaveLength(4);
    expect(KEEP_MS).toBe(fixture.hitShape.keepMs);
    expect(KEEP_MS * 3).toBe(fixture.hitShape.staleMs);
  });

  it("keep sending after a send that failed, and send no shape when he isn't drawn", async () => {
    const sent: HitSend[] = [];
    let fail = true;
    const hits = new HitPublisher(async (s) => {
      sent.push(s);
      if (fail) throw new Error("the shell didn't answer");
    }, () => 0);
    hits.offer(body, VIEW);
    for (let i = 0; i < 5; i++) await Promise.resolve();
    fail = false;
    hits.offer([], VIEW); // the context lost: no shape, said at once
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(sent.map((s) => s.rects.length)).toEqual([1, 0]);
  });
});

/** The page's own wiring, read from its source (main.ts runs only in his
 * window): the shape is offered at every fit, kept alive from the paced
 * tick, sent at once when drawing starts again, and none while the WebGL
 * context is lost. */
describe("the page's use of his shape", () => {
  const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8").replace(/\r\n/g, "\n");
  const body = (start: string, end = "\n}\n") => {
    const at = main.indexOf(start);
    expect(at, start).toBeGreaterThanOrEqual(0);
    return main.slice(at, main.indexOf(end, at));
  };

  it("offers it at the end of every fit", () => {
    expect(body("function fit(): void {").trimEnd().endsWith("offerShape();")).toBe(true);
    const offer = body("function offerShape(): void {");
    expect(offer).toContain("hits.offer(shapeNow(a !== null && !contextLost, () => a!.hitRects(camera, w, h)), { width: w, height: h });");
  });

  it("keeps it alive from the paced tick, and nowhere else", () => {
    expect(body("const loop = new PacedLoop((dt) => {", "\n}, FPS);")).toContain("renderer.render(scene, camera);\n  hits.keep();");
    expect(main.split("hits.keep()").length - 1).toBe(1);
  });

  it("sends it at once when drawing starts again", () => {
    expect(body("function draw(): void {")).toContain("hits.resume();\n    loop.start();");
  });

  it("offers none while the context is lost, and fits again once it is back", () => {
    expect(main).toContain('canvas.addEventListener("webglcontextlost", () => {\n  contextLost = true;\n  offerShape();\n});');
    expect(main).toContain('canvas.addEventListener("webglcontextrestored", () => {\n  contextLost = false;\n  fit();\n});');
  });

  it("stamps its sends with the shell's count of its loads", () => {
    expect(main).toContain("stamped(fetchPage, (page, s) => sendHitRects(hitArgs(instance, page, s)))");
  });
});

describe("the page's number", () => {
  it("is asked once, and stamps every send", async () => {
    const ask = vi.fn(async () => 4);
    const send = vi.fn(async (_page: number, _s: HitSend): Promise<boolean | null> => true);
    const go = stamped(ask, send);
    const s1: HitSend = { seq: 1, view: VIEW, rects: [] };
    const s2: HitSend = { seq: 2, view: VIEW, rects: [] };
    await go(s1);
    await go(s2);
    expect(ask).toHaveBeenCalledTimes(1);
    expect(send.mock.calls).toEqual([[4, s1], [4, s2]]);
  });

  it("is asked again after a send the shell took for an older page's, and when it couldn't be had", async () => {
    const answers: (number | null)[] = [null, 4, 5];
    const ask = vi.fn(async () => answers.shift() ?? null);
    const kept: (boolean | null)[] = [false, null, true];
    // Answered by position: a null in the list is the shell's null, not "true".
    const send = vi.fn(async (_page: number, _s: HitSend): Promise<boolean | null> => {
      const r = kept.shift();
      return r === undefined ? true : r;
    });
    const go = stamped(ask, send);
    const s: HitSend = { seq: 1, view: VIEW, rects: [] };
    await go(s); // couldn't ask: nothing is sent
    expect(send).not.toHaveBeenCalled();
    await go(s); // asked again: 4, which the shell takes for an older page's
    await go(s); // asked again: 5, refused outright (not a reason to ask again)
    await go(s); // still 5
    expect(ask).toHaveBeenCalledTimes(3);
    expect(send.mock.calls.map((c) => c[0])).toEqual([4, 5, 5]);
  });
});

describe("his shape's wire", () => {
  it("calls the commands the shell registers, with the arguments it takes", () => {
    const c = fixture.commands;
    expect(HIT_RECTS_CMD).toBe(c.setHitRects.name);
    expect(PAGE_CMD).toBe(c.avatarPage.name);
    expect(c.avatarPage.args).toEqual([]);
    const s: HitSend = { seq: 3, view: VIEW, rects: [rect(1, 2, 3, 4)] };
    const args = hitArgs(1, 2, s);
    expect(Object.keys(args)).toEqual(c.setHitRects.args);
    expect(args).toEqual({ instance: 1, page: 2, seq: 3, view: VIEW, rects: [rect(1, 2, 3, 4)] });
    // The fixture's sample is one such send.
    expect(Object.keys(fixture.hitRectsSample)).toEqual(c.setHitRects.args);
    expect(Object.keys(fixture.hitRectsSample.rects[0]).sort()).toEqual(["height", "width", "x", "y"]);
  });

  it("agrees with the shell on the numbers", () => {
    expect(MAX_RECTS).toBe(fixture.hitShape.maxRects);
    expect(BANDS).toBe(fixture.hitShape.bands);
    expect(fixture.hitRectsSample.rects.length).toBeLessThanOrEqual(MAX_RECTS);
  });

  it("reads the page's number strictly", () => {
    expect(parsePage(1)).toBe(1);
    expect(parsePage(7)).toBe(7);
    for (const bad of [0, -1, 1.5, "1", null, undefined, NaN, {}]) expect(parsePage(bad)).toBeNull();
  });
});
