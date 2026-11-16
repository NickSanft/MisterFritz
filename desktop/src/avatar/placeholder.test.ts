/**
 * The committed placeholder (public/placeholder.vrm, written by
 * scripts/make_placeholder_vrm.py) loads in three-vrm with everything the
 * body drives, so a fresh checkout and CI always have a working model.
 */
import { describe, expect, it } from "vitest";
import { VrmActor } from "./VrmActor";
import { asVrm0, placeholderBytes as bytes } from "./testModels";

describe("the placeholder VRM", () => {
  it("is a VRM 1.0 with a full humanoid", async () => {
    const actor = await VrmActor.load(bytes());
    expect(actor.vrm.meta.metaVersion).toBe("1");
    for (const bone of ["hips", "spine", "chest", "upperChest", "neck", "head", "leftUpperArm", "rightUpperArm",
      "leftLowerArm", "rightLowerArm", "leftHand", "rightHand", "leftUpperLeg", "rightUpperLeg", "leftLowerLeg",
      "rightLowerLeg", "leftFoot", "rightFoot"] as const) {
      expect(actor.vrm.humanoid.getRawBoneNode(bone), bone).not.toBeNull();
    }
  });

  it("has the expressions the body drives", async () => {
    const actor = await VrmActor.load(bytes());
    const em = actor.vrm.expressionManager!;
    for (const name of ["aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight", "happy", "angry", "sad",
      "relaxed", "surprised"]) {
      expect(em.getExpression(name), name).not.toBeNull();
    }
  });

  it("has spring bones, so the idle exercises them", async () => {
    const actor = await VrmActor.load(bytes());
    // The tie's three links and two on each coat tail (a chain's tip is not a joint).
    expect(actor.vrm.springBoneManager?.joints.size ?? 0).toBe(7);
  });

  it("is about a person's height, standing on the ground", async () => {
    const { bottom, top } = (await VrmActor.load(bytes())).bounds();
    expect(bottom).toBeGreaterThan(-0.05);
    expect(bottom).toBeLessThan(0.05);
    expect(top).toBeGreaterThan(1.6);
    expect(top).toBeLessThan(1.9);
  });

  it("blinks within six seconds of idling, and opens its eyes again", async () => {
    let r = 0.999; // the longest gap
    const actor = await VrmActor.load(bytes(), () => r);
    const em = actor.vrm.expressionManager!;
    let peak = 0;
    for (let i = 0; i < 6.3 * 30; i++) {
      actor.update(1 / 30);
      peak = Math.max(peak, em.getValue("blink") ?? 0);
    }
    expect(peak).toBeGreaterThan(0.5);
    r = 0.999;
    for (let i = 0; i < 10; i++) actor.update(1 / 30);
    expect(em.getValue("blink")).toBe(0);
  });

  it("stands with its arms down, not in the T-pose it is stored in", async () => {
    const actor = await VrmActor.load(bytes());
    actor.update(1 / 30);
    const hand = actor.vrm.humanoid.getRawBoneNode("leftHand")!;
    const shoulder = actor.vrm.humanoid.getRawBoneNode("leftUpperArm")!;
    const h = hand.getWorldPosition(new (await import("three")).Vector3());
    const s = shoulder.getWorldPosition(new (await import("three")).Vector3());
    expect(s.y - h.y).toBeGreaterThan(0.3);
  });
});

/** His shape on a page of this size, with the camera as the page's `fit()`
 * frames him, and where his head and his whole height come on that page. */
async function shapeOn(data: ArrayBuffer, width: number, height: number) {
  const THREE = await import("three");
  const { HEADROOM, MARGIN, frameSpan } = await import("./framing");
  const actor = await VrmActor.load(data);
  const { bottom, top } = actor.bounds();
  const camera = new THREE.PerspectiveCamera(22, width / height, 0.05, 20);
  const f = frameSpan(bottom, top, 22, MARGIN, HEADROOM);
  // Moved only (from there it looks straight down -Z at him, as fit()'s
  // lookAt leaves it), and nothing has drawn since: its matrices still say
  // it stands at the origin, and hitRects must bring them up to date.
  camera.position.set(0, f.centerY, f.distance);
  camera.updateProjectionMatrix();
  const rects = actor.hitRects(camera, width, height);
  const y = (world: number) => ((1 - new THREE.Vector3(0, world, 0).project(camera).y) / 2) * height;
  return { rects, head: actor.headScreen(camera, width, height), crown: y(top), soles: y(bottom), headroom: height * HEADROOM };
}

describe("the placeholder's shape", () => {
  const inside = (r: { x: number; y: number; width: number; height: number }, p: { x: number; y: number }) =>
    p.x >= r.x && p.x < r.x + r.width && p.y >= r.y && p.y < r.y + r.height;

  it("is a few boxes round him: his head and his feet in it, the corners beside his head not", async () => {
    for (const [name, data] of [["1.0", bytes()], ["0.x", asVrm0(bytes())]] as const) {
      const { rects, head, crown, soles, headroom } = await shapeOn(data, 340, 640);
      expect(rects.length, name).toBeGreaterThanOrEqual(2);
      expect(rects.length, name).toBeLessThanOrEqual(6);
      expect(rects.some((r) => inside(r, head)), `${name}: his head`).toBe(true);
      expect(rects.some((r) => inside(r, { x: 170, y: soles - 6 })), `${name}: his feet`).toBe(true);
      // From his crown to his soles, with the padding and little more (the
      // front of his head, nearer the camera, is seen a few pixels higher).
      const pad = 0.03 * 340;
      const top = Math.min(...rects.map((r) => r.y));
      const bottom = Math.max(...rects.map((r) => r.y + r.height));
      expect(top, name).toBeLessThanOrEqual(crown);
      expect(top, `${name}: padded`).toBeLessThanOrEqual(crown - Math.floor(pad));
      expect(top, name).toBeGreaterThanOrEqual(crown - pad - 8);
      expect(bottom, name).toBeGreaterThanOrEqual(Math.min(640, soles));
      expect(top, `${name}: under the bubble's room`).toBeGreaterThanOrEqual(headroom - pad - 2);
      // Beside his head (level with it, a third of the way in) is not his.
      expect(rects.some((r) => inside(r, { x: 340 * 0.12, y: head.y })), `${name}: beside his head`).toBe(false);
      expect(rects.some((r) => inside(r, { x: 340 * 0.88, y: head.y })), `${name}: beside his head`).toBe(false);
      // Nor is the room over his head, where his bubble shows.
      expect(rects.some((r) => inside(r, { x: 170, y: headroom / 2 })), `${name}: over his head`).toBe(false);
      const area = rects.reduce((a, r) => a + r.width * r.height, 0);
      expect(area / (340 * 640), name).toBeLessThan(0.5);
      // Measured standing, his arms down: in the T-pose the file is stored
      // in, his arms would span the page.
      expect(Math.max(...rects.map((r) => r.width)) / 340, `${name}: his arms are down`).toBeLessThan(0.6);
    }
  });

  it("follows the page's size", async () => {
    for (const [w, h] of [[200, 640], [600, 400], [680, 1280]] as const) {
      const { rects, head } = await shapeOn(bytes(), w, h);
      expect(rects.length, `${w}x${h}`).toBeGreaterThanOrEqual(1);
      for (const r of rects) {
        expect(r.x).toBeGreaterThanOrEqual(0);
        expect(r.y).toBeGreaterThanOrEqual(0);
        expect(r.x + r.width).toBeLessThanOrEqual(w);
        expect(r.y + r.height).toBeLessThanOrEqual(h);
      }
      expect(rects.some((r) => inside(r, head)), `${w}x${h}: his head`).toBe(true);
    }
  });
});

describe("a VRM 0.x export", () => {
  it("stands with its arms down too, not raised", async () => {
    const THREE = await import("three");
    const actor = await VrmActor.load(asVrm0(bytes()));
    expect(actor.vrm.meta.metaVersion).toBe("0");
    actor.update(1 / 30);
    for (const side of ["left", "right"] as const) {
      const hand = actor.vrm.humanoid.getRawBoneNode(`${side}Hand`)!.getWorldPosition(new THREE.Vector3());
      const shoulder = actor.vrm.humanoid.getRawBoneNode(`${side}UpperArm`)!.getWorldPosition(new THREE.Vector3());
      expect(shoulder.y - hand.y, side).toBeGreaterThan(0.3);
    }
  });
});

describe("the placeholder's licence", () => {
  it("claims nothing CC0 can't carry: every VRM permission is the most permissive", () => {
    const b = bytes();
    const json = JSON.parse(new TextDecoder().decode(new Uint8Array(b, 20, new DataView(b).getUint32(12, true))));
    const meta = json.extensions.VRMC_vrm.meta;
    expect(meta.otherLicenseUrl).toContain("publicdomain/zero");
    for (const [field, value] of Object.entries(meta)) {
      if (field.startsWith("allow")) expect(value, field).toBe(true);
    }
    expect(meta.avatarPermission).toBe("everyone");
    expect(meta.commercialUsage).toBe("corporation");
    expect(meta.creditNotation).toBe("unnecessary");
    expect(meta.modification).toBe("allowModificationRedistribution");
  });
});
