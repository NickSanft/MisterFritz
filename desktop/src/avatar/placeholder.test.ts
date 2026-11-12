/**
 * The committed placeholder (public/placeholder.vrm, written by
 * scripts/make_placeholder_vrm.py) loads in three-vrm with everything the
 * body drives, so a fresh checkout and CI always have a working model.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { VrmActor } from "./VrmActor";

const FILE = fileURLToPath(new URL("../../public/placeholder.vrm", import.meta.url));

function bytes(): ArrayBuffer {
  const b = readFileSync(FILE);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

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

/** The placeholder's GLB rewritten as a VRM 0.x export would arrive: the
 * `VRM` extension instead of `VRMC_vrm`, and the model facing -Z (the whole
 * scene turned half round about Y), as VRoid's "VRM 0.0" export does. */
function asVrm0(glb: ArrayBuffer): ArrayBuffer {
  const view = new DataView(glb);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(glb, 20, jsonLength)));
  const binAt = 20 + jsonLength;
  const bin = new Uint8Array(glb, binAt + 8, view.getUint32(binAt, true));
  const humanBones = json.extensions.VRMC_vrm.humanoid.humanBones as Record<string, { node: number }>;
  json.extensions = {
    VRM: {
      specVersion: "0.0",
      meta: { title: "Fritz (placeholder as VRM 0.x)", version: "1", author: "test", licenseName: "CC0" },
      humanoid: { humanBones: Object.entries(humanBones).map(([bone, b]) => ({ bone, node: b.node, useDefaultValues: true })) },
      blendShapeMaster: { blendShapeGroups: [] },
      firstPerson: { firstPersonBone: humanBones.head.node, meshAnnotations: [], lookAtTypeName: "BlendShape" },
      secondaryAnimation: { boneGroups: [], colliderGroups: [] },
      materialProperties: [],
    },
  };
  json.extensionsUsed = ["VRM"];
  for (const m of json.materials) delete m.extensions;
  json.nodes.push({ name: "Vrm0Root", rotation: [0, 1, 0, 0], children: json.scenes[0].nodes });
  json.scenes[0].nodes = [json.nodes.length - 1];

  const text = new TextEncoder().encode(JSON.stringify(json));
  const textPad = (4 - (text.length % 4)) % 4;
  const binPad = (4 - (bin.length % 4)) % 4;
  const total = 12 + 8 + text.length + textPad + 8 + bin.length + binPad;
  const out = new Uint8Array(total);
  const o = new DataView(out.buffer);
  o.setUint32(0, 0x46546c67, true);
  o.setUint32(4, 2, true);
  o.setUint32(8, total, true);
  o.setUint32(12, text.length + textPad, true);
  o.setUint32(16, 0x4e4f534a, true);
  out.set(text, 20);
  out.fill(0x20, 20 + text.length, 20 + text.length + textPad);
  const at = 20 + text.length + textPad;
  o.setUint32(at, bin.length + binPad, true);
  o.setUint32(at + 4, 0x004e4942, true);
  out.set(bin, at + 8);
  return out.buffer;
}

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
