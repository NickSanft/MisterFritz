/**
 * Models for the tests (never imported by the page): the committed
 * placeholder's bytes, and the same model rewritten as a VRM 0.x export.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const FILE = fileURLToPath(new URL("../../public/placeholder.vrm", import.meta.url));

export function placeholderBytes(): ArrayBuffer {
  const b = readFileSync(FILE);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

/** The placeholder's GLB rewritten as a VRM 0.x export would arrive: the
 * `VRM` extension instead of `VRMC_vrm`, and the model facing -Z (the whole
 * scene turned half round about Y), as VRoid's "VRM 0.0" export does. */
export function asVrm0(glb: ArrayBuffer): ArrayBuffer {
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
