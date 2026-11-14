/**
 * The theme on his body, on the placeholder (MToon, outlined) and its 0.x
 * rewrite (no MToon left): the tint leans his outlines and lights his rim
 * from the model's own colours, and the motion factor stills all but the
 * blink and the breathing.
 */
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { VRMExpression, VRMExpressionMaterialColorBind } from "@pixiv/three-vrm";
import { OUTLINE_MIX, RIM_FRESNEL, VrmActor } from "./VrmActor";
import { asVrm0, placeholderBytes } from "./testModels";

interface Toon extends THREE.Material {
  isMToonMaterial: true;
  isOutline: boolean;
  outlineColorFactor: THREE.Color;
  parametricRimColorFactor: THREE.Color;
  parametricRimFresnelPowerFactor: number;
  parametricRimLiftFactor: number;
}

function toons(actor: VrmActor): Toon[] {
  const out = new Set<Toon>();
  actor.vrm.scene.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
    for (const x of m ? (Array.isArray(m) ? m : [m]) : []) {
      if ((x as Partial<Toon>).isMToonMaterial) out.add(x as Toon);
    }
  });
  return [...out];
}

function head(actor: VrmActor): THREE.Vector3 {
  const h = actor.vrm.humanoid.getRawBoneNode("head")!;
  return new THREE.Vector3(0, 0, 1).applyQuaternion(h.getWorldQuaternion(new THREE.Quaternion()));
}

function run(actor: VrmActor, seconds: number, each?: () => void): void {
  for (let i = 0; i < seconds * 30; i++) {
    actor.update(1 / 30);
    each?.();
  }
}

describe("his tint", () => {
  it("leans every MToon outline toward the theme's colour, from his own", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const mats = toons(actor);
    expect(mats.length).toBeGreaterThan(0);
    const own = mats.map((m) => ({ outline: m.outlineColorFactor.clone(), rim: m.parametricRimColorFactor.clone() }));
    actor.setTint("#75694f", "#6D5BF0", 0.25);
    const toward = new THREE.Color("#75694f");
    const glow = new THREE.Color("#6D5BF0").multiplyScalar(0.25);
    mats.forEach((m, i) => {
      const want = own[i].outline.clone().lerp(toward, OUTLINE_MIX);
      expect(m.outlineColorFactor.r).toBeCloseTo(want.r, 6);
      expect(m.outlineColorFactor.g).toBeCloseTo(want.g, 6);
      expect(m.outlineColorFactor.b).toBeCloseTo(want.b, 6);
      expect(m.parametricRimColorFactor.b).toBeCloseTo(own[i].rim.b + glow.b, 6);
    });
  });

  it("never compounds: the same tint twice is the tint once, and another replaces it", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const m = toons(actor)[0];
    actor.setTint("#5d6178", "#4ABDE0", 0.25);
    const once = m.outlineColorFactor.clone();
    const rimOnce = m.parametricRimColorFactor.clone();
    actor.setTint("#5d6178", "#4ABDE0", 0.25);
    expect(m.outlineColorFactor.equals(once)).toBe(true);
    expect(m.parametricRimColorFactor.equals(rimOnce)).toBe(true);
    actor.setTint("#837560", "#F2A857", 0.15);
    const fresh = await VrmActor.load(placeholderBytes(), () => 0.5);
    fresh.setTint("#837560", "#F2A857", 0.15);
    expect(m.outlineColorFactor.equals(toons(fresh)[0].outlineColorFactor)).toBe(true);
  });

  it("gives a rim to a model whose own is off, and leaves a model's own rim its shape", async () => {
    // A VRoid 0.x export: rim black, falloff 1, as three-vrm reads one that
    // sets none of the _Rim properties.
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const [off, lit] = toons(actor);
    off.parametricRimColorFactor.setRGB(0, 0, 0);
    off.parametricRimFresnelPowerFactor = 1;
    off.parametricRimLiftFactor = 0.2;
    lit.parametricRimColorFactor.setRGB(0.1, 0, 0);
    lit.parametricRimFresnelPowerFactor = 2;
    lit.parametricRimLiftFactor = 0.1;
    actor.setTint("#75694f", "#6D5BF0", 0.25);
    expect(off.parametricRimFresnelPowerFactor).toBe(RIM_FRESNEL);
    expect(off.parametricRimLiftFactor).toBe(0);
    expect(lit.parametricRimFresnelPowerFactor).toBe(2);
    expect(lit.parametricRimLiftFactor).toBe(0.1);
    // At 45 degrees from the view, the accent stays a whisper, not a wash.
    const at45 = (1 - Math.SQRT1_2) ** off.parametricRimFresnelPowerFactor;
    expect(off.parametricRimColorFactor.b * at45).toBeLessThan(0.005);
  });

  it("keeps the tint on colours an expression owns, which three-vrm resets every frame", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const [rimmed, outlined] = toons(actor);
    const expression = new VRMExpression("tinted");
    expression.addBind(new VRMExpressionMaterialColorBind({ material: rimmed, type: "rimColor", targetValue: new THREE.Color(1, 0, 0) }));
    expression.addBind(new VRMExpressionMaterialColorBind({ material: outlined, type: "outlineColor", targetValue: new THREE.Color(0, 1, 0) }));
    actor.vrm.expressionManager!.registerExpression(expression);
    const ownRim = rimmed.parametricRimColorFactor.clone();
    const ownOutline = outlined.outlineColorFactor.clone();
    actor.setTint("#5d6178", "#4ABDE0", 0.25);
    run(actor, 0.5);
    const glow = new THREE.Color("#4ABDE0").multiplyScalar(0.25);
    expect(rimmed.parametricRimColorFactor.b).toBeCloseTo(ownRim.b + glow.b, 6);
    const want = ownOutline.clone().lerp(new THREE.Color("#5d6178"), OUTLINE_MIX);
    expect(outlined.outlineColorFactor.r).toBeCloseTo(want.r, 6);
  });

  it("leaves a model without MToon alone", async () => {
    const actor = await VrmActor.load(asVrm0(placeholderBytes()), () => 0.5);
    expect(toons(actor)).toEqual([]);
    expect(() => actor.setTint("#75694f", "#6D5BF0", 0.25)).not.toThrow();
  });
});

describe("his motion", () => {
  it("at 0 he doesn't sway, and looks ahead whatever he is told", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setMotion(0);
    actor.setLook({ yaw: 30, pitch: 15 });
    const hips = actor.vrm.humanoid.getNormalizedBoneNode("hips")!;
    run(actor, 10, () => expect(hips.rotation.y).toBe(0));
    expect(Math.abs(actor.gaze.yaw)).toBeLessThan(0.01);
    expect(Math.abs(head(actor).x)).toBeLessThan(0.05);
  });

  it("at 0 he still breathes and blinks", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setMotion(0);
    const chest = actor.vrm.humanoid.getNormalizedBoneNode("upperChest") ?? actor.vrm.humanoid.getNormalizedBoneNode("chest")!;
    const seen = new Set<number>();
    let blinked = false;
    run(actor, 8, () => {
      seen.add(Math.round(chest.rotation.x * 1e5));
      if ((actor.vrm.expressionManager?.getValue("blink") ?? 0) > 0.5) blinked = true;
    });
    expect(seen.size).toBeGreaterThan(10);
    expect(blinked).toBe(true);
  });

  it("scales his sway: ambient is 0.6 of rich", async () => {
    const sway = async (factor: number) => {
      const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
      actor.setMotion(factor);
      const hips = actor.vrm.humanoid.getNormalizedBoneNode("hips")!;
      let most = 0;
      run(actor, 12, () => (most = Math.max(most, Math.abs(hips.rotation.y))));
      return most;
    };
    const rich = await sway(1);
    expect(rich).toBeGreaterThan(0.01);
    expect(await sway(0.6)).toBeCloseTo(rich * 0.6, 6);
  });

  it("follows the gaze again once motion returns", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setMotion(0);
    actor.setLook({ yaw: 30, pitch: 0 });
    run(actor, 1);
    actor.setMotion(0.6);
    run(actor, 2);
    expect(actor.gaze.yaw).toBeGreaterThan(25);
  });
});
