/**
 * The look on a real model: the placeholder, which has no eyes of its own,
 * so his head's share is what shows; and three-vrm's look-at, which a VRoid
 * export's eyes follow. In a 1.0 rig and a 0.x rig, whose X axis runs the
 * other way.
 */
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { AHEAD, HEAD_SHARE_YAW, type Gaze } from "./look";
import { GLANCE, GLANCE_S, THINK_TILT_DEG, VrmActor } from "./VrmActor";
import { asVrm0, placeholderBytes } from "./testModels";

/** Where his head points, in the world: +X is the page's right, +Y up. */
function facing(actor: VrmActor): THREE.Vector3 {
  const head = actor.vrm.humanoid.getRawBoneNode("head")!;
  return new THREE.Vector3(0, 0, 1).applyQuaternion(head.getWorldQuaternion(new THREE.Quaternion()));
}

function settle(actor: VrmActor, g: Gaze, seconds = 2): void {
  actor.setLook(g);
  for (let i = 0; i < seconds * 30; i++) actor.update(1 / 30);
}

for (const [rig, load] of [
  ["a VRM 1.0 rig", () => VrmActor.load(placeholderBytes(), () => 0.5)],
  ["a VRM 0.x rig", () => VrmActor.load(asVrm0(placeholderBytes()), () => 0.5)],
] as const) {
  describe(`looking, in ${rig}`, () => {
    it("starts looking ahead", async () => {
      const actor = await load();
      actor.update(1 / 30);
      const f = facing(actor);
      expect(Math.abs(f.x)).toBeLessThan(0.05);
      expect(f.z).toBeGreaterThan(0.99);
    });

    it("turns his head toward the gaze: right and up, then left and down", async () => {
      const actor = await load();
      settle(actor, { yaw: 30, pitch: 15 });
      let f = facing(actor);
      // 40% of 30° across is 12°: sin 12° is about 0.21.
      expect(f.x).toBeGreaterThan(Math.sin((30 * HEAD_SHARE_YAW - 3) * (Math.PI / 180)));
      expect(f.y).toBeGreaterThan(0.03);
      settle(actor, { yaw: -30, pitch: -15 });
      f = facing(actor);
      expect(f.x).toBeLessThan(-0.15);
      expect(f.y).toBeLessThan(-0.03);
    });

    it("comes back to ahead", async () => {
      const actor = await load();
      settle(actor, { yaw: 30, pitch: 15 });
      settle(actor, AHEAD, 3);
      const f = facing(actor);
      expect(Math.abs(f.x)).toBeLessThan(0.03);
      expect(Math.abs(f.y)).toBeLessThan(0.03);
      expect(actor.gaze.yaw).toBeCloseTo(0, 2);
    });

    /** The eyes are three-vrm's: told a target the whole gaze away, they
     * take what the head hasn't already turned. */
    it("points three-vrm's look-at the whole way, and it turns the eyes the rest", async () => {
      const actor = await load();
      settle(actor, { yaw: 30, pitch: 0 });
      const head = actor.vrm.humanoid.getRawBoneNode("head")!.getWorldPosition(new THREE.Vector3());
      const to = actor.lookTarget.getWorldPosition(new THREE.Vector3()).sub(head);
      const angle = (Math.atan2(to.x, to.z) * 180) / Math.PI;
      expect(angle).toBeGreaterThan(29);
      expect(angle).toBeLessThan(31);
      const lookAt = actor.vrm.lookAt!;
      expect(lookAt.target).toBe(actor.lookTarget);
      // What is left for the eyes once the head has turned its share.
      expect(lookAt.yaw).toBeGreaterThan(30 * (1 - HEAD_SHARE_YAW) - 4);
      expect(lookAt.yaw).toBeLessThan(30 * (1 - HEAD_SHARE_YAW) + 4);
    });

    /** Breathing still nods the head itself while he looks: the look adds
     * to the nod, it doesn't replace it. */
    it("keeps nodding while he looks", async () => {
      const actor = await load();
      // Long enough for the look to have settled: what still moves is the nod.
      settle(actor, { yaw: 10, pitch: 10 }, 6);
      const head = actor.vrm.humanoid.getNormalizedBoneNode("head")!;
      const seen = new Set<number>();
      for (let i = 0; i < 60; i++) {
        actor.update(1 / 30);
        seen.add(Math.round(head.rotation.x * 1e4));
      }
      expect(seen.size).toBeGreaterThan(5);
    });

    /** A glance, not a snap: one frame in, he is on his way. */
    it("eases to a new gaze rather than snapping", async () => {
      const actor = await load();
      actor.update(1 / 30);
      actor.setLook({ yaw: 30, pitch: 0 });
      actor.update(1 / 30);
      expect(actor.gaze.yaw).toBeGreaterThan(5);
      expect(actor.gaze.yaw).toBeLessThan(25);
    });
  });
}

/** Thinking: he glances up and away, then back, tilting his head while the
 * turn runs; with no motion, none of it. */
for (const [rig, load] of [
  ["a VRM 1.0 rig", () => VrmActor.load(placeholderBytes(), () => 0.5)],
  ["a VRM 0.x rig", () => VrmActor.load(asVrm0(placeholderBytes()), () => 0.5)],
] as const) {
  describe(`thinking, in ${rig}`, () => {
    const roll = (actor: VrmActor) => actor.vrm.humanoid.getNormalizedBoneNode("head")!.rotation.z;
    const run = (actor: VrmActor, seconds: number) => {
      for (let i = 0; i < Math.round(seconds * 30); i++) actor.update(1 / 30);
    };

    it("glances up and to his left within half a second, and back after the glance", async () => {
      const actor = await load();
      run(actor, 0.5);
      actor.setPhase("thinking");
      run(actor, 0.5);
      let f = facing(actor);
      expect(actor.gaze.yaw).toBeLessThan(GLANCE.yaw / 2);
      expect(actor.gaze.pitch).toBeGreaterThan(GLANCE.pitch / 2);
      expect(f.x).toBeLessThan(-0.03);
      expect(f.y).toBeGreaterThan(0.02);
      run(actor, GLANCE_S + 1.5);
      f = facing(actor);
      expect(Math.abs(actor.gaze.yaw)).toBeLessThan(1);
      expect(Math.abs(f.x)).toBeLessThan(0.03);
      expect(actor.currentPhase).toBe("thinking");
    });

    it("glances only on starting to think, not again while thinking", async () => {
      const actor = await load();
      actor.setPhase("thinking");
      run(actor, GLANCE_S + 1.5);
      actor.setPhase("thinking");
      run(actor, 0.3);
      expect(Math.abs(actor.gaze.yaw)).toBeLessThan(1);
    });

    it("goes back to the cursor after the glance", async () => {
      const actor = await load();
      actor.setLook({ yaw: 25, pitch: 0 });
      run(actor, 1);
      actor.setPhase("thinking");
      run(actor, 0.5);
      expect(actor.gaze.yaw).toBeLessThan(0);
      run(actor, GLANCE_S + 1);
      expect(actor.gaze.yaw).toBeGreaterThan(24);
    });

    it("tilts his head while he thinks, easing in and out", async () => {
      const actor = await load();
      run(actor, 0.2);
      expect(Math.abs(roll(actor))).toBeLessThan(1e-6);
      actor.setPhase("thinking");
      actor.update(1 / 30);
      const first = Math.abs(roll(actor));
      run(actor, 1.5);
      const full = Math.abs(roll(actor));
      expect(first).toBeGreaterThan(0);
      expect(first).toBeLessThan(full / 3);
      expect(full).toBeCloseTo((THINK_TILT_DEG * Math.PI) / 180, 2);
      actor.setPhase("idle");
      actor.update(1 / 30);
      expect(Math.abs(roll(actor))).toBeGreaterThan(full / 2);
      run(actor, 2);
      expect(Math.abs(roll(actor))).toBeLessThan(0.002);
    });

    it("with no motion, neither glances nor tilts", async () => {
      const actor = await load();
      actor.setMotion(0);
      actor.setPhase("thinking");
      for (let i = 0; i < 60; i++) {
        actor.update(1 / 30);
        expect(actor.gaze).toEqual({ yaw: 0, pitch: 0 });
        expect(Math.abs(roll(actor))).toBeLessThan(1e-9);
      }
    });

    it("keeps the other phases without a pose of their own", async () => {
      const actor = await load();
      for (const p of ["listening", "transcribing", "speaking", "idle"] as const) {
        actor.setPhase(p);
        run(actor, 0.5);
        expect(actor.currentPhase).toBe(p);
        expect(Math.abs(actor.gaze.yaw)).toBeLessThan(1e-6);
        expect(Math.abs(roll(actor))).toBeLessThan(1e-6);
      }
    });

    /** The placeholder's hair tops out about 23 cm over its head bone. */
    it("knows how far the top of the model is over his head", async () => {
      const actor = await load();
      expect(actor.crownAboveHead).toBeGreaterThan(0.2);
      expect(actor.crownAboveHead).toBeLessThan(0.27);
      run(actor, 1);
      expect(actor.crownAboveHead, "worked out once").toBe(actor.crownAboveHead);
    });

    /** The bubble points at a place above his eyes, higher on the page. */
    it("places a point further over his head higher on the page", async () => {
      const actor = await load();
      run(actor, 0.1);
      const camera = new THREE.PerspectiveCamera(22, 340 / 640, 0.05, 20);
      camera.position.set(0, 1.1, 6);
      camera.lookAt(0, 1.1, 0);
      camera.updateMatrixWorld();
      const eyes = actor.headScreen(camera, 340, 640);
      const crown = actor.headScreen(camera, 340, 640, 0.2);
      expect(crown.y).toBeLessThan(eyes.y - 5);
      expect(crown.x).toBeCloseTo(eyes.x, 0);
    });
  });
}
