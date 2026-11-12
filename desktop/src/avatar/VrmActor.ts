/**
 * Fritz's body: a VRM loaded with @pixiv/three-vrm, posed and kept alive with
 * a procedural idle — breathing, a slow sway, and a blink every few seconds.
 * Spike 0 needs only this much; PR 2 adds look-at, mood and the mouth.
 *
 * Poses go through three-vrm's normalized humanoid. That rig is built from
 * the bones as the file has them, before `VRMUtils.rotateVRM0` turns a VRM
 * 0.x model round to face +Z: a 0.x rig keeps the -Z frame it was authored
 * in, which is the 1.0 frame turned half round about Y. That turn negates
 * rotations about X and Z (and leaves Y alone), so every X and Z angle below
 * is multiplied by `k`, -1 for a 0.x model (three-vrm's own retargeting
 * examples do the same).
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName } from "@pixiv/three-vrm";

/** Breaths a minute at rest. */
const BREATHS_PER_MINUTE = 14;
/** Seconds between blinks, drawn uniformly. */
const BLINK_GAP: readonly [number, number] = [2, 6];
/** Seconds a blink takes to close, then to open. */
const BLINK_CLOSE = 0.06;
const BLINK_OPEN = 0.1;

/** Arms down from the T-pose the file is stored in, in radians. */
const ARM_DOWN = 1.22;
const FOREARM_BEND = 0.12;

export class VrmActor {
  private t = 0;
  private nextBlink: number;
  private blinkAt = -1;
  /** The sign of X and Z rotations: -1 for a VRM 0.x rig (see above). */
  private readonly k: number;

  private constructor(
    readonly vrm: VRM,
    private readonly random: () => number,
  ) {
    this.k = vrm.meta?.metaVersion === "0" ? -1 : 1;
    this.nextBlink = this.gap();
    this.restPose();
    // Pose the bones now, so bounds() frames him standing, not in T-pose.
    vrm.humanoid?.update();
  }

  /** Parse a .vrm (glTF binary). Throws when the file isn't a VRM. */
  static async load(data: ArrayBuffer, random: () => number = Math.random): Promise<VrmActor> {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.parseAsync(data, "");
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) throw new Error("That file has no VRM in it.");
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(vrm);
    // Skinned meshes keep their bind-pose bounds; posed limbs would be culled.
    vrm.scene.traverse((o) => {
      o.frustumCulled = false;
    });
    return new VrmActor(vrm, random);
  }

  /** The model's vertical extent in world units, feet to crown. */
  bounds(): { bottom: number; top: number } {
    this.vrm.scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.vrm.scene);
    return { bottom: box.min.y, top: box.max.y };
  }

  /** Advance the idle by `dt` seconds. */
  update(dt: number): void {
    this.t += dt;
    this.breathe();
    this.blink(dt);
    this.vrm.update(dt);
  }

  private bone(name: VRMHumanBoneName): THREE.Object3D | null {
    return this.vrm.humanoid?.getNormalizedBoneNode(name) ?? null;
  }

  private restPose(): void {
    const set = (name: VRMHumanBoneName, z: number) => {
      const b = this.bone(name);
      if (b) b.rotation.z = z;
    };
    // The model's left arm lies along +X: turning it about Z by a negative
    // angle lowers it. The right arm mirrors.
    const k = this.k;
    set("leftUpperArm", -ARM_DOWN * k);
    set("rightUpperArm", ARM_DOWN * k);
    set("leftLowerArm", -FOREARM_BEND * k);
    set("rightLowerArm", FOREARM_BEND * k);
  }

  private breathe(): void {
    const k = this.k;
    const s = Math.sin((this.t * 2 * Math.PI * BREATHS_PER_MINUTE) / 60);
    const chest = this.bone("chest");
    const upper = this.bone("upperChest");
    if (chest) chest.rotation.x = -0.008 * s * k;
    if (upper) upper.rotation.x = -0.012 * s * k;
    const left = this.bone("leftShoulder");
    const right = this.bone("rightShoulder");
    if (left) left.rotation.z = 0.015 * s * k;
    if (right) right.rotation.z = -0.015 * s * k;
    const hips = this.bone("hips");
    if (hips) hips.rotation.y = 0.02 * Math.sin(this.t * 0.31);
    const head = this.bone("head");
    if (head) head.rotation.x = 0.01 * Math.sin(this.t * 0.47 + 1) * k;
  }

  private blink(dt: number): void {
    const em = this.vrm.expressionManager;
    if (!em) return;
    if (this.blinkAt < 0) {
      this.nextBlink -= dt;
      if (this.nextBlink > 0) return;
      this.blinkAt = 0;
    }
    this.blinkAt += dt;
    const p = this.blinkAt;
    let w: number;
    if (p < BLINK_CLOSE) w = p / BLINK_CLOSE;
    else if (p < BLINK_CLOSE + BLINK_OPEN) w = 1 - (p - BLINK_CLOSE) / BLINK_OPEN;
    else {
      w = 0;
      this.blinkAt = -1;
      this.nextBlink = this.gap();
    }
    em.setValue("blink", w);
  }

  private gap(): number {
    const [lo, hi] = BLINK_GAP;
    return lo + (hi - lo) * this.random();
  }
}
