/**
 * Fritz's body: a VRM loaded with @pixiv/three-vrm, posed and kept alive with
 * a procedural idle — breathing, a slow sway, a blink every few seconds — and
 * looking where he is told (`setLook`, from the cursor): his eyes through
 * three-vrm's look-at, his head turning a share of the way. Mood and the
 * mouth come with the PRs that drive them.
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
import { AHEAD, EYE_TAU, HEAD_SHARE_PITCH, HEAD_SHARE_YAW, HEAD_TAU, approach, type Gaze } from "./look";

const RAD = Math.PI / 180;
/** The eyes sit about this far above the head bone, in metres: where a
 * gaze starts on the page. */
const EYES_ABOVE_HEAD = 0.07;

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
  /** Where he has been told to look, and where his eyes and head are now
   * on the way there. */
  private aim: Gaze = AHEAD;
  private eyes: Gaze = { ...AHEAD };
  private headTurn: Gaze = { ...AHEAD };
  /** The point three-vrm's look-at turns the eyes to: a metre out from his
   * head, in the direction of `eyes`. It is told the whole gaze; it works
   * out the eyes' part from where the head already points. */
  readonly lookTarget = new THREE.Object3D();

  private constructor(
    readonly vrm: VRM,
    private readonly random: () => number,
  ) {
    this.k = vrm.meta?.metaVersion === "0" ? -1 : 1;
    this.nextBlink = this.gap();
    this.restPose();
    if (vrm.lookAt) {
      vrm.lookAt.target = this.lookTarget;
      vrm.lookAt.autoUpdate = true;
    }
    // Pose the bones now, so bounds() frames him standing, not in T-pose.
    vrm.humanoid?.update();
    this.placeLookTarget();
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
    this.look(dt);
    this.blink(dt);
    this.vrm.update(dt);
  }

  /** Look this way (degrees; right and up positive, as the page sees it).
   * He eases there over the next frames. */
  setLook(g: Gaze): void {
    this.aim = { yaw: g.yaw, pitch: g.pitch };
  }

  /** Where he looks now, on the way to the last `setLook`. */
  get gaze(): Gaze {
    return { ...this.eyes };
  }

  /** His eyes' place on a page `width` × `height` CSS pixels big, seen by
   * `camera`: where a gaze toward the cursor starts. */
  headScreen(camera: THREE.Camera, width: number, height: number): { x: number; y: number } {
    const head = this.vrm.humanoid?.getRawBoneNode("head");
    if (!head) return { x: width / 2, y: height / 4 };
    const p = head.getWorldPosition(new THREE.Vector3());
    p.y += EYES_ABOVE_HEAD;
    p.project(camera);
    return { x: ((p.x + 1) / 2) * width, y: ((1 - p.y) / 2) * height };
  }

  /** Ease toward the aim: the eyes quickly, the head a share of the way and
   * slower, on top of the breathing's nod. */
  private look(dt: number): void {
    this.eyes.yaw = approach(this.eyes.yaw, this.aim.yaw, dt, EYE_TAU);
    this.eyes.pitch = approach(this.eyes.pitch, this.aim.pitch, dt, EYE_TAU);
    this.headTurn.yaw = approach(this.headTurn.yaw, this.aim.yaw * HEAD_SHARE_YAW, dt, HEAD_TAU);
    this.headTurn.pitch = approach(this.headTurn.pitch, this.aim.pitch * HEAD_SHARE_PITCH, dt, HEAD_TAU);
    const head = this.bone("head");
    if (head) {
      // Facing +Z, turning about Y toward +X turns him to the page's right,
      // in either rig. About X a positive angle tips the face down in a 1.0
      // rig, so up is negative, times k.
      head.rotation.y = this.headTurn.yaw * RAD;
      head.rotation.x += -this.headTurn.pitch * RAD * this.k;
    }
    this.placeLookTarget();
  }

  /** A metre out from his eyes, along `eyes`, in the world (where he faces
   * +Z and the page's right is +X). */
  private placeLookTarget(): void {
    const head = this.vrm.humanoid?.getRawBoneNode("head");
    if (!head) return;
    const origin = head.getWorldPosition(new THREE.Vector3());
    origin.y += EYES_ABOVE_HEAD;
    const yaw = this.eyes.yaw * RAD;
    const pitch = this.eyes.pitch * RAD;
    this.lookTarget.position.set(
      origin.x + Math.sin(yaw) * Math.cos(pitch),
      origin.y + Math.sin(pitch),
      origin.z + Math.cos(yaw) * Math.cos(pitch),
    );
    this.lookTarget.updateMatrixWorld();
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
