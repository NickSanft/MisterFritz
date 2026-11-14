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
import {
  VRMExpressionMaterialColorBind, VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName,
} from "@pixiv/three-vrm";
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

/** How far `setTint` moves an outline from the model's own colour toward the
 * theme's: a lean, not a repaint. */
export const OUTLINE_MIX = 0.35;

/** The rim's falloff the tint gives a model whose own rim is off: VRM 1.0's
 * default, a light at the silhouette. A VRoid 0.x export leaves its rim black
 * with a falloff of 1, which would spread the accent over half his body. */
export const RIM_FRESNEL = 5;

/** An MToon material (three-vrm's), as far as the tint reaches into it. */
interface Toon {
  isMToonMaterial: true;
  outlineColorFactor: THREE.Color;
  parametricRimColorFactor: THREE.Color;
  parametricRimFresnelPowerFactor: number;
  parametricRimLiftFactor: number;
}

/** What the model's file says, kept so every tint starts from it. */
interface Authored {
  outline: THREE.Color;
  rim: THREE.Color;
  fresnel: number;
  lift: number;
}

function isToon(m: unknown): m is Toon {
  return (m as Partial<Toon> | null)?.isMToonMaterial === true;
}

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
  /** The theme's motion factor (theme.ts): 1 is everything, 0 leaves only
   * the blink and the breathing. */
  private motion = 1;
  /** Each MToon material's own outline and rim, as the file has them:
   * every tint starts from these, so tints never compound. */
  private readonly authored = new WeakMap<Toon, Authored>();
  /** The tint now, and the colours an expression owns: three-vrm puts those
   * back to the file's every frame (its colour binds), so the tint is laid on
   * them again after each update. */
  private tint: { toward: THREE.Color; glow: THREE.Color } | null = null;
  private boundRim = new Set<Toon>();
  private boundOutline = new Set<Toon>();

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
    if (this.tint) {
      for (const m of this.boundRim) m.parametricRimColorFactor.add(this.tint.glow);
      for (const m of this.boundOutline) m.outlineColorFactor.lerp(this.tint.toward, OUTLINE_MIX);
    }
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

  /** The theme's motion factor (0 to 1): it scales his sway, and at 0 he
   * looks ahead whatever `setLook` says (plan 14: with no motion, only the
   * blink and the breathing move). Neither the blink nor the breathing is
   * ever scaled. */
  setMotion(factor: number): void {
    this.motion = Math.min(1, Math.max(0, factor));
  }

  /** Lean his MToon outlines toward `outline` and add `rim` (sRGB hex) to his
   * rim light, `rimStrength` of it: always from the model's own colours, so
   * a second tint replaces the first. Where the model's own rim is off
   * (black), its falloff shows nothing, so the tint gives it a rim's
   * (RIM_FRESNEL, no lift); a rim the model lights itself keeps its shape.
   * Other materials are left alone. */
  setTint(outline: string, rim: string, rimStrength: number): void {
    const toward = new THREE.Color(outline);
    const glow = new THREE.Color(rim).multiplyScalar(rimStrength);
    this.tint = { toward, glow };
    this.vrm.scene.traverse((o) => {
      const material = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!material) return;
      for (const m of Array.isArray(material) ? material : [material]) {
        if (!isToon(m)) continue;
        let own = this.authored.get(m);
        if (!own) {
          own = {
            outline: m.outlineColorFactor.clone(),
            rim: m.parametricRimColorFactor.clone(),
            fresnel: m.parametricRimFresnelPowerFactor,
            lift: m.parametricRimLiftFactor,
          };
          this.authored.set(m, own);
        }
        m.outlineColorFactor.copy(own.outline).lerp(toward, OUTLINE_MIX);
        m.parametricRimColorFactor.copy(own.rim).add(glow);
        const off = own.rim.r === 0 && own.rim.g === 0 && own.rim.b === 0;
        m.parametricRimFresnelPowerFactor = off ? Math.max(own.fresnel, RIM_FRESNEL) : own.fresnel;
        m.parametricRimLiftFactor = off ? 0 : own.lift;
      }
    });
    // The colours an expression owns, looked for now (not at load), so an
    // expression registered since is seen.
    this.boundRim = new Set();
    this.boundOutline = new Set();
    for (const e of this.vrm.expressionManager?.expressions ?? []) {
      for (const b of e.binds) {
        if (!(b instanceof VRMExpressionMaterialColorBind) || !isToon(b.material)) continue;
        if (b.type === "rimColor") this.boundRim.add(b.material);
        else if (b.type === "outlineColor") this.boundOutline.add(b.material);
      }
    }
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
    const aim = this.motion === 0 ? AHEAD : this.aim;
    this.eyes.yaw = approach(this.eyes.yaw, aim.yaw, dt, EYE_TAU);
    this.eyes.pitch = approach(this.eyes.pitch, aim.pitch, dt, EYE_TAU);
    this.headTurn.yaw = approach(this.headTurn.yaw, aim.yaw * HEAD_SHARE_YAW, dt, HEAD_TAU);
    this.headTurn.pitch = approach(this.headTurn.pitch, aim.pitch * HEAD_SHARE_PITCH, dt, HEAD_TAU);
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
    if (hips) hips.rotation.y = 0.02 * Math.sin(this.t * 0.31) * this.motion;
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
