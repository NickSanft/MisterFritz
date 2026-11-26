/**
 * His face on a real model: `setFace` reaches the model's expressions at
 * the next update, each by the first of its names the model has: the
 * placeholder's VRM 1.0 presets, and a VRoid 0.x export's (Joy as happy,
 * Sorrow as sad, a custom Surprised by its own name), as the owner's is.
 */
import { describe, expect, it } from "vitest";
import { VrmActor } from "./VrmActor";
import { Face, FACES, IN_MS } from "./face";
import { VROID0_GROUPS, asVrm0, placeholderBytes } from "./testModels";

describe("his face on his model", () => {
  it("sets an expression as far as it is told, at the next update", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setFace([{ names: ["happy"], weight: 0.4 }]);
    expect(actor.expressionValue("happy"), "not before the update").toBe(0);
    actor.update(1 / 30);
    expect(actor.expressionValue("happy")).toBeCloseTo(0.4);
    actor.setFace([{ names: ["happy"], weight: 0 }]);
    actor.update(1 / 30);
    expect(actor.expressionValue("happy")).toBe(0);
  });

  it("each mood reaches the placeholder's own preset", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const face = new Face();
    for (const [mood, name] of [["pleased", "happy"], ["alarmed", "surprised"], ["weary", "sad"]] as const) {
      face.set(mood);
      for (let i = 0; i < 40; i++) face.tick(IN_MS / 10);
      actor.setFace(face.faces());
      actor.update(1 / 30);
      expect(actor.expressionValue(name), mood).toBeCloseTo(FACES[mood]!.weight);
    }
  });

  it("a VRoid 0.x export's: Joy, Sorrow and its custom Surprised", async () => {
    const actor = await VrmActor.load(asVrm0(placeholderBytes(), VROID0_GROUPS), () => 0.5);
    const em = actor.vrm.expressionManager!;
    expect(["happy", "sad", "Surprised", "aa"].map((n) => em.getExpression(n) != null)).toEqual([true, true, true, true]);
    expect(em.getExpression("surprised")).toBeFalsy();
    actor.setFace(new Face().faces().map((f) => ({ ...f, weight: 0.3 })));
    actor.setMouth(0.6);
    actor.update(1 / 30);
    expect(actor.expressionValue("happy")).toBeCloseTo(0.3);
    expect(actor.expressionValue("sad")).toBeCloseTo(0.3);
    expect(actor.expressionValue("Surprised")).toBeCloseTo(0.3);
    // His mouth moves under it.
    expect(actor.expressionValue("aa")).toBeCloseTo(0.6);
  });

  it("moves at motion 0 too", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setMotion(0);
    actor.setFace([{ names: ["sad"], weight: 0.4 }]);
    actor.update(1 / 30);
    expect(actor.expressionValue("sad")).toBeCloseTo(0.4);
  });

  it("is held to 0 to 1", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    for (const [asked, set] of [[1.7, 1], [-0.3, 0], [Number.NaN, 0], [Number.POSITIVE_INFINITY, 0]] as const) {
      actor.setFace([{ names: ["happy"], weight: asked }]);
      actor.update(1 / 30);
      expect(actor.expressionValue("happy"), String(asked)).toBe(set);
    }
  });

  it("leaves out what the model hasn't, unharmed", async () => {
    for (const bytes of [asVrm0(placeholderBytes()), placeholderBytes()]) {
      const actor = await VrmActor.load(bytes, () => 0.5);
      actor.setFace([{ names: ["smug", "Smug"], weight: 0.5 }, { names: ["happy"], weight: 0.2 }]);
      expect(() => actor.update(1 / 30)).not.toThrow();
      expect(actor.expressionValue("smug")).toBe(0);
    }
  });
});
