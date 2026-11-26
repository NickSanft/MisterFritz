/**
 * His mouth on a real model: the placeholder's `aa` follows `setMouth` at
 * the next update, at any motion; a model without expressions (a 0.x test
 * model has none) takes it without harm.
 */
import { describe, expect, it } from "vitest";
import { VrmActor } from "./VrmActor";
import { asVrm0, placeholderBytes } from "./testModels";

describe("his mouth on his model", () => {
  it("opens the model's aa as far as it is told, at the next update", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    const em = actor.vrm.expressionManager!;
    expect(em.getExpression("aa"), "the placeholder has an aa").toBeTruthy();
    actor.update(1 / 30);
    expect(em.getValue("aa")).toBe(0);
    actor.setMouth(0.6);
    expect(em.getValue("aa"), "not before the update").toBe(0);
    actor.update(1 / 30);
    expect(em.getValue("aa")).toBeCloseTo(0.6);
    expect(actor.mouthOpen).toBeCloseTo(0.6);
    actor.setMouth(0);
    actor.update(1 / 30);
    expect(em.getValue("aa")).toBe(0);
  });

  it("moves at motion 0 too: he is speaking", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    actor.setMotion(0);
    actor.setMouth(0.8);
    actor.update(1 / 30);
    expect(actor.vrm.expressionManager!.getValue("aa")).toBeCloseTo(0.8);
  });

  it("is held to 0 to 1", async () => {
    const actor = await VrmActor.load(placeholderBytes(), () => 0.5);
    for (const [asked, set] of [[1.7, 1], [-0.3, 0], [Number.NaN, 0], [Number.POSITIVE_INFINITY, 0]] as const) {
      actor.setMouth(asked);
      expect(actor.mouthOpen, String(asked)).toBe(set);
    }
  });

  it("a model without expressions takes it without harm", async () => {
    const actor = await VrmActor.load(asVrm0(placeholderBytes()), () => 0.5);
    actor.setMouth(0.5);
    expect(() => actor.update(1 / 30)).not.toThrow();
  });
});
