/**
 * Whether push-to-talk is muted, for the sign over his head: read strictly
 * from the settings, and followed as the theme is (an event wins over a
 * slower answer to the question asked at start).
 */
import { describe, expect, it } from "vitest";
import { mutedOf, watchMuted, type MutedWatchDeps } from "./muted";

const settings = (muted: unknown) => ({ audio: { inputDevice: null, outputDevice: null, muted, ttsVoice: "bm_george" } });

/** Deps whose event and answer the test gives. */
function deps(answer: unknown) {
  let emit: (s: unknown) => void = () => {};
  let answered: (s: unknown) => void = () => {};
  const asked = new Promise<unknown>((ok) => (answered = ok));
  const d: MutedWatchDeps = {
    listen: async (cb) => {
      emit = cb;
      return () => {
        emit = () => {};
      };
    },
    ask: () => asked,
  };
  return { d, emit: (s: unknown) => emit(s), answer: () => answered(answer) };
}

const settle = () => new Promise((r) => setTimeout(r, 0));

describe("push-to-talk muted, on his page", () => {
  it("is read strictly from the settings", () => {
    expect(mutedOf(settings(true))).toBe(true);
    expect(mutedOf(settings(false))).toBe(false);
    for (const bad of [settings("yes"), settings(undefined), { audio: null }, {}, null, 3]) {
      expect(mutedOf(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  it("is told from the answer when no event came first, and from every event", async () => {
    const seen: boolean[] = [];
    const { d, emit, answer } = deps(settings(true));
    const off = await watchMuted((m) => seen.push(m), d);
    answer();
    await settle();
    expect(seen).toEqual([true]);
    emit(settings(false));
    emit({ theme: {} });
    emit(settings(true));
    expect(seen).toEqual([true, false, true]);
    off();
    emit(settings(false));
    expect(seen, "no longer listening").toEqual([true, false, true]);
  });

  it("keeps an event over the answer that comes after it", async () => {
    const seen: boolean[] = [];
    const { d, emit, answer } = deps(settings(false));
    await watchMuted((m) => seen.push(m), d);
    emit(settings(true));
    answer();
    await settle();
    expect(seen).toEqual([true]);
  });

  it("waits for an event when the question goes unanswered", async () => {
    const seen: boolean[] = [];
    const d: MutedWatchDeps = { listen: async () => () => {}, ask: () => Promise.reject(new Error("no shell")) };
    await watchMuted((m) => seen.push(m), d);
    await settle();
    expect(seen).toEqual([]);
  });
});
