/**
 * The page's link to his voice in the shell: the names and keys the shell
 * has (wire.fixture.json), and what it makes of the shell's answers.
 */
import { describe, expect, it, vi } from "vitest";
import fixture from "./wire.fixture.json";
import {
  TTS_MAX_CHARS,
  TTS_READY_CMD,
  TTS_SAMPLE_RATE,
  TTS_SYNTHESIZE_CMD,
  speakSentence,
  ttsSynthesizeArgs,
  voiceReady,
} from "./voiceLink";

describe("his voice's link to the shell", () => {
  it("names the shell's commands and keys", () => {
    const c = fixture.commands;
    expect(TTS_READY_CMD).toBe(c.ttsReady.name);
    expect(c.ttsReady.args).toEqual([]);
    expect(TTS_SYNTHESIZE_CMD).toBe(c.ttsSynthesize.name);
    expect(Object.keys(ttsSynthesizeArgs("Good evening."))).toEqual(c.ttsSynthesize.args);
    expect(TTS_SAMPLE_RATE).toBe(24_000);
    expect(TTS_MAX_CHARS).toBe(2_000);
  });

  it("is ready only when the shell says so", async () => {
    const said = (v: unknown) => vi.fn(async () => v);
    const yes = said(true);
    expect(await voiceReady(yes)).toBe(true);
    expect(yes).toHaveBeenCalledWith(TTS_READY_CMD, {});
    for (const no of [false, "true", 1, null, undefined, {}]) {
      expect(await voiceReady(said(no)), JSON.stringify(no)).toBe(false);
    }
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await voiceReady(vi.fn(async () => Promise.reject(new Error("gone"))))).toBe(false);
    warn.mockRestore();
  });

  it("reads the shell's samples as they are", async () => {
    const samples = new Float32Array([0, 0.5, -0.25, 1]);
    const call = vi.fn(async () => samples.buffer);
    const got = await speakSentence("It is half past nine.", call);
    expect(call).toHaveBeenCalledWith(TTS_SYNTHESIZE_CMD, { text: "It is half past nine." });
    expect("samples" in got && Array.from(got.samples)).toEqual([0, 0.5, -0.25, 1]);
  });

  it("says why when there is no sound", async () => {
    for (const bad of [null, "bytes", new ArrayBuffer(0), new ArrayBuffer(6), new Uint8Array(8)]) {
      const got = await speakSentence("x", vi.fn(async () => bad));
      expect(got, String(bad)).toEqual({ why: "his voice gave back no sound" });
    }
    const words = "his voice isn't installed (no onnxruntime.dll): run desktop/scripts/fetch_voice.py";
    expect(await speakSentence("x", vi.fn(async () => Promise.reject(words)))).toEqual({ why: words });
    for (const odd of [new Error("boom"), "", "  ", undefined]) {
      expect(await speakSentence("x", vi.fn(async () => Promise.reject(odd)))).toEqual({ why: "his voice failed" });
    }
  });
});
