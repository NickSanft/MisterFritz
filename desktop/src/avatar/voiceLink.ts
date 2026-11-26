/**
 * His voice in the shell (kokoro_tts.rs, plan 14 PR 10): whether its files
 * are installed, and one of his sentences spoken, by the names the shell
 * has (wire.fixture.json). What comes back is 24 kHz mono samples, ready
 * for an AudioBuffer, which speech.ts plays.
 */
import { invoke } from "@tauri-apps/api/core";
import fixture from "./wire.fixture.json";

export const TTS_READY_CMD = "tts_ready";
export const TTS_SYNTHESIZE_CMD = "tts_synthesize";
/** Kokoro's rate, which every sample the shell gives is at. */
export const TTS_SAMPLE_RATE: number = fixture.voice.sampleRate;
/** The longest text the shell speaks at once, in characters (a sentence
 * from the brain is far below it). */
export const TTS_MAX_CHARS: number = fixture.voice.maxChars;

type Call = (cmd: string, args: Record<string, unknown>) => Promise<unknown>;

export const ttsSynthesizeArgs = (text: string) => ({ text });

/** Whether his voice is installed: false when the shell says not, or can't
 * say. */
export async function voiceReady(call: Call = invoke): Promise<boolean> {
  try {
    return (await call(TTS_READY_CMD, {})) === true;
  } catch (e) {
    console.warn(`avatar: ${TTS_READY_CMD} failed`, e);
    return false;
  }
}

/** A sentence of his, spoken: its samples, or why not in the shell's words. */
export type Spoken = { samples: Float32Array<ArrayBuffer> } | { why: string };

/** `text` spoken by the shell. Its answer is raw f32 little-endian (an
 * ArrayBuffer, as `ipc::Response` gives it); Windows on x86 and ARM is
 * little-endian, so a Float32Array reads it as it is. */
export async function speakSentence(text: string, call: Call = invoke): Promise<Spoken> {
  try {
    const got = await call(TTS_SYNTHESIZE_CMD, ttsSynthesizeArgs(text));
    if (!(got instanceof ArrayBuffer) || got.byteLength === 0 || got.byteLength % 4 !== 0) {
      return { why: "his voice gave back no sound" };
    }
    return { samples: new Float32Array(got) };
  } catch (e) {
    return { why: typeof e === "string" && e.trim() !== "" ? e : "his voice failed" };
  }
}
