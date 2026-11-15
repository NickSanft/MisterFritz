/**
 * The first run's steps and what its last one says: pure, so it is tested
 * without a window (Wizard.tsx draws them).
 */
import type { LinkStatus } from "../shared/companion";
import type { HotkeysBound, ModelInfo } from "./api";
import { winNamed } from "./view";

/** The steps, in order. */
export const WIZARD_STEPS = ["brain", "look", "keys", "ready"] as const;
export type WizardStep = (typeof WIZARD_STEPS)[number];

/** Each step's heading. */
export const STEP_TITLES: Record<WizardStep, string> = {
  brain: "Connect his brain",
  look: "Choose his look",
  keys: "Learn his keys",
  ready: "Summon him",
};

/** The step after `s`, or null after the last. */
export function stepAfter(s: WizardStep): WizardStep | null {
  return WIZARD_STEPS[WIZARD_STEPS.indexOf(s) + 1] ?? null;
}

/** The step before `s`, or null before the first. */
export function stepBefore(s: WizardStep): WizardStep | null {
  const i = WIZARD_STEPS.indexOf(s);
  return i > 0 ? WIZARD_STEPS[i - 1] : null;
}

/** Where the first run stands, as the live line says it. */
export function stepLine(s: WizardStep): string {
  return `Step ${WIZARD_STEPS.indexOf(s) + 1} of ${WIZARD_STEPS.length}: ${STEP_TITLES[s]}`;
}

export interface ReadyLine {
  text: string;
  /** `warn` for what keeps him from working until it's seen to. */
  tone: "ok" | "warn" | "quiet";
}

/** The last step's lines: whether he can reach his brain, what he wears,
 * and the key that asks him something. */
export function readyLines(
  link: LinkStatus | null,
  model: ModelInfo | null,
  bound: HotkeysBound | null,
  tokenPresent: boolean | null,
): ReadyLine[] {
  const brain: ReadyLine =
    link?.link === "online"
      ? { text: "He's connected to his brain.", tone: "ok" }
      : tokenPresent === false
        ? { text: "There's no token yet, so he can't reach his brain: paste it in Settings › Brain.", tone: "warn" }
        : { text: "He isn't connected to his brain yet: Settings › Brain says why.", tone: "warn" };
  const look: ReadyLine =
    model?.wearing === "export"
      ? model.found && !model.changed
        ? { text: "He wears your model.", tone: "ok" }
        : { text: "He wears the model he loaded last, until it's loaded again in Settings › Model.", tone: "quiet" }
      : { text: "He wears the placeholder until your VRoid export is in his folder and loaded.", tone: "quiet" };
  const keys: ReadyLine =
    bound === null
      ? { text: "His keys are in Settings › Hotkeys.", tone: "quiet" }
      : bound.prompt
        ? { text: winNamed(`Press ${bound.prompt} in any app to ask him something.`), tone: "ok" }
        : { text: "His prompt key isn't working: pick another in Settings › Hotkeys.", tone: "warn" };
  return [brain, look, keys];
}
