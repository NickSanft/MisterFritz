/**
 * The first run, in the control window in place of its sections: connect
 * his brain, choose his look, learn his keys, then summon him. Fritz's own
 * code, following the layout of DivoraVoice's first-run wizard (a rail of
 * steps, the step, a footer of Skip, Back and Continue).
 *
 * The steps are the sections' own rows (Brain, the Model rows, Hotkeys),
 * so the first run sets up exactly what Settings does. Each step's heading
 * takes the focus as it comes (inside the page only), and a live line says
 * where the run stands. Esc does nothing here, and nothing is kept: the
 * first run shows whenever there is no token yet, or from About.
 */
import { createSignal, For, Match, onMount, Show, Switch, type JSX } from "solid-js";
import { Sigil } from "../divora/sigils/Sigil";
import type { HotkeysBound, ModelInfo } from "./api";
import type { LinkStatus } from "../shared/companion";
import { Brain, type BrainProps } from "./sections/Brain";
import { Hotkeys, type HotkeysProps } from "./sections/Hotkeys";
import { ModelRows, type ModelRowsProps } from "./sections/Model";
import { Row } from "./layout";
import { readyLines, STEP_TITLES, stepAfter, stepBefore, stepLine, WIZARD_STEPS, type WizardStep } from "./firstRun";

/** What each step says under its heading. */
const INTROS: Record<WizardStep, string> = {
  brain: "Paste the COMPANION_TOKEN from his brain's .env, and he connects. Then try him with a question.",
  look: "He wears your VRoid Studio export once it's in his folder, or a placeholder until then.",
  keys: "These work in every app. Change any now, or later in Settings.",
  ready: "Summon him, and he appears on your desktop. Anything below that isn't ready yet can be seen to in Settings.",
};

export interface WizardProps {
  brain: BrainProps;
  model: ModelRowsProps;
  hotkeys: HotkeysProps;
  /** For the last step's lines. */
  link: LinkStatus | null;
  modelInfo: ModelInfo | null;
  bound: HotkeysBound | null;
  tokenPresent: boolean | null;
  /** Leave the first run for the sections. */
  skip(): void;
  /** Bring him onto the desktop and close the first run; refused, the
   * first run stays and the window's banner says why. */
  summon(): Promise<void>;
}

export function Wizard(props: WizardProps): JSX.Element {
  const [step, setStep] = createSignal<WizardStep>(WIZARD_STEPS[0]);
  const [summoning, setSummoning] = createSignal(false);
  let heading: HTMLHeadingElement | undefined;
  const at = () => WIZARD_STEPS.indexOf(step());
  const go = (s: WizardStep | null) => {
    if (!s) return;
    setStep(s);
    heading?.focus();
  };
  onMount(() => heading?.focus());
  const next = async () => {
    const after = stepAfter(step());
    if (after) {
      go(after);
      return;
    }
    if (summoning()) return;
    setSummoning(true);
    try {
      await props.summon();
    } finally {
      setSummoning(false);
    }
  };

  return (
    <div class="wizard" data-view="wizard">
      <nav class="wizard-rail" aria-label="First run">
        <div class="wizard-brand">
          <Sigil name="butler" size={22} /> Mister Fritz
        </div>
        <ol class="wizard-steps">
          <For each={WIZARD_STEPS}>
            {(s, i) => (
              <li
                classList={{ "wizard-step": true, "is-current": s === step(), "is-done": i() < at() }}
                aria-current={s === step() ? "step" : undefined}
              >
                <span class="wizard-num" aria-hidden="true">
                  <Show when={i() < at()} fallback={String(i() + 1)}>
                    <Sigil name="check" size={12} />
                  </Show>
                </span>
                <span>{STEP_TITLES[s]}</span>
              </li>
            )}
          </For>
        </ol>
      </nav>

      <div class="wizard-main">
        <div class="wizard-content">
          <p class="wizard-count" aria-live="polite">
            {stepLine(step())}
          </p>
          <h2 class="wizard-title" tabindex="-1" ref={heading}>
            {STEP_TITLES[step()]}
          </h2>
          <p class="wizard-intro">{INTROS[step()]}</p>
          <Switch>
            <Match when={step() === "brain"}>
              <Brain {...props.brain} />
            </Match>
            <Match when={step() === "look"}>
              <div class="panel">
                <ModelRows {...props.model} />
              </div>
            </Match>
            <Match when={step() === "keys"}>
              <Hotkeys {...props.hotkeys} />
            </Match>
            <Match when={step() === "ready"}>
              <div class="panel">
                <For each={readyLines(props.link, props.modelInfo, props.bound, props.tokenPresent)}>
                  {(line) => (
                    <Row title={line.text}>
                      <span classList={{ dot: true, [`tone-${line.tone}`]: true }} />
                    </Row>
                  )}
                </For>
              </div>
            </Match>
          </Switch>
        </div>

        <div class="wizard-footer">
          <button type="button" class="btn btn-ghost btn-sm" onClick={() => props.skip()}>
            Skip setup
          </button>
          <span class="wizard-spacer" />
          <Show when={at() > 0}>
            <button type="button" class="btn btn-secondary" onClick={() => go(stepBefore(step()))}>
              Back
            </button>
          </Show>
          <button type="button" class="btn btn-primary" disabled={summoning()} onClick={() => void next()}>
            {stepAfter(step()) ? "Continue" : "Summon Fritz"}
          </button>
        </div>
      </div>
    </div>
  );
}
