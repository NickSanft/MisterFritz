/**
 * Settings › Hotkeys: his three keys (plan 14 build § 4.8, DivoraVoice's
 * Hotkeys rows; push-to-talk's row comes with PR 8). A key is captured with
 * DivoraVoice's HotkeyCapture and handed to the shell's `set_hotkey`, which
 * checks it, takes it from Windows, saves it, and only then lets the old
 * one go. A refusal is said on its row, the old key still showing and
 * still working; the settings that come back are what every row shows.
 */
import { batch, createEffect, createSignal, For, on, type JSX } from "solid-js";
import { HotkeyCapture } from "../../divora/components/HotkeyCapture";
import { acceleratorToKeys, keysToAccelerator } from "../../divora/hotkeys";
import { HOTKEY_ACTIONS, HOTKEY_DEFAULTS, type HotkeyAction, type HotkeysBound } from "../api";
import { Row, Section } from "../layout";
import { hotkeyLine } from "../view";

/** Each row's name. */
export const HOTKEY_TITLES: Record<HotkeyAction, string> = {
  prompt: "Ask Fritz",
  arrange: "Arrange him",
  park: "Hide or show him",
};

export interface HotkeysProps {
  /** The keys as set; null until the settings have loaded. */
  keys: Record<HotkeyAction, string> | null;
  /** The keys the shell holds now; null until it has said. */
  bound: HotkeysBound | null;
  /** Bind a key and save it: null once done, else the shell's refusal.
   * It settles only once what the shell holds has been asked again. */
  set: (action: HotkeyAction, accelerator: string) => Promise<string | null>;
}

type ByAction<T> = Partial<Record<HotkeyAction, T>>;

export function Hotkeys(props: HotkeysProps): JSX.Element {
  const [refusals, setRefusals] = createSignal<ByAction<string>>({});
  /** Rows with a key being tried: their line waits for the answer, since
   * the new setting arrives (settings:changed) before what is held. */
  const [trying, setTrying] = createSignal<ByAction<boolean>>({});
  // What the shell holds was read again (the window came forward, or a key
  // was tried): a refusal from before may no longer be true.
  createEffect(on(() => props.bound, () => setRefusals({}), { defer: true }));
  /** Each row's tries, numbered: only the latest one's answer is said. */
  const tries: ByAction<number> = {};
  const tryKey = async (action: HotkeyAction, accelerator: string): Promise<void> => {
    if (!accelerator) return;
    const n = (tries[action] = (tries[action] ?? 0) + 1);
    // Cleared as it starts, so the same refusal again is a new alert.
    batch(() => {
      setRefusals((r) => ({ ...r, [action]: undefined }));
      setTrying((t) => ({ ...t, [action]: true }));
    });
    const why = await props.set(action, accelerator);
    if (tries[action] !== n) return;
    batch(() => {
      setRefusals((r) => ({ ...r, [action]: why ?? undefined }));
      setTrying((t) => ({ ...t, [action]: false }));
    });
  };

  return (
    <Section
      id="hotkeys"
      sigil="keyboard"
      eyebrow="Hotkeys"
      title="His keys"
      desc="They work in every app and change at once. A key Fritz or another app already uses does its job rather than being captured."
    >
      <For each={HOTKEY_ACTIONS}>
        {(action) => {
          const setting = () => props.keys?.[action] ?? null;
          const busy = () => trying()[action] === true;
          const line = () => {
            const s = setting();
            if (s === null) return null;
            const active = busy() ? undefined : props.bound?.[action];
            return hotkeyLine(action, s, active, refusals()[action] ?? null);
          };
          const name = HOTKEY_TITLES[action];
          return (
            <Row
              title={name}
              desc={line()?.alert ? undefined : line()?.text}
              alert={line()?.alert ? line()!.text : null}
            >
              <div
                class="row-line"
                // A held Enter repeats, and each repeat clicks the focused
                // button: after Default that is the capture, which would arm
                // and take Enter, again and again. Only the first press acts;
                // an armed capture still hears every key (on the window).
                onKeyDown={(e) => {
                  if (e.repeat && e.key === "Enter") e.preventDefault();
                }}
              >
                <HotkeyCapture
                  value={setting() === null ? null : acceleratorToKeys(setting()!)}
                  onChange={(chips) => void tryKey(action, keysToAccelerator(chips))}
                  ariaLabel={(state) =>
                    state === "capturing"
                      ? `${name} hotkey: press a key, Esc to cancel`
                      : state === "set"
                        ? `${name} hotkey: ${acceleratorToKeys(setting()!).join(" ")}`
                        : `${name} hotkey: not set`}
                />
                <button
                  type="button"
                  class="btn btn-ghost btn-sm"
                  aria-label={`${name}: Default (${HOTKEY_DEFAULTS[action]})`}
                  disabled={setting() === null || setting() === HOTKEY_DEFAULTS[action] || busy()}
                  onClick={async (e) => {
                    // From the keyboard, the focus stays in the row: the
                    // button is disabled while it tries, and for good once
                    // the default is back, which would drop it to the page.
                    const button = e.currentTarget;
                    const field = button.previousElementSibling as HTMLElement | null;
                    const fromKeys = e.detail === 0;
                    await tryKey(action, HOTKEY_DEFAULTS[action]);
                    if (fromKeys) (button.disabled ? field : button)?.focus();
                  }}
                >
                  Default
                </button>
              </div>
            </Row>
          );
        }}
      </For>
    </Section>
  );
}
