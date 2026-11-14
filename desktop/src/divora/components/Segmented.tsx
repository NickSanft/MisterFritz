// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Segmented.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// Segmented — small tab group. Selected = surface-4; with `accent`
// the active option uses the brand gradient instead.
//
// v1.54.0: the whole group can be `disabled`, and so can one option, with a
// `title` that says why (Replay's lengths over this PC's memory ceiling).
// v1.58.0: `blurOnPointer` hands focus back to the page after a mouse or
// touch click, so Space reaches push-to-modulate again (Push to modulate's
// Hold to apply / Hold to bypass).
// v1.61.0 (Follow OBS): `ariaLabel` names the group, which is then a radio
// group by the usual keys: `role="radiogroup"` with that name, each option
// a `radio` with `aria-checked`; Tab reaches the checked one only (the
// first one that can be picked when none is), and the arrows move to the
// next or previous option that can be picked, round the ends, and pick it.
// Without it, the tab list it has always been, every host before v1.61.0
// unchanged.

import type { JSX } from "solid-js";
import { For } from "solid-js";
import { blurOnPointer } from "./blurOnPointer";

export type SegmentedOption<T extends string> =
  | T
  | { value: T; label: string; disabled?: boolean; title?: string };

export interface SegmentedProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (next: T) => void;
  accent?: boolean;
  disabled?: boolean;
  /** A pointer click leaves the focus on the page, not on the option. */
  blurOnPointer?: boolean;
  /** v1.61.0: the group's name, which makes it a radio group ("Follow OBS
   *  for Uptime"). */
  ariaLabel?: string;
}

/** The keys that move a radio group's choice: on, or back. */
const NEXT_KEYS: readonly string[] = ["ArrowRight", "ArrowDown"];
const BACK_KEYS: readonly string[] = ["ArrowLeft", "ArrowUp"];

export function Segmented<T extends string>(props: SegmentedProps<T>): JSX.Element {
  let group: HTMLDivElement | undefined;
  const valueOf = (o: SegmentedOption<T>): T => (typeof o === "string" ? o : o.value);
  const offOf = (o: SegmentedOption<T>): boolean => !!props.disabled || (typeof o !== "string" && !!o.disabled);
  const radio = (): boolean => props.ariaLabel !== undefined;
  // A radio group's one tab stop: the checked option, else the first one
  // that can be picked.
  const tabStop = (): number => {
    const checked = props.options.findIndex((o) => valueOf(o) === props.value && !offOf(o));
    return checked >= 0 ? checked : props.options.findIndex((o) => !offOf(o));
  };
  /** From option `at`, the next (`step` 1) or previous (-1) one that can be
   *  picked, round the ends; `at` itself when it's the only one. */
  const stepFrom = (at: number, step: 1 | -1): number => {
    const n = props.options.length;
    for (let k = 1; k <= n; k++) {
      const j = (((at + step * k) % n) + n) % n;
      const o = props.options[j];
      if (o !== undefined && !offOf(o)) return j;
    }
    return at;
  };
  const onKeyDown = (e: KeyboardEvent, at: number): void => {
    if (!radio()) return;
    const step = NEXT_KEYS.includes(e.key) ? 1 : BACK_KEYS.includes(e.key) ? -1 : 0;
    if (step === 0) return;
    e.preventDefault();
    const j = stepFrom(at, step);
    const o = props.options[j];
    if (j === at || o === undefined) return;
    group?.querySelectorAll<HTMLButtonElement>("button")[j]?.focus();
    if (valueOf(o) !== props.value) props.onChange(valueOf(o));
  };
  return (
    <div
      ref={group}
      class="seg"
      role={radio() ? "radiogroup" : "tablist"}
      aria-label={props.ariaLabel}
    >
      <For each={props.options}>
        {(o, i) => {
          const value = valueOf(o);
          const label = typeof o === "string" ? o : o.label;
          const off = () => offOf(o);
          const on = () => props.value === value;
          return (
            <button
              type="button"
              role={radio() ? "radio" : "tab"}
              aria-selected={radio() ? undefined : on()}
              aria-checked={radio() ? on() : undefined}
              tabindex={radio() ? (i() === tabStop() ? 0 : -1) : undefined}
              class={`seg-btn ${on() ? "is-on" : ""} ${props.accent ? "is-accent" : ""}`}
              disabled={off()}
              title={typeof o === "string" ? undefined : o.title}
              onClick={(e) => {
                if (props.blurOnPointer) blurOnPointer(e);
                if (!off()) props.onChange(value);
              }}
              onKeyDown={(e) => onKeyDown(e, i())}
            >
              {label}
            </button>
          );
        }}
      </For>
    </div>
  );
}
