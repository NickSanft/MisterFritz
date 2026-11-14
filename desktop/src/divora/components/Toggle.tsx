// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Toggle.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// Toggle — 40x23 switch with sliding knob. Gradient track when on;
// `tone` lets you swap the on-tint to danger or success.
//
// v1.54.0: `disabled` (Settings' screen-recording rows while Replay is on).
// v1.58.0: `blurOnPointer` hands focus back to the page after a mouse or
// touch click, so Space reaches push-to-modulate again (Push to modulate's
// own switch); `ariaDescribedBy` names the line that says what it does.

import type { JSX } from "solid-js";
import { blurOnPointer } from "./blurOnPointer";

export interface ToggleProps {
  on: boolean;
  onChange?: (next: boolean) => void;
  tone?: "" | "danger" | "success";
  ariaLabel?: string;
  disabled?: boolean;
  title?: string;
  /** A pointer click leaves the focus on the page, not on the switch. */
  blurOnPointer?: boolean;
  ariaDescribedBy?: string;
}

export function Toggle(props: ToggleProps): JSX.Element {
  return (
    <button
      type="button"
      class={`toggle ${props.on ? "is-on" : ""} ${props.tone ? `tone-${props.tone}` : ""}`}
      role="switch"
      aria-checked={props.on}
      aria-label={props.ariaLabel}
      aria-describedby={props.ariaDescribedBy}
      disabled={props.disabled}
      title={props.title}
      onClick={(e) => {
        if (props.blurOnPointer) blurOnPointer(e);
        if (!props.disabled) props.onChange?.(!props.on);
      }}
    >
      <span class="toggle-knob" />
    </button>
  );
}
