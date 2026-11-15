// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/Kbd.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// Kbd — keycap chip. Used for hotkey indicators and keyboard hints.

import type { JSX } from "solid-js";

export interface KbdProps {
  children: JSX.Element;
}

export function Kbd(props: KbdProps): JSX.Element {
  return <span class="kbd">{props.children}</span>;
}
