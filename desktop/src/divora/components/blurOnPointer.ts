// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/blurOnPointer.ts at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// blurOnPointer — a mouse or touch click hands focus back to the page.
//
// A button keeps the focus a mouse press gave it, and a focused button
// takes Space and Enter. Space is the in-app push-to-modulate key, which
// App's listener acts on only when the page itself has the focus, so a
// control clicked with the mouse would otherwise swallow the next press, or
// worse, press itself again (v1.56.0's Mute button, the first to need this).
// v1.58.0: shared by that button, Push to modulate's controls, and any
// Toggle or Segmented given `blurOnPointer`.
//
// A click from the keyboard (Tab, then Space or Enter) has `detail` 0 and
// keeps the focus where it is, so the keyboard keeps working the control.

/** Blur the clicked element when the click came from a pointer. */
export function blurOnPointer(e: MouseEvent & { currentTarget: HTMLElement }): void {
  if (e.detail > 0) e.currentTarget.blur();
}
