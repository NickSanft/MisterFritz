// Lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors),
// src/components/HotkeyCapture.tsx at 62ed6a5, unchanged below this header. See
// desktop/LICENSES/DivoraVoiceMod.txt.

// HotkeyCapture — dashed field that captures the next keychord into
// Kbd chips. Click to arm; Esc cancels. Used by Settings → Hotkeys, the
// Soundboard's tile hotkeys and, v1.59.0, each timer's keys in
// Settings → Hourglass (v1.60.0: a countdown's third is End now).
//
// v1.59.0: the window `keydown` listener a click arms is held here, in the
// component, and removed however the capture ends: a key, Esc, a second
// click on the field, the focus leaving the field (a click elsewhere on the
// page, or another capture armed; a Tab is captured as the key), or the
// field going away while armed. Before, the cleanup was registered inside
// the click handler, where Solid has no owner and drops it, so a field
// removed while armed (or clicked again to cancel) left its listener
// behind, and the next key went to it. Hourglass puts up to 48 of these on
// one screen (16 countdowns, three keys each), each beside a Delete. A
// capture armed by a pointer gives the focus back to the page once it
// ends, so Space reaches push-to-modulate again (`blurOnPointer`'s rule);
// armed from the keyboard, the focus stays.

import type { JSX } from "solid-js";
import { createSignal, For, onCleanup, Show } from "solid-js";
import { Kbd } from "./Kbd";

export interface HotkeyCaptureProps {
  value: string[] | null;
  onChange: (next: string[]) => void;
  /** v1.59.0: the capture's accessible name in each state, for a row with
   *  many captures ("BRB start or pause hotkey: Ctrl Alt F13", "… not set",
   *  "… press a key, Esc to cancel"). Without it the name is the content,
   *  "Click to set" when unset, as before. */
  ariaLabel?: (state: "set" | "unset" | "capturing") => string;
}

const MODIFIER_KEYS = new Set(["Control", "Alt", "Shift", "Meta"]);

function formatKey(e: KeyboardEvent): string[] {
  const parts: string[] = [];
  if (e.ctrlKey) parts.push("Ctrl");
  if (e.altKey) parts.push("Alt");
  if (e.shiftKey) parts.push("Shift");
  if (e.metaKey) parts.push("Win");
  let k = e.key;
  if (k === " ") k = "Space";
  else if (k.length === 1) k = k.toUpperCase();
  if (!MODIFIER_KEYS.has(e.key)) parts.push(k);
  return parts;
}

export function HotkeyCapture(props: HotkeyCaptureProps): JSX.Element {
  const [capturing, setCapturing] = createSignal(false);
  let button: HTMLButtonElement | undefined;
  // The listener while armed; null otherwise.
  let active: ((e: KeyboardEvent) => void) | null = null;
  // Whether a pointer armed the capture now running.
  let byPointer = false;

  const disarm = (): void => {
    if (active) window.removeEventListener("keydown", active);
    active = null;
  };
  // At the component's level, so it runs when the field goes while armed.
  onCleanup(disarm);

  /** Ends the capture; a pointer-armed one hands the focus back. */
  const end = (): void => {
    disarm();
    setCapturing(false);
    if (byPointer) button?.blur();
  };

  const onArm = (e: MouseEvent): void => {
    if (capturing()) {
      // The cancel click. From a pointer, it hands the focus back too.
      byPointer = byPointer || e.detail > 0;
      end();
      return;
    }
    byPointer = e.detail > 0;
    disarm();
    const handler = (ev: KeyboardEvent): void => {
      ev.preventDefault();
      if (ev.key === "Escape") {
        end();
        return;
      }
      if (MODIFIER_KEYS.has(ev.key)) return;
      const keys = formatKey(ev);
      end();
      props.onChange(keys);
    };
    active = handler;
    window.addEventListener("keydown", handler);
    setCapturing(true);
  };

  const state = (): "set" | "unset" | "capturing" =>
    capturing() ? "capturing" : props.value && props.value.length > 0 ? "set" : "unset";

  return (
    <button
      ref={button}
      type="button"
      class={`hotkey-capture ${capturing() ? "is-capturing" : ""}`}
      aria-label={props.ariaLabel?.(state())}
      onClick={onArm}
      onBlur={() => {
        // The focus went elsewhere (a click on the page, another capture; a
        // Tab doesn't move it, being captured as the key): the key typed
        // there is not this field's. `end` clears `active` before its own
        // blur, so this doesn't run twice.
        if (active) {
          disarm();
          setCapturing(false);
        }
      }}
    >
      <Show
        when={capturing()}
        fallback={
          <Show
            when={props.value && props.value.length > 0}
            fallback={<span class="hotkey-capture-hint">Click to set</span>}
          >
            <For each={props.value ?? []}>{(k) => <Kbd>{k}</Kbd>}</For>
          </Show>
        }
      >
        <span class="hotkey-capture-hint">Press a key… (Esc to cancel)</span>
      </Show>
    </button>
  );
}
