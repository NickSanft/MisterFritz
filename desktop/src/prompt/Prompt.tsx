/**
 * The prompt (Ask Fritz): one field to ask him something, and a line on
 * whether he can be asked now. Enter asks; the shell closes the prompt once
 * the question is taken, and a refusal leaves the draft with the shell's
 * words. Esc closes it and keeps the draft. What keys do is keys.ts's; what
 * the line says is view.ts's.
 *
 * Every time the window gets the keyboard, the field takes it, with the
 * caret at the end, and any refusal and armed key are forgotten; a question
 * still out from the open before is let go, its answer kept from this one
 * (but a question taken leaves the field, if it is still what's there).
 * A click anywhere but the field leaves the keyboard in it, so Enter and Esc
 * always work.
 */
import { createEffect, createMemo, createSignal, on, onCleanup, onMount, Show, type JSX } from "solid-js";
import { Sigil } from "../divora/sigils/Sigil";
import { promptKey, type Armed } from "./keys";
import type { PromptStatus } from "./view";

export interface PromptProps {
  status: () => PromptStatus;
  /** Ask him: null once taken, else why not. */
  say: (text: string) => Promise<string | null>;
  close: () => Promise<void>;
  /** Hear the window getting, and losing, the keyboard; each returns the
   * unsubscribe. */
  onWindowFocus: (cb: () => void) => () => void;
  onWindowBlur: (cb: () => void) => () => void;
}

/** What the line says while the shell decides. */
export const ASKING = "Asking him…";

export function Prompt(props: PromptProps): JSX.Element {
  let field!: HTMLTextAreaElement;
  const [draft, setDraft] = createSignal("");
  const [refusal, setRefusal] = createSignal<string | null>(null);
  const [asking, setAsking] = createSignal(false);
  /** The last question taken, for ↑ (this page's memory only). */
  let last: string | null = null;
  let armed: Armed = null;
  /** Bumped each time the window gets the keyboard: a question's answer
   * counts only in the open it was asked in. */
  let opened = 0;

  const put = (text: string) => {
    setDraft(text);
    field.value = text;
    field.setSelectionRange(text.length, text.length);
  };
  const shown = () => {
    opened += 1;
    armed = null;
    setAsking(false);
    setRefusal(null);
    field.focus();
    field.setSelectionRange(field.value.length, field.value.length);
  };
  const send = async () => {
    if (asking()) return;
    const at = opened;
    const text = field.value;
    setAsking(true);
    setRefusal(null);
    try {
      const why = await props.say(text);
      if (why === null) {
        last = text;
        // Asked: not a draft any more, whichever open it was asked in,
        // unless the owner has typed something else since.
        if (field.value === text) put("");
      }
      if (opened !== at) return;
      if (why !== null) setRefusal(why);
    } finally {
      if (opened === at) setAsking(false);
    }
  };
  const onKey = (type: "keydown" | "keyup") => (e: KeyboardEvent) => {
    const out = promptKey(type, e, { armed, empty: field.value === "" });
    armed = out.armed;
    if (out.prevent) e.preventDefault();
    if (out.action === "send") void send();
    else if (out.action === "close") void props.close().catch(() => {});
    else if (out.action === "recall" && last !== null) put(last);
  };
  // The keyboard stays in the field: a press anywhere else in the window
  // (the line, the border, the corners) would leave it with the page.
  const keepKeyboard = (e: MouseEvent) => {
    if (e.target !== field) e.preventDefault();
  };
  // A refusal is of the moment it was given: once what the line would say
  // changes (the link, or his window), it says how things stand now. A
  // memo, so a report that changes nothing the line says changes nothing.
  const line = createMemo(() => props.status().text);
  createEffect(on(line, () => setRefusal(null), { defer: true }));

  onMount(() => {
    const offFocus = props.onWindowFocus(shown);
    const offBlur = props.onWindowBlur(() => {
      armed = null;
    });
    document.addEventListener("mousedown", keepKeyboard);
    onCleanup(() => {
      offFocus();
      offBlur();
      document.removeEventListener("mousedown", keepKeyboard);
    });
    shown();
  });

  return (
    <div class="prompt">
      <textarea
        ref={field}
        class="prompt-field"
        rows={2}
        aria-label="Ask Fritz"
        placeholder="Ask Fritz…"
        autocomplete="off"
        spellcheck={true}
        value={draft()}
        onInput={(e) => {
          setDraft(e.currentTarget.value);
          setRefusal(null);
        }}
        onKeyDown={onKey("keydown")}
        onKeyUp={onKey("keyup")}
      />
      <div class="prompt-meta">
        <span class={`prompt-dot ${props.status().tone}`} aria-hidden="true" />
        <Show
          when={refusal()}
          fallback={<span class="prompt-line" role="status">{asking() ? ASKING : props.status().text}</span>}
        >
          {(why) => <span class="prompt-line prompt-refusal" role="alert">{why()}</span>}
        </Show>
        <span class="prompt-local">
          <Sigil name="lock" size={12} />
          LOCAL · NO ACCOUNT
        </span>
      </div>
      <div class="prompt-rule" style={{ visibility: props.status().rule ? "visible" : "hidden" }} />
    </div>
  );
}
