/**
 * The speech bubble on the avatar's page: plain DOM, painted from the paced
 * loop's tick (bubble.css has no motion of its own).
 *
 * - Text is only ever written as text (`textContent`): the reply is the
 *   model's output, never trusted as markup.
 * - Everything it says sits in one clipped column, anchored at the bottom:
 *   when there is more than fits over his head, the oldest lines go off the
 *   top, inside the panel, and the newest (the apology, the note) stay.
 * - A style is written only when its value changed, and the text only when
 *   the turn's `revision` did; the bubble is measured only then (or when the
 *   window's width changes), so a held bubble costs the page nothing.
 * - It takes no input: `pointer-events: none`, nothing focusable, so Arrange
 *   drags through it and it never steals a click from the desktop.
 */
import { layout, plainText, type Envelope } from "./bubble";
import type { TurnView } from "./state";

/** What the bubble shows, worked out from the turn (pure, for the tests). */
export interface BubbleText {
  progress: string | null;
  body: string;
  apology: string | null;
  note: string | null;
}

/** Shown while he thinks and nothing of the reply has come yet. */
export const THINKING = "…";

/** A turn that ended with nothing at all to show (the model said nothing):
 * a line of his own rather than an empty bubble. */
export const NOTHING_TO_SAY = "I find I have nothing whatsoever to say on the matter, sir. Do try again.";

export function bubbleText(v: TurnView): BubbleText {
  const reply = plainText(v.reply);
  const notes: string[] = [];
  if (v.interrupted) notes.push("Interrupted");
  if (v.pictures === 1) notes.push("1 picture came with this; the bubble can't show it yet");
  else if (v.pictures > 1) notes.push(`${v.pictures} pictures came with this; the bubble can't show them yet`);
  const note = notes.length ? notes.join(" · ") : null;
  let body = reply;
  if (reply.trim() === "") {
    if (!v.ended) body = THINKING;
    else body = v.apology === null && note === null ? NOTHING_TO_SAY : "";
  }
  return { progress: v.ended ? null : v.progress, body, apology: v.apology, note };
}

export interface Bubble {
  readonly element: HTMLElement;
  /** Draw `v` as `env` says, over his head at `anchor`, in a window `view`
   * big (CSS pixels). */
  paint(v: TurnView, env: Envelope, anchor: { x: number; y: number }, view: { width: number; height: number }): void;
}

const round = (v: number, places = 0): string => {
  const f = 10 ** places;
  return String(Math.round(v * f) / f);
};

export function createBubble(doc: Document, parent: HTMLElement = doc.body): Bubble {
  const el = (tag: string, cls: string, into: HTMLElement): HTMLElement => {
    const e = doc.createElement(tag);
    e.className = cls;
    into.appendChild(e);
    return e;
  };
  const root = el("div", "bubble", parent);
  const clip = el("div", "bubble-clip", root);
  const content = el("div", "bubble-content", clip);
  const progress = el("div", "bubble-progress", content);
  el("span", "orb", progress);
  const progressText = el("span", "bubble-progress-text", progress);
  const text = el("div", "bubble-text", content);
  const apology = el("div", "bubble-apology", content);
  el("span", "orb warn", apology);
  const apologyText = el("span", "bubble-apology-text", apology);
  const note = el("div", "bubble-note", content);

  const written = new Map<string, string>();
  const style = (e: HTMLElement, name: string, prop: string, value: string): void => {
    const key = `${name}:${prop}`;
    if (written.get(key) === value) return;
    written.set(key, value);
    e.style.setProperty(prop, value);
  };
  const flags = new Map<string, boolean>();
  const toggle = (e: HTMLElement, name: string, cls: string, on: boolean): void => {
    const key = `${name}.${cls}`;
    if (flags.get(key) === on) return;
    flags.set(key, on);
    e.classList.toggle(cls, on);
  };
  /** A row shown as `display` with `value` in `span`, or hidden. */
  const row = (e: HTMLElement, name: string, display: string, span: HTMLElement, value: string | null): void => {
    style(e, name, "display", value === null || value === "" ? "none" : display);
    if (value !== null && span.textContent !== value) span.textContent = value;
  };

  let revision = -1;
  let measuredFor = -1;
  let width = 0;
  let clipFor = "";
  style(root, "root", "display", "none");

  return {
    element: root,
    paint(v, env, anchor, view) {
      if (!env.visible) {
        style(root, "root", "display", "none");
        return;
      }
      style(root, "root", "display", "flex");
      const fresh = v.revision !== revision;
      if (fresh) {
        revision = v.revision;
        const t = bubbleText(v);
        row(progress, "progress", "flex", progressText, t.progress);
        row(text, "text", "block", text, t.body);
        row(apology, "apology", "flex", apologyText, t.apology);
        row(note, "note", "block", note, t.note);
      }
      // Its widest and tallest first, then its own width as laid out in that.
      const room = layout(anchor, view, Number.POSITIVE_INFINITY);
      style(root, "root", "max-width", `${round(room.maxWidth)}px`);
      style(root, "root", "max-height", `${round(room.maxHeight)}px`);
      if (fresh || measuredFor !== view.width) {
        measuredFor = view.width;
        width = root.offsetWidth;
      }
      const at = layout(anchor, view, width);
      // The oldest lines fade out at the top only when there are more than
      // fit: looked at again when the text, the room or the width changes.
      const clipKey = `${revision}:${round(at.maxHeight)}:${view.width}`;
      if (clipKey !== clipFor) {
        clipFor = clipKey;
        toggle(clip, "clip", "clipped", content.offsetHeight > clip.clientHeight + 1);
      }
      toggle(root, "root", "no-tail", at.fallback);
      style(root, "root", "left", `${round(at.left)}px`);
      style(root, "root", "top", at.top === null ? "auto" : `${round(at.top)}px`);
      style(root, "root", "bottom", at.bottom === null ? "auto" : `${round(at.bottom)}px`);
      style(root, "root", "--tail-x", `${round(at.tailX)}px`);
      style(root, "root", "opacity", round(env.opacity, 3));
      style(root, "root", "transform", env.scale >= 1 ? "none" : `scale(${round(env.scale, 3)})`);
      style(root, "root", "filter", env.blurPx <= 0.05 ? "none" : `blur(${round(env.blurPx, 2)}px)`);
    },
  };
}
