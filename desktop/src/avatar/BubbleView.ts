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
 * - While push-to-talk is muted (the tray's Mute push-to-talk), the muted
 *   sign shows over his head the whole time: in the bubble's corner while
 *   it shows, else on its own where the bubble's tail would point.
 * - While he listens, DivoraVoice's level meter shows under *Listening…*:
 *   shown or hidden with the turn (so the bubble is measured as ever), its
 *   level and cap moved at every paint (meter.ts).
 *
 * Lifted in part from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora
 * Contributors) at 62ed6a5: the `muted` sigil's marks, in `mutedSigil`,
 * from src/components/Sigil.tsx; and `HMeter`'s three elements (a level and
 * a cap in a track), from src/components/Meters.tsx. See
 * desktop/LICENSES/DivoraVoiceMod.txt. The rest is Fritz's own.
 */
import { EDGE, GAP, layout, plainText, type Envelope } from "./bubble";
import type { MeterView } from "./meter";
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

/** Shown while push-to-talk is held and he listens. */
export const LISTENING = "Listening…";

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
    if (!v.ended) body = v.phase === "listening" ? LISTENING : THINKING;
    else body = v.apology === null && note === null ? NOTHING_TO_SAY : "";
  }
  return { progress: v.ended ? null : v.progress, body, apology: v.apology, note };
}

export interface Bubble {
  readonly element: HTMLElement;
  /** The muted sign shown on its own, while no bubble shows. */
  readonly sign: HTMLElement;
  /** Draw `v` as `env` says, over his head at `anchor`, in a window `view`
   * big (CSS pixels), with the muted sign while `muted`, and `meter` on the
   * meter while he listens (empty without it). */
  paint(
    v: TurnView,
    env: Envelope,
    anchor: { x: number; y: number },
    view: { width: number; height: number },
    muted?: boolean,
    meter?: MeterView | null,
  ): void;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** DivoraVoice's `muted` sigil (src/divora/sigils/Sigil.tsx), a circle
 * struck through, built as DOM: his page has no Solid. Its marks are the
 * Sigil's own (mutedSign.test.tsx). */
export function mutedSigil(doc: Document, size: number): SVGSVGElement {
  const make = (tag: string, attrs: Record<string, string>): SVGElement => {
    const e = doc.createElementNS(SVG_NS, tag) as SVGElement;
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };
  const svg = make("svg", {
    width: String(size),
    height: String(size),
    viewBox: "0 0 24 24",
    class: "sigil",
    "aria-hidden": "true",
  }) as SVGSVGElement;
  const g = make("g", { fill: "none", stroke: "currentColor", "stroke-width": "1.5", "stroke-linecap": "round" });
  g.appendChild(make("circle", { cx: "12", cy: "12", r: "7" }));
  g.appendChild(make("path", { d: "M7 7 L17 17" }));
  svg.appendChild(g);
  return svg;
}

/** The sign on its own: its plate's side (CSS pixels; bubble.css's
 * `.muted-sign`). */
export const SIGN_SIZE = 26;

/** Where the sign stands on its own: centred over `anchor`, its bottom
 * `GAP` above it, where the bubble's tail would end, and `EDGE` inside the
 * window, as the bubble keeps (pure, for the tests). */
export function signAt(anchor: { x: number; y: number }, view: { width: number; height: number }): { left: number; top: number } {
  const ax = Number.isFinite(anchor.x) ? anchor.x : view.width / 2;
  const ay = Number.isFinite(anchor.y) ? anchor.y : 0;
  const within = (v: number, room: number) => Math.min(Math.max(v, EDGE), Math.max(EDGE, room - SIGN_SIZE - EDGE));
  return { left: within(ax - SIGN_SIZE / 2, view.width), top: within(ay - GAP - SIGN_SIZE, view.height) };
}

/** Whether the turn is his listening: the meter shows. */
const listens = (v: TurnView): boolean => v.phase === "listening" && !v.ended;
const clamp01 = (n: number): number => (Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0);

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
  const meter = el("div", "hmeter bubble-meter", content);
  const meterLevel = el("div", "hmeter-level", meter);
  const meterPeak = el("div", "hmeter-peak", meter);
  const apology = el("div", "bubble-apology", content);
  el("span", "orb warn", apology);
  const apologyText = el("span", "bubble-apology-text", apology);
  const note = el("div", "bubble-note", content);
  const corner = el("div", "bubble-muted", root);
  corner.appendChild(mutedSigil(doc, 14));
  const sign = el("div", "muted-sign", parent);
  sign.appendChild(mutedSigil(doc, 16));

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
  style(corner, "corner", "display", "none");
  style(sign, "sign", "display", "none");

  return {
    element: root,
    sign,
    paint(v, env, anchor, view, muted = false, meter_ = null) {
      if (!env.visible) {
        style(root, "root", "display", "none");
        style(sign, "sign", "display", muted ? "flex" : "none");
        if (muted) {
          const at = signAt(anchor, view);
          style(sign, "sign", "left", `${round(at.left)}px`);
          style(sign, "sign", "top", `${round(at.top)}px`);
        }
        return;
      }
      style(sign, "sign", "display", "none");
      style(corner, "corner", "display", muted ? "flex" : "none");
      style(root, "root", "display", "flex");
      const fresh = v.revision !== revision;
      if (fresh) {
        revision = v.revision;
        const t = bubbleText(v);
        row(progress, "progress", "flex", progressText, t.progress);
        row(text, "text", "block", text, t.body);
        style(meter, "meter", "display", listens(v) ? "block" : "none");
        row(apology, "apology", "flex", apologyText, t.apology);
        row(note, "note", "block", note, t.note);
      }
      if (listens(v)) {
        const m = meter_ ?? { level: 0, peak: 0 };
        style(meterLevel, "meterLevel", "width", `${round(clamp01(m.level) * 100, 1)}%`);
        // DivoraVoice's cap shows above 2%.
        style(meterPeak, "meterPeak", "display", m.peak > 0.02 ? "block" : "none");
        style(meterPeak, "meterPeak", "left", `calc(${round(clamp01(m.peak) * 100, 1)}% - 1px)`);
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
