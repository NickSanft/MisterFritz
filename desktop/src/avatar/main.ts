/**
 * The avatar window's page: a transparent three.js scene with Fritz
 * in it, drawn by the paced loop, answering the shell's handshake.
 *
 * The rules the spike exists to test, kept here so they stay true:
 * - no requestAnimationFrame and no CSS animation, in anything this page
 *   loads (scripts/check-paced.mjs, scripts/check-no-css-animation.mjs);
 * - a transparent canvas, cleared to alpha 0, with premultiplied alpha;
 * - nothing on this page asks for focus. Input is only handled in Arrange,
 *   when the shell has made the window take clicks.
 *
 * At every fit the page tells the shell where his body is on it
 * (hitShape.ts), and again every few seconds from the same tick while it
 * draws, so the shell can tell a click on him from one beside him.
 *
 * The speech bubble over his head (BubbleView.ts) is drawn from the same
 * tick: the turn it shows is state.ts's, moved on by the brain's frames and
 * by `companion:say` (the shell's bridge, from Settings › Brain's Ask and
 * the prompt), each answered with whether it went on to the brain.
 */
import "../divora/tokens.css";
import "../divora/fonts.css";
import "./bubble.css";
import * as THREE from "three";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { PacedLoop } from "./PacedLoop";
import { VrmActor } from "./VrmActor";
import { watchTheme } from "../divora/theme";
import type { Theme } from "../shared/settings";
import { tintFor } from "./tint";
import { VRMUtils } from "@pixiv/three-vrm";
import { HEADROOM, MARGIN, frameSpan } from "./framing";
import { keyAction } from "./keys";
import { POWER_EVENT, afterAsking, fetchDisplayOn, parseDisplay, shouldDraw } from "./run";
import { AHEAD, CURSOR_EVENT, fetchCursor, gaze, parseCursor, type CursorFrame, type Gaze } from "./look";
import { CompanionClient } from "../shared/companion";
import { SAY_EVENT, answerSay, onSay, startCompanion, takeSay } from "./companionLink";
import { BubbleClock, anchorAbove, charCount, plainText } from "./bubble";
import { createBubble } from "./BubbleView";
import { INITIAL, cancelFirst, reduce, type TurnEvent, type TurnView } from "./state";
import { HitPublisher, shapeNow, stamped } from "./hitShape";
import { fetchPage, hitArgs, sendHitRects } from "./hitWire";
import {
  NO_READY_GLOBAL,
  RELOAD_EVENT,
  STATUS_EVENT,
  fetchOwnersModel,
  reportLoaded,
  vrmFetchError,
  type Wearing,
  pageInstance,
  parseStatus,
  requestLock,
  sendBeat,
  sendHandshake,
  type AvatarStatus,
} from "./status";

const FPS = 30;
const FOV = 22;
/** How often the page beats while arranging; the lease misses three. */
const BEAT_MS = 1000;

const instance = pageInstance(location.search);
const canvas = document.getElementById("stage") as HTMLCanvasElement;

const renderer = new THREE.WebGLRenderer({
  canvas,
  alpha: true,
  antialias: true,
  premultipliedAlpha: true,
  // An always-on companion should sit on the integrated GPU where it can.
  powerPreference: "low-power",
});
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;

/** Device pixels per CSS pixel, capped: follows the monitor he stands on. */
function pixelRatio(): number {
  return Math.min(window.devicePixelRatio || 1, 2);
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 20);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, Math.PI);
sun.position.set(0.6, 1.2, 1.5);
scene.add(sun);

let actor: VrmActor | null = null;
let status: AvatarStatus | null = null;
let displayOn = true;
/** Display events heard, so a late reply to the start-up question can't
 * undo a newer event. */
let displayEvents = 0;
/** The last cursor the shell sent (or answered with), and how many it has
 * sent: the gaze is worked out from it every frame, so it holds before the
 * model has loaded, and follows his head when Arrange resizes him. */
let cursorFrame: CursorFrame | null = null;
let cursorEvents = 0;
let paintIn = -1; // frames until the painted handshake; -1 = not due
let lastBeat = 0;
/** The owner's theme, once heard, and the motion it allows: kept, so a model
 * loaded later is tinted and paced the same. */
let theme: Theme | null = null;
let motionFactor = 1;
/** The device's one session with the brain (the control window has none:
 * it hears the link through the shell). */
const companion = new CompanionClient();
/** The turn the bubble shows, the bubble, and its clock (moved only by the
 * tick, so it waits while he is parked). */
let turn: TurnView = INITIAL;
const bubble = createBubble(document);
const bubbleClock = new BubbleClock();
let wasOnline = false;
/** His shape on the page, sent to the shell as it changes; and whether the
 * WebGL context is lost (a GPU reset), when nothing of him is drawn. */
const hits = new HitPublisher(
  stamped(fetchPage, (page, s) => sendHitRects(hitArgs(instance, page, s))),
  () => performance.now(),
);
let contextLost = false;

/** Move the turn on, and the bubble with it: up for a new turn, held once
 * it ends. */
function apply(e: TurnEvent): void {
  const was = turn;
  turn = reduce(turn, e);
  if (turn === was) return;
  if (turn.turnId !== was.turnId) bubbleClock.show();
  if (turn.ended && (!was.ended || turn.turnId !== was.turnId)) {
    bubbleClock.end(charCount(plainText(turn.reply)) + charCount(turn.apology ?? ""));
  }
}

/** Ask the brain: a turn still running is cancelled first (barge-in; the
 * brain starts the new one once the old has stopped). Whether it went on. */
function ask(text: string): boolean {
  const old = cancelFirst(turn);
  if (old !== null) companion.cancel(old);
  const id = companion.say(text);
  if (id === null) {
    console.warn("avatar: couldn't ask the brain (not online, or nothing to ask)");
    return false;
  }
  apply({ kind: "asked", turnId: id });
  return true;
}

for (const type of ["state", "token", "progress", "done", "error"] as const) {
  companion.on(type, (frame) => apply({ kind: "frame", frame }));
}
// The link lost mid-turn: the brain sends nothing more for it.
companion.onLink((s) => {
  if (wasOnline && s.link !== "online") apply({ kind: "lost" });
  wasOnline = s.link === "online";
});

/** Tint `a` and set its motion from the theme (none heard yet: as he was). */
function dress(a: VrmActor): void {
  a.setMotion(motionFactor);
  if (theme) {
    const t = tintFor(theme);
    a.setTint(t.outline, t.rim, t.rimStrength);
  }
}

function fit(): void {
  if (renderer.getPixelRatio() !== pixelRatio()) renderer.setPixelRatio(pixelRatio());
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  if (actor) {
    const { bottom, top } = actor.bounds();
    const f = frameSpan(bottom, top, FOV, MARGIN, HEADROOM);
    camera.position.set(0, f.centerY, f.distance);
    camera.lookAt(0, f.centerY, 0);
  }
  camera.updateProjectionMatrix();
  // setSize clears the drawing buffer: draw once now, or a resize in Arrange
  // shows a blank window until the next paced tick. One event-driven frame,
  // not a loop, so the paced-loop rule holds; never while he is parked.
  if (loop.running) renderer.render(scene, camera);
  offerShape();
}

/** Tell the shell where his body is on the page now: after every fit (a
 * model loaded, a resize, a new scale), parked or not, so the shape is
 * there when he is shown. None while there is no model, or nothing of him
 * can be drawn. */
function offerShape(): void {
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  const a = actor;
  hits.offer(shapeNow(a !== null && !contextLost, () => a!.hitRects(camera, w, h)), { width: w, height: h });
}

/** Moving onto a monitor with another scale changes devicePixelRatio without
 * always firing `resize`; the media query re-arms itself on each change. */
function watchPixelRatio(): void {
  window
    .matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
    .addEventListener("change", () => {
      fit();
      watchPixelRatio();
    }, { once: true });
}

/** Where the cursor says to look, from his eyes' place on the page now,
 * measured in the part of the window he stands in. */
function lookFor(c: CursorFrame, a: VrmActor): Gaze {
  if (!c.active) return AHEAD;
  const view = { width: window.innerWidth, height: window.innerHeight };
  return gaze(c, view, a.headScreen(camera, view.width, view.height), view.height * (1 - HEADROOM));
}

/** The bubble this frame: its clock moved on by the frame's time (held
 * while the cursor is over his window), drawn over his head. */
function paintBubble(dt: number): void {
  const hovered = cursorFrame !== null && cursorFrame.active && cursorFrame.over;
  const env = bubbleClock.tick(dt * 1000, hovered, motionFactor);
  const view = { width: window.innerWidth, height: window.innerHeight };
  const anchor = actor
    ? actor.headScreen(camera, view.width, view.height, anchorAbove(actor.crownAboveHead))
    : { x: view.width / 2, y: view.height * HEADROOM };
  bubble.paint(turn, env, anchor, view);
}

const loop = new PacedLoop((dt) => {
  if (actor) {
    actor.setPhase(turn.phase);
    if (cursorFrame) actor.setLook(lookFor(cursorFrame, actor));
  }
  actor?.update(dt);
  renderer.render(scene, camera);
  hits.keep();
  paintBubble(dt);
  if (paintIn > 0 && --paintIn === 0) {
    paintIn = -1;
    void sendHandshake(instance, "painted");
  }
  if (status?.mode === "arrange" && performance.now() - lastBeat >= BEAT_MS) {
    lastBeat = performance.now();
    void sendBeat(instance, status.arrangeGen);
  }
}, FPS);

function onStatus(next: AvatarStatus | null): void {
  if (!next || next.instance !== instance) return;
  const was = status;
  status = next;
  document.body.classList.toggle("arranging", next.mode === "arrange");
  if (next.phase === "shown" && was?.phase !== "shown") paintIn = 2; // two frames drawn since shown
  if (next.mode === "arrange" && was?.mode !== "arrange") lastBeat = 0; // beat at once
  draw();
}

/** Start or stop the loop as `shouldDraw` says: parked off-screen, or with
 * the display off, it draws nothing, so the compositor has nothing to
 * present. Only a change starts or stops it. */
function draw(): void {
  const on = shouldDraw(status, displayOn);
  if (on && !loop.running) {
    // The shell may have counted his shape as gone while nothing drew.
    hits.resume();
    loop.start();
  } else if (!on && loop.running) loop.stop();
}

/** The placeholder that ships with the app. */
async function placeholderBytes(): Promise<ArrayBuffer> {
  const r = await fetch("/placeholder.vrm");
  return r.arrayBuffer();
}

/** His model, else the placeholder; and the shell is told which, with why
 * when an export that is there couldn't be used. */
async function loadActor(): Promise<void> {
  let next: VrmActor | null = null;
  let error: string | null = null;
  let owners: ArrayBuffer | null = null;
  try {
    owners = await fetchOwnersModel();
  } catch (e) {
    error = vrmFetchError(e); // null: there's no export, which is no error
  }
  if (owners) {
    try {
      next = await VrmActor.load(owners);
    } catch (e) {
      console.error("avatar: couldn't load the owner's model; using the placeholder", e);
      error = e instanceof Error ? e.message : String(e);
    }
  }
  const wearing: Wearing = next ? "export" : "placeholder";
  next ??= await VrmActor.load(await placeholderBytes());
  // The old model goes only once the new one is ready, so he is never
  // missing for longer than a frame.
  const old = actor;
  if (old) {
    scene.remove(old.vrm.scene);
    VRMUtils.deepDispose(old.vrm.scene);
  }
  actor = next;
  dress(next);
  scene.add(next.vrm.scene);
  fit();
  await reportLoaded(wearing, error);
}

/** Loads one at a time, in the order asked for: two quick reloads can't
 * leave two models in the scene. */
let loading: Promise<void> = Promise.resolve();
function load(): Promise<void> {
  loading = loading.then(loadActor, loadActor);
  return loading;
}

// Arrange only: the shell has lifted click-through and given the window the
// keyboard. Drag to move, the corner grip to resize, Esc to lock.
function arranging(): boolean {
  return status?.mode === "arrange";
}
document.addEventListener("mousedown", (e) => {
  if (!arranging() || e.button !== 0) return;
  const grip = (e.target as Element | null)?.closest("#grip");
  void (grip ? getCurrentWindow().startResizeDragging("SouthEast") : getCurrentWindow().startDragging());
});
document.addEventListener("keydown", (e) => {
  const action = keyAction(e, arranging());
  if (action === "ignore") return;
  e.preventDefault();
  if (action === "lock") void requestLock();
});
// WebView2's own menu (Back, Reload, Print, Save as) is never offered.
document.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("resize", fit);
watchPixelRatio();
// A GPU reset: nothing of him is drawn until the context is back, so a
// click on where he was is no longer his.
canvas.addEventListener("webglcontextlost", () => {
  contextLost = true;
  offerShape();
});
canvas.addEventListener("webglcontextrestored", () => {
  contextLost = false;
  fit();
});

async function main(): Promise<void> {
  fit();
  await listen<unknown>(STATUS_EVENT, (e) => onStatus(parseStatus(e.payload)));
  await listen<unknown>(POWER_EVENT, (e) => {
    const on = parseDisplay(e.payload);
    if (on === null) return;
    displayEvents += 1;
    displayOn = on;
    draw();
  });
  // His eyes follow the cursor (the shell sends it only on a move, while he
  // is seen and the owner wants it; `active: false` when that stops). Kept,
  // and looked at from the next frame on.
  await listen<unknown>(CURSOR_EVENT, (e) => {
    const c = parseCursor(e.payload);
    if (!c) return;
    cursorEvents += 1;
    cursorFrame = c;
  });
  // The theme: the page's tokens (for what it draws in HTML) and his tint
  // and motion. Listening is awaited; the answer to the question isn't, so
  // nothing new stands between the page and its ready handshake.
  await watchTheme(document.documentElement, (t, motion) => {
    theme = t;
    motionFactor = motion.factor;
    if (actor) dress(actor);
  });
  // Where it is now: a cursor that hasn't moved since he opened sends
  // nothing, and an event before this page listened was lost.
  const cursorHeard = cursorEvents;
  const asked = await fetchCursor();
  if (asked) cursorFrame = afterAsking<CursorFrame | null>(asked, cursorHeard, cursorEvents, cursorFrame);
  // The state now, in case the display went off before the page listened.
  const heard = displayEvents;
  displayOn = afterAsking(await fetchDisplayOn(), heard, displayEvents, displayOn);
  draw();
  // Listening: tell the shell, which shows the window. The model may still be
  // loading; the window is transparent until it is.
  if ((window as unknown as Record<string, unknown>)[NO_READY_GLOBAL] !== true) {
    await sendHandshake(instance, "ready");
  }
  // The brain, once the shell has the page: nothing it waits for stands
  // between the page and its handshake. Asked through the shell's bridge,
  // listened for first, so no ask is lost to a connection made before it.
  await listen<unknown>(SAY_EVENT, (e) => {
    void onSay(e.payload, takeSay, ask, answerSay).then((done) => {
      if (done === "late") console.warn("avatar: the shell had stopped waiting for a turn, so it wasn't asked");
    });
  });
  void startCompanion(companion);
  await listen(RELOAD_EVENT, () => void load());
  await load();
}

void main();
