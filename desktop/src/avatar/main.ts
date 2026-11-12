/**
 * The avatar window's page: a transparent three.js scene with Fritz
 * in it, drawn by the paced loop, answering the shell's handshake.
 *
 * The rules the spike exists to test, kept here so they stay true:
 * - no requestAnimationFrame and no CSS animation (scripts/check-paced.mjs);
 * - a transparent canvas, cleared to alpha 0, with premultiplied alpha;
 * - nothing on this page asks for focus. Input is only handled in Arrange,
 *   when the shell has made the window take clicks.
 */
import * as THREE from "three";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { PacedLoop } from "./PacedLoop";
import { VrmActor } from "./VrmActor";
import { frameSpan } from "./framing";
import { keyAction } from "./keys";
import { POWER_EVENT, afterAsking, fetchDisplayOn, parseDisplay, shouldDraw } from "./run";
import { AHEAD, CURSOR_EVENT, fetchCursor, gaze, parseCursor, type CursorFrame, type Gaze } from "./look";
import {
  NO_READY_GLOBAL,
  STATUS_EVENT,
  fetchOwnersModel,
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

function fit(): void {
  if (renderer.getPixelRatio() !== pixelRatio()) renderer.setPixelRatio(pixelRatio());
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  if (actor) {
    const { bottom, top } = actor.bounds();
    const f = frameSpan(bottom, top, FOV);
    camera.position.set(0, f.centerY, f.distance);
    camera.lookAt(0, f.centerY, 0);
  }
  camera.updateProjectionMatrix();
  // setSize clears the drawing buffer: draw once now, or a resize in Arrange
  // shows a blank window until the next paced tick. One event-driven frame,
  // not a loop, so the paced-loop rule holds; never while he is parked.
  if (loop.running) renderer.render(scene, camera);
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

/** Where the cursor says to look, from his eyes' place on the page now. */
function lookFor(c: CursorFrame, a: VrmActor): Gaze {
  if (!c.active) return AHEAD;
  const view = { width: window.innerWidth, height: window.innerHeight };
  return gaze(c, view, a.headScreen(camera, view.width, view.height));
}

const loop = new PacedLoop((dt) => {
  if (actor && cursorFrame) actor.setLook(lookFor(cursorFrame, actor));
  actor?.update(dt);
  renderer.render(scene, camera);
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
  if (on && !loop.running) loop.start();
  else if (!on && loop.running) loop.stop();
}

async function modelBytes(): Promise<ArrayBuffer> {
  try {
    return await fetchOwnersModel();
  } catch {
    // No fritz.vrm in the app's data folder (or it can't be read): the
    // placeholder that ships with the app.
  }
  const r = await fetch("/placeholder.vrm");
  return r.arrayBuffer();
}

async function loadActor(): Promise<void> {
  try {
    actor = await VrmActor.load(await modelBytes());
  } catch (e) {
    console.error("avatar: couldn't load the model; using the placeholder", e);
    const r = await fetch("/placeholder.vrm");
    actor = await VrmActor.load(await r.arrayBuffer());
  }
  scene.add(actor.vrm.scene);
  fit();
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
  await loadActor();
}

void main();
