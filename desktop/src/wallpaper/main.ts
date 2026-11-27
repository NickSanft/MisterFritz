/**
 * The study scene (plan 14, PR 12): one of these pages per monitor, drawn
 * behind the desktop's icons (the shell's wallpaper.rs puts it there). It
 * paints the owner's mood as a slow radial light, inside the paced loop
 * and never with requestAnimationFrame or CSS animation
 * (scripts/check-paced.mjs), and it takes no input
 * (scripts/check-no-input.mjs): the icons above it get every click.
 *
 * It stops while the display is off and while the shell says its monitor
 * is covered (a fullscreen app, the lock screen), and rests as one still
 * frame at motion's functional level.
 */
import "../divora/tokens.css";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { PacedLoop } from "../avatar/PacedLoop";
import { POWER_EVENT, afterAsking, fetchDisplayOn, parseDisplay } from "../avatar/run";
import { watchTheme } from "../divora/theme";
import { glowAt, paint, rampOf, type Ramp } from "./backdrop";
import { COVERED_CMD, COVERED_EVENT, SCENE_FPS, SceneRun } from "./scene";

const canvas = document.getElementById("backdrop") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { alpha: false });

let ramp: Ramp | null = null;
let size = { width: 0, height: 0 };
let frames = 0;
let shown = false;
/** A GPU reset has the canvas's context: nothing drawn shows until it is
 * restored. */
let lost = false;
let covered = false;
let displayOn = true;

const loop = new PacedLoop((dt) => run.tick(dt), SCENE_FPS);
const run = new SceneRun((t) => {
  if (!ctx || !ramp || size.width <= 0 || lost) return;
  paint(ctx, size.width, size.height, ramp, glowAt(t));
  // Hidden until it has a frame in the owner's mood (wallpaper.html), and
  // again while a GPU reset has left it blank.
  if (!shown) {
    canvas.style.visibility = "visible";
    shown = true;
  }
  frames += 1;
}, loop);

// For the live checks, as the avatar's page has its own: what the scene is
// doing. Read-only.
(window as unknown as Record<string, unknown>).__fritzScene = () => ({
  drawing: run.now,
  running: loop.running,
  time: run.time,
  frames,
  covered,
  displayOn,
  width: canvas.width,
  height: canvas.height,
});

/** Size the canvas to the page, in device pixels (capped at 2 per CSS
 * pixel). The window is built at its monitor's size and given its real
 * one when it is moved behind the icons; a size under a pixel (not there
 * yet) keeps the last. */
function fit(): void {
  const width = window.innerWidth;
  const height = window.innerHeight;
  if (width < 1 || height < 1 || !ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  size = { width, height };
  run.redraw();
}

/** A move to a monitor with another scale changes devicePixelRatio without
 * always firing `resize`; the query re-arms itself on each change. */
function watchPixelRatio(): void {
  window
    .matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
    .addEventListener("change", () => {
      fit();
      watchPixelRatio();
    }, { once: true });
}

async function boot(): Promise<void> {
  fit();
  window.addEventListener("resize", fit);
  watchPixelRatio();
  // A GPU reset (a driver update, a TDR) blanks the canvas: it is hidden,
  // so the webview's own mood-coloured background shows, and nothing is
  // drawn into it until the context is back. A restored context has lost
  // its scale too, so it is fitted again, which draws it (a still scene
  // would otherwise wait for its next change).
  canvas.addEventListener("contextlost", () => {
    lost = true;
    shown = false;
    canvas.style.visibility = "hidden";
  });
  canvas.addEventListener("contextrestored", () => {
    lost = false;
    fit();
  });

  await watchTheme(document.documentElement, (t, motion) => {
    ramp = rampOf(getComputedStyle(document.documentElement), t.theme === "light");
    run.setTheme(motion.factor);
  });

  let displayEvents = 0;
  await listen<unknown>(POWER_EVENT, (e) => {
    const on = parseDisplay(e.payload);
    if (on === null) return;
    displayEvents += 1;
    displayOn = on;
    run.setDisplay(on);
  });
  const heardDisplay = displayEvents;
  displayOn = afterAsking(await fetchDisplayOn(), heardDisplay, displayEvents, displayOn);
  run.setDisplay(displayOn);

  // This window's own word: a plain listen would hear every study scene's.
  let coveredEvents = 0;
  await getCurrentWebviewWindow().listen<unknown>(COVERED_EVENT, (e) => {
    if (typeof e.payload !== "boolean") return;
    coveredEvents += 1;
    covered = e.payload;
    run.setCovered(covered);
  });
  const heardCovered = coveredEvents;
  let asked = false;
  try {
    asked = (await invoke<unknown>(COVERED_CMD)) === true;
  } catch {
    // Not asked as a study scene (a plain browser): uncovered.
  }
  covered = afterAsking(asked, heardCovered, coveredEvents, covered);
  run.setCovered(covered);
}

void boot();
