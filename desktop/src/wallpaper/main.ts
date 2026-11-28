/**
 * The study scene (plan 14, PR 12): one of these pages per monitor, drawn
 * behind the desktop's icons (the shell's wallpaper.rs puts it there). It
 * paints the owner's mood as a slow radial light and, on the monitor Fritz
 * stands on, the study circle around him (circle.ts), its threads lit
 * while the brain's widgets are fresh (feed.ts), and the parchment grain
 * while the owner's switch has it (grain.ts). All of it inside the paced
 * loop, never with requestAnimationFrame or CSS animation
 * (scripts/check-paced.mjs), and it takes no input
 * (scripts/check-no-input.mjs): the icons above it get every click.
 *
 * It stops while the display is off and while the shell says its monitor
 * is covered (a fullscreen app, the lock screen), and rests as one still
 * frame at motion's functional level. While the circle eases after Fritz,
 * or fades, the loop runs faster, and slows again once it has.
 */
import "../divora/tokens.css";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { PacedLoop } from "../avatar/PacedLoop";
import { POWER_EVENT, afterAsking, fetchDisplayOn, parseDisplay } from "../avatar/run";
import { watchTheme } from "../divora/theme";
import { COMPANION_STATUS_EVENT, parseCompanionStatus } from "../shared/link";
import { glowAt, paint, rampOf, type Ramp } from "./backdrop";
import { NODES, drawCircle, inkOf, makeGlyphs, nodeAngles, openArc, radiusFor, type Ink } from "./circle";
import { Freshness, NOW_CMD, SEEN_EVENT, STATUS_CMD, parseSeen } from "./feed";
import { FRITZ_CMD, FRITZ_EVENT, Follow, parsePlace } from "./follow";
import { TILE, grainOf, noiseTile, type Grain } from "./grain";
import { COVERED_CMD, COVERED_EVENT, SCENE_FPS, SceneRun } from "./scene";

/** Frames a second while the circle is on its way after him, or fading. */
const MOVING_FPS = 30;
/** Seconds of scene time for one turn of the ring, and for a spark to run
 * its thread. */
const SPIN_S = 900;
const SPARK_S = 7;
/** The least time between two frames drawn out of turn in a still scene
 * (his place, as he is dragged, comes as often as the mouse moves). */
const SOON_MS = 33;

const canvas = document.getElementById("backdrop") as HTMLCanvasElement;
const ctx = canvas.getContext("2d", { alpha: false });

let ramp: Ramp | null = null;
let ink: Ink | null = null;
let grain: Grain | null = null;
let grainOn = false;
let pattern: CanvasPattern | null = null;
let size = { width: 0, height: 0 };
let dpr = 1;
let frames = 0;
let shown = false;
/** A GPU reset has the canvas's context: nothing drawn shows until it is
 * restored. */
let lost = false;
let covered = false;
let displayOn = true;
let online = false;
/** The scene was covered, uncovered, or out of sight (the display off):
 * its next frame puts the circle where it now is, rather than fading from
 * what it was, so a fullscreen app's coming or going never blinks it. */
let snapNext = false;
const glyphs = makeGlyphs();
const follow = new Follow();
const fresh = new Freshness();

const loop = new PacedLoop((dt) => run.tick(dt), SCENE_FPS);
const run = new SceneRun((t, dt) => {
  if (!ctx || !ramp || size.width <= 0 || lost) return;
  paint(ctx, size.width, size.height, ramp, glowAt(t));
  const looping = run.now === "loop";
  const moving = follow.step(dt, !looping || snapNext);
  snapNext = false;
  if (ink && follow.presence > 0) drawStudyCircle(ctx, ink, t, looping);
  if (grainOn && grain && pattern) {
    ctx.save();
    ctx.globalCompositeOperation = grain.blend;
    ctx.globalAlpha = grain.opacity;
    ctx.fillStyle = pattern;
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.restore();
  }
  loop.setFps(moving ? MOVING_FPS : SCENE_FPS);
  // Hidden until it has a frame in the owner's mood (wallpaper.html), and
  // again while a GPU reset has left it blank.
  if (!shown) {
    canvas.style.visibility = "visible";
    shown = true;
  }
  frames += 1;
  if (!looping) watchStale();
}, loop);

/** The circle around Fritz now, at scene time `t`. */
function drawStudyCircle(c: CanvasRenderingContext2D, k: Ink, t: number, looping: boolean): void {
  const r = radiusFor(follow.size);
  const arc = openArc(follow.x, follow.y, r, size.width, size.height) ?? [-Math.PI, Math.PI];
  const angles = nodeAngles(NODES.length, arc);
  const now = performance.now();
  drawCircle(c, {
    x: follow.x,
    y: follow.y,
    r,
    presence: follow.presence,
    spin: (t / SPIN_S) * Math.PI * 2,
    spark: looping ? t / SPARK_S : null,
    nodes: NODES.map((name, i) => ({ name, angle: angles[i]!, lit: fresh.lit(name, now, online) })),
  }, k, glyphs);
}

/** Draw a still scene again soon: at most once per SOON_MS however often
 * it is asked. A running loop draws it at its next frame anyway. */
let soon: ReturnType<typeof setTimeout> | null = null;
function redrawSoon(): void {
  if (run.now === "loop" || soon !== null) return;
  soon = setTimeout(() => {
    soon = null;
    run.redraw();
  }, SOON_MS);
}

/** A still scene draws again when a lit thread's widget goes stale, which
 * no event says. */
let staleTimer: ReturnType<typeof setTimeout> | null = null;
function watchStale(): void {
  if (staleTimer !== null) clearTimeout(staleTimer);
  staleTimer = null;
  const left = fresh.untilStale(performance.now());
  if (left !== null) staleTimer = setTimeout(() => run.redraw(), left + 50);
}

// For the live checks, as the avatar's page has its own: what the scene is
// doing. Read-only.
(window as unknown as Record<string, unknown>).__fritzScene = () => ({
  drawing: run.now,
  running: loop.running,
  rate: loop.rate,
  time: run.time,
  frames,
  covered,
  displayOn,
  online,
  width: canvas.width,
  height: canvas.height,
  circle: { x: follow.x, y: follow.y, size: follow.size, presence: follow.presence },
  lit: NODES.filter((n) => fresh.lit(n, performance.now(), online)),
  grain: grainOn && grain !== null,
});

/** The grain's noise as a pattern, one noise pixel to a device pixel. */
function makePattern(c: CanvasRenderingContext2D, scale: number): CanvasPattern | null {
  const tile = document.createElement("canvas");
  tile.width = TILE;
  tile.height = TILE;
  const t = tile.getContext("2d");
  if (!t) return null;
  t.putImageData(new ImageData(noiseTile(), TILE, TILE), 0, 0);
  const p = c.createPattern(tile, "repeat");
  p?.setTransform(new DOMMatrix().scale(1 / scale));
  return p;
}

/** Size the canvas to the page, in device pixels (capped at 2 per CSS
 * pixel). The window is built at its monitor's size and given its real
 * one when it is moved behind the icons; a size under a pixel (not there
 * yet) keeps the last. */
function fit(): void {
  const width = window.innerWidth;
  const height = window.innerHeight;
  if (width < 1 || height < 1 || !ctx) return;
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  size = { width, height };
  // The shell's pixels are the monitor's own: devicePixelRatio, uncapped.
  follow.setScale(window.devicePixelRatio || 1);
  pattern = makePattern(ctx, dpr);
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

/** Ask the shell for something once listening, keeping a newer event's
 * word over the answer. */
async function asked<T>(cmd: string, heard: () => number, parse: (v: unknown) => T | undefined): Promise<T | undefined> {
  const before = heard();
  try {
    const v = parse(await invoke<unknown>(cmd));
    return heard() === before ? v : undefined;
  } catch {
    // Not asked as a study scene (a plain browser).
    return undefined;
  }
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
    const style = getComputedStyle(document.documentElement);
    ramp = rampOf(style, t.theme === "light");
    ink = inkOf(style);
    grain = grainOf(style);
    grainOn = t.grain;
    run.setTheme(motion.factor);
  });

  let displayEvents = 0;
  await listen<unknown>(POWER_EVENT, (e) => {
    const on = parseDisplay(e.payload);
    if (on === null) return;
    displayEvents += 1;
    if (on && !displayOn) snapNext = true;
    displayOn = on;
    run.setDisplay(on);
  });
  const heardDisplay = displayEvents;
  displayOn = afterAsking(await fetchDisplayOn(), heardDisplay, displayEvents, displayOn);
  run.setDisplay(displayOn);

  // This window's own word: a plain listen would hear every study scene's.
  const me = getCurrentWebviewWindow();
  let coveredEvents = 0;
  await me.listen<unknown>(COVERED_EVENT, (e) => {
    if (typeof e.payload !== "boolean") return;
    coveredEvents += 1;
    // Covered or uncovered, the next frame puts the circle where it now
    // is: the frame kept under a cover, and the first after it.
    if (covered !== e.payload) snapNext = true;
    covered = e.payload;
    run.setCovered(covered);
  });
  const wasCovered = await asked(COVERED_CMD, () => coveredEvents, (v) => v === true);
  if (wasCovered !== undefined) {
    covered = wasCovered;
    run.setCovered(covered);
  }

  // Where Fritz is on this monitor.
  let placeEvents = 0;
  await me.listen<unknown>(FRITZ_EVENT, (e) => {
    const place = parsePlace(e.payload);
    if (place === undefined) return;
    placeEvents += 1;
    follow.set(place);
    // Under a fullscreen app (or the lock screen) he is hidden: the frame
    // this page keeps, and shows the moment the cover goes, is put where
    // things are now (SceneRun.placed), at once, so no circle is left there
    // around an empty spot.
    if (covered) snapNext = true;
    const now = run.placed();
    // Else the loop takes it from its next frame, faster while it catches
    // up, or a still scene is drawn again soon.
    if (now === "loop") loop.setFps(MOVING_FPS);
    else if (now === "still") redrawSoon();
  });
  const place = await asked(FRITZ_CMD, () => placeEvents, parsePlace);
  if (place !== undefined) {
    follow.set(place);
    run.redraw();
  }

  // The brain's link, and its widgets as they come.
  let linkEvents = 0;
  await listen<unknown>(COMPANION_STATUS_EVENT, (e) => {
    const s = parseCompanionStatus(e.payload);
    if (!s) return;
    linkEvents += 1;
    online = s.link === "online";
    redrawSoon();
  });
  const link = await asked(STATUS_CMD, () => linkEvents, (v) => parseCompanionStatus(v) ?? undefined);
  if (link) online = link.link === "online";
  await me.listen<unknown>(SEEN_EVENT, (e) => {
    const s = parseSeen(e.payload);
    if (!s) return;
    fresh.hear(s, performance.now());
    redrawSoon();
  });
  try {
    const all = await invoke<unknown>(NOW_CMD);
    const at = performance.now();
    if (Array.isArray(all)) for (const item of all) {
      const s = parseSeen(item);
      // Each kept unless an event since was newer (Freshness.hear).
      if (s) fresh.hear(s, at);
    }
  } catch {
    // A plain browser: none.
  }
  run.redraw();
}

void boot();
