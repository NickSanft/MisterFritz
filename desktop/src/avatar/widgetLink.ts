/**
 * The brain's widgets, for the study scene behind the icons (plan 14 build
 * § 4.7): this page holds the device's one session with the brain, so it
 * asks for them, on every welcome whose brain offers `widgets`, and hands
 * each `widget` frame to the shell (`companion_widget`), which passes it on
 * to the scene's windows and keeps the newest of each. One connection per
 * device, as the plan chose, not one per window.
 *
 * Clock, weather and desk (the owner's call): asking for weather has the
 * brain fetch it from Open-Meteo about every quarter hour while this desk
 * is connected.
 */
import { invoke } from "@tauri-apps/api/core";
import type { CompanionClient } from "../shared/companion";
import { WIDGET_NAMES } from "../shared/protocol";

export const WIDGET_CMD = "companion_widget";
/** What this desk asks for: every widget the brain feeds. */
export const WANTED: readonly string[] = WIDGET_NAMES;

/** A widget as the shell takes it. */
export interface Handed {
  name: string;
  ttlS: number;
  payload: Record<string, unknown>;
}

export interface WidgetDeps {
  hand(w: Handed): Promise<void>;
}

const tauriDeps = (): WidgetDeps => ({
  hand: async (w) => {
    try {
      await invoke(WIDGET_CMD, { name: w.name, ttlS: w.ttlS, payload: w.payload });
    } catch (e) {
      console.warn(`avatar: ${WIDGET_CMD} failed`, e);
    }
  },
});

/** Ask `client`'s brain for the widgets on each welcome, and hand on each
 * that comes. Returns the unsubscribe. */
export function linkWidgets(client: CompanionClient, deps: WidgetDeps = tauriDeps()): () => void {
  const offWelcome = client.on("welcome", (w) => {
    if (w.server_caps.includes("widgets")) client.subscribe(WANTED);
  });
  const offWidget = client.on("widget", (f) => {
    void deps.hand({ name: f.name, ttlS: f.ttl_s, payload: f.payload });
  });
  return () => {
    offWelcome();
    offWidget();
  };
}
