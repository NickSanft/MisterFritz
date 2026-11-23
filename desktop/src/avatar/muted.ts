/**
 * Whether push-to-talk is muted (`settings.audio.muted`, which only the
 * tray's Mute push-to-talk sets), for the sign over his head: followed from
 * every settings event, and from the settings asked for once listening,
 * unless an event came first (as the theme is, divora/theme.ts).
 */
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { afterAsking } from "../shared/asking";
import { GET_SETTINGS_CMD, SETTINGS_EVENT } from "../shared/settings";

/** `settings.audio.muted` from the wire, or null when it isn't one. */
export function mutedOf(settings: unknown): boolean | null {
  const s = settings as { audio?: { muted?: unknown } } | null;
  const muted = typeof s === "object" && s !== null && typeof s.audio === "object" && s.audio !== null ? s.audio.muted : undefined;
  return typeof muted === "boolean" ? muted : null;
}

export interface MutedWatchDeps {
  listen(cb: (settings: unknown) => void): Promise<() => void>;
  ask(): Promise<unknown>;
}

const tauriDeps = (): MutedWatchDeps => ({
  listen: (cb) => listen<unknown>(SETTINGS_EVENT, (e) => cb(e.payload)),
  ask: () => invoke<unknown>(GET_SETTINGS_CMD),
});

/** Tell `onMuted` whether push-to-talk is muted, at every change. Resolves
 * once listening, without waiting for the answer to the question. */
export async function watchMuted(
  onMuted: (muted: boolean) => void,
  deps: MutedWatchDeps = tauriDeps(),
): Promise<() => void> {
  let current: boolean | null = null;
  let events = 0;
  const unlisten = await deps.listen((settings) => {
    const m = mutedOf(settings);
    if (m === null) return;
    events += 1;
    current = m;
    onMuted(m);
  });
  const before = events;
  void deps
    .ask()
    .then((settings) => {
      const m = mutedOf(settings);
      if (m === null) return;
      const next = afterAsking<boolean | null>(m, before, events, current);
      if (next !== null && next !== current) {
        current = next;
        onMuted(next);
      }
    })
    .catch(() => {
      // No answer: the next settings event says.
    });
  return unlisten;
}
