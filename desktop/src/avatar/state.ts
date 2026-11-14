/**
 * The turn the speech bubble shows, as a pure reducer: the page's one view
 * of a conversation with the brain, moved on by what the owner asks and
 * what the brain sends (companion_api.py's rules, plan 14 build § 2.2).
 *
 * - Asking switches the view to the new turn at once. Anything later for
 *   another turn is ignored: after a barge-in, the cancelled turn's last
 *   tokens and its done still arrive, and must not show.
 * - A token adds to the reply, or replaces it when it restarts (the model
 *   began its answer again after a tool).
 * - Progress (a tool's notice) shows the latest only, until the turn ends.
 * - Done ends the turn with the whole reply, which supersedes the tokens.
 *   An interrupted reply carries the brain's " [interrupted]" mark: it is
 *   taken off and shown as a note instead. Pictures are counted, never kept
 *   (each can be megabytes of data: URL).
 * - An error for the shown turn ends it with Fritz's apology, the reply so
 *   far kept. An error for no turn is not shown.
 * - The link lost mid-turn ends it with an apology of the page's own: the
 *   brain sends nothing more for a turn once its socket is gone.
 *
 * `revision` changes whenever what the bubble shows changes, and only then,
 * so the page writes and measures text only when it must.
 */
import type { ServerFrame, TurnState } from "../shared/protocol";

export type Phase = TurnState;

export interface TurnView {
  /** The turn shown; null before the first. */
  turnId: string | null;
  phase: Phase;
  reply: string;
  progress: string | null;
  apology: string | null;
  pictures: number;
  interrupted: boolean;
  /** Over: done, failed or lost. */
  ended: boolean;
  revision: number;
}

export type TurnEvent =
  | { kind: "asked"; turnId: string }
  | { kind: "frame"; frame: ServerFrame }
  | { kind: "lost" };

export const INITIAL: TurnView = Object.freeze({
  turnId: null,
  phase: "idle",
  reply: "",
  progress: null,
  apology: null,
  pictures: 0,
  interrupted: false,
  ended: true,
  revision: 0,
});

/** What he says when the link drops mid-reply (the brain's own register). */
export const LOST_APOLOGY =
  "Forgive me, sir: I lost my connection to the house mid-reply. I'll be with you again directly.";

/** The brain's mark on a reply cut short (mister_fritz.INTERRUPTED_MARK). */
export const INTERRUPTED_MARK = " [interrupted]";

/** The fields the bubble shows: a change to any of them is a new revision. */
const SHOWN = ["turnId", "reply", "progress", "apology", "pictures", "interrupted", "ended"] as const;

function next(v: TurnView, change: Partial<TurnView>): TurnView {
  const out = { ...v, ...change };
  const shown = SHOWN.some((k) => out[k] !== v[k]);
  if (!shown && out.phase === v.phase) return v;
  return shown ? { ...out, revision: v.revision + 1 } : out;
}

/** `text` without the interrupted mark (a reply cut before a word is the
 * mark alone, trimmed). */
function unmarked(text: string): string {
  if (text.endsWith(INTERRUPTED_MARK)) return text.slice(0, -INTERRUPTED_MARK.length);
  return text === INTERRUPTED_MARK.trim() ? "" : text;
}

/** An apology fit for the bubble: fritz_error writes Markdown's backticks
 * round its ref ("(ref `abcd1234`)"), which the bubble doesn't render. */
export function apologyText(s: string): string {
  return s.replace(/`/g, "");
}

/** The turn to cancel before asking anew (a barge-in): the one shown, if it
 * hasn't ended. */
export function cancelFirst(v: TurnView): string | null {
  return v.turnId !== null && !v.ended ? v.turnId : null;
}

export function reduce(v: TurnView, e: TurnEvent): TurnView {
  switch (e.kind) {
    case "asked":
      return {
        turnId: e.turnId,
        phase: "thinking",
        reply: "",
        progress: null,
        apology: null,
        pictures: 0,
        interrupted: false,
        ended: false,
        revision: v.revision + 1,
      };
    case "lost":
      if (v.turnId === null || v.ended) return v;
      return next(v, { apology: LOST_APOLOGY, progress: null, phase: "idle", ended: true });
    case "frame":
      return frame(v, e.frame);
  }
}

function frame(v: TurnView, f: ServerFrame): TurnView {
  if (!("turn_id" in f) || f.turn_id === null || f.turn_id !== v.turnId) return v;
  if (f.type === "state") return next(v, { phase: f.state });
  if (v.ended) return v;
  switch (f.type) {
    case "token":
      return next(v, { reply: f.restart ? f.delta : v.reply + f.delta });
    case "progress":
      return next(v, { progress: f.text });
    case "done":
      return next(v, {
        reply: f.interrupted ? unmarked(f.text) : f.text,
        interrupted: f.interrupted,
        pictures: f.images.length,
        progress: null,
        phase: "idle",
        ended: true,
      });
    case "error":
      return next(v, { apology: apologyText(f.text), progress: null, phase: "idle", ended: true });
    default:
      return v;
  }
}
