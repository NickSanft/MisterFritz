/**
 * Asking the shell for a state at start-up, when events about the same state
 * may arrive while the question is out: shared by every page.
 */

/** A state once the start-up question is answered (the display, the cursor,
 * the theme): the answer, unless an event arrived while it was being asked.
 * A reply isn't ordered against events, and any event is newer than the
 * question. */
export function afterAsking<T>(answer: T, eventsBefore: number, eventsNow: number, current: T): T {
  return eventsNow === eventsBefore ? answer : current;
}
