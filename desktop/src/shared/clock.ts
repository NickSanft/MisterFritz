/**
 * Time, for code that waits: the paced loop's frames and the companion
 * client's retries and heartbeat. Tests pass a fake (testClock.ts).
 */

export interface Clock {
  /** Milliseconds, monotonic: for intervals, never for the time of day. */
  now(): number;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export const browserClock: Clock = {
  now: () => performance.now(),
  setTimeout: (fn, ms) => window.setTimeout(fn, ms),
  clearTimeout: (handle) => window.clearTimeout(handle as number),
};
