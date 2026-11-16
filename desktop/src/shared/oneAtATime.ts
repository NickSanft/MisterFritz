/**
 * Send one value at a time, and only the newest of those that came while
 * one was on its way: two invokes in flight aren't ordered (WebView2 doesn't
 * keep them so), and the shell keeps the last it hears. His link's reports
 * go through it (connecting, then online a moment later, must never end as
 * connecting), and so does his shape. Newest only, not a queue: a value
 * overtaken before it left is stale, and is never sent.
 */
export function oneAtATime<T>(send: (v: T) => Promise<unknown>): (v: T) => void {
  let next: { v: T } | null = null;
  let busy = false;
  const pump = async (): Promise<void> => {
    busy = true;
    while (next) {
      const { v } = next;
      next = null;
      try {
        await send(v);
      } catch {
        // The sender says why; the next value still goes.
      }
    }
    busy = false;
  };
  return (v) => {
    next = { v };
    if (!busy) void pump();
  };
}
