import { describe, expect, it } from "vitest";
import { oneAtATime } from "./oneAtATime";

const flush = async () => {
  for (let i = 0; i < 5; i++) await Promise.resolve();
};

describe("one send at a time", () => {
  it("sends each value only once the one before it was taken, and only the newest", async () => {
    const sent: number[] = [];
    const settle: (() => void)[] = [];
    const push = oneAtATime<number>((v) => {
      sent.push(v);
      return new Promise<void>((r) => settle.push(r));
    });
    push(1);
    push(2);
    push(3);
    await flush();
    expect(sent).toEqual([1]);
    settle[0]();
    await flush();
    expect(sent).toEqual([1, 3]); // 2 was overtaken before it left
    settle[1]();
    await flush();
    push(4);
    await flush();
    expect(sent).toEqual([1, 3, 4]);
  });

  it("never has two on their way", async () => {
    let flying = 0;
    let most = 0;
    const settle: (() => void)[] = [];
    const push = oneAtATime<string>(() => {
      flying += 1;
      most = Math.max(most, flying);
      return new Promise<void>((r) => settle.push(() => {
        flying -= 1;
        r();
      }));
    });
    for (const v of ["a", "b", "c"]) push(v);
    await flush();
    while (settle.length) {
      settle.shift()!();
      await flush();
    }
    expect(most).toBe(1);
  });

  it("sends the next after one that failed", async () => {
    const sent: number[] = [];
    const push = oneAtATime<number>(async (v) => {
      sent.push(v);
      if (v === 1) throw new Error("refused");
    });
    push(1);
    await flush();
    push(2);
    await flush();
    expect(sent).toEqual([1, 2]);
  });

  it("sends a value that is nothing, too", async () => {
    const sent: unknown[] = [];
    const push = oneAtATime<number | null>(async (v) => {
      sent.push(v);
    });
    push(0);
    await flush();
    push(null);
    await flush();
    expect(sent).toEqual([0, null]);
  });
});
