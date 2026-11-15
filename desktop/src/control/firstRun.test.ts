import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import type { LinkStatus } from "../shared/companion";
import type { ModelInfo } from "./api";
import { readyLines, STEP_TITLES, stepAfter, stepBefore, stepLine, WIZARD_STEPS } from "./firstRun";

const link = (name: keyof typeof fixture.companionStatusSamples): LinkStatus =>
  fixture.companionStatusSamples[name] as LinkStatus;
const WEARING: ModelInfo = {
  from: "folder", path: "C:\\data\\avatar\\fritz.vrm", found: true, bytes: 1, tooBig: false,
  wearing: "export", loadError: null, changed: false,
};
const HELD = { prompt: "Ctrl+Alt+F", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" };

describe("the first run's steps", () => {
  it("are his brain, his look, his keys, then summoning him", () => {
    expect([...WIZARD_STEPS]).toEqual(["brain", "look", "keys", "ready"]);
    expect(WIZARD_STEPS.map((s) => STEP_TITLES[s])).toEqual(
      ["Connect his brain", "Choose his look", "Learn his keys", "Summon him"],
    );
  });

  it("go forward and back, and stop at either end", () => {
    expect(stepAfter("brain")).toBe("look");
    expect(stepAfter("look")).toBe("keys");
    expect(stepAfter("keys")).toBe("ready");
    expect(stepAfter("ready")).toBeNull();
    expect(stepBefore("ready")).toBe("keys");
    expect(stepBefore("look")).toBe("brain");
    expect(stepBefore("brain")).toBeNull();
  });

  it("say where the run stands", () => {
    expect(stepLine("brain")).toBe("Step 1 of 4: Connect his brain");
    expect(stepLine("ready")).toBe("Step 4 of 4: Summon him");
  });
});

describe("what the first run's last step says", () => {
  it("all set: connected, his model, his prompt key", () => {
    expect(readyLines(link("online"), WEARING, HELD, true)).toEqual([
      { text: "He's connected to his brain.", tone: "ok" },
      { text: "He wears your model.", tone: "ok" },
      { text: "Press Ctrl+Alt+F in any app to ask him something.", tone: "ok" },
    ]);
  });

  it("what still needs seeing to, and where", () => {
    const [noToken, placeholder, noKey] = readyLines(link("unconfigured"), { ...WEARING, found: false, wearing: "placeholder" },
      { ...HELD, prompt: null }, false);
    expect(noToken).toEqual({ text: "There's no token yet, so he can't reach his brain: paste it in Settings › Brain.", tone: "warn" });
    expect(placeholder.tone).toBe("quiet");
    expect(placeholder.text).toMatch(/placeholder/);
    expect(noKey).toEqual({ text: "His prompt key isn't working: pick another in Settings › Hotkeys.", tone: "warn" });
    // A token, but not connected: Brain says why.
    expect(readyLines(link("refused"), WEARING, HELD, true)[0]).toEqual(
      { text: "He isn't connected to his brain yet: Settings › Brain says why.", tone: "warn" },
    );
    // Before the shell says whether there is a token, no token is claimed.
    expect(readyLines(null, WEARING, HELD, null)[0]).toEqual(
      { text: "He isn't connected to his brain yet: Settings › Brain says why.", tone: "warn" },
    );
  });

  it("a model not yet loaded, or the shell yet to say, isn't claimed", () => {
    // A new export there, or the old one gone: he still wears the last one.
    for (const m of [{ ...WEARING, changed: true }, { ...WEARING, found: false, changed: true }]) {
      expect(readyLines(link("online"), m, HELD, true)[1]).toEqual(
        { text: "He wears the model he loaded last, until it's loaded again in Settings › Model.", tone: "quiet" },
      );
    }
    expect(readyLines(link("online"), null, HELD, true)[1].tone).toBe("quiet");
    expect(readyLines(link("online"), WEARING, null, true)[2]).toEqual({ text: "His keys are in Settings › Hotkeys.", tone: "quiet" });
  });

  it("calls the Windows key Win", () => {
    expect(readyLines(link("online"), WEARING, { ...HELD, prompt: "Ctrl+Super+F" }, true)[2].text)
      .toBe("Press Ctrl+Win+F in any app to ask him something.");
  });
});
