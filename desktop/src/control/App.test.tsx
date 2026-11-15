/**
 * The whole control window over a shell mocked at Tauri's IPC: what it asks
 * for, and where the answers go. The sections' own tests cover each
 * section; this covers the window's wiring of them.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, waitFor } from "@solidjs/testing-library";
import fixture from "../avatar/wire.fixture.json";
import { App } from "./App";
import { HOTKEY_DOES } from "./view";

type Args = Record<string, unknown>;

const SETTINGS = {
  version: 1,
  brain: { url: "ws://127.0.0.1:8001/companion", clientId: "nick-pc", clientName: "NICK-PC" },
  avatar: { rect: null, scale: 1, parked: false, vrmPath: null, lookatCursor: true },
  hotkeys: { prompt: "Ctrl+Alt+F", ptt: "Ctrl+Alt+Space", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" },
  diagnostics: { foregroundLog: false },
  theme: fixture.themeDefaults,
};
const MODEL = {
  from: "folder", path: "C:\\data\\avatar\\fritz.vrm", found: false, bytes: null, tooBig: false,
  wearing: "placeholder", loadError: null, changed: false,
};

let calls: { cmd: string; args: Args }[];
let answers: Record<string, (args: Args) => unknown>;

beforeEach(() => {
  calls = [];
  answers = {
    get_settings: () => SETTINGS,
    avatar_status: () => fixture.statusSamples.arranging,
    avatar_model: () => MODEL,
    get_token_present: () => true,
    companion_status: () => fixture.companionStatusSamples.online,
    hotkeys_bound: () => ({ prompt: "Ctrl+Alt+F", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" }),
    control_opening: () => null,
    "plugin:event|listen": (a) => a.handler,
    "plugin:event|unlisten": () => null,
    "plugin:app|version": () => "0.1.0",
  };
  let next = 1;
  Object.assign(window, {
    matchMedia: (media: string) => ({ matches: false, media, addEventListener() {}, removeEventListener() {} }),
    __TAURI_INTERNALS__: {
      metadata: { currentWindow: { label: "control" }, currentWebview: { windowLabel: "control", label: "control" } },
      async invoke(cmd: string, args: Args = {}) {
        calls.push({ cmd, args });
        const answer = answers[cmd];
        if (!answer) throw new Error(`the mock has no answer for ${cmd}`);
        return answer(args);
      },
      transformCallback: () => next++,
      unregisterCallback() {},
    },
    __TAURI_EVENT_PLUGIN_INTERNALS__: { unregisterListener() {} },
  });
});

afterEach(() => {
  document.documentElement.removeAttribute("data-theme");
});

const asked = (cmd: string) => calls.filter((c) => c.cmd === cmd);

/** A key pressed anywhere in the window, as the capture listens for it. */
const press = (key: string, more: KeyboardEventInit = {}) =>
  window.dispatchEvent(new KeyboardEvent("keydown", { key, cancelable: true, bubbles: true, ...more }));

describe("the control window", () => {
  it("names the key Windows holds, not the one set, and none when it holds none", async () => {
    answers.get_settings = () => ({ ...SETTINGS, hotkeys: { ...SETTINGS.hotkeys, arrange: "Ctrl+Alt+Nope" } });
    answers.hotkeys_bound = () => ({ prompt: "Ctrl+Alt+F", arrange: "Ctrl+Alt+A", park: null });
    const view = render(() => <App />);
    await waitFor(() => expect(view.getByText(/Esc locks him there\. Or press Ctrl\+Alt\+A\.$/)).toBeInTheDocument());
    expect(view.queryByText(/Or press Ctrl\+Alt\+Nope/)).toBeNull();
    expect(view.queryByText(/Or press Ctrl\+Alt\+H/)).toBeNull();
    expect(view.getByText(/^Not working: he couldn't have Ctrl\+Alt\+H/)).toBeInTheDocument();
    expect(view.getByText(/^Working on Ctrl\+Alt\+A: Ctrl\+Alt\+Nope couldn't be used/)).toBeInTheDocument();
  });

  it("binds a key through set_hotkey, shows what comes back, and asks again what is held", async () => {
    const after = { ...SETTINGS, hotkeys: { ...SETTINGS.hotkeys, prompt: "Ctrl+Alt+K" } };
    answers.set_hotkey = () => after;
    const view = render(() => <App />);
    const capture = await waitFor(() => view.getByRole("button", { name: "Ask Fritz hotkey: Ctrl Alt F" }));
    const boundBefore = asked("hotkeys_bound").length;
    answers.hotkeys_bound = () => ({ prompt: "Ctrl+Alt+K", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" });
    fireEvent.click(capture, { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    await waitFor(() => expect(view.getByRole("button", { name: "Ask Fritz hotkey: Ctrl Alt K" })).toBeInTheDocument());
    expect(asked("set_hotkey").map((c) => c.args)).toEqual([{ action: "prompt", accelerator: "Ctrl+Alt+K" }]);
    expect(asked("set_setting")).toEqual([]);
    await waitFor(() => expect(asked("hotkeys_bound").length).toBe(boundBefore + 1));
    // What is held, asked again, is what the row goes by: the new key works.
    await waitFor(() => expect(view.getByText(HOTKEY_DOES.prompt)).toBeInTheDocument());
    expect(view.queryByText(/^Working on /)).toBeNull();
  });

  it("says a refused key on its row, never in the window's banner", async () => {
    const why = "Another app already uses Ctrl+Alt+K: pick another.";
    answers.set_hotkey = () => {
      throw why;
    };
    const view = render(() => <App />);
    const capture = await waitFor(() => view.getByRole("button", { name: "Hide or show him hotkey: Ctrl Alt H" }));
    const boundBefore = asked("hotkeys_bound").length;
    fireEvent.click(capture, { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    const alert = await waitFor(() => view.getByRole("alert"));
    expect(alert).toHaveTextContent(why);
    expect(alert.closest(".row")).toHaveTextContent("Hide or show him");
    expect(view.container.querySelector(".problem")).toBeNull();
    expect(view.getByRole("button", { name: "Hide or show him hotkey: Ctrl Alt H" })).toBeInTheDocument();
    await waitFor(() => expect(asked("hotkeys_bound").length).toBe(boundBefore + 1));
  });
});
