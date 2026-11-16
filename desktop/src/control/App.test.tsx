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
  avatar: { rect: null, scale: 1, parked: false, vrmPath: null, lookatCursor: true, clicks: true },
  hotkeys: { prompt: "Ctrl+Alt+F", ptt: "Ctrl+Alt+Space", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" },
  diagnostics: { foregroundLog: false },
  theme: fixture.themeDefaults,
};
/** The mouse hook out while he is hidden, and in. */
const HOOK_OUT = {
  on: true, installed: false, why: fixture.hookOff.hidden, events: 0, presses: 0, onShape: 0, covered: 0, taken: 0,
  gestures: 0, slowestMicros: 0, putBack: 0, panicked: false,
};
const HOOK_IN = { ...HOOK_OUT, installed: true, why: null, events: 812, presses: 9, slowestMicros: 41 };
const MODEL = {
  from: "folder", path: "C:\\data\\avatar\\fritz.vrm", found: false, bytes: null, tooBig: false,
  wearing: "placeholder", loadError: null, changed: false,
};

let calls: { cmd: string; args: Args }[];
let answers: Record<string, (args: Args) => unknown>;
/** The page's listeners, by event, as Tauri's event plugin keeps them. */
let callbacks: Map<number, (data: unknown) => void>;
let listeners: Map<string, number[]>;

/** The shell sends an event, as Tauri delivers one. */
const emit = (event: string, payload: unknown) => {
  for (const id of listeners.get(event) ?? []) callbacks.get(id)?.({ event, id, payload });
};

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
    mouse_hook_status: () => HOOK_OUT,
    "plugin:event|listen": (a) => {
      const event = a.event as string;
      listeners.set(event, [...(listeners.get(event) ?? []), a.handler as number]);
      return a.handler;
    },
    "plugin:event|unlisten": () => null,
    "plugin:app|version": () => "0.1.0",
  };
  let next = 1;
  callbacks = new Map();
  listeners = new Map();
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
      transformCallback: (cb: (data: unknown) => void) => {
        const id = next++;
        callbacks.set(id, cb);
        return id;
      },
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
  it("turns clicks on him off and on, and never his eyes", async () => {
    answers.set_setting = (a) => ({ ...SETTINGS, avatar: { ...SETTINGS.avatar, clicks: a.value } });
    const view = render(() => <App />);
    const toggle = await waitFor(() => view.getByRole("switch", { name: "Clicks on him" }));
    expect(toggle).toHaveAttribute("aria-checked", "true");
    fireEvent.click(toggle);
    await waitFor(() => expect(view.getByRole("switch", { name: "Clicks on him" })).toHaveAttribute("aria-checked", "false"));
    fireEvent.click(view.getByRole("switch", { name: "Clicks on him" }));
    await waitFor(() => expect(asked("set_setting")).toHaveLength(2));
    expect(asked("set_setting").map((c) => c.args)).toEqual([
      { path: "avatar.clicks", value: false },
      { path: "avatar.clicks", value: true },
    ]);
    expect(view.getByRole("switch", { name: "His eyes follow the cursor" })).toHaveAttribute("aria-checked", "true");
  });

  it("says where the mouse hook stands, as the shell says it, and asks again after the switch and on focus", async () => {
    answers.set_setting = (a) => ({ ...SETTINGS, avatar: { ...SETTINGS.avatar, clicks: a.value } });
    const view = render(() => <App />);
    await waitFor(() => expect(view.getByText("Out: he's hidden.")).toBeInTheDocument());
    // The shell's word that it went in.
    emit("hook:status", HOOK_IN);
    await waitFor(() => expect(view.getByText("In while he's on screen; the slowest answer took 0.04 ms.")).toBeInTheDocument());
    // Something that isn't a status is no word at all.
    emit("hook:status", { installed: "yes" });
    await new Promise((r) => setTimeout(r, 0));
    expect(view.getByText("In while he's on screen; the slowest answer took 0.04 ms.")).toBeInTheDocument();
    // Asked again after the switch.
    const before = asked("mouse_hook_status").length;
    answers.mouse_hook_status = () => ({ ...HOOK_OUT, on: false, why: fixture.hookOff.switch });
    fireEvent.click(view.getByRole("switch", { name: "Clicks on him" }));
    await waitFor(() => expect(view.getByText("Out: clicks on him are off in Settings.")).toBeInTheDocument());
    expect(asked("mouse_hook_status").length).toBe(before + 1);
    // And when the window comes forward.
    answers.mouse_hook_status = () => HOOK_IN;
    window.dispatchEvent(new Event("focus"));
    await waitFor(() => expect(view.getByText("In while he's on screen; the slowest answer took 0.04 ms.")).toBeInTheDocument());
    expect(asked("mouse_hook_status").length).toBe(before + 2);
  });

  it("an answer read before the shell's word can't undo it", async () => {
    let answer: (v: unknown) => void = () => {};
    answers.mouse_hook_status = () => new Promise((r) => (answer = r));
    const view = render(() => <App />);
    await waitFor(() => expect(asked("mouse_hook_status")).toHaveLength(1));
    emit("hook:status", HOOK_IN);
    await waitFor(() => expect(view.getByText("In while he's on screen; the slowest answer took 0.04 ms.")).toBeInTheDocument());
    answer(HOOK_OUT);
    await new Promise((r) => setTimeout(r, 0));
    expect(view.queryByText("Out: he's hidden.")).toBeNull();
  });

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

  it("a hotkey's refusal outlives a change to another setting", async () => {
    const why = "Another app already uses Ctrl+Alt+K: pick another.";
    answers.set_hotkey = () => {
      throw why;
    };
    const view = render(() => <App />);
    const capture = await waitFor(() => view.getByRole("button", { name: "Ask Fritz hotkey: Ctrl Alt F" }));
    fireEvent.click(capture, { detail: 1 });
    press("k", { ctrlKey: true, altKey: true });
    await waitFor(() => expect(view.getByRole("alert")).toHaveTextContent(why));
    emit("settings:changed", { ...SETTINGS, theme: { ...SETTINGS.theme, mood: "midnight" } });
    await new Promise((r) => setTimeout(r, 0));
    expect(document.documentElement.dataset.mood).toBe("midnight");
    expect(view.getByRole("alert")).toHaveTextContent(why);
  });

  it("names the prompt's key in Brain's Ask line, as Windows holds it", async () => {
    answers.hotkeys_bound = () => ({ prompt: "Ctrl+Super+F", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" });
    const view = render(() => <App />);
    await waitFor(() => expect(view.getByText(/Anywhere else, press Ctrl\+Win\+F\.$/)).toBeInTheDocument());
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

describe("the first run in the control window", () => {
  const wizard = (view: ReturnType<typeof render>) => view.container.querySelector("[data-view=wizard]");
  const sections = (view: ReturnType<typeof render>) => view.container.querySelector("main.sections");
  const heading = (view: ReturnType<typeof render>) => view.container.querySelector("h2.wizard-title")?.textContent;

  it("opens in place of the sections when the shell says so, and not otherwise", async () => {
    answers.control_opening = () => "wizard";
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    expect(sections(view)).toBeNull();
    expect(view.container.querySelectorAll("section#brain")).toHaveLength(1);
    view.unmount();
    answers.control_opening = () => null;
    const plain = render(() => <App />);
    await waitFor(() => expect(asked("control_opening").length).toBe(2));
    await waitFor(() => expect(sections(plain)).not.toBeNull());
    // Let the opening's answer land before saying it changed nothing.
    await new Promise((r) => setTimeout(r, 0));
    expect(wizard(plain)).toBeNull();
  });

  it("shows neither view until the shell has said where to open", async () => {
    let answer: (at: string | null) => void = () => {};
    answers.control_opening = () => new Promise((r) => { answer = r; });
    const view = render(() => <App />);
    await waitFor(() => expect(asked("get_settings")).toHaveLength(1));
    await new Promise((r) => setTimeout(r, 0));
    expect(sections(view)).toBeNull();
    expect(wizard(view)).toBeNull();
    answer("wizard");
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    expect(sections(view)).toBeNull();
  });

  it("Skip setup shows the sections again", async () => {
    answers.control_opening = () => "wizard";
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    fireEvent.click(view.getByRole("button", { name: "Skip setup" }));
    expect(wizard(view)).toBeNull();
    expect(sections(view)).not.toBeNull();
  });

  it("Summon brings him out, then puts the first run and the window away, in that order", async () => {
    answers.control_opening = () => "wizard";
    answers.set_parked = () => fixture.statusSamples.arranging;
    answers["plugin:window|hide"] = () => null;
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    for (let i = 0; i < 3; i += 1) fireEvent.click(view.getByRole("button", { name: "Continue" }));
    fireEvent.click(view.getByRole("button", { name: "Summon Fritz" }));
    await waitFor(() => expect(asked("plugin:window|hide")).toHaveLength(1));
    const order = calls.map((c) => c.cmd).filter((c) => c === "set_parked" || c === "plugin:window|hide");
    expect(order).toEqual(["set_parked", "plugin:window|hide"]);
    expect(asked("set_parked")[0].args).toEqual({ on: false });
    expect(asked("plugin:window|hide")[0].args).toEqual({ label: "control" });
    expect(wizard(view)).toBeNull();
  });

  it("a summon refused leaves the first run and the window as they were, saying why", async () => {
    answers.control_opening = () => "wizard";
    answers.set_parked = () => {
      throw "His window couldn't open.";
    };
    answers["plugin:window|hide"] = () => null;
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    for (let i = 0; i < 3; i += 1) fireEvent.click(view.getByRole("button", { name: "Continue" }));
    fireEvent.click(view.getByRole("button", { name: "Summon Fritz" }));
    await waitFor(() => expect(view.container.querySelector(".problem")).toHaveTextContent("Couldn't summon him: His window couldn't open."));
    expect(asked("plugin:window|hide")).toEqual([]);
    expect(wizard(view)).not.toBeNull();
  });

  it("a token pasted on its first step is saved when the owner moves on, Connect or not", async () => {
    answers.control_opening = () => "wizard";
    answers.get_token_present = () => false;
    answers.companion_status = () => fixture.companionStatusSamples.unconfigured;
    answers.set_token = () => null;
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    const token = view.getByLabelText("His token") as HTMLInputElement;
    token.focus();
    fireEvent.input(token, { target: { value: "t0k-pasted" } });
    // Pressing Continue takes the focus from the field first.
    const next = view.getByRole("button", { name: "Continue" });
    next.focus();
    fireEvent.click(next);
    await waitFor(() => expect(asked("set_token").map((c) => c.args)).toEqual([{ token: "t0k-pasted" }]));
    expect(view.container.querySelector("h2.wizard-title")?.textContent).toBe("Choose his look");
  });

  it("a token refused while the window was away is still said when it comes back", async () => {
    answers.control_opening = () => "wizard";
    answers.get_token_present = () => false;
    answers.set_token = () => {
      throw "The token has a space in it.";
    };
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    const token = view.getByLabelText("His token") as HTMLInputElement;
    token.focus();
    fireEvent.input(token, { target: { value: "abc 123" } });
    // Alt-Tab: the field loses the focus to no element of the page.
    fireEvent.blur(token, { relatedTarget: null });
    await waitFor(() => expect(view.container.querySelector(".problem")).toHaveTextContent("The token has a space in it."));
    const before = asked("get_settings").length;
    window.dispatchEvent(new Event("focus"));
    await waitFor(() => expect(asked("get_settings").length).toBe(before + 1));
    await new Promise((r) => setTimeout(r, 0));
    expect(view.container.querySelector(".problem")).toHaveTextContent("Couldn't save the token: The token has a space in it.");
  });

  it("its look step opens his folder and loads his model, as Settings' Model rows do", async () => {
    answers.control_opening = () => "wizard";
    answers.open_folder = () => null;
    answers.reload_model = () => null;
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    fireEvent.click(view.getByRole("button", { name: "Continue" }));
    fireEvent.click(view.getByRole("button", { name: "Open" }));
    fireEvent.click(view.getByRole("button", { name: "Load" }));
    await waitFor(() => expect(asked("reload_model")).toHaveLength(1));
    expect(asked("open_folder").map((c) => c.args)).toEqual([{ which: "avatar" }]);
    fireEvent.click(view.getByRole("button", { name: "Skip setup" }));
    // And the same rows in Settings › Model.
    const model = view.container.querySelector("section#model")!;
    fireEvent.click(model.querySelector("button")!);
    fireEvent.click([...model.querySelectorAll("button")].find((b) => b.textContent === "Load")!);
    await waitFor(() => expect(asked("reload_model")).toHaveLength(2));
    expect(asked("open_folder")).toHaveLength(2);
  });

  it("About's Replay opens it again, at its first step", async () => {
    answers.control_opening = () => "wizard";
    const view = render(() => <App />);
    await waitFor(() => expect(wizard(view)).not.toBeNull());
    fireEvent.click(view.getByRole("button", { name: "Continue" }));
    fireEvent.click(view.getByRole("button", { name: "Continue" }));
    expect(heading(view)).toBe("Learn his keys");
    fireEvent.click(view.getByRole("button", { name: "Skip setup" }));
    fireEvent.click(view.getByRole("button", { name: "Replay" }));
    expect(wizard(view)).not.toBeNull();
    expect(heading(view)).toBe("Connect his brain");
  });
});
