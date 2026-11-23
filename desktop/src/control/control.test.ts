import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import {
  LOADED_CMD, RELOAD_EVENT, loadedArgs, parseStatus, vrmFetchError, type AvatarStatus,
} from "../avatar/status";
import {
  CMD, COMPANION_STATUS_EVENT, HOOK_KEYS, HOOK_STATUS_EVENT, HOTKEY_ACTIONS, HOTKEY_DEFAULTS, LOADED_EVENT, SECTIONS,
  SETTINGS_EVENT, WIZARD, args, parseCompanionStatus, parseHookStatus, parseHotkeysBound, parseModelInfo, parseOpening,
  parseSection, parseSettings, type HookStatus, type ModelInfo,
} from "./api";
import {
  CLICKS_DESC, HOTKEY_DOES, connectionLines, hookLine, hotkeyLine, identityLine, megabytes, modelLines, onThisPc, orPress,
  standing, winNamed,
} from "./view";
import { COMBOS, contrast, token } from "../shared/testContrast";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { LinkStatus } from "../shared/companion";

/** The mouse hook out while he is hidden, and in. */
const HOOK_OUT: HookStatus = {
  on: true, installed: false, why: fixture.hookOff.hidden, events: 0, presses: 0, onShape: 0, covered: 0, taken: 0,
  gestures: 0, slowestMicros: 0, putBack: 0, panicked: false,
};
const HOOK_IN: HookStatus = {
  ...HOOK_OUT, installed: true, why: null, events: 812, presses: 9, onShape: 2, gestures: 1, slowestMicros: 41,
};

/** A status sample from the fixture, read as the page reads one. */
function status(name: keyof typeof fixture.statusSamples): AvatarStatus {
  const s = parseStatus(fixture.statusSamples[name]);
  if (!s) throw new Error(`the fixture's ${name} sample isn't a status`);
  return s;
}

const info: ModelInfo = {
  from: "folder",
  path: "C:\\data\\avatar\\fritz.vrm",
  found: true,
  bytes: 12_582_912,
  tooBig: false,
  wearing: "export",
  loadError: null,
  changed: false,
};

describe("the control window's wire", () => {
  it("calls the commands the shell registers, with the arguments it takes", () => {
    const c = fixture.commands;
    expect(CMD.getSettings).toBe(c.getSettings.name);
    expect(CMD.setSetting).toBe(c.setSetting.name);
    expect(Object.keys(args.setSetting("avatar.lookatCursor", false))).toEqual(c.setSetting.args);
    expect(CMD.setParked).toBe(c.setParked.name);
    expect(Object.keys(args.setParked(true))).toEqual(c.setParked.args);
    expect(CMD.avatarStatus).toBe(c.status.name);
    expect(CMD.avatarModel).toBe(c.avatarModel.name);
    expect(c.avatarModel.args).toEqual([]);
    expect(CMD.openFolder).toBe(c.openFolder.name);
    expect(Object.keys(args.openFolder("logs"))).toEqual(c.openFolder.args);
    expect(CMD.reloadModel).toBe(c.reloadModel.name);
    expect(c.reloadModel.args).toEqual([]);
    expect(CMD.mode).toBe(c.mode.name);
    expect(fixture.modeRequests).toContain(args.mode("arrange").mode);
  });

  it("hears the events the shell sends", () => {
    expect(SETTINGS_EVENT).toBe(fixture.events.settings);
    expect(RELOAD_EVENT).toBe(fixture.events.reload);
    expect(LOADED_EVENT).toBe(fixture.events.loaded);
  });

  /** The avatar's page says what it loaded, by the command the shell has. */
  it("hears from the page which model it loaded", () => {
    expect(LOADED_CMD).toBe(fixture.commands.avatarLoaded.name);
    expect(Object.keys(loadedArgs("placeholder", "bad file"))).toEqual(fixture.commands.avatarLoaded.args);
    // "none" from avatar_vrm is no export at all, not an error.
    expect(vrmFetchError("none")).toBeNull();
    expect(vrmFetchError("C:\\m\\fritz.vrm is larger than 268435456 bytes")).toContain("larger");
    expect(vrmFetchError(new Error("read failed"))).toBe("read failed");
  });

  /** FRITZ_SHOW_SETTINGS=<section>: the shell answers with one of the
   * page's section ids, and anything else is no section. */
  it("opens at a section the page has", () => {
    expect(CMD.controlOpening).toBe(fixture.commands.controlOpening.name);
    expect(fixture.commands.controlOpening.args).toEqual([]);
    expect([...SECTIONS]).toEqual(fixture.controlSections);
    for (const s of SECTIONS) expect(parseSection(s)).toBe(s);
    for (const bad of [null, "", "Appearance", "Brain", 1, {}]) expect(parseSection(bad)).toBeNull();
    expect(SECTIONS[0], "Brain comes first").toBe("brain");
  });

  /** The first run: the shell's CONTROL_WIZARD, which no section is. */
  it("opens at the first run when the shell says so", () => {
    expect(WIZARD).toBe(fixture.controlWizard);
    expect(parseOpening("wizard")).toBe("wizard");
    for (const s of SECTIONS) expect(parseOpening(s)).toBe(s);
    for (const bad of [null, "", "Wizard", "wizards", 1, {}]) expect(parseOpening(bad)).toBeNull();
    expect(parseSection("wizard"), "not a section").toBeNull();
  });

  /** The brain's link: the window saves the token and asks for the link,
   * and never asks for the token itself (`companion_config` is the avatar
   * page's alone). */
  it("hears of the brain's link, and only saves the token", () => {
    const c = fixture.commands;
    expect(CMD.getTokenPresent).toBe(c.getTokenPresent.name);
    expect(c.getTokenPresent.args).toEqual([]);
    expect(CMD.setToken).toBe(c.setToken.name);
    expect(Object.keys(args.setToken("t0k"))).toEqual(c.setToken.args);
    expect(CMD.companionStatus).toBe(c.companionStatus.name);
    expect(c.companionStatus.args).toEqual([]);
    expect(CMD.companionConnect).toBe(c.companionConnect.name);
    expect(CMD.companionSay).toBe(c.companionSay.name);
    expect(Object.keys(args.companionSay("good evening"))).toEqual(c.companionSay.args);
    expect(c.companionConnect.args).toEqual([]);
    expect(Object.values(CMD)).not.toContain(c.companionConfig.name);
    expect(COMPANION_STATUS_EVENT).toBe(fixture.events.companionStatus);
  });

  it("reads the link strictly", () => {
    for (const [name, sample] of Object.entries(fixture.companionStatusSamples)) {
      expect(parseCompanionStatus(sample), name).toEqual(sample);
    }
    const online = fixture.companionStatusSamples.online;
    for (const bad of [
      { ...online, link: "sideways" },
      { ...online, reason: "sulking" },
      { ...online, retryIn: -1 },
      { ...online, retryIn: 1.5 },
      { ...online, retryIn: "4" },
      { ...online, identity: 3 },
      { ...online, threadId: undefined },
      // What the brain can take: the protocol's caps, in a list.
      { ...online, serverCaps: ["text", "hearing"] },
      { ...online, serverCaps: "audio-in" },
      { ...online, serverCaps: [3] },
      { ...online, serverCaps: undefined },
      [online],
      null,
    ]) {
      expect(parseCompanionStatus(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  /** Settings › Hotkeys: a key is bound through `set_hotkey` (never
   * `set_setting`, which the shell refuses for hotkeys), and what is held
   * is read from `hotkeys_bound`, strictly. */
  it("binds keys through the shell's own command, for its actions", () => {
    const c = fixture.commands;
    expect(CMD.setHotkey).toBe(c.setHotkey.name);
    expect(Object.keys(args.setHotkey("prompt", "Ctrl+Alt+K"))).toEqual(c.setHotkey.args);
    expect(CMD.hotkeysBound).toBe(c.hotkeysBound.name);
    expect(c.hotkeysBound.args).toEqual([]);
    expect([...HOTKEY_ACTIONS].sort()).toEqual(fixture.hotkeyActions);
    expect(HOTKEY_DEFAULTS).toEqual(fixture.hotkeyDefaults);
    const bound = { prompt: "Ctrl+Alt+F", arrange: null, park: "Ctrl+Alt+H", ptt: "Ctrl+Alt+V" };
    expect(parseHotkeysBound(bound)).toEqual(bound);
    for (const bad of [
      null,
      [bound],
      { ...bound, prompt: 3 },
      { ...bound, ptt: 3 },
      { prompt: "Ctrl+Alt+F", arrange: null, park: "Ctrl+Alt+H" },
      { ...bound, mute: "Ctrl+Alt+M" },
      { arrange: null, park: null },
      { ...bound, park: undefined },
    ]) {
      expect(parseHotkeysBound(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  /** Settings › Diagnostics › The mouse hook: its status is read from
   * `mouse_hook_status` and heard as `hook:status`, strictly. */
  it("reads the mouse hook's status strictly", () => {
    expect(CMD.mouseHookStatus).toBe(fixture.commands.mouseHookStatus.name);
    expect(fixture.commands.mouseHookStatus.args).toEqual([]);
    expect(HOOK_STATUS_EVENT).toBe(fixture.events.hookStatus);
    expect(HOOK_KEYS).toEqual([...fixture.hookStatusKeys].sort());
    expect(parseHookStatus(HOOK_OUT)).toEqual(HOOK_OUT);
    expect(parseHookStatus(HOOK_IN)).toEqual(HOOK_IN);
    const { putBack: _, ...short } = HOOK_IN;
    for (const bad of [
      null,
      [HOOK_IN],
      short,
      { ...HOOK_IN, x: 1 },
      { ...HOOK_IN, on: "yes" },
      { ...HOOK_IN, installed: 1 },
      { ...HOOK_IN, panicked: null },
      { ...HOOK_IN, why: 3 },
      { ...HOOK_IN, events: -1 },
      { ...HOOK_IN, slowestMicros: 0.5 },
      { ...HOOK_IN, putBack: "2" },
      { ...HOOK_IN, onShape: Number.NaN },
    ]) {
      expect(parseHookStatus(bad), JSON.stringify(bad)).toBeNull();
    }
  });

  it("asks only for the folders the shell opens", () => {
    for (const which of ["logs", "avatar"] as const) expect(fixture.folders).toContain(args.openFolder(which).which);
  });

  const settings = {
    version: 1,
    brain: { url: "ws://127.0.0.1:8001/companion", clientId: "nick-pc", clientName: "NICK-PC" },
    avatar: { rect: null, scale: 1, parked: false, vrmPath: null, lookatCursor: true, clicks: true },
    hotkeys: { prompt: "Ctrl+Alt+F", ptt: "Ctrl+Alt+V", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" },
    diagnostics: { foregroundLog: false },
    theme: { theme: "dark", mood: "ink", accent: "brand", motion: 0.6, grain: true, motionUser: false },
  };

  it("reads the settings it uses, strictly", () => {
    expect(parseSettings(settings)).toEqual({
      brain: { url: "ws://127.0.0.1:8001/companion", clientId: "nick-pc", clientName: "NICK-PC" },
      avatar: { lookatCursor: true, clicks: true, vrmPath: null },
      hotkeys: { prompt: "Ctrl+Alt+F", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H", ptt: "Ctrl+Alt+V" },
      diagnostics: { foregroundLog: false },
      theme: { theme: "dark", mood: "ink", accent: "brand", motion: 0.6, grain: true, motionUser: false },
    });
    const { prompt: _p, ...noPrompt } = settings.hotkeys;
    expect(parseSettings({ ...settings, hotkeys: noPrompt })).toBeNull();
    const { ptt: _t, ...noPtt } = settings.hotkeys;
    expect(parseSettings({ ...settings, hotkeys: noPtt })).toBeNull();
    expect(parseSettings({ ...settings, hotkeys: { ...settings.hotkeys, ptt: null } })).toBeNull();
    expect(parseSettings({ ...settings, hotkeys: { ...settings.hotkeys, prompt: 7 } })).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, vrmPath: "D:/m/fritz.vrm" } })?.avatar.vrmPath)
      .toBe("D:/m/fritz.vrm");
    expect(parseSettings(null)).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, lookatCursor: "yes" } })).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, clicks: false } })?.avatar.clicks).toBe(false);
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, clicks: "yes" } })).toBeNull();
    const { clicks: _c, ...noClicks } = settings.avatar;
    expect(parseSettings({ ...settings, avatar: noClicks })).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, vrmPath: 3 } })).toBeNull();
    expect(parseSettings({ ...settings, hotkeys: { ...settings.hotkeys, park: null } })).toBeNull();
    expect(parseSettings({ ...settings, diagnostics: {} })).toBeNull();
    expect(parseSettings({ ...settings, theme: { ...settings.theme, mood: 1 } })).toBeNull();
    expect(parseSettings({ ...settings, theme: { ...settings.theme, mood: "violet" } })).toBeNull();
    const { avatar: _, ...noAvatar } = settings;
    expect(parseSettings(noAvatar)).toBeNull();
    expect(parseSettings({ ...settings, brain: { ...settings.brain, url: null } })).toBeNull();
    const { brain: __, ...noBrain } = settings;
    expect(parseSettings(noBrain)).toBeNull();
  });

  it("reads the model's report strictly", () => {
    expect(Object.keys(info).sort()).toEqual(fixture.modelInfoKeys);
    expect(parseModelInfo(info)).toEqual(info);
    expect(parseModelInfo({ ...info, bytes: null })).toEqual({ ...info, bytes: null });
    expect(parseModelInfo({ ...info, from: "elsewhere" })).toBeNull();
    expect(parseModelInfo({ ...info, bytes: -1 })).toBeNull();
    expect(parseModelInfo({ ...info, found: "yes" })).toBeNull();
    expect(parseModelInfo([info])).toBeNull();
    expect(parseModelInfo({ ...info, wearing: null })).toEqual({ ...info, wearing: null });
    expect(parseModelInfo({ ...info, wearing: "a hat" })).toBeNull();
    expect(parseModelInfo({ ...info, loadError: 3 })).toBeNull();
    expect(parseModelInfo({ ...info, changed: true })).toEqual({ ...info, changed: true });
    const { changed: _, ...unsaid } = info;
    expect(parseModelInfo(unsaid)).toBeNull();
  });
});

describe("what the control window says", () => {
  it("where he is", () => {
    expect(standing(null)).toEqual({ line: "Waiting for Fritz…", offerShow: false, canArrange: false });
    const shown: AvatarStatus = { ...status("arranging"), mode: "locked", canArrange: true };
    expect(standing(shown)).toEqual({ line: "He's on your desktop.", offerShow: false, canArrange: true });
    const parked = standing(status("parked"));
    expect(parked.offerShow).toBe(true);
    expect(parked.canArrange).toBe(false);
    expect(parked.line).toContain("out of sight");
    const closed = standing(status("closed"));
    expect(closed.offerShow).toBe(true);
    expect(closed.line).toContain("closed");
    const closing = standing(status("watchdog"));
    expect(closing.offerShow).toBe(true);
    const arranging = standing(status("arranging"));
    expect(arranging.canArrange).toBe(false);
    expect(arranging.offerShow).toBe(false);
    expect(arranging.line).toContain("Esc");
  });

  it("which model he wears", () => {
    expect(megabytes(12_582_912)).toBe("12.0 MB");
    const ok = modelLines(info);
    expect(ok.tone).toBe("ok");
    expect(ok.detail).toContain("12.0 MB");
    expect(modelLines({ ...info, found: false, bytes: null }).tone).toBe("quiet");
    expect(modelLines({ ...info, found: false, bytes: null }).detail).toContain(info.path);
    const missing = modelLines({ ...info, from: "setting", found: false, bytes: null });
    expect(missing.tone).toBe("warn");
    expect(missing.title).toContain("vrmPath");
    const big = modelLines({ ...info, bytes: 300 * 1024 * 1024, tooBig: true });
    expect(big.tone).toBe("warn");
    expect(big.detail).toContain("placeholder");
    expect(modelLines(null).tone).toBe("quiet");
    // There, but the page couldn't use it: never "Your model".
    const failed = modelLines({ ...info, wearing: "placeholder", loadError: "That file has no VRM in it." });
    expect(failed.tone).toBe("warn");
    expect(failed.title).toBe("Your model didn't load");
    expect(failed.detail).toBe(`${info.path}: That file has no VRM in it. He wears the placeholder.`);
    expect(modelLines({ ...info, wearing: "placeholder", loadError: null }).tone).toBe("warn");
    // Before the page's first load, the file is what there is to go on.
    expect(modelLines({ ...info, wearing: null }).tone).toBe("ok");
  });

  it("a file that isn't the one he loaded", () => {
    // Dropped in where there was none, or in place of a broken one: not
    // tried yet, so never "didn't load", and an old error never shown.
    for (const was of [{ wearing: "placeholder", loadError: null }, { wearing: "export", loadError: null }] as const) {
      const fresh = modelLines({ ...info, ...was, changed: true });
      expect(fresh.title).toBe("A new model is here");
      expect(fresh.tone).toBe("quiet");
      expect(fresh.detail).toBe(`${info.path} (12.0 MB). Load it again to wear it.`);
    }
    // The export he wears taken away: he still wears it, until a load.
    const gone = modelLines({ ...info, found: false, bytes: null, changed: true });
    expect(gone.tone).toBe("warn");
    expect(gone.title).toBe("Your model isn't there any more");
    expect(gone.detail).toContain(info.path);
    // The placeholder he wears, and nothing there now: as ever.
    const none = { ...info, found: false, bytes: null, wearing: "placeholder", changed: true } as const;
    expect(modelLines(none)).toEqual(modelLines({ ...none, changed: false }));
    // A new file too large: it is too large, whatever he wears now.
    const big = modelLines({ ...info, bytes: 300 * 1024 * 1024, tooBig: true, changed: true });
    expect(big.title).toBe("Your model is too large to load");
    expect(big.detail).not.toContain("placeholder");
  });
});

describe("what the control window says of his brain", () => {
  const URL = "ws://127.0.0.1:8001/companion";
  const shown = { ...status("arranging"), mode: "locked" as const };
  const link = (name: keyof typeof fixture.companionStatusSamples, more: Partial<LinkStatus> = {}): LinkStatus => {
    const s = parseCompanionStatus(fixture.companionStatusSamples[name]);
    if (!s) throw new Error(`the fixture's ${name} sample isn't a link`);
    return { ...s, ...more };
  };

  it("asks for the token first", () => {
    for (const s of [null, link("online"), link("refused")]) {
      const l = connectionLines(s, shown, false, URL);
      expect(l.title).toBe("He needs his brain's token");
      expect(l.detail).toContain("COMPANION_TOKEN");
      expect(l.tone).toBe("warn");
    }
  });

  it("waits to hear before saying anything", () => {
    expect(connectionLines(link("online"), shown, null, URL).tone).toBe("quiet");
    expect(connectionLines(null, shown, true, URL).title).toContain("Asking");
  });

  it("says his window is closed, since his page holds the connection", () => {
    for (const s of [status("closed"), status("watchdog")]) {
      const l = connectionLines(link("online"), s, true, URL);
      expect(l.title).toBe("His window is closed");
      expect(l.tone).toBe("quiet");
    }
    // Not yet known: what the page last said stands.
    expect(connectionLines(link("online"), null, true, URL).title).toBe("Connected");
  });

  it("connecting, and connected", () => {
    for (const s of [link("connecting"), link("unconfigured")]) {
      expect(connectionLines(s, shown, true, URL)).toEqual({ title: "Connecting…", detail: `To his brain at ${URL}.`, tone: "quiet" });
    }
    expect(connectionLines(link("online"), shown, true, URL)).toEqual({ title: "Connected", detail: `To his brain at ${URL}.`, tone: "ok" });
  });

  it("while he tries again: why, and when", () => {
    const unreachable = connectionLines(link("offline"), shown, true, URL);
    expect(unreachable.title).toBe("Can't reach his brain");
    expect(unreachable.detail).toBe(`Nothing answers at ${URL}: is the brain running, with COMPANION_ENABLED on? Trying again in 4 s.`);
    expect(unreachable.tone).toBe("warn");
    expect(connectionLines(link("offline", { retryIn: null }), shown, true, URL).detail).toContain("Trying again now.");
    const full = connectionLines(link("offline", { reason: "full", retryIn: 30 }), shown, true, URL);
    expect(full.title).toContain("all the desks");
    expect(full.detail).toBe("COMPANION_MAX_CLIENTS desks are connected to it already. Trying again in 30 s.");
    const quiet = connectionLines(link("offline", { reason: "quiet", retryIn: 1 }), shown, true, URL);
    expect(quiet.title).toBe("His brain stopped answering");
    expect(quiet.tone).toBe("warn");
  });

  it("each refusal, and what to do about it", () => {
    const token = connectionLines(link("refused"), shown, true, URL);
    expect(token.title).toBe("His brain refused the token");
    expect(token.detail).toContain("COMPANION_TOKEN");
    expect(token.tone).toBe("bad");
    const version = connectionLines(link("refused", { reason: "version" }), shown, true, URL);
    expect(version.title).toBe("His brain speaks another version");
    expect(version.tone).toBe("bad");
    const hello = connectionLines(link("refused", { reason: "hello" }), shown, true, URL);
    expect(hello.title).toBe("His brain couldn't read his hello");
    expect(hello.detail).toContain("log");
    expect(hello.tone).toBe("bad");
  });

  it("who he is to the brain, while online", () => {
    expect(identityLine(link("online"))).toBe("desktop-nick-pc, talking with Nick, in the thread desktop-nick-pc#desktop.");
    expect(identityLine(link("online", { displayName: null, threadId: null }))).toBe("desktop-nick-pc.");
    expect(identityLine(link("offline"))).toBeNull();
    expect(identityLine(link("online", { identity: null }))).toBeNull();
    expect(identityLine(null)).toBeNull();
  });

  it("whether the brain is on this PC", () => {
    for (const url of [URL, "ws://localhost:8001/companion", "WSS://LOCALHOST/x", "ws://[::1]:8001/companion",
      "ws://127.255.0.9:1/x", "ws://fritz.localhost/x"]) {
      expect(onThisPc(url), url).toBe(true);
    }
    for (const url of ["ws://10.0.0.5:8001/companion", "wss://brain.example/companion", "ws://128.0.0.1/x",
      "ws://localhost.example/x", "ws://127.0.0.1.example/x", "not a url", ""]) {
      expect(onThisPc(url), url).toBe(false);
    }
  });
});

describe("what Settings says of clicks on him", () => {
  it("the switch says what is watched, and that no record is kept", () => {
    expect(CLICKS_DESC).toMatch(/^Double-click him to ask him something; right-click him for his menu\. /);
    expect(CLICKS_DESC).toMatch(/A click on his body is his and never reaches the window behind; around him, clicks go through\./);
    expect(CLICKS_DESC).toMatch(/mouse buttons across the desktop/);
    expect(CLICKS_DESC).toMatch(/keeps no record of where you click/);
  });

  it("the mouse hook's line: in and how quick, or out and why, and every time it went quiet and was put back", () => {
    expect(hookLine(null)).toBe("Not known.");
    expect(hookLine(HOOK_OUT)).toBe(`Out: ${fixture.hookOff.hidden}.`);
    expect(hookLine({ ...HOOK_OUT, on: false, why: fixture.hookOff.switch })).toBe(
      "Out: clicks on him are off in Settings.",
    );
    expect(hookLine({ ...HOOK_OUT, why: null })).toBe("Out.");
    expect(hookLine(HOOK_IN)).toBe("In while he's on screen; the slowest answer took 0.04 ms.");
    expect(hookLine({ ...HOOK_IN, slowestMicros: 1234 })).toBe("In while he's on screen; the slowest answer took 1.23 ms.");
    expect(hookLine({ ...HOOK_IN, events: 0, slowestMicros: 0 })).toBe("In while he's on screen.");
    // "In" only while it is in, whatever else it has heard.
    expect(hookLine({ ...HOOK_IN, installed: false, why: "he's hidden" })).toBe("Out: he's hidden.");
    expect(hookLine({ ...HOOK_IN, putBack: 1 })).toMatch(/ It went quiet once while the mouse moved, and Fritz put it back\.$/);
    expect(hookLine({ ...HOOK_IN, putBack: 3 })).toMatch(/ It went quiet 3 times while the mouse moved, and Fritz put it back each time\.$/);
    expect(hookLine({ ...HOOK_OUT, panicked: true })).toBe(
      "Out: he's hidden. It failed, and lets every click through until it goes in again.",
    );
    // Failed while still in: what puts it in again.
    expect(hookLine({ ...HOOK_IN, panicked: true })).toBe(
      "In while he's on screen; the slowest answer took 0.04 ms. It failed, and lets every click through until it is put in again: turn Clicks on him off and on.",
    );
  });
});

describe("what Settings › Hotkeys says", () => {
  it("each row's line: the refusal, a key not working, or what it does", () => {
    const set = "Ctrl+Alt+K";
    expect(hotkeyLine("prompt", set, set, null)).toEqual({ text: HOTKEY_DOES.prompt, alert: false });
    expect(hotkeyLine("park", set, ` ${set} `, null)).toEqual({ text: HOTKEY_DOES.park, alert: false });
    // Until the shell says what it holds, the key set is taken to work.
    expect(hotkeyLine("arrange", set, undefined, null)).toEqual({ text: HOTKEY_DOES.arrange, alert: false });
    expect(hotkeyLine("prompt", set, null, null)).toEqual({
      text: "Not working: he couldn't have Ctrl+Alt+K when he started (another app may hold it). Pick another.",
      alert: false,
    });
    expect(hotkeyLine("prompt", "Ctrl+Alt+Nope", "Ctrl+Alt+F", null)).toEqual({
      text: "Working on Ctrl+Alt+F: Ctrl+Alt+Nope couldn't be used. Pick another, or press Default.",
      alert: false,
    });
    // The shell's refusal of the key just tried comes first, as an alert,
    // and the row still says if its key isn't working.
    const why = "Ctrl+Alt+A is already Fritz's key to arrange him.";
    for (const active of [set, undefined]) {
      expect(hotkeyLine("prompt", set, active, why)).toEqual({ text: why, alert: true });
    }
    expect(hotkeyLine("prompt", set, null, why)).toEqual({
      text: `${why} Not working: he couldn't have Ctrl+Alt+K when he started (another app may hold it). Pick another.`,
      alert: true,
    });
    expect(hotkeyLine("prompt", set, "Ctrl+Alt+F", why)).toEqual({
      text: `${why} Working on Ctrl+Alt+F: Ctrl+Alt+K couldn't be used. Pick another, or press Default.`,
      alert: true,
    });
    expect(new Set(Object.values(HOTKEY_DOES)).size).toBe(4);
  });

  /** The settings and the shell say Super; the owner's keycap says Win. */
  it("calls the Windows key Win", () => {
    expect(winNamed("Ctrl+Super+P is already Fritz's key to arrange him.")).toBe("Ctrl+Win+P is already Fritz's key to arrange him.");
    expect(winNamed("Super+F9, cmd+K and Command+J")).toBe("Win+F9, Win+K and Win+J");
    expect(winNamed("A Superb key: Super")).toBe("A Superb key: Super");
    expect(hotkeyLine("park", "Ctrl+Super+P", null, null).text).toContain("he couldn't have Ctrl+Win+P when");
    expect(hotkeyLine("park", "Ctrl+Super+P", "Super+J", null).text).toBe(
      "Working on Win+J: Ctrl+Win+P couldn't be used. Pick another, or press Default.",
    );
    expect(hotkeyLine("park", "F13", "F13", "Another app already uses Super+F13: pick another.").text)
      .toBe("Another app already uses Win+F13: pick another.");
    expect(orPress("Ctrl+Super+P")).toBe("Or press Ctrl+Win+P.");
  });

  it("offers a key only while it is held", () => {
    expect(orPress("Ctrl+Alt+H")).toBe("Or press Ctrl+Alt+H.");
    expect(orPress(null)).toBe("");
    expect(orPress(undefined)).toBe("");
  });

  /** DivoraVoice's hint is --text-lo at 11 px, below AA on the capture's
   * --surface-2 in every mood (worst in the dark ones); the lift says it
   * in --text-mid, and its keycaps are --text-hi on --surface-3. */
  it("the capture's words and keycaps are AA in every mood", () => {
    const css = readFileSync(fileURLToPath(new URL("../divora/components.css", import.meta.url)), "utf8")
      .replace(/\r\n/g, "\n");
    const rule = (sel: string) => {
      const at = css.indexOf(`\n${sel} {`);
      expect(at, sel).toBeGreaterThan(0);
      return css.slice(at, css.indexOf("}", at));
    };
    expect(rule(".hotkey-capture-hint")).toContain("color: var(--text-mid);");
    expect(rule(".kbd")).toContain("color: var(--text-hi);");
    expect(rule(".kbd")).toContain("background: var(--surface-3);");
    expect(rule(".hotkey-capture")).toContain("background: var(--surface-2);");
    // The ring follows the accent rather than DivoraVoice's violet.
    expect(rule(".hotkey-capture.is-capturing")).toContain("var(--indigo) 20%");
    let loFails = 0;
    for (const [name, selectors] of Object.entries(COMBOS)) {
      const s2 = token("--surface-2", selectors);
      expect(contrast(token("--text-mid", selectors), s2), `hint on ${name}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(token("--text-hi", selectors), token("--surface-3", selectors)), `keycap on ${name}`)
        .toBeGreaterThanOrEqual(4.5);
      if (contrast(token("--text-lo", selectors), s2) < 4.5) loFails += 1;
    }
    expect(loFails, "--text-lo fails in every mood, which is why it changed").toBe(Object.keys(COMBOS).length);
  });

  /** The first run's words: its step line in --text-mid (not the
   * eyebrow's --text-lo) on the page, the rail in --text-mid on
   * --surface-1, and the numbers and ticks in the text's colours on
   * --surface-3 (--success is far below 3:1 on the light surfaces). */
  it("the first run's words are AA in every mood", () => {
    const css = readFileSync(fileURLToPath(new URL("./control.css", import.meta.url)), "utf8").replace(/\r\n/g, "\n");
    const rule = (sel: string) => {
      const at = css.indexOf(`\n${sel} {`);
      expect(at, sel).toBeGreaterThan(0);
      return css.slice(at, css.indexOf("}", at));
    };
    expect(rule(".wizard-count")).toContain("color: var(--text-mid);");
    expect(rule(".wizard-step")).toContain("color: var(--text-mid);");
    expect(rule(".wizard-num")).toContain("color: var(--text-mid);");
    expect(rule(".wizard-num")).toContain("background: var(--surface-3);");
    expect(rule(".wizard-step.is-done .wizard-num")).toContain("color: var(--text-hi);");
    expect(rule(".wizard-rail")).toContain("background: var(--surface-1);");
    for (const [name, selectors] of Object.entries(COMBOS)) {
      const t = (n: string) => token(n, selectors);
      expect(contrast(t("--text-mid"), t("--bg")), `step line on ${name}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t("--text-mid"), t("--surface-1")), `rail on ${name}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t("--text-mid"), t("--surface-3")), `number on ${name}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(t("--text-hi"), t("--surface-3")), `tick on ${name}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
