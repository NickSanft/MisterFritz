import { describe, expect, it } from "vitest";
import fixture from "../avatar/wire.fixture.json";
import {
  LOADED_CMD, RELOAD_EVENT, loadedArgs, parseStatus, vrmFetchError, type AvatarStatus,
} from "../avatar/status";
import {
  CMD, LOADED_EVENT, SECTIONS, SETTINGS_EVENT, args, parseModelInfo, parseSection, parseSettings, type ModelInfo,
} from "./api";
import { megabytes, modelLines, standing } from "./view";

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
    for (const bad of [null, "", "Appearance", "brain", 1, {}]) expect(parseSection(bad)).toBeNull();
  });

  it("asks only for the folders the shell opens", () => {
    for (const which of ["logs", "avatar"] as const) expect(fixture.folders).toContain(args.openFolder(which).which);
  });

  const settings = {
    version: 1,
    avatar: { rect: null, scale: 1, parked: false, vrmPath: null, lookatCursor: true },
    hotkeys: { prompt: "Ctrl+Alt+F", ptt: "Ctrl+Alt+Space", arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" },
    diagnostics: { foregroundLog: false },
    theme: { theme: "dark", mood: "ink", accent: "brand", motion: 0.6, grain: true, motionUser: false },
  };

  it("reads the settings it uses, strictly", () => {
    expect(parseSettings(settings)).toEqual({
      avatar: { lookatCursor: true, vrmPath: null },
      hotkeys: { arrange: "Ctrl+Alt+A", park: "Ctrl+Alt+H" },
      diagnostics: { foregroundLog: false },
      theme: { theme: "dark", mood: "ink", accent: "brand", motion: 0.6, grain: true, motionUser: false },
    });
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, vrmPath: "D:/m/fritz.vrm" } })?.avatar.vrmPath)
      .toBe("D:/m/fritz.vrm");
    expect(parseSettings(null)).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, lookatCursor: "yes" } })).toBeNull();
    expect(parseSettings({ ...settings, avatar: { ...settings.avatar, vrmPath: 3 } })).toBeNull();
    expect(parseSettings({ ...settings, hotkeys: { ...settings.hotkeys, park: null } })).toBeNull();
    expect(parseSettings({ ...settings, diagnostics: {} })).toBeNull();
    expect(parseSettings({ ...settings, theme: { ...settings.theme, mood: 1 } })).toBeNull();
    expect(parseSettings({ ...settings, theme: { ...settings.theme, mood: "violet" } })).toBeNull();
    const { avatar: _, ...noAvatar } = settings;
    expect(parseSettings(noAvatar)).toBeNull();
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
