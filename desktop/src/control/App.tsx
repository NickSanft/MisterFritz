/**
 * The control window (tray > Settings…): his brain, where Fritz is, his
 * keys, how the house looks, his model, and diagnostics; or, at the first
 * run (no token yet) and from About, the first-run wizard in their place.
 * Plan 14 spreads it across PRs 2-4 and 8: Voice comes with PR 8.
 *
 * What it shows comes from the shell (status events, settings:changed,
 * companion:status and a fresh read whenever the window is focused); what
 * it says comes from view.ts, which is tested.
 */
import { createSignal, For, Match, onCleanup, onMount, Show, Switch, type JSX } from "solid-js";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { STATUS_EVENT, parseStatus, type AvatarStatus } from "../avatar/status";
import {
  COMPANION_STATUS_EVENT, HOOK_STATUS_EVENT, LOADED_EVENT, SETTINGS_EVENT, WIZARD, arrange, avatarModel, avatarStatus,
  companionConnect, companionSay, companionStatus, controlOpening, getSettings, getTokenPresent, hotkeysBound,
  mouseHookStatus, openFolder, parseCompanionStatus, parseHookStatus, parseSettings, reloadModel, setHotkey, setParked,
  setSetting, setToken, type ControlSettings, type HookStatus, type HotkeyAction, type HotkeysBound, type ModelInfo,
} from "./api";
import type { LinkStatus } from "../shared/companion";
import { afterAsking } from "../shared/asking";
import { Toggle } from "../divora/components/Toggle";
import { Sigil } from "../divora/sigils/Sigil";
import { reducedMotion } from "../divora/theme";
import { Row, Section } from "./layout";
import { Appearance } from "./sections/Appearance";
import { Brain } from "./sections/Brain";
import { Hotkeys, type HotkeysProps } from "./sections/Hotkeys";
import { ModelRows, type ModelRowsProps } from "./sections/Model";
import type { BrainProps } from "./sections/Brain";
import { CLICKS_DESC, hookLine, orPress, standing } from "./view";
import { Wizard } from "./Wizard";
import { wearTheme } from "./wear";

export function App(): JSX.Element {
  const [status, setStatus] = createSignal<AvatarStatus | null>(null);
  const [settings, setSettings] = createSignal<ControlSettings | null>(null);
  const [model, setModel] = createSignal<ModelInfo | null>(null);
  const [version, setVersion] = createSignal("");
  const [problem, setProblem] = createSignal<string | null>(null);
  const [link, setLink] = createSignal<LinkStatus | null>(null);
  const [tokenPresent, setTokenPresent] = createSignal<boolean | null>(null);
  /** The keys the shell holds now, which may not be the ones set. */
  const [bound, setBound] = createSignal<HotkeysBound | null>(null);
  /** The mouse hook, for Diagnostics: read with the rest, and kept up to
   * date by the shell's word whenever it goes in or out. */
  const [hook, setHook] = createSignal<HookStatus | null>(null);
  /** Hook reports heard, so a late answer can't undo a newer one. */
  let hookEvents = 0;
  const readHook = (): void => {
    const heard = hookEvents;
    void mouseHookStatus().then((h) => {
      if (h && heard === hookEvents) setHook(h);
    }, () => {});
  };
  /** The first run, shown in place of the sections; null until the shell
   * has said where to open, so neither view shows only to be replaced. */
  const [wizard, setWizard] = createSignal<boolean | null>(null);
  /** Link reports heard, so a late answer to `companion_status` can't undo
   * a newer one. */
  let linkEvents = 0;

  const reduced = reducedMotion();
  const [lessMotion, setLessMotion] = createSignal(reduced.now());
  const took = (s: ControlSettings | null) => {
    if (!s) return;
    setSettings(s);
    wearTheme(document, s.theme, reduced.now());
  };
  /** What the window's own look at the shell is called in the banner. */
  const ASKING = "ask Fritz how he is";
  /** Run a command, showing what went wrong rather than nothing; whether
   * it went through. */
  const attempt = async (what: string, act: () => Promise<unknown>): Promise<boolean> => {
    setProblem(null);
    try {
      await act();
      return true;
    } catch (e) {
      setProblem(`Couldn't ${what}: ${typeof e === "string" ? e : String(e)}`);
      return false;
    }
  };
  /** Ask the shell how everything stands. Its own failure goes in the
   * banner, and only that is cleared when it next succeeds: another
   * command's refusal (a token refused while the window was away) stays to
   * be read when the window comes back. */
  const refresh = async (): Promise<void> => {
    try {
      const heard = linkEvents;
      const [st, se, mo, tp, li, bo] = await Promise.all([
        avatarStatus(), getSettings(), avatarModel(), getTokenPresent(), companionStatus(), hotkeysBound(),
      ]);
      if (st) setStatus(st);
      took(se);
      setModel(mo);
      setTokenPresent(tp);
      if (bo) setBound(bo);
      if (li) setLink(afterAsking<LinkStatus | null>(li, heard, linkEvents, link()));
      if (problem()?.startsWith(`Couldn't ${ASKING}:`)) setProblem(null);
    } catch (e) {
      setProblem(`Couldn't ${ASKING}: ${typeof e === "string" ? e : String(e)}`);
    }
    readHook();
  };

  const unlisten: Promise<UnlistenFn>[] = [];
  const onFocus = () => void refresh();
  let offReduced = () => {};
  onMount(() => {
    unlisten.push(listen<unknown>(STATUS_EVENT, (e) => {
      const s = parseStatus(e.payload);
      if (s) setStatus(s);
    }));
    unlisten.push(listen<unknown>(SETTINGS_EVENT, (e) => {
      const s = parseSettings(e.payload);
      const was = settings()?.audio.muted;
      took(s);
      // Muted or unmuted from the tray: push-to-talk's key was let go of,
      // or taken again, before this was said, so what is held is asked anew.
      if (s && was !== undefined && s.audio.muted !== was) {
        void hotkeysBound().then((b) => b && setBound(b), () => {});
      }
    }));
    // The mouse hook went in or out.
    unlisten.push(listen<unknown>(HOOK_STATUS_EVENT, (e) => {
      const h = parseHookStatus(e.payload);
      if (!h) return;
      hookEvents += 1;
      setHook(h);
    }));
    // The avatar's page says where the brain's link stands, on every change.
    unlisten.push(listen<unknown>(COMPANION_STATUS_EVENT, (e) => {
      const s = parseCompanionStatus(e.payload);
      if (!s) return;
      linkEvents += 1;
      setLink(s);
      // The token may have been saved some other way than this window's
      // Connect (or the file taken away): ask again with each report.
      void getTokenPresent().then(setTokenPresent, () => {});
    }));
    // The avatar's page has loaded a model (at start, or after Load): what it
    // wears now, and why if not the export.
    unlisten.push(listen<unknown>(LOADED_EVENT, () => void avatarModel().then(setModel, () => {})));
    // The owner may have dropped a model in the folder since: look again
    // whenever the window comes forward.
    window.addEventListener("focus", onFocus);
    // Windows asking for less motion (or no longer) changes the level.
    offReduced = reduced.onChange(() => {
      setLessMotion(reduced.now());
      const s = settings();
      if (s) wearTheme(document, s.theme, reduced.now());
    });
    void getVersion().then(setVersion, () => setVersion("?"));
    // Open at the first run (no token yet), or at the section the shell
    // names (FRITZ_SHOW_SETTINGS=appearance) once what's above it has its
    // content. Never smoothly: motion here follows the theme, and this is a
    // jump, not a flourish.
    const opening = controlOpening().catch(() => null);
    void opening.then((at) => setWizard(at === WIZARD));
    void Promise.all([refresh(), opening]).then(([, at]) => {
      if (at && at !== WIZARD) document.getElementById(at)?.scrollIntoView({ block: "start" });
    });
  });
  onCleanup(() => {
    window.removeEventListener("focus", onFocus);
    offReduced();
    for (const u of unlisten) void u.then((f) => f());
  });

  const set = (path: string, value: unknown, what: string) =>
    attempt(what, async () => took(await setSetting(path, value)));
  /** Bind a key: null once it is bound and saved, else the shell's words,
   * said on its own row rather than the window's banner. What the shell
   * holds is asked again either way. */
  const rebind = async (action: HotkeyAction, accelerator: string): Promise<string | null> => {
    try {
      took(await setHotkey(action, accelerator));
      return null;
    } catch (e) {
      return typeof e === "string" ? e : String(e);
    } finally {
      await hotkeysBound().then((b) => b && setBound(b), () => {});
    }
  };
  const saveToken = async (token: string) => {
    const ok = await attempt("save the token", () => setToken(token));
    if (ok) setTokenPresent(true);
    return ok;
  };

  /** The sections' props, which the first run's steps share. */
  const brainProps = (): BrainProps => ({
    url: settings()?.brain.url ?? null,
    link: link(),
    avatar: status(),
    tokenPresent: tokenPresent(),
    promptKey: bound()?.prompt ?? null,
    setUrl: (url) => set("brain.url", url, "change his brain's address"),
    saveToken,
    reconnect: () => void attempt("reconnect", companionConnect),
    ask: (text) => attempt("ask him", () => companionSay(text)),
  });
  const hotkeysProps = (): HotkeysProps => ({
    keys: settings()?.hotkeys ?? null,
    bound: bound(),
    muted: settings()?.audio.muted ?? false,
    set: rebind,
  });
  const modelProps = (): ModelRowsProps => ({
    model: model(),
    openFolder: () => void attempt("open the folder", () => openFolder("avatar")),
    reload: () => void attempt("load his model", reloadModel),
  });
  /** The first run's last step: bring him onto the desktop (opening his
   * window again if need be), then put the first run and this window
   * away, as the × does. Refused, both stay, and the banner says why. */
  const summon = async () => {
    const shown = await attempt("summon him", async () => {
      const s = await setParked(false);
      if (s) setStatus(s);
    });
    if (!shown) return;
    setWizard(false);
    await getCurrentWindow().hide();
  };

  const here = () => standing(status());

  return (
    <div class="shell">
      <header class="titlebar" data-tauri-drag-region>
        <span class="brand" data-tauri-drag-region>Mister Fritz</span>
        <span class="eyebrow" data-tauri-drag-region>Settings</span>
        <span class="titlebar-local" data-tauri-drag-region>
          <Sigil name="lock" size={13} /> LOCAL · NO ACCOUNT
        </span>
        <div class="titlebar-buttons">
          <button type="button" class="icon-btn" aria-label="Minimise" onClick={() => void getCurrentWindow().minimize()}>
            –
          </button>
          <button type="button" class="icon-btn" aria-label="Close" onClick={() => void getCurrentWindow().hide()}>
            ×
          </button>
        </div>
      </header>

      <Switch>
        <Match when={wizard() === true}>
          <Wizard
            brain={brainProps()}
            model={modelProps()}
            hotkeys={hotkeysProps()}
            link={link()}
            modelInfo={model()}
            bound={bound()}
            tokenPresent={tokenPresent()}
            skip={() => setWizard(false)}
            summon={summon}
          />
        </Match>
        <Match when={wizard() === false}>
          <main class="sections">
            <Brain {...brainProps()} />

            <Section id="fritz" sigil="butler" eyebrow="Fritz" title="Where he is" desc={here().line}>
              <Row
                title={here().offerShow ? "Show him" : "Hide him"}
                desc={orPress(bound()?.park) || undefined}
              >
                <button
                  type="button"
                  class="btn btn-secondary"
                  disabled={!status()}
                  onClick={() => attempt(here().offerShow ? "show him" : "hide him", async () => {
                    const s = await setParked(!here().offerShow);
                    if (s) setStatus(s);
                  })}
                >
                  {here().offerShow ? "Show" : "Hide"}
                </button>
              </Row>
              <Row
                title="Arrange him"
                desc={`Drag him where you like and resize him from the corner; Esc locks him there.${
                  orPress(bound()?.arrange) ? ` ${orPress(bound()?.arrange)}` : ""}`}
              >
                <button type="button" class="btn btn-secondary" disabled={!here().canArrange} onClick={() => attempt("arrange him", arrange)}>
                  Arrange…
                </button>
              </Row>
              <Row title="His eyes follow the cursor" desc="His head turns a little toward it too.">
                <Toggle
                  ariaLabel="His eyes follow the cursor"
                  on={settings()?.avatar.lookatCursor ?? false}
                  disabled={!settings()}
                  onChange={(on) => set("avatar.lookatCursor", on, "change that")}
                />
              </Row>
              <Row title="Clicks on him" desc={CLICKS_DESC}>
                <Toggle
                  ariaLabel="Clicks on him"
                  on={settings()?.avatar.clicks ?? false}
                  disabled={!settings()}
                  onChange={(on) => void set("avatar.clicks", on, "change that").then(readHook)}
                />
              </Row>
            </Section>

            <Hotkeys {...hotkeysProps()} />

            <Appearance
              theme={settings()?.theme ?? null}
              wallpaper={settings()?.wallpaper.on ?? null}
              lessMotion={lessMotion()}
              set={(path, value) => void set(path, value, "change that")}
            />

            <Section
              id="model"
              sigil="image"
              eyebrow="Model"
              title="What he looks like"
              desc="A VRoid Studio export, or the placeholder until there is one."
            >
              <ModelRows {...modelProps()} />
            </Section>

            <Section id="diagnostics" sigil="search" eyebrow="Diagnostics" title="If something goes wrong" desc="Everything stays on this PC.">
              <Row title="Fritz's logs" desc="startup.log and crash.log, this run's log, and the settings.">
                <button type="button" class="btn btn-secondary" onClick={() => attempt("open the logs", () => openFolder("logs"))}>
                  Open
                </button>
              </Row>
              <Row
                title="Record which app has the keyboard"
                desc="Spike 0's check that Fritz never takes it. From the next start; it writes other apps' window titles to diagnostics\foreground.log."
              >
                <Toggle
                  ariaLabel="Record which app has the keyboard"
                  on={settings()?.diagnostics.foregroundLog ?? false}
                  disabled={!settings()}
                  onChange={(on) => set("diagnostics.foregroundLog", on, "change that")}
                />
              </Row>
              <Row title="The mouse hook" desc={hookLine(hook())} />
            </Section>

            <Section id="about" sigil="info" eyebrow="About" title={`Mister Fritz ${version()}`} desc="Fritz's body on the Windows desktop.">
              <For each={[
                "Parts of the window shell, and this window's styling, controls and sigils, are lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors).",
                "The typefaces are Bricolage Grotesque, Space Grotesk and Space Mono, under the SIL Open Font License.",
                "Some plumbing follows the owner's LorraineAtelier.",
              ]}>
                {(line) => <Row title={line} />}
              </For>
              <Row title="Replay the first run" desc="Its four steps again: his brain, his look, his keys, and summoning him.">
                <button type="button" class="btn btn-secondary" onClick={() => setWizard(true)}>
                  Replay
                </button>
              </Row>
            </Section>
          </main>
        </Match>
      </Switch>

      <Show when={problem()}>
        <div class="problem" role="alert">
          {problem()}
        </div>
      </Show>
    </div>
  );
}
