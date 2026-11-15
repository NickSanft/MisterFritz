/**
 * The control window (tray > Settings…): his brain, where Fritz is, his
 * keys, how the house looks, his model, and diagnostics. Plan 14 spreads it
 * across PRs 2-4 and 8: Voice comes with PR 8.
 *
 * What it shows comes from the shell (status events, settings:changed,
 * companion:status and a fresh read whenever the window is focused); what
 * it says comes from view.ts, which is tested.
 */
import { createSignal, For, onCleanup, onMount, Show, type JSX } from "solid-js";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { STATUS_EVENT, parseStatus, type AvatarStatus } from "../avatar/status";
import {
  COMPANION_STATUS_EVENT, LOADED_EVENT, SETTINGS_EVENT, arrange, avatarModel, avatarStatus, companionConnect, companionSay,
  companionStatus, controlOpening, getSettings, getTokenPresent, hotkeysBound, openFolder, parseCompanionStatus,
  parseSettings, reloadModel, setHotkey, setParked, setSetting, setToken, type ControlSettings, type HotkeyAction,
  type HotkeysBound, type ModelInfo,
} from "./api";
import type { LinkStatus } from "../shared/companion";
import { afterAsking } from "../shared/asking";
import { Toggle } from "../divora/components/Toggle";
import { Sigil } from "../divora/sigils/Sigil";
import { reducedMotion } from "../divora/theme";
import { Row, Section } from "./layout";
import { Appearance } from "./sections/Appearance";
import { Brain } from "./sections/Brain";
import { Hotkeys } from "./sections/Hotkeys";
import { modelLines, orPress, standing } from "./view";
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
  const refresh = () =>
    attempt("ask Fritz how he is", async () => {
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
    });

  const unlisten: Promise<UnlistenFn>[] = [];
  const onFocus = () => void refresh();
  let offReduced = () => {};
  onMount(() => {
    unlisten.push(listen<unknown>(STATUS_EVENT, (e) => {
      const s = parseStatus(e.payload);
      if (s) setStatus(s);
    }));
    unlisten.push(listen<unknown>(SETTINGS_EVENT, (e) => took(parseSettings(e.payload))));
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
    // Open at the section the shell names (FRITZ_SHOW_SETTINGS=appearance),
    // once what's above it has its content. Never smoothly: motion here
    // follows the theme, and this is a jump, not a flourish.
    void Promise.all([refresh(), controlOpening().catch(() => null)]).then(([, at]) => {
      if (at) document.getElementById(at)?.scrollIntoView({ block: "start" });
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

  const here = () => standing(status());
  const lines = () => modelLines(model());

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

      <main class="sections">
        <Brain
          url={settings()?.brain.url ?? null}
          link={link()}
          avatar={status()}
          tokenPresent={tokenPresent()}
          setUrl={(url) => set("brain.url", url, "change his brain's address")}
          saveToken={saveToken}
          reconnect={() => void attempt("reconnect", companionConnect)}
          ask={(text) => attempt("ask him", () => companionSay(text))}
        />

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
        </Section>

        <Hotkeys keys={settings()?.hotkeys ?? null} bound={bound()} set={rebind} />

        <Appearance
          theme={settings()?.theme ?? null}
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
          <Row title={lines().title} desc={lines().detail}>
            <span classList={{ dot: true, [`tone-${lines().tone}`]: true }} />
          </Row>
          <Row title="His model's folder" desc="Where avatar\fritz.vrm goes, unless avatar.vrmPath points elsewhere.">
            <button type="button" class="btn btn-secondary" onClick={() => attempt("open the folder", () => openFolder("avatar"))}>
              Open
            </button>
          </Row>
          <Row title="Load it again" desc="After you've put a new export in place.">
            <button
              type="button"
              class="btn btn-secondary"
              onClick={() => attempt("load his model", reloadModel)}
            >
              Load
            </button>
          </Row>
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
        </Section>

        <Section id="about" sigil="info" eyebrow="About" title={`Mister Fritz ${version()}`} desc="Fritz's body on the Windows desktop.">
          <For each={[
            "Parts of the window shell, and this window's styling, controls and sigils, are lifted from DivoraVoiceMod (MIT, Copyright (c) 2026 Divora Contributors).",
            "The typefaces are Bricolage Grotesque, Space Grotesk and Space Mono, under the SIL Open Font License.",
            "Some plumbing follows the owner's LorraineAtelier.",
          ]}>
            {(line) => <Row title={line} />}
          </For>
        </Section>
      </main>

      <Show when={problem()}>
        <div class="problem" role="alert">
          {problem()}
        </div>
      </Show>
    </div>
  );
}
