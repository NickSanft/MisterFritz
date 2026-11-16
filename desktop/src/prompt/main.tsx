/**
 * The prompt's page (Ask Fritz): see Prompt.tsx. It wears the owner's theme
 * (never the control window's grain or stylesheets), and hears where the
 * brain's link and his window stand as every window does: listening first,
 * then asking, and again each time the prompt gets the keyboard.
 */
import { render } from "solid-js/web";
import { createSignal } from "solid-js";
import { listen } from "@tauri-apps/api/event";
import "../divora/tokens.css";
import "../divora/fonts.css";
import "./prompt.css";
import { watchTheme } from "../divora/theme";
import { STATUS_EVENT, parseStatus, type AvatarStatus } from "../avatar/status";
import type { LinkStatus } from "../shared/companion";
import { COMPANION_STATUS_EVENT, parseCompanionStatus } from "../shared/link";
import { afterAsking } from "../shared/asking";
import { WAITING_EVENT, avatarStatus, close, companionStatus, parseWaiting, ready, say } from "./api";
import { Prompt } from "./Prompt";
import { status } from "./view";

const [link, setLink] = createSignal<LinkStatus | null>(null);
const [avatar, setAvatar] = createSignal<AvatarStatus | null>(null);
/** The shell says the prompt waits for a click. */
const [waiting, setWaiting] = createSignal(false);
/** Events heard, so a late answer can't undo a newer event. */
let linkEvents = 0;
let avatarEvents = 0;

async function ask(): Promise<void> {
  const [l, a] = [linkEvents, avatarEvents];
  try {
    const [li, st] = await Promise.all([companionStatus(), avatarStatus()]);
    if (li) setLink(afterAsking<LinkStatus | null>(li, l, linkEvents, link()));
    if (st) setAvatar(afterAsking<AvatarStatus | null>(st, a, avatarEvents, avatar()));
  } catch {
    // No answer: the next event brings it.
  }
}

const on = (type: "focus" | "blur") => (cb: () => void) => {
  window.addEventListener(type, cb);
  return () => window.removeEventListener(type, cb);
};

async function main(): Promise<void> {
  // The browser's own menu (Print, Save as) has no place here.
  document.addEventListener("contextmenu", (e) => e.preventDefault());
  void watchTheme(document.documentElement, () => {});
  await listen<unknown>(COMPANION_STATUS_EVENT, (e) => {
    const s = parseCompanionStatus(e.payload);
    if (!s) return;
    linkEvents += 1;
    setLink(s);
  });
  await listen<unknown>(STATUS_EVENT, (e) => {
    const s = parseStatus(e.payload);
    if (!s) return;
    avatarEvents += 1;
    setAvatar(s);
  });
  await listen<unknown>(WAITING_EVENT, (e) => {
    const w = parseWaiting(e.payload);
    if (w !== null) setWaiting(w);
  });
  window.addEventListener("focus", () => void ask());
  void ask();
  const root = document.getElementById("root");
  if (root) {
    render(
      () => (
        <Prompt
          status={() => status(link(), avatar())}
          say={say}
          close={close}
          onWindowFocus={on("focus")}
          onWindowBlur={on("blur")}
          waiting={waiting}
        />
      ),
      root,
    );
    // Mounted and listening: only now may the hotkey open it (a page that
    // failed to load never says so, and the prompt never shows empty).
    ready().catch((e) => console.warn("prompt: couldn't tell the shell it is ready", e));
  }
}

void main();
