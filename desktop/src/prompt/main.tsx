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
import { avatarStatus, close, companionStatus, say } from "./api";
import { Prompt } from "./Prompt";
import { status } from "./view";

const [link, setLink] = createSignal<LinkStatus | null>(null);
const [avatar, setAvatar] = createSignal<AvatarStatus | null>(null);
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
        />
      ),
      root,
    );
  }
}

void main();
