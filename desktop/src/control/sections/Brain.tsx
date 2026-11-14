/**
 * Settings › Brain: where his brain is, the token he connects with, whether
 * he is connected, and a question to try him with. The avatar's page holds
 * the connection; this window only hears of it (`companion:status`) and is
 * never given the token: what is typed is sent to the shell once, then
 * cleared. An Ask goes the same way, through the shell to his page, and the
 * reply shows in his bubble, not here.
 *
 * What it says comes from view.ts (`connectionLines`, `identityLine`).
 */
import { createSignal, Show, type JSX } from "solid-js";
import type { AvatarStatus } from "../../avatar/status";
import type { LinkStatus } from "../../shared/companion";
import { Sigil } from "../../divora/sigils/Sigil";
import { Row, Section } from "../layout";
import { connectionLines, identityLine, onThisPc } from "../view";

export interface BrainProps {
  /** `brain.url` in the settings; null before they are read. */
  url: string | null;
  link: LinkStatus | null;
  avatar: AvatarStatus | null;
  /** null until the shell has said. */
  tokenPresent: boolean | null;
  /** Set `brain.url`; whether the shell took it (if not, it says why). */
  setUrl(url: string): Promise<boolean>;
  /** Save a pasted token; whether the shell took it. */
  saveToken(token: string): Promise<boolean>;
  /** Connect again with the saved token. */
  reconnect(): void;
  /** Ask him something; whether the shell took it (if not, it says why). */
  ask(text: string): Promise<boolean>;
}

export function Brain(props: BrainProps): JSX.Element {
  const [urlDraft, setUrlDraft] = createSignal<string | null>(null);
  const [token, setToken] = createSignal("");
  const [saving, setSaving] = createSignal(false);
  const [question, setQuestion] = createSignal("");
  const [asking, setAsking] = createSignal(false);
  const [asked, setAsked] = createSignal(false);

  const url = () => props.url ?? "";
  const lines = () => connectionLines(props.link, props.avatar, props.tokenPresent, url());
  const who = () => identityLine(props.link);
  const here = () => onThisPc(url());

  const commitUrl = async () => {
    const draft = urlDraft();
    if (draft === null) return;
    if (draft === url()) {
      setUrlDraft(null);
      return;
    }
    // Refused, the draft stays, so it can be mended; the shell says why.
    if (await props.setUrl(draft)) setUrlDraft(null);
  };

  const pasted = () => token().trim() !== "";
  const connect = async () => {
    if (saving()) return;
    if (!pasted()) {
      if (props.tokenPresent) props.reconnect();
      return;
    }
    setSaving(true);
    try {
      // Taken or not, the field is cleared: the token isn't kept here.
      await props.saveToken(token());
    } finally {
      setToken("");
      setSaving(false);
    }
  };

  const online = () => props.link?.link === "online";
  const askable = () => online() && question().trim() !== "" && !asking();
  const ask = async () => {
    if (!askable()) return;
    setAsking(true);
    try {
      // Taken, the field clears and says where the answer is; refused, it
      // stays, and the shell says why.
      if (await props.ask(question())) {
        setQuestion("");
        setAsked(true);
      }
    } finally {
      setAsking(false);
    }
  };
  const askDesc = () =>
    asked()
      ? "He's answering on your desktop."
      : online()
        ? "Try him: his answer shows over his head."
        : "Once he's connected, you can try him from here.";

  return (
    <Section
      id="brain"
      sigil="link"
      eyebrow="Brain"
      title="Where his mind is"
      desc="His brain does the thinking, on this PC or another; this desk connects to it."
    >
      <Row
        title="Where his brain is"
        desc={props.url === null ? undefined : here() ? "On this PC." : "On another machine."}
      >
        <div class="row-line">
          <Show when={props.url !== null && here()}>
            <span class="place" title="His brain is on this PC">
              <Sigil name="shield" size={13} /> On-device
            </span>
          </Show>
          <input
            type="text"
            class="field mono brain-url"
            value={urlDraft() ?? url()}
            aria-label="His brain's address"
            spellcheck={false}
            autocomplete="off"
            disabled={props.url === null}
            onInput={(e) => setUrlDraft(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void commitUrl();
              } else if (e.key === "Escape") {
                setUrlDraft(null);
              }
            }}
            onBlur={() => void commitUrl()}
          />
        </div>
      </Row>
      <Row
        title="His token"
        desc={
          props.tokenPresent
            ? "Saved. Paste a new one to replace it, or reconnect with this one."
            : "COMPANION_TOKEN, from the brain's .env. It is kept beside his settings, never shown again."
        }
      >
        <div class="row-line">
          <input
            type="password"
            class="field mono"
            value={token()}
            placeholder={props.tokenPresent ? "Saved" : "Paste the token"}
            aria-label="His token"
            spellcheck={false}
            autocomplete="off"
            onInput={(e) => setToken(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void connect();
              }
            }}
          />
          <button
            type="button"
            class="btn btn-secondary"
            disabled={saving() || (!pasted() && props.tokenPresent !== true)}
            onClick={() => void connect()}
          >
            {!pasted() && props.tokenPresent ? "Reconnect" : "Connect"}
          </button>
        </div>
      </Row>
      <Row title={lines().title} desc={lines().detail || undefined}>
        <span classList={{ dot: true, [`tone-${lines().tone}`]: true }} />
      </Row>
      <Show when={who()}>
        <Row title="Who he is to it" desc={who()!} />
      </Show>
      <Row title="Ask him something" desc={askDesc()}>
        <div class="row-line">
          <input
            type="text"
            class="field brain-ask"
            value={question()}
            placeholder="Good evening"
            aria-label="Ask him something"
            spellcheck={true}
            autocomplete="off"
            disabled={!online()}
            onInput={(e) => {
              setQuestion(e.currentTarget.value);
              setAsked(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void ask();
              }
            }}
          />
          <button type="button" class="btn btn-secondary" disabled={!askable()} onClick={() => void ask()}>
            Ask
          </button>
        </div>
      </Row>
    </Section>
  );
}
