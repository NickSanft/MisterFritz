/**
 * The control window's page shape, DivoraVoice's Settings: a section is a
 * sigil, an eyebrow, a title and a line, then a panel of rows, each a label
 * (and a line saying more) with its control on the right.
 */
import { Show, type JSX } from "solid-js";
import { Sigil, type SigilName } from "../divora/sigils/Sigil";
import type { SectionId } from "./api";

export function Row(props: {
  title: string;
  desc?: string;
  /** Said in place of `desc`, as an alert: what went wrong on this row. */
  alert?: string | null;
  children?: JSX.Element;
}): JSX.Element {
  return (
    <div class="row">
      <div class="row-text">
        <div class="row-title">{props.title}</div>
        <Show
          when={props.alert}
          fallback={
            <Show when={props.desc}>
              <div class="row-desc">{props.desc}</div>
            </Show>
          }
        >
          {(said) => (
            <div class="row-desc row-alert" role="alert">
              <Sigil name="warning" size={13} class="row-alert-sigil" />
              {said()}
            </div>
          )}
        </Show>
      </div>
      <div class="row-control">{props.children}</div>
    </div>
  );
}

export function Section(props: {
  id: SectionId;
  sigil?: SigilName;
  eyebrow: string;
  title: string;
  desc: string;
  children: JSX.Element;
}): JSX.Element {
  return (
    <section class="section" id={props.id}>
      <div class="section-head">
        <Show when={props.sigil}>
          {(name) => <Sigil name={name()} size={19} class="section-sigil" />}
        </Show>
        <div>
          <div class="eyebrow">{props.eyebrow}</div>
          <h2>{props.title}</h2>
        </div>
      </div>
      <p class="section-desc">{props.desc}</p>
      <div class="panel">{props.children}</div>
    </section>
  );
}
