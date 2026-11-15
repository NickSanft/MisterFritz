/**
 * The Model rows: which model he wears, its folder, and loading it again.
 * Settings › Model shows them, and so does the first run's "His look".
 */
import type { JSX } from "solid-js";
import type { ModelInfo } from "../api";
import { Row } from "../layout";
import { modelLines } from "../view";

export interface ModelRowsProps {
  /** null until the shell has said. */
  model: ModelInfo | null;
  openFolder(): void;
  reload(): void;
}

export function ModelRows(props: ModelRowsProps): JSX.Element {
  const lines = () => modelLines(props.model);
  return (
    <>
      <Row title={lines().title} desc={lines().detail}>
        <span classList={{ dot: true, [`tone-${lines().tone}`]: true }} />
      </Row>
      <Row title="His model's folder" desc="Where avatar\fritz.vrm goes, unless avatar.vrmPath points elsewhere.">
        <button type="button" class="btn btn-secondary" onClick={() => props.openFolder()}>
          Open
        </button>
      </Row>
      <Row title="Load it again" desc="After you've put a new export in place.">
        <button type="button" class="btn btn-secondary" onClick={() => props.reload()}>
          Load
        </button>
      </Row>
    </>
  );
}
