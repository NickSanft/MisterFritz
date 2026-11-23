/**
 * The muted sign over his head is DivoraVoice's `muted` sigil, drawn as DOM
 * on his page (which has no Solid): mark for mark the Sigil's own.
 */
import { describe, expect, it } from "vitest";
import { render } from "@solidjs/testing-library";
import { Sigil } from "../divora/sigils/Sigil";
import { mutedSigil } from "./BubbleView";

describe("the muted sign's sigil", () => {
  it("is the Sigil set's muted, mark for mark", () => {
    const { container } = render(() => <Sigil name="muted" size={16} />);
    const theirs = container.querySelector("svg")!;
    const ours = mutedSigil(document, 16);
    expect(ours.querySelector("g")!.outerHTML).toBe(theirs.querySelector("g")!.outerHTML);
    for (const attr of ["width", "height", "viewBox", "aria-hidden"]) {
      expect(ours.getAttribute(attr), attr).toBe(theirs.getAttribute(attr));
    }
    expect(ours.namespaceURI).toBe("http://www.w3.org/2000/svg");
  });
});
