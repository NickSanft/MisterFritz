/**
 * Settings › Appearance: the theme every window wears (plan 14 build § 4.8,
 * DivoraVoice's own Appearance rows). Each pick is one `set_setting`; the
 * shell tells every window, the tray included, so this section shows what
 * the settings say rather than what was clicked.
 */
import type { JSX } from "solid-js";
import { Segmented, type SegmentedOption } from "../../divora/components/Segmented";
import { Toggle } from "../../divora/components/Toggle";
import { MOTION_LEVELS, motionLevel, type MotionLevel } from "../../divora/theme";
import type { Theme } from "../../shared/settings";
import { Row, Section } from "../layout";

const THEMES: SegmentedOption<Theme["theme"]>[] = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];
const MOODS: SegmentedOption<Theme["mood"]>[] = [
  { value: "ink", label: "Ink + candle" },
  { value: "dusk", label: "Dusk violet" },
  { value: "midnight", label: "Midnight" },
];
const ACCENTS: SegmentedOption<Theme["accent"]>[] = [
  { value: "brand", label: "Brand" },
  { value: "abyssal", label: "Abyssal" },
  { value: "ember", label: "Ember" },
];
const MOTIONS: SegmentedOption<MotionLevel>[] = [
  { value: "functional", label: "Functional" },
  { value: "ambient", label: "Ambient" },
  { value: "rich", label: "Rich" },
];

/** What the motion row says: what the level does, or that Windows has him
 * keeping still whatever it is. */
export function motionLine(t: Theme | null, lessMotion: boolean): string {
  if (t && lessMotion && !t.motionUser) {
    return "Windows is asking for less motion, so for now he only blinks and breathes.";
  }
  switch (motionLevel(t?.motion ?? MOTION_LEVELS.ambient)) {
    case "functional":
      return "He only blinks and breathes, and nothing here animates.";
    case "ambient":
      return "He sways a little.";
    case "rich":
      return "He sways freely.";
  }
}

export interface AppearanceProps {
  /** null until the settings have loaded: every control waits. */
  theme: Theme | null;
  /** Windows is asking apps for less motion (its Animation effects off). */
  lessMotion: boolean;
  set: (path: string, value: unknown) => void;
}

export function Appearance(props: AppearanceProps): JSX.Element {
  const t = () => props.theme;
  const off = () => !t();
  return (
    <Section
      id="appearance"
      sigil="eye"
      eyebrow="Appearance"
      title="How the house looks"
      desc="He and this window both wear these, and change as soon as you pick."
    >
      <Row title="Theme" desc="Light, for a bright room.">
        <Segmented
          ariaLabel="Theme"
          options={THEMES}
          value={t()?.theme ?? "dark"}
          disabled={off()}
          onChange={(v) => props.set("theme.theme", v)}
        />
      </Row>
      <Row title="Colour mood" desc="Ink + candle is his own.">
        <Segmented
          ariaLabel="Colour mood"
          options={MOODS}
          value={t()?.mood ?? "ink"}
          disabled={off()}
          onChange={(v) => props.set("theme.mood", v)}
        />
      </Row>
      <Row title="Accent" desc="His rim light, and this window's highlights.">
        <Segmented
          ariaLabel="Accent"
          accent
          options={ACCENTS}
          value={t()?.accent ?? "brand"}
          disabled={off()}
          onChange={(v) => props.set("theme.accent", v)}
        />
      </Row>
      <Row title="Motion" desc={motionLine(t(), props.lessMotion)}>
        <Segmented
          ariaLabel="Motion"
          options={MOTIONS}
          value={motionLevel(t()?.motion ?? MOTION_LEVELS.ambient)}
          disabled={off()}
          onChange={(level) => props.set("theme.motion", MOTION_LEVELS[level])}
        />
      </Row>
      <Row
        title="Keep his motion when Windows asks for less"
        desc={props.lessMotion
          ? "Windows is asking for less motion right now."
          : "For when Windows' Animation effects are turned off."}
      >
        <Toggle
          ariaLabel="Keep his motion when Windows asks for less"
          on={t()?.motionUser ?? false}
          disabled={off()}
          onChange={(on) => props.set("theme.motionUser", on)}
        />
      </Row>
      <Row title="Parchment grain" desc="A paper texture over this window, never over him.">
        <Toggle
          ariaLabel="Parchment grain"
          on={t()?.grain ?? false}
          disabled={off()}
          onChange={(on) => props.set("theme.grain", on)}
        />
      </Row>
    </Section>
  );
}
