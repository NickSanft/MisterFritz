# Handoff: Mister Fritz — Web Chat Restyle ("the butler restyle")

## Overview
A dark-academia redesign of the Mister Fritz web chat surface (`admin_panel.py` + `admin_templates/chat.html`). It implements the visual direction for plan item 07 (web-chat-redesign) and visually anticipates the streaming UX from item 03 (token-streaming): streamed replies with a status line and Stop control, syntax-highlighted code blocks with copy buttons, image attachments, inline notices replacing `alert()`, and a styled New-conversation confirm dialog.

Vibe: **dark academia base** (serif type, candle-lit purple, ornate butler copy) + **crystalline accents** (faceted clip-path geometry everywhere — no rounded corners) + **sparing glitch** (scan-line overlay, occasional wordmark distortion).

## About the Design Files
The files in this bundle are **design references created in HTML** — a working prototype showing intended look and behavior, NOT production code to copy directly. The task is to **recreate this design in the Fritz codebase's existing environment**: Jinja templates (`admin_templates/`), the `_theme.html` token include introduced by PR 11, and the vanilla-JS chat client in `chat.html`. No framework should be added; the prototype's React-style state maps onto the existing SSE + DOM-update code.

- `Fritz Chat.dc.html` — the prototype (open in a browser; keep `support.js` beside it). All styles are inline on elements; global CSS (fonts, keyframes, scrollbar, markdown/code classes) is in the `<style>` block at the top.
- `support.js` — prototype runtime only. Ignore for implementation.

## Fidelity
**High-fidelity.** Colors, typography, spacing, and interactions are final. Recreate pixel-perfectly, translating inline styles into `_theme.html` CSS custom properties + classes.

## Screens / Views

### 1. Chat transcript (main)
Full-viewport column: header / (optional notice) / scrollable transcript / composer footer.

- **Page background**: `radial-gradient(ellipse 120% 90% at 50% -20%, #1a1128 0%, #0e0916 48%, #0b0710 100%)`; base text `#d5cbe5`; body font **EB Garamond** 17px, line-height 1.65.
- **Atmosphere layers** (both `pointer-events: none`):
  - Scan lines (z-index above content, `mix-blend-mode: overlay`): `repeating-linear-gradient(0deg, transparent 0 3px, rgba(20,10,34,0.14) 4px)`, animated `background-position 0 → 0 128px` over 9s linear infinite.
  - Candle glow (behind content, top-center, ~720×420px): `radial-gradient(ellipse, rgba(150,105,215,0.16) → transparent 65%)`, opacity flickering 0.5–0.75 over 7s (irregular keyframes).
- **Header** (sticky top, `padding: 14px 26px`, bottom border `1px rgba(147,100,210,0.22)`, bg `linear-gradient(180deg, rgba(24,15,38,0.85), rgba(15,10,24,0.65))` + `backdrop-filter: blur(8px)`):
  - Hexagonal "F" seal, 44px, `clip-path: polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%)`, bg `linear-gradient(135deg, #241638, #120b1e)`, border `rgba(147,100,210,0.45)`, pulsing purple glow shadow (5s).
  - Wordmark "Mister Fritz": **Cormorant Garamond** 600, 26px, letter-spacing 0.06em, `#f0e9fa`. Glitch: every ~8s a few frames of `translate(±2px)` + `clip-path: inset()` slices (see `@keyframes glitchShiftA`).
  - Kicker beside it: JetBrains Mono 10.5px uppercase, ls 0.22em, `#6f5f92` — "est. MMXXIV — the household model".
  - Presence row: 7px diamond (clip-path rotated square) `#7fc98f` idle / `#d8b45a` streaming, with matching glow; italic 13.5px `#9d8bc0` label — "In attendance · the candles are lit" / "Occupied — composing a reply".
  - "✦ New conversation" button: transparent, border `rgba(147,100,210,0.4)`, Cormorant 16px `#cdb9ea`, parallelogram `clip-path: polygon(10px 0%, 100% 0%, calc(100% - 10px) 100%, 0% 100%)`. Hover: bg `rgba(147,100,210,0.14)`, brighter border/text.
- **Transcript**: scrollable, `padding: 30px 26px`; inner column `max-width: 860px`, `gap: 26px`. Custom scrollbar: 10px, thumb `#2c2140` on `#0b0710`.
- **Fritz message row**: left-aligned, 34px hexagon "F" avatar + bubble.
  - Bubble: max-width 78%, `padding: 14px 18px 15px`, bg `linear-gradient(160deg, rgba(32,20,52,0.75), rgba(19,12,32,0.75))`, border `1px rgba(147,100,210,0.3)` with **2px left border** `rgba(180,140,235,0.55)`, faceted corners: `clip-path: polygon(14px 0%, 100% 0%, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0% 100%, 0% 14px)` (top-left and bottom-right corners cut).
  - Speaker line: "MISTER FRITZ" Cormorant 600 15px uppercase ls 0.12em `#b48ceb` + timestamp JetBrains Mono 10px `#574a75`.
  - Body text `#d5cbe5`; `<em>` `#cdb9ea`, `<strong>` `#f0e9fa`, inline `<code>` JetBrains Mono 0.82em on `rgba(147,100,210,0.12)` with border `rgba(147,100,210,0.22)`, text `#d8c7f0`.
- **User message row**: right-aligned, no avatar. Bubble max-width 72%, bg `rgba(147,100,210,0.09)`, border `rgba(147,100,210,0.24)`, mirrored facet clip-path (top-right and bottom-left cut), speaker "YOU" `#8d7fa8`, text `#c4b8d8`.
- **Code block** (inside Fritz bubble, `margin-top: 12px`): border `rgba(147,100,210,0.32)`, bg near-black vertical gradient.
  - Title bar: `padding: 7px 12px`, bottom border, bg `rgba(147,100,210,0.07)`; left: 6px purple diamond + language label (JetBrains Mono 10.5px uppercase ls 0.18em `#8d7fa8`, e.g. "python — tests/conftest.py"); right: copy button (bordered, JetBrains Mono 10.5px, "⧉ Copy" → "✓ Transcribed" for 2.2s after click).
  - Code: JetBrains Mono 13px / 1.6, base `#c9bce0`, `padding: 14px 16px`, horizontal scroll.
  - Pygments/codehilite token colors: keyword `#c792ea`, string `#b8a1e3`, function/name `#e0cff5`, comment `#64578a` italic, number `#d4b8f0`, operator `#9d8bc0`.
- **Image attachment (in a sent message)**: framed card max-width 340px, `padding: 8px`, bg `rgba(11,7,16,0.6)`, border `rgba(147,100,210,0.3)`; image area 150px tall; caption row: filename (JetBrains Mono 11px `#9d8bc0`) + meta "1440×900 · PNG" (10px `#574a75`).
- **Status line** (while Fritz works, indented 46px to align with bubbles): parallelogram chip, border `rgba(147,100,210,0.3)`, bg `rgba(147,100,210,0.06)`, italic 14.5px `#9d8bc0`; a soft prism highlight sweeps across it (2.2s loop). Text: "Fritz is consulting the archives…" → "Fritz is taking pen to paper…".
  - **Stop button** beside it, labeled **"Enough"** with a small square stop glyph: rose tint — bg `rgba(178,82,102,0.08)`, border `rgba(198,106,126,0.45)`, text `#d89aac`, Cormorant 15px, parallelogram clip.
- **Streaming caret**: 9×18px vertical bar at the end of the streaming message, gradient `#b48ceb → #7a4fc0`, blinking ~1.1s.
- **Inline notice bar** (below header, replaces `alert()`): `padding: 10px 16px`, 3px left border, italic message + × dismiss. Info: bg `rgba(147,100,210,0.08)`, border `rgba(147,100,210,0.4)`, glyph "❦" `#b48ceb`. Error: bg `rgba(178,82,102,0.1)`, border `rgba(198,106,126,0.5)`, glyph "✕" `#d89aac`. Slides in 6px from above, 0.3s.
- **Composer footer** (sticky bottom, top border, bg gradient + blur like header):
  - Attach button: 46px hexagon, "✧" glyph, purple-tint hover.
  - Textarea: flex-1, min-height 46px, max 150px, bg `rgba(11,7,16,0.75)`, border `rgba(147,100,210,0.35)`, EB Garamond **16.5px** (≥16px prevents iOS zoom), placeholder "Address the butler…". Focus: brighter border + `box-shadow: 0 0 0 1px rgba(180,140,235,0.35), 0 0 22px rgba(147,100,210,0.15)`.
  - Send button "Dispatch ❖": 46px tall, `padding: 0 22px`, bg `linear-gradient(135deg, #4b2f78, #33205a)`, border `rgba(180,140,235,0.55)`, Cormorant 600 17px ls 0.12em `#eee6f9`, parallelogram clip (12px). Hover: lighter gradient + purple glow.
  - Attachment chip (above input when staged): bordered chip with gem-shaped pentagon swatch, filename in mono, × to remove.
  - Hint row below: JetBrains Mono 10px ls 0.14em `#574a75` — left "ENTER TO DISPATCH · SHIFT+ENTER FOR A NEW LINE", right "CANDLE NO. 7 STILL BURNING" (flavor; optional).

### 2. Empty / new-conversation state
Centered in the transcript area: 92px hexagonal "F" seal (pulsing), headline "The study is lit. The ledger is open." (Cormorant 600 34px `#f0e9fa`), italic subcopy (max-width 520px, 17px `#9d8bc0`), then three suggestion chips (italic EB Garamond 15.5px, parallelogram clip, purple-tint hover) that prefill the composer:
- "What is dirtying my git status?"
- "Run ruff and report back."
- "Are my secrets safe in exec?"

### 3. New-conversation confirm dialog
Full-screen veil `rgba(6,4,10,0.72)` + `backdrop-filter: blur(4px)`. Card: min(480px, vw−48px), `padding: 34px 36px 30px`, bg `linear-gradient(160deg, #1c1230, #100a1b)`, border `rgba(147,100,210,0.5)`, large faceted clip-path (24px cuts, top-left + bottom-right). Rose diamond "!" badge; title "Burn the correspondence?" (Cormorant 600 26px); body italic `#9d8bc0`: "The present conversation shall be committed to the fire — every word, every confession. The ashes cannot be reassembled, sir. Fritz has tried."
Buttons right-aligned: "Spare it" (ghost) / "Into the fire" (rose gradient `#7c3350 → #5a2138`, border `rgba(198,106,126,0.6)`). Veil click and "Spare it" cancel; card click must not propagate. **Wire via `addEventListener` above any early-return guard** (the current inline `onsubmit` has the SyntaxError this replaces — plan item 07 / PR 6).

## Interactions & Behavior
- **Send**: Enter dispatches; Shift+Enter newline. Empty draft (no attachment) is a no-op. Sending while Fritz streams is blocked with an error notice: "One moment, sir — Fritz is mid-sentence. Interrupt him with "Enough" if you must."
- **Streaming lifecycle**: send → status chip ("consulting the archives…", swaps to "taking pen to paper…" after ~1.4s) → Fritz bubble appears and fills token-by-token with blinking caret → on completion the code block (if any) attaches, caret and status clear. Maps onto the `applyToken(delta, accumulated, restart)` seam from PR 12/16.
- **Stop ("Enough")**: halts the stream, appends " —" (em-dash) to the truncated reply, clears status, shows info notice: "Very well — Fritz has set down his pen mid-sentence. The model, regrettably, mutters on in the servants' quarters." (Honest labeling: server keeps generating — PR 12 note.)
- **Copy code**: writes plain code text to clipboard (with non-secure-context fallback per PR 13); label → "✓ Transcribed" for 2.2s.
- **Attach (✧)**: stages one image chip; × unstages. Sent image renders as the framed card. (Prototype fakes the file; real impl uses the existing upload endpoint + Pillow sniffing from PR 8.)
- **New conversation**: opens confirm dialog; confirming clears the thread, cancels any stream, shows info notice "The correspondence has been burned. A fresh ledger awaits, sir." (auto-dismisses ~4.5s), and reveals the empty state.
- **Auto-scroll**: smooth scroll-to-bottom on each new message/token — but implement pinned-to-bottom detection (`isPinnedToBottom()`/`followScroll()` per PR 12) so a user scrolled up isn't yanked down. **Do not use `scrollIntoView`**; set `scrollTop`/`scrollTo` on the container.
- **Message entrance**: fade + 10px rise, 0.35s ease. Honor `prefers-reduced-motion` (disable scan drift, glitch, sweeps, entrances).
- **Notices**: single slot below header; info vs error styling; × dismisses; some auto-dismiss (timings above).

## State Management
- `messages: [{id, role: 'user'|'fritz', text, time, image?, code?, streaming?}]` — server-rendered history + SSE appends.
- `draft` (textarea), `attachment` (staged upload or null).
- `streaming: bool` — drives presence dot/label, send-blocking, status chip, Stop visibility.
- `statusLine: string|null` — progress callback text.
- `notice: {kind: 'info'|'error', text}|null`.
- `confirmOpen: bool`.
- `copiedId` transient for copy-button feedback.
- Data flow: POST message → SSE stream (delta frames per PR 16, or cumulative pre-PR-16 — `applyToken()` isolates the difference) → done frame carries sanitized rendered HTML (`_sanitise_html`, class attrs preserved for codehilite).

## Design Tokens
Suggested `_theme.html` custom properties:
- **Colors — ground**: `--bg-void #0b0710`, `--bg-deep #0e0916`, `--bg-raised #1a1128`, `--panel-hi rgba(24,15,38,0.85)`, `--panel-lo rgba(15,10,24,0.65)`.
- **Purples**: `--amethyst #9364d2` (core accent, borders at 0.22–0.5 alpha), `--amethyst-bright #b48ceb`, `--lilac #cdb9ea`, `--lavender-ink #d8c7f0`, `--text-hi #f0e9fa`, `--text-body #d5cbe5`, `--text-dim #9d8bc0`, `--text-faint #8d7fa8`, `--text-ghost #6f5f92`, `--text-trace #574a75`.
- **States**: rose (danger) `#d89aac` text / `rgba(198,106,126,0.45–0.6)` border / `#7c3350→#5a2138` gradient; presence green `#7fc98f`; busy amber `#d8b45a`.
- **Send gradient**: `#4b2f78 → #33205a` (hover `#5d3b96 → #41296f`).
- **Type**: Cormorant Garamond (display/buttons; 600–700), EB Garamond (body/prose; 400 + italic), JetBrains Mono (code/meta). Scale: 34/26/17/16.5/15/13/11/10.5/10px. Meta text uppercase with wide tracking (0.1–0.22em).
- **Spacing**: 4/8/12/16/26px rhythm; transcript column max-width 860px; header/footer x-padding 26px.
- **Geometry**: **no border-radius anywhere** — facets via `clip-path`: hexagon (avatars/attach), parallelogram (buttons/chips, 8–12px skew cut), corner-cut rectangles (bubbles 14px, dialog 24px), diamond (bullets/badges).
- **Motion**: hover transitions 0.2–0.25s ease; entrances 0.3–0.35s; candle flicker 7s; scan drift 9s; seal pulse 5s; glitch cycle 8s (a ~0.5s burst); prism sweep 2.2s; caret blink 1.1s.

## Assets
No image assets. Fonts from Google Fonts: Cormorant Garamond, EB Garamond, JetBrains Mono. Glyphs are Unicode characters (✦ ✧ ❖ ❦ ⧉ ✕ ×) — no icon font.

## Files
- `Fritz Chat.dc.html` — full prototype: inline-styled markup + a `Component` class (bottom `<script>`) holding all interaction logic (streaming simulation, stop, confirm, notices, copy, attach). The canned reply logic (`composeReply`) is demo-only.
- `support.js` — prototype runtime; not for implementation.

## Codebase mapping (from the implementation plan)
- Tokens → `_theme.html` (PR 11); transcript/composer restyle + streaming UX → `chat.html` (PR 12); codehilite colors (PR 13) — requires the nh3 sanitizer allowing `class` on `div/pre/code/span` (PR 5) or highlighting is silently stripped; confirm-dialog fix + viewport meta + a11y (`role="log"`, `#chat-status`, `aria-busy`) → PR 6. Keep the "Type a message" placeholder swap in mind: tests assert on it — update them with the copy change ("Address the butler…").
