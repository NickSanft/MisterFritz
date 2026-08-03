# 7. Web chat — the dark-academia restyle

[← back to index](README.md) · [decisions](DECISIONS.md)

**Effort:** L → **XL (>3 days)** with the mockup incorporated
**Depends on:** PR 5 (nh3 sanitiser — **hard**, see §0.1), PR 8 (image sniffing, for the card meta), PR 12/16 (the `applyToken` seam)
**Design source:** `NewMockup/` — `Fritz Chat.dc.html` (prototype), `README.md` (handoff)

> **Rewritten 2026-08-03** to incorporate the Claude Design mockup. The visual direction changed completely: this plan previously specified a **light** parchment/mahogany/brass palette with dark mode as a `prefers-color-scheme` variant. The mockup is **dark-only dark academia** — candle-lit purple, serif type, zero border-radius, clip-path facets throughout. The correctness and accessibility work survives intact (§3); the palette work is replaced.

---

## 0. Blockers resolved before implementation

Four things would have failed silently or shipped a regression.

### 0.1 The sanitiser strips the syntax highlighting — plan 05 must change first

**The one that fails silently.** Plan 05 calls `nh3.clean(raw_html)` with no `attributes=` and explicitly blesses losing `class`. Verified empirically: that strips `class="codehilite"` and every `class="k"` / `class="nb"` from Pygments output. Every code block renders flat. No error, no failing test.

Plan 05 must gain a shared helper before either item lands:

```python
_NH3_ATTRS = dict(nh3.ALLOWED_ATTRIBUTES)           # dict() copy is load-bearing —
for _t in ("div", "pre", "code", "span", "table"):   # in-place mutation poisons the constant
    _NH3_ATTRS[_t] = (_NH3_ATTRS.get(_t) or set()) | {"class"}   # `or set()` — default is None

def _sanitise_html(html: str) -> str:
    return nh3.clean(html, attributes=_NH3_ATTRS)
```

Verified against nh3 0.3.6 this session. **PR 5 before PR 13, no exceptions.** Ship the guard tests with it or this dies the next time either plan is edited:

```python
def test_codehilite_classes_survive_sanitiser(self):
    out = _render_markdown("```python\nprint(1)\n```")
    self.assertIn('class="codehilite"', out)
    self.assertIn('class="k"', out)

def test_onclick_stripped_from_classed_span(self):
    self.assertNotIn("onclick", _sanitise_html('<span class="x" onclick="evil()">t</span>'))
```

### 0.2 The CSP blocks every font — and `script-src 'none'` is not survivable

Plan 05's CSP is `default-src 'none'` with no `font-src`, which blocks `@font-face` whether self-hosted or CDN. And `script-src 'none'` kills the chat client outright — `chat.html` is one large inline script.

**Ship exactly:**

```
default-src 'none'; script-src 'nonce-{nonce}'; style-src 'self' 'unsafe-inline';
font-src 'self'; img-src 'self' data:; connect-src 'self'; form-action 'self';
base-uri 'none'; frame-ancestors 'none'; object-src 'none'
```

Three edits versus plan 05: add `font-src 'self'`, add `'self'` to `style-src`, add `object-src 'none'`.

Tightening path for later, in order: (1) move the chat script to `admin_static/chat.js`, then `script-src 'self'` and delete the nonce plumbing; (2) move `_theme`'s `<style>` and the remaining `style=""` attributes into `admin_static/theme.css`, then drop `'unsafe-inline'`.

**Never write `style-src 'nonce-x' 'unsafe-inline'`** — the nonce makes `'unsafe-inline'` be ignored and every `style=""` attribute dies.

**Forbidden in the port:** translating the prototype's `onClick="{{ handler }}"` (12 sites) into `onclick=""` attributes — dead under this CSP. And no Jinja interpolation into `style=""` — that locks `'unsafe-inline'` in forever and opens a CSS-injection sink.

### 0.3 clip-path eats focus rings — eight controls would have none

`clip-path` clips `outline` and outer `box-shadow`. The previous focus rule would leave the seal, presence dot, New-conversation button, attach button, send button, stop button, suggestion chips and dialog buttons with **zero visible focus indicator** — WCAG 2.4.7 failure, and a regression against today's chat.

**Fix:** wrap each faceted control in an unclipped `<span class="facet">` carrying `:focus-within { outline: 2px solid var(--focus); outline-offset: 2px }`. A rectangular ring around a faceted shape is what real design systems do. Ring colour `#c9a75f` (8.1-8.6:1 on every dark ground).

Same root cause: `sealPulse` and the presence-dot glow are **invisible today** in the prototype because their `box-shadow` is clipped — see §0.5.

Also raise the textarea focus shadow from `rgba(180,140,235,0.35)` (1.90:1) to solid `#b48ceb`; the indicator itself must clear 3:1 per SC 1.4.11.

### 0.4 Three palette tokens fail AA — worse than what this redesign exists to fix

Measured against **composited** backgrounds (bubble = 0.75 alpha over the page gradient = `#1c112c`; code = 0.92 alpha over the bubble), not the raw declared hex. Measuring against `#0b0710` flatters every number and is how a second failing palette gets shipped.

| Token | Used for | Ratio | Verdict |
|---|---|---|---|
| `#574a75` | 10px timestamps, hint row, image meta | **2.26:1** | fails even the 3:1 non-text floor |
| `#6f5f92` | 10.5px kicker, × dismiss buttons | **3.28:1** | fails |
| `#64578a` | 13px italic code comments | **3.11:1** | fails |
| `#8d7fa8` | cite, code language label, user speaker | 4.31-4.97:1 | borderline; fails under candle-glow peak |

Today's `--muted` fails at 3.67:1 and fixing it was a HIGH-impact audit finding. Shipping `#574a75` at 2.26:1 is a regression dressed as a fix.

**Resolution:** collapse `#574a75`, `#6f5f92` and `#8d7fa8` into one `--text-meta: #9082aa` (4.77:1 worst case), and move the code comment `#64578a` → `#8579ab` (4.80-5.06:1). Keep `--text-dim #9d8bc0`, `--text-body #d5cbe5`, `--text-hi #f0e9fa` exactly as designed — all pass with margin.

Annotate every token line with its measured ratio, as the previous palette table did, so the next person cannot regress it blind.

### 0.5 Two prototype/handoff contradictions — resolved

**Scan lines.** The prototype renders them at `z-index: 6`, *beneath* `<main>` (7) and header/footer (8) — a background texture. The handoff (README:24) claims "above content". Visibly different products.
**Decided: beneath content, as the prototype actually renders.** An above-content `mix-blend-mode: overlay` layer tints text and would undo §0.4. Reverse only if you want the overlay look and will re-measure every ratio through it.

**Seal pulse.** The handoff describes a "pulsing purple glow"; the prototype's `box-shadow` is clipped, so nothing pulses.
**Decided: implement the intended glow via `filter: drop-shadow()`**, which follows the clipped silhouette. Applies to the seal, presence dot, send button and "Into the fire".

**Collapsing prototype drift:** four parallelogram skews (8/9/10/12px) → `--cut-sm: 8px` / `--cut-lg: 12px`; two near-identical seal hexagons → one.

---

## 1. Decisions incorporated

From [DECISIONS.md](DECISIONS.md), answered 2026-08-03:

| Question | Decision |
|---|---|
| Theme scope | **Chat only** — decision 11 overridden, see §2.1 |
| Fonts | **Self-host woff2** (~150-170 KB, six faces) |
| Failing contrast | **Lighten the three tokens** per §0.4 |
| Atmosphere | **Ship on, honour `prefers-reduced-motion`** |
| Timestamps | **Keep** — needs a server-side source |
| Image meta caption | **Keep** — extend PR 8's response |
| Code language label | **Keep** — costs a treeprocessor, see §6.3 |
| Identity / sign-out | **In the presence row** |
| Login page | **Reuse the confirm-dialog card shape** |
| Admin doc upload | **Moves to the admin panel** |

---

## 2. Structural changes

### 2.1 The theme file splits — decision 11 is overridden

One shared token file no longer holds. This palette is built for a conversation; its faceted clip-paths fight the `table { display: block; overflow-x: auto }` mobile rule, and styling data grids in candle-lit purple buys nothing.

```
admin_templates/
  _theme_base.html    ← reset, responsive, a11y (focus, reduced-motion) — both chromes
  _theme_chat.html    ← dark-academia palette + geometry + motion
  _theme_admin.html   ← admin palette (today's look; --muted fixed to pass AA)
  chat_base.html      ← NEW chat chrome, no admin nav
  base.html           ← unchanged structurally; consumes _theme_base + _theme_admin
```

The admin panel still gets its contrast fix — that was a real audit finding and survives the scope change. It does not get the purple.

### 2.2 `chat_base.html` is mandatory, not optional

Previously step 6 and marked deferrable. Now a hard prerequisite: the mockup shell is `position: fixed; inset: 0; display: flex; flex-direction: column; overflow: hidden`, which **cannot exist** inside `base.html`'s `main { padding: 2rem; max-width: 1100px; margin: 0 auto; }` (base.html:32).

This also flips the follow-scroll implementation. The previous helpers used `document.documentElement.scrollHeight` / `window.scrollY` / `window.scrollTo` — all wrong for a container-scrolled shell. Read and set `scrollTop` on the transcript container. **Do not use `scrollIntoView`.**

### 2.3 A `/static` mount is new

Self-hosting fonts needs `Mount("/static", StaticFiles(directory="admin_static"))`, which does not exist in `admin_panel.py`. It interacts with `_BasicAuthMiddleware`'s path exemption (admin_panel.py:77), which exempts only `/chat` and `/chat/*`.

**Extend the exemption to `/static`**, or `/chat/login` renders unstyled behind an admin-password prompt. Static assets are public — they are fonts and CSS, and gating them buys nothing while breaking the login page.

This contradicts the previous plan's "no new ports, secrets, or routes" claim, which is now false and removed.

---

## 3. Verified current state that still holds

All verified against the working tree; unchanged by the new design direction.

**Mobile.** `base.html:3-5` has no viewport meta; `grep -rn "@media\|viewport" admin_templates/` returns zero matches across all eight templates. Two faults beyond the audit's: `chat.html:58` sets the textarea to `0.95rem` (15.2px) and `chat_login.html:25` the same — iOS force-zooms under 16px. And `chat.html:396` calls `textarea.focus()` in `.finally`, re-popping the soft keyboard after every send.

**Broken confirm — verified empirically, not by inspection.** `chat.html:105-106` carries `onsubmit="return confirm('...only this thread\\'s context is reset.');"`. `cat -A` confirms two literal backslashes; Jinja passes them through; HTML attribute parsing does not process backslashes. Feeding the exact attribute bytes to `new Function()` under node yields **`SyntaxError - missing ) after argument list`**. A handler that fails to compile is null, so `POST /chat/forget` fires with no confirmation. `chat_forget` (admin_panel.py:636-646) calls `privacy.forget_conversation`, which `DELETE`s from `checkpoints` and `writes` — unrecoverable.

**Contrast — the audit undercounted the blast radius.** `--muted #738291` on `--bg #f7f7f5` = **3.672:1**, on `--card #fff` = **3.939:1**. Consumers the audit missed: `h2 { color: var(--muted) }` (base.html:33) — every section heading on every admin page; `.chat-pending-attach` (chat.html:70); `.chat-doc-upload label` (:89); `.chat-empty` (:95-98); and the inline `color: var(--muted)` on chat_login.html:19. **These are why the admin panel still needs its contrast fix even though it keeps its own palette.**

**Screen readers.** `chat.html:116` is a bare `<div id="chat-messages">` — no `role`, no `aria-live`, no `aria-busy`.

**Streaming wire format.** `admin_panel.py:492` documents `event=token data=<accumulated text so far>`; `mister_fritz.py:433` and `:494` both call `streaming_callback(accumulated_text)`. Token frames are cumulative today, which is why `chat.html:354` assigns rather than appends. `tests/test_admin_panel.py:516` asserts exactly `["Very", "Very well", "Very well, sir."]` — **that assertion is the wire contract**, and it changes when PR 16 flips to deltas.

**Server-side cancellation does not exist.** `chat_stream` spawns a plain daemon thread (admin_panel.py:558); `_event_generator` (:560-577) never checks `request.is_disconnected()`. Aborting the client fetch cannot stop `ask_stuff` — which is why the mockup's honest Stop copy matters.

**Two server-side bugs found while reading.** `chat_send` renders chat.html at admin_panel.py:462-468 and :473-480 without an `is_admin` key, so after a no-JS send the admin upload panel silently vanishes; and the error-path message dicts at :465-466 omit `html`, so an error reply skips the markdown branch. Both become moot when the no-JS path is deleted (decision 3) and upload moves to the admin panel.

---

## 4. What survives from the previous plan

- **Viewport meta** — add to both chromes.
- **Confirm handler** — delete the attribute; register via `addEventListener` **above** the `if (!form || !list) return;` guard at chat.html:168. The handoff independently restates this rule (README:62). The `window.confirm()` half is replaced by the faceted dialog.
- **Enter-to-send — keep this plan's version, not the prototype's.** The prototype (`:409`) is a bare `if (e.key === 'Enter' && !e.shiftKey)`. This plan's handler additionally guards `ev.isComposing || ev.keyCode === 229` (IME) and `matchMedia("(pointer: coarse)")`. Both strictly better.
- **The 16px composer floor** — mockup is 16.5px with the same stated rationale.
- **`applyToken(delta, accumulated, restart)`** — explicitly endorsed by the handoff (README:66, :83). Land the streaming-UX work **before** token-streaming so the seam exists when the wire format flips.
- **AbortController Stop, `alert()` removal, `aria-live`, codehilite with `guess_lang=False`, the `CHAT_CODE_HIGHLIGHT` knob, `Pygments>=2.17`**, and the `.env.example` backfill of `CHAT_COOKIE_SECRET` / `CHAT_IMAGE_UPLOAD_MAX_BYTES` / `CHAT_DOC_UPLOAD_MAX_BYTES` (all three present in `fritz_utils.py`, all undocumented).
- **Unchanged risks:** the `navigator.clipboard` non-secure-context fallback; codehilite colouring prose; theme-move blast radius.

---

## 5. Design system

Full extraction is in `NewMockup/README.md`. What follows is the delta between that and what should actually be written.

### 5.1 Three token layers, not one flat list

The prototype hand-types 41 `rgba()` strings off four base colours.

```css
/* Layer 1 — channel triplets for the colours carrying alpha ladders */
--amethyst-rgb: 147, 100, 210;
--amethyst-bright-rgb: 180, 140, 235;
--rose-rgb: 178, 82, 102;
--rose-border-rgb: 198, 106, 126;

/* Layer 2 — named alpha steps */
--edge-subtle: rgba(var(--amethyst-rgb), .22);
--edge:        rgba(var(--amethyst-rgb), .30);
--edge-strong: rgba(var(--amethyst-rgb), .45);
--fill-ghost:  rgba(var(--amethyst-rgb), .06);
--fill-faint:  rgba(var(--amethyst-rgb), .08);
--fill-soft:   rgba(var(--amethyst-rgb), .12);
--fill-hover:  rgba(var(--amethyst-rgb), .16);

/* Layer 3 — semantic component tokens */
```

### 5.2 Collapse the duplicates first

~45 hex values reduce to ~32 real tokens:

- candle glow `rgba(150,105,215,.16)` → `rgba(var(--amethyst-rgb), .16)`
- prism sweep `rgba(190,150,245,.14)` → `rgba(var(--amethyst-bright-rgb), .14)`
- send-button label `#eee6f9` → `--text-hi #f0e9fa`
- link `#b79ce0` → `--amethyst-bright #b48ceb`, hover `--lavender-ink #d8c7f0`
- seal gradient tails `#120b1e` and `#100a1b` → one `--obsidian`
- footer panel alphas `0.90/0.70` → the header's `0.85/0.65`
- `#9d8bc0` currently serves as **both** `--text-dim` and the code operator colour — emit two tokens that happen to start equal, or tuning one silently moves the other.

### 5.3 Name the clip-paths as tokens

`--clip-seal` (pointy-top hex, 3 uses), `--clip-hex-flat` (attach button — genuinely different geometry, keep separate), `--clip-diamond` (3 uses), `--clip-cut-sm/lg` (parallelograms, collapsed per §0.5), `--clip-facet-l` (Fritz bubble, 14px TL+BR), `--clip-facet-r` (user bubble, mirrored), `--clip-facet-lg` (dialog, 24px), `--clip-gem` (pentagon).

Custom properties hold `clip-path` values fine, and this makes "no border-radius, facets everywhere" enforceable by review.

### 5.4 Fonts — self-hosted, six faces

`admin_static/fonts/`, latin subset, woff2 only, `font-display: swap`:

- Cormorant Garamond 400, 600, 700
- EB Garamond 400, 400 italic, **600** ← the prototype's Google request omits it, but `.fritz-md strong` needs it
- JetBrains Mono 400

The prototype requests 13 faces and uses 6. Dropping Cormorant 500, both Cormorant italics and JetBrains 500/700 takes ~350-400 KB to ~150-170 KB. All three families are SIL OFL 1.1 — ship `OFL.txt` alongside.

Fix the bare fallback stacks:
```css
--font-display: 'Cormorant Garamond', Cormorant, Garamond, 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
--font-body:    'EB Garamond', Garamond, Georgia, 'Times New Roman', serif;
--font-mono:    'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
```

### 5.5 Responsive — must be written from scratch

**The prototype contains zero `@media` queries.** The 860px column, 26px gutters, 78%/72% bubble widths, 46px controls, 720×420px glow and the single-row header are desktop-only. Without a breakpoint spec this ships a horizontally-scrolling header on every phone.

At `max-width: 640px`:
- Header: drop the kicker; wrap the presence row below the wordmark; New-conversation collapses to the ✦ glyph with an `aria-label`.
- Transcript: gutters 26px → 14px; bubbles to 92% / 88%.
- Composer: hint row keeps only the left half.
- Candle glow: scale to ~90vw.
- Use `100dvh`, not `100vh`.

### 5.6 `prefers-reduced-motion` checklist

The prototype has none. Write against this list:

- `animation: none` on `scanDrift`, `candleFlicker`, `glitchShiftA`, `prismSweep`, `sealPulse`
- keep `quillBlink` but slower, or a static caret
- `msgIn` / `noticeIn` / `cardIn` / `veilIn` → opacity-only (zero the translate)
- scroll behaviour `'smooth'` → `'auto'`

---

## 6. Components new in the mockup

- **Empty state** — 92px pulsing seal, "The study is lit. The ledger is open.", italic subcopy, three suggestion chips that prefill the composer.
- **Presence row** — 7px diamond, green `#7fc98f` idle / amber `#d8b45a` streaming, italic label. **Also carries identity** per §1: "in attendance · {{ username }}" with a ghost sign-out link beside New-conversation. This keeps `POST /chat/logout` reachable and `test_authed_user_sees_chat_ui`'s `assertIn("alice", ...)` passing.
- **Notice bar** — single slot below the header, info (❦, amethyst) vs error (✕, rose), × to dismiss, slides in 6px. Replaces all three `alert()` calls. Use the handoff's `{kind, text}` shape, not the prototype's two sibling fields. Info auto-dismisses (4-4.5s); errors do not.
- **Status chip** — parallelogram with prism sweep, "Fritz is consulting the archives…" → "…taking pen to paper…" after ~1.4s. Give the sweep `will-change: transform` and keep the parent `overflow: hidden`, or it paints outside the parallelogram.
- **Stop button "Enough"** — rose tint, beside the status chip. Settles the previous open question about the Stop label; the honest notice copy is the mitigation for it being cosmetic server-side.
- **Faceted confirm dialog** — replaces `window.confirm()`. Veil click and "Spare it" cancel; card click must not propagate.
- **Image card** — framed, 340px max, filename + meta caption.
- **Attachment chip** — staged upload with pentagon gem swatch and × to unstage.
- **Send blocked state** — the prototype never disables Send; it lets you click and scolds you. Keep the notice, but **also** add real `:disabled` styling (reduced-alpha gradient, `cursor: not-allowed`, `aria-disabled`) driven by `streaming`. The design system has no disabled treatment at all.
- **Textarea auto-grow** — declared (`rows="1"`, min 46px, max 150px, `resize: none`) but **never implemented** in the prototype. Write it: reset height to `auto`, set to `scrollHeight`, clamp at 150px, then enable `overflow-y: auto`. Verify against the sticky footer.

---

## 7. Server-side work the mockup requires

### 7.1 Timestamps

`_doc_to_message` (admin_panel.py:326-341) returns exactly `{role, content, html}`, and LangGraph checkpoint messages carry no timestamp. Showing times only on live messages makes every reload look like data loss.

Add a `ts` field written at message-creation time. Rehydrated history with no `ts` renders the speaker line without a timestamp rather than a wrong one.

### 7.2 Image metadata

`chat_upload_image` returns `{ok, url, name}` (admin_panel.py:719-723). PR 8's Pillow `verify()` already computes format and dimensions — **extend PR 8's response** to `{ok, url, name, width, height, format}`. The `done` frame (`:543`) carries only URLs and needs the same.

### 7.3 Code block language label — the expensive one

You chose to keep "python — tests/conftest.py". The cost:

- Enabling `codehilite` **removes** the `class="language-python"` that `fenced_code` alone emits. Verified.
- Markdown fences carry **no filename** — there is no source for the second half.

Implementation: a custom Markdown treeprocessor re-attaching the language as `data-lang`, plus adding `data-lang` to the nh3 allowlist in §0.1. The filename half needs a fence convention you invent (e.g. ```python:tests/conftest.py) and parse.

The analysis called this "a day of work for a caption." **If you cut scope later, cut this first** — a static "code" chip preserves the visual rhythm at zero cost.

### 7.4 Post-forget flow

`chat_forget` returns a 303 redirect (admin_panel.py:646); the mockup shows an auto-dismissing notice plus the empty state with **no reload**. Since decision 3 deletes the no-JS path, convert forget to `fetch()` + client-side clear.

### 7.5 Restate the code colours as Pygments classes

The handoff's colour table uses prototype-only names (`.hl-kw`, `.hl-str`…) that will **never appear** in production HTML. Drop `.hl-*` from the spec entirely and write:

| Role | Pygments classes | Colour |
|---|---|---|
| keyword | `.k, .kn, .kd` | `#c792ea` |
| string | `.s, .s1, .s2` | `#b8a1e3` |
| name/function | `.n, .nb, .nf, .nc, .nt` | `#e0cff5` |
| comment | `.c, .c1, .cm` | `#8579ab` *(lightened, §0.4)* |
| number | `.m, .mi, .mf` | `#d4b8f0` |
| operator | `.o, .ow, .p` | `#9d8bc0` |

Base `<pre>` colour `#c9bce0`.

---

## 8. Tests

### 8.1 The only hard break

**Exactly one** test asserts on chat.html markup: `TestChatPageWithCookie::test_authed_user_sees_chat_ui` (tests/test_admin_panel.py:386-392). `assertIn("alice", ...)` survives (identity moves to the presence row but stays in the body). `assertIn("Type a message", ...)` **breaks** — the placeholder becomes "Address the butler…".

The previous plan said "do NOT change the placeholder". **That instruction is stale and reversed.**

### 8.2 Login page assertions

Three tests assert "Sign in to chat" against `chat_login.html:6`: `:333`, `:382`, `:398`. Preserve that copy through the confirm-card restyle and all three pass. `test_empty_username_renders_error:356` asserts "at least one letter", which comes from `admin_panel.py:408` via the `{% if error %}` block at `chat_login.html:14-16` — **keep that block**.

### 8.3 Zero tests protect the UI

No test references any CSS class or element id in chat.html, and no test reads `admin_templates/*.html` off disk. `test_authed_send_invokes_ask_stuff_with_username:424` asserts `"Very well."` as a bare substring over the whole document — it would keep passing against a completely broken page.

Add template-source tests, stable across styling changes:

```python
class TestChatTemplateAccessibility(TestCase):
    def test_viewport_meta_present(self): ...
    def test_transcript_is_a_log_region(self): ...          # role="log" aria-live="polite"
    def test_status_region_present(self): ...               # #chat-status
    def test_forget_form_has_no_inline_onsubmit(self): ...
    def test_no_alert_calls_in_chat_script(self): ...
    def test_chat_extends_chat_base_not_admin_base(self): ...
    def test_chat_base_has_no_admin_nav(self): ...
    def test_every_faceted_control_has_focus_wrapper(self): ...   # §0.3 guard
```

### 8.4 The baseline is currently broken

`python-multipart` is not installed in `.venv`, so **30 of 65 tests in test_admin_panel.py fail before any change** — including the one test the redesign is supposed to break. Run `pip install -r requirements.txt` first; there is no valid baseline until then.

---

## 9. Steps

1. **PR 5 first** — `_sanitise_html` with the class allowlist + both guard tests (§0.1). Blocks everything else.
2. Fix the test baseline (§8.4).
3. `admin_static/` + `/static` mount + Basic-auth exemption + self-hosted woff2 + `OFL.txt` (§2.3, §5.4).
4. Split `_theme_base` / `_theme_chat` / `_theme_admin` (§2.1). Land the admin contrast fix here. **Highest blast radius in the set — alone in its own commit for bisectability.**
5. `chat_base.html`: shell, header, seal, wordmark, presence row with identity + sign-out (§2.2, §6).
6. Correctness and a11y: viewport meta, confirm via `addEventListener`, `aria-live`, focus wrappers (§0.3), reduced-motion block (§5.6).
7. Transcript restyle: bubbles, avatars, speaker lines, markdown scoping, entrance animation.
8. Composer: attach, textarea + auto-grow, send with disabled state, attachment chip, hint row.
9. Streaming UX: status chip, "Enough", caret, container follow-scroll, `applyToken` seam.
10. Notice bar + faceted confirm dialog + post-forget `fetch()` flow (§7.4).
11. Empty state + suggestion chips.
12. Codehilite CSS against real Pygments classes (§7.5) + copy buttons.
13. `chat_login.html` on the confirm-card shape, preserving the two asserted strings (§8.2).
14. Server-side: timestamps (§7.1), image meta via PR 8 (§7.2), language label treeprocessor (§7.3).
15. Move admin document upload to the admin panel; delete the `{% if is_admin %}` block from chat.html and retire its four tests.
16. Responsive pass (§5.5).
17. CSP header (§0.2) — last, applied to finished markup.

---

## 10. Risks

- **The sanitiser conflict fails silently.** No error, no failing test, just flat code blocks. §0.1's guard test is the only thing that catches it.
- **The contrast fix is easy to quietly undo** by anyone "restoring" the mockup's declared hexes, since the handoff calls them final. The trailing ratio comments are the defence.
- **Focus indicators are easy to lose again** — any new faceted control added without the wrapper repeats the bug. §8.3's guard test is worth the effort.
- **The theme split is the highest blast radius change in the whole plan set** — a dropped brace breaks all eight templates at once.
- **The atmosphere layers are permanent compositor work.** Watch battery on a laptop; the reduced-motion block is the escape hatch and promoting it to an env knob is ten lines.
- **Copy strings are load-bearing** for the persona and several are asserted. Port verbatim: the two status lines, the stop/blocked/burned notices, the empty-state headline and subcopy, the three suggestion labels with their curly quotes, the hint row, and the dialog's title/body/buttons.
- **Timer cancellation is the real state machine.** Every path that ends a stream (stop, confirm-new, unmount) must clear the fetch abort, the status-swap timeout, the auto-dismiss timeout and the accumulator. Miss one and you get a zombie status chip.

## 11. Rollback

The theme split is the risky commit; everything after is additive per-component work revertible individually. Keep `CHAT_CODE_HIGHLIGHT` as the codehilite escape hatch. If the atmosphere annoys you, the reduced-motion block already lists every selector to disable.

## 12. Still open

- Whether the atmosphere gets an env knob in addition to reduced-motion.
- Whether Enter-to-send should be enabled on touch devices (currently disabled on the theory that soft keyboards lack a reliable Shift).
- Whether the admin panel eventually adopts a restrained version of the faceted geometry, or stays permanently plain.
