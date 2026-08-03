# Decisions

Answers to the open questions raised across the ten plans, recorded August 2026. Where a decision changes a plan as written, the required amendment is spelled out. **These override the corresponding plan text.**

---

## 1. `execute_command` — Config A, admin-gated

Interpreters stay (`python`, `node`, `pip`, `git`, …), the subprocess environment is scrubbed of secrets, and the tool becomes admin-only via `is_admin`. `EXEC_REQUIRE_ADMIN=false` remains as a one-line escape hatch.

**Amends plan 04.** Take Config A; drop the Config B branch. Non-admin workspace users keep read/write/edit/search/list but lose `execute_command`.

**Follow-through:** the env scrub must keep `SystemRoot`, `COMSPEC` and `PATHEXT` on Windows or subprocess breaks. Verify on the production host by running `git status`, `python -c "import tempfile;print(tempfile.gettempdir())"` and `npm --version` through the bot after deploying; anything that breaks is an `EXEC_ENV_PASSTHROUGH` addition, not a bug.

**Note:** this makes `/workspace enable` staying ungated much less urgent, since the exec path is now behind the admin gate. Unbounded disk use under `WORKSPACES_ROOT` is still unaddressed.

---

## 2. Web chat auth — shared password now, tokens later

Gate `/chat/login` behind a shared secret. Per-user invite tokens are deferred, not rejected.

**Amends plan 05.** Ship the password gate; drop the token design from scope but keep the sketch for later.

**Understand what this does and doesn't fix:** it closes anonymous access to the port. It does *not* stop one person who knows the password from typing another person's username and reading their conversation. That gap only closes with invite tokens or the thread separation in decision 2b below.

### 2b. `CHAT_PASSWORD` is separate — no fallback to `ADMIN_PANEL_PASSWORD`

Chat is **off after upgrade until `CHAT_PASSWORD` is set**. That is the intended loud failure. Log a clear startup line saying chat is disabled and why, rather than 404-ing silently.

Giving someone chat access must never hand them the admin panel's password.

---

## 3. Delete the no-JS `/chat/send` fallback

Remove `chat_send` (admin_panel.py:425-480) and its four tests along with the streaming hardening.

**Amends plan 05.** One less place to keep sanitization, identity and thread-ID changes in sync.

---

## 4. Delete plan mode outright — the largest amendment

Remove `PLANNER_NODE` and `SYNTHESIZER_NODE`. The graph collapses to `START → executor → (summarize | END)`.

This goes further than any plan assumed and **changes three of them**:

### Plan 08 (latency-tax)
The heuristic planner gate is moot — there is no planner to gate. Delete `planner()` (mister_fritz.py:162-223) rather than short-circuiting it, and drop `PLANNER_MIN_CHARS` from the config batch. The off-critical-path summarization work is unaffected and still wanted. Of the three hand-rolled JSON sites, the planner one disappears; **profile-signal extraction and memory extraction still need native structured outputs.**

### Plan 01 (history-window)
The plan-mode branch of `executor` disappears — no `_history_window(messages[:-1])` variant, no synthetic step prompt. Only the simple-mode path survives, which makes this plan strictly simpler than written.

### Plan 03 (token-streaming)
Everything about plan-mode streaming suppression (mister_fritz.py:405) is deleted rather than handled. Note that `synthesizer` was the **only** place doing true token streaming today (`ollama_instance.stream` at :490-499) — deleting it is safe *only because* this item replaces it with token streaming at the executor. **Land token streaming before or with the plan-mode deletion**, or there is a window with no streaming at all.

### Also
- `EnhancedState` fields `plan`, `current_step`, `step_results` and `original_request` become vestigial — remove them.
- `route_executor` (:141-157) and `should_continue` (:136-138) collapse into one edge, which incidentally resolves the duplicated-logic defect the audit flagged.
- Plan-step progress notices vanish from `progress_callback`, shrinking plan 10's progress work.

**The real risk to watch:** multi-step requests previously got an explicit decomposition plus a final persona-voiced synthesis pass. Now the ReAct loop must decompose on its own, and replies land in whatever voice the executor produced. The executor's system prompt already carries the persona so simple replies are already in character — but **test a genuinely multi-step request** ("search for X, then summarize it against what you know about me") before and after, and compare both correctness and tone.

---

## 5. Identity separator — dash, not colon

`discord-123456`, not `discord:123456`. Colon is illegal in Windows filenames and you are on Windows.

**Amends plan 09.** Use `partition("-")` with a closed platform allowlist (`discord`, `telegram`, `web`) so a display name containing a dash still parses correctly — `web-alice-bob` splits to platform `web`, id `alice-bob`. This removes the `safe_user_token` discipline the plan required at main_discord.py:161, :170, bot_commands.py:295 and admin_panel.py:711.

---

## 6. `ADMIN_LEGACY_NAME_MATCH` defaults to **false**

Secure from day one. No impersonation window, not even for one release.

**Before deploying the migration**, put your numeric Discord ID in `ADMIN_USERS` — otherwise you lose admin commands until you edit `.env`. Get it by enabling Developer Mode in Discord and right-clicking your name → Copy User ID.

---

## 7. Delete `browser_tools.py`

Remove the module rather than wiring it up. Playwright does not enter the dependency set.

**Amends plan 06** — drop the `[browser]` extra.

**Do not lose this:** the 8000-char output cap that `browser_tools.py` had and the live `scrape_web` lacks must still be copied into `agent_tools.scrape_web` (agent_tools.py:241-248). That was always a separate fix; deleting the dead module does not do it for you.

---

## 8. Raise `SUMMARIZE_THRESHOLD` once summarization is off-path

The current value of 15 is aggressive because summarizing was expensive and blocking. Once it runs in the background, raise it for longer in-thread memory.

**Ordering is load-bearing:** off-path summarization (plan 08) must land *first*. Raising the threshold while summarization is still synchronous makes the stall on those turns worse, not better.

Suggested target 30-40, tuned by feel. Each summarization gets slower since it feeds the whole message list to the 20B model, but you no longer wait on it. Keep the trim-to-one-message behavior for now; revisit if continuity still feels short.

---

## 9. `requirements.txt` stays a hand-reviewed freeze

No pip-tools or uv. Keeps the Dockerfile consuming the file unchanged and adds no dev tooling.

**Amends plan 06** — take the freeze path, drop the lock-pipeline alternative.

---

## 10. `/health` — ephemeral, not admin-gated

Stops metrics leaking into channel history; any guild member can still run it themselves. Preserves asking a friend to run it while debugging remotely.

---

## 11. ~~One shared theme for chat and admin panel~~ — **SUPERSEDED by decision 12**

~~The parchment/mahogany token set applies to both surfaces.~~

Overtaken by the `NewMockup/` design. See decision 12.

---

# Mockup decisions (2026-08-03)

Answered after the `NewMockup/` dark-academia design landed. These drove the rewrite of [plan 07](07-web-chat-redesign.md); see that document for the implementation detail.

## 12. Theme scope — chat only. Decision 11 is overridden

The dark-academia palette applies to the chat surface only. `overview` / `users` / `schedules` / `documents` keep a plainer utility look.

The faceted clip-paths fight the `table { display: block; overflow-x: auto }` mobile rule, and candle-lit purple buys nothing on a data grid. `_theme.html` splits three ways: `_theme_base` (reset, responsive, a11y — shared), `_theme_chat`, `_theme_admin`.

**The admin panel still gets its contrast fix.** `--muted` failing at 3.67:1 was a real audit finding and it survives the scope change — it just gets fixed in the admin palette rather than a shared one. Note the audit undercounted the consumers: `h2 { color: var(--muted) }` at base.html:33 means every section heading on every admin page is currently sub-AA.

## 13. Fonts — self-hosted woff2

Six faces (Cormorant Garamond 400/600/700, EB Garamond 400/400i/**600**, JetBrains Mono 400), latin subset, ~150-170 KB cached forever. The prototype's Google request asks for 13 faces and uses 6 — and omits EB Garamond 600, which `.fritz-md strong` needs.

Rejected the CDN: a 127.0.0.1 admin panel should not beacon to a third party on every page load, it renders unstyled on an egress-less host or in Docker, and it forces `fonts.googleapis.com` / `fonts.gstatic.com` into the CSP.

**This adds a `/static` mount that does not exist today**, and its path must be added to `_BasicAuthMiddleware`'s exemption (admin_panel.py:77) or `/chat/login` renders unstyled behind a password prompt.

## 14. Contrast — lighten the failing tokens

Three mockup colours fail WCAG AA against composited backgrounds: `#574a75` at **2.26:1** (worse than the 3.67:1 this redesign exists to fix), `#6f5f92` at 3.28:1, `#64578a` at 3.11:1.

Collapse `#574a75` / `#6f5f92` / `#8d7fa8` into one `--text-meta #9082aa` (4.77:1 worst case); move the code comment to `#8579ab`. Keep `--text-dim`, `--text-body` and `--text-hi` exactly as designed.

**This departs from the handoff**, which declares colours final and asks for pixel-perfect recreation. Measure against the *composited* background (bubble = 0.75 alpha over the page gradient), not the raw hex — measuring against `#0b0710` flatters every number.

## 15. Atmosphere — ships on, honours reduced-motion

Scan lines, candle glow and the wordmark glitch all ship enabled. The prototype implements **zero** `prefers-reduced-motion` handling; write it against the checklist in plan 07 §5.6.

An env knob was considered and left open — the reduced-motion block already names every selector, so promoting it later is ten lines.

## 16. Keep all three "missing data" details

Timestamps, the image metadata caption, and the code-block language label all stay, and each needs new server-side plumbing:

- **Timestamps** — `_doc_to_message` returns only `{role, content, html}` and checkpoints carry no time. Add a `ts` field; render history without a timestamp rather than a wrong one.
- **Image meta** — extend PR 8's response to `{ok, url, name, width, height, format}`; Pillow already computes it.
- **Language label** — the expensive one. `codehilite` strips the `language-*` class `fenced_code` emits, and fences carry no filename. Needs a custom treeprocessor plus `data-lang` in the nh3 allowlist, and a fence convention you invent for the filename half. **If scope is cut later, cut this first** — a static "code" chip preserves the rhythm at zero cost.

## 17. Identity lives in the presence row

"in attendance · {{ username }}" beside the status dot, with a ghost sign-out link next to New conversation. The mockup header has neither, but `POST /chat/logout` is a live route and `test_authed_user_sees_chat_ui` asserts the username appears in the body.

## 18. Login page reuses the confirm-dialog card

`chat_login.html` is not in the mockup. Reuse the "Burn the correspondence?" card shape — 24px faceted card, seal badge, radial ground — so it matches with no new design work.

Preserve two literal strings or four tests break: **"Sign in to chat"** (asserted at test_admin_panel.py:333, :382, :398) and the `{% if error %}` block at chat_login.html:14-16 (carries "at least one letter", asserted at :356).

## 19. Admin document upload moves to the admin panel

Consistent with decision 2 — a privileged action does not belong on a surface where identity is a claim. Delete the `{% if is_admin %}` block from chat.html; its four tests move or retire.

---

# Amendments to plan 05 forced by the mockup

Both were found by verification, and both would have failed silently.

## 20. `_sanitise_html` must allow `class` — plan 05 as written breaks all syntax highlighting

Plan 05 calls `nh3.clean(raw_html)` with no `attributes=`, which strips `class="codehilite"` and every Pygments token class. Code blocks render flat with no error and no failing test.

```python
_NH3_ATTRS = dict(nh3.ALLOWED_ATTRIBUTES)           # dict() copy load-bearing
for _t in ("div", "pre", "code", "span", "table"):
    _NH3_ATTRS[_t] = (_NH3_ATTRS.get(_t) or set()) | {"class"}   # default is None
```

**PR 5 before PR 13, always.** Ship `test_codehilite_classes_survive_sanitiser` with it.

## 21. The CSP must gain `font-src`, and `script-src 'none'` is not viable

Plan 05's `default-src 'none'` with no `font-src` blocks every `@font-face` — self-hosted or CDN. And `script-src 'none'` kills the chat client outright, since chat.html is one large inline script.

Ship: `default-src 'none'; script-src 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'`

Never write `style-src 'nonce-x' 'unsafe-inline'` — the nonce makes `'unsafe-inline'` be ignored and every `style=""` attribute dies.

---

## Still open — deliberately not decided

These came up during planning and remain unresolved. None blocks tranche 1.

- Whether `/workspace enable` should be gated or size-capped (unbounded disk use under `WORKSPACES_ROOT`).
- Whether `./output` assets should be per-user, or whether "any signed-in user can see any generated image" is acceptable.
- Whether generated images and `temp_audio/` need a retention policy at all.
- Whether `DISCORD_ERROR_DETAIL` is worth adding, given the `(ref …)` token already lets you grep the log.
- `/draw` upper bound — 40 fits one message; higher is safe once chunking lands, but may just be a footgun.
- Whether `/schedule list_all` stays plain chunked text while `/schedule list` becomes an embed.
- Enter-to-send behavior on touch devices (currently disabled on the theory that soft keyboards lack a reliable Shift).
- ~~Whether the Stop button should say "Stop" or "Hide reply"~~ — **resolved by the mockup: "Enough"**, with honest notice copy explaining the model keeps generating server-side.
- Whether `search_memories_internal`'s `limit=30` should drop to ~10, and whether the injected memory blob should stop carrying internal `namespace` / `original_key` metadata on every turn.
- Whether the atmosphere layers get an env knob in addition to `prefers-reduced-motion`.
- Whether the admin panel eventually adopts a restrained version of the faceted geometry, or stays permanently plain.
