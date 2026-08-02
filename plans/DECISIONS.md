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

## 11. One shared theme for chat and admin panel

The parchment/mahogany token set applies to both surfaces. Smallest change, and it fixes the `--muted` contrast failures on the admin tables for free.

**Confirms plan 07** as written — `_theme.html` carries the full palette, both surfaces consume it.

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
- Whether the Stop button should say "Stop" (honest about the UI, not the server) or "Hide reply".
- Whether `search_memories_internal`'s `limit=30` should drop to ~10, and whether the injected memory blob should stop carrying internal `namespace` / `original_key` metadata on every turn.
