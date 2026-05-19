# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance
- **Phase 10 — quick wins.**
  - `config.json` parse is now cached via `functools.lru_cache`. Repeated key lookups during import no longer re-open the file.
  - Automatic memory extraction now skips trivial turns (user message < `MEMORY_EXTRACT_MIN_USER_CHARS` or reply < `MEMORY_EXTRACT_MIN_REPLY_CHARS`). Saves an LLM call + embedding writes on "hi" / "lol" turns. Skipped turns increment the `memory_extract_skipped` counter so the rate is visible in `/health`.
- **Phase 11 — per-tool latency observability.**
  - New `Metrics.time_block(name)` context manager: increments a counter, records latency, and tracks errors as a single atomic operation around a block.
  - New `time_tool(name)` helper that auto-prefixes with `tool.` for the canonical tool-call metric namespace.
  - All `agent_tools` tools and `file_tools` tools now record latency, not just call count. The admin overview page surfaces the per-tool average automatically, so it's now possible to see which tool is dragging response times.

### Added
- `/help` slash command listing capabilities and slash commands.
- `/about` slash command showing bot version, active models, uptime, and a brief data-storage disclosure.
- `__version__` constant in `fritz_utils.py` as the single source of truth for the bot version.
- `CONTRIBUTING.md` with development setup, test/lint instructions, and how to add tools, commands, and config knobs.
- GitHub issue templates (bug report, feature request) and a PR template.
- `ruff` lint gate in CI.
- Interactive setup wizard at `scripts/setup.py` — checks Ollama, pulls models, validates Discord token, writes `.env`.
- `ADMIN_USERS` env var: comma-separated list of additional admin usernames. Anyone listed gets the same powers as `ROOT_USER`.
- `fritz_utils.is_admin(user_id)` — single source of truth for admin authorisation, decoupled from the single-`ROOT_USER` assumption.

### Changed
- `_require_root` helper renamed to `_require_admin` to reflect that it now allows anyone in `ADMIN_USERS`, not just `ROOT_USER`.
- All `user_id == ROOT_USER` comparisons across `bot_commands`, `file_tools`, `main_discord`, and `mister_fritz` migrated to `fritz_utils.is_admin()`. Zero behavioural change for single-admin deployments; multi-admin via `ADMIN_USERS` now works.
- **Phase 7b — per-user workspaces.** File tools are no longer admin-gated; any user can run `/workspace enable` to get a sandboxed directory at `WORKSPACES_ROOT/<user_id>/` and use the file tools (read/write/edit/search/list/run) scoped to it. Workspaces persist across bot restarts in a new SQLite table.
- `/workspace` is now a subcommand group: `/workspace status` (anyone), `/workspace enable` (anyone, creates sandbox), `/workspace disable` (anyone), `/workspace set <path>` (admin only, registers arbitrary host path). **Breaking change** to the prior `/workspace [path]` shape.
- `file_tools._authorize` now gates on "workspace is set", not "user is admin". The shell allowlist and `_resolve_safe_path` keep each user inside their own directory.
- New `WORKSPACES_ROOT` env var (default `./workspaces`).
- **Phase 7c — per-user schedules.** Dropped the Phase 2 admin gate on `/schedule add` and `/schedule remove`. Any user can manage their own schedules; the scheduler's per-user ownership check still prevents cross-user mutation.
- New per-user cap: `MAX_SCHEDULES_PER_USER` (default 10). Hit it and you get a clear error pointing at `/schedule remove`.
- New `/schedule list_all` admin subcommand for cross-user visibility.
- **Phase 8 — privacy commands.** New `/forget` subcommand group lets any user delete data Fritz has stored about them:
  - `/forget memories` — drops the user's Chroma namespace (memories + profile).
  - `/forget conversation` — clears the LangGraph SqliteSaver state for the user's thread; next message starts fresh.
  - `/forget schedules` — bulk-cancels every recurring task the user owns.
  - `/forget all` — runs the above plus drops workspace registration. Requires confirmation via a 30-second button view.
- New `/export` command sends the user a JSON attachment of every memory, schedule, profile entry, workspace path, and conversation-checkpoint count Fritz has on them. 8 MB cap.
- Every `/forget` and `/export` event appends an NDJSON line to `AUDIT_LOG_PATH` (default `./audit.log`) so deletions are reconstructable.
- New `privacy` module centralises all per-user data ops so the upcoming web admin panel can reuse them.
- **Phase 9a — read-only web admin panel.** New Starlette-based HTML admin UI at `http://127.0.0.1:8001/` (port configurable). HTTP Basic auth gated by `ADMIN_PANEL_PASSWORD`; if unset the panel doesn't start at all. Pages: overview (version/uptime/counters/errors), users list, per-user detail, all-schedules, document inventory, and a `/health` JSON route. Bound to localhost only — SSH-forward for remote access.
- Reuses the existing Jinja2 + uvicorn deps; no new packages added (Starlette comes in transitively via the LLM stack).
- **Phase 9b — mutating admin actions.** POST-only routes wired up to buttons on the existing pages:
  - "Forget everything about <user>" on the user detail page (runs `privacy.forget_all`).
  - "Disable workspace" on the user detail page.
  - "Cancel" next to each row on `/schedules` (admin override — bypasses the per-user ownership check).
  - "Re-index" next to each row on `/documents` (re-enqueues the file for ingestion; rejects paths outside `DOC_FOLDER`).
- Every mutating action writes an `admin_*` event to `AUDIT_LOG_PATH` with the admin's Basic-auth username, the target resource, and the result.

## [0.1.0] — 2026-05-18

First semver-tagged release. Backfills work landed across four prior phases.

### Added
- **Phase 1 — safety hotfixes**
  - `VECTORSTORE_LOCK` in `document_engine.py` to prevent races between the watchdog worker thread and the main thread.
  - `OLLAMA_TIMEOUT` env var (default 120 s) wired into both `ChatOllama` clients so a hung model cannot wedge the bot.
  - `execute_command` file tool now parses with `shlex`, validates `argv[0]` against `EXEC_ALLOWED_COMMANDS`, and rejects `..` and out-of-workspace absolute paths. No more `shell=True`.
  - Graceful shutdown for the watchdog observer and ingestion worker, via `atexit` and a queue sentinel.
  - Regression tests covering the new `execute_command` sandboxing rules.
- **Phase 2 — auth and config hygiene**
  - `_require_root` helper gating `schedule_add` and `schedule_remove` to `ROOT_USER`. `schedule_list` remains open (read-only).
  - Magic numbers hoisted to env-overridable constants in `fritz_utils.py`: `SUMMARIZE_THRESHOLD`, `MAX_READ_LINES`, `MAX_FILE_SIZE_BYTES`, `EXEC_OUTPUT_TRUNCATE`, `SCHEDULE_MIN_DELAY_MIN`.
  - All new knobs documented in `.env.example`.
- **Phase 3 — refactor**
  - New `bot_adapters` module owns `split_into_chunks`; `main_discord` and `bot_commands` import it instead of redefining their own copies.
  - `agent_tools.scrape_web` now uses a shared `httpx.Client` with split connect/read timeouts, giving connection pooling and a cleaner failure mode than the previous blocking `requests` call.
- **Phase 4 — tests and docs**
  - `tests/test_mister_fritz.py` — 12 tests covering the `planner()` JSON-extraction logic.
  - `tests/test_bot_commands.py` — 7 tests covering Phase 2 ROOT_USER gating.
  - `tests/test_document_engine.py` — 5 tests covering Phase 1 thread-safety.
  - README: new **System Requirements** section with per-model VRAM/RAM estimates, and an expanded **Troubleshooting** section covering Ollama OOM, hung requests, sandbox rejections, Chroma locks, ffmpeg, and schedule permissions.

### Changed
- `document_engine.py` module-level Mermaid PNG write is now wrapped in `try/except`, matching the pattern already used in `mister_fritz.py`. Lets the module import cleanly in offline / sandboxed environments.

### Fixed
- Replaced two bare `except: pass` blocks in `document_engine.ingestion_worker` and `initialize_vectorstore` with explicit exception handling at `debug` level so silent failures are observable.

### Security
- Tightened the `execute_command` file tool — argv allowlist, no shell interpretation, traversal and out-of-workspace argument rejection.
