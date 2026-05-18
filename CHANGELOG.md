# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
