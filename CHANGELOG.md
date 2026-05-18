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
