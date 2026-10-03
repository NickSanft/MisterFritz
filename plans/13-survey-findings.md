# 13. What is left — the survey findings, batched

[← back to index](README.md) · [decisions](DECISIONS.md)

_October 2026. Produced by a 117-agent survey workflow run against the tree at `7440e52`. Eight independent lenses swept it — the roadmap in `plans/`, hot-path correctness, persistence and migrations, security and privacy, test coverage, dependencies and unadopted upstream capability, ops/CI/observability, and UX/documentation drift. Every candidate was then reality-checked against the code by a separate agent that had not seen the lens's reasoning, and each significant survivor faced two adversarial refuters: one attacking the mechanism ("can this sequence actually happen here?"), one attacking the value ("even if real, is it worth it for this app?")._

**40 candidates in, 23 out.** Two lenses independently reported the same `"scheduled"` defect, so there are **22 distinct items**. The 17 that were dropped are recorded at the bottom with the reason, so they are not re-proposed next time.

Every file:line below was verified at plan time. The six items in batch A were additionally re-read by hand, because a batch that claims something is broken today has to be right about that.

> Three of these already have background task chips queued: **A2** (`on_ready` idempotency), **A3** (`migrate_db`), and the `file_tools` symlink hardening mentioned under batch C. Starting one of those from its chip and also doing it here would duplicate the work.

## Grouping rules

Carried over from [plan 11](11-minor-findings.md), which these batches follow:

- **one concern per commit**, so a reviewer holds one idea at a time;
- **independently revertible** — nothing in batch N is a prerequisite for batch N+1;
- **docs, tests and behaviour are never mixed.** A commit that changes behaviour and rewrites the docs describing it hides the behaviour change in the diff.

Batch A is ordered by blast radius, not by effort. Every batch ends with `pytest tests/ -q && ruff check .` before pushing, and every behavioural fix carries a test verified to fail against the old code.

---

## Batch A — the six that break something today

### A1. A failed TTS load silently removes every slash command

**Where:** `main_discord.py:196-204`, `tts.py:50-54`

`on_ready` wraps the TTS bootstrap in `except ImportError` only. But `tts.py:52-54` logs and then **re-raises** whatever `TTS(self.model_name).to(self.device)` threw — a failed download of the ~2 GB XTTS weights, a CUDA OOM, a corrupt model cache, an unsupported device string. None of those is an `ImportError`, so the exception escapes `on_ready` and **every line from `main_discord.py:205` onward never runs**: the scheduler is never built, `relay_housekeeping` never reconciles the reservations a previous process left open or purges expired messages, the admin panel never binds, models are never pre-warmed, `client.add_cog(FritzCommands(...))` at :216 never registers, `client.tree.on_error` at :220 is never installed, and `tree.sync()` at :223 never runs.

The bot logs in and sits there with no commands and no diagnostic. It also makes `README.md:157` ("Without an extra its feature reports itself unavailable rather than crashing") false — true for an absent extra, false for a present but broken one.

**Fix:** broaden the handler to `except Exception`, with an error-level log for the present-but-broken case and the existing warning wording kept for `ImportError` so the README claim stays true for that path. `sayer` stays `None`, so `FritzCommands` still registers and `/voice` reports itself unavailable.

**Risk:** none — it strictly widens a handler. **Verify:** a test that makes the loader raise a non-`ImportError` and asserts the cog still registers.

### A2. `on_ready` is not idempotent — a reconnect doubles every schedule

**Where:** `main_discord.py:205-216`, `scheduler.py:279-315`, `scheduler.py:332`

discord.py re-dispatches `on_ready` on every full reconnect, not just the first. Lines 205-206 unconditionally do `schedule_manager = ScheduleManager(client)` then `.start()`, and `start()` reads every row out of `schedules` and `add_job`s it onto a brand-new `AsyncIOScheduler`. The previous manager is never stopped — `ScheduleManager.stop()` is called from nowhere in production code. Both schedulers then run on the same loop with the same job ids, so after one reconnect every persisted schedule fires twice; after N reconnects, N+1 times. Each firing is a full `ask_stuff` turn plus a channel post, and two of them land concurrently on the same LangGraph thread. `client.add_cog` at :216 also raises on the second pass without `override=True`, which is what currently stops `tree.sync()` from running on a reconnect.

**Fix:** guard the process-wide singletons the way `sayer` already is — build the scheduler once (or stop the previous one via the already-defined `stop()`), and pass `override=True` to `add_cog` so the rest of `on_ready` still completes on a reconnect.

**Risk:** low. **Verify:** dispatch `on_ready` twice against a fake client and assert one scheduler, one cog, jobs registered once.

### A3. `migrate_db` bricks a fresh deployment

**Where:** `migrate_db.py:121-133` (the column), `migrate_db.py:58`, `migrate_db.py:63-65`, `migrate_db.py:214-216`

`_ensure_target_schema` pre-creates LangGraph's `writes` table with `blob BLOB NOT NULL`. The pinned checkpointer names that column `value` and declares it nullable (`langgraph/checkpoint/sqlite/__init__.py:160`). Because `CREATE TABLE IF NOT EXISTS` no-ops on an existing table, `SqliteSaver.setup()` cannot repair the drift: once `migrate_db.py` has run against a fresh `fritz.db`, every `put_writes` fails with `OperationalError: table writes has no column named value`. So the documented upgrade path leaves the bot unable to finish a single agent turn, with a cryptic SQLite error and nothing pointing at the migration.

Two companions in the same file: `_migrate_table` copies with `INSERT OR IGNORE INTO {table} SELECT * FROM {alias}.{table}` — positional mapping, so any column-set difference misfiles data or raises; and that raise is swallowed at :63-65, after which main() can still print "0 row(s) copied — Done" and advise removing the old files.

**Fix:** delete the LangGraph table pre-creation entirely and let the checkpointer own its schema — `_migrate_table`'s existing missing-target skip already handles the consequence. Make a swallowed `OperationalError` reach main()'s exit code and suppress the "remove the old files" advice when any table warned.

**Risk:** low, and it only affects a path that is currently broken. **Verify:** a new `tests/test_migrate_db.py` that runs the migration against a temp DB, then has a real `SqliteSaver` complete a `put_writes` against the result.

### A4. The agent books every reminder under one shared pseudo-identity

**Where:** `agent_tools.py:537-552` (`"scheduled"` at :546), `scheduler.py:113-120`, `mister_fritz.py:851`, `mister_fritz.py:944`

`make_schedule_message_tool` takes no identity, so it passes the literal string `"scheduled"` as `user_id` to `schedule_once` — and it is the only caller. `ask_stuff`'s own docstring at `mister_fritz.py:850-851` states the contract this breaks: the value "must be a CANONICAL identity … used verbatim as the memory namespace and, with `channel_key`, the LangGraph thread." Four consequences, each traced through the code:

1. **One memory namespace for everyone.** `extract_memories_background("scheduled", …)` writes into Chroma namespace `("scheduled",)`, which `search_memories_internal` then reads back for any later reminder. A cross-user read, not a theoretical one.
2. **One LangGraph thread for everyone.** `THREADS_PER_CHANNEL` defaults to false, so `thread_id_for("scheduled", channel_key)` returns the bare `"scheduled"` — a single thread shared by every user's reminders in every channel and guild. One person's reminder conversation is replayed into another person's reminder turn.
3. **Invisible to the privacy commands.** `/forget` and `/export` resolve the caller's canonical id, so nothing stored under `"scheduled"` is reachable by either.
4. **`cancel_reminder` cannot cancel it.** That tool keys on the real identity, so Fritz confirms a reminder and then denies it exists.

The tool is not admin-gated: any guild member saying "remind me in ten minutes" reaches it. The `/schedule add` path is unaffected — it stores the real identity.

**Fix:** give `make_schedule_message_tool` a `user_id` parameter like its two siblings at `agent_tools.py:493` and `:515`, and pass it instead of the literal. The call site already has it in scope at `mister_fritz.py:594`. Keep a guarded fallback for the empty/legacy id rather than writing to an empty namespace. No migration — one-shot jobs are not persisted, so nothing in `schedules` carries the bad value.

**Risk:** low. Note the agent cache key comment at `mister_fritz.py:623-625` — the key is meant to be exactly what the closures capture, and `user_id` is already in it. **Verify:** assert `schedule_once` receives the caller's canonical id, and that a reminder's memories land in that namespace.

### A5. Neither deployment persists the database

**Where:** `docker-compose.yml:28`, `fritz_utils.py:71-73`, `infra/k8s/configmap.yaml:16`, `infra/k8s/deployment.yaml:58`

`DB_NAME` is `fritz.db`, and both `CHAT_DB_NAME` and `SCHEDULE_DB` default to it, so one file holds the LangGraph checkpoints, the relay messages and block table, the identity aliases and the schedules. Compose mounts `./chat_history.db:/app/chat_history.db` — a path nothing in the codebase opens, as `.env.example:214` itself documents. So `/app/fritz.db` lives on the container's writable layer and is destroyed by `docker compose down`, a rebuild, or any image update. The k8s manifests mount nothing either.

This lands squarely on a recorded decision. [DECISIONS.md](DECISIONS.md) #22 keeps placed blocks out of `/forget all` on the grounds that a privacy command must not become a safety regression — and a container restart does exactly what that decision forbade, silently, with no command run and no audit line. Second trap: because `./chat_history.db` does not exist in the repo, Docker creates a *directory* at that host path on first `up`.

**Fix:** mount what the code actually opens. Prefer a `./data` directory plus `DB_NAME=/app/data/fritz.db` over a bare file bind, which avoids the missing-source-becomes-a-directory trap entirely; add `./workspaces` and the audit log. For k8s, either drop `CHAT_DB_NAME` from the configmap so the three stay unified, or give the deployment a volume.

**Risk:** medium — it changes where data lives, so it needs a note about moving an existing `fritz.db` into `./data`. **Verify:** `docker compose up`, send a message, `docker compose down && up`, confirm the conversation and a placed block survive.

### A6. Two overlapping turns silently lose an exchange

**Where:** `mister_fritz.py:926` (`app.stream`), `mister_fritz.py:962`, `main_discord.py:417-427`, `fritz_utils.py:550`

Nothing serialises `ask_stuff` per thread. `thread_id_for` returns the bare identity by default, so one person's DMs and every channel they mention Fritz in share a single LangGraph thread. The blocking pool allows 8 concurrent calls and `on_message` offloads every message onto it with no per-user gate. Two turns that overlap — a follow-up sent while the first reply is still streaming, a DM plus a channel mention, or the duplicated firings from A2 — both stream against the one `SqliteSaver`. The saver's internal lock makes each individual DB operation safe, which is the only property plan 08 ever claimed; it does not make a read-modify-write turn atomic. The later writer wins and one question with its answer disappears from history. No error, no metric, no log line: it presents as the agent randomly forgetting things, indistinguishable from the model being bad.

**Fix:** a per-thread-id gate around the `app.stream` loop inside `ask_stuff`, modelled on the existing `_summarize_inflight` / `_summarize_lock` pattern at `mister_fritz.py:323-324` — a dict of thread id to lock behind one module lock, entry removed when the last waiter releases so it does not grow per channel. Inside `ask_stuff` rather than in `on_message`, so the Telegram adapter and the scheduler get it too.

**Risk:** medium — it introduces waiting where there was none, so a long turn now delays a follow-up instead of corrupting history. That is the intended trade. **Verify:** two concurrent `ask_stuff` calls on one thread; assert both exchanges survive in the checkpoint.

---

## Batch B — the privacy and feedback commands tell the truth

| # | Where | What |
|---|---|---|
| B1 | `bot_commands.py:674`, `:684`, `:695`, `:233` | Four destructive handlers do their blocking store work **before** responding, so they can blow Discord's 3-second deadline after the data is already gone — the user is told it failed. Insert `defer(ephemeral=True, thinking=True)` and switch to `followup`, copying the shape already at `:471-492`. |
| B2 | `privacy.py:57`, `:79`, `:152`, `:180`, `:198`, `:235`, `:277` | Every operation behind `/forget` and `/export` catches `Exception` and returns a neutral `0` / `False` / `[]`, so a total store failure renders as success. `count_conversation_checkpoints` has a bare `except` with no log line at all. Give the command layer a way to tell the two apart. |
| B3 | `main_discord.py:110-118`, `:128` | `_perform_update` retries a failing `message.edit` with no attempt limit and never clears `pending_text`. `NotFound` and `Forbidden` are both `HTTPException` subclasses, so a deleted placeholder is an infinite 404-every-2-seconds loop for the life of the process. Treat those two as terminal; bound the rest. |
| B4 | `stt.py:103-108`, `main_discord.py:392-394` | `transcribe` returns `None` both for an exception and for an empty transcription (`return text or None` — silence, a very short clip). For a voice-note-only message that `None` becomes the prompt. Tell the user it could not be transcribed. |

**Risk:** low throughout. **Verify:** B1 and B3 each need a test that fails against the old code; B4 is observable by sending a silent clip.

---

## Batch C — exposure and workflow hygiene

| # | Where | What |
|---|---|---|
| C1 | `observability.py:224-225`, `main_discord.py:467` | `start_metrics_server` hard-codes `0.0.0.0` with no bind knob, serving the full counter set and `/health` unauthenticated on every interface. Add `METRICS_HOST`, default `127.0.0.1`, and a README row. |
| C2 | `admin_panel.py:86`, `:125`, `:1231`, `admin_templates/*.html` | The five destructive POST routes have no CSRF defence of any kind, and because the panel authenticates with **Basic** rather than a cookie there is no `SameSite` protecting it. A `Sec-Fetch-Site` check inside `_BasicAuthMiddleware`, falling back to an Origin-vs-host compare and allowing the header-less non-browser case. |
| C3 | `.github/workflows/ci.yml:15`, `release.yml:58`, `:81` | No `permissions:` block on `ci.yml` (which runs PR-head code through `pip install -e` and `conftest.py`) or on `test-image` / `canary-deploy` (which holds cluster credentials). Every third-party action is on a moving tag. Add least-privilege blocks and pin by SHA; `release.yml:17-19`'s `packages: write` is load-bearing — leave it. |

Also here, for sequencing only: the queued **`file_tools` symlink hardening** — `os.path.normpath` collapses `..` textually without following links, so a symlink inside a workspace that points outside it passes containment at `file_tools.py:108-110`, `:506`, `:524`. It has its own chip.

---

## Batch D — dependency moves with real advisory content

| # | Where | What |
|---|---|---|
| D1 | `requirements.txt:279` | `pypdf` 6.4.2 → **6.19.0**. 45 in-range CVEs, every one fixed, highest required release 6.19.0. No release exists in the 6.4.x line after 6.4.2, so this is a minor move, not a patch bump. The cluster reads like a fuzzing campaign against malformed PDFs — infinite loops and unbounded allocation on crafted files, which `document_engine` feeds from the watched folder. Worth capping that ingestion path at the same time. |
| D2 | `requirements.txt:239` | `nltk` 3.9.2 → **3.10.3**. 44 in-range CVEs, the most of any pin. 3.10.3 clears all but CVE-2026-33236 (Downloader path traversal), which has no fixed release — so also stop ingestion reaching for corpora at import (`document_engine.py:27`, `:33`, `Dockerfile:51`). |
| D3 | `requirements.txt:277`, `pyproject.toml:31` | discord.py 2.6.4's own metadata declares `PyNaCl<1.6,>=1.5.0` for its voice extra; this repo pins `PyNaCl==1.6.1` and declares `PyNaCl>=1.5` as a core dependency, above that ceiling. pip never evaluates the extra's constraint because PyNaCl is declared directly. Record the ceiling rather than discovering it through a voice bug. |

> OSV's version filter is unreliable for these packages — it returns every advisory for a package regardless of version. The counts above come from walking the `introduced`/`fixed` events in the full records. Re-verify the same way rather than trusting a filtered query.

---

## Batch E — the code that has never run in a test

| # | Where | What |
|---|---|---|
| E1 | `storage.py:134-211` | `ChromaStore` is the only store production code constructs, and **not one of its data methods is ever executed** — `get`, `mget`, `put`, `mset`, `delete`, `mdelete`, `yield_keys`, `search` all have zero executed body lines. `/forget` and `/export` of memories are proven only against `MagicMock`s. A real-store class modelled on `tests/test_store_trust.py:270-295` with `FakeEmbeddings` needs no Ollama and runs in seconds. |
| E2 | `scheduler.py:90-130`, `:215`, `:258` | `tests/test_scheduler.py` has 20 tests, all about rows and trigger parsing. The callback the jobs are registered with never runs: `_run_task` has zero executed body lines, including the `fetch_channel` recovery and its `NotFound` / `Forbidden` skips. `schedule_once` and `remove_all_for_user` never execute either — which is why A4 survived this long. |
| E3 | `observability.py:185-230` | The documented operational surface is entirely unexecuted: `start_metrics_server`, its thread body, the uptime updater, and `_MetricsHandler.do_GET` with its `/metrics`, `/health`, 503-without-prometheus and 404 branches. README documents the port and both endpoints; Prometheus scrapes them. Stand it up on port 0 and assert the four branches. |

**Note:** E1 and E2 are not coverage theatre — each covers a path a user's data actually travels. Do them after batch A so they test the fixed behaviour, not the broken behaviour.

---

## Batch F — CI never builds or boots the image

**Where:** `.github/workflows/ci.yml:33`, `release.yml:45`, `:71-72`, `Dockerfile:63`

`Dockerfile`'s `ENTRYPOINT ["python", "main_discord.py"]` is never executed anywhere in CI. `ci.yml` has no docker step, so the image is first built on a `v*` tag. Four gaps compound: the dependency set the image uses is never installed in CI (`ci.yml` installs `-e ".[dev]"` from pyproject's floors, the Dockerfile installs pinned `requirements.txt`, and `test_packaging.py` only parses that file as text); the release smoke test runs a different program from the entrypoint; and it pulls a tag that may never have been pushed.

**Fix:** build the image in `ci.yml` and boot it with a fail-fast command that exercises the real entrypoint's imports. **Verify:** break a pinned version deliberately and confirm CI goes red.

---

## Batch G — documentation that is missing rather than stale

| # | Where | What |
|---|---|---|
| G1 | `README.md:7-22`, `:26-44`, `:221-245` | The relay shipped in six PRs and is **invisible** in every document a user or operator would read: zero occurrences of "relay" or "/tell" in the README — not in Features, not in the slash-command table (which does list `/forget` and `/export`), and not one row for any of the seven `RELAY_*` knobs, including `RELAY_ENABLED`, which defaults to false. The Tunables table is 41 rows short overall. |
| G2 | `fritz_utils.py:242`, `:797-801`, `.env.example:111-112` | `RELAY_AGENT_TOOL_ENABLED` gates nothing: plan 12's PRs 8-9 are unbuilt, and the knob's only consumer warns solely when it is true *and* `RELAY_ENABLED` is false. So `true`+`true` is accepted silently and does nothing, while `.env.example` advertises it as a working feature. Either build phase 2 or make the knob say it is inert — the ten-line version is the warning plus a comment clause. |

---

## Needs a decision before it can be planned

The completeness critic asked what the eight lenses structurally could not see. Four answers, each verified against the code, and each raising a question that belongs in [DECISIONS.md](DECISIONS.md) rather than being settled by whoever writes the patch.

**1. Nothing in the system says what "9am" means.** `ScheduleManager` builds bare `CronTrigger(minute=…, hour=…)` triggers (`scheduler.py:79-82`), and APScheduler resolves an unspecified timezone to the host's local zone. The `schedules` table has no timezone column — only `created_at`, which is UTC. So `/schedule list` echoes `0 9 * * *` back with no zone, and the hour is defined by an implicit property of the machine, never recorded, and silently changes meaning across a DST boundary or a host move. **Question:** per-user timezone, one configured bot timezone, or UTC-with-display-conversion?

**2. There is no stop path.** No signal handling anywhere in the repo, and discord.py installs none — `client.run` catches only `KeyboardInterrupt`. Under `SIGTERM`, which is how `docker compose stop`, a k8s rolling update and the Argo canary all stop a process, Python's default handler kills the interpreter outright: `atexit` never fires, so the document ingestion worker and the watchdog observer are abandoned, in-flight turns vanish, and pending reminders are silently voided. The relay already accepted the principle that a promise the user watched the bot make must not vanish without a word. **Question:** how much of a turn is worth draining on shutdown, and what should the next boot say about what the stop interrupted?

**3. The `skills/` extension point has no contract.** It is the one documented way a user extends this bot, and the one surface with neither validation nor a single test. The loader checks only that `register()` returned a dict (`agent_tools.py:478-483`), not that each value is the `(tool, description)` tuple its consumers require — so the easy mistake survives the loader's `try/except` and detonates later at import. A skill may also silently shadow a core tool, including the relay and schedule closures whose guardrails live in their factories. **Question:** reject-and-log a malformed skill, or refuse to boot? And is shadowing a core tool ever legitimate?

**4. Nothing bounds a turn or a user's share of the pool.** `BLOCKING_POOL_SIZE` defaults to 8 and every Discord message holds one of those threads for the whole turn. `OLLAMA_TIMEOUT` caps a single request, not a turn: with `AGENT_RECURSION_LIMIT` at 40 and `ToolRetryMiddleware(max_retries=2)`, one turn can legitimately hold a thread far longer than the comment at `fritz_utils.py:85` implies. Bounding the pool was right, but it turned a slow path into a shared one — one pathological turn now degrades the whole bot, and the symptom is a placeholder that never resolves. **Question:** a turn deadline, per-user admission, or both?

---

## Dropped — considered and rejected

Recorded so they are not re-proposed. Each was killed because the mechanism was real but the trigger cannot occur here, or because something already in the code blocks the consequence. One was simply not real.

| Lens | Candidate | Why dropped |
|---|---|---|
| roadmap | Relay docs including `/help` and an `/about` data-storage disclosure | Refuted as framed; the narrower README gap survived as **G1** |
| roadmap | Mirror `relay.*` / `discord_commands.*` counters into Prometheus | Factual core right, mechanism does not survive reading the code |
| roadmap | Move the last `run_in_executor(None, …)` sites onto the bounded pool | Line numbers accurate, but the sequence cannot produce the claimed consequence |
| data | A retention policy for `checkpoints` / `writes` | "One unbounded store" premise is false |
| data | Key extracted memories by derived key rather than a fresh `uuid4` | Trigger essentially does not occur here |
| data | Make `migrate_identity --apply` survive a primary-key collision | Mechanism real and reproduced; headline framing and trigger refuted |
| data | Make `migrate_identity` follow `CHAT_DB_NAME` | Mechanism real; consequence, reachability and the proposed fix all refuted |
| security | Keep `.chat_cookie_secret` and the audit log out of the image and git | Facts right, mechanism for the claimed consequence does not exist here |
| security | Enforce `CHAT_ALLOWED_USERS` on the request path; rotation revokes sessions | Refuted — the gap is narrower than claimed |
| tests | The document engine's query graph and loaders are mocked away (37.5%) | Coverage arithmetic right; every consequence is blocked by something already present |
| tests | No `@tool` entry point is ever invoked; memory/profile readers untested | **Not real** — headline and principal consequence both wrong |
| deps | A second patch-bump pass across ten pins | Refuted on mechanism; D1/D2 are the two that actually matter |
| deps | Adopt langchain 1.4 `ContextEditingMiddleware` for the 4096-token budget | Individual facts accurate, the sequence they imply is not |
| ops | Make `/health` able to fail on gateway disconnect | Refuted as framed — the literal-response observation is true, the framing is not |
| ops | `infra/k8s` as committed runs two bots and resets the token to `REPLACE_ME` | Each sub-claim's consequence is blocked by a distinct mechanism in the files |
| ops | The canary analysis gate cannot detect a bad canary of this workload | Central mechanism wrong, both examples wrong, trigger cannot occur |
| ux | `/gen` should report the missing `[image]` extra; stop advertising `generate_image` | Central code reading accurate; the mechanism that would make it matter is not |

---

## Status

| Batch | Items | State |
|---|---|---|
| A | 6 | A1, A3, A4 in progress — see commits following this plan |
| B | 4 | not started |
| C | 3 (+ queued symlink chip) | not started |
| D | 3 | not started |
| E | 3 | not started |
| F | 1 | not started |
| G | 2 | not started |
| Decisions | 4 | awaiting answers in [DECISIONS.md](DECISIONS.md) |
