# 11. The minor findings — a batching plan

[← back to index](README.md) · [decisions](DECISIONS.md)

The post-implementation audit produced 39 minor findings alongside the criticals and majors (all of which are now closed). None of these is load-bearing; collectively they are the difference between "works" and "maintained".

**2 are already done** — both were the `/chat/send` fallback (#12, #23), deleted in `cb5f119`. **37 remain**, grouped below into 9 commits.

## Grouping rules

Each batch is one commit and one push, chosen so that:

- **it touches one concern**, so a reviewer holds one idea at a time;
- **it is independently revertible** — nothing in batch N is a prerequisite for batch N+1, so a bad batch can be dropped without unpicking the rest;
- **docs, tests and behaviour are never mixed.** A commit that both changes behaviour and rewrites the docs describing it hides the behaviour change in the diff.

Batches A–D are near-zero risk and can go in any order. E–G change runtime behaviour and deserve a live smoke test. H–I are infrastructure.

Every batch ends with `pytest tests/ -q && ruff check .` before pushing.

---

## Batch A — documentation that now lies (8 findings, no code)

The highest-value cheap batch: each of these actively misleads the next reader.

| # | Where | What |
|---|---|---|
| 8 | `chat.html:893` | `applyToken` seam comment still says token frames are cumulative. They carry deltas — the flip already landed. |
| 9 | `CHANGELOG.md:40` | Same stale "cumulative today" claim, plus a wrong parameter name. |
| 16 | `admin_panel.py` | Comment above the chat-identity helpers still says "cookie-based, no password" — the exact claim `CHAT_PASSWORD` removed. |
| 17 | `CHANGELOG.md` | A bullet describing the `is_admin` render fix that the same release superseded by removing `is_admin` from the chat surface entirely. **Verify it is still present** — a grep for it came back empty. |
| 14 | `README.md:435` | Advertises the chat-side "Add to shared docs" control, which moved to the admin panel. |
| 15 | `README.md:441-446` | Chat config table missing `CHAT_PASSWORD`, `CHAT_ALLOWED_USERS`, `CHAT_COOKIE_SECURE`, `CHAT_CODE_HIGHLIGHT`. `CHAT_PASSWORD`'s "no fallback, chat is off until set" is the one a new deployer most needs. |
| 38 | `README.md:35,38` | `/draw` documented without its 1–40 bound; `/health` without "(ephemeral)". |
| 20 | `.env.example` | `SCRAPE_MAX_CHARS` undocumented while both its siblings are. |

**Risk:** none. **Verify:** read the diff.

---

## Batch B — Discord surface leftovers (5 findings)

Everything plan 10 specified but did not ship, plus one data-hygiene item.

| # | Where | What |
|---|---|---|
| 36 | `bot_commands.py` | `/draw` is the one command with no `METRICS.increment`. |
| 39 | `bot_commands.py:612,422` | `split_into_chunks` returns `[]` for empty text, so `/lore` and `/draw` can defer and then send nothing — the user watches "thinking…" forever. Restore the `or [text]` fallback. |
| 31 | `main_discord.py:220` | The bot records **itself** as a user: `identity_store.record` runs before the `ctx.author == client.user` guard at :223, so every message Fritz sends upserts an alias row for the bot account. Move the call below the early returns. |
| 37 | `temp_audio/` | 9 leaked files from before the cleanup fix. The leak is closed; the historical files remain. |
| 7 | `bot_commands.py:336,153` | `privacy.forget_memories` / `forget_all` still run Chroma work on the event loop. Wrap in `run_blocking`. |

**Risk:** low. #31 changes what lands in `user_aliases` — confirm the bot's own row stops appearing.
**Verify:** run `/draw`, `/lore` with a query that returns nothing, and `/forget memories`.

---

## Batch C — metrics that measure the wrong thing (3 findings)

Grouped because they are all `observability` semantics, and all silently produce plausible-but-wrong numbers.

| # | Where | What |
|---|---|---|
| 3 | `bot_commands.py:589` | Errors on `/gen` and `/voice` are counted **twice** — `METRICS.time_block` records and re-raises, then `_reply_error` → `fritz_error` records again. |
| 5 | `bot_commands.py:589` | Recorded latency includes semaphore **queue** time, so with `IMAGE_GEN_MAX_CONCURRENCY=1` a queued `/gen` reports the wait as render time. Move `time_block` inside the semaphore, or add a separate queue metric. |
| 28 | `mister_fritz.py:212` | The `summarize_background` histogram was never wired up, so `/health` shows nothing for it. |

**Risk:** low, but it changes what `/health` reports. **Verify:** trigger an error on `/gen`, check the error count moves by 1 not 2.

---

## Batch D — config guards (2 findings)

Both turn a silent hang into a loud failure.

| # | Where | What |
|---|---|---|
| 4 | `fritz_utils.py:174,177` | `IMAGE_GEN_MAX_CONCURRENCY=0` is accepted; `asyncio.Semaphore(0)` never releases, so `/gen` hangs until the 15-minute interaction expiry with no log line. Clamp with `max(1, …)` or reject in `validate_config()`. |
| 30 | `migrate_identity.py` | The plan required refusing `--apply` when `fritz.db-wal` is non-empty — the guard against running the migration while the bot is live. It was never added. |

**Risk:** none. **Verify:** set the knob to 0 and confirm a clear error; run the migration with the bot up and confirm it refuses.

---

## Batch E — memory and summarisation quality (4 findings)

These change what the model sees, so they go together and get a live smoke test.

| # | Where | What |
|---|---|---|
| 1 | `mister_fritz.py:482` | The injection cap under-fills by roughly half: `break` on the first entry that would cross the budget means one 2001-char entry stops a 4000-char budget dead. Measured: 30×2000-char entries under a 4000 cap yields **one** key. Use `continue`, or accumulate against serialized length. |
| 2 | `mister_fritz.py:285` | The turn right after every summarisation gets **zero** in-thread history — identical to the bug plan 01 exists to fix. The trim keeps only `messages[-1]`, which is Fritz's own reply. Keep the last **two**. |
| 25 | `mister_fritz.py:198` | `_make_memory_key` takes `words[:8]` verbatim, producing stopword soup. The plan specified a `_MEMORY_KEY_STOPWORDS` frozenset and content-word filtering. |
| 26 | `mister_fritz.py:169` | `ProfileSignals` gives every field a default, so its JSON schema has **no** `required` array — the plan declared them without defaults precisely to avoid that. |

**Risk:** medium — this is the model's context. **Verify:** live Ollama, a multi-turn conversation that crosses `SUMMARIZE_THRESHOLD`, confirming recall survives the summarisation boundary.

---

## Batch F — identity consistency (4 findings)

All four are places the identity work landed unevenly. None is a live bug; each is a trap for the next change.

| # | Where | What |
|---|---|---|
| 32 | `mister_fritz.py:123` | `get_source_info` drops the canonical id and mislabels the display name as the user id — the prompt reads "(User ID: Divora)". The plan specified both values. |
| 33 | `admin_panel.py:452` | `_chat_thread_id` passes no channel key. Identical today, but the moment `THREADS_PER_CHANNEL` is enabled, Discord branches and the web surface does not. |
| 35 | `privacy.py:50` | `resolve_identity` is applied in 2 of 7 operations. Harmless now; a correctness bug the first time a legacy id reaches one of the other 5. |
| 13 | `file_tools.py:429` | The exec admin gate and the slash-command admin gate disagree about `ADMIN_LEGACY_NAME_MATCH` — one passes `display_name`, the other does not. Decide deliberately, then make them agree. |

**Risk:** low. **Verify:** flip `THREADS_PER_CHANNEL` on and confirm both surfaces branch; run `/forget all` for a legacy-named user.

---

## Batch G — wasted per-request work (3 findings)

| # | Where | What |
|---|---|---|
| 27 | `mister_fritz.py:204` | `config_values` is threaded through the summarisation path and never used — dead per-request work added by the item whose purpose was deleting dead per-request work. |
| 29 | `mister_fritz.py:455` | The cached-agent fast path is **dead on Discord**: `main_discord` passes `channel_id` and `schedule_manager` on every message, so the tool registry is rebuilt per request anyway. Memoise on the tool-set identity. |
| 22 | `admin_panel.py:474` | `_label_code_languages` mislabels blocks whenever an unlabelled fence precedes a labelled one — it collects only labelled fences but substitutes across every block in document order. A real rendering bug. |

**Risk:** #29 touches agent construction — the highest-risk item in this document. Consider splitting it out if the batch gets noisy.
**Verify:** a reply containing an unlabelled fence followed by a ```python fence; confirm the labels land on the right blocks.

---

## Batch H — packaging (3 findings)

| # | Where | What |
|---|---|---|
| 18 | `Dockerfile:14` | Still `COPY requirements.txt .`; the plan specified copying `pyproject.toml` too so the lock's provenance travels with the image. |
| 19 | `requirements.txt` | The header claims the lock was fixed by adding `zstandard`, but faster-whisper's other transitives (`av`, `ctranslate2`) are still unpinned — so either pin them or stop claiming the file locks. |
| 21 | `requirements.txt:279` | Pins and floors are mixed incoherently and the block comment describes the wrong entries. |

**Risk:** none to runtime; **verify** by building the image.

---

## Batch I — test integrity (5 findings, tests only)

Last because it is the least urgent and the most tedious — but two of these are tests that currently prove nothing.

| # | Where | What |
|---|---|---|
| 10 | `tests/test_mister_fritz.py:427` | `test_sub_threshold_tail_is_flushed` is **vacuous**: patching the module constant cannot affect `_DeltaEmitter.__init__`'s default, which binds at import. Patch the constructor instead. |
| 24 | `tests/test_admin_panel.py:628` | The focus-ring guard enumerates three control names, so it cannot catch a new one — it did not catch the suggestion chips or copy buttons. Invert it: find every selector carrying `clip-path`, intersect with focusable elements, assert each is wrapped. |
| 11 | `main_discord.py:264` | `streaming_callback` has no test at all — not its 3-arg arity, not the cross-thread scheduling guard, not restart-bypasses-the-guard. It is a closure inside `on_message`; extract a `_make_streaming_callback` factory to make it reachable. |
| 34 | `tests/test_fritz_utils.py` | Two tests the plan named were never written (`TestIsAdminCanonical`, `test_validate_config_warns_on_legacy_admin_names`). |
| 6 | `tests/conftest.py:63` | The SDXL double-load guard has no test; `conftest` stubs `image_generator` before any import. Needs a `pytest.importorskip("diffusers")`-guarded test. |

**Risk:** none to production. #11 requires a small refactor of `main_discord`, so it may deserve its own commit.

---

## Suggested order

**A → B → D → C → H** first: five commits, all low-risk, no live testing needed beyond the suite. That clears 21 of the 37 and leaves the repo honest about itself.

**Then E → F → G**, each with a live Ollama smoke test, because these three change what the model sees or how the agent is built.

**I last**, or opportunistically — fold each test fix into whichever batch touches the same file, if you would rather not have a tests-only commit.

If you only do one: **Batch A**. Stale documentation is the finding class that costs future-you the most, and it carries no risk at all.
