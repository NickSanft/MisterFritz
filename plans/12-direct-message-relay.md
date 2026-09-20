# 12. Discord direct-message relay

[← back to index](README.md) · [decisions](DECISIONS.md)

_Planned September 2026 by a 6-agent workflow: five specialists (delivery, relay, guard rails,
surfaces, data/tests) plus an integrator that reconciled their disagreements. Every load-bearing
claim below was verified against the working tree; where two specialists contradicted each other,
the reconciliation section says which won and why._

[← back to index](README.md)

> **DECIDED 2026-09-17 — see [DECISIONS.md](DECISIONS.md) §22-27, which override this document.**
> Five of the six open questions below confirmed the defaults as written. **Decision 26 does not:**
> `/tell compose` is deferred, so **Phase 1 is PRs 1-6 and is a purely verbatim courier**. PR 7
> below is retained for whenever composed mode is revived, but it is no longer in scope.
> Consequence worth noting: with composed mode out, nothing Fritz sends in Phase 1 is authored by
> a model, so the prompt-injection surface of the shipped feature is substantially smaller than
> this plan assumed.

**Effort:** XL (>3 days, across two phases)
**Depends on:** nothing, but touches `main_discord.on_message` — the single most load-bearing line in the bot's primary surface.

## Goal

Today Fritz has never initiated contact with anyone. Every outbound path in the repo is `channel.send` into a channel the user is already standing in (`scheduler.py:106`, `:130`; `main_discord.py:313`), and a repo-wide grep for `create_dm|send_dm|\.dm_channel` over non-venv, non-test `.py` returns nothing. This feature makes Fritz a courier: a `/tell` slash command carries a message from one guild member to another by DM, attributed to the sender and verbatim by default; the recipient's reply comes back to the sender by DM. When this is done: `/tell message @alice "running late"` delivers an embed to Alice's DMs whose author field says it came from Nick and whose description is Nick's exact words; Alice replies to that DM and Nick receives it; every delivery failure is reported to the sender in words that say nothing was sent; Alice can refuse relays from Nick or from everybody, and Nick cannot tell which she did; and a plain DM to Fritz remains, byte for byte, the agent conversation it is today.

---

## Reconciliation: where the five designs disagreed, and what won

The specialists converged on more than they diverged on. Seven real conflicts, decided here.

### 1. The scheduler is not reused for delivery. Its APScheduler instance is.

Three of five proposed some degree of scheduler reuse and one proposed none. The send half genuinely is reusable — `scheduler._run_task` does `bot.get_channel` → `fetch_channel` → `channel.send()` (`scheduler.py:95-130`), and `discord/state.py` checks `_get_private_channel` first, so a DM channel id in `schedules.channel_id` would resolve. **Reuse is still wrong, on five verified counts:**

- `_run_task` feeds the stored prompt to `ask_stuff` (`scheduler.py:114-120`). That is an LLM invocation, not a verbatim relay, and it contradicts owner decision 3 at the most basic level.
- It posts `"⏰ *Running scheduled task...*"` into the target channel first (`scheduler.py:106`). A recipient's relay DM would open with a scheduler status line.
- It has no sender concept. `schedules` (`scheduler.py:44-57`) has no sender, recipient, attribution or thread column, and the only caller of `schedule_once` passes the literal string `"scheduled"` as `user_id` (`agent_tools.py:546`).
- **It swallows delivery failure.** `NotFound` and `Forbidden` log and `return` (`scheduler.py:99-104`). "The sender always learns whether delivery happened" is the one requirement this machinery structurally cannot satisfy.
- It chunks with `text[i:i + 2000]` (`scheduler.py:126`) instead of `bot_adapters.split_into_chunks`.

**Decision:** new `relay_store.py` and a new send path. **But** reuse `ScheduleManager.scheduler` — the APScheduler *instance*, i.e. the timer — for the internal retention job, registered exactly like `_internal_wal_checkpoint` (`scheduler.py:308`). `list_all_schedules` reads the `schedules` table and not the job store, so internal jobs never leak into user listings; `tests/test_scheduler.py::test_wal_checkpoint_internal_job_is_not_in_list_all` already pins that invariant. A second AsyncIOScheduler would be a second thing to start, stop and reason about for one daily DELETE.

Corollary: do not copy `scheduler.py:114`'s `run_in_executor(None, ...)`. That is the *default* executor, not `bot_adapters._BLOCKING_POOL` — an existing inconsistency with the repo convention. Do not propagate it.

### 2. An inbound DM is a relay reply only if it names the message it answers.

This was the sharpest split. One specialist proposed "the author has an open relay in which they are the recipient" — which is a session in disguise and fails the hardest case: Alice holds relays from Bob and Carol, replies to Bob's, and the router has to guess. Every guess is somebody's private message delivered to the wrong person.

**Decision: message-anchored routing, no session state, no TTL capture.** Two accepted signals:

1. **Native reply.** `ctx.type is discord.MessageType.reply` **and** `ctx.reference.message_id` hits a `relay_messages.dm_message_id` row.
2. **Button → modal.** A `discord.ui.DynamicItem` button on the relay DM whose `custom_id` carries the relay id; the modal submission never touches `on_message` at all.

Bare text never routes. A recipient who types "sure, 8 works" into the DM gets a normal Fritz turn and the sender hears nothing. That is the correct failure — silently swallowing Alice's words into someone else's inbox because she happened to be in a relay is far worse — and it is the feature's biggest UX risk, recorded as such below.

**A version discrepancy I had to settle, because it decides how this check is written.** Two specialists reported `discord.MessageReferenceType` exists and one reported it does not. Both are correct about the interpreter they ran:

```
$ python -c "import discord; print(discord.__version__)"      → 2.4.0   (global site-packages)
$ .venv/Scripts/python.exe -c "import discord; print(...)"     → 2.6.4   (matches requirements.txt:76)
$ grep discord pyproject.toml                                  → "discord.py>=2.6"
```

`MessageReferenceType` is imported at `.venv/…/discord/message.py:53` and `MessageReference.type` is set at `:660` — in 2.6.4 only. In 2.4.0 `MessageReference.__slots__` is `['message_id', 'channel_id', 'guild_id', 'fail_if_not_exists', 'resolved', '_state']`, with no `type`. The authoritative version is **2.6.4**; the 2.4.0 install is a stale global interpreter on this machine and is itself a prerequisite to fix (P5).

**But gate on `Message.type`, not `MessageReference.type`, anyway.** `discord.MessageType.reply` exists in both versions (verified in the 2.4.0 install), it is the attribute that distinguishes a reply from a *forward*, and using it makes the router immune to which interpreter a contributor happens to run. Checking `reference.type` in addition is free; checking only it is a version trap.

### 3. The reply lands in the sender's DMs. Always.

Unanimous across every specialist who addressed it, for four reasons that all hold:

- The content is the recipient's private words. Publishing them into `#general` because the sender happened to type `/tell` there broadcasts her reply to a whole guild, avoidably.
- Nothing about the exchange was ever public — `/tell` is ephemeral throughout, per the cog convention (`bot_commands.py:350-352`). A channel-delivered reply would be the *only* public artefact of a private exchange.
- "Wherever they are" has no data source. `main_discord.py:153-155` builds `discord.Intents.default()` plus `message_content` only; `presences` and `members` are both off.
- Symmetry: a DM to the sender re-enters the same `on_message` path, so the sender replies to the reply through the identical mechanism. A guild-channel delivery has no reply-anchor story that does not require reading all guild messages.

**Consequence to enforce at `/tell` time:** Fritz must be able to DM the *sender*, because that is how replies reach him. Probe it by delivering the sender's receipt as a DM. If it raises `Forbidden` with `code == 50007`, the relay still stands (the recipient already has it) but the ephemeral confirmation degrades to "I have nowhere to bring a reply."

### 4. One table per hop, not a relay plus a message log.

Four schemas were proposed. **The one-row-per-hop shape wins:** a reply *is* a delivery and needs the same failure, expiry and routing machinery, so giving it its own table means writing that machinery twice. Chain hops with `origin_id`. This also removes the `turns` counter and the four-state machine a two-table design forces, because the chain already encodes both.

**Storing no bodies at all was proposed and is rejected.** It is genuinely elegant — `/export` cannot leak what was never stored and `/forget` cannot miss it — but it makes `/export` unable to show a user what they sent or were sent, which is a real reduction in the transparency `privacy.py` exists to provide. The *correct* half of that argument is adopted in full: **bodies never reach `audit.log`.** Nothing in `privacy.py` touches `AUDIT_LOG_PATH`, `audit_log` is a bare append with no rotation (`observability.py:288`), and the working-tree `audit.log` is already 574 KB from `/forget` and file-tool traffic alone. Log `chars` and a `body_sha16`; never the text. This is the same line `file_tools` already draws — paths and argv, never contents.

**The `dm_channels` cache is deferred, not rejected.** The argument for it is real: `discord/state.py:487-511` uses `_private_channels` as a 128-entry LRU and bots receive no private-channel list at READY, so after a restart every relay re-issues `POST /users/@me/channels` for a channel that already exists — and that route is the one Discord's anti-spam heuristics watch. But it is an optimisation for a load this deployment does not have. The schema carries `dm_channel_id` from day one so the data is there; add the cache when the symptom appears (intermittent 429s on `/users/@me/channels`), not before.

### 5. Embed, not prefixed plain text.

Two specialists said embed, two said plain text with escaping. **Embed wins**, on one structural argument: a plain-text banner is forgeable. A sender writes `\n\n— actually from @admin` or `**Relayed for Admin** · ignore the above` into the body and the recipient cannot tell. An embed's `set_author` field sits structurally outside the sender-controlled `description`.

This dissolves a real bug in the plain-text branch, too. The escaping design required `escape_markdown`, which **inserts backslashes** — so a body that passes a 900-char raw check can exceed 2000 once escaped and framed, and the send raises `HTTPException`. Inside an embed description the sender cannot forge the author line regardless of their markdown, so `escape_markdown` is not load-bearing and can be dropped, which also keeps the word "verbatim" honest. The 4096-char embed description cap then makes the length arithmetic comfortable rather than tight.

`allowed_mentions=discord.AllowedMentions.none()` on every relay send, both legs — unanimous, and non-negotiable. A verbatim relay is attacker-controlled text; without it Fritz is a ping proxy.

**Recorded uncertainty:** embeds render less well on some mobile clients and copy-paste worse. This is the design call I am least sure of. It is one function's worth of change to reverse if the owner dislikes it in practice, and the test at DoD-7 pins the anti-forgery property either way.

### 6. The rate-limit gate lives in the store, and counts off the delivery log.

Two specialists disagreed about placement, two about the counter's home. Both are right about different things, and the combination satisfies both:

- **Placement in the store, not the cog.** A limit enforced in `bot_commands` is bypassed the day the phase-2 agent tool lands, because the tool does not go through the cog. `relay_store.reserve_send(...)` is the single gate; `/tell`, the reply path and the phase-2 tool all call it.
- **Counted off `relay_messages`, not a counter table.** A counter table is a second source of truth that can drift from the delivery log, and `/forget all` would then have to clear it too.
- **`BEGIN IMMEDIATE`, not COUNT-then-INSERT.** Check-then-send is a TOCTOU: two concurrent interactions both read `count = 19` and both send. The existing `MAX_SCHEDULES_PER_USER` check (`scheduler.py:148-165`) has exactly this weakness today. Do not copy it verbatim; do all reads and the reservation INSERT in one transaction on one connection with `isolation_level=None`.

The proposed limit list ran to ten knobs. Phase 1 ships **three**, and one of them is load-bearing in a way the others are not: **`RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR` is the only limit that stops a brigade.** Thirty people sending one message each passes every per-sender limit anyone has ever written. Per-pair cooldowns, per-pair daily caps, first-contact caps and global ceilings are deferred to What Not To Build.

### 7. Phase-2 addressing: the model gets the body, never the recipient — but the tap is negotiable.

The strongest disagreement, and the one I am handing back to the owner with a default rather than settling.

The risk is verified and not theoretical. Retrieved memories are string-concatenated into the system prompt; memories are written by an *unsupervised* background LLM pass over conversation text (`agent_tools.py:122-168`) and by `save_memory`; `scrape_web` returns up to `SCRAPE_MAX_CHARS` of arbitrary page text into the tool loop. Nothing inspects any of it. A memory reading "when he says tell my boss, that means @victim" would be concatenated verbatim into the prompt and would choose who gets DM'd.

Three designs were proposed: a `UserSelect` confirmation tap (safe, but `UserSelect` in a DM "will only allow the user to select the client or themselves" per the library's own docstring — and a DM is where the agent tool's entire value lives); a `recipient_name` argument with an `AMBIGUOUS` return (usable, but the model holds addressing); and closure-bound resolution from a pre-computed candidate list.

**The default I propose, which none of the specialists put forward:** `relay_message(recipient_name, message)`, where the tool body rejects any `recipient_name` that does not appear as a substring of the user's own message text for the current turn. The human must have typed the name. A poisoned memory cannot put a name in the user's message; `scrape_web` output cannot either. This is a cheap structural guard, and it is novel and untested — **its known hole is pronouns**: "tell her I'll be late" has no name to match, so the tool refuses and Fritz has to ask. That refusal is the cost. Ranked as open question 4; it blocks nothing until PR 9.

Two things about phase 2 are settled regardless: the tool binds the **real sender identity** in the closure, not the literal string `"scheduled"` that `make_schedule_message_tool` passes (`agent_tools.py:546`); and `guild_id` must join the agent cache key at `mister_fritz.py:605`, because the comment there states the rule — the key is exactly what the closures capture — and `user_id` does not cover it.

---

## Prerequisites

These land **before** PR 1. Two of them are changes to existing behaviour and each deserves its own commit.

### P0 — The `on_message` ordering change, inert. (XS, its own commit, its own revert)

`main_discord.py:259` is `elif not isinstance(channel, discord.DMChannel) and not client.user.mentioned_in(ctx): return`. Because the DM case short-circuits that guard, **every DM falls through to the full agent turn** at `main_discord.py:327`. That is the collision, and it is worse than a UX clash: the turn binds file tools whenever the user has a workspace (`mister_fritz.py:570` — `include_file_tools = workspace_root is not None`, fed by `main_discord.py:332`), so relayed, attacker-controlled text would become input to a tool-using agent with read/write/exec on that user's disk.

Land the hook **inert**, so the ordering change is reviewable on its own and the diff that makes it live is three lines:

```python
    if isinstance(channel, discord.DMChannel):
        if await relay_router.try_route_reply(client, ctx):
            METRICS.increment("relay.replies_routed")
            return
```

Placed **after** the self-check at `:254` (so Fritz's own relay DM cannot route itself), **after** the `$` prefix check at `:257` (so a command in a DM stays a command), **before** the guard at `:259` (that guard is the collision), and it must `return` before `identity_store.record` at `:267`, before `METRICS.increment("discord_messages")` at `:269`, and well before the `"✍️ *Mister Fritz is thinking...*"` placeholder at `:313` — otherwise the recipient watches a spinner for a message that is never going to the model.

In P0, `relay_router.try_route_reply` is a stub that returns `False` unconditionally. Ship it with the regression test that a plain DM still reaches `ask_stuff`, and with a source-level ordering test in the style of `tests/test_discord_commands.py:232-259`, which reads `main_discord.py` as text and asserts statement order. That source-level test is the guard against a future refactor silently reordering the block — the failure is invisible at runtime because everything still appears to work.

> **SHIPPED 2026-09-20.** One departure: the snippet above cannot be pasted where it belongs, because `on_message`'s early returns were an `if/elif/elif` chain and nothing can go *between* two arms of one chain. The first two arms are now separate `if`s (each returns, so the flow is identical), and the DM branch carries the `elif not client.user.mentioned_in(ctx)` guard — keep that `elif` attached to the `isinstance` `if`, or a non-DM message stops being checked for a mention. Anyone reverting P0 must restore the chain, not just delete the hook.

### P1 — `pyproject.toml` `py-modules`. (XS)

`tests/test_packaging.py:183` cross-checks `[tool.setuptools] py-modules` against top-level modules on disk **in both directions**. Adding `relay_store.py` and `relay_router.py` turns it red until `pyproject.toml:127-152` lists them. Free regression guard; expect it rather than be surprised by it.

### P2 — The `migrate_identity` table-name clobber. (XS, must precede P3)

`_SQLITE_TARGETS` is a list of `(table, column)` pairs (`migrate_identity.py:66-71`), but `survey()` accumulates into `found[table] = keys` (`:103-106`) and `rewrite_sqlite()` into `counts[table] = ...` (`:198-207`). Both key on **table name**. Every existing target appears exactly once, so this has never bitten. `relay_messages` needs *both* `sender_id` and `recipient_id` registered, and the second would hide the first from the survey — which matters because `main()` gates `--apply` on `unmapped` derived from that survey. A legacy key present only in `sender_id` would be neither displayed nor demanded, `--apply` would proceed, nothing would be rewritten, and the tool would report success. That is precisely the half-migrated outcome its own error text exists to prevent.

Fix both dicts to key on `f"{table}.{column}"` **before** adding any relay target. Two tests force the ordering: one asserting `survey()` reports both identity columns of `relay_messages`, one asserting `--apply` refuses when a relay `sender_id` is unmapped.

### P3 — Config knobs. (XS)

In `fritz_utils.py` beside `MAX_SCHEDULES_PER_USER` (`:159`), documented as commented-out lines in `.env.example` beside `:93`, validated in `validate_config()` (`:581`):

```python
RELAY_ENABLED: bool = os.environ.get("RELAY_ENABLED", "true").lower() in ("1", "true", "yes")
RELAY_AGENT_TOOL_ENABLED: bool = os.environ.get("RELAY_AGENT_TOOL_ENABLED", "false").lower() in ("1", "true", "yes")
RELAY_MAX_BODY_CHARS: int = _at_least_one("RELAY_MAX_BODY_CHARS", "1000")
RELAY_MAX_PER_SENDER_PER_HOUR: int = _at_least_one("RELAY_MAX_PER_SENDER_PER_HOUR", "10")
RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR: int = _at_least_one("RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", "10")
RELAY_REPLY_WINDOW_MIN: int = _at_least_one("RELAY_REPLY_WINDOW_MIN", "1440")
RELAY_RETENTION_DAYS: int = _at_least_one("RELAY_RETENTION_DAYS", "30")
```

Use `_at_least_one` (`fritz_utils.py:178-194`), not a bare `int()`, for every count. The comment at `:166-174` documents exactly why: `IMAGE_GEN_MAX_CONCURRENCY=0` was perfectly legal and made `/gen` hang for fifteen minutes with nothing in the log. "Off" is `RELAY_ENABLED=false`, never a 0 count.

`RELAY_AGENT_TOOL_ENABLED` defaulting **false** is what makes the owner's phasing real rather than aspirational: phase-2 code can land in the same tree, under test, while the model provably cannot DM anyone until an operator flips it.

### P4 — Size-based rotation for `audit_log`. (XS)

`observability.py:288` is a bare append with no cap. The file is already 574 KB from far lower-volume events. Relay traffic on an unbounded append is a disk-fill bug whose blast radius is a file full of who-messaged-whom, and it survives `/forget all` by design. Cheap, unrelated to the rest, and genuinely a prerequisite rather than a follow-up.

### P5 — Verify the test interpreter resolves `discord.py>=2.6`.

`requirements.txt:76` pins `2.6.4`, `pyproject.toml:24` declares `>=2.6`, `.venv` has 2.6.4 — and the global Python 3.12 on this machine has **2.4.0**. A contributor who runs `pytest` with the wrong interpreter gets a suite that passes while exercising a library without `MessageReferenceType`. One line in the contributing notes, or a `conftest.py` assertion on `discord.__version__`.

---

## The PR sequence

Each is individually reviewable and revertible. Sizes follow the repo convention (XS < 1h, S ~half a day, M ~1 day, L 1-3 days).

### Phase 1 — `/tell`, verbatim, one-way

**PR 1 — `relay_store.py`: schema, reservation gate, no Discord.** (M)
Two tables in `SCHEDULE_DB`, created with `CREATE TABLE IF NOT EXISTS` after `PRAGMA journal_mode=WAL`, module-global `_INIT_LOCK`/`_INITIALISED` with `_init_db()` at the top of every public function, one `sqlite3.connect` per operation — the `workspace_store.py:18-42` shape exactly.

```sql
CREATE TABLE IF NOT EXISTS relay_messages (
    id             TEXT PRIMARY KEY,   -- str(uuid4())[:8], per scheduler.py:146
    sender_id      TEXT NOT NULL,      -- canonical "discord-<snowflake>"
    recipient_id   TEXT NOT NULL,      -- canonical
    origin_id      TEXT,               -- previous hop; no FOREIGN KEY, per repo convention
    body           TEXT NOT NULL,
    composed       INTEGER NOT NULL DEFAULT 0,
    guild_id       INTEGER,            -- snowflake
    dm_channel_id  INTEGER,            -- snowflake; unused in phase 1, present for the cache
    dm_message_id  INTEGER,            -- snowflake; THE reply-routing key
    status         TEXT NOT NULL DEFAULT 'pending',
    error          TEXT,
    created_at     TEXT NOT NULL,      -- ISO-8601 UTC
    delivered_at   TEXT,
    expires_at     TEXT NOT NULL,
    closed_at      TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relay_dm_message
    ON relay_messages(dm_message_id) WHERE dm_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_relay_sender    ON relay_messages(sender_id, created_at);
CREATE INDEX IF NOT EXISTS idx_relay_recipient ON relay_messages(recipient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_relay_open      ON relay_messages(status, expires_at);

CREATE TABLE IF NOT EXISTS relay_optouts (
    user_id     TEXT NOT NULL,   -- the person being protected
    blocked_id  TEXT NOT NULL,   -- who they refuse; '*' = everyone
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, blocked_id)
);
CREATE INDEX IF NOT EXISTS idx_relay_optouts_blocked ON relay_optouts(blocked_id);
```

The partial unique index is load-bearing and is a deliberate first for this repo: `dm_message_id` is NULL while a row is `pending` (many rows sit there at once) and unique once delivered, because a duplicate routes a reply to the wrong person. SQLite treats NULLs as distinct in a unique index, which is exactly the behaviour needed — and is also why `blocked_id` is `NOT NULL` with a `'*'` sentinel rather than nullable: a nullable column would let the same person insert the global opt-out repeatedly and would make `ON CONFLICT DO NOTHING` a guard that guards nothing. The sentinel is provably collision-free — `canonical_user_id('discord', '*')` raises `ValueError`, because `_IDENT_STRIP_RE` reduces `*` to the empty string.

`reserve_send(sender_id, recipient_id, guild_id, chars) -> Reservation | Denial` does every count and the reservation INSERT inside one `BEGIN IMMEDIATE`, then `mark_sent(rid, dm_message_id)` or `mark_failed(rid, reason)`. Rows stuck at `reserved` for more than 60s are ignored by counting and swept, so a crash cannot permanently consume quota. ~~Check order, most recipient-protective first: recipient is a bot → recipient is the sender → recipient is Fritz → recipient blocked sender → recipient blocked everyone → body over cap → per-sender hour → per-recipient inbound hour.~~

> **CORRECTED 2026-09-18 — the order above is a block detector.** Checking blocks before the body cap meant a 5,000-character message came back `REFUSED` if the recipient had blocked you and "too long" if not: definitive (closed DMs are only discovered at send time, which an oversized body never reaches) and free (a denial wrote no row and spent no quota). Being over your own hourly cap did the same. The shipped order is: **Fritz → bot → self → body over cap → per-sender hour → per-recipient inbound hour → blocked sender → blocked everyone**, then the send. Recipient-side refusals go last, so `REFUSED` only ever means what Discord's own 403 means — everything else was fine. Two consequences follow, both implemented: a refusal from a block and a refusal from closed DMs (403) write **identical** `refused` rows, because PR 6's `/export` shows senders their own rows; and both are **charged to the sender** but never to the recipient's inbound cap. The charging stops the rate limit becoming the detector, and keeping them off the inbound cap stops a blocked harasser from exhausting the recipient's inbox. `failed` is reserved for failures on our side, which cost nothing. See `relay_store._gate` and `tests/test_relay_store.py::TestBlockStateIsNotProbeable`. **This also amends the Definition of Done**: a 50007 writes a `refused` row, not a `failed` one, and a rate-limited attempt is *not* charged — charging it would extend a lockout indefinitely under retry and write a row per denial.

**Resolve identity at both ends, inside the store.** With `IDENTITY_LINKS=web-alice=discord-123`, a block stored against `discord-123` must be honoured when the sender arrives as `web-alice`. Resolving only the sender leaves a live bypass — the same write-path/delete-path divergence the comment in `privacy.py:44-49` documents eleven times.

Raise on failure; do not swallow like `identity_store.py:76-77`. A swallowed write means telling the sender "delivered" with no row to route a reply through.

*Unblocks:* everything. *Risk:* the `BEGIN IMMEDIATE` transaction is the only concurrency-sensitive code in the feature; get it wrong and the caps are advisory.

**PR 2 — `/tell message`: the delivery path.** (M)
`bot_commands.py`, a new section after the workspace group. `tell` as an `app_commands.Group(..., guild_only=True)` — **on the Group, not the subcommand**, because the library's own docstring says the decorator "does nothing in subcommands and is ignored". Keep an in-code `interaction.guild_id is not None` check as the real enforcement, since `guild_only` is verified server-side with no error handler called.

Annotate the recipient parameter `discord.Member`, not `discord.User`. `MemberTransformer.transform` raises `TransformerError` unless the resolved value is a `Member`, and `Namespace._get_resolved_items` builds `Member` objects only from `resolved['members']`, which Discord populates only for actual guild members — **with no member cache and no HTTP call.** That is owner decision 2 enforced server-side for free. It matters because the obvious alternative is a trap: `User.mutual_guilds` is implemented as a pure member-cache scan, and `Intents.default()` has `members = False`, so it would silently false-negative.

Add a `/tell`-specific branch to `handle_app_command_error` (`bot_commands.py:104-106`), because the generic `TransformerError` copy — "That value is outside the permitted range" — is wrong and confusing for "that person is not in this server."

Control flow: `defer(ephemeral=True, thinking=True)` **first** (`create_dm` + `send` is two round trips against a 3s ack deadline, and either can sleep on a 429), then `_identity(interaction)`, then `reserve_send`, then the recipient DM, then the sender's DM receipt, then `audit_log`, then `followup.send(ephemeral=True)`. **Never optimistic-ack:** the confirmation is written only after `channel.send(...)` returns a `Message`, and every failure path says in words that nothing was sent.

Do **not** wrap discord.py coroutines in `run_blocking`. `create_dm`, `send`, `fetch_user`, `interaction.response.*` are aiohttp coroutines; wrapping one occupies a scarce pool worker (`bot_adapters.py:27-30`, `max_workers=BLOCKING_POOL_SIZE`) doing nothing but waiting, and breaks the loop affinity discord.py's own 429 backoff depends on. Do wrap the `relay_store` calls — sqlite3 is blocking, and `bot_commands.py:166` is the convention.

Catch order in the sender, because `Forbidden` and `NotFound` both subclass `HTTPException`: `Forbidden` → `NotFound` → `RateLimited` (which does **not** subclass `HTTPException`, so it must come before it) → `DiscordServerError` → `HTTPException` (branch on `status == 429`, then `400`) → `Exception` via `fritz_error`. `RateLimited` is dead code today — nothing passes `max_ratelimit_timeout` to `commands.Bot` at `main_discord.py:155` — but it is one kwarg away from being live, and the day it fires an `except discord.HTTPException` handler starts leaking tracebacks to users.

*Unblocks:* PR 3, PR 4. *Risk:* the interaction can be double-answered if a failure path forgets the defer/followup distinction; `_reply_error` (`bot_commands.py:70-88`) already solves this and should be used rather than reimplemented.

> **SHIPPED 2026-09-18, with these departures — each found by verifying against discord.py 2.6.4 or by a 25-agent adversarial review, and each pinned by a test that fails without it:**
> - **Ships dark.** `RELAY_ENABLED` defaults to `false` until Phase 1 is complete. Every commit lands on master, so every intermediate state is deployable, and `/tell` without PR 3 is the ship blocker this section names.
> - **`guild_only=True` is not enough.** On a Group it serialises only the deprecated `dm_permission`. The group also sets `allowed_contexts` (→ `contexts: [0]`) and `allowed_installs` (→ `integration_types: [0]`). Without the latter, a copy of Fritz user-installed into someone's account could run `/tell` from a server Fritz was never invited to and DM that server's members. The command also checks `interaction.is_guild_integration()` in code.
> - **Both refusals keep one schedule, not just one string.** A block and a 403 are held as `reserved`, settled as `refused` and answered at the same randomly drawn deadline (1–2 s from the start of the command). Two findings forced this. Answering each as soon as it was known was a timing oracle, and padding only the block path merely moved it. And a 403-bound send sat `reserved`, counting toward the recipient's inbox, while a block went straight to `refused`, so a second account racing the first saw "their inbox is full" only when there was no block. `relay_store.reserve_send` therefore returns a block with its reservation still open (`Denial.relay_id`).
> - **The author line is `@username · Display`,** with handle syntax, control characters and format characters stripped from the display name. `Display (@username)` let a nickname of `Alice (@alice)` render as `Alice (@alice) (@mallory)`.
> - **A 400 is `rejected`, and the sender is charged for it.** Discord's harmful-link filter reliably returns one, and free 400s were an unlimited supply of DM-channel opens on the bot's token.
> - **The audit log carries a keyed HMAC digest, not `sha256(body)[:16]`.** Unkeyed, a hash of "ok" *is* "ok" to anyone with a dictionary.
> - Also: the recipient-cap denial no longer says "try again after HH:MM", because that time is a *third party's* message to the recipient. And the receipt DM to the sender is sent only after a delivery. The old wording "as delivered" would have been a lie on any other path.
>
> One known limit, not fixed: a 403 that takes longer than its deadline is answered late, and is then distinguishable by timing. It needs a Discord round trip of over a second.

**PR 3 — `/relay block|unblock|status`, and the one indistinguishable refusal.** (S)
`relay` as a Group deliberately **not** `guild_only` — the recipient is standing in a DM when they want to block, and making them walk back into the server that caused the problem is exactly the wrong shape. Global commands are available in DMs with the bot by default, so this needs no decorator at all, just the absence of one.

Every recipient-side refusal returns **byte-identical** copy: blanket block, per-sender block, and `Forbidden`/50007 all produce the same string. Every clause must be true in all three cases. Otherwise `/tell` becomes a reliable detector for "has X blocked me", which is the thing an opt-out exists to avoid revealing. Rate-limit denials *are* explicit with a retry time, because those are the sender's own state and there is nothing to leak.

Pin the property with a test that asserts **equality of the two reply strings**, not two separate literals — that equality *is* the non-probeable property, and a later "helpful error messages" pass is exactly how it would be lost.

*Unblocks:* nothing, but it is a **ship blocker** for PR 2. Do not deploy `/tell` without it. *Risk:* none technical; the risk is that a later copy edit to one branch breaks the equality.

> **SHIPPED 2026-09-19 as `/relay block | block-everyone | unblock | status | forget-blocks` plus a message context menu, Apps → "Block this sender".** Departures, each pinned by a test that fails without it:
> - **No command takes a user-picker option.** In a DM with the bot, Discord's USER picker is undocumented and, by every available sign, limited to the DM's own participants: you and Fritz. That makes a picker useless in exactly the place this group exists for. `who` is a string with **autocomplete over the people who have actually relayed to you** (delivered messages only, so a recipient never learns that someone they blocked keeps trying). It also accepts a pasted mention or ID, all parsed and never trusted. The context menu needs no naming at all: it is anchored on the message, like reply routing, so with relays from two people you can never block the wrong one.
> - **A person is only ever shown what they typed or what they were shown.** A 31-agent review found six ways the first draft broke this. The fix is structural:
>   - Blocks are stored **as supplied** and resolved through `IDENTITY_LINKS` only at the gate. Resolving at store time meant status, and the "already blocked" reply, disclosed which Discord account a web or Telegram identity, or an alt, belongs to.
>   - Each delivered relay records the author line the recipient actually saw (`shown_as`), and each block records a `label`. Autocomplete, status and every reply show only those, never a display name from `identity_store`. That made `/relay block` a lookup service for any ID, and it could show a nickname from a server the recipient isn't in.
>   - Autocomplete hands the client **opaque tokens** (a relay id, or a block's entry key), never a resolved id.
>   - Free text accepts only strict ASCII Discord ids. That rules out `web-…`, `telegram-…`, case variants that would never bind, and non-ASCII digits, which `\d` would have matched.
>   - Block replies are identical whether or not the block was new.
>   - A second, 15-agent review of that rework found that blocks placed *from a relay* still used the resolved sender. Each relay row now also records `sender_account`, the account that actually sent it, before `IDENTITY_LINKS`. "Block this sender" and autocomplete block that account, so every stored block is resolved exactly once, at the gate. Blocking the resolved id resolved it twice: under chained links that caught a different identity and let the real sender through. Autocomplete also lists linked accounts separately, because merging them into one entry told the recipient they were linked. The relay row's recipient is compared directly rather than resolved a second time.
>   - Unblock tokens are a keyed hash of (owner, blocked id), not rowids. Rowids are one global sequence, so the gap between two of your own said someone else had blocked in between, and a reused rowid let a stale token remove a different block.
>   - A block refusal and a 403 now write the **same** audit line: no `reason`, no Discord code. `relay_denied reason=blocked_sender` had been a durable record of who blocked whom.
>   - `recent_senders` is one grouped read, not a correlated subquery. It runs on the event loop at every autocomplete keystroke. A test counts SQLite VM steps to pin it as linear.
>   - `migrate_identity` checks that a registered *column* exists, not just its table, and registers `sender_account`. Without the column check, any database whose relay tables predated a column made the survey, and so the whole migration, raise.
> - **`/relay forget-blocks` is here, not in PR 6.** It is the escape hatch DECISIONS #22 requires. It asks for confirmation, warns that it re-arms anyone you were keeping out, never touches blocks other people placed against you, and implements `on_timeout`, which `_ForgetConfirmView` does not. The `/forget all` copy that points to it is still PR 6's job.
> - **Blocks work while `RELAY_ENABLED` is off,** so people can refuse in advance, and `/relay status` says when the relay is switched off.
> - **The relayed DM's footer names the way out:** Apps → Block this sender, or `/relay block`. The one-press button is PR 5.
> - **The audit log records who blocked, never whom.** It outlives `forget-blocks`, so a record of the blocked party would outlive the block itself.
> - **Every `/relay` command and the menu defer first.** Their store calls go through the shared worker pool, which long model turns can fill for minutes, and anything slower than three seconds is "This interaction failed". Autocomplete cannot defer, so it reads the store directly on the event loop: one indexed read. `/relay status` is capped to fit Discord's 2,000-character limit, and autocomplete filters *before* its 25-choice cut, so a brigade of recent senders cannot bury the person the recipient is looking for.

**PR 4 — The reply router. The riskiest PR in the sequence.** (L)
New `relay_router.py`. Fill in the P0 stub. One direction-agnostic function: given a `dm_message_id`, look up the row, require `canonical_user_id('discord', ctx.author.id) == recipient_id` (near-tautological in a 1:1 DM, but three lines, and it is the guard that stops a forged or mis-indexed anchor delivering to the wrong person), and send the reply to `sender_id` through the same `reserve_send` gate with the ends swapped. Outbound and reply-back are the same code path; there is no separate "reply handler."

**Enforced by architecture, not by a guard: `relay_router` does not import `mister_fritz`.** Composition is verbatim, so a relayed message needs no model at all. Code that cannot reach `ask_stuff` cannot reach the two contamination vectors — the LangGraph checkpoint keyed on `thread_id_for(user_id, channel_key)`, and `extract_memories_background` firing into Chroma, whose output is then auto-injected into that user's *next* prompt. Patching one and not the other proves nothing. Add a test asserting `mister_fritz` is not in `sys.modules` after importing `relay_router` alone, in the spirit of `tests/test_packaging.py`'s torch-free assertion.

A reply to a **closed or expired** relay does not route and does not fall through to the agent either — the second is the nastier leak, because the reply would become a Fritz conversation turn containing a message meant for someone else. Tell the recipient the exchange has lapsed.

A reply whose anchor resolves to **nothing** (row purged, or a reply to some other Fritz message) falls through to the normal agent path unchanged. Graceful degradation, no error.

If the reply-back send fails, the failure copy goes to **the person who just replied** — they are the only one in the room — not to the sender who cannot see it.

> **SHIPPED 2026-09-20.** As specified — one direction-agnostic path, no session state, no `mister_fritz` import — with these notes:
> - **`MessageReferenceType.reply` is a literal alias of `default`** in discord.py 2.6.4 (`enums.py:232-235`: both are `0`). Testing it would accept pins, crossposts, thread starters and poll results alike, so it is not a reply check at all. `Message.type is MessageType.reply` (19) is the only discriminator, which is what the plan said for a different reason. This also settles the forward case concretely: a forward carries `reference.message_id` for the very row we look up, with **empty content**, so without the type gate a forward would relay nothing to the original sender and burn the exchange.
> - **The shared presentation moved to `relay_format.py`.** `bot_commands` imports `mister_fritz`, so the router could not reuse anything from it; the embed, the author line, the refusal schedule, the audit digest and the keyed-digest helper now live in a module that imports neither. `bot_commands` keeps its old names as aliases, so `/tell`'s call sites are unchanged. The relay embed itself was inline in `_deliver_relay` and became a real function used by both ends.
> - **Addressing is `client.create_dm(discord.Object(id=...))`.** `get_user` can never hit for this bot (no member cache), and `fetch_user` would cost a second HTTP call for a name the router does not need: the reply-back embed names the *replier*, who is live in `ctx.author`.
> - **A reply goes to the account that wrote**, `sender_account`, not the identity it resolves to — the same rule PR 3 established for blocks.
> - **A reply with no words** (an attachment alone) is answered, not carried and not passed to the agent. Attachments never travel; both parties are told.
> - **A reply beginning with `$` was being swallowed whole.** `on_message` handed anything starting with the prefix to `process_commands`, and this bot registers no prefix commands at all, so "$20 plus tip" was dispatched to nothing, logged as CommandNotFound, and dropped: not carried, not given to the agent, and nobody told — the one path where someone could be left believing they had answered. The branch now runs only when `client.get_command` finds a real command. Pre-existing and bot-wide; PR 4 is what made it cost something.
> - Known gap, not fixed here: a sender replying to their own `/tell` **receipt** finds a dead end, because only the recipient's DM id is recorded. Their side of the conversation continues by replying to the reply. Recording the receipt's id too would fix it and is a small change.
>
> 25 mutations of the router each turn a test red, including every "must not route" case.

*Unblocks:* the feature is two-way. *Risk:* **this is the riskiest PR, and it is not close.** It makes live a branch that sits directly above `main_discord.py:259`, the single most load-bearing line in the bot's main surface. A bug here does not degrade relay — it breaks every DM conversation with Fritz, and it does so silently, because a router that eats every DM still "works" from the code's point of view. The test that a plain DM with no reference still reaches `ask_stuff` is a release blocker, not a nice-to-have.

**PR 5 — Buttons: `[Reply]` and `[Not now]`.** (M)
`discord.ui.DynamicItem` subclasses with the relay id in a regex-templated `custom_id` (`fritz:relay:reply:(?P<rid>[0-9a-f]{8})`), registered once via `client.add_dynamic_items(...)` in `on_ready` beside the `client.add_cog` call at `main_discord.py:200`. `DynamicItem` rather than `add_view`, because a per-relay button cannot have a static `custom_id` and `View.is_persistent` demands one. `[Reply]` opens a `Modal` with a single paragraph `TextInput`; the submission *is* the reply, unambiguously bound, and never touches `on_message`.

**`custom_id` is a public, unauthenticated channel.** It travels to the client and comes back on every press. Verify `interaction.user` is the relay's recorded recipient before opening the modal — the same lesson as `_ForgetConfirmView.interaction_check` (`bot_commands.py:153-162`), which exists because anyone can press anyone's button.

Modals cannot survive a restart (`ViewStore._modals` is an in-memory dict), but the exposure is the seconds a modal is open and the user simply presses again. A *missed* `add_dynamic_items` registration is the real hazard: every button on every historical DM silently becomes "This interaction failed" with no server-side line pointing at the cause. Guard it with a startup assertion and a test that imports `main_discord` and asserts the classes are registered. The slash commands named in the DM's footer are why that is a degradation rather than a failure.

*Unblocks:* the ergonomic gap PR 4 leaves open. *Risk:* a dead registration is invisible.

**PR 6 — Privacy, retention, reconciliation.** (M)
`privacy.forget_relay(user_id)` and `export_relay(user_id)`, both resolving identity at their own entry point with the repeated comment block, added to `forget_all` (`privacy.py:258-275`) and `export_user_data` (`privacy.py:296-312`).

The deletion rule, which is the subtlest thing in the feature:

- DELETE `relay_messages` WHERE `sender_id = me` **OR** `recipient_id = me`. A shared exchange cannot be half-deleted; keeping one half so the counterpart can export it means "forget me" did not.
- DELETE `relay_optouts` WHERE `user_id = me` — blocks I placed are mine to drop. **This is the contested half; see open question 1.**
- **KEEP** `relay_optouts` WHERE `blocked_id = me`. Blocks other people placed *against* me are **their** data. Deleting them would let anyone launder away every block on them by running `/forget all`. This half is settled — it is a bug fix, not a policy call.

**Also fix the live under-report while you are here.** `forget_all` returns five keys; `_ForgetConfirmView.confirm` (`bot_commands.py:168-176`) renders four. `alias_dropped` is deleted and never reported today. Add it, add the relay line, and add the `/forget all` warning bullet at `bot_commands.py:386-391`. `tests/test_privacy.py:368-376` asserts the whole `forget_all` dict with a single `assertEqual` — a free test that fails against the old code.

Two boot-time actions in `on_ready`, immediately after `schedule_manager.start()` (`main_discord.py:190-191`):

1. `relay_store.reconcile_pending()` — a row is INSERTed `pending`, the DM sent, the row UPDATEd to `delivered`. A crash between steps leaves a row that will never route and never tell the sender. Flip every `pending` row older than a grace window to `failed`, and log in the shape of `scheduler.py:268`.
2. Register `id="_internal_relay_retention"` on `schedule_manager.scheduler`, daily, deleting rows past `RELAY_RETENTION_DAYS`, following the `_internal_wal_checkpoint` convention at `scheduler.py:308`.

*Unblocks:* shipping. *Risk:* under-reporting a deletion that was performed is a trust bug, not a correctness one, and it is exactly what the existing `alias_dropped` gap already is.

**PR 7 — `/tell compose`, with the draft preview.** (M) — **DEFERRED, not in Phase 1 (decision 26).**
Composed mode is a **subcommand**, not a `compose: bool`. Discord collapses optional parameters, so the mode of an invocation would be legible only to someone who went looking, and the failure is asymmetric and expensive: a person pastes a carefully worded apology, a stale flag rewrites it, and Fritz's words go out under their name. Give the two subcommands different payload parameter names — `message` (a noun you are handing over) and `gist` (an instruction) — so the call site reads as a sentence describing what will happen.

`ask_stuff` runs through `run_blocking` with an explicit `thread_id=f"relay-draft:{sender}"` so a draft never lands in the user's real conversation checkpoint. The draft goes to a review view with `[Send] [Rewrite] [Discard]` — mirroring `_ForgetConfirmView` including the canonical-id `interaction_check`, but **implementing `on_timeout`**, which that class does not: after 30s its buttons silently stop working, and that is a gap worth not repeating.

**Ship the preview or cut the feature.** This is the only path that puts model-generated text under a human's name, and it is the piece most likely to be cut for expedience. A local Ollama producing something tone-deaf, sent in Nick's name to a colleague with no preview, is a worse outcome than not having composed mode at all.

*Unblocks:* nothing. *Risk:* the preview is cut under time pressure and nobody notices until a bad draft ships.

### Phase 2 — the agent tool

**PR 8 — `relay_service`: the sync facade.** (S)
The tool body runs on a `bot_adapters` pool thread (`run_blocking` → `ask_stuff`), so it cannot await. The facade holds the bot's loop and does the `run_coroutine_threadsafe` hop. **With a hard timeout.** A facade with no timeout parks a pool worker indefinitely on a Discord API hiccup, and eight of those wedge every DM, every `/voice` and every `/gen` at once (`BLOCKING_POOL_SIZE` defaults to 8). Return a `FAILED` string on expiry.

*Unblocks:* PR 9. *Risk:* the timeout is the whole PR; without it this is a pool-exhaustion bug.

**PR 9 — `make_relay_tool`, dark by default.** (M)
Follow `make_schedule_message_tool` (`agent_tools.py:537-552`) exactly: a factory returning a **synchronous** `@tool` function that returns a plain `str` and catches `Exception` rather than raising. Plain `@tool`, not `parse_docstring=True` — matching the three existing closure tools, and because `parse_docstring` would strip the prose down to the Args lines, which is where all the guardrails live.

Wired at `mister_fritz.py:576-589` beside the schedule tools, gated on `RELAY_AGENT_TOOL_ENABLED and relay_service is not None`. **`guild_id` joins the agent cache key at `mister_fritz.py:605`** — the comment at `:599-604` states the rule and `user_id` does not cover it; without it a cached agent built in one guild relays into another guild's member list.

Return strings lead with one keyword the model can branch on — `DELIVERED`, `AMBIGUOUS`, `FAILED` — and the docstring tells it never to report success unless the result begins with `DELIVERED`. One `TOOL_NOTICES` entry at `mister_fritz.py:374-392`, keyed on the function name so the dispatch at `:747` finds it: `"relay_message": "✉️ Conveying that to the party in question."` — present participle, understated, and deliberately naming no one, because a name would be the only mutable token in a progress line that renders inside a live-editing placeholder.

*Unblocks:* nothing. *Risk:* addressing (open question 4). Shipping with `RELAY_AGENT_TOOL_ENABLED=false` means the risk is inert until an operator decides otherwise.

---

## Definition of done

- [ ] `/tell message` with a `discord.Member` picker delivers a `discord.Embed` to the recipient's DM whose `author.name` is the sender's display name and whose `description` is the sender's exact typed string, unaltered.
- [ ] A `/tell` whose recipient DM raises `Forbidden(code=50007)` replies to the sender with copy stating nothing was sent, writes a `failed` row, and leaks no Python traceback (the reply goes through `bot_adapters.fritz_error` and carries a ref).
- [ ] The refusal string for "recipient blocked this sender", "recipient blocked everyone", and "recipient's DMs are closed" is **the same object**, asserted by `assertEqual` between two invocations, not by two literals.
- [ ] Every reply on every `/tell` and `/relay` path passes `ephemeral=True`, asserted per path.
- [ ] Every relay send passes `allowed_mentions=discord.AllowedMentions.none()`, asserted on both legs.
- [ ] A DM that is a Discord reply to a delivered relay message routes to the original sender **and** `ask_stuff` is never called; `privacy.count_conversation_checkpoints` is unchanged for both parties before and after; `extract_memories_background` is never called.
- [ ] With two relays open to the same recipient from two different senders, replying to the first reaches only the first sender. (Any last-relay-wins or time-window implementation passes the single-relay test and fails this one.)
- [ ] A plain DM with no reference still reaches `ask_stuff` exactly as it does today — **release blocker.**
- [ ] A forwarded relay DM does not route (gated on `ctx.type is discord.MessageType.reply`).
- [ ] A reply to an expired relay neither routes nor falls through to the agent; the recipient is told the exchange lapsed.
- [ ] `SELECT typeof(dm_message_id) FROM relay_messages` returns `'integer'`, not `'text'`.
- [ ] Many `pending` rows coexist before delivery (the partial unique index tolerates repeated NULLs); the same `dm_message_id` cannot be recorded twice.
- [ ] `reserve_send` called from two concurrent threads never exceeds `RELAY_MAX_PER_SENDER_PER_HOUR`.
- [ ] Blocked and rate-limited attempts both still answer the interaction (mirroring `TestDeferredCommandsAlwaysAnswer` at `tests/test_bot_commands.py:629-646`) and both count toward the rate limit.
- [ ] `privacy.forget_all` returns a `relays` key; `tests/test_privacy.py:368-376` is updated rather than deleted; `_ForgetConfirmView.confirm` renders `relays` **and** the currently-missing `alias_dropped`.
- [ ] `forget_all` removes rows where the user is the **recipient**, not only the sender; and does **not** remove `relay_optouts` rows where `blocked_id = me`.
- [ ] `forget_relay` honours an `IDENTITY_LINKS` alias on both ends of a block check.
- [ ] `export_user_data` includes relayed messages in both directions and never the other party's opt-out state.
- [ ] No relay body, in either direction, appears anywhere in `audit.log` — asserted by writing a relay with a distinctive body and grepping the test's audit file. Only `chars` and `body_sha16`.
- [ ] `import relay_router` does not put `mister_fritz` in `sys.modules`.
- [ ] `migrate_identity.survey()` reports both `relay_messages.sender_id` and `relay_messages.recipient_id`; `--apply` refuses when a relay `sender_id` is unmapped.
- [ ] `tests/test_packaging.py::test_py_modules_lists_every_top_level_module` passes with `relay_store` and `relay_router` in `pyproject.toml`.
- [ ] With `RELAY_ENABLED=false`, DM handling is byte-for-byte unchanged and `/tell` refuses with copy saying so.
- [ ] With `RELAY_AGENT_TOOL_ENABLED=false`, `get_conversation_tools_description` output contains no `relay_message`.
- [ ] `"relay_message" in mister_fritz.TOOL_NOTICES`.
- [ ] The agent cache key at `mister_fritz.py:605` includes `guild_id`: building in guild A then guild B yields two distinct cache entries.
- [ ] `ruff check .` is clean and `pytest tests/ --cov-fail-under=60` passes **under the `.venv` interpreter**, with `discord.__version__` starting `2.6`.

---

## What not to build

Named explicitly so the next person does not gold-plate this.

- **Deferred relays.** "Tell Alice at 5pm" needs its own persisted rows and a `restore()` pass next to `ScheduleManager.start()`. Shipping it without persistence means a restart silently drops a message the sender was told would be delivered — which violates the one rule this feature exists to enforce. Phase 1 refuses deferred relays outright.
- **Attachments.** Re-uploading one person's file to a third party is a malware-distribution channel with the bot's name on it, and none of the attribution reasoning applies to a file. Say so in the DM: "Alice attached 1 file, which I have not passed along — I carry words, not parcels." This will be the first feature request. Decline it.
- **Chunking a relay body.** `split_into_chunks` is banned on this path. Chunk 2 arrives with no attribution and reads as Fritz speaking in his own voice. Over the cap is refused, never split.
- **The `dm_channels` cache.** Deferred with a named trigger (see reconciliation 4). The column is in the schema; the cache is not.
- **Per-guild opt-in and a `relay_guilds` table.** A genuinely strong platform control, and the right answer the day Fritz is installed in a 5,000-member server the owner does not run. `RELAY_ENABLED` is the phase-1 stand-in for a single-owner deployment. Revisit *before* the first such install, not after.
- **The full rate-limit lattice.** Per-pair cooldowns, per-pair hourly and daily caps, first-contact-per-sender-per-day, global first-contact-per-hour. Three limits ship; seven do not. Add the first-contact caps if the bot ever leaves guilds the owner controls.
- **A relay session, a TTL capture window, or first-reply-wins.** Any of these silently captures the recipient's *next* DM whatever she meant by it. Bob relays "dinner?", Alice ten minutes later unrelatedly DMs Fritz "remind me to take my meds at 9", Bob receives it. That is a privacy breach manufactured by our own heuristic, and no disclosure copy repairs it.
- **A "current conversation partner" mode for a DM channel.** Same failure, dressed differently.
- **Auto-routing bare text**, even if telemetry shows most replies are bare text. If the ratio is bad, the answer is a better affordance — a one-time nudge, a select menu, a pinned prompt — not a heuristic that guesses.
- **Read receipts.** "Delivered" is Fritz's own action and the sender needs it. "Alice has seen it" is presence data about a third party and is not on offer.
- **Expiry notifications.** A DM saying "Alice never replied" is noise and a small social accusation. The relay simply leaves the list.
- **Rendering relay bodies in the admin panel.** Metadata only — who, when, how many, open or closed. Different category of private from schedules and workspaces.
- **A migration script.** There is nothing to migrate. `migrate_db.py` merges pre-existing database *files*; `migrate_identity.py` rewrites rows whose key format changed. A brand-new empty table is neither, and the repo has already shipped exactly this way twice — `workspaces` and `user_aliases`. `migrate_identity.py:82-84` states the precedent in its own comment. An unrun script is worse than none. (The `_SQLITE_TARGETS` registration in P2 is not a migration; it is making the *existing* tool aware of the new columns.)

---

## The honest risk summary

**The shape of the exposure.** "Anyone in a shared server" plus a two-way relay means: any member of any guild Fritz is in can cause Fritz to DM any other member of that guild, and can then hold a sustained back-and-forth with them, without either party ever having agreed to be reachable. The recipient sees a DM from a bot. Discord's own block and report tooling targets **the bot**, not the sender. Fritz is the face on a message he did not write.

**What the guard rails do protect against.**

- *Spoofed attribution.* The embed author field is structurally outside the sender-controlled description. A sender cannot write a fake "— from @admin" that renders as chrome.
- *Fritz as a ping proxy.* `AllowedMentions.none()` on every send, both legs.
- *One person nagging one person.* `RELAY_MAX_PER_SENDER_PER_HOUR`, plus `/relay block <user>`.
- *A brigade.* `RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR` is the only limit that does this, and it is the reason it ships in phase 1 while six others do not.
- *Block-probing.* One indistinguishable refusal for every recipient-side reason, pinned by an equality assertion.
- *Block laundering.* `/forget all` does not delete blocks other people placed against you.
- *Relay text reaching the agent.* `relay_router` cannot import `mister_fritz`. Attacker-controlled text never becomes input to a tool-bound agent with file access.
- *Memory and checkpoint contamination.* Same mechanism, asserted at the storage layer rather than by a mock, so it survives a rewrite of the interception point.

**What they do not protect against, plainly.**

- **A determined harasser with patience.** Ten messages an hour, every hour, is still ten messages an hour. The block is the real defence, and it is reactive: it works only after the first message has landed.
- **The first message.** There is no first-contact gate. A stranger in a 500-person guild reaches you once, in full, before you have any opportunity to refuse. That is the direct consequence of owner decision 2, and it is the thing a first-contact gate would fix at the cost of partially re-imposing an allowlist.
- **A sender who churns identities.** Blocks key on canonical id. Someone with two accounts in the same guild has two turns.
- **Bodies at rest.** Relay bodies sit in plaintext in `fritz.db` for `RELAY_RETENTION_DAYS`, in a file that also holds schedules, aliases and LangGraph checkpoints, with no encryption story. Anyone with host access — including any admin-gated `execute_command` user, per DECISIONS.md decision 1 — can read every relayed message. State this in `/about` and the README rather than letting someone discover it.
- **Platform enforcement.** A bot that DMs users who have not interacted with it is the pattern that gets applications disabled, and enforcement lands on the **application token**, not on the abusing user. The rate limits protect the bot's own survival at least as much as its users'. A 403/50007 must be treated as **terminal** — never retried — because retrying is exactly the behaviour Discord's heuristics read as spam.
- **Anything at all, socially.** Every control here is a policy mitigation wearing technical clothes. A permanent `relay_messages` row and an audit line tell you *who* sent *what many* characters *when*. They do not stop anyone.

**What to watch after shipping.**

1. `relay.replies_routed` ÷ `relay.delivered`. If that ratio is bad after a week, the reply affordance has failed and people are typing bare text into the void. Fix the affordance; do not reach for a session heuristic.
2. `relay.denied.*` by reason. A rising `blocked_optout` count against one sender is the abuse signal, and it is the only one that arrives before a complaint does.
3. The absolute count of distinct `(sender, recipient)` pairs per week. If it grows past what a small friend group produces, the per-guild opt-in table moved from What Not To Build to a prerequisite.
4. 429s on `POST /users/@me/channels` in the discord.py logs. That is the `dm_channels` cache's trigger, and it will not appear in testing with two accounts.
5. `audit.log` size, until P4's rotation is confirmed working.

Note also that `METRICS.increment` mirrors into Prometheus only for the exact name `discord_messages` and names starting with `tool.` (`observability.py:104-112`). A `relay.*` counter reaches the internal snapshot and `/health` but **never Prometheus** — so if any of the above is meant to be graphable, `observability.py` needs a `relay.` branch backed by a new Counter with a `kind` label.

---

## Open questions, ranked by how much they block work

**1. Does `/forget all` clear the blocks *you placed*?** — *Blocks PR 6.*
The half where blocks placed *against* you survive is settled (anti-laundering). This half is not. Two defensible answers: strict erasure means a harassment target who runs `/forget all` re-arms their harasser, having been told they were protecting themselves; keeping them means `/forget all` does not forget everything. **Default if you say nothing: blocks you placed survive, with an explicit `/relay forget-blocks` escape hatch named unmissably in the confirmation text.** A privacy command must not be a safety regression. Say which, because the confirmation copy and one test depend on it.

**2. First contact: full content, or a gated notice?** — *Blocks PR 2's copy, and only its copy.*
Deliver the stranger's message immediately with the opt-out button visible (the design as written), or tell the recipient that Bob has something waiting and withhold the content until she accepts? The gate is meaningfully kinder to people who never asked to be reachable, and it partially re-imposes an allowlist by the back door — which decision 2 settled against. This is a values call. **Default: deliver in full, with the opt-out one press away.**

**3. `RELAY_REPLY_WINDOW_MIN` — is 24 hours right?** — *Blocks nothing; a default ships either way.*
Too short and someone who opens Discord the next evening finds their reply refused; too long and a stale thread routes surprising text to someone who has forgotten the exchange. Your own usage pattern should set this, not mine.

**4. Phase-2 addressing.** — *Blocks PR 9 only, and PR 9 ships dark.*
Three options: the name-must-appear-in-the-user's-own-message guard I propose (novel, untested, fails on pronouns); a `UserSelect` confirmation tap (safe, but `UserSelect` in a DM can only offer the bot or the viewer themselves — and a DM is where the agent tool's entire value lives); or model-supplied names with no structural guard (which I would decline to design, given that `scrape_web` output and unsupervised memory extraction both reach the system prompt verbatim). **Default: the substring guard, shipped behind `RELAY_AGENT_TOOL_ENABLED=false`.**

**5. Does `/tell compose` ship at all in phase 1?** — *Blocks PR 7's existence.*
It is the only path that puts model-generated text under a human's name, and it is the part of decision 3 you hedged most specifically. Splitting it out is cheap and defensible. **Default: ship it, with the draft preview. If the preview goes, the feature goes with it.**

**6. Does a relay survive the two parties ceasing to share a guild?** — *Blocks nothing; affects one lookup.*
**Default: yes.** Authorisation is checked at `/tell` time and recorded in `guild_id`; a reply the recipient chose to send should not silently vanish because of a membership change she may not know about. The counter-argument — that the shared guild *is* the ongoing basis of the relationship — is reasonable.

**7. Admin visibility of relay bodies.** — *Blocks nothing.*
**Default: metadata only.** The panel already lists users, schedules and workspaces; relay bodies are a different category of private. You run the box and may want otherwise.

### Experiments, not questions

Four things nobody could settle by reading. None blocks phase 1; the first two should be run before PR 2 ships to a real guild.

- **Does `create_dm()` succeed when the recipient has DMs closed, with the 403/50007 arriving only at `send()`?** Two test accounts, recipient turns off "Allow direct messages from server members", call `create_dm()` and `ch.send('x')` as separate awaits, log `type(exc).__name__, exc.status, exc.code` for each. The design is correct either way — it persists nothing until a send succeeds — but the answer decides whether `create_dm` is ever worth calling speculatively.
- **Does a forwarded bot DM populate `reference.message_id`?** Forward a Fritz DM from a real client and log `message.type`, `message.reference.to_dict()`. The `MessageType.reply` gate should already handle it; this confirms rather than assumes.
- **Does `guild.fetch_member(id)` work with the members intent off?** The docstring asymmetry is suggestive — `fetch_members()` (the LIST route) explicitly documents `ClientException: The members intent is not enabled` and `fetch_member()` documents only `Forbidden`/`HTTPException`/`NotFound` — but phase 2's shared-guild check would depend on it. One call against the live bot.
- **Does `guild.query_members(query='ali', limit=5)` return results with the members intent off?** The load-bearing unknown for any phase-2 name resolution that is not the substring guard. Documented as requiring only the *presences* intent, but op-8 behaviour is the kind of thing Discord changes. If it times out, phase 2 falls back to `identity_store` alone — which only knows people who have already talked to Fritz.

---

## Also worth doing, noticed in passing

Not part of this feature; each is a few lines and each is a real defect.

- `_ForgetConfirmView.confirm` (`bot_commands.py:168-176`) renders four of the five keys `forget_all` returns. `alias_dropped` is deleted on every `/forget all` and reported to nobody. Fix it in PR 6.
- `scheduler.py:114` uses `run_in_executor(None, ...)` — the default executor, not `bot_adapters._BLOCKING_POOL`. Inconsistent with the repo's own rule, and any code that copies `_run_task`'s shape inherits it.
- `agent_tools.py:546` passes the literal string `"scheduled"` as `user_id`, so every one-shot scheduled message writes memories into a shared `"scheduled"` Chroma namespace and is attributed to no person. Do not let the relay tool copy this.
- `_ForgetConfirmView` implements no `on_timeout`, so after 30s its buttons silently do nothing. PR 7's draft view should not repeat it.