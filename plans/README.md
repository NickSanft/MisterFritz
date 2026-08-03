# Mister Fritz — Implementation Plan for the Top 10 Improvements

_August 2026. Produced by an 11-agent planning workflow: one planner per item — each re-read the actual source and test suite rather than trusting the earlier audit's line numbers — plus an integrator that resolved cross-item conflicts and sequenced the work into shippable PRs._

Every file:line reference was verified against the working tree at plan time. Where a planner found the earlier audit wrong, the correction is recorded in that item's **Current state** section.

> **[DECISIONS.md](DECISIONS.md) records the answers to every open question these plans raised, and overrides the plan text where the two disagree.** Read it alongside this file. Most consequentially: plan mode is being deleted outright, which amends plans 01, 03 and 08.

> **2026-08-03 — the `NewMockup/` design landed.** [Plan 07](07-web-chat-redesign.md) was rewritten around it: the chat surface is now dark-academia (candle-lit purple, serif, zero border-radius, clip-path facets) rather than the light parchment palette originally planned. Three knock-on changes to the PR sequence below:
> - **PR 11 splits three ways** — `_theme_base` / `_theme_chat` / `_theme_admin`. Decision 11 (one shared theme) is superseded by decision 12; the admin panel keeps its own palette and still gets its contrast fix.
> - **A new PR is needed before PR 11**: `admin_static/` + a `/static` mount + self-hosted fonts, plus extending the Basic-auth path exemption to `/static`.
> - **PR 5 becomes a hard blocker for PR 13.** Plan 05's sanitiser as written strips every Pygments class, so syntax highlighting renders flat with no error and no failing test. See decisions 20 and 21 — plan 05's CSP also needs `font-src` and cannot use `script-src 'none'`.

## Contents

1. [Feed conversation history to the executor](01-history-window.md) — M (half day), depends on nothing
2. [Get blocking work off the Discord event loop](02-event-loop.md) — M (half day), depends on nothing
3. [Real token streaming end-to-end](03-token-streaming.md) — M (half day), depends on nothing
4. [Contain execute_command](04-exec-env-scrub.md) — M (half day), depends on nothing
5. [Authenticate and harden the web chat surface](05-web-auth.md) — L (1-3 days), depends on nothing
6. [Purge the dependency freeze and fix CI](06-deps-ci.md) — L (1-3 days), depends on nothing
7. [Web chat: mobile, bug fixes, and the butler restyle](07-web-chat-redesign.md) — L (1-3 days), depends on nothing
8. [Cut fixed per-message latency](08-latency-tax.md) — L (1-3 days), depends on nothing
9. [Stable identity and per-channel threads](09-identity-threads.md) — XL (>3 days), depends on nothing
10. [Discord surface polish](10-discord-polish.md) — L (1-3 days), depends on token-streaming

This file is the integration layer: what to build once, what collides with what, and the order to ship it in. **Read this before any individual plan.**


---

## A discovery that reframes item 1

The planners found something the audit missed, and it changes the shape of the first item.

`add_messages` coerces a bare `str` into a `HumanMessage`, not an `AIMessage`. Both `executor` (mister_fritz.py:450-453) and `synthesizer` (:502-507) return `{"messages": [<bare str>]}`. So **every reply Fritz has ever written is checkpointed as a HumanMessage** — the stored transcript is an unbroken run of user turns with no speaker distinction.

Two consequences. First, feeding that list to the model as a history window would be worse than useless, so the two-line role fix is a hard prerequisite for item 1, not optional polish. Second, this is already a live user-visible bug: `admin_panel._doc_to_message` maps `human` to a user bubble and `ai` to a Fritz bubble, so the web chat's history hydration paints **every one of Fritz's past replies as if the user had said it**.

Verified in this repo's venv:

```
>>> add_messages([HumanMessage(content='what is your name')], ['I am Fritz, sir.'])
HumanMessage | what is your name
HumanMessage | I am Fritz, sir.
```


---

## The three tranches

### Tranche 1 — first weekend: hygiene and the two real security holes

Eight PRs, six of them XS/S/M and mostly mechanical. This tranche is deliberately all correctness and containment, no performance and no refactors. It closes the two holes that are verified rather than theoretical — a workspace user can currently print DISCORD_BOT_TOKEN and ADMIN_PANEL_PASSWORD into a Discord channel, and anyone who can reach :8001 can claim any username and read that person's conversation. It also fixes the one live data-loss bug (the New conversation confirm never fires because of a JavaScript SyntaxError) and stops the test suite writing databases and PNGs into the working tree, which is the thing that will otherwise make every later PR's diff noisy. PR 2 and PR 3 are cheap insurance: two lines and a config batch that between them defuse the AIMessage import trap and seven separate fritz_utils merge conflicts. Nothing here requires a live Ollama to verify except PR 4's manual command re-check.

### Tranche 2 — second pass: finish the web surface, then the streaming stack

Finish web-auth and web-chat-redesign together, because they interleave in admin_panel.py and admin_templates/ and doing them apart means touching the same files twice. The ordering inside is load-bearing in exactly two places: PR 5 must precede PR 13 or syntax highlighting is silently inert, and PR 12 should precede PR 16 so the applyToken() seam already exists when the SSE wire format flips from cumulative to delta. Then the streaming stack: PR 2 → PR 15 → PR 16 is a hard chain through mister_fritz.py's executor, and PR 17 must precede PR 19 because both rewrite the same three slash-command bodies. This is where the user-visible payoff lives — the bot stops freezing during /gen, replies stream token by token, and the Discord surface stops leaking Python tracebacks.

### Tranche 3 — longer horizon: performance, identity, packaging

These are the three items that most need a live Ollama, a real production host, and unhurried attention — and the one (PR 20) that can only be tuned once streaming is actually running. latency-tax's core assumption (that format=<json schema> works reliably on the configured models) is explicitly unverifiable by reading code and needs a smoke test against the real models. identity-threads is XL, touches every adapter boundary, and ships a one-time migration that must run with the bot stopped — it is not weekend work, and its separator decision ('discord:123' vs 'discord-123') should be made with the Windows filename trap in mind. deps-ci goes last on purpose: it regenerates requirements.txt wholesale, so it should absorb the nh3 and Pygments pins added in tranches 1 and 2 rather than racing them. Realistically this tranche may never happen, and that is fine — see the tranche-one note.

### If tranche 1 is all that ever ships

"Tranche 1 is deliberately constructed to be a complete, coherent stopping point rather than the first eighth of a project — and for a one-person hobby bot, stopping there is a defensible outcome, not a failure.\n\nWhat the owner would have: a bot that can no longer be made to print its own Discord token and admin password into a chat channel by any user who ran /workspace enable; a web chat that requires a shared secret instead of letting anyone on the network type a username and read someone else's conversation; an XSS hole in the reply renderer closed; a 'New conversation' button that actually asks before destroying a thread; Fritz's replies stored with the correct speaker role so the web history stops painting every one of them as a user bubble; and a test suite that no longer writes SQLite databases, a Chroma directory and PNGs into the working tree on every run. Every one of those is a defect or a live exposure. None of them is a refactor.\n\nWhat they would NOT have is entirely performance and polish: the planner still costs a FAST-model round trip on 'hi', /gen still freezes the whole bot for a minute, replies still appear all at once instead of streaming, the executor is still functionally amnesiac inside a thread, the chat is still desktop-only, requirements.txt still drags a 2016 neuroimaging stack into every install. All annoying; none of it can hurt the owner or a user, and all of it stays exactly as annoying as it is today rather than degrading.\n\nThe single most important thing to get right if tranche 1 is all that ever ships: land PR 0, PR 2 and PR 3 anyway, even though none of them changes user-visible behaviour. PR 0 is what makes any future weekend cheap to restart — without it, the first thing a returning owner sees is a dirty git status they no longer remember the cause of. PR 2 is two lines that silently corrupt the summarisation prompt and the web history renderer for as long as they are absent. PR 3 costs twenty minutes now and saves seven merge conflicts later. They are the cheapest possible option on the rest of the plan.\n\nOne caveat to flag explicitly: PR 4 requires a real product decision (Config A, interpreters allowed but admin-gated, versus Config B, inspection-only allowlist open to everyone) before the mechanical work starts. If the owner cannot make that call in the moment, ship Config A — it preserves today's agent capability, breaks the fewest tests, and EXEC_REQUIRE_ADMIN=false is a one-line env flip to undo. Do not let an unmade decision block the env scrub, which is the part that actually stops the token leak and is identical under both configurations."


---

## Shared prerequisites — build these once

Multiple plans each independently invented the following. Extracting them once turns duplicated, conflicting work into a single change.

### conftest.py test-environment sandbox

**Lives in:** tests/conftest.py (NEW — verified absent; 'find . -name conftest.py' outside .venv returns nothing)

**Consumed by:** history-window, token-streaming, latency-tax, identity-threads, web-auth, web-chat-redesign, deps-ci, event-loop, discord-polish

Importing mister_fritz has three CWD-relative side effects, all at module scope: SQLiteStore(CHAT_DB_NAME) at :604 and SqliteSaver.from_conn_string(CHAT_DB_NAME) at :606 (CHAT_DB_NAME defaults to DB_NAME='fritz.db'), the Chroma store rooted at CHROMA_DB_PATH='./chroma_store', and a daemon thread at :652 writing mister_fritz_diagram.png. That is exactly the debris in git status: tests/chat_history.db, tests/chroma_store/, tests/mister_fritz_diagram.png untracked, plus 'M mister_fritz_diagram.png' and 'M document_engine_diagram.png' dirtied at the repo root on every run from there. The conftest must set DB_NAME, CHAT_DB_NAME, SCHEDULE_DB and CHROMA_DB_PATH to a per-session temp dir as TOP-LEVEL MODULE STATEMENTS, not inside a fixture — every test module (test_mister_fritz.py:29, test_admin_panel.py, test_agent_tools.py:36-44) imports the real module at module scope, which runs during collection, long before any fixture. It should also host the _ensure_mock sys.modules stub preamble currently duplicated across four test files, and gitignore the three artifacts.

### canonical identity + thread_id derivation

**Lives in:** fritz_utils.py, next to is_admin (~line 218)

**Consumed by:** web-auth, identity-threads, privacy.py, admin_panel.py, mister_fritz.ask_stuff, main_discord, bot_commands, main_telegram, scheduler

Three verbatim copies of re.sub(r'[^a-zA-Z0-9]','',user_id) exist today — mister_fritz.py:536, admin_panel.py:353, privacy.py:22-25 (whose docstring literally says 'Mirror mister_fritz.ask_stuff's transformation'). This drift IS the live /forget bug for punctuated usernames. Both web-auth and identity-threads reinvent the fix separately with incompatible shapes. Extract one helper — minimally thread_id_for(user_id, channel_key=None) plus safe_user_token(user_id) for filesystem-safe names — before either item starts. safe_user_token is not optional polish: main_discord.py:161/:170, bot_commands.py:295 and admin_panel.py:711 all build filenames from the identity, and a ':' separator is illegal on NTFS.

### streaming callback contract (delta, accumulated, restart)

**Lives in:** mister_fritz.py as _DeltaEmitter (producer); the contract itself documented on ask_stuff's docstring

**Consumed by:** token-streaming, main_discord.streaming_callback, admin_panel._streaming_callback, discord-polish (set_status coexistence), web-chat-redesign (applyToken seam)

Four surfaces consume this callback and its arity change is atomic — a PR that migrates main_discord but not admin_panel raises TypeError on the first web chat message. token-streaming already defines _DeltaEmitter correctly; the point is that the arity flip must land in ONE PR touching mister_fritz, main_discord, admin_panel, admin_templates/chat.html and the four test fakes together. Do NOT split it across PRs. Related: web-chat-redesign's applyToken() is precisely the client-side seam that makes this a one-function change on the browser side — land that first.

### run_blocking bounded-offload helper

**Lives in:** bot_adapters.py (platform-neutral, respects the module's 'no discord import' rule)

**Consumed by:** event-loop, bot_commands voice/gen/lore, main_discord.on_message + TTSEngine load + speech_to_text, deferred: scheduler.py:109, main_telegram.py:26/45/52, admin_panel.py:388/457/630/642

event-loop invents this. Four other call sites already offload correctly but via the unbounded default executor, so the bounded-pool guarantee is only partial until they migrate. Land the helper once with the knob, migrate the Discord surfaces in event-loop, and file the rest as a follow-up. Do NOT convert admin_panel.py:554-558 — the comment there records that the plain daemon thread is deliberate for TestClient loop teardown, and the five TestChatStream* classes are the detector if someone tries.

### fritz_error(operation, exc) butler-voiced error helper

**Lives in:** bot_adapters.py, alongside run_blocking

**Consumed by:** discord-polish (nine bot_commands sites + two main_discord sites), admin_panel error SSE frame (currently leaks str(e) at the 'error' event), event-loop (the new voice_slash except block it introduces)

discord-polish scopes this to Discord only, but admin_panel.chat_stream currently does event_queue.put(('error', str(e))) — the same raw-exception leak on the web surface. Putting the helper in bot_adapters (stdlib + observability.METRICS only, no cycle) lets both surfaces use it. event-loop adds a new exception path to voice_slash that will want it too, so land fritz_error before or with event-loop rather than after.

### _sanitise_html — nh3 with an explicit class allowlist

**Lives in:** admin_panel.py, adjacent to _render_markdown

**Consumed by:** web-auth, web-chat-redesign codehilite, _doc_to_message history rendering, chat_stream done-frame payload

See the conflicts entry for the verified detail. The helper must start from dict(nh3.ALLOWED_ATTRIBUTES) and union {'class'} onto div/pre/code/span/table, because nh3's attributes= parameter REPLACES the default map rather than extending it — passing a bare {'span': {'class'}} would also strip href from <a> and src/alt from <img>. Both the streamed done-frame and the server-rendered history go through _render_markdown, so a single helper covers both paths.

### tests/test_bot_adapters.py

**Lives in:** tests/ (NEW)

**Consumed by:** event-loop (TestRunBlocking), discord-polish (TestFenceAwareChunking, fritz_error tests)

Both items independently create this file. bot_adapters has no heavy imports, so tests for it need none of the sys.modules mocking preamble that test_discord_commands.py requires — which is a genuine reason to put the chunker tests here rather than leaving them beside main_discord's re-export tests. Create it once, in whichever of the two PRs lands first.

### lazy import of image_generator in agent_tools

**Lives in:** agent_tools.py:20 → move into generate_image (~:306)

**Consumed by:** deps-ci (load-bearing for the whole extras split), every test file that stubs image_generator in sys.modules

Confirmed 'import image_generator' sits at agent_tools.py:20, module scope. Four test files (test_agent_tools.py:30, test_mister_fritz.py:24, test_bot_commands.py:25, test_discord_commands.py:31) exist partly to stub it. Making it lazy is a two-line change that makes those stubs inert rather than mandatory, reduces collection-order coupling, and is the single change that makes a torch-free install able to import mister_fritz at all. It costs nothing to land early and it de-risks both deps-ci and the conftest consolidation.

### one batched config-knob PR

**Lives in:** fritz_utils.py '# Tunables' block + .env.example

**Consumed by:** history-window, token-streaming, latency-tax, event-loop, web-auth, web-chat-redesign, discord-polish

Seven items each open with an identical 'Commit 1 — add knobs, no behaviour change' step, and four of them target the same two insertion points (line 97 and line 119). Merging those seven commits into one converts seven unmergeable additive conflicts into zero, and gives .env.example a single coherent update instead of seven. Keep exec-env-scrub's _DEFAULT_EXEC_ALLOWED out of it (its value encodes the Config A/B product decision) and identity-threads' is_admin change out of it (behavioural, not a knob).


---

## Conflicts between items

Every place two or more plans modify the same code, with the correct order and what the later item must adapt to.

### admin_panel.py:313-323 — _render_markdown / _MARKDOWN_EXTENSIONS

**Items:** web-auth (nh3 sanitiser), web-chat-redesign (codehilite syntax highlighting)

**Collision.** HARD SEMANTIC CONFLICT, verified empirically this session and missed by both plans. web-auth wraps the markdown output in nh3.clean(). web-chat-redesign adds the codehilite extension, whose entire output is class-attribute-driven. I installed nh3 into a scratch dir and ran it: markdown+codehilite emits '<div class="codehilite"><pre><span></span><code><span class="nb">print</span>…' and nh3.clean() returns '<div><pre><span></span><code><span>print</span>…' — every class attribute gone. nh3's default attribute map (nh3.ALLOWED_ATTRIBUTES) has no entry for span/div/pre/code at all. So codehilite becomes dead weight: Pygments runs, spans are emitted, all styling hooks are stripped, and the _theme.html highlight CSS matches nothing. Nothing errors and no test catches it — web-chat-redesign's own guard test only asserts '<pre>' and 'print' are present, and both survive. web-auth even records the wrong conclusion in its risk list ('class="language-py" being stripped is fine'), which is true for fenced_code but fatal for codehilite.

**Resolution.** web-auth's sanitiser lands FIRST, codehilite second. Rationale: if codehilite lands first and nh3 arrives later with defaults, highlighting dies silently and cosmetically — nobody notices for weeks. If nh3 is already in place, the developer adding codehilite sees unhighlighted code on the first manual check and is forced to fix it. Security also outranks cosmetics. The second item must own a single shared helper (suggested: _sanitise_html(html) in admin_panel.py next to _render_markdown) that builds the allowlist explicitly: d = dict(nh3.ALLOWED_ATTRIBUTES); for t in ('div','pre','code','span','table'): d[t] = (d.get(t) or set()) | {'class'}; then nh3.clean(html, attributes=d). Verified this session: that exact call preserves '<div class="codehilite">' and '<span class="nb">' while still stripping onclick from '<span class="x" onclick="evil()">'. Note nh3's attributes= REPLACES the default map wholesale — you must start from dict(nh3.ALLOWED_ATTRIBUTES), not pass a bare {'span': {'class'}}, or you also lose href on <a> and src/alt on <img>.

### mister_fritz.py:388-435 — executor's inputs construction and agent.stream loop

**Items:** history-window, token-streaming, discord-polish

**Collision.** Line 411 ('inputs = {"messages": [("system", system_prompt), ("user", agent_prompt)]}') is the seam. history-window DELETES line 411 and rebuilds inputs per-branch inside the is_plan_mode/else tail at 388-410. token-streaming's declared range is 411-435 — it starts on the same line history-window removes and rewrites the whole 'for s in agent.stream(...)' loop below it. Git will conflict; more importantly the semantics interlock: token-streaming's _DeltaEmitter assumes it is the sole writer of accumulated text, while history-window changes what the agent is fed (a multi-message window rather than one user turn), which changes how many AIMessage turns the messages stream produces per superstep. Separately, discord-polish rewrites the tool_messages dict at 364-382 and the plan-step notice at 403-404, which sits inside history-window's 388-411 edit window.

**Resolution.** ORDER: history-window → token-streaming → discord-polish. history-window touches only the CONSTRUCTION of inputs (it explicitly does not touch the streaming loop at 416-433, the resp extraction at 435, or the image_paths scan at 437-441), so it is the cleaner base. token-streaming then rewrites the loop against an inputs variable that is already per-branch — it must adapt by reading inputs from whichever branch built it rather than from a single hoisted assignment. discord-polish lands last and only rewrites string literals in the dict at 364-382 plus the notice at 404, which by then are untouched by either predecessor. Reverse order forces token-streaming's brand-new loop to be re-cut by history-window's branch split.

### mister_fritz.py:8 — langchain_core.messages import line

**Items:** history-window, token-streaming

**Collision.** VERIFIED: AIMessage has exactly one use in the file today, at line 429 inside the streaming loop. token-streaming's plan is therefore correct AS WRITTEN ON MASTER when it says 'swap AIMessage for AIMessageChunk — AIMessage's only use is line 429, which this change deletes; leaving it triggers ruff F401 and fails CI.' But history-window ADDS two new AIMessage uses (wrapping the executor return at ~:451 and the synthesizer return at ~:503). Once history-window has landed, performing token-streaming's documented import swap removes a name that is still used and breaks the module.

**Resolution.** Since history-window lands first, token-streaming must ADD AIMessageChunk alongside AIMessage rather than swapping — 'from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage'. Correct token-streaming's changeSites note before handing it to an implementer, because the stated rationale ('its only use is line 429') will be false by then. CI does catch the mistake: ruff's F select set includes F821 undefined-name, so 'ruff check .' fails rather than the bot crashing at runtime — but the engineer will waste time on a confusing error whose plan text says the opposite.

### mister_fritz.py:523-593 — ask_stuff signature and body

**Items:** web-auth, identity-threads, latency-tax, token-streaming, history-window

**Collision.** Five-way. web-auth adds a keyword-only thread_id= parameter with its own sanitiser that preserves [_-] so 'web-alice' cannot collide. identity-threads rewrites the same function wholesale: it deletes the user_id_clean regex at :536, adds display_name and channel_key parameters, and derives thread_id via a new thread_id_for() helper. latency-tax changes :565 ('original_request': '' → full_prompt) and deletes the dead system_prompt rebuild at :543-545 and the pretty_print() call at :571-576. token-streaming only edits the docstring at :523-534. history-window changes nothing here but depends on what the checkpoint under that thread_id contains. web-auth and identity-threads are solving the SAME problem (thread_id must stop being a lossy transform of the display name) with two incompatible mechanisms.

**Resolution.** ORDER: latency-tax's :543-545 / :565 / :571-576 edits are independent one-liners and can land any time (fold them into the config/dead-work PR). web-auth lands its thread_id= keyword-only parameter next — it is the minimum viable change and it is needed urgently for the security split. identity-threads lands LAST and generalises: thread_id_for(identity, channel_key) becomes the default, and web-auth's explicit thread_id= survives as an override for callers that want to pick a checkpoint directly. Do not let identity-threads delete the thread_id= kwarg — admin_panel and privacy will both still be passing it. Also note identity-threads' test-update list instructs you to edit tests/test_admin_panel.py::TestChatUploadDocument::setUp, but web-auth DELETES that class wholesale; that instruction is stale by the time identity-threads runs.

### Three copies of re.sub(r'[^a-zA-Z0-9]','',user_id) — mister_fritz.py:536, admin_panel.py:353, privacy.py:22-25

**Items:** web-auth, identity-threads, history-window

**Collision.** I confirmed all three exist verbatim. mister_fritz.ask_stuff computes user_id_clean; admin_panel._load_chat_history recomputes the identical regex inline to find the checkpoint; privacy._sanitise_thread_id has a docstring that literally says 'Mirror mister_fritz.ask_stuff's transformation'. Three plans each rewrite a different subset: web-auth replaces admin_panel:353 with _chat_thread_id and adds thread_id to privacy.forget_conversation; identity-threads deletes all three; history-window relies on the checkpoint these produce. Any two landing independently leaves the write path and the delete/read path disagreeing — which is already the live /forget bug for usernames with punctuation.

**Resolution.** Extract ONE helper before either item touches it (see sharedPrerequisites). Whichever of web-auth or identity-threads lands first must delete all three copies and route every caller through the single fritz_utils helper, even if it only needs one of them. Leaving two copies behind is how the current bug happened.

### bot_commands.py:390-442 — voice_slash / gen_slash / lore_slash bodies

**Items:** event-loop, discord-polish

**Collision.** event-loop rewrites all three bodies structurally: wraps the blocking calls in await run_blocking(...), adds per-cog asyncio.Semaphore acquisition, and adds a missing try/except to voice_slash so a deferred interaction is always answered. discord-polish rewrites the SAME bodies' user-facing strings: the 'Something crazy happened!' AttributeError handler at :409-410, the raw f'Failed to generate image: {e}' leak at :422-423, and lore_slash's header + chunking at :425-442. They overlap line-for-line. discord-polish declares DependsOn: token-streaming but says nothing about event-loop, which is the collision that actually matters.

**Resolution.** ORDER: event-loop FIRST. It is the structural change (new await points, new semaphores, a new except block in voice_slash); discord-polish then reskins the copy inside that structure and routes the new except through fritz_error. Reverse order means discord-polish's butler strings get hand-retyped into event-loop's rewritten bodies. Note event-loop also ADDS an exception path to voice_slash that discord-polish will want to butler-voice — so discord-polish's change list grows by one site after event-loop lands.

### bot_adapters.py — whole file (17 lines today)

**Items:** event-loop, discord-polish

**Collision.** event-loop appends _BLOCKING_POOL + async run_blocking() and imports asyncio/functools/ThreadPoolExecutor/BLOCKING_POOL_SIZE. discord-polish REPLACES split_into_chunks with a boundary- and fence-aware version and appends fritz_error() importing METRICS from observability. Both then want a tests/test_bot_adapters.py, which does not exist — I checked, tests/ has 19 files and none is test_bot_adapters.py. Both will create it and conflict on the whole file.

**Resolution.** Either order works (the two additions are disjoint), but the NEW TEST FILE must be created exactly once. Land discord-polish's chunker PR first with tests/test_bot_adapters.py containing TestSplitIntoChunks-style fence tests plus a placeholder import, then event-loop appends TestRunBlocking to the same file. Both plans independently note the module's 'keep it dependency-light, no discord import' rule — that rule survives both additions (asyncio/functools are stdlib; observability imports only stdlib + optional prometheus_client), so there is no import cycle. Verify once with a bare 'python -c "import bot_adapters"'.

### main_discord.py:183-189 — streaming_callback / progress_callback closures

**Items:** token-streaming, discord-polish, event-loop, identity-threads

**Collision.** A 7-line span that four items edit. token-streaming changes streaming_callback's arity to (delta, accumulated, restart) and adds a DISCORD_STREAM_MIN_INTERVAL guard before run_coroutine_threadsafe so ~40 tokens/s do not queue 40 coroutines/s. discord-polish repoints progress_callback from ctx.channel.send(message) to streaming_handler.set_status(message). event-loop replaces the loop.run_in_executor(None, lambda: ask_stuff(...)) call immediately below at :193-203 with run_blocking(...). identity-threads changes the 'author' argument passed into that same ask_stuff call into identity/display_name/file_token and fixes :161 and :170 to use file_token.

**Resolution.** ORDER: event-loop → token-streaming → discord-polish → identity-threads. event-loop's edit is below the callbacks and mechanical. token-streaming owns the callback arity (it must change main_discord and admin_panel in ONE PR — a half-migrated arity raises TypeError in the other consumer). discord-polish then only repoints progress_callback's body. identity-threads last. CRITICAL for identity-threads: do NOT replace 'author' in place — main_discord.py:161 and :170 build temp_images/{author}_... and temp_audio/{author}_... paths, and a colon in 'discord:123' creates an NTFS alternate data stream on Windows silently on write. That is the single most likely way to break this change on the owner's actual OS.

### main_discord.py:34-83 — StreamingMessageHandler class

**Items:** discord-polish, token-streaming

**Collision.** token-streaming states the handler body at 48-82 is untouched and only changes the min_update_interval default at :38 to read DISCORD_STREAM_MIN_INTERVAL. discord-polish substantially rewrites the same class: adds status_text/_compose/set_status, replaces the [:2000] head-truncation at :63 and :77 with a tail window, and moves chunking out of on_message (:229-238) into final_update. discord-polish's declared DependsOn: token-streaming is correct for THIS part but not for the rest of the item.

**Resolution.** ORDER: token-streaming → discord-polish's handler work. But SPLIT discord-polish: its fence-aware chunker, fritz_error, embeds, app-command error handler, /draw Range bound, butler copy and temp_audio cleanup have ZERO dependency on token-streaming and can ship immediately. Only the StreamingMessageHandler status-line/tail-window work truly needs it (discord-polish's own risk list concedes 'resolve the edit-scheduling policy there, not here'). Treating the whole item as blocked on token-streaming needlessly delays a live bug fix — /draw 500 currently raises an uncaught HTTPException and the user sees nothing.

### fritz_utils.py:96-97 and :118-119 — the '# Tunables' block

**Items:** history-window, token-streaming, latency-tax, event-loop, web-auth, web-chat-redesign, discord-polish

**Collision.** Seven items insert new constants into a ~25-line window. history-window and token-streaming both target 'immediately after SUMMARIZE_THRESHOLD' (line 97) — literally the same insertion point. latency-tax and event-loop both target 'after MEMORY_EXTRACT_MIN_REPLY_CHARS' (line 119) — again the same point. web-auth rewrites the CHAT_ALLOWED_IMAGE_TYPES block at 155-169 while web-chat-redesign inserts CHAT_CODE_HIGHLIGHT at 164-169, inside it. Every one of these is an additive conflict that git cannot auto-merge, and each rebase risks a fat-fingered default.

**Resolution.** Land ONE config PR up front that adds every new tunable at once with defaults that exactly reproduce today's behaviour, then have each feature PR only CONSUME its knob. This is not extra work — every single plan already opens with 'Commit 1 — config, no behaviour change', so this merely merges seven identical commit-1s into one. It converts seven conflicting inserts into zero. The two exceptions that must stay with their own item: exec-env-scrub's _DEFAULT_EXEC_ALLOWED (its value is the Config A/B decision, not a mechanical addition) and identity-threads' is_admin signature change (behavioural).

### fritz_utils.py:221-231 — is_admin()

**Items:** identity-threads, exec-env-scrub, web-auth

**Collision.** Verified signature today is is_admin(user_id: str | None) -> bool, one positional parameter. identity-threads changes it to is_admin(user_id, display_name=None) with an ADMIN_LEGACY_NAME_MATCH fallback. exec-env-scrub adds a brand-new caller inside file_tools._exec_denied_reason. web-auth DELETES the admin_panel caller (and the 'import fritz_utils' at admin_panel.py:39 along with it).

**Resolution.** Low friction if ordered right: exec-env-scrub's new call site uses one positional argument, which survives identity-threads' signature change untouched. Land exec-env-scrub before identity-threads. web-auth's deletion is independent. The one stale instruction to fix: identity-threads tells you to change tests/test_admin_panel.py::TestChatUploadDocument::setUp's patch to 'side_effect=lambda u, display_name=None: u == "web:alice"' — that class no longer exists after web-auth. Drop that line from identity-threads' test plan.

### requirements.txt:291-292 (tail)

**Items:** web-auth (nh3), web-chat-redesign (Pygments), deps-ci (full regeneration)

**Collision.** web-auth and web-chat-redesign both append a single pin to the last two lines of the file (note the file's final line is unterminated, which makes appended diffs conflict more often than they should). deps-ci regenerates all 292 lines from a clean venv, which will either absorb or silently drop both.

**Resolution.** Order the small appends first, deps-ci LAST. Because deps-ci regenerates from 'pip install ".[voice,image,ocr,telegram]"', nh3 and Pygments must be added to the [project] dependencies table in pyproject.toml at that point, not just left in requirements.txt — otherwise the regeneration drops them and the chat surface loses its sanitiser, which is a security regression disguised as a dependency cleanup. Add an explicit item to deps-ci's Step 4 diff-review checklist: nh3 and Pygments must both appear in the regenerated file.

### tests/test_admin_panel.py — the 27 /chat/login call sites and the streaming fakes

**Items:** web-auth, token-streaming, identity-threads, web-chat-redesign

**Collision.** web-auth adds a password field to all 27 client.post('/chat/login', ...) calls. token-streaming rewrites four streaming fakes to the 3-arg callback and changes the token assertion from ['Very','Very well','Very well, sir.'] to ['Very',' well',', sir.']. identity-threads changes user_id assertions from 'alice' to 'web:alice' across roughly eight tests plus the _drain_for_user helper and a tearDown filename prefix. web-chat-redesign touches TestChatPageWithCookie's placeholder assertion. All four edit the same 900-line file heavily.

**Resolution.** Introduce the _login(client, username='alice', password=PASSWORD) helper in web-auth's PR and mechanically replace all 27 sites there — that is the single largest mechanical churn and it should happen once, early. Every later item then edits at most a handful of lines. Also: token-streaming's assertion change at line ~516 is exactly the seam web-chat-redesign's applyToken() function exists to protect, so land web-chat-redesign's streaming-UX PR BEFORE token-streaming and the collision shrinks to one two-line function plus one assertion.

### .claude/worktrees/nifty-hoover-6fe3d6/ — full duplicate checkout

**Items:** all ten

**Collision.** Confirmed via 'git worktree list': there is a second checkout of the entire repo at .claude/worktrees/nifty-hoover-6fe3d6 sitting at the same commit (bbe655b, detached HEAD). exec-env-scrub flags this for file_tools.py/fritz_utils.py, but it applies to every file every item touches. Any repo-wide search-and-replace, any 'ruff check .' from the wrong root, and any IDE refactor will hit both copies.

**Resolution.** Delete or prune the worktree before starting ('git worktree remove .claude/worktrees/nifty-hoover-6fe3d6'), or add it to ruff's extend-exclude in pyproject.toml alongside the existing entries. Do this in the conftest/hygiene PR so it is handled once for all ten items rather than rediscovered by each.


---

## Dependency order

- conftest-sandbox — no blockers; soft-unblocks everything by making test runs reproducible and git status clean
- agent-tools-lazy-image-import — no blockers; soft prerequisite for deps-ci and for simplifying conftest stubs
- aimessage-role-fix — no blockers; HARD BLOCKER for history-window, token-streaming, and correct web chat history rendering
- config-knobs-batch — no blockers; soft-unblocks seven items by pre-resolving the fritz_utils insertion conflict
- exec-env-scrub — soft dep on config-knobs-batch; HARD dep on the Config A/B product decision being made first (everything else in that item is mechanical either way)
- web-auth-sanitiser — no blockers; HARD BLOCKER for web-chat-redesign-codehilite (nh3 strips the class attributes codehilite depends on — verified)
- web-chat-redesign-quickwins — no blockers (viewport meta, confirm SyntaxError, aria wiring, two chat_send gaps); safe in parallel with anything
- web-auth-password-gate — soft dep on config-knobs-batch; introduces the _login test helper that later items depend on
- web-auth-image-sniffing — soft dep on web-auth-password-gate (shares the _login helper and the fixture rewrite)
- web-auth-thread-split — soft dep on web-auth-password-gate; HARD BLOCKER-BY-CONVENTION for identity-threads (establishes the ask_stuff thread_id= kwarg identity-threads must preserve)
- web-auth-drop-chat-admin — soft dep on web-auth-password-gate; deletes TestChatUploadDocument, invalidating one instruction in identity-threads' test plan
- web-chat-redesign-theme — soft dep on web-chat-redesign-quickwins; highest-blast-radius mechanical edit, keep it alone in its own commit for bisectability
- web-chat-redesign-chat-restyle — HARD dep on web-chat-redesign-theme (consumes its tokens)
- web-chat-redesign-streaming-ux — soft dep on web-chat-redesign-chat-restyle; SHOULD precede token-streaming so the applyToken() seam already exists
- web-chat-redesign-codehilite — HARD dep on web-auth-sanitiser (else highlighting is silently stripped); soft dep on web-chat-redesign-theme
- web-chat-redesign-chat-base-split — soft dep on web-chat-redesign-theme; soft dep on web-auth-drop-chat-admin (the doc-upload UI must be gone before the chrome is restructured)
- history-window — HARD dep on aimessage-role-fix; soft dep on config-knobs-batch
- token-streaming — HARD dep on history-window (both rewrite mister_fritz.py:411); HARD dep on aimessage-role-fix; must ADD AIMessageChunk rather than swap out AIMessage
- event-loop — no blockers; SHOULD precede discord-polish-bodies (structural rewrite of the same three command bodies)
- discord-polish-chunker — no blockers despite the item's declared DependsOn: token-streaming; fence-aware split_into_chunks + fritz_error touch nothing streaming-related
- discord-polish-bodies — HARD dep on event-loop (same three command bodies) and on discord-polish-chunker (consumes fritz_error); includes embeds, app-command error handler, /draw Range bound, temp_audio cleanup, and the uptime_sec key fix
- discord-polish-streaming-status — HARD dep on token-streaming (edit-scheduling policy) and on discord-polish-bodies
- latency-tax — no hard blockers, but WILL conflict textually with history-window in summarize_conversation and route_executor; land history-window first and rebase
- identity-threads — soft dep on web-auth-thread-split and web-auth-drop-chat-admin (otherwise you mint web:* identities that need re-migrating); soft dep on exec-env-scrub (its is_admin caller is positional and survives)
- deps-ci — soft dep on agent-tools-lazy-image-import (hard, within the item); MUST land after web-auth-sanitiser and web-chat-redesign-codehilite so the requirements regeneration absorbs nh3 and Pygments instead of dropping them


---

## Test infrastructure

"Verified: there is no conftest.py anywhere in the repo (find . -name conftest.py outside .venv returns nothing), and pyproject.toml configures only testpaths=[\"tests\"] and pythonpath=[\".\"]. A conftest.py should absolutely land as PR zero.\n\nThe pollution mechanism, read from the code rather than inferred: importing mister_fritz triggers three CWD-relative side effects at module scope — SQLiteStore(CHAT_DB_NAME) at :604 and SqliteSaver.from_conn_string(CHAT_DB_NAME) at :606 (CHAT_DB_NAME defaults to DB_NAME='fritz.db'), the Chroma store rooted at fritz_utils.CHROMA_DB_PATH='./chroma_store', and a daemon thread started at :652 that writes mister_fritz_diagram.png. document_engine does the same for its own diagram. That is exactly the debris in git status: tests/chat_history.db, tests/chroma_store/ and tests/mister_fritz_diagram.png untracked, plus 'M mister_fritz_diagram.png' and 'M document_engine_diagram.png' dirtied whenever the suite runs from the repo root. The repo-root PNG is currently 0 bytes, so the writer is already racing or failing.\n\nThe critical implementation constraint: the env vars must be set as TOP-LEVEL STATEMENTS in tests/conftest.py, not inside a fixture. Every test module imports the real module at import time — test_mister_fritz.py:29-30, test_agent_tools.py:36-44, test_admin_panel.py reloads fritz_utils/admin_panel — which happens during collection, long before any fixture body executes. A tmp_path-based autouse fixture is too late and will look like it works while the files still land in tests/.\n\nWhat PR zero should contain: (1) module-level os.environ.setdefault for DB_NAME, CHAT_DB_NAME, SCHEDULE_DB, CHROMA_DB_PATH, WORKSPACES_ROOT pointed at a per-session temp directory; (2) an env guard or monkeypatch that makes _write_diagram a no-op under test, since it is a daemon thread whose failure is swallowed and whose only effect is dirtying a tracked file; (3) the _ensure_mock sys.modules stub preamble currently duplicated across test_mister_fritz.py, test_agent_tools.py, test_bot_commands.py and test_discord_commands.py, hoisted to one place; (4) deletion of the three existing artifacts plus .gitignore entries; (5) either 'git worktree remove .claude/worktrees/nifty-hoover-6fe3d6' or adding it to ruff's extend-exclude, since it is a full duplicate checkout that will confuse every repo-wide search in every subsequent PR.\n\nBeyond conftest, three pieces of shared test infrastructure are each invented twice across the plans and should be created once: tests/test_bot_adapters.py (event-loop wants TestRunBlocking, discord-polish wants TestFenceAwareChunking and fritz_error tests — the module has no heavy imports, so tests there need none of the sys.modules preamble); the _login(client, username, password) helper in test_admin_panel.py (web-auth introduces it to rewrite 27 call sites, and identity-threads then edits many of the same tests); and a real decodable _TINY_PNG fixture (the current b'\\x89PNG\\r\\n\\x1a\\n' + nulls is not parseable by Pillow, which web-auth correctly caught).\n\nOne coverage caution that cuts across several PRs: CI gates at --cov-fail-under=60 with no documented headroom. latency-tax removes net lines from mister_fritz.py (good), but deps-ci's torch-free core job changes which modules get imported at all, and web-auth deletes a whole test class. Capture the baseline coverage number before starting and re-check it on any PR that deletes tests or changes what CI installs."


---

## PR sequence

| # | PR | Contains | Size | Unblocks | Risk |
|---|---|---|---|---|---|
| 0 | tests: add conftest.py sandboxing import-time side effects | NEW tests/conftest.py, module-level env: DB_NAME / CHAT_DB_NAME / SCHEDULE_DB / CHROMA_DB_PATH → per-session tmp, hoist the _ensure_mock sys.modules preamble duplicated in 4 test files, delete tests/chat_history.db, tests/chroma_store/, tests/mister_fritz_diagram.png; add to .gitignore, prune or ruff-exclude .claude/worktrees/nifty-hoover-6fe3d6 | S | Every subsequent PR. Makes test runs reproducible, stops the suite dirtying tracked diagram PNGs, and removes cross-test state bleed that will otherwise be blamed on feature PRs. | The env vars MUST be top-level statements in conftest.py, not a fixture — test modules import mister_fritz at module scope during collection, which is before any fixture runs. |
| 1 | agent_tools: lazy-import image_generator inside generate_image *(any order)* | deps-ci Step 1 only (agent_tools.py:20 → function-local at ~:306) | XS | deps-ci's entire extras split; makes the image_generator sys.modules stub in four test files inert rather than mandatory. | Two lines; if the import is placed after first use inside the function you get a NameError on the first /gen. |
| 2 | mister_fritz: store Fritz's replies as AIMessage, not bare str | history-window Commit 1 only: mister_fritz.py:451 and :503 wrapped in AIMessage(content=...) | XS | history-window (the window is meaningless without speaker roles), token-streaming, and admin_panel._doc_to_message — which currently paints every Fritz reply as a user bubble because add_messages coerces the bare str to HumanMessage. | Pre-existing checkpoints replay Fritz's old replies as user turns until the summariser's RemoveMessage sweep clears them — self-healing within ~8 exchanges. Do not write a migration; you cannot tell which historical HumanMessages were Fritz's. |
| 3 | config: add all new tunables in one pass (no behaviour change) *(any order)* | HISTORY_TOKEN_BUDGET, MEMORY_INJECT_MAX_CHARS, STREAM_MIN_CHARS, DISCORD_STREAM_MIN_INTERVAL, PLANNER_MODE, PLANNER_MIN_CHARS, SUMMARIZE_ASYNC, BLOCKING_POOL_SIZE, IMAGE_GEN_MAX_CONCURRENCY, TTS_MAX_CONCURRENCY, CHAT_CODE_HIGHLIGHT, DISCORD_ERROR_DETAIL, backfill CHAT_COOKIE_SECRET / CHAT_IMAGE_UPLOAD_MAX_BYTES / CHAT_DOC_UPLOAD_MAX_BYTES into .env.example, one TestConstantDefaults extension asserting the numeric tunables are sane | M | Seven items whose Commit 1 is otherwise an unmergeable insert at fritz_utils.py:97 or :119. | A typo'd default silently changes behaviour when the consuming PR lands weeks later; assert every numeric default in the test. |
| 4 | security: contain execute_command (env scrub, argv[0] bare-name, timeout clamp, admin gate) *(any order)* | exec-env-scrub Steps 2-5 and 7, _build_exec_env() + EXEC_ENV_PASSTHROUGH, bare-program-name rule in _validate_command_argv (closes the verified Windows git.bat bypass), EXEC_REQUIRE_ADMIN gate placed AFTER the shlex parse, _invoke_capturing_audit helper fix + the two python-dependent test wraps | M | Nothing downstream — but it closes a verified live hole where any /workspace enable user can print DISCORD_BOT_TOKEN and ADMIN_PANEL_PASSWORD into chat. | The scrubbed env breaks a legitimate command on the owner's host; EXEC_ENV_PASSTHROUGH exists so that is a config fix, not a code change. Requires the Config A/B decision up front. |
| 5 | web chat: sanitise rendered markdown with nh3 | web-auth Commit 1, NEW _sanitise_html() built from dict(nh3.ALLOWED_ATTRIBUTES) with class unioned onto div/pre/code/span/table, nh3>=0.3,<0.4 in requirements.txt, TestRenderMarkdown: script/onerror/javascript: stripping tests | S | web-chat-redesign's codehilite work (PR 13). Also closes the XSS path where python-markdown passes <script> verbatim into `\| safe` and innerHTML. | nh3's attributes= REPLACES the default map — start from dict(nh3.ALLOWED_ATTRIBUTES) or you also lose href on <a> and src/alt on <img>. |
| 6 | web chat: mobile viewport, working confirm dialog, a11y wiring *(any order)* | web-chat-redesign Step 1 (a)-(f), base.html viewport meta (fixes all eight templates), delete the inline onsubmit SyntaxError; register confirm via addEventListener ABOVE the early-return guard, role=log + #chat-status + aria-busy; prefers-reduced-motion guard, admin_panel.chat_send: missing is_admin and missing html keys | S | Nothing, but it is the only live data-loss bug in the set — the 'New conversation' confirm silently does not fire today. | If the listener is registered after `if (!form \|\| !list) return;` the confirm is silently disabled again; verify the Cancel path manually once, the attribute-absence test won't catch misplacement. |
| 7 | web chat: require a password at /chat/login | web-auth Commit 5, CHAT_PASSWORD (defaults to ADMIN_PANEL_PASSWORD, fail-closed if neither set), CHAT_ALLOWED_USERS, CHAT_COOKIE_SECURE, create_app(chat_password=...) + app.state, the _login(client, username, password) test helper + mechanical rewrite of all 27 /chat/login call sites | M | Every later test-heavy web PR (they inherit _login instead of each rewriting 27 sites). Closes the hole where anyone reaching :8001 can claim any username. | The route is unthrottled; secrets.compare_digest stops timing leaks but nothing rate-limits. Acceptable while bound to 127.0.0.1 — say so in the README rather than implying the surface is safe for untrusted users. |
| 8 | web chat: decode-based image sniffing and hardened asset serving *(any order)* | web-auth Commits 3 and 4, Pillow verify() sniff; canonical extension from the sniffed format, never the client's, /chat/assets content-type allowlist + nosniff + default-src 'none'; sandbox CSP + temp_images ownership check, replace the _TINY_PNG fixture (the current one is not a decodable PNG) and rewrite TestChatAsset | M | Nothing; closes the stored-XSS path where an HTML/SVG upload is served back same-origin. | Keep the size cap BEFORE the decode — it preserves the existing 413 test and bounds what Pillow is asked to parse. |
| 9 | web chat: separate the web LangGraph thread from Discord | web-auth Commit 6, ask_stuff(..., *, thread_id=None) keyword-only, privacy.forget_conversation optional thread_id, admin_panel._chat_thread_id replacing the duplicated regex at :353, CHAT_SHARE_DISCORD_THREAD rollback flag | M | identity-threads — establishes the thread_id= kwarg it must preserve rather than delete. Also deletes one of the three copies of the thread-id regex. | Existing web conversations appear to reset; ship it and say so in the CHANGELOG, or set CHAT_SHARE_DISCORD_THREAD=true for a single-user deployment. |
| 10 | web chat: remove admin privilege from the self-asserted chat identity | web-auth Commit 7, delete chat_upload_document + its route + the {% if is_admin %} block and uploader JS, re-home as POST /documents/upload behind the existing Basic-auth middleware, delete admin_panel's `import fritz_utils` (ruff F401 otherwise) | M | web-chat-redesign's chat_base split (the doc-upload UI must be gone before the chrome is restructured). | Deletes TestChatUploadDocument wholesale, which identity-threads' test plan still references — strike that line from identity-threads before starting it. |
| 11 | web chat: extract _theme.html and lay in the palette | web-chat-redesign Step 2, move base.html's entire <style> into a shared include; swap the eight hardcoded hexes for tokens, light + dark token sets, mobile breakpoint, focus rings | M | All later chat styling work. | Highest-blast-radius mechanical edit in the whole set — a dropped brace breaks all eight templates at once. Keep it alone in its own commit so a bisect is trivial. |
| 12 | web chat: restyle the transcript, sticky composer, streaming UX | web-chat-redesign Steps 3 and 4, serif prose / mono code, 16px inputs to stop iOS zoom, pre-wrap reflow fix, applyToken() / isPinnedToBottom() / followScroll() / showNotice() / setStatus(), Enter-to-send, AbortController Stop button, inline notices replacing the three alert() calls | L | token-streaming — applyToken() is the seam that reduces the cumulative→delta switch to a two-line change plus one assertion. | Stop is cosmetic server-side (the daemon thread and the Ollama call keep running); label it honestly or users will report it as broken. |
| 13 | web chat: syntax-highlighted code blocks and copy buttons | web-chat-redesign Step 5, codehilite with guess_lang=False; Pygments>=2.17 pinned explicitly, Pygments token CSS for light + dark in _theme.html, addCopyButtons() with the non-secure-context clipboard fallback | M | Nothing. | HARD dependency on PR 5 having allowed class on div/pre/code/span — verified this session that nh3's defaults strip them and reduce codehilite to inert markup with no error and no failing test. |
| 14 | web chat: split chat_base.html from the admin chrome *(any order)* | web-chat-redesign Step 6 and Step 7 docs, chat.html and chat_login.html reparented; {% block chrome_actions %} seam, keep <h2>Sign in to chat</h2> and the 'Type a message' placeholder verbatim — three and one test assert on them | M | Nothing; stops chat users seeing six nav links, five of which lead to Basic-auth walls. | A future admin-panel style added only to base.html will silently miss /chat; everything shared must go in _theme.html. |
| 15 | mister_fritz: feed a token-budgeted conversation window to the executor | history-window Commits 2-7 (Commit 1 already landed as PR 2), _history_window helper using trim_messages + count_tokens_approximately (note: the latter is in langchain_core.messages.utils, NOT .messages), per-branch inputs construction replacing the single hoisted assignment at :411, MEMORY_INJECT_MAX_CHARS cap on the Chroma blob | M | token-streaming (must rebase onto the per-branch inputs). | Adding up to 4096 history tokens moves the worst case closer to num_ctx=32768, at which point Ollama silently drops the front of the prompt including the entire Fritz persona. scrape_web is still untruncated. |
| 16 | streaming: real token deltas end-to-end | token-streaming Commits 1-8 (config already in PR 3), _chunk_text + _DeltaEmitter defining the (delta, accumulated, restart) contract, stream_mode=['values','messages'] in the executor; synthesizer deltas, main_discord callback arity + cross-thread scheduling guard; admin_panel delta frames + reset event; chat.html applyToken flip, the LangGraph messages-mode canary test | M | discord-polish's StreamingMessageHandler status line. | ADD AIMessageChunk to the import, do NOT swap out AIMessage — PR 2 introduced two new uses and the plan's stated rationale is stale. Ruff F821 will catch it, confusingly. |
| 17 | discord: get blocking work off the event loop *(any order)* | event-loop Steps 2-8 (config already in PR 3), bot_adapters.run_blocking + bounded pool, TTSEngine.generate_speech becomes genuinely synchronous (+ instance lock), image_generator _PIPELINE_LOCK; per-cog asyncio semaphores; on_message onto the same pool, _fake_interaction gains defer/followup.send/channel.send AsyncMocks | M | discord-polish's rewrite of the same three command bodies. | Keep the semaphores on the cog instance — hoisting them to module scope binds them to whichever test's loop ran first and wedges /gen after a production reconnect. |
| 18 | discord: fence-aware chunking and a butler-voiced error helper *(any order)* | discord-polish Steps 1-3, boundary- and fence-aware split_into_chunks, bot_adapters.fritz_error(operation, exc) + DISCORD_ERROR_DETAIL, NEW tests/test_bot_adapters.py | M | discord-polish's body rewrites; also gives admin_panel.chat_stream a helper for its own raw str(e) leak. | All six existing TestSplitIntoChunks tests must survive unmodified — run them FIRST after the rewrite and reconcile before touching anything else. The item's declared DependsOn: token-streaming does not apply to this part. |
| 19 | discord: embeds, app-command error handler, /draw bound, butler copy | discord-polish Steps 4-6 and 8-11, cog_app_command_error + tree.on_error backstop, /help /about /health /schedule list as embeds; /health ephemeral, fix the uptime_seconds → uptime_sec key bug (verified: observability.py:304 emits uptime_sec, bot_commands.py:367 reads uptime_seconds, so /about always shows 0s), app_commands.Range[int,1,40] on /draw; temp_audio cleanup | M | Nothing. | Leave bot_commands.py:149 and :190 untouched — they carry author-written validation text that two tests assert on. |
| 20 | discord: in-placeholder progress status and chunked final update | discord-polish Step 7, StreamingMessageHandler status_text / _compose / set_status, tail window replacing the [:2000] head-truncation, chunking moved into final_update; the duplicated block in on_message deleted | S | Nothing. | Progress notices now compete with token edits for the 1.5s edit budget; a 6-tool plan will coalesce some away. Resolve the edit-scheduling policy here, with token-streaming already in place. |
| 21 | latency: heuristic planner gate, off-path summarisation, structured outputs | latency-tax Commits 1-10 (config already in PR 3), should_run_planner + route_start conditional START edge, summarize_conversation snapshot + daemon thread; _PROFILE_LOCK in agent_tools, Pydantic structured output at the three JSON-scrape sites; _make_memory_key replacing a 20B call, dead work: the debug-only system_prompt rebuild and unconditional pretty_print() | L | Nothing. | Structured output via format=<schema> may stall small models under a tight GBNF grammar — unverifiable statically, and the four rewritten planner tests will pass VACUOUSLY against a bare MagicMock unless each returns a real PlanDecision. |
| 22 | identity: canonical namespaced user ids and per-channel threads | identity-threads Steps 1-9, fritz_utils.canonical_user_id / thread_id_for / safe_user_token / split_user_id, NEW identity_store.py; privacy.py thread predicate fix, adapter cut-over at all four boundaries; migrate_identity.py (dry-run default), THREADS_PER_CHANNEL shipped false | XL | Nothing — it is the terminal cleanup. | A colon in 'discord:123' used in a filename creates a silent NTFS alternate data stream on the owner's actual OS; route every filename through safe_user_token. Strongly consider '-' as the separator instead. |
| 23 | deps: declare intent in pyproject, purge the polluted freeze, slim CI | deps-ci Steps 2-11 (Step 1 already landed as PR 1), [project] table with voice/image/ocr/telegram/browser/dev/all extras, requirements.txt regenerated from a clean venv; the fitz vs PyMuPDF shadowing removed, pip caching + torch-free core CI job + weekly full-deps job, scripts/check_imports.py and tests/test_packaging.py | L | Nothing. | The regeneration will DROP nh3 and Pygments unless both are added to [project] dependencies — that would silently un-fix the XSS hole from PR 5. Add an explicit check to the Step 4 diff review. |
