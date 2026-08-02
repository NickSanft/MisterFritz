# 1. Feed conversation history to the executor

[← back to index](README.md)

**Effort:** M (half day)  
**Depends on:** nothing

## Goal
Today Mister Fritz is functionally amnesiac inside a single thread: the executor hands the sub-agent exactly one system message and one user message, so the LLM never sees the previous turns even though LangGraph faithfully checkpoints them. When this is done, the executor passes the newest slice of `state["messages"]` that fits a configurable token budget (`HISTORY_TOKEN_BUDGET`, default 4096) into the ReAct agent, trimmed with `langchain_core.messages.trim_messages`, so follow-ups like "do that again but in French" and "what did I just ask you?" resolve from real transcript instead of a lossy Chroma re-embedding. Chroma memory injection stays exactly where it is and keeps its job (long-range recall across threads/sessions); the window covers short-range continuity. As a hard prerequisite, Fritz's own replies stop being written into state as `HumanMessage` (which is what `add_messages` does with the bare `str` the executor returns today) and become `AIMessage`, so the window actually carries speaker roles — this simultaneously fixes the web chat history renderer, which currently paints every Fritz reply as a user bubble.

## Definition of done

- [ ] `executor()` in simple mode sends the agent `[SystemMessage, *window]` where `window` is the trimmed suffix of `state["messages"]` and `window[-1]` is the current user turn — verified by a unit test that records what the model received.
- [ ] The current user turn appears exactly once in the agent input (no duplicate `("user", latest_message)` appended after the window).
- [ ] `HISTORY_TOKEN_BUDGET=0` reproduces today's exact behaviour byte-for-byte in BOTH simple mode and plan mode (kill switch / rollback path), covered by a test.
- [ ] When the newest single message alone exceeds the budget, `trim_messages` returns `[]` and the executor still sends that message (never sends a system-prompt-only input to the model).
- [ ] `executor()` simple-mode return is `{"messages": [AIMessage(content=resp)], ...}` and `synthesizer()` return is `{"messages": [AIMessage(content=accumulated_text)], ...}` — no bare `str` is ever handed to the `add_messages` reducer.
- [ ] Plan mode passes `_history_window(messages[:-1])` (current turn excluded, since it is already restated verbatim as `original_request` in the step prompt) followed by the synthetic step prompt.
- [ ] The auto-injected Chroma memory blob is capped at `MEMORY_INJECT_MAX_CHARS` (default 4000) so the system prompt cannot silently evict the window from `num_ctx=32768`.
- [ ] `HISTORY_TOKEN_BUDGET` and `MEMORY_INJECT_MAX_CHARS` are read via `fritz_utils` env-var config and documented in `.env.example` under `# ----- Tunables -----`.
- [ ] `ruff check .` clean; `pytest tests/ --cov=. --cov-fail-under=60` green (baseline today: `pytest tests/test_mister_fritz.py -q` = 12 passed).
- [ ] CHANGELOG.md has a phase-style bullet under `## [Unreleased]`; README.md's ASCII architecture block (line ~232) no longer implies the executor is stateless.

## Current state (verified against the working tree)
VERIFIED IN THIS SESSION — every claim below was read out of the file, and I re-ran the relevant library calls locally.

1. `mister_fritz.py:406-411` — the audit's line numbers are still exact. Simple mode is:
```
406    else:
407        latest_message = messages[-1].content if messages else original_request
408        agent_prompt = latest_message
409        effective_streaming_callback = streaming_callback
410
411    inputs = {"messages": [("system", system_prompt), ("user", agent_prompt)]}
```
`state["messages"]` (bound at `:284`) is read only for `messages[-1]`. Everything before it is discarded.

2. Sub-agent has no checkpointer — confirmed. `mister_fritz.py:321` (`create_agent(ollama_instance, tools=active_tools)`) and `:625` (`create_agent(ollama_instance, tools=conversation_tools)`) both omit `checkpointer=`. `get_config_values` (`:510-518`) rebuilds `configurable` from scratch with only `user_id`/`thread_id`, so the parent `SqliteSaver` (`:606`, wired at `:640`) cannot leak in. Checkpointed history accumulates and is never read by the model.

3. **The audit under-reports the severity.** `add_messages` coerces bare strings to `HumanMessage`, not `AIMessage`. Verified:
```
>>> add_messages([HumanMessage(content='hi')], ['I am Fritz, sir.'])
[HumanMessage('hi'), HumanMessage('I am Fritz, sir.')]
```
`executor` returns `{"messages": [resp]}` at `:450-453` and `synthesizer` returns `{"messages": [accumulated_text]}` at `:502-507`, both bare `str`. So the checkpointed transcript for every thread is an **unbroken run of HumanMessages** — Fritz's replies are indistinguishable from the user's. Feeding that list to the model as-is would be worse than useless. Fixing the role tag is a prerequisite, not an optional polish.

4. This same defect is live in the web chat. `admin_panel.py:326-341` `_doc_to_message` maps `msg_type == "human"` → `{"role": "user"}` and `"ai"` → `{"role": "fritz"}`; `_load_chat_history` (`:344-368`) feeds it straight from `agent_app.get_state(config).values["messages"]`. Because everything is `human` today, the last-40-messages hydration on `/chat` renders Fritz's replies as user bubbles. The `AIMessage` fix repairs that for free (for turns written after the change).

5. Memory + profile injection confirmed at `:326-335` and `:337-362`, both appending to `system_prompt`. `search_memories_internal` (`agent_tools.py:75-84`) calls `ChromaStore.search(..., limit=30)`, which is `vectorstore.similarity_search(k=limit)` (`storage.py:185-192`) — relevance-ordered, dict-deduped, `json.dumps`'d, and **completely uncapped**. Each stored value can be a full conversation summary written by `summarize_conversation` (`:241`). This block is the single biggest unbounded consumer of `num_ctx` and must be capped for the history budget to mean anything.

6. `SUMMARIZE_THRESHOLD=15` (`fritz_utils.py:97`); `should_continue` (`:136-138`) and `route_executor` (`:141-157`) both gate on `len(state["messages"]) > SUMMARIZE_THRESHOLD`; `summarize_conversation` deletes all but the last message via `RemoveMessage` at `:271`. So the parent message list is hard-bounded at 16 entries — the token budget is a safety valve for pasted walls of text, not the normal binding constraint.

7. `num_ctx 32768` confirmed in all five files under `modelfiles/`.

8. Measured sizes (via `count_tokens_approximately`, this repo, this session):
   - base system prompt: 2935 chars / **739 tokens**; with file tools: 3551 chars / **893 tokens**
   - bound tool JSON schemas: **1019 tokens** (10 tools) / **1897 tokens** (16 tools with file tools)
   - `format_prompt` wrapper overhead: **96 chars (~24 tokens) per stored user turn**

9. Library facts, verified against the *installed* versions (`langchain-core 1.2.19` — note `requirements.txt:114` pins `1.2.5`, the venv has drifted):
   - `trim_messages` IS exported from `langchain_core.messages`.
   - `count_tokens_approximately` is **NOT** exported from `langchain_core.messages` in 1.2.19 — `ImportError`. It lives at `langchain_core.messages.utils`. This is a real trap; use the string literal `token_counter="approximate"` for the trim itself.
   - `trim_messages(..., strategy="last", start_on="human", allow_partial=False)` **returns `[]`** when the newest message alone busts the budget. Verified. Must be guarded.
   - Message identity is preserved by trim (`[m is orig for ...] == [True, True, True]`), so ids/roles survive.
   - End-to-end smoke test through a real `create_agent` with a recording fake model confirms `{"messages": [("system", sp), *window]}` reaches the model as `[('system',...), ('human','turn one'), ('ai','reply one'), ('human','turn two')]`, and `out["messages"][-1]` is still the fresh `AIMessage` (so `:435` and `:438-441` keep working).

10. Test baseline: `tests/test_mister_fritz.py` covers **only** `planner()` (12 tests, all passing). There is no existing test of `executor`, `synthesizer`, `should_continue`, or `route_executor`. `tests/test_admin_panel.py` mocks the entire `mister_fritz` module via `patch.dict(sys.modules, ...)`. **No existing test breaks.** I am stating that as a verified fact, not a hope.

## Change sites

### `fritz_utils.py:96-97`

Add two tunables directly under the existing SUMMARIZE_THRESHOLD block, following the file's comment-then-constant convention.

# BEFORE (fritz_utils.py:96-97)
# Number of conversation messages before the agent triggers a summarisation pass.
SUMMARIZE_THRESHOLD: int = int(os.environ.get("SUMMARIZE_THRESHOLD", "15"))

# AFTER — append immediately below:

# Token budget for the slice of conversation history handed to the executor's
# ReAct sub-agent each turn. The sub-agent is compiled WITHOUT a checkpointer,
# so this window is the only short-term memory the model gets; Chroma memory
# injection covers long-range recall. Sized against num_ctx=32768 (see
# modelfiles/): ~900 system prompt + ~1900 tool schemas + ~1000 injected
# memories + 4096 history still leaves ~24k for tool output and the reply.
# Set to 0 to disable the window entirely and restore pre-window behaviour.
HISTORY_TOKEN_BUDGET: int = int(os.environ.get("HISTORY_TOKEN_BUDGET", "4096"))

# Hard cap (characters) on the Chroma memory blob auto-injected into the
# system prompt. search_memories_internal pulls up to 30 stored summaries with
# no size limit; uncapped, that block alone can evict the history window — and
# the system prompt itself — from the model's context.
MEMORY_INJECT_MAX_CHARS: int = int(os.environ.get("MEMORY_INJECT_MAX_CHARS", "4000"))

### `mister_fritz.py:8`

Add trim_messages to the existing langchain_core.messages import, and add a separate import for count_tokens_approximately from .utils (it is NOT re-exported from .messages in langchain-core 1.2.19 — verified ImportError).

# BEFORE (line 8)
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

# AFTER
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage, trim_messages
# NOT exported from langchain_core.messages in 1.2.19 — import from .utils explicitly.
from langchain_core.messages.utils import count_tokens_approximately

### `mister_fritz.py:29-38`

Extend the fritz_utils import block with the two new constants (keep alphabetical order — the block is sorted), and pull METRICS in alongside init_logging.

# BEFORE (29-38)
from fritz_utils import (
    CHAT_DB_NAME,
    FAST_OLLAMA_MODEL,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    SUMMARIZE_THRESHOLD,
    THINKING_OLLAMA_MODEL,
)
from observability import init_logging

# AFTER
from fritz_utils import (
    CHAT_DB_NAME,
    FAST_OLLAMA_MODEL,
    HISTORY_TOKEN_BUDGET,
    MEMORY_INJECT_MAX_CHARS,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    SUMMARIZE_THRESHOLD,
    THINKING_OLLAMA_MODEL,
)
from observability import METRICS, init_logging

### `mister_fritz.py:273-274`

Insert the _history_window helper in the blank lines between summarize_conversation (ends :272) and executor (starts :275).

def _history_window(messages: list) -> list:
    """Newest slice of the conversation that fits HISTORY_TOKEN_BUDGET.

    The executor's ReAct agent is compiled without a checkpointer, so whatever
    this returns is the model's entire short-term memory for the turn. Returns
    [] when there is nothing usable — callers must handle that and fall back to
    the single latest message.
    """
    if not messages or HISTORY_TOKEN_BUDGET <= 0:
        return []
    try:
        window = trim_messages(
            messages,
            max_tokens=HISTORY_TOKEN_BUDGET,
            token_counter="approximate",   # chars/4 heuristic; no tokenizer for gpt-oss
            strategy="last",
            start_on="human",              # never open the window on a dangling AI turn
            include_system=False,
            allow_partial=False,           # never split a message mid-content
        )
    except Exception as e:
        logger.warning("History trim failed (non-fatal): %s", e)
        return []
    if not window:
        # trim_messages returns [] when the newest message alone busts the
        # budget (verified). Caller falls back to sending just that message.
        METRICS.increment("history_window_overflow")
        return []
    if len(window) < len(messages):
        METRICS.increment("history_window_trimmed")
    logger.debug(
        "History window: %d/%d messages, ~%d tokens",
        len(window), len(messages), count_tokens_approximately(window),
    )
    return window

### `mister_fritz.py:327-333`

Cap the injected Chroma memory blob. Drops least-relevant entries first — ChromaStore.search is similarity-ordered (storage.py:185-192) and search_memories_internal preserves that order in its dict, so keeping the head keeps the best matches.

# BEFORE (327-333)
    try:
        memory_query = original_request or (messages[-1].content if messages else "")
        if memory_query and user_id:
            past_context = search_memories_internal(config, memory_query)
            if past_context and past_context != "{}":
                system_prompt = system_prompt + f"\n\nWhat I know about this user:\n{past_context}"
                logger.debug("Injected memory context for %s (%d chars)", user_id, len(past_context))

# AFTER
    try:
        memory_query = original_request or (messages[-1].content if messages else "")
        if memory_query and user_id:
            past_context = search_memories_internal(config, memory_query)
            if past_context and past_context != "{}":
                if len(past_context) > MEMORY_INJECT_MAX_CHARS:
                    try:
                        # Chroma returns most-similar-first and the dict keeps that
                        # order, so truncating the tail drops the weakest matches.
                        kept, size = {}, 0
                        for k, v in json.loads(past_context).items():
                            size += len(k) + len(str(v))
                            if size > MEMORY_INJECT_MAX_CHARS:
                                break
                            kept[k] = v
                        past_context = json.dumps(kept)
                    except Exception:
                        past_context = past_context[:MEMORY_INJECT_MAX_CHARS]
                    METRICS.increment("memory_inject_truncated")
                system_prompt = system_prompt + f"\n\nWhat I know about this user:\n{past_context}"
                logger.debug("Injected memory context for %s (%d chars)", user_id, len(past_context))

### `mister_fritz.py:388-411`

The core change: build `inputs` per-branch instead of once at :411. Simple mode sends the window (whose last element already IS the current turn). Plan mode sends the window minus the current turn, then the synthetic step prompt.

# BEFORE (388-411)
    if is_plan_mode:
        step_instruction = plan[current_step]
        ... context_parts built ...
        agent_prompt = "\n".join(context_parts)
        if progress_callback:
            progress_callback(f"Step {current_step + 1}/{len(plan)}: {step_instruction}")
        effective_streaming_callback = None
    else:
        latest_message = messages[-1].content if messages else original_request
        agent_prompt = latest_message
        effective_streaming_callback = streaming_callback

    inputs = {"messages": [("system", system_prompt), ("user", agent_prompt)]}

# AFTER — context_parts block at 389-401 is untouched; only the tail changes
    if is_plan_mode:
        step_instruction = plan[current_step]
        ... context_parts built (UNCHANGED) ...
        agent_prompt = "\n".join(context_parts)
        if progress_callback:
            progress_callback(f"Step {current_step + 1}/{len(plan)}: {step_instruction}")
        effective_streaming_callback = None
        # messages[-1] is this very request, already restated verbatim as
        # "Original request:" inside agent_prompt — excluding it avoids a
        # duplicate and keeps the window identical across every step of a plan.
        history = _history_window(messages[:-1])
        inputs = {"messages": [("system", system_prompt), *history, ("user", agent_prompt)]}
    else:
        effective_streaming_callback = streaming_callback
        history = _history_window(messages)
        if history:
            # history[-1] IS the current user turn (ask_stuff appended it before
            # the planner ran), so no separate ("user", ...) entry is needed.
            inputs = {"messages": [("system", system_prompt), *history]}
        else:
            # Budget disabled, empty state, or a single oversized message.
            latest_message = messages[-1].content if messages else original_request
            inputs = {"messages": [("system", system_prompt), ("user", latest_message)]}

### `mister_fritz.py:449-453`

PREREQUISITE FIX. Wrap the executor's reply in AIMessage so add_messages stops coercing it to HumanMessage. Without this the window replays Fritz's own answers as if the user said them, and admin_panel._doc_to_message keeps mislabelling them.

# BEFORE (449-453)
    else:
        return {
            "messages": [resp],
            "image_paths": image_paths,
        }

# AFTER
    else:
        # Bare strings are coerced to HumanMessage by the add_messages reducer,
        # which makes the checkpointed transcript unreadable (and mislabels
        # every reply in the /chat history renderer). Tag it explicitly.
        return {
            "messages": [AIMessage(content=resp)],
            "image_paths": image_paths,
        }

### `mister_fritz.py:502-507`

Same fix on the plan-mode exit path — synthesizer also returns a bare str today.

# BEFORE (502-507)
    return {
        "messages": [accumulated_text],
        "plan": [],
        "current_step": 0,
        "step_results": [],
    }

# AFTER
    return {
        "messages": [AIMessage(content=accumulated_text)],
        "plan": [],
        "current_step": 0,
        "step_results": [],
    }

### `.env.example:42-43`

Document both knobs immediately after the existing SUMMARIZE_THRESHOLD entry, and note the relationship between them (the summariser is what bounds the window in practice).

# BEFORE (42-43)
# Messages-in-conversation threshold before the agent summarises and trims.
# SUMMARIZE_THRESHOLD=15

# AFTER
# Messages-in-conversation threshold before the agent summarises and trims.
# This is what bounds the history window in practice: the window can never hold
# more than SUMMARIZE_THRESHOLD+1 messages, because the summariser removes all
# but the last one. Raise both together for longer in-thread continuity.
# SUMMARIZE_THRESHOLD=15
# Token budget for the conversation history fed to the agent each turn. The
# window is a suffix of the thread transcript, trimmed to fit. Budget against
# num_ctx (32768 in modelfiles/): ~900 system prompt + ~1900 tool schemas +
# MEMORY_INJECT_MAX_CHARS/4 + this + room for tool output and the reply.
# Set to 0 to disable the window and send only the latest message.
# HISTORY_TOKEN_BUDGET=4096
# Cap (characters) on the memory blob auto-injected into the system prompt from
# Chroma. Uncapped it can be tens of thousands of chars and evict everything else.
# MEMORY_INJECT_MAX_CHARS=4000

### `tests/test_mister_fritz.py:1-8, 128`

Update the module docstring (it currently says "Tests for the LangGraph planner node") and append three new TestCase classes. Reuse the existing _ensure_mock preamble at lines 14-25 rather than duplicating it in a new file.

# Add to the existing imports at line 27:
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from mister_fritz import _history_window, executor, planner  # noqa: E402


def _thread(n_pairs: int, body: str = "x" * 40) -> list:
    """A realistic checkpointed transcript: alternating Human/AI turns."""
    msgs = []
    for i in range(n_pairs):
        msgs.append(HumanMessage(content=f"q{i} {body}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i} {body}", id=f"a{i}"))
    return msgs


class _RecordingAgent:
    """Stands in for the compiled ReAct agent; captures the inputs dict."""
    def __init__(self):
        self.seen = None
    def stream(self, inputs, config=None, stream_mode=None):
        self.seen = inputs
        yield {"messages": [AIMessage(content="Very well, sir.")]}


class TestHistoryWindow(unittest.TestCase):
    def test_returns_all_messages_when_under_budget(self): ...
    def test_trims_oldest_when_over_budget(self): ...
    def test_returns_empty_when_newest_message_alone_exceeds_budget(self): ...
    def test_returns_empty_for_empty_input(self): ...
    def test_zero_budget_returns_empty_kill_switch(self): ...
    def test_window_never_opens_on_an_ai_message(self): ...


class TestExecutorInputs(unittest.TestCase):
    # patch.object(mister_fritz, "_get_conversation_agent", return_value=rec),
    # patch.object(mister_fritz, "search_memories_internal", return_value="{}"),
    # patch.object(mister_fritz, "get_user_profile", return_value={})
    def test_simple_mode_passes_full_window_with_roles(self): ...
    def test_simple_mode_does_not_duplicate_latest_turn(self): ...
    def test_simple_mode_returns_ai_message_not_bare_string(self): ...
    def test_zero_budget_reproduces_single_message_input(self): ...
    def test_plan_mode_excludes_current_turn_and_appends_step_prompt(self): ...


class TestMemoryInjectionCap(unittest.TestCase):
    def test_oversized_memory_blob_is_truncated_to_cap(self): ...
    def test_small_memory_blob_passes_through_unmodified(self): ...

### `tests/test_fritz_utils.py:99-120`

Add one test method to the existing TestConstantDefaults class covering the new tunables (there is currently no test asserting numeric tunables are sane).

    def test_numeric_tunables_are_sane(self):
        self.assertGreater(fu.SUMMARIZE_THRESHOLD, 0)
        self.assertGreaterEqual(fu.HISTORY_TOKEN_BUDGET, 0)  # 0 = window disabled
        self.assertGreater(fu.MEMORY_INJECT_MAX_CHARS, 0)
        for attr in ("SUMMARIZE_THRESHOLD", "HISTORY_TOKEN_BUDGET", "MEMORY_INJECT_MAX_CHARS"):
            with self.subTest(attr=attr):
                self.assertIsInstance(getattr(fu, attr), int)

### `CHANGELOG.md:8-9`

Add a phase-style bullet at the top of ## [Unreleased], under a ### Fixed heading (the file has Performance / Added / Changed today; this is genuinely a correctness fix plus a tunable).

## [Unreleased]

### Fixed
- **Phase 15 — the executor can finally see the conversation.** The ReAct sub-agent was being handed exactly `[system_prompt, latest_user_message]` every turn and is compiled without a checkpointer, so LangGraph's per-user SQLite history accumulated but the model never read a word of it — in-thread continuity was entirely dependent on lossy background Chroma extraction. The executor now passes a token-budgeted suffix of `state["messages"]` (`langchain_core.messages.trim_messages`, `strategy="last"`, approximate char/4 counter — there is no tokenizer for `gpt-oss`), governed by the new `HISTORY_TOKEN_BUDGET` env var (default 4096, `0` disables). Chroma memory injection is unchanged and keeps its job: cross-thread long-range recall.
  - Prerequisite fix: `executor` and `synthesizer` returned their reply as a bare `str`, which the `add_messages` reducer coerces to a **`HumanMessage`**. Every thread's checkpointed transcript was therefore an unbroken run of user turns with Fritz's replies indistinguishable from the user's. Both now return `AIMessage`. Side effect: the `/chat` history hydration (`admin_panel._doc_to_message`), which keys off `.type`, stops rendering Fritz's replies as user bubbles.
  - The auto-injected Chroma memory blob is now capped by `MEMORY_INJECT_MAX_CHARS` (default 4000). `search_memories_internal` pulls up to 30 stored conversation summaries with no size limit; uncapped it could evict the window — and the system prompt itself — from `num_ctx=32768`. Least-relevant entries are dropped first (Chroma returns similarity-ordered).

### `README.md:232-235`

The architecture block and the sentence after it imply history is retained and used. Correct it to describe what the executor actually receives.

# BEFORE (232-235)
    └─ SUMMARIZE_NODE   ──  auto-summarise at 15+ messages, store to Chroma
```

Conversation state is checkpointed per-user in `chat_history.db` (SQLite).

# AFTER
    └─ SUMMARIZE_NODE   ──  auto-summarise at 15+ messages, store to Chroma
```

Conversation state is checkpointed per-user in `chat_history.db` (SQLite). Each
turn the executor replays the newest slice of that transcript that fits
`HISTORY_TOKEN_BUDGET` (default 4096 tokens) into the ReAct agent — short-term
continuity. Anything older is reachable only through the Chroma memory store,
which the summariser writes to and which is auto-injected into the system
prompt (capped at `MEMORY_INJECT_MAX_CHARS`) — long-term recall.

## Steps

1. Commit 1 — role tagging (independently valuable, ship first). Change `mister_fritz.py:451` to `AIMessage(content=resp)` and `:503` to `AIMessage(content=accumulated_text)`. `AIMessage` is already imported at line 8; no new imports. Run `pytest tests/ -q` — expect zero failures (verified: nothing asserts on these return shapes). Manually confirm via `/chat` that new replies now render as Fritz bubbles.
2. Commit 2 — config. Add `HISTORY_TOKEN_BUDGET` and `MEMORY_INJECT_MAX_CHARS` to `fritz_utils.py` below `SUMMARIZE_THRESHOLD` (line 97), document both in `.env.example` after line 43, add `test_numeric_tunables_are_sane` to `tests/test_fritz_utils.py::TestConstantDefaults`.
3. Commit 3 — the helper. Add the `trim_messages` + `count_tokens_approximately` imports (line 8; remember `count_tokens_approximately` comes from `langchain_core.messages.utils`, NOT `langchain_core.messages` — it will `ImportError` otherwise on 1.2.19), extend the `fritz_utils` import block, swap `from observability import init_logging` → `from observability import METRICS, init_logging`, then insert `_history_window` between `summarize_conversation` and `executor`. Add `TestHistoryWindow` to `tests/test_mister_fritz.py`. This commit changes no behaviour — the helper is not called yet.
4. Commit 4 — wire it in. Rewrite the `is_plan_mode` / `else` tail at `mister_fritz.py:388-411` so each branch builds its own `inputs`. Delete the standalone `inputs = ...` line at 411. Do not touch the `context_parts` construction (389-401), the streaming loop (416-433), the `resp` extraction (435), or the `image_paths` scan (437-441) — all verified compatible with a longer input list.
5. Commit 5 — cap the memory blob at `mister_fritz.py:327-333`. `json` is already imported at line 1.
6. Commit 6 — tests. Add `TestExecutorInputs` and `TestMemoryInjectionCap` to `tests/test_mister_fritz.py`, update its module docstring (it says "Tests for the LangGraph planner node"). Patch `mister_fritz._get_conversation_agent` with a recording stub exposing `.stream(inputs, config=..., stream_mode=...)`; also patch `search_memories_internal` → `"{}"` and `get_user_profile` → `{}` so the system prompt stays deterministic. Note `executor` takes the no-file-tools path (and therefore `_get_conversation_agent()`) only when `metadata` has no `workspace_root`, no `channel_id`, and no `schedule_manager` — pass `config={"metadata": {"user_id": "tester"}}`.
7. Commit 7 — docs. CHANGELOG.md `### Fixed` bullet under `## [Unreleased]`; README.md lines 232-235.
8. Verify: `ruff check .` then `pytest tests/ --tb=short --cov=. --cov-fail-under=60 -q`.
9. Live smoke test against real Ollama with `LOG_LEVEL=DEBUG` — see manualVerification. This is the only step that proves the model actually uses the window.

## Config and env changes

- `HISTORY_TOKEN_BUDGET` (int, default `4096`) — token budget for the conversation window fed to the executor's ReAct agent. `0` disables the window and restores the previous single-message behaviour in both simple and plan mode. Added to `fritz_utils.py` and documented in `.env.example`.
- `MEMORY_INJECT_MAX_CHARS` (int, default `4000`) — hard cap on the Chroma memory blob appended to the system prompt at `mister_fritz.py:332`. Added to `fritz_utils.py` and documented in `.env.example`.
- No change to `SUMMARIZE_THRESHOLD` (stays 15). Rationale in openQuestions — the summariser already caps the window at 16 messages, so raising the threshold, not the budget, is the lever for longer continuity, and raising it also raises the summariser's own invoke cost (it feeds the whole list to the thinking model at `mister_fritz.py:231`). The relationship is documented in `.env.example` so an operator can raise both together.
- Budget arithmetic against `num_ctx=32768` (all figures measured in this repo, not estimated): 893 (system prompt w/ file tools) + 1897 (bound tool schemas, 16 tools) + ~1000 (memory blob at the 4000-char cap) + ~200 (profile block) + 4096 (history) ≈ 8.1k, leaving ~24.6k for ReAct tool results, reasoning, and the reply.

## Tests
### New

- tests/test_mister_fritz.py::TestHistoryWindow::test_returns_all_messages_when_under_budget — 6 short turns, budget 4096, asserts len(window) == len(messages) and identity is preserved.
- tests/test_mister_fritz.py::TestHistoryWindow::test_trims_oldest_when_over_budget — 20 turns of 400-char bodies, budget 500, asserts 0 < len(window) < len(messages) and window[-1] is messages[-1].
- tests/test_mister_fritz.py::TestHistoryWindow::test_returns_empty_when_newest_message_alone_exceeds_budget — HumanMessage('x'*100000) as the tail; asserts _history_window(...) == [] (verified real trim_messages behaviour) so callers know to fall back.
- tests/test_mister_fritz.py::TestHistoryWindow::test_returns_empty_for_empty_input.
- tests/test_mister_fritz.py::TestHistoryWindow::test_zero_budget_returns_empty_kill_switch — patch.object(mister_fritz, 'HISTORY_TOKEN_BUDGET', 0); asserts [].
- tests/test_mister_fritz.py::TestHistoryWindow::test_window_never_opens_on_an_ai_message — input [AIMessage, HumanMessage, AIMessage, HumanMessage]; asserts window[0].type == 'human' (start_on='human' guard).
- tests/test_mister_fritz.py::TestExecutorInputs::test_simple_mode_passes_full_window_with_roles — recording agent; asserts the inputs list is [('system', ...), Human, AI, Human, AI, Human] with roles alternating and the system entry first.
- tests/test_mister_fritz.py::TestExecutorInputs::test_simple_mode_does_not_duplicate_latest_turn — asserts the current turn's content appears exactly once across inputs['messages'].
- tests/test_mister_fritz.py::TestExecutorInputs::test_simple_mode_returns_ai_message_not_bare_string — asserts isinstance(result['messages'][0], AIMessage). This is the regression guard for the add_messages coercion bug; without it the defect silently returns.
- tests/test_mister_fritz.py::TestExecutorInputs::test_zero_budget_reproduces_single_message_input — patch HISTORY_TOKEN_BUDGET=0; asserts inputs['messages'] == [('system', sp), ('user', latest_content)], i.e. byte-identical to pre-change behaviour. This is the rollback contract.
- tests/test_mister_fritz.py::TestExecutorInputs::test_plan_mode_excludes_current_turn_and_appends_step_prompt — state with plan=['a','b'], current_step=0; asserts the current turn is NOT in the window and the final entry is ('user', <step prompt containing 'Current task (step 1/2)'>).
- tests/test_mister_fritz.py::TestMemoryInjectionCap::test_oversized_memory_blob_is_truncated_to_cap — patch search_memories_internal to return json.dumps of 30 x 2000-char summaries and MEMORY_INJECT_MAX_CHARS to 4000; asserts the injected system string is bounded and the highest-relevance (first) key survives.
- tests/test_mister_fritz.py::TestMemoryInjectionCap::test_small_memory_blob_passes_through_unmodified.
- tests/test_fritz_utils.py::TestConstantDefaults::test_numeric_tunables_are_sane — SUMMARIZE_THRESHOLD / HISTORY_TOKEN_BUDGET / MEMORY_INJECT_MAX_CHARS are ints with sane bounds.

### Existing tests affected

- NONE BREAK — verified, not assumed. Full audit of every test that touches this code: tests/test_mister_fritz.py contains only TestPlannerParsing (12 tests, all exercising planner(); baseline `pytest tests/test_mister_fritz.py -q` = 12 passed in this session). There is no existing test of executor, synthesizer, should_continue, or route_executor.
- tests/test_mister_fritz.py — module docstring at lines 1-8 says "Tests for the LangGraph planner node in mister_fritz.py"; it becomes inaccurate once the new classes land. Update it (documentation change, not a failure).
- tests/test_mister_fritz.py:30 — `from mister_fritz import planner` must be extended to `from mister_fritz import _history_window, executor, planner`.
- tests/test_admin_panel.py::TestChatHistory::test_authed_returns_messages_from_loader (line 651) — this is the closest thing to a test of the human/ai role mapping, but it patches admin_panel._load_chat_history with a hand-built fake list, so it does not touch _doc_to_message and is unaffected by the AIMessage change. admin_panel._doc_to_message itself (lines 326-341) has NO unit test — consider adding one asserting HumanMessage→'user' and AIMessage→'fritz', since this change is what finally makes that mapping correct in production.
- tests/test_admin_panel.py — every chat test (TestChatSend, TestChatStreamProgressEvents, TestPendingImagePlumbing, etc.) replaces the whole mister_fritz module via `patch.dict(sys.modules, {"mister_fritz": fake_module})`, so none of them execute any changed code.
- tests/test_agent_tools.py:43 imports only format_prompt and get_source_info from mister_fritz — untouched.
- tests/test_bot_commands.py imports mister_fritz transitively (module-level init only) — untouched.

### Manual verification

- Import smoke test first — the count_tokens_approximately import path is the likeliest breakage: `python -c "import mister_fritz"` must not raise (it will `ImportError` if imported from langchain_core.messages instead of .utils).
- Live two-turn continuity test with real Ollama, `LOG_LEVEL=DEBUG`. Turn 1: "My favourite colour is green." Turn 2: "What did I just tell you?" Fritz must answer from the window, not from Chroma. Confirm by watching for the new `History window: N/M messages, ~T tokens` DEBUG line and, critically, by first running `/forget` (privacy.forget_conversation) plus clearing the user's Chroma namespace so the answer cannot come from memory injection.
- Pronoun/reference resolution: "Roll 3d6." then "Do it again." — the second turn must call roll_dice without asking what to roll.
- Plan mode: give a request that trips the planner (e.g. "Research X on the web, then write me a short report"), confirm from DEBUG logs that the window in each executor iteration excludes the current turn and is identical across steps 1..N (it must be — plan-mode executor returns no messages, so state['messages'] is frozen for the duration of the plan).
- Oversized-input path: paste ~60k characters as a single message. Confirm the log shows `history_window_overflow` (or just no window line) and Fritz still answers rather than replying to an empty prompt.
- Context-overflow check: run a turn that calls scrape_web on a long article. agent_tools.scrape_web (lines 240-248) returns the FULL page text with no truncation — verified — so this is the one path that can still blow past num_ctx=32768 and cause Ollama to silently evict the front of the prompt (i.e. the system prompt and persona). Watch for Fritz breaking character. If it happens, that is the scrape_web cap, not this change; log it as a separate item.
- Latency A/B: time 5 identical turns with HISTORY_TOKEN_BUDGET=0 vs 4096 and record the delta. Expect sub-second on a warm GPU (the window is typically ~500-1500 tokens because SUMMARIZE_THRESHOLD caps it at 16 messages), but this is the number to hand to the latency-tax item.
- /chat history check: after a few new turns, load http://127.0.0.1:8001/chat and confirm Fritz's replies now render in assistant bubbles rather than user bubbles.

## Risks

- Pre-existing checkpoints are poisoned. Every reply written before commit 1 is stored as a HumanMessage, so for existing threads the window will replay Fritz's old answers as user turns until the summariser's RemoveMessage sweep (mister_fritz.py:271) clears them — at most SUMMARIZE_THRESHOLD+1 = 16 messages, so ~8 turns. Detect: `python -c` over `app.get_state(config).values['messages']` for a known user and count `.type == 'ai'`. Mitigation: none needed (self-healing); or tell affected users to hit "New conversation" (/chat/forget → privacy.forget_conversation). Do NOT write a migration — you cannot reliably tell which historical HumanMessages were Fritz's.
- Context overflow. The memory blob is now capped but agent_tools.scrape_web (:240-248) still returns whole pages untruncated, and search_web returns 5 full DDG results. Adding up to 4096 history tokens brings the worst case closer to num_ctx=32768, at which point Ollama silently drops the front of the prompt — including the system message and the entire Fritz persona. Detect: Fritz breaking character or ignoring tool instructions on turns that follow a big scrape. Mitigation: lower HISTORY_TOKEN_BUDGET; the real fix (capping scrape_web output the way EXEC_OUTPUT_TRUNCATE caps execute_command) is deliberately out of scope here.
- Latency. Prefill cost scales with prompt length. Worse: the volatile memory/profile blocks are appended to the SYSTEM message at position 0, so every turn's prompt differs at the very first token and Ollama's KV prefix cache is useless for the whole prompt — the history window is re-prefilled from scratch every turn. Detect: the A/B timing in manualVerification. This is the strongest argument for the deferred refactor noted in openQuestions, and should be fed into the latency-tax item.
- Merge conflict with token-streaming. That item almost certainly rewrites the executor's `for s in agent.stream(...)` loop at mister_fritz.py:416-433, which sits ~5 lines below this change. Not a dependency (this change touches only the construction of `inputs`), but land one before starting the other or expect a hand-merge.
- approximate token counting is a chars/4 heuristic tuned for English on OpenAI tokenizers. gpt-oss's tokenizer will differ, and code blocks / JSON / non-English text tokenize far worse than 4 chars/token. The counter can therefore underestimate by 30-50% on code-heavy turns. Mitigation: the 4096 default has ~24k of headroom, so a 50% underestimate is still safe. If it ever matters, the real fix is a HuggingFace tokenizer for the model (transformers 4.57.3 is already in requirements.txt) passed as `token_counter=` — a callable is accepted.
- Two consecutive human-role messages in plan mode (window ends on the previous user turn, then the synthetic step prompt). In practice the parent transcript alternates Human/AI so `messages[:-1]` ends on an AI turn (verified with real trim_messages: [H,A,H,A] → ['human','ai','human','ai']). If it ever does happen, Ollama's chat template renders both fine. Hardening if wanted: pass `end_on="ai"` in the plan-mode trim call.
- Stale image paths in replayed history. Historic user turns built by format_prompt include "User has attached images: [...]" with paths that may no longer exist. Low risk, verified: analyze_image (agent_tools.py:359-375) reads user_image_paths from config metadata, never from the prompt text, so a stale path in the transcript cannot cause a bad file read — worst case the model mentions an old image.

## Rollback
"Two independent levers. (1) Runtime, no deploy: set `HISTORY_TOKEN_BUDGET=0` in `.env` and restart. `_history_window` short-circuits to `[]`, both executor branches take their fallback paths, and the agent input becomes byte-identical to today's `[(\"system\", system_prompt), (\"user\", latest_message)]` in simple mode and `[(\"system\", system_prompt), (\"user\", agent_prompt)]` in plan mode. This is the reason the kill switch returns `[]` rather than `[messages[-1]]` — the latter would NOT reproduce plan-mode behaviour. `tests/test_mister_fritz.py::TestExecutorInputs::test_zero_budget_reproduces_single_message_input` is the contract that keeps this true. (2) Code: `git revert` commits 3-5. Note that the `AIMessage` role fix (commit 1) is deliberately separate and is NOT covered by the kill switch — it is correct and desirable on its own (it fixes the /chat history renderer regardless of the window), so leave it in place unless something specifically implicates it, in which case revert commit 1 alone. No feature-flag scaffolding beyond the env var is warranted: the change is ~40 lines confined to one function in one module."

## Open questions for you to decide

- Should SUMMARIZE_THRESHOLD move? My recommendation is NO, keep 15, and I want to be explicit about why rather than hand-wave it. The summariser removes all but the last message (mister_fritz.py:271), so the window is structurally capped at 16 messages ≈ 8 exchanges ≈ well under 4096 tokens for typical Discord chat — meaning the token budget almost never binds and is really a safety valve for pasted walls of text. If the owner wants longer in-thread continuity, the lever is SUMMARIZE_THRESHOLD, not HISTORY_TOKEN_BUDGET. The cost of raising it is real though: summarize_conversation feeds the entire message list to the thinking model at :231, so a threshold of 40 makes every 40th turn noticeably slow. Suggested experiment: run at 15 for a week, watch the `history_window_trimmed` counter in /health; if it stays at zero, the budget is not the constraint and the owner should consider 25-30.
- Where should the volatile memory/profile blocks live? Appending them to the system message (:332, :359) puts turn-varying text at position 0, which defeats Ollama's KV prefix cache for the entire prompt — including the now-larger history. Moving them to a trailing SystemMessage placed AFTER the history (or into the last human turn) would make the system prompt + older history a stable prefix that Ollama can reuse across turns. That is a genuinely valuable latency win but it changes prompt structure enough to alter model behaviour, so I am explicitly deferring it — it belongs in latency-tax, measured, not bundled here.
- Should the format_prompt wrapper (mister_fritz.py:124-131) be stripped from replayed history? Each stored user turn carries ~96 chars / ~24 tokens of "Context: User is texting from Discord (User ID: X) Question:" boilerplate, repeated on every historical turn in the window. It is cheap (~350 tokens across a full 15-message window) but it is also noise the model has to read past, and for DISCORD_VOICE it repeats "Please answer in 30 words or less" on turns where that no longer applies — which could bleed terseness into a text reply. Cleanest fix is to store the raw base_prompt in state and apply format_prompt only to the current turn, but that changes what ask_stuff writes into the checkpoint. Deferred; flag if voice/text mixing produces oddly short answers.
- Should `search_memories_internal`'s `limit=30` (agent_tools.py:77) come down now that the blob is char-capped? Fetching 30 results and discarding most of them wastes embedding-search time on every single turn. Dropping to 10 would be strictly cheaper with near-identical output post-cap. It is a one-token change in another module, so I left it out of this item's blast radius — but it is nearly free and the owner may want it folded in.
- Is the ~4 chars/token approximation good enough for gpt-oss, or should a real tokenizer be wired in? Cannot be settled statically. The experiment that settles it: for ~20 real turns, log `count_tokens_approximately(window)` alongside the `prompt_eval_count` Ollama returns in its response metadata (langchain_ollama surfaces it in `response_metadata`), and compute the ratio. If it is consistently within ±20% the heuristic is fine forever; if it is worse than 40% low on code-heavy turns, pass a `transformers` AutoTokenizer-backed callable as `token_counter=`. Do not do this speculatively — the 24k of headroom in the budget makes it very likely unnecessary.
