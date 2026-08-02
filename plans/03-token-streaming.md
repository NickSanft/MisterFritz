# 3. Real token streaming end-to-end

[← back to index](README.md)

**Effort:** M (half day)  
**Depends on:** nothing

## Goal
Today the executor streams whole per-superstep state snapshots, so the user sees nothing until an entire AIMessage has been generated, and the web client re-sends the full accumulated reply on every event (O(n²) bytes). After this change the ReAct executor streams real LLM token chunks via LangGraph's `stream_mode=["values","messages"]`, and a single well-defined callback contract — `streaming_callback(delta, accumulated, restart)` — feeds both surfaces: Discord keeps using `accumulated` for its whole-body message edits (throttled, unchanged handler), and the web SSE endpoint emits delta-only `token` frames plus a new `reset` frame that the browser appends/clears against. Tool-call progress notices keep firing from the `values` stream before each tool runs, and plan mode stays deliberately silent until the synthesizer, which now also emits deltas.

## Definition of done

- [ ] `mister_fritz.executor` calls `agent.stream(..., stream_mode=["values", "messages"])` when a streaming callback is present and `["values"]` (a one-element list, so the (mode, payload) tuple shape is uniform) otherwise.
- [ ] With `STREAM_MIN_CHARS=1`, a scripted 5-chunk model produces >= 5 `streaming_callback` invocations — not one per completed AIMessage.
- [ ] Concatenating every `delta` since the most recent `restart=True` equals the `accumulated` value of the final call in that segment.
- [ ] `POST /chat/stream` `token` frames carry only new text; total `token` payload bytes for an N-char reply are O(N), not O(N^2).
- [ ] After a tool-calling turn where the model emitted preamble text, the web bubble's final rendered text equals the `done` frame's text — no duplicated preamble (a `reset` frame cleared it).
- [ ] Discord's final message content still equals `response_data["text"]` (via `final_update`), unchanged.
- [ ] Plan mode emits zero `token`/`reset` frames during executor steps; the synthesizer's first emission carries `restart=True`.
- [ ] `ruff check .` is clean (note: `AIMessage` becomes an unused import in `mister_fritz.py` if not swapped — F401 is enabled and `tests/*` is the only per-file-ignore).
- [ ] `pytest tests/` green with `--cov-fail-under=60`.
- [ ] `.env.example` documents `STREAM_MIN_CHARS` and `DISCORD_STREAM_MIN_INTERVAL`; `CHANGELOG.md` has a phase-style entry under `## [Unreleased]`.

## Current state (verified against the working tree)
VERIFIED AGAINST THE CODE. The audit's line numbers are all still accurate; two of its implicit assumptions needed correcting, and I found one behavioural trap it missed.

CONFIRMED:
- `mister_fritz.py:416` — `for s in agent.stream(inputs, config=get_config_values(config), stream_mode="values"):` inside `executor()` (`def executor` at :275).
- `mister_fritz.py:420-427` — tool-call detection on the values snapshot (`hasattr(latest, 'tool_calls') and latest.tool_calls`) driving `progress_callback` with the `tool_messages` dict defined at :364-382.
- `mister_fritz.py:428-433` — the streaming branch: `elif hasattr(latest,'content') and isinstance(latest.content,str) and effective_streaming_callback:` → `if isinstance(latest, AIMessage): ... effective_streaming_callback(accumulated_text)`. It fires only once the whole AIMessage lands, and passes FULL text.
- `mister_fritz.py:405` — `effective_streaming_callback = None` in plan mode; `:409` assigns the real callback in simple mode.
- `mister_fritz.py:488-499` — synthesizer is the only true token-streamer (`ollama_instance.stream` at :490), but calls `streaming_callback(accumulated_text)` (:494) with full text; the `except` fallback at :495-499 never calls the callback at all.
- `mister_fritz.py:572` — outer graph `app.stream(..., stream_mode="values")`. This is correct to leave alone: the callbacks fire from inside nodes, so tokens never traverse this loop; `final_state` extraction at :578-583 depends on it.
- `main_discord.py:35-82` — `StreamingMessageHandler`; `.edit(content=self.pending_text[:2000])` at :63 and `[:2000]` again at :78; `min_update_interval: float = 1.5` hardcoded at :38. `:183-189` builds the handler and defines `streaming_callback(partial_text)` / `progress_callback(message)`, both hopping threads with `asyncio.run_coroutine_threadsafe`.
- `admin_panel.py:509-511` — `_streaming_callback(partial_text)` puts `("token", partial_text)` (full accumulated text) on `event_queue`; `:513-517` the progress callback; `:576-577` the hand-rolled SSE framing.
- `admin_templates/chat.html:354` — `body.textContent = data;` replaces wholesale.
- Pins: `requirements.txt:110 langchain==1.1.3`, `:119 langgraph==1.0.5`, `:116 langchain-ollama==1.0.1`, `:114 langchain-core==1.2.5`.
- Only two in-repo callers pass a streaming callback (`main_discord.py:197`, `admin_panel.py:527`). `bot_commands.py:395`, `main_telegram.py:28/54`, `scheduler.py:111` all call `ask_stuff` with no callbacks and are unaffected.

VERIFIED EMPIRICALLY IN THIS SESSION (installed langgraph 1.0.5 / langchain 1.1.3 / langchain-core 1.2.19 / langchain-ollama 1.0.1):
- `stream_mode=["values","messages"]` yields `(mode, data)`; messages data is `(message, metadata)`. A plain string `stream_mode` yields raw payloads, a LIST of even one mode yields tuples — so always pass a list.
- messages-mode emits `AIMessageChunk` per token from `langgraph_node="model"` AND whole `ToolMessage` objects from `langgraph_node="tools"`. Consumers MUST filter on `isinstance(chunk, AIMessageChunk)`.
- `AIMessageChunk.id` is stable within one model turn and changes across turns (`lc_run--<uuid>`) — a free segment key.
- It works when the sub-agent is streamed from INSIDE an outer StateGraph node (exactly `executor`'s shape, with `get_config_values` rebuilding the config): I ran a nested reproduction and got 7 token chunks. This is the riskiest assumption and it holds.

CORRECTIONS TO THE AUDIT'S FRAMING:
1. `create_agent`'s model node calls `model_.invoke(messages)` (site-packages/langchain/agents/factory.py:1102), NOT `.stream()`. Token streaming still works because `BaseChatModel._generate_with_cache` switches to the `_stream` path whenever a `_StreamingCallbackHandler` is in the run manager's handlers, and LangGraph's `StreamMessagesHandler` (site-packages/langgraph/pregel/_messages.py) subclasses exactly that. No change to `create_agent` or `ChatOllama` is required.
2. Note a latent trap in `langchain_ollama`: `ChatOllama._stream` calls `run_manager.on_llm_new_token(chunk.text, verbose=...)` WITHOUT `chunk=chunk`, and `StreamMessagesHandler.on_llm_new_token` early-returns when `chunk` isn't a `ChatGenerationChunk`. We are saved because `_generate_with_cache` (not `_stream` directly) is what drives the callback and it does pass `chunk=`. Do not "optimise" the executor into calling `ollama_instance.stream()` directly expecting the same events.

TRAP THE AUDIT MISSED: today's `elif` at :428 is skipped for any message that has `tool_calls`, which silently SUPPRESSES pre-tool-call preamble text. With token deltas that preamble streams live, then the real answer follows in a NEW model turn. Without a reset signal the two would concatenate in the UI. I reproduced this with a scripted model ("Let me look. " streamed, then a tool call, then "I found it sir."). This is why the contract needs `restart`, not just `delta`.

## Change sites

### `fritz_utils.py:96-97 (insert after the SUMMARIZE_THRESHOLD block, inside the '# Tunables' section)`

Two new env-var knobs, following the existing `int(os.environ.get(...))` convention.

# Number of conversation messages before the agent triggers a summarisation pass.
SUMMARIZE_THRESHOLD: int = int(os.environ.get("SUMMARIZE_THRESHOLD", "15"))

# Token streaming: characters to accumulate before firing streaming_callback.
# 1 = emit every token (smoothest). Raise it to cut SSE frame count / Discord
# edit pressure on slow links; set it absurdly high to collapse a whole reply
# into a single emission (the closest thing to a streaming kill-switch).
STREAM_MIN_CHARS: int = max(1, int(os.environ.get("STREAM_MIN_CHARS", "1")))

# Minimum seconds between Discord message edits while streaming. Discord's
# edit rate limit is the binding constraint; 1.5 s was the hardcoded default
# in StreamingMessageHandler before this knob existed.
DISCORD_STREAM_MIN_INTERVAL: float = float(os.environ.get("DISCORD_STREAM_MIN_INTERVAL", "1.5"))

### `mister_fritz.py:8`

Swap `AIMessage` for `AIMessageChunk` in the langchain_core.messages import. `AIMessage`'s only use is line 429, which this change deletes — leaving it triggers ruff F401 and fails CI.

-from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
+from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage

### `mister_fritz.py:29-37`

Import the new knob from fritz_utils (alphabetically STREAM_MIN_CHARS sorts before SUMMARIZE_THRESHOLD, i.e. insert before line 35).

from fritz_utils import (
    CHAT_DB_NAME,
    FAST_OLLAMA_MODEL,
    MessageSource,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
+   STREAM_MIN_CHARS,
    SUMMARIZE_THRESHOLD,
    THINKING_OLLAMA_MODEL,
)

### `mister_fritz.py:274 (new block immediately before `def executor` at 275)`

Add `_chunk_text()` and the `_DeltaEmitter` class — the single place that defines and enforces the new callback contract, shared by `executor` and `synthesizer`.

# ── Streaming callback plumbing ───────────────────────────────────────────────

def _chunk_text(message) -> str:
    """Plain text of a streamed message chunk. ChatOllama emits `content` as a
    str today, but langchain-core's v1 output format can make it a list of
    content blocks; `.text` normalises both."""
    text = getattr(message, "text", None)
    if callable(text):  # some langchain-core versions expose text() as a method
        try:
            text = text()
        except Exception:
            text = None
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else ""


_SEGMENT_UNSET = object()


class _DeltaEmitter:
    """Adapts LLM token chunks to the streaming-callback contract:

        streaming_callback(delta: str, accumulated: str, restart: bool) -> None

    delta       — text produced since the previous call.
    accumulated — full text of the CURRENT answer segment.
    restart     — True on the first emission of a new segment. A new segment
                  starts whenever the model begins a fresh turn (e.g. it wrote
                  a preamble, called a tool, then began the real answer) or the
                  synthesizer takes over from the executor.

    Consumers that replace content wholesale (Discord message edits) read
    `accumulated` and can ignore `restart`. Consumers that append (web SSE)
    read `delta` and clear their buffer when `restart` is True.
    """

    def __init__(self, callback, min_chars: int = STREAM_MIN_CHARS):
        self._callback = callback
        self._min_chars = max(1, min_chars)
        self._segment = _SEGMENT_UNSET
        self._accumulated = ""
        self._buffer = ""
        self._restart_pending = False

    def feed(self, text: str, segment_id=None) -> None:
        if not text or self._callback is None:
            return
        if segment_id != self._segment:
            self._segment = segment_id
            self._accumulated = ""
            self._buffer = ""
            self._restart_pending = True
        self._accumulated += text
        self._buffer += text
        if len(self._buffer) >= self._min_chars:
            self.flush()

    def flush(self) -> None:
        """Emit whatever is buffered. Call once when the stream ends so a
        sub-threshold tail isn't stranded."""
        if self._callback is None or not self._buffer:
            return
        delta, self._buffer = self._buffer, ""
        restart, self._restart_pending = self._restart_pending, False
        try:
            self._callback(delta, self._accumulated, restart)
        except Exception as e:  # a broken UI consumer must not kill the turn
            logger.warning("streaming_callback raised (non-fatal): %s", e)

### `mister_fritz.py:411-435`

Replace the values-only stream loop. Values events keep driving `final_state` and tool-call progress notices (unchanged semantics — the values snapshot containing the tool_calls AIMessage still arrives before the tools node runs; I verified the ordering). Messages events become token deltas.

    inputs = {"messages": [("system", system_prompt), ("user", agent_prompt)]}

    final_state = None
    emitter = _DeltaEmitter(effective_streaming_callback)
    # A LIST (even of one mode) is what makes LangGraph tuple its output as
    # (mode, payload); a bare string yields raw payloads. Keep it uniform.
    stream_modes = ["values", "messages"] if effective_streaming_callback else ["values"]

    for mode, payload in agent.stream(
        inputs, config=get_config_values(config), stream_mode=stream_modes
    ):
        if mode == "values":
            final_state = payload
            if "messages" in payload and payload["messages"]:
                latest = payload["messages"][-1]
                if getattr(latest, "tool_calls", None):
                    logger.debug("Detected tool calls: %s", [tc.get('name', '') for tc in latest.tool_calls])
                    if progress_callback:
                        for tool_call in latest.tool_calls:
                            tool_name = tool_call.get('name', '')
                            if tool_name in tool_messages and tool_name not in notified_tools:
                                progress_callback(tool_messages[tool_name])
                                notified_tools.add(tool_name)
            continue

        # mode == "messages" → (chunk, metadata). StreamMessagesHandler also
        # emits whole ToolMessages from the tools node; only AIMessageChunks
        # are LLM tokens. chunk.id is stable per model turn → segment key.
        chunk, _meta = payload
        if isinstance(chunk, AIMessageChunk):
            emitter.feed(_chunk_text(chunk), segment_id=chunk.id)

    emitter.flush()

    resp = final_state["messages"][-1].content if final_state and "messages" in final_state else ""

# NOTE: lines 437-453 (image_paths collection from ToolMessages, and the
# plan-mode vs simple-mode return) are unchanged.

### `mister_fritz.py:488-499`

Synthesizer emits deltas through the same emitter under a constant segment id (so its first emission carries restart=True and wipes whatever the executor left on screen), and the invoke() fallback now corrects the UI instead of leaving partial text.

    emitter = _DeltaEmitter(streaming_callback)
    accumulated_text = ""
    try:
        for chunk in ollama_instance.stream(synthesis_prompt, config=get_config_values(config)):
            text = _chunk_text(chunk)
            if text:
                accumulated_text += text
                emitter.feed(text, segment_id="synthesis")
        emitter.flush()
    except Exception as e:
        logger.warning("Synthesizer stream failed: %s; falling back to invoke", e)
        accumulated_text = ollama_instance.invoke(
            synthesis_prompt, config=get_config_values(config)
        ).content
        # Replace whatever partial text the UI is showing with the real answer.
        if streaming_callback:
            streaming_callback(accumulated_text, accumulated_text, True)

### `mister_fritz.py:402-409`

NO CODE CHANGE — plan mode keeps `effective_streaming_callback = None` (line 405). Add a one-line comment recording that this is deliberate: intermediate steps stay silent, the synthesizer owns the visible stream. Called out so the next reader doesn't 'fix' it.

        if progress_callback:
            progress_callback(f"Step {current_step + 1}/{len(plan)}: {step_instruction}")
        # Intermediate steps stay silent — the synthesizer streams the one
        # answer the user actually sees, with restart=True on its first token.
        effective_streaming_callback = None

### `mister_fritz.py:523-534`

Document the contract on `ask_stuff`'s docstring so out-of-tree callers see the arity change without reading the executor.

    """Process user input and return structured output with text and attachments.

    streaming_callback(delta, accumulated, restart) is called as tokens arrive:
      delta       — new text since the last call
      accumulated — full text of the current answer segment
      restart     — True when a new segment begins (clear anything shown)
    progress_callback(message) is called with human-readable tool notices.
    """

### `main_discord.py:13-22, 38`

Import `DISCORD_STREAM_MIN_INTERVAL` and use it as the handler's default instead of the hardcoded 1.5. `StreamingMessageHandler`'s body (48-82) is untouched — it already coalesces via `pending_text`, and it still receives full text.

from fritz_utils import (
    DISCORD_BOT_TOKEN,
+   DISCORD_STREAM_MIN_INTERVAL,
    EMBEDDING_MODEL,
    ...
)

-    def __init__(self, message: discord.Message, loop: asyncio.AbstractEventLoop, min_update_interval: float = 1.5):
+    def __init__(self, message: discord.Message, loop: asyncio.AbstractEventLoop,
+                 min_update_interval: float = DISCORD_STREAM_MIN_INTERVAL):

### `main_discord.py:183-189`

New callback arity; use `accumulated` (Discord edits replace the whole body, and `accumulated` already resets on `restart`); add a cheap cross-thread scheduling guard so ~40 tokens/s don't queue 40 coroutines/s onto the gateway loop.

    streaming_handler = StreamingMessageHandler(status_msg, loop)
    last_stream_schedule = 0.0

    def streaming_callback(delta: str, accumulated: str, restart: bool = False):
        # A Discord edit replaces the entire message body, so we send
        # `accumulated` (which resets by itself when `restart` fires), not the
        # delta. Called once per token from a worker thread — rate-limit the
        # cross-thread hop; final_update() always writes the complete text.
        nonlocal last_stream_schedule
        now = time.monotonic()
        if not restart and now - last_stream_schedule < DISCORD_STREAM_MIN_INTERVAL:
            return
        last_stream_schedule = now
        asyncio.run_coroutine_threadsafe(streaming_handler.update_text(accumulated), loop)

    def progress_callback(message: str):
        asyncio.run_coroutine_threadsafe(ctx.channel.send(message), loop)

# `time` is already imported at main_discord.py:4. Lines 191-240 unchanged.

### `admin_panel.py:487-494`

Update the `chat_stream` docstring's event list — `token` is now a delta and `reset` is new.

    """Server-Sent Events endpoint. Runs ask_stuff in a worker thread; the
    streaming_callback puts events on a queue.Queue that the SSE generator
    drains. Emits:
      - event=reset    data=              (clear the bubble; new answer segment)
      - event=token    data=<delta text>  (append — NOT the accumulated text)
      - event=progress data=<tool notice>
      - event=done     data=<JSON {text, html, images}>   (exactly one)
      - event=error    data=<message>     (instead of done on failure)
    """

### `admin_panel.py:509-517`

Delta-only `token` frames plus a `reset` frame on segment boundaries. This is the O(n^2) → O(n) fix. The SSE framing at 576-577 and the generator at 560-579 need no change: every data line carries a literal `data: ` prefix, so no payload can accidentally produce a frame-terminating blank line.

    def _streaming_callback(delta: str, accumulated: str, restart: bool = False) -> None:
        # ask_stuff invokes this from a worker thread; queue.Queue is thread-safe.
        # `token` carries ONLY the new text (the client appends), so a long
        # reply costs O(n) bytes on the wire instead of O(n^2). `reset` means
        # the model started a fresh answer — after a tool call, or when the
        # synthesizer took over — so the client must clear the bubble first.
        if restart:
            event_queue.put(("reset", ""))
        event_queue.put(("token", delta))

### `admin_templates/chat.html:352-360`

Append deltas instead of replacing; handle the new `reset` event. Nothing else in the parser changes — `line.slice(5).replace(/^ /, "")` already strips exactly one leading space, so a token like " well" round-trips intact.

                        if (eventName === "token") {
                            // Deltas: append. Markdown is applied only at 'done'.
                            body.textContent += data;
                        } else if (eventName === "reset") {
                            // Fresh answer segment (post-tool-call, or the
                            // synthesizer) — drop anything streamed so far.
                            body.textContent = "";
                        } else if (eventName === "progress") {
                            const line = makeProgressLine(data);
                            list.insertBefore(line, fritzBubble);
                            progressLines.push(line);
                        } else if (eventName === "done") {

### `.env.example:57 (insert after MEMORY_EXTRACT_MIN_REPLY_CHARS, within the '----- Tunables -----' section)`

Document both new knobs.

# Characters to buffer before pushing a streamed token to the UI. 1 = emit
# every token (smoothest). Raise to reduce SSE frame count / Discord edits.
# STREAM_MIN_CHARS=1
# Minimum seconds between Discord message edits while a reply streams in.
# DISCORD_STREAM_MIN_INTERVAL=1.5

### `CHANGELOG.md:8-10 (top of '## [Unreleased]', under '### Performance')`

Phase-style entry matching the surrounding prose (see the existing 'Web chat — Phase 2: SSE streaming' bullet, which explicitly documents `token` as 'each accumulated state of the response' — that sentence is now wrong and must be superseded by this entry).

- **Phase 15 — real token streaming.** The ReAct executor now streams LLM tokens instead of whole state snapshots. `agent.stream(...)` runs with `stream_mode=["values", "messages"]`: `values` still drives final-state extraction and the tool-progress notices, `messages` yields `AIMessageChunk` tokens as the model produces them.
  - **New callback contract.** `streaming_callback(delta, accumulated, restart)`. `delta` is the new text, `accumulated` is the full text of the current answer segment, and `restart` is True when the model begins a fresh segment (it wrote a preamble, called a tool, and started over; or the synthesizer took over from a plan run). Consumers that replace content wholesale use `accumulated`; consumers that append use `delta` + `restart`.
  - **Web chat: O(n) instead of O(n^2).** `token` SSE frames now carry only the delta and the client appends, with a new `reset` frame clearing the bubble on a segment boundary. Previously every event re-sent the entire reply so far.
  - **Discord** keeps editing with the full accumulated body (the API requires it) but the worker→loop hop is now rate-limited by `DISCORD_STREAM_MIN_INTERVAL` so a fast model can't queue dozens of coroutines per second onto the gateway loop.
  - Plan mode still suppresses streaming for intermediate steps by design; the synthesizer streams the single answer the user sees.
  - New knobs: `STREAM_MIN_CHARS` (default 1) and `DISCORD_STREAM_MIN_INTERVAL` (default 1.5).

### `README.md:328 (Chat UI feature table row)`

One-line accuracy fix: the row already promises token-by-token SSE — it is now actually true. Optionally note the `reset` event alongside the existing 'blinking cursor' description. Low priority; include in the same commit as the CHANGELOG.

| **Streaming responses** | Fritz's reply appears token-by-token via Server-Sent Events (delta frames), with a blinking cursor while he writes. |

## Steps

1. Commit 1 — config: add `STREAM_MIN_CHARS` and `DISCORD_STREAM_MIN_INTERVAL` to `fritz_utils.py` and `.env.example`. No behaviour change yet.
2. Commit 2 — contract: add `_chunk_text()` and `_DeltaEmitter` to `mister_fritz.py` above `executor` (with the docstring that IS the contract spec), plus the `AIMessageChunk` import swap and the `STREAM_MIN_CHARS` fritz_utils import. Add `TestDeltaEmitter` unit tests in `tests/test_mister_fritz.py`. Still no call sites — suite stays green.
3. Commit 3 — third-party canary test: add `TestLangGraphMessagesContract` to `tests/test_mister_fritz.py` that builds a scripted `BaseChatModel` (implements `_stream` + `bind_tools`; no network, no Ollama), wraps it in `langchain.agents.create_agent`, and asserts `stream_mode=["values","messages"]` yields `(mode, payload)` tuples with `AIMessageChunk` tokens and distinct `chunk.id` per model turn. This pins the exact upstream behaviour the executor depends on, so a `langgraph`/`langchain-core` bump fails CI loudly instead of silently reverting to no streaming.
4. Commit 4 — executor: rewrite the stream loop at `mister_fritz.py:411-435` per the sketch. Add `TestExecutorTokenStreaming` (stub agent). Verify `pytest tests/test_mister_fritz.py` green and `ruff check .` clean (the F401 on `AIMessage` bites here if step 2's import swap was skipped).
5. Commit 5 — synthesizer: rewrite `mister_fritz.py:488-499` to feed the emitter under `segment_id="synthesis"` and to correct the UI from the invoke() fallback. Add the plan-mode comment at :405.
6. Commit 6 — Discord consumer: `main_discord.py` import + handler default + new `streaming_callback` arity and scheduling guard. Confirm `tests/test_discord_commands.py::TestStreamingMessageHandler` still passes untouched (it constructs the handler with an explicit `min_update_interval`, so the default change is invisible to it).
7. Commit 7 — web consumer: `admin_panel.py` `_streaming_callback` + docstring, and `admin_templates/chat.html` token/reset handling. Update the four existing fakes in `tests/test_admin_panel.py` to the new arity and add the reset-ordering / leading-space tests.
8. Commit 8 — docs: CHANGELOG entry and the README row. Run the full suite with coverage and `ruff check .`.
9. Manual smoke: run through the five scenarios in testPlan.manualVerification against a live Ollama before merging — the hermetic tests cannot tell you whether your actual model emits pre-tool-call preamble or leaks reasoning into `content`.

## Config and env changes

- STREAM_MIN_CHARS (fritz_utils.py, default 1) — characters buffered before firing streaming_callback. 1 emits every token. Setting it very high collapses a reply into a single end-of-stream emission, which is the practical mitigation if streaming misbehaves in production without a redeploy.
- DISCORD_STREAM_MIN_INTERVAL (fritz_utils.py, default 1.5) — minimum seconds between Discord message edits AND between cross-thread callback hops. Replaces the hardcoded 1.5 default at main_discord.py:38.
- Both documented in .env.example under the '----- Tunables -----' section. No changes to docker-compose.yml, Dockerfile, or infra/k8s/configmap needed — both have sane defaults and neither is required.

## Tests
### New

- tests/test_mister_fritz.py :: TestDeltaEmitter — test_single_segment_emits_delta_and_accumulated (first call has restart=True, subsequent False; accumulated grows); test_new_segment_id_resets_accumulator_and_flags_restart; test_min_chars_buffers_until_threshold (min_chars=5, feed 'a'*3 → no call, feed 'bb' → one call with delta 'aaabb'); test_flush_emits_sub_threshold_tail; test_flush_is_noop_when_buffer_empty; test_none_callback_is_a_noop (no exception); test_callback_exception_is_swallowed_and_logged.
- tests/test_mister_fritz.py :: TestChunkText — test_str_content_returned; test_text_property_preferred; test_list_content_returns_empty_string (guards the langchain-core v1 content-block form).
- tests/test_mister_fritz.py :: TestExecutorTokenStreaming — a `_StubAgent` whose `.stream(inputs, config=..., stream_mode=...)` records `stream_mode` and yields a scripted list of ('values', {...}) / ('messages', (chunk, {})) tuples, injected via `patch.object(mister_fritz, '_get_conversation_agent', return_value=stub)`. Pass `config={'metadata': {'streaming_callback': cb, 'progress_callback': pcb}}` — with no `user_id` in metadata the memory-injection (mister_fritz.py:327-335) and profile-injection (337-362) blocks are skipped, so no Chroma/Ollama is touched. Cases: test_ai_message_chunks_become_deltas (three AIMessageChunks with the same id → three callbacks, deltas concatenate to accumulated, only the first has restart=True); test_tool_messages_in_messages_stream_are_ignored; test_new_chunk_id_sets_restart_and_resets_accumulated (reproduces the preamble-then-tool-call-then-answer sequence); test_tool_calls_in_values_still_fire_progress_callback (asserts 'Searching the web...' from the tool_messages dict, and that a repeated tool name only notifies once via notified_tools); test_requests_values_and_messages_when_callback_present (stub.stream_mode == ['values','messages']); test_plan_mode_requests_values_only_and_never_streams (state with plan=['a','b'] → stub.stream_mode == ['values'], cb never called, progress_callback got the 'Step 1/2: a' line).
- tests/test_mister_fritz.py :: TestSynthesizerStreaming — patch `mister_fritz.ollama_instance`; `.stream` yields AIMessageChunks → assert first callback has restart=True and deltas concatenate; second case where `.stream` raises → assert `.invoke` fallback fires exactly one callback with (full, full, True).
- tests/test_mister_fritz.py :: TestLangGraphMessagesContract — the third-party canary described in step 3. Hermetic (scripted BaseChatModel, no network). Asserts tuple shape, AIMessageChunk filtering, and per-turn id stability.
- tests/test_admin_panel.py :: TestChatStreamReset — fake ask_stuff emits two segments (restart=True 'A', then restart=True 'B'); assert the parsed SSE event order is [('reset',''),('token','A'),('reset',''),('token','B'), ('done', ...)].
- tests/test_admin_panel.py :: TestChatStreamTokenFidelity — test_token_delta_preserves_leading_space: fake emits streaming_callback(' well', 'Very well', False); assert the parsed token equals ' well' exactly (guards the `data: ` + single-leading-space-strip round trip in admin_panel.py:576 / chat.html:349); test_token_delta_with_embedded_newlines_round_trips ('a\n\nb').

### Existing tests affected

- tests/test_admin_panel.py:493-521 TestChatStreamSuccess.test_streams_token_events_then_done — the fake at 498-504 calls streaming_callback with ONE arg three times and asserts tokens == ['Very','Very well','Very well, sir.']. Rewrite the fake to streaming_callback('Very','Very',True) / (' well','Very well',False) / (', sir.','Very well, sir.',False) and change the assertion to tokens == ['Very',' well',', sir.'] plus exactly one 'reset' event.
- tests/test_admin_panel.py:523-543 TestChatStreamSuccess.test_audit_log_records_streamed_message — fake at 527-530 calls streaming_callback('ok'); change to streaming_callback('ok','ok',True). Audit assertions unchanged.
- tests/test_admin_panel.py:595-618 TestChatStreamDonePayload.test_done_event_carries_html_and_text — fake at 601-604 calls streaming_callback('Very well'); change to streaming_callback('Very well','Very well',True). The done-payload assertions are unaffected.
- tests/test_admin_panel.py:621-642 TestChatStreamProgressEvents.test_progress_callback_yields_progress_events — line 631 calls streaming_callback('Here is what I found.'); change to the 3-arg form. The progress assertion is unaffected.
- tests/test_discord_commands.py:78-128 TestStreamingMessageHandler (all six tests) — NO CHANGES REQUIRED, and that is the point: `StreamingMessageHandler`'s public surface (update_text/final_update/current_text/pending_text/last_update_time) is untouched, and every test passes `min_update_interval=` explicitly so the new fritz_utils-sourced default is invisible. If any of these fail, the change went further than intended.
- tests/test_admin_panel.py:876-924 (TestChatPendingImages) — unaffected; those fakes go through /chat/send, which never builds a streaming callback (admin_panel.py:444-457).
- tests/test_fritz_utils.py — unaffected; TestConstantDefaults (99-121) checks specific named constants, not an exhaustive list, so adding two is safe.
- tests/test_scheduler.py, tests/test_bot_commands.py — unaffected; those ask_stuff call sites (scheduler.py:111, bot_commands.py:395) pass no callbacks.

### Manual verification

- Long simple reply: with Ollama up, DM the bot 'write me 300 words on the merits of a well-pressed shirt'. Discord message should grow in ~1.5 s increments and the final text must match the full reply (final_update). Watch the log for 'Error editing message' — none expected.
- Web O(n) check: open :8001/chat, send the same prompt, DevTools → Network → /chat/stream → EventStream. Every `token` frame must be a short delta, not a growing prefix. Sum the token payload bytes; it should be within ~2x the reply length, not quadratic.
- Tool-call turn: ask 'search the web for the current price of tea in China'. Expect a `progress` italic line, then — if the model narrated before calling the tool — a `reset` frame that clears the bubble before the real answer streams. Confirm the final bubble text equals the `done` payload text with no duplicated preamble. This is the scenario the hermetic tests can only simulate.
- Plan mode: give a request the planner splits into >1 step (e.g. 'research X, then write me a short brief on it'). Confirm zero token/reset frames during the executor steps (only `progress` 'Step n/N: ...' lines), then a single clean streamed answer from the synthesizer.
- Reasoning leak check: confirm THINKING_OLLAMA_MODEL (default `gpt-oss`) is not emitting chain-of-thought into `content`. If <think> text appears in the bubble it was already appearing before this change (the old values-mode path used the same `.content`), but it is now visible as it is produced — decide whether to filter.
- Degradation check: kill Ollama mid-reply and confirm the SSE stream ends with a single `error` event and the Discord path lands the '❌ An error occurred' edit (main_discord.py:208).
- `ruff check .` and `pytest tests/ --cov=. --cov-fail-under=60` locally, matching .github/workflows/ci.yml.

## Risks

- Pre-tool-call preamble becomes visible, then the bubble clears. Today's `elif` at mister_fritz.py:428 silently suppresses any AIMessage that carries tool_calls; token streaming does not. I reproduced this with a scripted model. Detection: manual scenario 3. If the flicker is judged worse than the latency, the fallback is to buffer a segment until the turn is known to be text-only — a bigger change, explicitly deferred.
- Upstream contract drift. Token flow depends on `BaseChatModel._generate_with_cache` auto-switching to `_stream` because LangGraph's `StreamMessagesHandler` is a `_StreamingCallbackHandler`. That is a private-ish internal. Also note requirements.txt:114 pins `langchain-core==1.2.5` while this machine has 1.2.19 — I verified on 1.2.19 only. Detection: the `TestLangGraphMessagesContract` canary test (step 3) fails in CI on any bump that breaks it, instead of streaming silently regressing to nothing.
- Callback arity break for out-of-tree callers. Only main_discord.py:197 and admin_panel.py:527 pass a streaming callback in this repo (grep-verified), but any external caller passing a 1-arg function now raises TypeError. `_DeltaEmitter.flush` catches and logs it, so the turn completes without streaming rather than failing — degraded, not broken.
- Discord rate-limit pressure. Deltas arrive ~40x/s. Mitigated twice: the callback's `DISCORD_STREAM_MIN_INTERVAL` guard before `run_coroutine_threadsafe`, and `StreamingMessageHandler._perform_update`'s own sleep. Detection: 'Error editing message' warnings at main_discord.py:68, and Discord 429s in the gateway log.
- Stale browser tab with cached old chat.html against a new server would show only the last token until `done` corrects it. Self-healing and transient — the JS is inlined in the Jinja template (admin_templates/chat.html:162-407), so it is served fresh with every page load and there is no separate cached asset to bust.
- SSE frame count: ~1 frame per token, so a 2000-char reply is ~500 frames. Fine over localhost/LAN (the documented deployment model), potentially chatty over a slow tunnel. `STREAM_MIN_CHARS` is the dial.
- Whitespace fidelity on the wire. The encoder writes `data: {chunk}` (admin_panel.py:576) and the client strips exactly one leading space (chat.html:349), so ' well' round-trips — but this was previously untested because full-text payloads made it invisible. Covered by the new TestChatStreamTokenFidelity tests.

## Rollback
"Plain `git revert` of the eight commits — there is no schema, migration, or persisted state involved, so revert is total and instant. No feature flag is warranted: a true on/off switch would require keeping the old whole-AIMessage `elif` branch alive as a second code path in the executor, which doubles the surface for something meant to fully replace it. Instead, `STREAM_MIN_CHARS` is the live mitigation: set it to a huge value (e.g. 1000000) and the emitter buffers the entire reply, firing exactly one `flush()` at end-of-stream — one `reset` + one `token` with the complete text, behaviourally close to today's coarse updates, with no redeploy. If only Discord is misbehaving, raise `DISCORD_STREAM_MIN_INTERVAL` to slow edits without touching the web surface. If only the web surface is misbehaving, `chat.html` is server-rendered, so reverting just that one file's token/reset branch (back to `body.textContent = data`) plus `admin_panel._streaming_callback` (back to putting `accumulated`) restores the old web behaviour while Discord keeps the new streaming."

## Open questions for you to decide

- Should pre-tool-call preamble text be shown at all? Showing it is more transparent and gives the fastest first token, but produces a visible clear when the real answer starts. Swallowing it (buffer the first segment until the turn is known to be text-only) costs first-token latency on every tool-using turn. I chose to show it + reset; the owner should look at manual scenario 3 with the real model and decide.
- `STREAM_MIN_CHARS` default of 1 vs something chunkier like 8. 1 is the smoothest and matches what the README already promises; 8 cuts SSE frames ~8x with barely perceptible coarseness. I defaulted to 1.
- Should a `reset` also clear the accumulated `progress` lines in the web UI? Today they're removed only on `done`/`error` (chat.html:376, 381). Arguably a new answer segment should clear the 'Searching the web...' line too. Left as-is to keep the diff tight.
- Should plan-mode steps stream into the web UI as collapsible per-step output rather than being silent? That is a UX decision that belongs with `web-chat-redesign`, not here — the plumbing (a per-step segment id) would already support it.
- Bigger refactor deliberately deferred: collapsing `progress_callback` + `streaming_callback` into one `event_callback(event: dict)` channel carrying {'type': 'token'|'reset'|'progress'|'tool_start'|'image'} — it would unify the two surfaces, let the outer graph emit events too, and make the SSE endpoint a near-passthrough. Tempting while touching all these files, but it changes every consumer and every test at once. Do it separately if a third surface (Telegram streaming) ever appears.
