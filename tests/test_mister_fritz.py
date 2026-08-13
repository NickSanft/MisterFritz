"""
Tests for the LangGraph nodes in mister_fritz.py — the planner's JSON parsing,
and the conversation window the executor feeds its ReAct sub-agent.

mister_fritz initialises ChatOllama clients, SqliteSaver, and writes a Mermaid
diagram at import time. We stub heavy modules (image_generator, document_engine)
before importing, but ChatOllama itself is real — we patch the instance after
import to control what invoke() returns.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

# ddgs / image_generator / document_engine are stubbed in tests/conftest.py
# before any test module is collected.

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

import mister_fritz  # noqa: E402
from mister_fritz import _history_window, executor, planner  # noqa: E402


def _state_with_message(text: str) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "image_paths": [],
        "user_image_paths": [],
        "original_request": "",
        "plan": [],
        "current_step": 0,
        "step_results": [],
    }


def _thread(n_pairs: int, body: str = "x" * 40) -> list:
    """A realistic checkpointed transcript: alternating Human/AI turns."""
    msgs = []
    for i in range(n_pairs):
        msgs.append(HumanMessage(content=f"q{i} {body}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i} {body}", id=f"a{i}"))
    return msgs


def _fake_response(content: str) -> MagicMock:
    r = MagicMock()
    r.content = content
    return r


class TestPlannerParsing(unittest.TestCase):
    """Cover the brittle JSON-extraction logic in planner()."""

    def _run(self, llm_content: str, message: str = "do a thing"):
        state = _state_with_message(message)
        with patch.object(mister_fritz, "fast_ollama_instance") as m:
            m.invoke.return_value = _fake_response(llm_content)
            return planner(state, config={"metadata": {}})

    def test_simple_mode_when_needs_planning_false(self):
        result = self._run('{"needs_planning": false}')
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["current_step"], 0)
        self.assertEqual(result["step_results"], [])

    def test_multi_step_plan_extracted(self):
        result = self._run(
            '{"needs_planning": true, "steps": ["fetch data", "summarise it"]}'
        )
        self.assertEqual(result["plan"], ["fetch data", "summarise it"])
        self.assertEqual(result["current_step"], 0)

    def test_plan_capped_at_five_steps(self):
        many = ", ".join(f'"step {i}"' for i in range(10))
        result = self._run(f'{{"needs_planning": true, "steps": [{many}]}}')
        self.assertEqual(len(result["plan"]), 5)

    def test_handles_markdown_code_fences(self):
        result = self._run(
            '```json\n{"needs_planning": true, "steps": ["a", "b"]}\n```'
        )
        self.assertEqual(result["plan"], ["a", "b"])

    def test_handles_bare_code_fences(self):
        result = self._run('```\n{"needs_planning": false}\n```')
        self.assertEqual(result["plan"], [])

    def test_handles_surrounding_text(self):
        result = self._run(
            'Sure! Here is my plan: {"needs_planning": true, "steps": ["x", "y"]} '
            'Hope that helps.'
        )
        self.assertEqual(result["plan"], ["x", "y"])

    def test_malformed_json_falls_back_to_simple_mode(self):
        result = self._run("not json at all")
        self.assertEqual(result["plan"], [])

    def test_empty_response_falls_back_to_simple_mode(self):
        result = self._run("")
        self.assertEqual(result["plan"], [])

    def test_single_step_plan_collapses_to_simple_mode(self):
        # The planner explicitly requires >1 step to enter plan mode.
        result = self._run('{"needs_planning": true, "steps": ["just one"]}')
        self.assertEqual(result["plan"], [])

    def test_non_string_steps_coerced(self):
        # If the LLM returns numeric steps, planner stringifies them.
        result = self._run('{"needs_planning": true, "steps": [1, 2, 3]}')
        self.assertEqual(result["plan"], ["1", "2", "3"])

    def test_original_request_preserved(self):
        result = self._run('{"needs_planning": false}', message="my original ask")
        self.assertEqual(result["original_request"], "my original ask")

    def test_llm_exception_falls_back_to_simple_mode(self):
        state = _state_with_message("anything")
        with patch.object(mister_fritz, "fast_ollama_instance") as m:
            m.invoke.side_effect = RuntimeError("ollama exploded")
            result = planner(state, config={"metadata": {}})
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["current_step"], 0)


class _RecordingAgent:
    """Stands in for the compiled ReAct agent; captures inputs and stream_mode.

    Yields (mode, payload) tuples, which is what LangGraph produces whenever
    stream_mode is a LIST — even a one-element one. A bare string yields raw
    payloads instead, which is why the executor always passes a list.
    """

    def __init__(self, reply="Very well, sir.", script=None):
        self.seen = None
        self.stream_mode = None
        self._reply = reply
        self._script = script

    def stream(self, inputs, config=None, stream_mode=None):
        self.seen = inputs
        self.stream_mode = stream_mode
        if self._script is not None:
            yield from self._script
            return
        yield ("values", {"messages": [AIMessage(content=self._reply)]})


def _run_executor(state, agent, **patches):
    """Invoke executor with the heavy collaborators stubbed out.

    executor takes the no-file-tools path (and therefore
    _get_conversation_agent()) only when metadata has no workspace_root, no
    channel_id and no schedule_manager.
    """
    defaults = {
        "search_memories_internal": lambda *a, **k: "{}",
        "get_user_profile": lambda *a, **k: {},
    }
    defaults.update(patches)
    with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent), \
         patch.object(mister_fritz, "search_memories_internal",
                      side_effect=defaults["search_memories_internal"]), \
         patch.object(mister_fritz, "get_user_profile",
                      side_effect=defaults["get_user_profile"]):
        return executor(state, config={"metadata": {"user_id": "tester"}})


class TestHistoryWindow(unittest.TestCase):
    """The sub-agent is compiled without a checkpointer, so this window is the
    model's entire short-term memory for the turn."""

    def test_returns_all_messages_when_under_budget(self):
        msgs = _thread(3)
        window = _history_window(msgs)
        self.assertEqual(len(window), len(msgs))
        # Identity is preserved, so ids and roles survive the trim.
        self.assertTrue(all(a is b for a, b in zip(window, msgs)))

    def test_trims_oldest_when_over_budget(self):
        msgs = _thread(20, "y" * 400)
        with patch.object(mister_fritz, "HISTORY_TOKEN_BUDGET", 500):
            window = _history_window(msgs)
        self.assertGreater(len(window), 0)
        self.assertLess(len(window), len(msgs))
        # The newest turn is always kept — it is the question being asked.
        self.assertIs(window[-1], msgs[-1])

    def test_returns_empty_when_newest_message_alone_exceeds_budget(self):
        # Verified real trim_messages behaviour, and the reason callers must
        # have a fallback rather than trusting the window to be non-empty.
        msgs = [HumanMessage(content="x" * 100000, id="h")]
        self.assertEqual(_history_window(msgs), [])

    def test_returns_empty_for_empty_input(self):
        self.assertEqual(_history_window([]), [])

    def test_zero_budget_returns_empty_kill_switch(self):
        with patch.object(mister_fritz, "HISTORY_TOKEN_BUDGET", 0):
            self.assertEqual(_history_window(_thread(3)), [])

    def test_window_never_opens_on_an_ai_message(self):
        # A window starting mid-exchange reads as though the user said nothing.
        msgs = [AIMessage(content="a", id="1"), HumanMessage(content="b", id="2"),
                AIMessage(content="c", id="3"), HumanMessage(content="d", id="4")]
        window = _history_window(msgs)
        self.assertEqual(window[0].type, "human")

    def test_trim_failure_is_non_fatal(self):
        with patch.object(mister_fritz, "trim_messages",
                          side_effect=RuntimeError("boom")):
            self.assertEqual(_history_window(_thread(2)), [])


class TestExecutorInputs(unittest.TestCase):
    """What the executor actually hands the model."""

    def _state(self, messages, **over):
        state = {
            "messages": messages, "image_paths": [], "user_image_paths": [],
            "original_request": "", "plan": [], "current_step": 0,
            "step_results": [],
        }
        state.update(over)
        return state

    def test_simple_mode_passes_the_window_with_roles(self):
        msgs = _thread(2) + [HumanMessage(content="and now?", id="h9")]
        agent = _RecordingAgent()
        _run_executor(self._state(msgs), agent)
        sent = agent.seen["messages"]
        self.assertEqual(sent[0][0], "system")
        # Speaker roles survive — this is what the AIMessage fix bought.
        self.assertEqual([m.type for m in sent[1:]],
                         ["human", "ai", "human", "ai", "human"])

    def test_simple_mode_does_not_duplicate_the_latest_turn(self):
        # history[-1] IS the current turn; appending a ("user", ...) entry too
        # would ask the question twice.
        msgs = _thread(1) + [HumanMessage(content="unique-marker-42", id="h9")]
        agent = _RecordingAgent()
        _run_executor(self._state(msgs), agent)
        rendered = [m if isinstance(m, tuple) else m.content
                    for m in agent.seen["messages"]]
        occurrences = sum(1 for m in rendered
                          if isinstance(m, str) and "unique-marker-42" in m)
        self.assertEqual(occurrences, 1)

    def test_simple_mode_returns_ai_message_not_bare_string(self):
        # Regression guard for the add_messages coercion bug: a bare str is
        # stored as a HumanMessage, making the transcript unreadable.
        agent = _RecordingAgent()
        result = _run_executor(
            self._state([HumanMessage(content="hi", id="h1")]), agent)
        self.assertIsInstance(result["messages"][0], AIMessage)

    def test_zero_budget_reproduces_the_old_single_message_input(self):
        # The rollback contract: HISTORY_TOKEN_BUDGET=0 must be byte-identical
        # to pre-window behaviour, with no deploy required.
        msgs = _thread(2) + [HumanMessage(content="latest turn", id="h9")]
        agent = _RecordingAgent()
        with patch.object(mister_fritz, "HISTORY_TOKEN_BUDGET", 0):
            _run_executor(self._state(msgs), agent)
        sent = agent.seen["messages"]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0][0], "system")
        self.assertEqual(sent[1], ("user", "latest turn"))

    def test_oversized_latest_message_still_reaches_the_model(self):
        # The window is [] here; the executor must fall back rather than send
        # a system-prompt-only input.
        msgs = [HumanMessage(content="z" * 100000, id="h1")]
        agent = _RecordingAgent()
        _run_executor(self._state(msgs), agent)
        sent = agent.seen["messages"]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1][0], "user")
        self.assertEqual(len(sent[1][1]), 100000)

    def test_plan_mode_excludes_the_current_turn(self):
        # The current request is already restated verbatim as "Original
        # request:" inside the step prompt.
        msgs = _thread(2) + [HumanMessage(content="do-the-thing", id="h9")]
        agent = _RecordingAgent()
        state = self._state(msgs, plan=["step one", "step two"],
                            original_request="do-the-thing")
        _run_executor(state, agent)
        sent = agent.seen["messages"]
        window = [m for m in sent[1:-1]]
        self.assertNotIn("do-the-thing", [getattr(m, "content", "") for m in window])
        # Final entry is the synthetic step prompt.
        self.assertEqual(sent[-1][0], "user")
        self.assertIn("Current task (step 1/2)", sent[-1][1])


class TestMemoryInjectionCap(unittest.TestCase):
    """Uncapped, the Chroma blob can evict the window and the persona itself."""

    def _state(self):
        return {
            "messages": [HumanMessage(content="hello", id="h1")],
            "image_paths": [], "user_image_paths": [], "original_request": "",
            "plan": [], "current_step": 0, "step_results": [],
        }

    def test_oversized_memory_blob_is_truncated(self):
        blob = json.dumps({f"memory_{i}": "v" * 2000 for i in range(30)})
        agent = _RecordingAgent()
        with patch.object(mister_fritz, "MEMORY_INJECT_MAX_CHARS", 4000):
            _run_executor(self._state(), agent,
                          search_memories_internal=lambda *a, **k: blob)
        system_prompt = agent.seen["messages"][0][1]
        injected = system_prompt.split("What I know about this user:\n")[1]
        self.assertLess(len(injected), len(blob))
        # Chroma returns most-similar-first, so the best match must survive.
        self.assertIn("memory_0", injected)

    def test_small_memory_blob_passes_through_unmodified(self):
        blob = json.dumps({"memory_of_pie": "the user likes pie"})
        agent = _RecordingAgent()
        _run_executor(self._state(), agent,
                      search_memories_internal=lambda *a, **k: blob)
        system_prompt = agent.seen["messages"][0][1]
        self.assertIn("the user likes pie", system_prompt)

    def test_non_json_blob_falls_back_to_a_hard_cut(self):
        agent = _RecordingAgent()
        with patch.object(mister_fritz, "MEMORY_INJECT_MAX_CHARS", 100):
            _run_executor(self._state(), agent,
                          search_memories_internal=lambda *a, **k: "not json " * 500)
        system_prompt = agent.seen["messages"][0][1]
        injected = system_prompt.split("What I know about this user:\n")[1]
        self.assertEqual(len(injected), 100)


class TestChunkText(unittest.TestCase):
    """ChatOllama emits str content today, but langchain-core v1 can make it a
    list of content blocks. `.text` normalises both."""

    def test_str_content_returned(self):
        self.assertEqual(mister_fritz._chunk_text(AIMessageChunk(content="hi")), "hi")

    def test_text_property_is_preferred(self):
        msg = MagicMock()
        msg.text = "from text"
        msg.content = "from content"
        self.assertEqual(mister_fritz._chunk_text(msg), "from text")

    def test_callable_text_is_invoked(self):
        msg = MagicMock()
        msg.text = lambda: "called"
        msg.content = "ignored"
        self.assertEqual(mister_fritz._chunk_text(msg), "called")

    def test_list_content_returns_empty_string(self):
        msg = MagicMock()
        msg.text = None
        msg.content = [{"type": "text", "text": "block"}]
        self.assertEqual(mister_fritz._chunk_text(msg), "")


class TestDeltaEmitter(unittest.TestCase):
    """The contract: streaming_callback(delta, accumulated, restart)."""

    def setUp(self):
        self.calls = []

    def _emitter(self, min_chars=1):
        return mister_fritz._DeltaEmitter(
            lambda d, a, r: self.calls.append((d, a, r)), min_chars=min_chars)

    def test_single_segment_emits_delta_and_accumulated(self):
        em = self._emitter()
        em.feed("Very", segment_id="s1")
        em.feed(" well", segment_id="s1")
        em.feed(", sir.", segment_id="s1")
        self.assertEqual(self.calls, [
            ("Very", "Very", True),
            (" well", "Very well", False),
            (", sir.", "Very well, sir.", False),
        ])
        # The invariant the whole design rests on.
        self.assertEqual("".join(d for d, _, _ in self.calls), self.calls[-1][1])

    def test_new_segment_resets_accumulator_and_flags_restart(self):
        # The preamble-then-tool-call-then-answer case, verified to happen for
        # real: the two halves arrive under different chunk ids.
        em = self._emitter()
        em.feed("Let me look. ", segment_id="turn1")
        em.feed("I found it sir.", segment_id="turn2")
        self.assertEqual(self.calls, [
            ("Let me look. ", "Let me look. ", True),
            ("I found it sir.", "I found it sir.", True),
        ])

    def test_min_chars_buffers_until_threshold(self):
        em = self._emitter(min_chars=5)
        em.feed("a", segment_id="s")
        em.feed("a", segment_id="s")
        em.feed("a", segment_id="s")
        self.assertEqual(self.calls, [])
        em.feed("bb", segment_id="s")
        self.assertEqual(self.calls, [("aaabb", "aaabb", True)])

    def test_flush_emits_sub_threshold_tail(self):
        # Without this the last few characters of every reply are stranded.
        em = self._emitter(min_chars=100)
        em.feed("short tail", segment_id="s")
        self.assertEqual(self.calls, [])
        em.flush()
        self.assertEqual(self.calls, [("short tail", "short tail", True)])

    def test_flush_is_a_noop_when_buffer_empty(self):
        em = self._emitter()
        em.flush()
        em.feed("x", segment_id="s")
        em.flush()
        self.assertEqual(len(self.calls), 1)

    def test_none_callback_is_a_noop(self):
        em = mister_fritz._DeltaEmitter(None)
        em.feed("anything", segment_id="s")
        em.flush()  # must not raise

    def test_empty_text_is_ignored(self):
        em = self._emitter()
        em.feed("", segment_id="s")
        self.assertEqual(self.calls, [])

    def test_callback_exception_is_swallowed(self):
        # A broken UI consumer must not kill the turn.
        def boom(d, a, r):
            raise RuntimeError("consumer exploded")
        em = mister_fritz._DeltaEmitter(boom)
        em.feed("x", segment_id="s")  # must not raise


def _chunk(text, cid):
    return AIMessageChunk(content=text, id=cid)


class TestExecutorTokenStreaming(unittest.TestCase):
    """The executor's stream loop: values drive progress, messages drive tokens."""

    def _state(self, **over):
        state = {
            "messages": [HumanMessage(content="hello", id="h1")],
            "image_paths": [], "user_image_paths": [], "original_request": "",
            "plan": [], "current_step": 0, "step_results": [],
        }
        state.update(over)
        return state

    def _run(self, script, state=None, with_callback=True):
        calls, progress = [], []
        agent = _RecordingAgent(script=script)
        metadata = {"user_id": "tester"}
        if with_callback:
            metadata["streaming_callback"] = lambda d, a, r: calls.append((d, a, r))
        metadata["progress_callback"] = progress.append
        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent), \
             patch.object(mister_fritz, "search_memories_internal", return_value="{}"), \
             patch.object(mister_fritz, "get_user_profile", return_value={}):
            result = executor(state or self._state(), config={"metadata": metadata})
        return agent, calls, progress, result

    def test_ai_message_chunks_become_deltas(self):
        script = [
            ("messages", (_chunk("Very", "t1"), {})),
            ("messages", (_chunk(" well", "t1"), {})),
            ("messages", (_chunk(", sir.", "t1"), {})),
            ("values", {"messages": [AIMessage(content="Very well, sir.")]}),
        ]
        _agent, calls, _p, _r = self._run(script)
        self.assertEqual([d for d, _, _ in calls], ["Very", " well", ", sir."])
        self.assertEqual([r for _, _, r in calls], [True, False, False])
        self.assertEqual(calls[-1][1], "Very well, sir.")

    def test_tool_messages_in_the_messages_stream_are_ignored(self):
        # The messages stream carries whole ToolMessages from the tools node;
        # only AIMessageChunks are LLM tokens.
        script = [
            ("messages", (ToolMessage(content="tool output", tool_call_id="c1"), {})),
            ("messages", (_chunk("real", "t1"), {})),
            ("values", {"messages": [AIMessage(content="real")]}),
        ]
        _agent, calls, _p, _r = self._run(script)
        self.assertEqual([d for d, _, _ in calls], ["real"])

    def test_new_chunk_id_restarts_the_segment(self):
        # Preamble, tool call, then the real answer under a NEW id. Without
        # restart the client would render "Let me look. I found it sir."
        script = [
            ("messages", (_chunk("Let me look. ", "t1"), {})),
            ("messages", (_chunk("I found it sir.", "t2"), {})),
            ("values", {"messages": [AIMessage(content="I found it sir.")]}),
        ]
        _agent, calls, _p, _r = self._run(script)
        self.assertEqual([r for _, _, r in calls], [True, True])
        self.assertEqual(calls[-1][1], "I found it sir.")

    def test_tool_calls_in_values_still_fire_progress_once(self):
        tool_call = {"name": "search_web", "args": {}, "id": "c1", "type": "tool_call"}
        ai = AIMessage(content="", tool_calls=[tool_call])
        script = [
            ("values", {"messages": [ai]}),
            ("values", {"messages": [ai]}),          # same tool again
            ("values", {"messages": [AIMessage(content="done")]}),
        ]
        _agent, _calls, progress, _r = self._run(script)
        self.assertEqual(progress, ["Searching the web..."])

    def test_requests_values_and_messages_when_a_callback_is_present(self):
        script = [("values", {"messages": [AIMessage(content="x")]})]
        agent, _c, _p, _r = self._run(script)
        self.assertEqual(agent.stream_mode, ["values", "messages"])

    def test_requests_values_only_without_a_callback(self):
        # Still a LIST — a bare string would change the payload shape and the
        # loop's tuple unpacking would blow up.
        script = [("values", {"messages": [AIMessage(content="x")]})]
        agent, _c, _p, _r = self._run(script, with_callback=False)
        self.assertEqual(agent.stream_mode, ["values"])

    def test_plan_mode_never_streams(self):
        # Intermediate steps stay silent; the synthesizer owns the visible one.
        state = self._state(plan=["step a", "step b"], original_request="do it")
        script = [("values", {"messages": [AIMessage(content="step result")]})]
        agent, calls, progress, _r = self._run(script, state=state)
        self.assertEqual(agent.stream_mode, ["values"])
        self.assertEqual(calls, [])
        self.assertIn("Step 1/2: step a", progress)

    def test_sub_threshold_tail_is_flushed(self):
        script = [
            ("messages", (_chunk("tail", "t1"), {})),
            ("values", {"messages": [AIMessage(content="tail")]}),
        ]
        with patch.object(mister_fritz, "STREAM_MIN_CHARS", 999):
            _agent, calls, _p, _r = self._run(script)
        self.assertEqual([d for d, _, _ in calls], ["tail"])


class TestSynthesizerStreaming(unittest.TestCase):
    def _state(self):
        return {
            "messages": [], "image_paths": [], "user_image_paths": [],
            "original_request": "do it", "plan": ["a", "b"],
            "current_step": 2, "step_results": ["r1", "r2"],
        }

    def test_streams_deltas_under_one_segment(self):
        calls = []
        fake = MagicMock()
        fake.stream.return_value = iter([
            AIMessageChunk(content="Very"), AIMessageChunk(content=" well."),
        ])
        with patch.object(mister_fritz, "ollama_instance", fake):
            mister_fritz.synthesizer(self._state(), config={"metadata": {
                "streaming_callback": lambda d, a, r: calls.append((d, a, r))}})
        # First emission restarts, wiping whatever the executor steps left.
        self.assertEqual([r for _, _, r in calls], [True, False])
        self.assertEqual(calls[-1][1], "Very well.")

    def test_invoke_fallback_corrects_the_ui(self):
        # A failed stream may already have painted partial text; the fallback
        # must replace it, not append to it.
        calls = []
        fake = MagicMock()
        fake.stream.side_effect = RuntimeError("stream died")
        fake.invoke.return_value = MagicMock(content="The real answer.")
        with patch.object(mister_fritz, "ollama_instance", fake):
            mister_fritz.synthesizer(self._state(), config={"metadata": {
                "streaming_callback": lambda d, a, r: calls.append((d, a, r))}})
        self.assertEqual(calls, [("The real answer.", "The real answer.", True)])


class TestLangGraphMessagesContract(unittest.TestCase):
    """Third-party canary. Pins the exact upstream behaviour the executor
    depends on, so a langgraph / langchain-core bump fails loudly here instead
    of silently reverting streaming to nothing.

    Hermetic: a scripted BaseChatModel, no Ollama and no network.
    """

    @staticmethod
    def _build_agent():
        from typing import Iterator, Optional

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
        from langchain_core.tools import tool
        from langchain.agents import create_agent

        @tool(parse_docstring=True)
        def lookup(query: str) -> str:
            """Look something up.

            Args:
                query: what to look up.
            """
            return "5"

        class ScriptedModel(BaseChatModel):
            turn: int = 0

            @property
            def _llm_type(self) -> str:
                return "scripted"

            def bind_tools(self, tools, **kw):
                return self

            def _stream(self, messages, stop=None,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        **kwargs) -> Iterator[ChatGenerationChunk]:
                self.turn += 1
                if self.turn == 1:
                    for piece in ["Let ", "me ", "look. "]:
                        c = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                        if run_manager:
                            run_manager.on_llm_new_token(piece, chunk=c)
                        yield c
                    yield ChatGenerationChunk(message=AIMessageChunk(
                        content="",
                        tool_calls=[{"name": "lookup", "args": {"query": "x"},
                                     "id": "call_1", "type": "tool_call"}]))
                else:
                    for piece in ["I ", "found ", "it."]:
                        c = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                        if run_manager:
                            run_manager.on_llm_new_token(piece, chunk=c)
                        yield c

            def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
                text = "".join(
                    c.message.content for c in self._stream(messages, stop, run_manager))
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content=text))])

        return create_agent(ScriptedModel(), tools=[lookup])

    def setUp(self):
        self.agent = self._build_agent()
        self.inputs = {"messages": [("system", "terse"), ("user", "go")]}

    def test_list_stream_mode_yields_mode_payload_tuples(self):
        for item in self.agent.stream(self.inputs, stream_mode=["values", "messages"]):
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_one_element_list_still_yields_tuples(self):
        # Why the executor always passes a list, even for values-only.
        first = next(iter(self.agent.stream(self.inputs, stream_mode=["values"])))
        self.assertIsInstance(first, tuple)
        self.assertEqual(first[0], "values")

    def test_bare_string_yields_raw_payloads(self):
        # The shape the executor must NOT ask for.
        first = next(iter(self.agent.stream(self.inputs, stream_mode="values")))
        self.assertNotIsInstance(first, tuple)

    def test_messages_mode_emits_ai_message_chunks(self):
        chunks = [p[0] for m, p in
                  self.agent.stream(self.inputs, stream_mode=["values", "messages"])
                  if m == "messages" and isinstance(p[0], AIMessageChunk)]
        self.assertGreaterEqual(len(chunks), 6)

    def test_messages_mode_also_emits_tool_messages(self):
        # Which is exactly why the executor filters on AIMessageChunk.
        from langchain_core.messages import ToolMessage as _TM
        tools = [p[0] for m, p in
                 self.agent.stream(self.inputs, stream_mode=["values", "messages"])
                 if m == "messages" and isinstance(p[0], _TM)]
        self.assertGreaterEqual(len(tools), 1)

    def test_chunk_id_is_stable_within_a_turn_and_changes_across_turns(self):
        # The free segment key the emitter uses for `restart`.
        runs = []
        for m, p in self.agent.stream(self.inputs, stream_mode=["values", "messages"]):
            if m == "messages" and isinstance(p[0], AIMessageChunk):
                if not runs or runs[-1] != p[0].id:
                    runs.append(p[0].id)
        self.assertEqual(len(runs), 2, "preamble and answer must be separate segments")


if __name__ == "__main__":
    unittest.main()
