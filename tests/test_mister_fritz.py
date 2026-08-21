"""
Tests for the LangGraph nodes in mister_fritz.py — the conversation window the
executor feeds its ReAct sub-agent, token streaming, and off-path summarisation.

mister_fritz initialises ChatOllama clients, SqliteSaver, and writes a Mermaid
diagram at import time. We stub heavy modules (image_generator, document_engine)
before importing, but ChatOllama itself is real — we patch the instance after
import to control what invoke() returns.
"""
import json
import threading
import time
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
from mister_fritz import _history_window, executor  # noqa: E402


def _thread(n_pairs: int, body: str = "x" * 40) -> list:
    """A realistic checkpointed transcript: alternating Human/AI turns."""
    msgs = []
    for i in range(n_pairs):
        msgs.append(HumanMessage(content=f"q{i} {body}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i} {body}", id=f"a{i}"))
    return msgs


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

    def test_turn_framing_is_applied_to_the_current_turn_only(self):
        """The framing is an instruction about THIS turn. Replaying it on older
        turns spent context on boilerplate and carried stale per-turn
        instructions (e.g. voice mode's "30 words or less") forward."""
        older = HumanMessage(content="first question", id="h1",
                             additional_kwargs={"ctx": "User is speaking from Discord. "
                                                       "Please answer in 30 words or less."})
        current = HumanMessage(content="second question", id="h2",
                               additional_kwargs={"ctx": "User is texting from Discord (User ID: alice)"})
        agent = _RecordingAgent()
        _run_executor(self._state([older, AIMessage(content="ok", id="a1"), current]), agent)
        sent = agent.seen["messages"]
        bodies = [m.content for m in sent[1:] if not isinstance(m, tuple)]
        # The old turn is replayed as the user's actual words, unwrapped.
        self.assertEqual(bodies[0], "first question")
        self.assertNotIn("30 words or less", bodies[0])
        # The current turn carries its framing.
        self.assertIn("Context:", bodies[-1])
        self.assertIn("User is texting from Discord (User ID: alice)", bodies[-1])
        self.assertIn("second question", bodies[-1])

    def test_message_without_ctx_is_passed_through_unchanged(self):
        """Fritz's own replies, and anything checkpointed before ctx existed."""
        msgs = [HumanMessage(content="plain old turn", id="h1")]
        agent = _RecordingAgent()
        _run_executor(self._state(msgs), agent)
        bodies = [m.content for m in agent.seen["messages"][1:] if not isinstance(m, tuple)]
        self.assertEqual(bodies, ["plain old turn"])

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

    def test_single_memory_larger_than_the_budget_is_cut_not_dropped(self):
        """Regression: the cap deleted the memory instead of shortening it.

        The keep-loop breaks on the first entry that crosses the budget, so a
        lone oversized memory left `kept` empty and json.dumps({}) == "{}" —
        the prompt then carried the literal text "What I know about this
        user:\n{}" and the user's most relevant memory was gone entirely.
        This is the common case, not a corner one: summarize_conversation
        writes one large value per memory and a summary of a long thread
        routinely runs past the budget.
        """
        blob = json.dumps({"summary_of_everything": "v" * 10000})
        agent = _RecordingAgent()
        with patch.object(mister_fritz, "MEMORY_INJECT_MAX_CHARS", 4000):
            _run_executor(self._state(), agent,
                          search_memories_internal=lambda *a, **k: blob)
        system_prompt = agent.seen["messages"][0][1]
        injected = system_prompt.split("What I know about this user:\n")[1]
        self.assertNotEqual(injected.strip(), "{}")
        self.assertEqual(len(injected), 4000)
        self.assertIn("summary_of_everything", injected)

    def test_empty_memory_result_injects_no_header(self):
        """An empty blob must not leave a dangling 'What I know about this
        user:' header with nothing under it."""
        agent = _RecordingAgent()
        _run_executor(self._state(), agent,
                      search_memories_internal=lambda *a, **k: "{}")
        system_prompt = agent.seen["messages"][0][1]
        self.assertNotIn("What I know about this user:", system_prompt)

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
        # Asserted against the registry rather than a literal, so rewording a
        # notice does not fail a test that is really about de-duplication.
        self.assertEqual(progress, [mister_fritz.TOOL_NOTICES["search_web"]])
        self.assertEqual(len(progress), 1, "the same tool must notify once")

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

    def test_sub_threshold_tail_is_flushed(self):
        script = [
            ("messages", (_chunk("tail", "t1"), {})),
            ("values", {"messages": [AIMessage(content="tail")]}),
        ]
        with patch.object(mister_fritz, "STREAM_MIN_CHARS", 999):
            _agent, calls, _p, _r = self._run(script)
        self.assertEqual([d for d, _, _ in calls], ["tail"])


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


class TestGraphShape(unittest.TestCase):
    """Plan mode is gone: START → executor → (summarize | END)."""

    def test_graph_has_only_executor_and_summarize(self):
        nodes = set(mister_fritz.app.get_graph().nodes)
        self.assertEqual(nodes, {"__start__", "__end__", "executor",
                                 "summarize_conversation"})

    def test_planner_and_synthesizer_are_gone(self):
        self.assertFalse(hasattr(mister_fritz, "planner"))
        self.assertFalse(hasattr(mister_fritz, "synthesizer"))
        self.assertFalse(hasattr(mister_fritz, "route_executor"))

    def test_state_has_no_vestigial_plan_fields(self):
        keys = set(mister_fritz.EnhancedState.__annotations__)
        for gone in ("plan", "current_step", "step_results", "original_request"):
            with self.subTest(field=gone):
                self.assertNotIn(gone, keys)

    def test_should_continue_routes_on_the_threshold(self):
        under = {"messages": ["m"] * (mister_fritz.SUMMARIZE_THRESHOLD)}
        over = {"messages": ["m"] * (mister_fritz.SUMMARIZE_THRESHOLD + 1)}
        self.assertEqual(mister_fritz.should_continue(under), "__end__")
        self.assertEqual(mister_fritz.should_continue(over), "summarize_conversation")


class TestMemoryKey(unittest.TestCase):
    """Replaces a full 20B round trip whose only output was a label string."""

    def test_builds_a_memory_of_slug(self):
        key = mister_fritz._make_memory_key(
            "Summary made at 2026\r\n The user discussed the pie incident at length")
        self.assertTrue(key.startswith("memory_of_"))
        self.assertIn("pie", key)

    def test_skips_the_timestamp_preamble(self):
        key = mister_fritz._make_memory_key(
            "Summary made at 2026-08-14T10:00:00\r\n They like tea")
        self.assertNotIn("2026", key)

    def test_is_bounded(self):
        key = mister_fritz._make_memory_key("word " * 200)
        self.assertLessEqual(len(key), 64)

    def test_handles_empty_and_punctuation_only(self):
        self.assertEqual(mister_fritz._make_memory_key(""), "memory_of_conversation")
        self.assertEqual(mister_fritz._make_memory_key("!!! ???"), "memory_of_conversation")

    def test_makes_no_network_call(self):
        # The point of the change: no model is consulted for a label.
        with patch.object(mister_fritz, "ollama_instance") as m:
            mister_fritz._make_memory_key("something happened")
        m.invoke.assert_not_called()


class TestOffPathSummarisation(unittest.TestCase):
    """The reply used to block behind three LLM calls, two on the 20B model."""

    def setUp(self):
        # The in-flight guard is keyed per user and module-global. A worker
        # still draining from a previous test would make the next one's
        # summarisation correctly SKIP, which looks like a failure.
        with mister_fritz._summarize_lock:
            mister_fritz._summarize_inflight.clear()
        # Distinct user per test, so even a straggler cannot collide.
        self._user = f"user_{self._testMethodName[:24]}"

    tearDown = setUp

    def _await_worker(self, timeout=5):
        """Block until the background summariser has fully finished.

        Called INSIDE the patch context. A worker that outlives its patches
        would otherwise wander into the next test's mocks — which is exactly
        how a passing suite starts reporting phantom calls.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with mister_fritz._summarize_lock:
                if self._user not in mister_fritz._summarize_inflight:
                    return True
            time.sleep(0.01)
        return False

    def _state(self, n=5):
        return {"messages": [HumanMessage(content=f"m{i}", id=f"h{i}")
                             for i in range(n)],
                "image_paths": [], "user_image_paths": []}

    def _config(self):
        return {"metadata": {"user_id": self._user, "thread_id": self._user}}

    def test_llm_calls_happen_off_the_graph_thread(self):
        # Asserting "not yet called" would be racy — the worker can legitimately
        # win. What matters is WHERE the call runs: not on the thread the graph
        # (and therefore the user's reply) is waiting on.
        graph_thread = threading.get_ident()
        seen = {}
        done = threading.Event()

        def record(*a, **k):
            seen["thread"] = threading.get_ident()
            done.set()
            return MagicMock(content="a summary")

        with patch.object(mister_fritz, "ollama_instance") as thinking, \
             patch.object(mister_fritz, "fast_ollama_instance"), \
             patch.object(mister_fritz, "add_memory"), \
             patch.object(mister_fritz, "update_user_profile"):
            thinking.invoke.side_effect = record
            result = mister_fritz.summarize_conversation(self._state(), self._config())
            self.assertTrue(done.wait(timeout=5), "summariser never ran")
            self.assertTrue(self._await_worker(), "worker did not finish")

        self.assertNotEqual(seen["thread"], graph_thread)
        # The trim is what the node actually returns, and it stays synchronous.
        self.assertTrue(all(type(m).__name__ == "RemoveMessage"
                            for m in result["messages"]))

    def test_the_node_returns_before_the_summary_finishes(self):
        # The latency win: the reply is not gated on the 20B call.
        release = threading.Event()
        started = threading.Event()

        def slow(*a, **k):
            started.set()
            release.wait(timeout=5)
            return MagicMock(content="s")

        with patch.object(mister_fritz, "ollama_instance") as thinking, \
             patch.object(mister_fritz, "fast_ollama_instance"), \
             patch.object(mister_fritz, "add_memory"), \
             patch.object(mister_fritz, "update_user_profile"):
            thinking.invoke.side_effect = slow
            began = time.monotonic()
            result = mister_fritz.summarize_conversation(self._state(), self._config())
            elapsed = time.monotonic() - began
            self.assertTrue(started.wait(timeout=5))
            release.set()
            self.assertTrue(self._await_worker(), "worker did not finish")
        self.assertLess(elapsed, 1.0, "node blocked on the summariser")
        self.assertTrue(result["messages"])

    def test_trim_leaves_the_last_two_messages(self):
        """The trim runs AFTER the executor appended its reply, so keeping only
        the last message left a lone AIMessage — and the next turn opened its
        history window on an answer with no question attached. That is the same
        amnesia the window exists to fix, arriving every SUMMARIZE_THRESHOLD
        messages."""
        state = self._state(n=5)
        with patch.object(mister_fritz, "_summarize_and_profile"):
            result = mister_fritz.summarize_conversation(state, self._config())
        removed = {m.id for m in result["messages"]}
        survivors = [m.id for m in state["messages"] if m.id not in removed]
        self.assertEqual(survivors, ["h3", "h4"])

    def test_the_surviving_pair_is_a_real_exchange(self):
        """Same rule against a realistic alternating transcript: what is left
        behind is a question and its answer, not a dangling half."""
        state = {"messages": _thread(3), "image_paths": [], "user_image_paths": []}
        with patch.object(mister_fritz, "_summarize_and_profile"):
            result = mister_fritz.summarize_conversation(state, self._config())
        removed = {m.id for m in result["messages"]}
        survivors = [m for m in state["messages"] if m.id not in removed]
        self.assertEqual([m.type for m in survivors], ["human", "ai"])

    def test_trim_is_a_no_op_on_a_thread_shorter_than_one_exchange(self):
        state = self._state(n=1)
        with patch.object(mister_fritz, "_summarize_and_profile"):
            result = mister_fritz.summarize_conversation(state, self._config())
        self.assertEqual(result["messages"], [])

    def test_sync_mode_runs_inline(self):
        # The rollback lever: SUMMARIZE_ASYNC=false restores the old ordering.
        with patch.object(mister_fritz, "SUMMARIZE_ASYNC", False), \
             patch.object(mister_fritz, "_summarize_and_profile") as work:
            mister_fritz.summarize_conversation(self._state(), self._config())
        work.assert_called_once()

    def test_concurrent_turns_do_not_stack_summaries(self):
        # Rapid-fire turns would otherwise run several 20B summaries of nearly
        # the same transcript at once.
        release = threading.Event()
        started = threading.Event()

        def slow(*a, **k):
            started.set()
            release.wait(timeout=5)

        with patch.object(mister_fritz, "_summarize_and_profile",
                          side_effect=slow) as work:
            mister_fritz.summarize_conversation(self._state(), self._config())
            self.assertTrue(started.wait(timeout=5))
            # Second turn while the first is still in flight.
            mister_fritz.summarize_conversation(self._state(), self._config())
            release.set()
            self.assertTrue(self._await_worker(), "worker did not finish")
        # One summary, not two, despite two threshold crossings.
        self.assertEqual(work.call_count, 1)

    def test_worker_failure_does_not_break_the_node(self):
        with patch.object(mister_fritz, "SUMMARIZE_ASYNC", False), \
             patch.object(mister_fritz, "ollama_instance") as thinking:
            thinking.invoke.side_effect = RuntimeError("ollama down")
            result = mister_fritz.summarize_conversation(self._state(), self._config())
        # The trim still happened; the turn is not lost to a summariser failure.
        self.assertTrue(result["messages"])


class TestProfileSignalsSchema(unittest.TestCase):
    def test_schema_has_the_expected_fields(self):
        fields = set(mister_fritz.ProfileSignals.model_fields)
        self.assertEqual(fields, {"communication_style", "interests",
                                  "dislikes", "notes"})

    def test_defaults_are_empty_not_none(self):
        # An absent field must mean "no evidence", not a null the profile
        # writer then has to defend against.
        s = mister_fritz.ProfileSignals()
        self.assertEqual(s.communication_style, "")
        self.assertEqual(s.interests, [])
        self.assertEqual(s.dislikes, [])

    def test_extraction_uses_structured_output(self):
        captured = {}

        def fake_structured(schema):
            captured["schema"] = schema
            m = MagicMock()
            m.invoke.return_value = mister_fritz.ProfileSignals(interests=["tea"])
            return m

        with patch.object(mister_fritz, "ollama_instance") as thinking, \
             patch.object(mister_fritz, "fast_ollama_instance") as fast, \
             patch.object(mister_fritz, "add_memory"), \
             patch.object(mister_fritz, "update_user_profile") as upd:
            thinking.invoke.return_value = MagicMock(content="a summary")
            fast.with_structured_output = fake_structured
            mister_fritz._summarize_and_profile(
                [HumanMessage(content="hi", id="h")], "alice", {})
        self.assertIs(captured["schema"], mister_fritz.ProfileSignals)
        upd.assert_called_once()
        self.assertEqual(upd.call_args.args[1]["interests"], ["tea"])


class TestCheckpointStoresWhatTheUserSaid(unittest.TestCase):
    """The transcript holds the user's words, not the prompt scaffolding.

    ask_stuff used to checkpoint the fully-framed prompt, so /chat rendered
    "Context: User is texting from Discord (User ID: nick) Question: …" as the
    user's own message on every reload.
    """

    def _captured_inputs(self, text, source=None, **kwargs):
        captured = {}

        def _fake_stream(payload, config, **_kw):
            captured["inputs"] = payload
            return iter(())

        with patch.object(mister_fritz.app, "stream", side_effect=_fake_stream), \
             patch.object(mister_fritz.app, "get_state") as get_state:
            get_state.return_value = MagicMock(values={"messages": []})
            mister_fritz.ask_stuff(
                text, source or mister_fritz.MessageSource.DISCORD_TEXT,
                "discord-1", display_name="nick", **kwargs)
        return captured["inputs"]

    def test_content_is_the_raw_user_text(self):
        inputs = self._captured_inputs("what is dirtying my git status?")
        msg = inputs["messages"][0]
        self.assertEqual(msg.content, "what is dirtying my git status?")
        self.assertNotIn("Context:", msg.content)
        self.assertNotIn("User ID", msg.content)

    def test_framing_travels_in_additional_kwargs(self):
        inputs = self._captured_inputs("hello")
        kw = inputs["messages"][0].additional_kwargs
        self.assertIn("User is texting from Discord", kw["ctx"])
        self.assertIn("nick", kw["ctx"])
        self.assertIn("ts", kw)

    def test_voice_terseness_does_not_enter_the_transcript(self):
        """A DISCORD_VOICE turn's "30 words or less" is an instruction for that
        turn. Stored in the content it would replay into every later text
        reply through the history window."""
        inputs = self._captured_inputs(
            "how are you", source=mister_fritz.MessageSource.DISCORD_VOICE)
        msg = inputs["messages"][0]
        self.assertEqual(msg.content, "how are you")
        self.assertIn("30 words or less", msg.additional_kwargs["ctx"])


class TestAskStuffIdentity(unittest.TestCase):
    """ask_stuff is where the identity used to be mangled.

    It did `re.sub(r'[^a-zA-Z0-9]', '', user_id)` and put the result in both
    `configurable` and `metadata`. The executor read it back out of metadata to
    key the memory namespace — so every memory was written under the STRIPPED
    id while privacy.forget_memories deleted the RAW one. Nothing here may
    transform the id.
    """

    def _invoke(self, user_id, **kwargs):
        captured = {}

        def _fake_stream(payload, config, **_kw):
            captured["config"] = config
            return iter(())

        with patch.object(mister_fritz.app, "stream", side_effect=_fake_stream), \
             patch.object(mister_fritz.app, "get_state") as get_state:
            get_state.return_value = MagicMock(values={"messages": []})
            mister_fritz.ask_stuff(
                "hello", mister_fritz.MessageSource.LOCAL, user_id, **kwargs)
        return captured["config"]

    def test_identity_reaches_metadata_verbatim(self):
        uid = "web-alice_smith-42"
        config = self._invoke(uid)
        self.assertEqual(config["metadata"]["user_id"], uid)
        self.assertEqual(config["configurable"]["user_id"], uid)

    def test_thread_id_defaults_to_the_identity(self):
        uid = "discord-123456789"
        config = self._invoke(uid)
        self.assertEqual(config["configurable"]["thread_id"], uid)

    def test_explicit_thread_id_wins(self):
        config = self._invoke("web-alice", thread_id="web-alice#7")
        self.assertEqual(config["configurable"]["thread_id"], "web-alice#7")
        # …but does not contaminate the memory namespace.
        self.assertEqual(config["metadata"]["user_id"], "web-alice")

    def test_channel_key_scopes_the_thread_only_when_enabled(self):
        import fritz_utils
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", False):
            off = self._invoke("discord-1", channel_key="999")
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", True):
            on = self._invoke("discord-1", channel_key="999")
        self.assertEqual(off["configurable"]["thread_id"], "discord-1")
        self.assertEqual(on["configurable"]["thread_id"], "discord-1#999")
        # The namespace never branches by channel — memories are per person.
        self.assertEqual(on["metadata"]["user_id"], "discord-1")

    def test_identity_links_are_resolved(self):
        import fritz_utils
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "discord-1"}):
            config = self._invoke("web-a")
        self.assertEqual(config["metadata"]["user_id"], "discord-1")
        self.assertEqual(config["configurable"]["thread_id"], "discord-1")


if __name__ == "__main__":
    unittest.main()


class TestMemoryCapFillsTheBudget(unittest.TestCase):
    """The cap is a budget, not a tripwire.

    Breaking on the first entry that did not fit let one long summary block
    every shorter memory behind it, so a 4000-char budget routinely carried
    ~2000 chars of the available context.
    """

    def _state(self, msgs=None):
        return {"messages": msgs or [HumanMessage(content="q", id="h1")],
                "image_paths": [], "user_image_paths": []}

    def _inject(self, blob, cap):
        agent = _RecordingAgent()
        with patch.object(mister_fritz, "MEMORY_INJECT_MAX_CHARS", cap):
            _run_executor(self._state(), agent,
                          search_memories_internal=lambda *a, **k: blob)
        prompt = agent.seen["messages"][0][1]
        if "What I know about this user:" not in prompt:
            return ""
        return prompt.split("What I know about this user:")[1].lstrip()

    def test_a_long_entry_does_not_block_the_shorter_ones_behind_it(self):
        blob = json.dumps({
            "huge": "v" * 3000,      # fits alone, but not with others
            "small_a": "a" * 100,
            "small_b": "b" * 100,
        })
        injected = self._inject(blob, 1000)
        # The oversized head is skipped; the ones that fit are kept.
        self.assertIn("small_a", injected)
        self.assertIn("small_b", injected)

    def test_injected_block_never_exceeds_the_cap(self):
        blob = json.dumps({f"k{i}": "v" * 300 for i in range(30)})
        for cap in (500, 1500, 4000):
            with self.subTest(cap=cap):
                self.assertLessEqual(len(self._inject(blob, cap)), cap)

    def test_a_single_oversized_memory_is_cut_not_dropped(self):
        blob = json.dumps({"only_one": "v" * 10000})
        injected = self._inject(blob, 4000)
        self.assertNotEqual(injected.strip(), "{}")
        self.assertEqual(len(injected), 4000)


class TestMemoryKeyIsMadeOfContentWords(unittest.TestCase):
    """The key is a Chroma metadata key AND part of the embedded document, so
    a slug of pure grammar makes every memory look alike."""

    def test_stopwords_are_filtered_out(self):
        key = mister_fritz._make_memory_key(
            "The user said that they would like to discuss the pie incident")
        for stop in ("the", "that", "they", "would", "to"):
            self.assertNotIn(f"_{stop}_", key)
        self.assertIn("user", key)
        self.assertIn("pie", key)

    def test_an_all_stopword_summary_still_yields_something(self):
        key = mister_fritz._make_memory_key("the and of to it")
        self.assertTrue(key.startswith("memory_of_"))
        self.assertGreater(len(key), len("memory_of_"))

    def test_key_stays_within_the_length_cap(self):
        key = mister_fritz._make_memory_key("supercalifragilistic " * 30)
        self.assertLessEqual(len(key), mister_fritz._MEMORY_KEY_MAX)

    def test_timestamp_preamble_is_skipped(self):
        key = mister_fritz._make_memory_key(
            "Summary made at 2026-01-01T00:00:00 \r\n Nick prefers oxblood")
        self.assertNotIn("2026", key)
        self.assertIn("oxblood", key)


class TestProfileSignalsWireSchema(unittest.TestCase):
    """Two contracts on purpose: the wire schema requires all four fields so
    the model cannot answer {} and produce no profile update, while the Python
    defaults keep "absent" meaning "no evidence"."""

    def test_every_field_is_required_in_the_json_schema(self):
        schema = mister_fritz.ProfileSignals.model_json_schema()
        self.assertEqual(sorted(schema["required"]),
                         ["communication_style", "dislikes", "interests", "notes"])

    def test_python_defaults_are_unchanged(self):
        s = mister_fritz.ProfileSignals()
        self.assertEqual((s.communication_style, s.interests, s.dislikes, s.notes),
                         ("", [], [], ""))

