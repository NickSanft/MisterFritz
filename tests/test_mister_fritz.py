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

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

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
    """Stands in for the compiled ReAct agent; captures the inputs dict."""

    def __init__(self, reply="Very well, sir."):
        self.seen = None
        self._reply = reply

    def stream(self, inputs, config=None, stream_mode=None):
        self.seen = inputs
        yield {"messages": [AIMessage(content=self._reply)]}


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


if __name__ == "__main__":
    unittest.main()
