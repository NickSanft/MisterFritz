"""
Tests for the LangGraph planner node in mister_fritz.py.

mister_fritz initialises ChatOllama clients, SqliteSaver, and writes a Mermaid
diagram at import time. We stub heavy modules (image_generator, document_engine)
before importing, but ChatOllama itself is real — we patch the instance after
import to control what invoke() returns.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch


def _ensure_mock(name: str) -> MagicMock:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# ddgs (DuckDuckGo search) and image_generator/document_engine all initialise
# heavy resources at import time. Stub them so this test doesn't require the
# full runtime environment.
_ensure_mock("ddgs")
_ensure_mock("image_generator")
_ensure_mock("document_engine")

from langchain_core.messages import HumanMessage  # noqa: E402

import mister_fritz  # noqa: E402
from mister_fritz import planner  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
