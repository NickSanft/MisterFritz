"""
Tests for the conversation tools defined in mister_fritz.py.

The module has unavoidable side-effects at import time (creates SQLite DB,
chroma directory, and attempts to write a Mermaid PNG). The PNG write is
wrapped in a try/except in the source, so it won't fail. The DB and chroma
directory creation are lightweight and acceptable in a test environment.
"""
import re
import sys
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Import mister_fritz with mocked heavy dependencies so unit tests do not
# require a live Ollama instance.
# ---------------------------------------------------------------------------

def _ensure_mock(module_name: str, mock: MagicMock = None) -> MagicMock:
    """Place mock in sys.modules if not already imported."""
    if module_name not in sys.modules:
        m = mock or MagicMock()
        sys.modules[module_name] = m
        return m
    return sys.modules[module_name]


# image_generator and document_engine have no state we care about in unit tests
_ensure_mock("image_generator")
_ensure_mock("document_engine")

# Import the functions we want to test AFTER ensuring mocks are in place.
# (chroma_store, langchain_ollama, and langchain.agents are real packages in
# the venv; they don't connect to Ollama at import / __init__ time.)
from mister_fritz import (  # noqa: E402  (import after sys.modules setup)
    get_current_time_internal,
    format_prompt,
    get_source_info,
    scrape_web,
    search_web,
    roll_dice,
)
from fritz_utils import MessageSource  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetCurrentTime(unittest.TestCase):
    def test_returns_rfc3339_string(self):
        result = get_current_time_internal()
        # RFC3339: YYYY-MM-DDTHH:MM:SS…±HH:MM or Z
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_contains_timezone_offset(self):
        result = get_current_time_internal()
        # Central time offset present (+ or -)
        self.assertTrue("+" in result or "-" in result)

    def test_returns_string(self):
        self.assertIsInstance(get_current_time_internal(), str)


class TestFormatPromptAndSource(unittest.TestCase):
    def test_format_prompt_contains_question(self):
        result = format_prompt("What is 2+2?", MessageSource.DISCORD_TEXT, "user1")
        self.assertIn("What is 2+2?", result)

    def test_format_prompt_contains_source_info(self):
        result = format_prompt("hello", MessageSource.DISCORD_TEXT, "user1")
        self.assertIn("user1", result)

    def test_get_source_info_discord_text(self):
        result = get_source_info(MessageSource.DISCORD_TEXT, "alice")
        self.assertIn("alice", result)
        self.assertIn("Discord", result)

    def test_get_source_info_discord_voice(self):
        result = get_source_info(MessageSource.DISCORD_VOICE, "bob")
        self.assertIn("30 words", result)

    def test_get_source_info_discord_image(self):
        result = get_source_info(MessageSource.DISCORD_TEXT_AND_IMAGE, "charlie")
        self.assertIn("image", result.lower())

    def test_get_source_info_local(self):
        result = get_source_info(MessageSource.LOCAL, "dave")
        self.assertIn("CLI", result)


class TestScrapeWebTool(unittest.TestCase):
    def _mock_response(self, html: str):
        resp = MagicMock()
        resp.text = html
        resp.raise_for_status = MagicMock()
        return resp

    def test_extracts_visible_text(self):
        html = "<html><body><p>Hello world</p></body></html>"
        with patch("mister_fritz.requests.get", return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertIn("Hello world", result)

    def test_strips_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Clean</p></body></html>"
        with patch("mister_fritz.requests.get", return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertNotIn("alert", result)
        self.assertIn("Clean", result)

    def test_returns_error_string_on_exception(self):
        with patch("mister_fritz.requests.get", side_effect=ConnectionError("failed")):
            result = scrape_web.invoke({"url": "http://bad-host.invalid"})
        self.assertIn("Error", result)

    def test_passes_timeout_to_requests(self):
        with patch("mister_fritz.requests.get",
                   return_value=self._mock_response("<p>ok</p>")) as mock_get:
            scrape_web.invoke({"url": "http://example.com"})
        _, kwargs = mock_get.call_args
        self.assertIn("timeout", kwargs)
        self.assertEqual(kwargs["timeout"], 10)


class TestSearchWebTool(unittest.TestCase):
    def test_returns_list_of_results(self):
        fake_results = [{"title": "Test", "body": "Test body", "href": "http://test.com"}]
        with patch("mister_fritz.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = fake_results
            result = search_web.invoke({"text_to_search": "python testing"})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_returns_empty_list_on_no_results(self):
        with patch("mister_fritz.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = []
            result = search_web.invoke({"text_to_search": "xyzzy_no_results"})
        self.assertEqual(result, [])


class TestRollDiceTool(unittest.TestCase):
    def _invoke(self, num_dice: int, num_sides: int):
        config = {"configurable": {}, "metadata": {"user_id": "testplayer"}}
        return roll_dice.invoke({"num_dice": num_dice, "num_sides": num_sides}, config=config)

    def test_result_is_string(self):
        self.assertIsInstance(self._invoke(1, 6), str)

    def test_contains_user_id(self):
        result = self._invoke(1, 6)
        self.assertIn("testplayer", result)

    def test_single_die_value_in_valid_range(self):
        result = self._invoke(1, 6)
        # Extract numbers from the list in the string e.g. "[4]"
        numbers = re.findall(r"\d+", result.split("[")[-1].replace("]", ""))
        for n in numbers:
            self.assertGreaterEqual(int(n), 1)
            self.assertLessEqual(int(n), 6)

    def test_multiple_dice_produces_multiple_values(self):
        result = self._invoke(3, 6)
        # The list portion should contain 3 numbers
        list_match = re.search(r"\[([^\]]+)\]", result)
        self.assertIsNotNone(list_match)
        values = [v.strip() for v in list_match.group(1).split(",")]
        self.assertEqual(len(values), 3)

    def test_zero_dice_raises(self):
        with self.assertRaises(Exception):
            self._invoke(0, 6)

    def test_zero_sides_raises(self):
        with self.assertRaises(Exception):
            self._invoke(2, 0)


if __name__ == "__main__":
    unittest.main()
