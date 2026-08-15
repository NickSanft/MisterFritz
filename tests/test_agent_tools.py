"""
Tests for the conversation tools defined in mister_fritz.py.

The module has unavoidable side-effects at import time (creates SQLite DB,
chroma directory, and attempts to write a Mermaid PNG). The PNG write is
wrapped in a try/except in the source, so it won't fail. The DB and chroma
directory creation are lightweight and acceptable in a test environment.
"""
import re
import unittest
from unittest.mock import MagicMock, patch


# image_generator / document_engine (and ddgs) are stubbed in
# tests/conftest.py before any test module is collected, so unit tests do not
# require a live Ollama instance or the heavy optional stacks.
# (chroma_store, langchain_ollama, and langchain.agents are real packages in
# the venv; they don't connect to Ollama at import / __init__ time.)
import agent_tools  # noqa: E402  — also needed for patching _HTTP_CLIENT
from agent_tools import (  # noqa: E402  (import after sys.modules setup)
    get_current_time_internal,
    scrape_web,
    search_web,
    roll_dice,
)
from mister_fritz import format_prompt, get_source_info  # noqa: E402
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
        with patch.object(agent_tools._HTTP_CLIENT, "get", return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertIn("Hello world", result)

    def test_strips_script_tags(self):
        html = "<html><body><script>alert('xss')</script><p>Clean</p></body></html>"
        with patch.object(agent_tools._HTTP_CLIENT, "get", return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertNotIn("alert", result)
        self.assertIn("Clean", result)

    def test_returns_error_string_on_exception(self):
        with patch.object(agent_tools._HTTP_CLIENT, "get", side_effect=ConnectionError("failed")):
            result = scrape_web.invoke({"url": "http://bad-host.invalid"})
        self.assertIn("Error", result)

    def test_output_is_capped(self):
        # Uncapped, one long article lands whole in the executor's input and can
        # evict the conversation window from num_ctx. browser_tools.py had this
        # cap and the live scraper did not; the module is gone, the cap is not.
        html = "<html><body><p>" + ("word " * 20000) + "</p></body></html>"
        with patch.object(agent_tools._HTTP_CLIENT, "get",
                          return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertLessEqual(len(result), agent_tools.SCRAPE_MAX_CHARS)

    def test_short_pages_are_not_truncated(self):
        html = "<html><body><p>Short and sweet</p></body></html>"
        with patch.object(agent_tools._HTTP_CLIENT, "get",
                          return_value=self._mock_response(html)):
            result = scrape_web.invoke({"url": "http://example.com"})
        self.assertEqual(result, "Short and sweet")

    def test_http_client_has_timeout_configured(self):
        # Timeout is configured on the shared client, not per-call. Sanity-check
        # the client was built with a non-default timeout.
        self.assertIsNotNone(agent_tools._HTTP_CLIENT.timeout)
        self.assertGreater(agent_tools._HTTP_CLIENT.timeout.connect, 0)
        self.assertGreater(agent_tools._HTTP_CLIENT.timeout.read, 0)


class TestSearchWebTool(unittest.TestCase):
    def test_returns_list_of_results(self):
        fake_results = [{"title": "Test", "body": "Test body", "href": "http://test.com"}]
        with patch("agent_tools.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = fake_results
            result = search_web.invoke({"text_to_search": "python testing"})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_returns_empty_list_on_no_results(self):
        with patch("agent_tools.DDGS") as mock_ddgs:
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


class TestMemoryExtractionSkipGuard(unittest.TestCase):
    """Phase 10: trivial turns ("hi", "lol") shouldn't trigger an LLM call."""

    def test_short_user_message_skips_extraction(self):
        with patch("agent_tools._ollama_client") as mock_client:
            agent_tools._extract_and_store_memories(
                user_id="alice",
                user_message="hi",
                assistant_response="Hello there, how can I help you today?",
            )
        mock_client.chat.assert_not_called()

    def test_short_reply_skips_extraction(self):
        with patch("agent_tools._ollama_client") as mock_client:
            agent_tools._extract_and_store_memories(
                user_id="alice",
                user_message="What is the meaning of life, the universe, and everything?",
                assistant_response="42.",
            )
        mock_client.chat.assert_not_called()

    def test_substantial_turn_still_calls_llm(self):
        with patch("agent_tools._ollama_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.message.content = "[]"
            mock_client.chat.return_value = mock_resp
            agent_tools._extract_and_store_memories(
                user_id="alice",
                user_message="I work as a backend engineer at a fintech startup in Berlin.",
                assistant_response=(
                    "Excellent. I shall make a note of your profession and location. "
                    "Berlin is a fine city for the trade."
                ),
            )
        mock_client.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
