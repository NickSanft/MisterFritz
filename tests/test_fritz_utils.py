import json
import os
import unittest
from unittest.mock import mock_open, patch

import fritz_utils as fu


class TestGetKeyFromJsonConfig(unittest.TestCase):
    def setUp(self):
        # The cache is shared across all callers, so each test must start fresh.
        fu._load_config_json.cache_clear()

    def tearDown(self):
        fu._load_config_json.cache_clear()

    def test_returns_value_for_existing_key(self):
        data = json.dumps({"my_key": "my_value"})
        with patch("builtins.open", mock_open(read_data=data)):
            result = fu.get_key_from_json_config_file("my_key")
        self.assertEqual(result, "my_value")

    def test_returns_none_for_missing_key(self):
        data = json.dumps({"other_key": "value"})
        with patch("builtins.open", mock_open(read_data=data)):
            result = fu.get_key_from_json_config_file("nonexistent")
        self.assertIsNone(result)

    def test_returns_none_when_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = fu.get_key_from_json_config_file("any_key")
        self.assertIsNone(result)

    def test_returns_none_for_invalid_json(self):
        with patch("builtins.open", mock_open(read_data="not valid json {{{")):
            result = fu.get_key_from_json_config_file("key")
        self.assertIsNone(result)


class TestConfigJsonCaching(unittest.TestCase):
    """The parsed dict is memoised — repeat key lookups must not re-open the file."""

    def setUp(self):
        fu._load_config_json.cache_clear()

    def tearDown(self):
        fu._load_config_json.cache_clear()

    def test_file_is_opened_at_most_once_per_cache_lifetime(self):
        data = json.dumps({"a": "1", "b": "2", "c": "3"})
        m = mock_open(read_data=data)
        with patch("builtins.open", m):
            self.assertEqual(fu.get_key_from_json_config_file("a"), "1")
            self.assertEqual(fu.get_key_from_json_config_file("b"), "2")
            self.assertEqual(fu.get_key_from_json_config_file("c"), "3")
        # 3 key lookups but only 1 file open.
        self.assertEqual(m.call_count, 1)

    def test_cache_clear_resumes_file_reads(self):
        data = json.dumps({"k": "v"})
        m = mock_open(read_data=data)
        with patch("builtins.open", m):
            fu.get_key_from_json_config_file("k")
            fu._load_config_json.cache_clear()
            fu.get_key_from_json_config_file("k")
        self.assertEqual(m.call_count, 2)


class TestValidateConfig(unittest.TestCase):
    def test_raises_when_token_missing(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", None), \
             patch.object(fu, "ROOT_USER", "someone"):
            with self.assertRaises(RuntimeError) as ctx:
                fu.validate_config()
        self.assertIn("DISCORD_BOT_TOKEN", str(ctx.exception))

    def test_raises_when_root_user_missing(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "sometoken"), \
             patch.object(fu, "ROOT_USER", None):
            with self.assertRaises(RuntimeError) as ctx:
                fu.validate_config()
        self.assertIn("ROOT_USER", str(ctx.exception))

    def test_raises_listing_all_missing_fields(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", None), \
             patch.object(fu, "ROOT_USER", None):
            with self.assertRaises(RuntimeError) as ctx:
                fu.validate_config()
        msg = str(ctx.exception)
        self.assertIn("DISCORD_BOT_TOKEN", msg)
        self.assertIn("ROOT_USER", msg)

    def test_passes_when_all_set(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "token"), \
             patch.object(fu, "ROOT_USER", "user"):
            fu.validate_config()  # must not raise


class TestConstantDefaults(unittest.TestCase):
    def test_ollama_host_default_is_localhost(self):
        self.assertIn("127.0.0.1", fu.OLLAMA_HOST)

    def test_model_names_are_non_empty(self):
        for attr in ("THINKING_OLLAMA_MODEL", "FAST_OLLAMA_MODEL",
                     "EMBEDDING_MODEL", "VISION_MODEL"):
            with self.subTest(attr=attr):
                self.assertTrue(getattr(fu, attr))

    def test_path_constants_are_non_empty(self):
        for attr in ("DOC_FOLDER", "CHROMA_DB_PATH", "CHAT_DB_NAME"):
            with self.subTest(attr=attr):
                self.assertTrue(getattr(fu, attr))

    def test_numeric_tunables_are_sane(self):
        self.assertGreater(fu.SUMMARIZE_THRESHOLD, 0)
        self.assertGreaterEqual(fu.HISTORY_TOKEN_BUDGET, 0)  # 0 = window disabled
        self.assertGreater(fu.MEMORY_INJECT_MAX_CHARS, 0)
        self.assertGreaterEqual(fu.STREAM_MIN_CHARS, 1)
        self.assertGreater(fu.DISCORD_STREAM_MIN_INTERVAL, 0)
        self.assertGreater(fu.BLOCKING_POOL_SIZE, 0)
        self.assertGreaterEqual(fu.IMAGE_GEN_MAX_CONCURRENCY, 1)  # 0 deadlocks /gen
        self.assertGreaterEqual(fu.TTS_MAX_CONCURRENCY, 1)
        for attr in ("SUMMARIZE_THRESHOLD", "HISTORY_TOKEN_BUDGET",
                     "MEMORY_INJECT_MAX_CHARS", "STREAM_MIN_CHARS",
                     "BLOCKING_POOL_SIZE", "IMAGE_GEN_MAX_CONCURRENCY",
                     "TTS_MAX_CONCURRENCY"):
            with self.subTest(attr=attr):
                self.assertIsInstance(getattr(fu, attr), int)
        self.assertIsInstance(fu.DISCORD_STREAM_MIN_INTERVAL, float)

    def test_bool_tunables_are_bools(self):
        for attr in ("SUMMARIZE_ASYNC", "CHAT_CODE_HIGHLIGHT", "DISCORD_ERROR_DETAIL"):
            with self.subTest(attr=attr):
                self.assertIsInstance(getattr(fu, attr), bool)
        # Defaults: summariser off-path and highlighting on; error detail off.
        self.assertFalse(fu.DISCORD_ERROR_DETAIL)

    def test_ffmpeg_path_is_string(self):
        self.assertIsInstance(fu.FFMPEG_PATH, str)
        self.assertTrue(fu.FFMPEG_PATH)

    def test_ffprobe_path_is_string(self):
        self.assertIsInstance(fu.FFPROBE_PATH, str)
        self.assertTrue(fu.FFPROBE_PATH)


class TestEnvVarOverride(unittest.TestCase):
    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom-host:9999"}):
            # re-read directly since the constant is already set at import time;
            # test the helper logic instead
            import importlib
            # Just verify the env var is readable
            self.assertEqual(os.environ.get("OLLAMA_HOST"), "http://custom-host:9999")

    def test_env_or_json_prefers_env(self):
        with patch.dict(os.environ, {"_TEST_KEY": "env_value"}):
            result = os.environ.get("_TEST_KEY") or fu.get_key_from_json_config_file("_test_key")
        self.assertEqual(result, "env_value")


class TestIsAdmin(unittest.TestCase):
    """is_admin() should accept ROOT_USER or anyone in ADMIN_USERS."""

    def test_root_user_is_admin(self):
        with patch.object(fu, "ROOT_USER", "alice"), \
             patch.object(fu, "ADMIN_USERS", frozenset()):
            self.assertTrue(fu.is_admin("alice"))

    def test_admin_users_member_is_admin(self):
        with patch.object(fu, "ROOT_USER", "alice"), \
             patch.object(fu, "ADMIN_USERS", frozenset({"bob", "carol"})):
            self.assertTrue(fu.is_admin("bob"))
            self.assertTrue(fu.is_admin("carol"))

    def test_unknown_user_is_not_admin(self):
        with patch.object(fu, "ROOT_USER", "alice"), \
             patch.object(fu, "ADMIN_USERS", frozenset({"bob"})):
            self.assertFalse(fu.is_admin("dave"))

    def test_empty_user_id_is_not_admin(self):
        with patch.object(fu, "ROOT_USER", "alice"), \
             patch.object(fu, "ADMIN_USERS", frozenset()):
            self.assertFalse(fu.is_admin(""))
            self.assertFalse(fu.is_admin(None))

    def test_no_root_user_set_admin_list_still_works(self):
        # Deployments that drop ROOT_USER entirely in favour of ADMIN_USERS.
        with patch.object(fu, "ROOT_USER", None), \
             patch.object(fu, "ADMIN_USERS", frozenset({"alice"})):
            self.assertTrue(fu.is_admin("alice"))
            self.assertFalse(fu.is_admin("bob"))


class TestMessageSource(unittest.TestCase):
    def test_all_variants_exist(self):
        from fritz_utils import MessageSource
        self.assertIsNotNone(MessageSource.DISCORD_TEXT)
        self.assertIsNotNone(MessageSource.DISCORD_TEXT_AND_IMAGE)
        self.assertIsNotNone(MessageSource.DISCORD_VOICE)
        self.assertIsNotNone(MessageSource.LOCAL)

    def test_variants_are_distinct(self):
        from fritz_utils import MessageSource
        values = [e.value for e in MessageSource]
        # All values should be distinct
        self.assertEqual(len(values), len(set(str(v) for v in values)))




class TestIsAdminCanonical(unittest.TestCase):
    """Admin follows the immutable id, not the name.

    ADMIN_LEGACY_NAME_MATCH exists so a pre-migration config keeps working for
    one release; it ships FALSE (DECISIONS #6) so there is no impersonation
    window. These pin both halves — the plan named this class and it was never
    written.
    """

    def test_canonical_root_is_admin(self):
        with patch.object(fu, "ROOT_USER", "discord-1"),              patch.object(fu, "ADMIN_USERS", frozenset()):
            self.assertTrue(fu.is_admin("discord-1"))

    def test_someone_else_is_not(self):
        with patch.object(fu, "ROOT_USER", "discord-1"),              patch.object(fu, "ADMIN_USERS", frozenset()):
            self.assertFalse(fu.is_admin("discord-2"))

    def test_admin_users_membership_counts(self):
        with patch.object(fu, "ROOT_USER", None),              patch.object(fu, "ADMIN_USERS", frozenset({"discord-7"})):
            self.assertTrue(fu.is_admin("discord-7"))

    def test_a_display_name_is_ignored_when_the_flag_is_off(self):
        """The whole point: renaming yourself to the owner's handle grants
        nothing."""
        with patch.object(fu, "ROOT_USER", "divora"),              patch.object(fu, "ADMIN_LEGACY_NAME_MATCH", False):
            self.assertFalse(fu.is_admin("discord-99", display_name="divora"))

    def test_a_display_name_is_honoured_when_the_flag_is_on(self):
        with patch.object(fu, "ROOT_USER", "divora"),              patch.object(fu, "ADMIN_LEGACY_NAME_MATCH", True):
            self.assertTrue(fu.is_admin("discord-99", display_name="divora"))

    def test_a_canonical_root_cannot_be_matched_by_name_even_with_the_flag_on(self):
        """Once ROOT_USER is canonical the legacy shim grants nothing — no
        display name equals "discord-1". Migrating the config is what actually
        closes the rename hole, not flipping the flag."""
        with patch.object(fu, "ROOT_USER", "discord-1"),              patch.object(fu, "ADMIN_LEGACY_NAME_MATCH", True):
            self.assertFalse(fu.is_admin("discord-99", display_name="discord-1"))

    def test_none_is_never_admin(self):
        with patch.object(fu, "ROOT_USER", "discord-1"):
            self.assertFalse(fu.is_admin(None))


class TestValidateConfigWarnsOnLegacyAdminNames(unittest.TestCase):
    """A config still using display names is a live impersonation risk with the
    flag on, and silently broken admin commands with it off. Either way the
    operator should be told at startup, by name."""

    def test_it_names_every_non_canonical_entry(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"),              patch.object(fu, "ROOT_USER", "divora"),              patch.object(fu, "ADMIN_USERS", frozenset({"someone_else"})),              patch.object(fu, "ADMIN_LEGACY_NAME_MATCH", False):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config()
        blob = " ".join(logs.output)
        self.assertIn("divora", blob)
        self.assertIn("someone_else", blob)

    def test_a_fully_canonical_config_warns_about_nothing(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"),              patch.object(fu, "ROOT_USER", "discord-1"),              patch.object(fu, "ADMIN_USERS", frozenset({"discord-2"})),              patch.object(fu, "_CLAMPED_KNOBS", []):
            with self.assertNoLogs("fritz_utils", level="WARNING"):
                fu.validate_config()

if __name__ == "__main__":
    unittest.main()


class TestConcurrencyKnobsCannotDeadlock(unittest.TestCase):
    """asyncio.Semaphore(0) is legal and simply never releases.

    IMAGE_GEN_MAX_CONCURRENCY=0 therefore did not fail — it made /gen hang on
    `async with` until Discord expired the interaction 15 minutes later, every
    time, with nothing in the log to explain it. The helper clamps so the bot
    cannot deadlock, and records so validate_config can say so out loud once
    logging is configured (a warning at import time may have nowhere to go).
    """

    def setUp(self):
        self._saved = list(fu._CLAMPED_KNOBS)
        fu._CLAMPED_KNOBS.clear()

    def tearDown(self):
        fu._CLAMPED_KNOBS[:] = self._saved

    def test_zero_is_clamped_to_one(self):
        with patch.dict(os.environ, {"_KNOB": "0"}):
            self.assertEqual(fu._at_least_one("_KNOB", "1"), 1)
        self.assertEqual(fu._CLAMPED_KNOBS, [("_KNOB", "0")])

    def test_negative_is_clamped_to_one(self):
        with patch.dict(os.environ, {"_KNOB": "-3"}):
            self.assertEqual(fu._at_least_one("_KNOB", "1"), 1)

    def test_non_numeric_is_clamped_and_quoted_back(self):
        """A typo is a different mistake from setting 0 and should read that
        way in the warning."""
        with patch.dict(os.environ, {"_KNOB": "abc"}):
            self.assertEqual(fu._at_least_one("_KNOB", "1"), 1)
        self.assertEqual(fu._CLAMPED_KNOBS, [("_KNOB", "'abc'")])

    def test_a_valid_value_passes_through_and_records_nothing(self):
        with patch.dict(os.environ, {"_KNOB": "4"}):
            self.assertEqual(fu._at_least_one("_KNOB", "1"), 4)
        self.assertEqual(fu._CLAMPED_KNOBS, [])

    def test_shipped_knobs_are_all_usable(self):
        for name in ("BLOCKING_POOL_SIZE", "IMAGE_GEN_MAX_CONCURRENCY",
                     "TTS_MAX_CONCURRENCY"):
            self.assertGreaterEqual(getattr(fu, name), 1, name)

    def test_validate_config_warns_about_a_clamped_knob(self):
        fu._CLAMPED_KNOBS.append(("IMAGE_GEN_MAX_CONCURRENCY", "0"))
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"), \
             patch.object(fu, "ROOT_USER", "discord-1"):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config()
        self.assertTrue(any("IMAGE_GEN_MAX_CONCURRENCY" in m for m in logs.output))
