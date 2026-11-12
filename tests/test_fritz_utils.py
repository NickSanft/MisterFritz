import contextlib
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
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

    def test_without_discord_it_still_names_every_non_canonical_entry(self):
        """require_discord=False drops the token requirement and nothing else:
        main_local.py runs with it, and the same admin config applies there."""
        with patch.object(fu, "DISCORD_BOT_TOKEN", None), \
             patch.object(fu, "ROOT_USER", "divora"), \
             patch.object(fu, "ADMIN_USERS", frozenset({"someone_else"})), \
             patch.object(fu, "ADMIN_LEGACY_NAME_MATCH", False):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config(require_discord=False)
        blob = " ".join(logs.output)
        self.assertIn("divora", blob)
        self.assertIn("someone_else", blob)

    def test_without_discord_a_missing_token_is_not_an_error(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", None), \
             patch.object(fu, "ROOT_USER", "discord-1"):
            fu.validate_config(require_discord=False)
            with self.assertRaises(RuntimeError) as caught:
                fu.validate_config()
        self.assertIn("DISCORD_BOT_TOKEN", str(caught.exception))

    def test_a_fully_canonical_config_warns_about_nothing_without_discord_either(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", None), \
             patch.object(fu, "ROOT_USER", "discord-1"), \
             patch.object(fu, "ADMIN_USERS", frozenset({"discord-2"})), \
             patch.object(fu, "_CLAMPED_KNOBS", []):
            with self.assertNoLogs("fritz_utils", level="WARNING"):
                fu.validate_config(require_discord=False)

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
                     "TTS_MAX_CONCURRENCY", "RELAY_MAX_BODY_CHARS",
                     "RELAY_MAX_PER_SENDER_PER_HOUR",
                     "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR",
                     "RELAY_REPLY_WINDOW_MIN", "RELAY_RETENTION_DAYS"):
            self.assertGreaterEqual(getattr(fu, name), 1, name)

    def test_relay_counts_are_clamped_not_bare_ints(self):
        """A relay cap of 0 rejects every message and reads as a real limit.

        Checking the source rather than the values, because the values are
        identical either way until someone sets the env var to 0 - which is
        exactly the case a bare int() would sail straight through.
        """
        src = pathlib.Path(fu.__file__).read_text(encoding="utf-8")
        for name in ("RELAY_MAX_BODY_CHARS", "RELAY_MAX_PER_SENDER_PER_HOUR",
                     "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR",
                     "RELAY_REPLY_WINDOW_MIN", "RELAY_RETENTION_DAYS"):
            line = next(ln for ln in src.splitlines()
                        if ln.startswith(name + ":"))
            self.assertIn("_at_least_one", line, name)

    def test_validate_config_warns_when_the_agent_tool_knob_is_set(self):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"),              patch.object(fu, "ROOT_USER", "discord-1"),              patch.object(fu, "_CLAMPED_KNOBS", []),              patch.object(fu, "RELAY_ENABLED", False),              patch.object(fu, "RELAY_AGENT_TOOL_ENABLED", True):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config()
        self.assertTrue(any("RELAY_AGENT_TOOL_ENABLED" in m for m in logs.output))

    def test_validate_config_quiet_when_relay_knobs_are_coherent(self):
        # (True, True) is deliberately NOT here any more. It is the one
        # combination that reads as "the agent tool is on", and the tool does
        # not exist, so it warns — see the class below.
        for enabled, tool in ((True, False), (False, False)):
            with self.subTest(relay=enabled, tool=tool):
                with patch.object(fu, "DISCORD_BOT_TOKEN", "x"),                      patch.object(fu, "ROOT_USER", "discord-1"),                      patch.object(fu, "ADMIN_USERS", frozenset()),                      patch.object(fu, "_CLAMPED_KNOBS", []),                      patch.object(fu, "RELAY_ENABLED", enabled),                      patch.object(fu, "RELAY_AGENT_TOOL_ENABLED", tool):
                    with self.assertNoLogs("fritz_utils", level="WARNING"):
                        fu.validate_config()

    def test_validate_config_warns_about_a_clamped_knob(self):
        fu._CLAMPED_KNOBS.append(("IMAGE_GEN_MAX_CONCURRENCY", "0"))
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"), \
             patch.object(fu, "ROOT_USER", "discord-1"):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config()
        self.assertTrue(any("IMAGE_GEN_MAX_CONCURRENCY" in m for m in logs.output))


class TestAKnobThatGatesNothingSaysSo(unittest.TestCase):
    """RELAY_AGENT_TOOL_ENABLED is reserved for an agent-facing relay tool that
    is not built. Nothing reads it except validate_config.

    It used to warn only when RELAY_ENABLED was ALSO false, which meant the one
    configuration that looks like it works — both set to true — was
    accepted in silence, while .env.example described the feature as shipping
    and merely off. An operator who set it deliberately got no sign that
    nothing had happened.
    """

    def _validate(self, relay: bool, tool: bool):
        with patch.object(fu, "DISCORD_BOT_TOKEN", "x"), \
             patch.object(fu, "ROOT_USER", "discord-1"), \
             patch.object(fu, "ADMIN_USERS", frozenset()), \
             patch.object(fu, "_CLAMPED_KNOBS", []), \
             patch.object(fu, "RELAY_ENABLED", relay), \
             patch.object(fu, "RELAY_AGENT_TOOL_ENABLED", tool):
            with self.assertLogs("fritz_utils", level="WARNING") as logs:
                fu.validate_config()
        return "; ".join(logs.output)

    def test_it_warns_even_when_the_relay_itself_is_on(self):
        """The case that used to pass in silence."""
        said = self._validate(relay=True, tool=True)
        self.assertIn("RELAY_AGENT_TOOL_ENABLED", said)
        self.assertIn("does nothing", said)

    def test_it_still_warns_when_the_relay_is_off(self):
        said = self._validate(relay=False, tool=True)
        self.assertIn("RELAY_AGENT_TOOL_ENABLED", said)

    def test_it_mentions_the_relay_being_off_only_when_it_is(self):
        """Two different situations, and an operator debugging one should not
        be told about the other."""
        self.assertIn("relay itself is off", self._validate(relay=False, tool=True))
        self.assertNotIn("relay itself is off", self._validate(relay=True, tool=True))

    def test_the_warning_says_the_tool_does_not_exist(self):
        """Rather than only that the setting had no effect, which reads as a
        configuration mistake the operator could fix."""
        said = self._validate(relay=True, tool=True)
        for claim in ("not built", "reserved"):
            with self.subTest(claim=claim):
                self.assertIn(claim, said)

    def test_nothing_in_the_tree_reads_the_knob_except_this_check(self):
        """The warning's central claim. If a module starts reading it, the
        warning becomes the lie instead."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        readers = []
        for path in sorted(repo.glob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "RELAY_AGENT_TOOL_ENABLED" not in line or line.strip().startswith("#"):
                    continue
                readers.append(f"{path.name}:{number}")
        self.assertEqual(
            [r.split(":")[0] for r in readers], ["fritz_utils.py"] * len(readers),
            f"something outside fritz_utils reads the knob: {readers}. The "
            "startup warning says nothing does, so either it is wrong or the "
            "tool has landed and the warning should go.")

    def test_env_example_no_longer_advertises_it_as_shipping(self):
        import pathlib
        env = (pathlib.Path(__file__).resolve().parents[1] / ".env.example"
               ).read_text(encoding="utf-8")
        section = env[env.index("RELAY_AGENT_TOOL_ENABLED") - 600:
                      env.index("RELAY_AGENT_TOOL_ENABLED")]
        self.assertIn("does nothing today", section)
        self.assertNotIn("Ships off; turn on only", section)


class TestDesktopCompanionKnobs(unittest.TestCase):
    """Plan 14: six knobs, declared by P4 ahead of the code that reads them.

    Three refusals hold whatever is built, because each describes a
    configuration no build should allow. PR 1 made three knobs live; the rest
    follow the RELAY_AGENT_TOOL_ENABLED convention: a knob nothing reads says
    so, and a scan of the tree keeps that true, as another keeps the live ones
    read.
    """

    LIVE = ("COMPANION_ENABLED", "COMPANION_TOKEN", "COMPANION_MAX_CLIENTS")
    RESERVED = ("COMPANION_BIND", "COMPANION_TTS", "GOOGLE_MCP_ENABLED")
    TOKEN = "t" * 43

    def _patched(self, **knobs):
        values = {"DISCORD_BOT_TOKEN": "x", "ROOT_USER": "discord-1", "ADMIN_USERS": frozenset(),
                  "_CLAMPED_KNOBS": [], "RELAY_AGENT_TOOL_ENABLED": False, "ADMIN_PANEL_PASSWORD": "pw",
                  "COMPANION_ENABLED": False, "COMPANION_TOKEN": None, "COMPANION_BIND": "127.0.0.1",
                  "COMPANION_MAX_CLIENTS": 2, "COMPANION_TTS": "local", "GOOGLE_MCP_ENABLED": False}
        values.update(knobs)
        stack = contextlib.ExitStack()
        for name, value in values.items():
            stack.enter_context(patch.object(fu, name, value))
        return stack

    def _warnings(self, **knobs) -> str:
        with self._patched(**knobs), self.assertLogs("fritz_utils", level="WARNING") as logs:
            fu.validate_config()
        return "; ".join(logs.output)

    def _logged(self, **knobs) -> str:
        """Like _warnings, but nothing logged is an answer too: ""."""
        lines = []

        class _Collect(logging.Handler):
            def emit(self, record):
                lines.append(record.getMessage())

        log = logging.getLogger("fritz_utils")
        handler, level = _Collect(logging.WARNING), log.level
        log.addHandler(handler)
        log.setLevel(logging.WARNING)
        try:
            with self._patched(**knobs):
                fu.validate_config()
        finally:
            log.removeHandler(handler)
            log.setLevel(level)
        return "; ".join(lines)

    def _refusal(self, **knobs) -> str:
        with self._patched(**knobs), self.assertRaises(RuntimeError) as caught:
            fu.validate_config()
        return str(caught.exception)

    def test_the_shipped_defaults(self):
        """Read from source, so a developer's own .env cannot decide it."""
        src = pathlib.Path(fu.__file__).read_text(encoding="utf-8")
        for expected in ('os.environ.get("COMPANION_ENABLED", "false")',
                         'os.environ.get("COMPANION_TOKEN") or None',
                         'os.environ.get("COMPANION_BIND", "127.0.0.1")',
                         '_at_least_one("COMPANION_MAX_CLIENTS", "2")',
                         'os.environ.get("COMPANION_TTS", "local")',
                         'os.environ.get("GOOGLE_MCP_ENABLED", "false")'):
            with self.subTest(expected=expected):
                self.assertIn(expected, src)

    def test_the_defaults_start_cleanly_and_quietly(self):
        with self._patched(), self.assertNoLogs("fritz_utils", level="WARNING"):
            fu.validate_config()

    # Every spelling here binds every interface: asyncio resolves the host with
    # getaddrinfo, and glibc (the Docker image) reads "0" and "0x0" as 0.0.0.0.
    WILDCARDS = ("0.0.0.0", " 0.0.0.0 ", "::", "[::]", "", "::0", "[::0]", "0::",
                 "0:0:0:0:0:0:0:0", "::0.0.0.0", "::%1", "0", "0x0", "000.000.000.000")

    def test_a_wildcard_bind_is_refused_enabled_or_not(self):
        """The token is the only gate on /companion; every interface is too
        many. Decided by the address, not the spelling."""
        for bind in self.WILDCARDS:
            for enabled in (False, True):
                with self.subTest(bind=bind, enabled=enabled):
                    said = self._refusal(COMPANION_BIND=bind, COMPANION_ENABLED=enabled,
                                         COMPANION_TOKEN=self.TOKEN)
                    self.assertIn("COMPANION_BIND", said)
                    self.assertIn("every network interface", said)

    def test_a_named_address_or_host_is_allowed(self):
        """Phase 5's case is a Tailscale address."""
        for bind in ("100.64.0.7", "::1", "[::1]", "localhost", "fritz-pc.tailnet.ts.net"):
            with self.subTest(bind=bind):
                said = self._warnings(COMPANION_BIND=bind)
                self.assertIn("COMPANION_BIND", said)   # reserved, so it warns; it does not refuse

    def test_an_enabled_companion_needs_a_token(self):
        said = self._refusal(COMPANION_ENABLED=True, COMPANION_TOKEN=None)
        self.assertIn("COMPANION_TOKEN is not set", said)
        self.assertIn("token_urlsafe", said)

    def test_the_live_knobs_set_well_start_quietly(self):
        self.assertEqual(self._logged(COMPANION_ENABLED=True, COMPANION_TOKEN=self.TOKEN,
                                      COMPANION_MAX_CLIENTS=3), "")

    def test_an_enabled_companion_with_no_panel_to_serve_it_says_so(self):
        """main_discord starts without ADMIN_PANEL_PASSWORD, panel and all
        left off, and /companion with it."""
        said = self._warnings(COMPANION_ENABLED=True, COMPANION_TOKEN=self.TOKEN,
                              ADMIN_PANEL_PASSWORD=None)
        self.assertIn("ADMIN_PANEL_PASSWORD is not set", said)
        self.assertIn("/companion", said)
        self.assertEqual(self._logged(COMPANION_ENABLED=False, ADMIN_PANEL_PASSWORD=None), "")

    def test_an_unknown_tts_mode_is_refused(self):
        # Values as they arrive after parsing, which strips and lowercases.
        for mode in ("kokoro", "piper", "none"):
            with self.subTest(mode=mode):
                self.assertIn("COMPANION_TTS", self._refusal(COMPANION_TTS=mode))
        for mode in ("local", "xtts", "off"):
            with self.subTest(mode=mode), self._patched(COMPANION_TTS=mode):
                fu.validate_config()          # accepted: no refusal

    def test_a_short_token_is_named_and_a_long_one_is_not(self):
        self.assertIn("COMPANION_TOKEN is 8 characters", self._warnings(COMPANION_TOKEN="abcdefgh"))
        self.assertIn("COMPANION_TOKEN is 31 characters", self._warnings(COMPANION_TOKEN="t" * 31))
        self.assertNotIn("characters", self._logged(COMPANION_TOKEN="t" * 32))
        self.assertNotIn("characters", self._logged(COMPANION_TOKEN=self.TOKEN))

    def test_values_are_parsed_as_the_environment_gives_them(self):
        """Every other test patches the parsed values, so this one reads a real
        environment in a fresh interpreter: whitespace and case are forgiven,
        and a padded wildcard is still refused."""
        repo = pathlib.Path(__file__).resolve().parents[1]
        probe = ("import fritz_utils as f, json\n"
                 "out = {'tts': f.COMPANION_TTS, 'bind': f.COMPANION_BIND}\n"
                 "try:\n    f.validate_config(); out['started'] = True\n"
                 "except RuntimeError as e:\n    out['started'] = str(e)\n"
                 "print(json.dumps(out))")

        def parse(**env):
            full = {**os.environ, "DISCORD_BOT_TOKEN": "x", "ROOT_USER": "discord-1", **env}
            proc = subprocess.run([sys.executable, "-c", probe], cwd=repo, env=full,
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            return json.loads(proc.stdout.strip().splitlines()[-1])

        out = parse(COMPANION_TTS=" XTTS ", COMPANION_BIND=" 127.0.0.1 ")
        self.assertEqual((out["tts"], out["bind"], out["started"]), ("xtts", "127.0.0.1", True))
        out = parse(COMPANION_BIND=" 0.0.0.0 ")
        self.assertIn("every network interface", out["started"])

    def test_every_reserved_knob_says_it_does_nothing_when_set(self):
        for name, value, why in (("COMPANION_BIND", "100.64.0.7", "Phase 5"),
                                 ("COMPANION_TTS", "xtts", "not built")):
            with self.subTest(knob=name):
                said = self._warnings(**{name: value})
                self.assertIn(f"{name}={value} does nothing yet", said)
                self.assertIn("no code reads this setting", said)
                self.assertIn(why, said)

    def test_a_reserved_bind_does_not_move_the_panel(self):
        """What COMPANION_BIND's warning promises: /companion stays on
        loopback with the panel, whatever the knob says."""
        import admin_panel
        with self._patched(COMPANION_BIND="100.64.0.7"):
            self.assertEqual(admin_panel.server_config(app=None, port=0).host, "127.0.0.1")

    def test_google_says_nothing_is_connected(self):
        said = self._warnings(GOOGLE_MCP_ENABLED=True)
        self.assertIn("GOOGLE_MCP_ENABLED", said)
        self.assertIn("Nothing is connected", said)

    @staticmethod
    def _readers(knob):
        """Every line outside fritz_utils that names `knob` other than as a
        comment."""
        repo = pathlib.Path(__file__).resolve().parents[1]
        return [f"{path.name}:{number}"
                for path in sorted(repo.glob("*.py")) if path.name != "fritz_utils.py"
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if knob in line and not line.strip().startswith("#")]

    def test_nothing_in_the_tree_reads_them_yet(self):
        """The warnings' central claim. The PR that starts reading one of
        these must retire its warning too, or the warning becomes the lie."""
        for knob in self.RESERVED:
            with self.subTest(knob=knob):
                self.assertEqual(self._readers(knob), [], f"{knob} is read outside fritz_utils")

    def test_the_live_ones_are_read(self):
        """The other side of it: a knob no longer warned about must do
        something. Read by code, as `fritz_utils.KNOB` at call time, not
        merely named in a docstring or a comment. That it is read where it
        matters is test_companion_api's to show, by behaviour (a log line
        reading it would satisfy this scan)."""
        import ast
        repo = pathlib.Path(__file__).resolve().parents[1]
        for knob in self.LIVE:
            readers = [path.name for path in sorted(repo.glob("*.py")) if path.name != "fritz_utils.py"
                       for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                       if isinstance(node, ast.Attribute) and node.attr == knob
                       and isinstance(node.value, ast.Name) and node.value.id == "fritz_utils"
                       and isinstance(node.ctx, ast.Load)]
            with self.subTest(knob=knob):
                self.assertIn("companion_api.py" if knob != "COMPANION_ENABLED" else "admin_panel.py",
                              readers, f"no code reads fritz_utils.{knob}")

    @staticmethod
    def _env_example_blocks() -> dict[str, str]:
        """.env.example's companion section: each knob's own comment, the
        lines between the previous knob (or the header's bare "#") and its
        `# KNOB=` line."""
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / ".env.example").read_text(encoding="utf-8")
        lines = env[env.index("Desktop companion and Google"):].splitlines()[1:]
        blocks, current = {}, []
        for line in lines:
            knob = re.match(r"# ([A-Z_]+)=", line)
            if knob:
                blocks[knob.group(1)], current = " ".join(current), []
            elif line.strip() == "#":
                current = []
            elif line.startswith("#"):
                current.append(line.lstrip("# "))
            else:
                break
        return blocks

    def test_the_documentation_says_which_are_reserved(self):
        blocks = self._env_example_blocks()
        self.assertEqual(sorted(blocks), sorted(self.LIVE + self.RESERVED))
        # Its header says "the first three are live".
        self.assertEqual(list(blocks)[:len(self.LIVE)], list(self.LIVE))
        # Decision 32: the token is its own secret.
        self.assertIn("deliberately separate from", blocks["COMPANION_TOKEN"])
        for knob in self.LIVE + self.RESERVED:
            with self.subTest(env=knob):
                if knob in self.RESERVED:
                    self.assertIn("RESERVED", blocks[knob])
                else:
                    self.assertNotIn("RESERVED", blocks[knob])
        repo = pathlib.Path(__file__).resolve().parents[1]
        readme = (repo / "README.md").read_text(encoding="utf-8")
        for knob in self.RESERVED + self.LIVE:
            row = next(line for line in readme.splitlines() if line.startswith(f"| `{knob}` |"))
            with self.subTest(readme=knob):
                if knob in self.RESERVED:
                    self.assertIn("Reserved", row)
                else:
                    self.assertNotIn("Reserved", row)
        # The section's own sentence counts the live ones off the top of the table.
        section = readme[readme.index("### Desktop companion and Google"):]
        section = section[:section.index("\n### ", 1)]
        self.assertIn(f"The first {len(self.LIVE)} are live", section.replace("three", "3"))
        rows = re.findall(r"^\| `([A-Z_]+)` \|", section, flags=re.M)
        self.assertEqual(rows[:len(self.LIVE)], list(self.LIVE))
        self.assertEqual(sorted(rows[len(self.LIVE):]), sorted(self.RESERVED))
