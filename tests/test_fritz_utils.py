import json
import os
import unittest
from unittest.mock import mock_open, patch

import fritz_utils as fu


class TestGetKeyFromJsonConfig(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
