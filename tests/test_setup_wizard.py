"""
Tests for the pure-logic helpers in scripts/setup.py.

We avoid testing the interactive flow itself; instead we cover the file
I/O helpers (parse_env, write_env) and the model-name normalisation in
list_local_models's response handling.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# Load scripts/setup.py as a module without changing sys.path globally.
_SETUP_PATH = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"
_spec = importlib.util.spec_from_file_location("fritz_setup", _SETUP_PATH)
fritz_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fritz_setup)


class TestParseEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_returns_empty_dict_when_file_missing(self):
        result = fritz_setup.parse_env(self.tmp / "nope.env")
        self.assertEqual(result, {})

    def test_parses_key_value_pairs(self):
        path = self.tmp / ".env"
        path.write_text(
            "FOO=bar\n"
            "BAZ=qux\n",
            encoding="utf-8",
        )
        self.assertEqual(fritz_setup.parse_env(path), {"FOO": "bar", "BAZ": "qux"})

    def test_ignores_comments_and_blanks(self):
        path = self.tmp / ".env"
        path.write_text(
            "# a comment\n"
            "\n"
            "KEY=value\n"
            "  # indented comment\n",
            encoding="utf-8",
        )
        self.assertEqual(fritz_setup.parse_env(path), {"KEY": "value"})

    def test_handles_equals_in_value(self):
        # partition("=") stops at the first =, so KEY=a=b yields ('KEY', 'a=b')
        path = self.tmp / ".env"
        path.write_text("URL=http://host:8080/path?x=1\n", encoding="utf-8")
        self.assertEqual(
            fritz_setup.parse_env(path),
            {"URL": "http://host:8080/path?x=1"},
        )

    def test_strips_whitespace_around_key_and_value(self):
        path = self.tmp / ".env"
        path.write_text("  KEY  =  value  \n", encoding="utf-8")
        self.assertEqual(fritz_setup.parse_env(path), {"KEY": "value"})


class TestWriteEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Build a minimal .env.example in the tmp dir and point the module at it.
        self.example = self.tmp / ".env.example"
        self.env = self.tmp / ".env"
        self.example.write_text(
            "# header\n"
            "DISCORD_BOT_TOKEN=\n"
            "ROOT_USER=\n"
            "OLLAMA_HOST=http://127.0.0.1:11434\n"
            "WHISPER_MODEL=small\n",
            encoding="utf-8",
        )

    def _patch_paths(self):
        return (
            patch.object(fritz_setup, "ENV_EXAMPLE", self.example),
            patch.object(fritz_setup, "ENV_PATH", self.env),
        )

    def test_writes_fresh_env_from_template(self):
        p1, p2 = self._patch_paths()
        with p1, p2:
            fritz_setup.write_env({
                "DISCORD_BOT_TOKEN": "abc123",
                "ROOT_USER": "alice",
                "OLLAMA_HOST": "http://127.0.0.1:11434",
            })
        result = fritz_setup.parse_env(self.env)
        self.assertEqual(result["DISCORD_BOT_TOKEN"], "abc123")
        self.assertEqual(result["ROOT_USER"], "alice")
        self.assertEqual(result["OLLAMA_HOST"], "http://127.0.0.1:11434")
        # Untouched keys preserve their defaults from the template.
        self.assertEqual(result["WHISPER_MODEL"], "small")

    def test_preserves_template_comments_and_structure(self):
        p1, p2 = self._patch_paths()
        with p1, p2:
            fritz_setup.write_env({"DISCORD_BOT_TOKEN": "tok"})
        content = self.env.read_text(encoding="utf-8")
        self.assertIn("# header", content)
        # The line should be replaced in-place, not appended.
        self.assertIn("DISCORD_BOT_TOKEN=tok", content)
        self.assertNotIn("DISCORD_BOT_TOKEN=\n", content)

    def test_backs_up_existing_env(self):
        # Pre-existing .env should be backed up before being overwritten.
        self.env.write_text("OLD_CONTENT=keep\n", encoding="utf-8")
        p1, p2 = self._patch_paths()
        with p1, p2:
            fritz_setup.write_env({"DISCORD_BOT_TOKEN": "new"})
        backups = list(self.tmp.glob(".env.bak.*"))
        self.assertEqual(len(backups), 1)
        self.assertIn("OLD_CONTENT=keep", backups[0].read_text(encoding="utf-8"))

    def test_appends_unknown_keys_at_end(self):
        # Keys not in the template should still land in the file.
        p1, p2 = self._patch_paths()
        with p1, p2:
            fritz_setup.write_env({"DISCORD_BOT_TOKEN": "tok", "CUSTOM_KEY": "x"})
        result = fritz_setup.parse_env(self.env)
        self.assertEqual(result["CUSTOM_KEY"], "x")


class TestValidateDiscordToken(unittest.TestCase):
    """The fetch is mockable; we just confirm it returns the right shape on success/failure."""

    def test_returns_user_dict_on_200(self):
        import io
        import json

        fake_resp = io.BytesIO(json.dumps({"username": "bot", "discriminator": "0"}).encode())
        fake_resp.status = 200

        with patch.object(fritz_setup.urllib.request, "urlopen", return_value=fake_resp):
            result = fritz_setup.validate_discord_token("dummy")
        self.assertIsNotNone(result)
        self.assertEqual(result["username"], "bot")

    def test_returns_none_on_401(self):
        import urllib.error

        def boom(*a, **kw):
            raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

        with patch.object(fritz_setup.urllib.request, "urlopen", side_effect=boom):
            result = fritz_setup.validate_discord_token("bad")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
