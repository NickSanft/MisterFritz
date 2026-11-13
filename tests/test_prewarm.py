"""
Tests for the Ollama pre-warm helper (Phase 12).

We mock the `ollama` module so the tests don't require a live Ollama
instance — what we care about is that prewarm_models passes the right
arguments and silently swallows failures.
"""
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch


def _install_fake_ollama() -> types.ModuleType:
    """Replace the real ollama module in sys.modules with a MagicMock-backed stub.

    ollama_recovery (which the pre-warm calls through) is imported first, so it
    and langchain_ollama bind the real module: imported under the stub, they
    would fail and every pre-warm with them."""
    import ollama_recovery  # noqa: F401
    fake = types.ModuleType("ollama")
    fake.chat = MagicMock()
    fake.embeddings = MagicMock()
    sys.modules["ollama"] = fake
    return fake


def _wait_for(thread: threading.Thread, timeout: float = 2.0) -> None:
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise AssertionError("Pre-warm thread did not complete in time")


class TestPrewarmModels(unittest.TestCase):
    def setUp(self):
        self._real_ollama = sys.modules.get("ollama")
        self.fake = _install_fake_ollama()
        # Import after sys.modules is patched so prewarm's `import ollama`
        # picks up the stub on first use.
        import importlib
        import prewarm
        importlib.reload(prewarm)
        self.prewarm = prewarm

    def tearDown(self):
        if self._real_ollama is not None:
            sys.modules["ollama"] = self._real_ollama
        else:
            sys.modules.pop("ollama", None)

    def test_chat_models_are_pre_warmed(self):
        t = self.prewarm.prewarm_models(
            chat_models=("alpha", "beta"),
            embedding_models=(),
            keep_alive="5m",
        )
        _wait_for(t)
        self.assertEqual(self.fake.chat.call_count, 2)
        models_called = [c.kwargs.get("model") for c in self.fake.chat.call_args_list]
        self.assertEqual(set(models_called), {"alpha", "beta"})

    def test_chat_prewarm_uses_single_token_and_passes_keep_alive(self):
        t = self.prewarm.prewarm_models(
            chat_models=("alpha",),
            embedding_models=(),
            keep_alive="-1",
        )
        _wait_for(t)
        call = self.fake.chat.call_args_list[0]
        self.assertEqual(call.kwargs["model"], "alpha")
        self.assertEqual(call.kwargs["keep_alive"], "-1")
        self.assertEqual(call.kwargs["options"], {"num_predict": 1})

    def test_embedding_models_use_embeddings_endpoint(self):
        t = self.prewarm.prewarm_models(
            chat_models=(),
            embedding_models=("mxbai-embed-large",),
            keep_alive="5m",
        )
        _wait_for(t)
        self.fake.embeddings.assert_called_once()
        kwargs = self.fake.embeddings.call_args.kwargs
        self.assertEqual(kwargs["model"], "mxbai-embed-large")
        self.assertEqual(kwargs["keep_alive"], "5m")

    def test_empty_model_names_are_filtered_out(self):
        # Passing None or "" for a model (e.g. unset VISION_MODEL) must not crash.
        t = self.prewarm.prewarm_models(
            chat_models=("alpha", "", None),
            embedding_models=("",),
            keep_alive="5m",
        )
        _wait_for(t)
        self.assertEqual(self.fake.chat.call_count, 1)
        self.fake.embeddings.assert_not_called()

    def test_chat_failures_are_swallowed(self):
        # A failing pre-warm must not propagate or crash subsequent ones.
        self.fake.chat.side_effect = [RuntimeError("ollama down"), None]
        t = self.prewarm.prewarm_models(
            chat_models=("alpha", "beta"),
            embedding_models=(),
            keep_alive="5m",
        )
        _wait_for(t)
        # Both models were attempted; the first failed, the second still went through.
        self.assertEqual(self.fake.chat.call_count, 2)

    def test_thread_is_daemon(self):
        t = self.prewarm.prewarm_models(
            chat_models=("alpha",),
            embedding_models=(),
            keep_alive="5m",
        )
        self.assertTrue(t.daemon, "Pre-warm thread must be a daemon so it doesn't block bot exit")
        _wait_for(t)


class TestKeepAliveConfig(unittest.TestCase):
    """Phase 12: OLLAMA_KEEP_ALIVE is plumbed end-to-end."""

    def test_default_is_five_minutes(self):
        # Reload fritz_utils in a clean env so the env override doesn't leak in
        # from outside.
        import importlib
        import os
        # And without the developer's .env, which reload would read again: the
        # default is the code's, whatever a local .env pins.
        with patch.dict(os.environ, {}, clear=False) as env, patch("dotenv.load_dotenv"):
            env.pop("OLLAMA_KEEP_ALIVE", None)
            import fritz_utils
            importlib.reload(fritz_utils)
            self.assertEqual(fritz_utils.OLLAMA_KEEP_ALIVE, "5m")
        importlib.reload(fritz_utils)

    def test_env_var_overrides_default(self):
        import importlib
        import os
        with patch.dict(os.environ, {"OLLAMA_KEEP_ALIVE": "30m"}):
            import fritz_utils
            importlib.reload(fritz_utils)
            self.assertEqual(fritz_utils.OLLAMA_KEEP_ALIVE, "30m")
        importlib.reload(fritz_utils)

    def test_a_bare_number_goes_to_ollama_as_a_number(self):
        # Ollama parses a STRING keep_alive with Go's time.ParseDuration, which
        # refuses "-1" or "300" ("time: missing unit in duration": HTTP 400 on
        # every request; only a bare "0" gets through); as numbers of seconds
        # it takes them all.
        from fritz_utils import parse_keep_alive
        self.assertEqual(parse_keep_alive("-1"), -1)
        self.assertIs(type(parse_keep_alive("-1")), int)
        self.assertEqual(parse_keep_alive("0"), 0)
        self.assertEqual(parse_keep_alive(" 300 "), 300)
        self.assertEqual(parse_keep_alive("+60"), 60)

    def test_a_duration_goes_as_written(self):
        from fritz_utils import parse_keep_alive
        for d in ("5m", "-1m", "30s", "1h", "1h30m"):
            self.assertEqual(parse_keep_alive(d), d)

    def test_unset_or_blank_is_five_minutes(self):
        from fritz_utils import parse_keep_alive
        for raw in (None, "", "   "):
            self.assertEqual(parse_keep_alive(raw), "5m")

    def test_the_setting_is_parsed(self):
        import importlib
        import os
        with patch.dict(os.environ, {"OLLAMA_KEEP_ALIVE": "-1"}):
            import fritz_utils
            importlib.reload(fritz_utils)
            self.assertEqual(fritz_utils.OLLAMA_KEEP_ALIVE, -1)
        importlib.reload(fritz_utils)


class TestWarmingOrder(unittest.TestCase):
    """The last model warmed is the one left loaded on a small GPU, so the
    thinking model goes last; a model in two roles is warmed once."""

    def setUp(self):
        self._real_ollama = sys.modules.get("ollama")
        # ollama_recovery binds the real ollama before the fake goes in.
        import ollama_recovery  # noqa: F401
        self.fake = _install_fake_ollama()
        import importlib
        import prewarm
        importlib.reload(prewarm)
        self.prewarm = prewarm

    def tearDown(self):
        if self._real_ollama is not None:
            sys.modules["ollama"] = self._real_ollama
        else:
            sys.modules.pop("ollama", None)

    def _warmed(self, **kw):
        _wait_for(self.prewarm.prewarm_models(**kw))
        return [c.kwargs["model"] for c in self.fake.chat.call_args_list]

    def test_the_thinking_model_is_warmed_last(self):
        import fritz_utils
        with patch.multiple(fritz_utils, THINKING_OLLAMA_MODEL="gpt-oss", FAST_OLLAMA_MODEL="llama3.2",
                            VISION_MODEL="llava", EMBEDDING_MODEL="mxbai-embed-large"):
            models = self.prewarm.configured_models()
        self.assertEqual(models["chat_models"][-1], "gpt-oss")
        self.assertEqual(set(models["chat_models"]), {"gpt-oss", "llama3.2", "llava"})
        self.assertEqual(self._warmed(**models), ["llava", "llama3.2", "gpt-oss"])

    def test_a_model_in_two_roles_is_warmed_once_at_its_last_place(self):
        self.assertEqual(self._warmed(chat_models=("gpt-oss", "llava", "gpt-oss"), keep_alive="5m"),
                         ["llava", "gpt-oss"])

    def test_zero_is_kept(self):
        # `keep_alive or "5m"` once turned 0 (unload at once) into five minutes.
        self._warmed(chat_models=("alpha",), keep_alive=0)
        self.assertEqual(self.fake.chat.call_args.kwargs["keep_alive"], 0)

    def test_a_runner_crash_is_recovered(self):
        import ollama_recovery
        crash = ollama_recovery.ollama.ResponseError(
            "llama-server process has terminated: exit status 0xc0000409", 500)
        self.fake.chat.side_effect = [crash, None]
        with patch.object(ollama_recovery, "make_room", return_value=["llava:latest"]) as room, \
                self.assertLogs("ollama_recovery", "WARNING"):
            self.assertEqual(self._warmed(chat_models=("gpt-oss",), keep_alive="5m"),
                             ["gpt-oss", "gpt-oss"])
        room.assert_called_once()


if __name__ == "__main__":
    unittest.main()
