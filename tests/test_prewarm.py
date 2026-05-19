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
    """Replace the real ollama module in sys.modules with a MagicMock-backed stub."""
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
        with patch.dict(os.environ, {}, clear=False) as env:
            env.pop("OLLAMA_KEEP_ALIVE", None)
            import fritz_utils
            importlib.reload(fritz_utils)
            self.assertEqual(fritz_utils.OLLAMA_KEEP_ALIVE, "5m")

    def test_env_var_overrides_default(self):
        import importlib
        import os
        with patch.dict(os.environ, {"OLLAMA_KEEP_ALIVE": "-1"}):
            import fritz_utils
            importlib.reload(fritz_utils)
            self.assertEqual(fritz_utils.OLLAMA_KEEP_ALIVE, "-1")


if __name__ == "__main__":
    unittest.main()
