"""Pre-warm Ollama models so the first user request doesn't pay the cold-load tax.

Sends a 1-token completion to each configured model in a background thread.
Silent on failure — if Ollama is unreachable we'll find out at first real
request, no need to alarm anyone at startup.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _prewarm_one(client, model: str, keep_alive: str) -> None:
    """Send a 1-token chat to load the model into Ollama's resident set."""
    try:
        client.chat(
            model=model,
            messages=[{"role": "user", "content": "."}],
            options={"num_predict": 1},
            keep_alive=keep_alive,
        )
        logger.info("Pre-warmed Ollama model: %s", model)
    except Exception as e:
        # Don't alarm on startup — the first real request will surface the issue.
        logger.debug("Pre-warm of %s failed (non-fatal): %s", model, e)


def _prewarm_embedding(model: str, keep_alive: str) -> None:
    """Embedding endpoint takes a different request shape than chat."""
    try:
        import ollama
        ollama.embeddings(model=model, prompt=".", keep_alive=keep_alive)
        logger.info("Pre-warmed Ollama embedding model: %s", model)
    except Exception as e:
        logger.debug("Pre-warm of embedding %s failed (non-fatal): %s", model, e)


def configured_models() -> dict:
    """The models every entry point warms, as keyword arguments for prewarm_models.

    The three chat models and the embedder. The vision model is not optional:
    Discord and the web chat both accept images. One definition, so that
    main_discord and main_local cannot drift apart. fritz_utils is imported
    here, not at module scope, to keep this module cheap to import.
    """
    import fritz_utils
    return {
        "chat_models": (fritz_utils.THINKING_OLLAMA_MODEL, fritz_utils.FAST_OLLAMA_MODEL,
                        fritz_utils.VISION_MODEL),
        "embedding_models": (fritz_utils.EMBEDDING_MODEL,),
        "keep_alive": fritz_utils.OLLAMA_KEEP_ALIVE,
    }


def prewarm_models(
    chat_models: Iterable[str],
    embedding_models: Iterable[str] = (),
    keep_alive: Optional[str] = None,
) -> threading.Thread:
    """Pre-warm a set of Ollama models in a background daemon thread.

    Returns the thread so callers can join() in tests; production paths
    should just fire-and-forget.
    """
    chat_models = [m for m in chat_models if m]
    embedding_models = [m for m in embedding_models if m]
    ka = keep_alive or "5m"

    def _run() -> None:
        try:
            import ollama
            client = ollama
            for m in chat_models:
                _prewarm_one(client, m, ka)
            for m in embedding_models:
                _prewarm_embedding(m, ka)
        except Exception as e:
            logger.debug("Pre-warm thread aborted (non-fatal): %s", e)

    t = threading.Thread(target=_run, name="ollama-prewarm", daemon=True)
    t.start()
    return t
