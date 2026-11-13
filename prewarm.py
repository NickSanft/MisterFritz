"""Pre-warm Ollama models so the first user request doesn't pay the cold-load tax.

Sends a 1-token completion to each configured model in a background thread.
Silent on failure — if Ollama is unreachable we'll find out at first real
request, no need to alarm anyone at startup.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional, Union

logger = logging.getLogger(__name__)


def _prewarm_one(client, model: str, keep_alive: Union[str, int]) -> None:
    """Send a 1-token chat to load the model into Ollama's resident set,
    once more if Ollama's runner crashes loading it (ollama_recovery)."""
    try:
        from ollama_recovery import call_with_recovery
        call_with_recovery(
            model,
            client.chat,
            model=model,
            messages=[{"role": "user", "content": "."}],
            options={"num_predict": 1},
            keep_alive=keep_alive,
        )
        logger.info("Pre-warmed Ollama model: %s", model)
    except Exception as e:
        # Don't alarm on startup — the first real request will surface the issue.
        logger.debug("Pre-warm of %s failed (non-fatal): %s", model, e)


def _prewarm_embedding(model: str, keep_alive: Union[str, int]) -> None:
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

    The thinking model is warmed LAST of the chat models. On a GPU too small
    for all of them at once (16 GB holds gpt-oss or llava, not both), each
    load evicts the one before, so the last one warmed is the one left loaded,
    and that should be the model every turn needs. Warmed first, it was
    evicted by the vision model before the first message, which then waited
    for it to load again. The embedder comes after it: it is small and loads
    beside it.
    """
    import fritz_utils
    return {
        "chat_models": (fritz_utils.VISION_MODEL, fritz_utils.FAST_OLLAMA_MODEL,
                        fritz_utils.THINKING_OLLAMA_MODEL),
        "embedding_models": (fritz_utils.EMBEDDING_MODEL,),
        "keep_alive": fritz_utils.OLLAMA_KEEP_ALIVE,
    }


def _once_each(models: Iterable[Optional[str]]) -> list[str]:
    """The named models in order, each once: the same model in two roles
    (FAST_OLLAMA_MODEL=gpt-oss, say) is warmed once, at its LAST place, so
    the order's meaning (last warmed, left loaded) holds."""
    named = [m for m in models if m]
    return [m for i, m in enumerate(named) if m not in named[i + 1:]]


def prewarm_models(
    chat_models: Iterable[str],
    embedding_models: Iterable[str] = (),
    keep_alive: Union[str, int, None] = None,
) -> threading.Thread:
    """Pre-warm a set of Ollama models in a background daemon thread, in the
    order given.

    Returns the thread so callers can join() in tests; production paths
    should just fire-and-forget.
    """
    chat_models = _once_each(chat_models)
    embedding_models = _once_each(embedding_models)
    # Not `keep_alive or "5m"`: 0 (unload at once) is a setting, not a gap.
    ka = "5m" if keep_alive is None else keep_alive

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
