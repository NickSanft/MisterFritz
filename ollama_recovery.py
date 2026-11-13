"""Recovering when Ollama's model runner crashes as it loads a model.

Seen with Ollama 0.40.16 (its CUDA 13 runner) on an RTX 5080 Laptop GPU,
driver 621.14: loading gpt-oss while any other model's runner is alive kills
the new runner before it answers. Ollama returns HTTP 500 "llama-server process
has terminated: exit status 0xc0000409", and its server.log says "CUDA error:
shared object initialization failed". Measured by loading gpt-oss from cold
ten times in each case:

- nothing else loaded: 10 of 10 loads worked;
- the embedder loaded beside it: 5 of 10;
- a CPU-only model loaded beside it: 3 of 10 (its runner still holds the GPU);
- Ollama's CUDA 12 runner instead: 1 of 15, even alone.

Every other model loaded fine every time. Before this module, each such crash
was a turn that failed ("I'm afraid that did not go to plan"), and it happened
on most reloads of the thinking model: the fast model or the vision model
evicting it on a 16 GB GPU, or every model unloading after OLLAMA_KEEP_ALIVE.

The crash comes before the first token, so the request can safely be made
again. What makes the second try work is the finding above: unload every other
model first, so the one asked for loads alone. One retry only: if it fails
again, the error is the caller's, as before.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import ollama
from langchain_ollama import ChatOllama

import fritz_utils
from observability import METRICS

logger = logging.getLogger(__name__)

T = TypeVar("T")

# What Ollama says when the runner it started for a model died as it loaded.
RUNNER_CRASH = "llama-server process has terminated"
# ...and, after it, what llama.cpp's loader says when the model itself can't be
# loaded (an unknown architecture, a corrupt file): that fails every time, and
# unloading the other models for it would only make them load again.
LOADER_FAILED = "error loading model"
# How long make_room waits for the models it unloaded to leave, and how long
# any one of its requests to Ollama may take.
UNLOAD_WAIT_S = 15.0
REQUEST_TIMEOUT_S = 10.0


def is_runner_crash(error: BaseException) -> bool:
    """Whether `error` is Ollama reporting that a model's runner died as it
    loaded, for a reason another try might not meet."""
    if not (isinstance(error, ollama.ResponseError) and error.status_code == 500):
        return False
    text = str(error.error)
    return RUNNER_CRASH in text and LOADER_FAILED not in text.lower()


def _base_name(model: str | None) -> str:
    """"gpt-oss" for "gpt-oss:latest": Ollama names a model with its tag."""
    name = (model or "").strip()
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _client() -> ollama.Client:
    # A timeout of its own: the client the brain's other direct calls use has
    # none, and recovery must not hold a turn longer than it saves.
    return ollama.Client(host=fritz_utils.OLLAMA_HOST, timeout=REQUEST_TIMEOUT_S)


def make_room(model: str, client: Any = None, wait_s: float = UNLOAD_WAIT_S) -> list[str]:
    """Unload every loaded model but `model`, and wait until they have gone
    (the wait bounded by `wait_s`, each request by REQUEST_TIMEOUT_S).
    Returns the names unloaded."""
    client = client or _client()
    keep = _base_name(model)
    others = [m.model for m in client.ps().models if m.model and _base_name(m.model) != keep]
    for name in others:
        try:
            # A generate with no prompt and keep_alive 0 is Ollama's unload.
            client.generate(model=name, keep_alive=0)
        except Exception as e:  # noqa: BLE001 - recovery is best effort
            logger.warning("Couldn't unload %s to make room for %s: %s", name, model, e)
    deadline = time.monotonic() + wait_s
    while others and time.monotonic() < deadline:
        loaded = {m.model for m in client.ps().models}
        if not loaded.intersection(others):
            break
        time.sleep(0.25)
    return others


def recover(model: str, error: BaseException, client: Any = None) -> None:
    """Log the crash and make room for `model`, before the one retry."""
    METRICS.increment("ollama_runner_crash")
    try:
        unloaded = make_room(model, client)
    except Exception as e:  # noqa: BLE001 - the retry may still work
        logger.warning("Couldn't list Ollama's models to make room for %s: %s", model, e)
        unloaded = []
    logger.warning(
        "Ollama's runner for %s crashed as it loaded (%s). Unloaded %s and asking "
        "again: on some GPUs Ollama can't load gpt-oss beside another model.",
        model, str(getattr(error, "error", error))[:160], ", ".join(unloaded) or "nothing",
    )


def call_with_recovery(model: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """`fn(*args, **kwargs)`, made once more, after make_room(model), if
    Ollama's runner crashed loading `model`. For calls that return whole
    (not streamed) replies. `model` and `fn` are positional-only, so `fn`
    can be given its own `model=`, as Ollama's chat is."""
    try:
        return fn(*args, **kwargs)
    except ollama.ResponseError as e:
        if not is_runner_crash(e):
            raise
        recover(model, e)
    return fn(*args, **kwargs)


class RecoveringChatOllama(ChatOllama):
    """ChatOllama that asks once more when Ollama's runner crashes loading
    the model.

    Every ChatOllama request, streamed or not, structured output and tool
    calls included, goes through _create_chat_stream (or its async twin). The
    retry is made only if the crash came before the first chunk: a stream that
    had started would have given its caller words it can't take back.
    """

    def _create_chat_stream(self, messages, stop=None, **kwargs):
        started = False
        try:
            for part in super()._create_chat_stream(messages, stop, **kwargs):
                started = True
                yield part
            return
        except ollama.ResponseError as e:
            if started or not is_runner_crash(e):
                raise
            recover(self.model, e)
        yield from super()._create_chat_stream(messages, stop, **kwargs)

    async def _acreate_chat_stream(self, messages, stop=None, **kwargs):
        import asyncio

        started = False
        try:
            async for part in super()._acreate_chat_stream(messages, stop, **kwargs):
                started = True
                yield part
            return
        except ollama.ResponseError as e:
            if started or not is_runner_crash(e):
                raise
            await asyncio.to_thread(recover, self.model, e)
        async for part in super()._acreate_chat_stream(messages, stop, **kwargs):
            yield part
