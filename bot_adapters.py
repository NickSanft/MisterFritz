"""Shared helpers used by both the Discord and Telegram entrypoints.

Keep this module deliberately small. Anything platform-specific (streaming,
attachments, voice) belongs in the respective main_*.py adapter, not here.

Dependency-light on purpose — stdlib plus fritz_utils only, and specifically no
`discord` import, so the Telegram adapter and the tests can use it freely.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from fritz_utils import BLOCKING_POOL_SIZE

# One shared, *bounded* pool for every blocking call the bots make.
#
# Bounded on purpose. Ollama and the GPU serialise anyway, so an unbounded pool
# does not make the work finish sooner — it just queues the contention one layer
# deeper, where it is invisible and where it competes with discord.py's own use
# of the loop's default executor. One named pool means one number to reason
# about and one knob to turn.
_BLOCKING_POOL = ThreadPoolExecutor(
    max_workers=BLOCKING_POOL_SIZE,
    thread_name_prefix="fritz-blocking",
)


async def run_blocking(func, *args, **kwargs):
    """Run a blocking callable on the shared pool and await its result.

    Use this for anything that touches Ollama, the GPU, or the disk from inside
    an async Discord handler. Exceptions propagate to the awaiting coroutine.

    Note this is admission control for *threads*, not for the GPU. Callers that
    must not run concurrently (SDXL, XTTS) additionally take an asyncio
    semaphore BEFORE calling in, so waiters park on the event loop rather than
    occupying a pool thread while they wait.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _BLOCKING_POOL, functools.partial(func, *args, **kwargs)
    )


def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    """Split a string into chunks of at most chunk_size characters.

    chunk_size defaults to Discord's 2000-char message cap. Pass 4096 for
    Telegram's reply limit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]
