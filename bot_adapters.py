"""Shared helpers used by both the Discord and Telegram entrypoints.

Keep this module deliberately small. Anything platform-specific (streaming,
attachments, voice) belongs in the respective main_*.py adapter, not here.

Dependency-light on purpose — stdlib plus fritz_utils only, and specifically no
`discord` import, so the Telegram adapter and the tests can use it freely.
"""
import asyncio
import functools
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from fritz_utils import BLOCKING_POOL_SIZE, DISCORD_ERROR_DETAIL
from observability import METRICS

logger = logging.getLogger(__name__)

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


FENCE = "```"


def _find_break(text: str, limit: int) -> int:
    """Index to cut `text` at (<= limit), preferring paragraph > line > word.

    Falls back to a hard cut at `limit` when no boundary sits in the second half
    of the window — a boundary at index 3 of a 2000-char budget would waste the
    whole message to avoid one split word.

    The separator is kept in the PRECEDING chunk, so "".join(chunks) still
    reproduces the input exactly whenever no fence had to be repaired.
    """
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    for sep in ("\n\n", "\n", " "):
        idx = window.rfind(sep)
        if idx > limit // 2:
            return idx + len(sep)
    return limit


def _open_fence_lang(text: str) -> str | None:
    """Info string of a ``` fence left open at the end of `text`, else None.

    Toggles: an opening fence records its language tag, the next fence closes
    it. Returns "" for a bare ``` with no language, which is still "open".
    """
    lang = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            lang = stripped[len(FENCE):].strip() if lang is None else None
    return lang


def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    """Split a string into chunks of at most chunk_size characters.

    chunk_size defaults to Discord's 2000-char message cap. Pass 4096 for
    Telegram's reply limit.

    Breaks on paragraph/line/word boundaries where one exists, and never leaves
    a ``` code fence unbalanced: a fence spanning a cut is closed at the end of
    one chunk and reopened, with its language tag, at the top of the next.
    Without that, Discord renders the tail of a split code block as prose and
    the rest of the message as code.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not s:
        return []
    if len(s) <= chunk_size:
        return [s]

    chunks: list[str] = []
    remaining = s
    carry_lang: str | None = None

    while remaining:
        prefix = f"{FENCE}{carry_lang}\n" if carry_lang is not None else ""
        budget = max(1, chunk_size - len(prefix))
        cut = _find_break(remaining, budget)
        piece = prefix + remaining[:cut]
        open_lang = _open_fence_lang(piece)
        if open_lang is not None and cut < len(remaining):
            # Ending inside a fence — re-cut to reserve room for the closer.
            budget = max(1, budget - (len(FENCE) + 1))
            cut = _find_break(remaining, budget)
            piece = prefix + remaining[:cut]
            open_lang = _open_fence_lang(piece)
            if open_lang is not None:
                piece = piece.rstrip("\n") + "\n" + FENCE
        remaining = remaining[cut:]
        carry_lang = open_lang if remaining else None
        chunks.append(piece)
    return chunks


_ERROR_COPY = (
    "I'm afraid that did not go to plan. The particulars have been noted in "
    "the log, where such things belong."
)


def fritz_error(operation: str, exc: BaseException | None = None, *,
                note: str | None = None, record: bool = True) -> str:
    """Record `exc` and return user-facing copy in Mister Fritz's voice.

    Pass record=False when the caller has already recorded this failure —
    METRICS.time_block records and re-raises, so a handler that wraps its work
    in time_block AND routes the exception through here counted every error
    twice.

    The real exception goes to METRICS and the log only. The returned string
    carries a short ref so an admin can grep the log for the exact failure
    without the user ever seeing a Python traceback — which is both a poor
    experience and a small information leak.

    Set DISCORD_ERROR_DETAIL=1 to append the exception text when debugging a
    private instance.
    """
    ref = uuid.uuid4().hex[:8]
    if exc is not None:
        if record:
            METRICS.record_error(operation, exc)
        logger.exception("[%s] %s failed", ref, operation, exc_info=exc)
    else:
        logger.error("[%s] %s failed", ref, operation)
    parts = [_ERROR_COPY if note is None else note]
    if DISCORD_ERROR_DETAIL and exc is not None:
        parts.append(f"`{type(exc).__name__}: {exc}`")
    parts.append(f"(ref `{ref}`)")
    return " ".join(parts)
