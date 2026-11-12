"""Run the brain's web surfaces without Discord or Telegram.

    python main_local.py

Serves the admin panel and the web chat on 127.0.0.1:ADMIN_PANEL_PORT. It is
the same Starlette app main_discord starts from on_ready, so until now the web
surfaces existed only while a Discord bot was logged in. With
COMPANION_ENABLED, the same server also serves the desktop companion's
/companion WebSocket (companion_api).

Run this INSTEAD of main_discord.py, never beside it: the two share ports, the
SQLite checkpointer and the Chroma store. This one refuses to start while
another brain is listening on ADMIN_PANEL_PORT. main_discord.py does not check,
so stop this before starting that.

Needs ADMIN_PANEL_PASSWORD (without it there is nothing to serve), ROOT_USER,
and CHAT_PASSWORD for the web chat's sign-in. DISCORD_BOT_TOKEN is not needed.

What runs, compared with main_discord:

- The brain is loaded at startup, as main_discord loads it at import, so a
  configuration it cannot start with fails here and not on the first message.
- The scheduler does NOT run (plan 14, Q8): it can only post to Discord until
  plan 14's DeliverySink (PR 14). A ScheduleManager is still built, and never
  started, because the admin panel's Forget, export, /users and /schedules
  reach the stored schedules through it.
- The relay's retention purge does run, at startup and hourly. The relay is a
  Discord feature, but its retention is a privacy promise about message bodies
  stored in plaintext, and it must not lapse because Discord is not running.
- TTS does not load. Nothing on the web surfaces speaks yet (plan 14, PR 9).

Stopping: the first Ctrl+C or SIGTERM asks it to stop, and it exits once
nothing is left running. A turn caught in the middle of a model call finishes
that call first, which can take up to OLLAMA_TIMEOUT. A second Ctrl+C, or
Ctrl+Break, ends it at once and abandons that turn. How much of a turn is
worth draining on shutdown is plan 13's open stop-path question.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Callable

# Named, not __name__: run as a script this module is "__main__", and the log
# line should say which entry point wrote it.
logger = logging.getLogger("main_local")

# How often the main thread wakes to check whether it has been asked to stop.
# It must not block untimed: on Windows, a bare threading.Event.wait() is not
# interrupted by Ctrl+C, so the handler would never run. Measured on Python
# 3.12: a bare wait() was still blocked 8 s after SIGINT, while this loop
# stopped within one poll of it.
_STOP_POLL_S = 0.5

# relay_store.purge_expired's cadence, as main_discord schedules it.
_RELAY_PURGE_EVERY_S = 3600

# Ctrl+Break (SIGBREAK) is deliberately left alone: its Windows default is to
# end the process at once, which is what someone pressing it wants.
_STOP_SIGNALS = ("SIGINT", "SIGTERM")


class _StopFlag:
    """Set from a signal handler, so setting it must take no lock.

    threading.Event.set() acquires the Event's condition lock, and the handler
    runs on the main thread, which may hold that very lock inside Event.wait()
    at that instant. The handler would then wait on itself forever. Assigning
    an attribute takes no lock.
    """

    def __init__(self) -> None:
        self.requested: int | None = None   # the signal number, once asked


def _serve_until_stopped(flag: _StopFlag, still_serving: Callable[[], bool],
                         sleep: Callable[[float], None] = time.sleep) -> int:
    """Block until a stop is requested (0), or until the web server dies (1).

    The second case is a bind failure, almost always the port already being in
    use because main_discord or another main_local is running. uvicorn logs the
    error and its thread ends, after start_admin_panel has already returned.
    Without this check the process would sit there, apparently up, serving
    nothing.
    """
    while flag.requested is None:
        if not still_serving():
            logger.error(
                "The web server stopped. If uvicorn logged a bind error above, "
                "the port is already in use - is main_discord.py or another "
                "main_local.py running?"
            )
            return 1
        sleep(_STOP_POLL_S)
    logger.info(
        "Stopping (signal %s). A turn mid-way through a model call finishes it "
        "first; press Ctrl+C again to stop at once.", flag.requested,
    )
    return 0


def _install_stop_handlers(flag: _StopFlag) -> None:
    """The first SIGINT or SIGTERM asks politely; the defaults then come back.

    A stop does not end the process by itself: interpreter shutdown still
    waits for a model call in flight on LangGraph's worker pool. If the
    handlers stayed installed, every later Ctrl+C would land on them and do
    nothing, and the only way out would be to kill the process. Restoring the
    defaults makes the second Ctrl+C the hard stop people expect.
    """
    sigs = [s for s in (getattr(signal, n, None) for n in _STOP_SIGNALS) if s is not None]

    def _handler(signum, _frame):
        # No logging here: the logging module takes a lock too.
        flag.requested = signum
        for sig in sigs:
            signal.signal(sig, signal.SIG_DFL)

    for sig in sigs:
        signal.signal(sig, _handler)


def _start_relay_retention(flag: _StopFlag) -> threading.Thread:
    """relay_store.purge_expired at startup and hourly, until asked to stop.

    main_discord runs the same purge from relay_housekeeping on its scheduler,
    which is not started here. The rest of that housekeeping (reconciling
    half-sent relays) needs Discord and stays there.
    """
    import relay_store

    def _run() -> None:
        while flag.requested is None:
            try:
                relay_store.purge_expired()
            except Exception as e:
                logger.error("relay: could not purge expired messages: %s", e)
            time.sleep(_RELAY_PURGE_EVERY_S)

    t = threading.Thread(target=_run, name="relay-retention", daemon=True)
    t.start()
    return t


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        # Same rule as main_discord: an argument that is silently ignored is how
        # the release smoke test once ran the real bot instead of a mock.
        print(f"main_local.py takes no arguments, not {args}", file=sys.stderr)
        return 2

    # Everything heavy is imported here rather than at module scope, so that
    # importing this module (as the tests do) starts nothing and loads nothing.
    import fritz_utils
    from observability import init_logging, start_metrics_server
    from prewarm import configured_models, prewarm_models

    init_logging()

    if not fritz_utils.ADMIN_PANEL_PASSWORD:
        logger.error(
            "ADMIN_PANEL_PASSWORD is not set, so there is nothing to serve: the "
            "admin panel, the web chat and the desktop companion all run in the "
            "one web server it guards. Set it in .env and start again."
        )
        return 2

    # Raises on a missing ROOT_USER or a CHROMA_API_IMPL pointing at a remote
    # server, exactly as main_discord's call does.
    fritz_utils.validate_config(require_discord=False)
    try:
        start_metrics_server()
    except OSError as e:
        # Both entry points listen on METRICS_PORT, and this binds synchronously
        # before the web server, so on Linux a brain that is already running is
        # caught here. Not on Windows: HTTPServer sets SO_REUSEADDR, which there
        # lets a second process bind a port that is still in use (seen live),
        # so the check in _serve_until_stopped is what catches it instead.
        logger.error(
            "Could not start the metrics server (%s). Is main_discord.py or "
            "another main_local.py already running? Run one at a time.", e,
        )
        return 1

    # The brain, loaded now as main_discord loads it at import. Left to the
    # first chat message, a configuration it cannot start with (an unopenable
    # CHAT_DB_NAME, say) would fail inside a request thread: this process would
    # say it was serving while every message hung.
    import mister_fritz  # noqa: F401

    # Built for its store, never started: see the module docstring.
    from scheduler import ScheduleManager
    schedules = ScheduleManager(None)

    import admin_panel
    port = admin_panel.start_admin_panel(schedule_manager=schedules)
    if port is None:
        return 2

    prewarm_models(**configured_models())

    flag = _StopFlag()
    _install_stop_handlers(flag)
    _start_relay_retention(flag)
    logger.info("Serving http://127.0.0.1:%d/ without Discord. Ctrl+C to stop.", port)
    return _serve_until_stopped(flag, admin_panel.panel_is_serving)


if __name__ == "__main__":
    raise SystemExit(main())
