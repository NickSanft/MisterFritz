"""
main_local.py - the brain's web surfaces with no Discord login
(P1 of plans/14-desktop-companion-build.md).

The process half of this (that `python main_local.py` really serves /chat with
Discord unconfigured) was checked by hand when it landed. What is tested here
is everything that decides whether it starts, what it starts, and whether it
can be stopped.

Every call that could block runs through _bounded, so a regression fails an
assertion instead of hanging the suite: CI has no per-test timeout.
"""
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import admin_panel
import fritz_utils
import main_local
import observability
import prewarm
import privacy
import relay_store
import scheduler

REPO = Path(__file__).resolve().parents[1]


def _run_python(code: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """A fresh interpreter. In this one, other tests have imported everything."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO,
                          capture_output=True, text=True, timeout=timeout)


def _bounded(test: unittest.TestCase, fn, timeout: float = 5.0):
    """fn() on a daemon thread; fail, rather than hang, if it does not return."""
    box = {}

    def run():
        box["value"] = fn()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        test.fail(f"did not return within {timeout}s")
    return box["value"]


class TestImportingStartsNothing(unittest.TestCase):
    def test_import_loads_no_server_no_agent_and_starts_no_thread(self):
        proc = _run_python("""
            import sys, threading
            import main_local
            heavy = ("admin_panel", "mister_fritz", "uvicorn", "prewarm", "fritz_utils",
                     "scheduler", "relay_store")
            print(sorted(m for m in heavy if m in sys.modules))
            print(threading.active_count())
        """)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        loaded, threads = proc.stdout.splitlines()
        self.assertEqual(loaded, "[]")
        self.assertEqual(threads, "1")


class _MainHarness(unittest.TestCase):
    """main() with everything that would start a thread, bind a port or touch
    the process's signal handlers replaced by a mock."""

    def setUp(self):
        self.started = {}
        for name, target, kwargs in (
            ("init_logging", observability, {}),
            ("start_metrics_server", observability, {"return_value": 8000}),
            ("prewarm_models", prewarm, {}),
            ("start_admin_panel", admin_panel, {"return_value": 8001}),
        ):
            p = patch.object(target, name, **kwargs)
            self.started[name] = p.start()
            self.addCleanup(p.stop)
        for name, kwargs in (("_install_stop_handlers", {}),
                             ("_start_relay_retention", {}),
                             ("_serve_until_stopped", {"return_value": 0})):
            p = patch.object(main_local, name, **kwargs)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        for name, value in (("ADMIN_PANEL_PASSWORD", "pw"), ("ROOT_USER", "discord-1"),
                            ("DISCORD_BOT_TOKEN", None)):
            p = patch.object(fritz_utils, name, value)
            p.start()
            self.addCleanup(p.stop)

    def assert_nothing_started(self):
        for name in ("start_metrics_server", "prewarm_models", "start_admin_panel"):
            with self.subTest(name=name):
                self.started[name].assert_not_called()
        self._install_stop_handlers.assert_not_called()
        self._start_relay_retention.assert_not_called()


class TestWhenItRefusesToStart(_MainHarness):
    def test_arguments_are_refused_not_ignored(self):
        """main_discord learned this the hard way: an ignored argument let the
        release smoke test boot the real bot when it asked for a mock."""
        self.assertEqual(main_local.main(["--preflight"]), 2)
        self.assert_nothing_started()

    def test_no_admin_password_means_nothing_to_serve(self):
        with patch.object(fritz_utils, "ADMIN_PANEL_PASSWORD", None):
            self.assertEqual(main_local.main([]), 2)
        self.assert_nothing_started()

    def test_a_missing_root_user_still_stops_it(self):
        """Discord is optional here; the owner's identity is not."""
        with patch.object(fritz_utils, "ROOT_USER", None):
            with self.assertRaises(RuntimeError) as caught:
                main_local.main([])
        self.assertIn("ROOT_USER", str(caught.exception))
        self.assertNotIn("DISCORD_BOT_TOKEN", str(caught.exception))
        self.assert_nothing_started()

    def test_a_brain_already_running_is_named_not_a_traceback(self):
        """The metrics port is bound synchronously and first, so on Linux it is
        where a running main_discord.py is discovered. (On Windows the bind
        succeeds anyway - SO_REUSEADDR - and the web-server check catches it.)"""
        self.started["start_metrics_server"].side_effect = OSError(10048, "address in use")
        with self.assertLogs("main_local", level="ERROR") as logs:
            self.assertEqual(main_local.main([]), 1)
        self.assertIn("already running", " ".join(logs.output))
        self.started["start_admin_panel"].assert_not_called()
        self.started["prewarm_models"].assert_not_called()

    def test_a_brain_that_cannot_load_stops_it_before_it_serves(self):
        """Left to the first chat message, the failure would happen in a
        request thread while the log said it was serving."""
        with patch.dict(sys.modules, {"mister_fritz": None}):
            with self.assertRaises(ImportError):
                main_local.main([])
        self.started["start_admin_panel"].assert_not_called()

    def test_a_panel_that_did_not_start_is_reported(self):
        self.started["start_admin_panel"].return_value = None
        self.assertEqual(main_local.main([]), 2)
        self.started["prewarm_models"].assert_not_called()
        self._serve_until_stopped.assert_not_called()


class TestWhatItStarts(_MainHarness):
    def test_it_serves_with_no_discord_token(self):
        self.assertEqual(main_local.main([]), 0)
        self.started["start_admin_panel"].assert_called_once()
        self.started["start_metrics_server"].assert_called_once()
        flag = self._install_stop_handlers.call_args.args[0]
        self.assertIsInstance(flag, main_local._StopFlag)
        self._start_relay_retention.assert_called_once_with(flag)
        self.assertEqual(self._serve_until_stopped.call_args.args,
                         (flag, admin_panel.panel_is_serving))

    def test_the_panel_gets_a_schedule_store_that_never_fires(self):
        """With None, the panel's Forget reported "0 schedules" and left every
        row for main_discord's next boot to replay. A real manager, never
        started, reaches the rows without anything firing or posting."""
        main_local.main([])
        manager = self.started["start_admin_panel"].call_args.kwargs["schedule_manager"]
        self.assertIsInstance(manager, scheduler.ScheduleManager)
        self.assertFalse(manager.scheduler.running)

    def test_it_warms_the_shared_model_set(self):
        """The same call main_discord makes (tests/test_discord_commands.py)."""
        main_local.main([])
        self.started["prewarm_models"].assert_called_once_with(**prewarm.configured_models())

    def test_its_exit_code_is_the_serve_loops(self):
        self._serve_until_stopped.return_value = 1
        self.assertEqual(main_local.main([]), 1)


class TestForgetReachesSchedulesUnderMainLocal(unittest.TestCase):
    """The privacy half of the test above, against the real SQLite store."""

    USER = "discord-919191"

    def test_forget_deletes_the_rows_an_unstarted_manager_can_see(self):
        manager = scheduler.ScheduleManager(None)
        with sqlite3.connect(scheduler.SCHEDULE_DB) as conn:
            conn.execute(
                "INSERT INTO schedules (id, user_id, channel_id, prompt, schedule_expr, created_at) "
                "VALUES ('p1test01', ?, 1, 'a private thing', '0 9 * * *', '2026-11-10T00:00:00')",
                (self.USER,))
        self.addCleanup(self._delete_rows)
        self.assertIn(self.USER, [s["user_id"] for s in manager.list_all_schedules()])
        self.assertEqual(privacy.forget_schedules(self.USER, manager), 1)
        self.assertNotIn(self.USER, [s["user_id"] for s in manager.list_all_schedules()])
        self.assertFalse(manager.scheduler.running)

    def _delete_rows(self):
        with sqlite3.connect(scheduler.SCHEDULE_DB) as conn:
            conn.execute("DELETE FROM schedules WHERE user_id = ?", (self.USER,))


class TestServeUntilStopped(unittest.TestCase):
    def test_a_stop_request_ends_it_cleanly(self):
        flag = main_local._StopFlag()
        threading.Timer(0.05, setattr, args=(flag, "requested", 2)).start()
        with patch.object(main_local, "_STOP_POLL_S", 0.01):
            code = _bounded(self, lambda: main_local._serve_until_stopped(flag, lambda: True))
        self.assertEqual(code, 0)

    def test_a_dead_web_server_ends_it_with_an_error(self):
        """A bind failure kills uvicorn's thread after start_admin_panel has
        returned. Without this the process would stand there serving nothing."""
        with patch.object(main_local, "_STOP_POLL_S", 0.01), \
             self.assertLogs("main_local", level="ERROR") as logs:
            code = _bounded(self, lambda: main_local._serve_until_stopped(
                main_local._StopFlag(), lambda: False))
        self.assertEqual(code, 1)
        self.assertIn("port is already in use", " ".join(logs.output))

    def test_every_wait_is_bounded(self):
        """On Windows an untimed wait is not interrupted by Ctrl+C, so the
        handler would never run. Platform-independent, so CI catches the
        regression too; TestSignalsStopIt proves the behaviour itself."""
        flag = main_local._StopFlag()
        naps = []

        def sleep(seconds):
            naps.append(seconds)
            if len(naps) == 3:
                flag.requested = 2

        code = _bounded(self, lambda: main_local._serve_until_stopped(flag, lambda: True, sleep))
        self.assertEqual(code, 0)
        self.assertEqual(len(naps), 3)
        self.assertTrue(all(0 < n <= 1 for n in naps), naps)


class TestSignalsStopIt(unittest.TestCase):
    """The real handlers in a real interpreter. On Windows the first test is
    the one that hangs - and times out here - if the loop ever blocks untimed;
    on Linux an untimed wait happens to be interruptible, so there it proves
    the handlers are wired and test_every_wait_is_bounded guards the loop."""

    def test_sigint_and_sigterm_each_end_the_wait_with_status_0(self):
        for name in ("SIGINT", "SIGTERM"):
            with self.subTest(signal=name):
                proc = _run_python(f"""
                    import signal, threading
                    import main_local
                    flag = main_local._StopFlag()
                    main_local._install_stop_handlers(flag)
                    threading.Timer(0.3, signal.raise_signal, args=(signal.{name},)).start()
                    raise SystemExit(main_local._serve_until_stopped(flag, lambda: True))
                """, timeout=20)
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_the_second_ctrl_c_ends_the_process_at_once(self):
        """Interpreter shutdown waits for a model call in flight. If the first
        handler stayed installed, every later Ctrl+C would land on it and do
        nothing. The first one puts the default back instead."""
        proc = _run_python("""
            import signal
            import main_local
            flag = main_local._StopFlag()
            main_local._install_stop_handlers(flag)
            signal.raise_signal(signal.SIGINT)
            print("first:", flag.requested == signal.SIGINT,
                  signal.getsignal(signal.SIGINT) is signal.SIG_DFL,
                  signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, flush=True)
            signal.raise_signal(signal.SIGINT)
            print("survived the second", flush=True)
        """, timeout=20)
        self.assertIn("first: True True True", proc.stdout)
        self.assertNotIn("survived", proc.stdout)
        self.assertNotEqual(proc.returncode, 0)

    def test_ctrl_break_is_left_as_the_hard_stop(self):
        if not hasattr(__import__("signal"), "SIGBREAK"):
            self.skipTest("SIGBREAK exists only on Windows")
        proc = _run_python("""
            import signal
            import main_local
            before = signal.getsignal(signal.SIGBREAK)
            main_local._install_stop_handlers(main_local._StopFlag())
            print(signal.getsignal(signal.SIGBREAK) is before)
        """)
        self.assertEqual(proc.stdout.strip(), "True", proc.stderr)


class TestRelayRetention(unittest.TestCase):
    def test_it_purges_now_and_again_and_survives_a_failure(self):
        flag = main_local._StopFlag()
        calls = []
        two_runs = threading.Event()

        def purge():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            two_runs.set()

        with patch.object(relay_store, "purge_expired", purge), \
             patch.object(main_local, "_RELAY_PURGE_EVERY_S", 0.01), \
             self.assertLogs("main_local", level="ERROR") as logs:
            t = main_local._start_relay_retention(flag)
            self.assertTrue(two_runs.wait(5), "the purge did not run again after a failure")
            flag.requested = 2
            t.join(5)
        self.assertFalse(t.is_alive(), "the purge thread did not stop when asked")
        self.assertTrue(t.daemon)
        self.assertIn("could not purge", " ".join(logs.output))


class TestPanelIsServing(unittest.TestCase):
    """The one guard that makes a second brain on Windows exit."""

    def setUp(self):
        p = patch.object(admin_panel, "_PANEL_THREAD", None)
        p.start()
        self.addCleanup(p.stop)

    def test_never_started(self):
        self.assertFalse(admin_panel.panel_is_serving())

    def test_a_thread_that_has_ended(self):
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        admin_panel._PANEL_THREAD = t
        self.assertFalse(admin_panel.panel_is_serving())

    def test_a_live_thread(self):
        release = threading.Event()
        t = threading.Thread(target=release.wait, daemon=True)
        t.start()
        self.addCleanup(release.set)
        admin_panel._PANEL_THREAD = t
        self.assertTrue(admin_panel.panel_is_serving())

    # Both expected. The thread dying of uvicorn's sys.exit(1) is the event
    # under test. The deprecation is uvicorn's "auto" WebSocket protocol
    # importing websockets.legacy, recorded in b73d28b and left for PR 1.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning:uvicorn")
    def test_a_real_bind_failure_turns_it_false(self):
        """End to end: the port is taken, uvicorn's thread dies, and this
        notices, which is what main_local's loop polls."""
        import socket
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        port = blocker.getsockname()[1]
        with patch.object(admin_panel, "ADMIN_PANEL_PASSWORD", "pw"), \
             patch.object(admin_panel, "ADMIN_PANEL_PORT", port), \
             patch.object(admin_panel, "logger", MagicMock()):
            self.assertEqual(admin_panel.start_admin_panel(schedule_manager=None), port)
            deadline = time.monotonic() + 10
            while admin_panel.panel_is_serving() and time.monotonic() < deadline:
                time.sleep(0.05)
        self.assertFalse(admin_panel.panel_is_serving())


if __name__ == "__main__":
    unittest.main()
