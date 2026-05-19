import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


def _make_manager(db_path: str):
    """Return a ScheduleManager pointed at db_path with a mock bot and scheduler."""
    import fritz_utils

    with patch.object(fritz_utils, "SCHEDULE_DB", db_path):
        # Re-import scheduler so it picks up the patched SCHEDULE_DB at module level.
        import importlib
        import scheduler as sched_mod
        importlib.reload(sched_mod)

        bot = MagicMock()
        manager = sched_mod.ScheduleManager(bot)
        # Replace the real APScheduler so no background threads start.
        manager.scheduler = MagicMock()
        return manager, sched_mod


class TestScheduleManagerDB(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.manager, self.sched_mod = _make_manager(self.db_path)

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        # Also clean up any WAL/SHM sidecar files that SQLite may have created.
        for ext in ("-wal", "-shm"):
            try:
                os.unlink(self.db_path + ext)
            except OSError:
                pass

    # ── add_schedule ────────────────────────────────────────────────────────

    def test_add_schedule_interval_returns_id(self):
        sid = self.manager.add_schedule("user1", 111, 999, "ping", "30m")
        self.assertEqual(len(sid), 8)

    def test_add_schedule_interval_writes_row(self):
        self.manager.add_schedule("user1", 111, 999, "ping", "30m", "my task")
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT * FROM schedules WHERE user_id='user1'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], "30m")   # schedule_expr column
        self.assertEqual(rows[0][6], "my task")  # description column

    def test_add_schedule_cron_writes_row(self):
        self.manager.add_schedule("user2", 222, 888, "report", "0 9 * * *")
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT schedule_expr FROM schedules WHERE user_id='user2'").fetchall()
        self.assertEqual(rows[0][0], "0 9 * * *")

    def test_add_schedule_invalid_expr_raises_and_no_row(self):
        with self.assertRaises(ValueError):
            self.manager.add_schedule("user1", 111, 999, "ping", "bad_expr")
        with sqlite3.connect(self.db_path) as c:
            count = c.execute("SELECT COUNT(*) FROM schedules").fetchone()[0]
        self.assertEqual(count, 0)

    def test_add_schedule_registers_job_with_apscheduler(self):
        self.manager.add_schedule("user1", 111, 999, "ping", "1h")
        self.manager.scheduler.add_job.assert_called_once()

    def _manager_with_cap(self, cap: int):
        """Reload scheduler with both SCHEDULE_DB (so we stay on the temp DB)
        and MAX_SCHEDULES_PER_USER patched. Returns a fresh manager with the
        APScheduler stubbed out.
        """
        import fritz_utils
        import importlib
        with patch.object(fritz_utils, "SCHEDULE_DB", self.db_path), \
             patch.object(fritz_utils, "MAX_SCHEDULES_PER_USER", cap):
            importlib.reload(self.sched_mod)
            mgr = self.sched_mod.ScheduleManager(MagicMock())
            mgr.scheduler = MagicMock()
        return mgr

    def test_add_schedule_enforces_per_user_cap(self):
        # Phase 7c: MAX_SCHEDULES_PER_USER caps abuse. Patch low to keep the test
        # fast; the production default is 10.
        mgr = self._manager_with_cap(2)
        mgr.add_schedule("user1", 111, 999, "p1", "1h")
        mgr.add_schedule("user1", 111, 999, "p2", "1h")
        with self.assertRaises(ValueError) as ctx:
            mgr.add_schedule("user1", 111, 999, "p3", "1h")
        self.assertIn("max", str(ctx.exception).lower())

    def test_per_user_cap_is_per_user_not_global(self):
        # user1 hitting their cap doesn't stop user2 from adding their own.
        mgr = self._manager_with_cap(1)
        mgr.add_schedule("user1", 111, 999, "p1", "1h")
        with self.assertRaises(ValueError):
            mgr.add_schedule("user1", 111, 999, "p2", "1h")
        # user2 still gets their first slot.
        mgr.add_schedule("user2", 222, 999, "p3", "1h")

    # ── list_all_schedules ──────────────────────────────────────────────────

    def test_list_all_schedules_returns_every_user(self):
        self.manager.add_schedule("alice", 111, 999, "p1", "1h")
        self.manager.add_schedule("bob", 222, 999, "p2", "2h")
        rows = self.manager.list_all_schedules()
        users = {r["user_id"] for r in rows}
        self.assertEqual(users, {"alice", "bob"})

    def test_list_all_schedules_includes_user_id_field(self):
        self.manager.add_schedule("alice", 111, 999, "p1", "1h")
        rows = self.manager.list_all_schedules()
        self.assertEqual(rows[0]["user_id"], "alice")
        for key in ("id", "prompt", "schedule", "created", "description"):
            self.assertIn(key, rows[0])

    # ── remove_schedule ─────────────────────────────────────────────────────

    def test_remove_schedule_success_returns_true(self):
        sid = self.manager.add_schedule("user1", 111, 999, "ping", "1h")
        result = self.manager.remove_schedule(sid, "user1")
        self.assertTrue(result)

    def test_remove_schedule_deletes_row(self):
        sid = self.manager.add_schedule("user1", 111, 999, "ping", "1h")
        self.manager.remove_schedule(sid, "user1")
        with sqlite3.connect(self.db_path) as c:
            count = c.execute("SELECT COUNT(*) FROM schedules").fetchone()[0]
        self.assertEqual(count, 0)

    def test_remove_schedule_not_found_returns_false(self):
        result = self.manager.remove_schedule("nonexistent", "user1")
        self.assertFalse(result)

    def test_remove_schedule_wrong_user_raises_permission_error(self):
        sid = self.manager.add_schedule("user1", 111, 999, "ping", "1h")
        with self.assertRaises(PermissionError):
            self.manager.remove_schedule(sid, "other_user")

    # ── list_schedules ──────────────────────────────────────────────────────

    def test_list_schedules_filters_by_user(self):
        self.manager.add_schedule("alice", 111, 999, "ping", "1h", "Alice's task")
        self.manager.add_schedule("bob", 222, 888, "pong", "2h", "Bob's task")
        result = self.manager.list_schedules("alice")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "Alice's task")

    def test_list_schedules_returns_expected_keys(self):
        self.manager.add_schedule("alice", 111, 999, "ping", "1h", "desc")
        result = self.manager.list_schedules("alice")
        self.assertIn("id", result[0])
        self.assertIn("prompt", result[0])
        self.assertIn("schedule", result[0])
        self.assertIn("created", result[0])
        self.assertIn("description", result[0])

    def test_list_schedules_empty_for_unknown_user(self):
        result = self.manager.list_schedules("nobody")
        self.assertEqual(result, [])

    # ── start ───────────────────────────────────────────────────────────────

    def test_start_restores_all_schedules(self):
        self.manager.add_schedule("user1", 111, 999, "ping", "30m")
        self.manager.add_schedule("user2", 222, 888, "pong", "1h")
        # Reset the mock call count before calling start().
        self.manager.scheduler.reset_mock()
        self.manager.start()
        # add_job called once per restored schedule.
        self.assertEqual(self.manager.scheduler.add_job.call_count, 2)

    def test_start_skips_corrupt_schedule_gracefully(self):
        # Insert a row with an invalid schedule_expr directly.
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO schedules VALUES (?,?,?,?,?,?,?,?)",
                ("badid123", "user1", 111, 999, "ping", "INVALID", None, "2024-01-01T00:00:00+00:00"),
            )
        self.manager.scheduler.reset_mock()
        # Should not raise — bad rows are warned and skipped.
        self.manager.start()
        self.manager.scheduler.add_job.assert_not_called()


class TestParseTrigger(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.manager, _ = _make_manager(self.db_path)

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        for ext in ("-wal", "-shm"):
            try:
                os.unlink(self.db_path + ext)
            except OSError:
                pass

    def test_parse_trigger_minutes(self):
        trigger = self.manager._parse_trigger("30m")
        self.assertIsInstance(trigger, IntervalTrigger)

    def test_parse_trigger_hours(self):
        trigger = self.manager._parse_trigger("2h")
        self.assertIsInstance(trigger, IntervalTrigger)

    def test_parse_trigger_days(self):
        trigger = self.manager._parse_trigger("1d")
        self.assertIsInstance(trigger, IntervalTrigger)

    def test_parse_trigger_cron(self):
        trigger = self.manager._parse_trigger("0 9 * * *")
        self.assertIsInstance(trigger, CronTrigger)

    def test_parse_trigger_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager._parse_trigger("not_valid")

    def test_parse_trigger_partial_cron_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager._parse_trigger("0 9 * *")  # only 4 parts


if __name__ == "__main__":
    unittest.main()
