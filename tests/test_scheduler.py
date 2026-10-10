import os
import sqlite3
import tempfile
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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
        # 2 user schedules + 1 internal WAL checkpoint job = 3 add_job calls.
        self.assertEqual(self.manager.scheduler.add_job.call_count, 3)
        # User-supplied IDs are restored.
        ids_passed = {c.kwargs.get("id") for c in self.manager.scheduler.add_job.call_args_list}
        self.assertIn("_internal_wal_checkpoint", ids_passed)

    def test_start_skips_corrupt_schedule_gracefully(self):
        # Insert a row with an invalid schedule_expr directly.
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO schedules VALUES (?,?,?,?,?,?,?,?)",
                ("badid123", "user1", 111, 999, "ping", "INVALID", None, "2024-01-01T00:00:00+00:00"),
            )
        self.manager.scheduler.reset_mock()
        # Should not raise — bad rows are warned and skipped. The internal
        # WAL checkpoint job still registers (it's not from the DB).
        self.manager.start()
        ids_passed = {c.kwargs.get("id") for c in self.manager.scheduler.add_job.call_args_list}
        self.assertEqual(ids_passed, {"_internal_wal_checkpoint"})

    # ── WAL checkpoint ──────────────────────────────────────────────────────

    def test_wal_checkpoint_runs_pragma_against_schedule_db(self):
        # _wal_checkpoint just executes a PRAGMA on the configured DB.
        # We can't easily inspect the result of TRUNCATE on a tiny test DB,
        # but the method must not raise.
        self.manager.add_schedule("user1", 111, 999, "ping", "1h")
        self.manager._wal_checkpoint()  # must not raise

    def test_wal_checkpoint_internal_job_is_not_in_list_all(self):
        # The internal job lives only in APScheduler, not the schedules table.
        # list_all_schedules() queries the DB, so it should never surface
        # the internal job's ID.
        self.manager.start()  # registers the internal job
        all_schedules = self.manager.list_all_schedules()
        ids = {s["id"] for s in all_schedules}
        self.assertNotIn("_internal_wal_checkpoint", ids)


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


class TestAOneShotCanBeCancelledAndForgotten(unittest.TestCase):
    """`schedule_once` deliberately does not persist - a reminder that missed
    its moment should not fire on the next boot - so the APScheduler job store
    is a one-shot's only record.

    `remove_schedule` only ever read the schedules table, so it returned False
    for every id `schedule_message` handed out: Fritz confirmed a reminder and
    then denied it existed. `/forget schedules` missed them for the same
    reason, leaving the prompt booked and still due to fire.

    These tests use a real (unstarted) AsyncIOScheduler rather than the
    MagicMock the rest of this file uses, because the behaviour under test IS
    the job store: pending jobs are visible to get_job/get_jobs/remove_job,
    and the owner is args[1] exactly as schedule_once writes it.
    """

    OWNER = "discord-111"
    OTHER = "discord-222"

    def setUp(self):
        import os
        import tempfile
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.manager, self.sched_mod = _make_manager(self.db_path)
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        self.manager.scheduler = AsyncIOScheduler()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import os
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass

    def _book(self, owner=OWNER, minutes=10, prompt="stand up"):
        return self.manager.schedule_once(4242, owner, minutes, prompt)

    def test_the_id_fritz_hands_out_can_be_cancelled(self):
        sid = self._book()
        self.assertTrue(self.manager.remove_schedule(sid, self.OWNER))
        self.assertIsNone(self.manager.scheduler.get_job(sid))

    def test_cancelling_it_twice_reports_the_second_as_absent(self):
        sid = self._book()
        self.manager.remove_schedule(sid, self.OWNER)
        self.assertFalse(self.manager.remove_schedule(sid, self.OWNER))

    def test_an_unknown_id_is_still_simply_absent(self):
        self.assertFalse(self.manager.remove_schedule("nosuchid", self.OWNER))

    def test_one_person_cannot_cancel_anothers(self):
        sid = self._book(owner=self.OTHER)
        with self.assertRaises(PermissionError):
            self.manager.remove_schedule(sid, self.OWNER)
        self.assertIsNotNone(self.manager.scheduler.get_job(sid),
                             "the job was removed despite the refusal")

    def test_a_job_with_no_identifiable_owner_is_not_claimed(self):
        """Reporting it as somebody else's would assert an owner there is no
        evidence for, so it is reported the way an absent id is."""
        self.manager.scheduler.add_job(
            lambda _only_arg: None, trigger=self.sched_mod.DateTrigger(
                run_date=self.sched_mod.datetime.now(self.sched_mod.timezone.utc)
                + self.sched_mod.timedelta(minutes=5)),
            args=["orphan"], id="orphan")
        self.assertFalse(self.manager.remove_schedule("orphan", self.OWNER))
        self.assertIsNotNone(self.manager.scheduler.get_job("orphan"))

    def test_forget_schedules_removes_one_shots_and_counts_them(self):
        """The privacy consequence: /forget schedules reports what it removed,
        and a one-shot used to be neither removed nor counted."""
        self._book(prompt="first")
        self._book(prompt="second")
        self.assertEqual(self.manager.remove_all_for_user(self.OWNER), 2)
        self.assertEqual(self.manager.scheduler.get_jobs(), [])

    def test_forget_schedules_counts_persisted_and_one_shot_together(self):
        persisted = self.manager.add_schedule(self.OWNER, 111, 4242, "ping", "30m")
        self._book()
        self.assertEqual(self.manager.remove_all_for_user(self.OWNER), 2)
        self.assertEqual(self.manager.list_schedules(self.OWNER), [])
        self.assertIsNone(self.manager.scheduler.get_job(persisted))
        self.assertEqual(self.manager.scheduler.get_jobs(), [])

    def test_forget_leaves_other_peoples_reminders_alone(self):
        mine = self._book()
        theirs = self._book(owner=self.OTHER)
        self.assertEqual(self.manager.remove_all_for_user(self.OWNER), 1)
        self.assertIsNone(self.manager.scheduler.get_job(mine))
        self.assertIsNotNone(self.manager.scheduler.get_job(theirs))

    def test_a_persisted_job_is_not_counted_twice(self):
        """remove_all_for_user detaches the persisted jobs before sweeping the
        store, so the sweep must not find them again."""
        self.manager.add_schedule(self.OWNER, 111, 4242, "ping", "30m")
        self.assertEqual(self.manager.remove_all_for_user(self.OWNER), 1)


if __name__ == "__main__":
    unittest.main()


class RunTaskCase(unittest.IsolatedAsyncioTestCase):
    """_run_task is what a schedule actually DOES, and almost none of it ran.

    Its recovery paths are the ones that matter: a reminder booked months ago
    fires into whatever the channel is now, which may be gone, or private, or
    hand back something too long to post. Each of those used to be reasoned
    about rather than executed.
    """

    def _manager(self):
        import scheduler
        manager = scheduler.ScheduleManager.__new__(scheduler.ScheduleManager)
        manager.bot = MagicMock()
        manager.scheduler = MagicMock()
        return manager, scheduler

    @staticmethod
    def _channel():
        channel = MagicMock()
        status = MagicMock()
        status.edit = AsyncMock()
        channel.send = AsyncMock(return_value=status)
        return channel, status

    @staticmethod
    def _discord_error(cls, status: int):
        response = MagicMock()
        response.status = status
        return cls(response, {"code": 0, "message": "no"})

    async def _run(self, manager, scheduler_module, *, answer="the reminder",
                   raises=None):
        ask = MagicMock(side_effect=raises) if raises else MagicMock(
            return_value={"text": answer, "image_paths": []})
        with patch.dict(sys.modules,
                        {"mister_fritz": MagicMock(ask_stuff=ask)}):
            with patch.object(scheduler_module.identity_store, "display_name",
                              return_value="Nick"):
                await manager._run_task("sid1", "discord-1", 4242, "stand up")
        return ask


class TestAReminderFiresIntoWhateverTheChannelIsNow(RunTaskCase):
    async def test_a_channel_the_cache_has_lost_is_fetched(self):
        """get_channel only sees what is cached, and a bot restarted since the
        schedule was made has nothing cached."""
        manager, module = self._manager()
        channel, status = self._channel()
        manager.bot.get_channel.return_value = None
        manager.bot.fetch_channel = AsyncMock(return_value=channel)
        await self._run(manager, module)
        manager.bot.fetch_channel.assert_awaited_once_with(4242)
        status.edit.assert_awaited_once()
        self.assertIn("the reminder", status.edit.await_args.kwargs["content"])

    async def test_a_deleted_channel_is_skipped_not_raised(self):
        """An exception here would escape into APScheduler's job runner, which
        logs it and moves on — so the reminder is lost either way, but the
        skip says which channel in the log."""
        import discord
        manager, module = self._manager()
        manager.bot.get_channel.return_value = None
        manager.bot.fetch_channel = AsyncMock(
            side_effect=self._discord_error(discord.NotFound, 404))
        with self.assertLogs("scheduler", level="WARNING") as caught:
            ask = await self._run(manager, module)
        ask.assert_not_called()
        self.assertIn("4242", "; ".join(caught.output))

    async def test_a_channel_it_may_no_longer_read_is_skipped(self):
        import discord
        manager, module = self._manager()
        manager.bot.get_channel.return_value = None
        manager.bot.fetch_channel = AsyncMock(
            side_effect=self._discord_error(discord.Forbidden, 403))
        with self.assertLogs("scheduler", level="WARNING"):
            ask = await self._run(manager, module)
        ask.assert_not_called()

    async def test_a_turn_that_fails_is_reported_in_the_channel(self):
        """Rather than vanishing: somebody asked to be reminded, and silence
        is indistinguishable from the reminder never having been set."""
        manager, module = self._manager()
        channel, status = self._channel()
        manager.bot.get_channel.return_value = channel
        with self.assertLogs("scheduler", level="ERROR"):
            await self._run(manager, module, raises=RuntimeError("ollama is down"))
        said = status.edit.await_args.kwargs["content"]
        self.assertIn("failed", said)
        self.assertIn("ollama is down", said)

    async def test_an_empty_answer_still_posts_something(self):
        manager, module = self._manager()
        channel, status = self._channel()
        manager.bot.get_channel.return_value = channel
        await self._run(manager, module, answer="")
        self.assertIn("No response generated",
                      status.edit.await_args.kwargs["content"])

    async def test_a_long_answer_is_chunked_across_follow_up_sends(self):
        """Discord refuses anything over 2000 characters, and the placeholder
        can only hold the first chunk."""
        manager, module = self._manager()
        channel, status = self._channel()
        manager.bot.get_channel.return_value = channel
        await self._run(manager, module, answer="y" * 4500)
        first = status.edit.await_args.kwargs["content"]
        self.assertEqual(len(first), 2000)
        # The placeholder send is call one; the rest are the extra chunks.
        extra = [c.args[0] for c in channel.send.await_args_list[1:]]
        self.assertEqual(len(first) + sum(len(c) for c in extra), 4500)


class TestRemovalSurvivesAJobThatIsAlreadyGone(unittest.TestCase):
    """The schedules table and APScheduler's job store can disagree — a
    restart that failed partway, or a job already fired. Removal has to be
    about the row, with the job detached best-effort."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.manager, self.sched_mod = _make_manager(self.db_path)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass

    def test_remove_schedule_still_reports_the_row_it_deleted(self):
        sid = self.manager.add_schedule("discord-1", 111, 4242, "ping", "30m")
        self.manager.scheduler.remove_job.side_effect = Exception("no such job")
        self.assertTrue(self.manager.remove_schedule(sid, "discord-1"))
        self.assertEqual(self.manager.list_schedules("discord-1"), [])

    def test_remove_all_still_counts_the_rows_it_deleted(self):
        self.manager.add_schedule("discord-1", 111, 4242, "a", "30m")
        self.manager.add_schedule("discord-1", 111, 4242, "b", "30m")
        self.manager.scheduler.remove_job.side_effect = Exception("no such job")
        self.manager.scheduler.get_jobs.return_value = []
        self.assertEqual(self.manager.remove_all_for_user("discord-1"), 2)

    def test_a_one_shot_that_vanishes_mid_sweep_is_not_counted(self):
        """The count is what /forget reports, so it must not include a job the
        store refused to remove."""
        job = MagicMock()
        job.id = "oneshot1"
        job.args = ["oneshot1", "discord-1", 4242, "later"]
        self.manager.scheduler.get_jobs.return_value = [job]
        self.manager.scheduler.remove_job.side_effect = Exception("gone")
        self.assertEqual(self.manager.remove_all_for_user("discord-1"), 0)

    def test_a_delay_under_the_minimum_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.manager.schedule_once(4242, "discord-1", 0, "now please")
        self.assertIn("at least", str(caught.exception))


class TestTheInternalUpkeepJob(unittest.TestCase):
    """The WAL checkpoint and stop() had never run. The checkpoint is the thing
    keeping fritz.db's write-ahead log from growing into hundreds of MB."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.manager, self.sched_mod = _make_manager(self.db_path)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass

    def _wal_size(self) -> int:
        path = self.db_path + "-wal"
        return os.path.getsize(path) if os.path.exists(path) else 0

    def test_the_checkpoint_truncates_the_write_ahead_log(self):
        """Asserted on the file, not on the log line. The whole point of this
        job is that the WAL stops growing: under WAL mode with heavy writes it
        reaches hundreds of MB before SQLite checkpoints on its own, and a log
        line saying "complete" is equally available to a PRAGMA that truncates
        nothing.

        The writer connection stays open, because the WAL is what a closed
        connection has already checkpointed away.
        """
        writer = sqlite3.connect(self.db_path)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE IF NOT EXISTS t (x TEXT)")
        writer.executemany("INSERT INTO t VALUES (?)", [("y" * 400,)] * 2000)
        writer.commit()
        self.assertGreater(self._wal_size(), 100_000,
                           "no write-ahead log to checkpoint")

        with self.assertLogs("scheduler", level="INFO") as caught:
            self.manager._wal_checkpoint()

        self.assertEqual(self._wal_size(), 0,
                         "the write-ahead log was not truncated")
        self.assertIn("WAL checkpoint", "; ".join(caught.output))

    def test_a_checkpoint_failure_is_logged_not_raised(self):
        """It runs from inside APScheduler; raising would only land in its job
        log, and the next run would try again anyway."""
        with patch.object(self.sched_mod.sqlite3, "connect",
                          side_effect=RuntimeError("disk gone")):
            with self.assertLogs("scheduler", level="WARNING") as caught:
                self.manager._wal_checkpoint()      # must not raise
        self.assertIn("disk gone", "; ".join(caught.output))

    def test_a_scheduler_that_refuses_the_upkeep_job_still_starts(self):
        """The checkpoint is upkeep. Failing to register it must not stop the
        schedules themselves from being restored."""
        self.manager.scheduler.add_job.side_effect = [
            Exception("refused"), None]
        with self.assertLogs("scheduler", level="WARNING"):
            self.manager.start()
        self.manager.scheduler.start.assert_called_once()

    def test_stop_shuts_the_scheduler_down_without_waiting(self):
        """Called when on_ready replaces a manager. Waiting would block the
        event loop on whatever job happened to be running."""
        self.manager.stop()
        self.manager.scheduler.shutdown.assert_called_once_with(wait=False)
