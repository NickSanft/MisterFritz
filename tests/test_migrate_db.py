"""Tests for the one-time schedules.db + chat_history.db -> fritz.db migration.

This script had never been executed by the suite, and it had made the upgrade
path it documents unusable. It pre-created LangGraph's `writes` table with a
column named `blob BLOB NOT NULL`; the pinned checkpointer reads and writes a
nullable `value`. `CREATE TABLE IF NOT EXISTS` no-ops on an existing table, so
`SqliteSaver.setup()` could not repair the drift afterwards - once this script
had run against a fresh fritz.db, every put_writes failed with "table writes
has no column named value" and the bot could not finish one agent turn, with
nothing in the error pointing back at the migration.

So the first test here is the one that matters: run the migration, then ask a
real checkpointer to write through the result. The rest cover the two things
found alongside it - a positional `SELECT *` copy that misfiled data whenever
the column sets disagreed, and a swallowed OperationalError that still ended
with "Done" and an invitation to delete the sources.
"""
import io
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fritz_utils  # noqa: E402
import migrate_db  # noqa: E402

CONFIG = {"configurable": {"thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": "c1"}}


def _saver_schema(path: str) -> None:
    """The authoritative LangGraph schema: whatever the checkpointer makes."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    with sqlite3.connect(path) as conn:
        SqliteSaver(conn).setup()


class MigrationCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.target = str(self.tmp / "fritz.db")
        self.chat = str(self.tmp / "chat_history.db")
        self.sched = str(self.tmp / "schedules.db")

    def run_migration(self, target=None, chat=None, sched=None) -> tuple[int, str, str]:
        """Drive main() the way an operator does. It reads the three paths from
        fritz_utils inside the function, so patching the module is enough."""
        out, err = io.StringIO(), io.StringIO()
        with unittest.mock.patch.multiple(
            fritz_utils,
            DB_NAME=target or self.target,
            CHAT_DB_NAME=chat or self.chat,
            SCHEDULE_DB=sched or self.sched,
        ):
            with redirect_stdout(out), redirect_stderr(err):
                code = migrate_db.main()
        return code, out.getvalue(), err.getvalue()

    def a_chat_source(self, rows=1) -> None:
        """A chat_history.db shaped the way the checkpointer shapes it."""
        _saver_schema(self.chat)
        with sqlite3.connect(self.chat) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS store (
                namespace TEXT, key TEXT, value TEXT, PRIMARY KEY (namespace, key))""")
            for i in range(rows):
                conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata) VALUES (?,?,?,?,?,?,?)",
                    (f"thread{i}", "", f"ckpt{i}", None, "msgpack", b"body", b"meta"),
                )
                conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, "
                    "idx, channel, type, value) VALUES (?,?,?,?,?,?,?,?)",
                    (f"thread{i}", "", f"ckpt{i}", f"task{i}", 0, "messages", "msgpack", b"v"),
                )
                conn.execute("INSERT INTO store (namespace, key, value) VALUES (?,?,?)",
                             (f"ns{i}", f"k{i}", '{"a": 1}'))
            conn.commit()

    def a_schedule_source(self) -> None:
        with sqlite3.connect(self.sched) as conn:
            conn.execute("""CREATE TABLE schedules (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, channel_id INTEGER NOT NULL,
                guild_id INTEGER, prompt TEXT NOT NULL, schedule_expr TEXT NOT NULL,
                description TEXT, created_at TEXT NOT NULL)""")
            conn.execute(
                "INSERT INTO schedules VALUES (?,?,?,?,?,?,?,?)",
                ("abc12345", "discord-1", 42, None, "ping", "30m", "", "2026-01-01T00:00:00Z"),
            )
            conn.commit()

    def rows(self, table: str, db: str = None) -> int:
        with sqlite3.connect(db or self.target) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestTheBotCanStillCheckpointAfterwards(MigrationCase):
    """The acceptance test. Everything else here is detail."""

    def test_a_real_checkpointer_can_write_to_the_migrated_database(self):
        self.a_chat_source()
        code, out, err = self.run_migration()
        self.assertEqual(code, 0, out + err)

        from langgraph.checkpoint.sqlite import SqliteSaver
        with sqlite3.connect(self.target) as conn:
            SqliteSaver(conn).put_writes(CONFIG, [("messages", {"a": 1})], "task-new")
        self.assertEqual(self.rows("writes"), 2)   # the copied row plus this one

    def test_the_writes_table_has_the_column_the_checkpointer_names(self):
        """The specific regression, asserted on the schema rather than through
        it, so a failure says what is wrong rather than only that it broke."""
        self.a_chat_source()
        self.run_migration()
        with sqlite3.connect(self.target) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(writes)")]
        self.assertIn("value", columns)
        self.assertNotIn("blob", columns)

    def test_a_null_write_value_is_accepted(self):
        """The old DDL also declared the column NOT NULL, which the
        checkpointer's own schema does not."""
        self.a_chat_source()
        self.run_migration()
        with sqlite3.connect(self.target) as conn:
            conn.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, "
                "idx, channel, type, value) VALUES (?,?,?,?,?,?,?,?)",
                ("t9", "", "c9", "task9", 0, "messages", None, None),
            )
            conn.commit()
        self.assertEqual(self.rows("writes"), 2)


class TestTheDataActuallyArrives(MigrationCase):
    def test_conversation_history_is_copied(self):
        self.a_chat_source(rows=3)
        code, out, err = self.run_migration()
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.rows("checkpoints"), 3)
        self.assertEqual(self.rows("writes"), 3)
        self.assertEqual(self.rows("store"), 3)

    def test_schedules_are_copied(self):
        self.a_schedule_source()
        code, out, err = self.run_migration()
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.rows("schedules"), 1)

    def test_running_it_twice_copies_nothing_the_second_time(self):
        self.a_chat_source(rows=2)
        self.run_migration()
        code, out, _ = self.run_migration()
        self.assertEqual(code, 0)
        self.assertEqual(self.rows("checkpoints"), 2)
        self.assertIn("0 row(s) copied", out)

    def test_a_missing_source_is_reported_and_not_fatal(self):
        code, out, err = self.run_migration()
        self.assertEqual(code, 0, err)
        self.assertIn("not found", out)

    def test_nothing_to_do_when_every_source_is_the_target(self):
        code, out, _ = self.run_migration(chat=self.target, sched=self.target)
        self.assertEqual(code, 0)
        self.assertIn("nothing to migrate", out)


class TestAColumnSetMismatchIsReportedNotMisfiled(MigrationCase):
    """`INSERT OR IGNORE INTO t SELECT * FROM old.t` maps by position. A source
    written by a different checkpointer version - the whole reason this script
    exists - therefore misfiled every column after the first difference, with
    no error, and the migration still said "Done"."""

    def _chat_with_an_extra_column(self) -> None:
        with sqlite3.connect(self.chat) as conn:
            conn.execute("""CREATE TABLE writes (
                thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, idx INTEGER NOT NULL,
                legacy_flag TEXT, channel TEXT NOT NULL, type TEXT, value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))""")
            conn.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                "legacy_flag, channel, type, value) VALUES (?,?,?,?,?,?,?,?,?)",
                ("t1", "", "c1", "task1", 0, "dropped", "messages", "msgpack", b"payload"),
            )
            conn.commit()

    def test_the_shared_columns_land_in_the_right_places(self):
        self._chat_with_an_extra_column()
        self.run_migration()
        with sqlite3.connect(self.target) as conn:
            row = conn.execute(
                "SELECT thread_id, channel, type, value FROM writes").fetchone()
        self.assertEqual(row[0], "t1")
        self.assertEqual(row[1], "messages")      # not "dropped", which is what
        self.assertEqual(row[2], "msgpack")       # a positional copy would put here
        self.assertEqual(bytes(row[3]), b"payload")

    def test_the_dropped_column_is_named_in_the_output(self):
        self._chat_with_an_extra_column()
        _code, out, _err = self.run_migration()
        self.assertIn("legacy_flag", out)

    def test_it_does_not_end_by_inviting_the_sources_to_be_deleted(self):
        self._chat_with_an_extra_column()
        code, out, err = self.run_migration()
        self.assertEqual(code, 1)
        self.assertIn("KEEP the old files", err)
        self.assertNotIn("Remove them once", out)


class TestAFailedCopyIsNotReportedAsDone(MigrationCase):
    """A copy that raised printed one [warn] line, was counted as zero rows,
    and then the run still finished with "Done." and advice to remove the
    sources - which is how an operator deletes data that never arrived."""

    def test_an_unreadable_table_fails_the_run(self):
        self.a_chat_source()
        with unittest.mock.patch.object(
            migrate_db, "_migrate_table",
            side_effect=lambda *a, **k: (0, True),
        ):
            code, out, err = self.run_migration()
        self.assertEqual(code, 1)
        self.assertIn("INCOMPLETE", err)
        self.assertNotIn("Done.", out)

    def test_a_source_missing_a_required_column_fails_the_run(self):
        """A real sqlite error rather than a patched one: this source predates
        created_at, which the target declares NOT NULL, so the insert raises.
        The old handler caught only OperationalError and counted the table as
        zero rows copied."""
        with sqlite3.connect(self.sched) as conn:
            conn.execute("""CREATE TABLE schedules (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, channel_id INTEGER NOT NULL,
                guild_id INTEGER, prompt TEXT NOT NULL, schedule_expr TEXT NOT NULL,
                description TEXT)""")
            conn.execute("INSERT INTO schedules VALUES (?,?,?,?,?,?,?)",
                         ("abc12345", "discord-1", 42, None, "ping", "30m", ""))
            conn.commit()
        code, out, err = self.run_migration()
        self.assertEqual(code, 1, out + err)
        self.assertIn("INCOMPLETE", err)
        self.assertIn("created_at", out)       # named as absent from the source
        self.assertEqual(self.rows("schedules"), 0)
        self.assertNotIn("Remove them once", out)

    def test_a_sqlite_error_during_the_copy_marks_the_run_incomplete(self):
        """The defensive branch, driven directly.

        A rejected row does not raise - OR IGNORE skips it, which is what the
        arrival check above is for - so this covers the other case: the copy
        itself failing. The proxy raises on the INSERT only, leaving every
        PRAGMA and COUNT real, because the branch's contract is "warn, and do
        not let main() call this a success".

        Both error classes, because the handler used to catch
        sqlite3.OperationalError alone and IntegrityError is not one - a
        constraint rejection would have escaped to main()'s outer handler and
        ended the run with a bare traceback instead of a per-table warning.
        """
        self.a_chat_source()
        self.run_migration()                      # a target with real tables

        class RaisesOnInsert:
            def __init__(self, conn, error):
                self._conn = conn
                self._error = error

            def execute(self, sql, *args):
                if sql.strip().upper().startswith("INSERT"):
                    raise self._error
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        failures = [
            sqlite3.OperationalError("database is locked"),
            sqlite3.IntegrityError("NOT NULL constraint failed: writes.value"),
        ]
        for error in failures:
            with self.subTest(error=type(error).__name__):
                out = io.StringIO()
                with sqlite3.connect(self.target) as conn:
                    conn.execute(f"ATTACH DATABASE '{self.chat}' AS old_chat")
                    try:
                        with redirect_stdout(out):
                            copied, warned = migrate_db._migrate_table(
                                RaisesOnInsert(conn, error), "old_chat", "writes")
                    finally:
                        conn.execute("DETACH DATABASE old_chat")
                self.assertEqual(copied, 0)
                self.assertTrue(warned, "a failed copy must not be reported as success")
                self.assertIn(str(error), out.getvalue())

    def test_a_clean_run_still_says_done(self):
        self.a_chat_source()
        code, out, err = self.run_migration()
        self.assertEqual(code, 0, err)
        self.assertIn("Done.", out)
        self.assertIn("Remove them once", out)


class TestTheCheckpointerOwnsItsSchema(MigrationCase):
    def test_without_the_checkpointer_installed_nothing_is_invented(self):
        """Better to skip the LangGraph tables than to hand-write a schema
        that drifts from the one the bot will use. _migrate_table's existing
        missing-target skip then does the rest."""
        with unittest.mock.patch.dict(sys.modules, {"langgraph.checkpoint.sqlite": None}):
            with sqlite3.connect(self.target) as conn:
                self.assertFalse(migrate_db._ensure_langgraph_schema(conn))
        with sqlite3.connect(self.target) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("writes", tables)

    def test_the_tables_it_creates_are_the_ones_the_bot_uses(self):
        self.run_migration()
        reference = str(self.tmp / "reference.db")
        _saver_schema(reference)
        for table in ("checkpoints", "writes"):
            with self.subTest(table=table):
                self.assertEqual(self._schema(self.target, table),
                                 self._schema(reference, table))

    @staticmethod
    def _schema(db: str, table: str) -> list:
        with sqlite3.connect(db) as conn:
            return [(row[1], row[2], row[3]) for row in
                    conn.execute(f"PRAGMA table_info({table})")]

    def test_the_store_and_schedules_tables_are_still_this_scripts_own(self):
        """Only the LangGraph tables moved to the checkpointer; these two have
        no owner but this script and the stores that read them."""
        self.run_migration()
        with sqlite3.connect(self.target) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("store", tables)
        self.assertIn("schedules", tables)


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
