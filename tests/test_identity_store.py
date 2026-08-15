"""Tests for the display-name side-channel.

Once user_id is `discord-123456789`, this table is the only thing that knows
the human's name — so it has to be reliable enough for the prompt builder and
harmless enough that a failure never breaks a conversation turn.
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import fritz_utils


class _TempDBTestCase(unittest.TestCase):
    """Point SCHEDULE_DB at a throwaway file and reload identity_store so it
    picks it up. The module caches both the path and the names."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.db_path)          # let sqlite create it
        self._patch = patch.object(fritz_utils, "SCHEDULE_DB", self.db_path)
        self._patch.start()
        import importlib
        import identity_store
        importlib.reload(identity_store)
        self.store = identity_store

    def tearDown(self):
        self._patch.stop()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


class TestRecord(_TempDBTestCase):
    def test_creates_the_table_on_a_fresh_db(self):
        self.store.record("discord-1", "Alice", "discord")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT display_name, platform FROM user_aliases WHERE user_id = ?",
                ("discord-1",),
            ).fetchone()
        self.assertEqual(row, ("Alice", "discord"))

    def test_repeat_with_the_same_name_does_not_write(self):
        # This runs on every inbound message; a gratuitous SQLite round trip
        # per conversation turn is the thing being avoided.
        self.store.record("discord-1", "Alice", "discord")
        with sqlite3.connect(self.db_path) as conn:
            before = conn.execute(
                "SELECT updated_at FROM user_aliases WHERE user_id = ?", ("discord-1",),
            ).fetchone()[0]
        with patch("identity_store.sqlite3.connect") as spy:
            self.store.record("discord-1", "Alice", "discord")
            spy.assert_not_called()
        with sqlite3.connect(self.db_path) as conn:
            after = conn.execute(
                "SELECT updated_at FROM user_aliases WHERE user_id = ?", ("discord-1",),
            ).fetchone()[0]
        self.assertEqual(before, after)

    def test_a_changed_name_upserts(self):
        self.store.record("discord-1", "Alice", "discord")
        self.store.record("discord-1", "Alice Renamed", "discord")
        self.assertEqual(self.store.display_name("discord-1"), "Alice Renamed")
        with sqlite3.connect(self.db_path) as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM user_aliases").fetchone()
        self.assertEqual(count, 1)

    def test_missing_id_or_name_is_a_no_op(self):
        self.store.record("", "Alice")
        self.store.record("discord-1", "")
        self.store.record("discord-1", None)
        with sqlite3.connect(self.db_path) as conn:
            try:
                (count,) = conn.execute("SELECT COUNT(*) FROM user_aliases").fetchone()
            except sqlite3.OperationalError:
                count = 0                 # table never created — also fine
        self.assertEqual(count, 0)

    def test_a_db_error_is_swallowed(self):
        # A conversation turn must not die because the alias table is locked.
        with patch("identity_store.sqlite3.connect", side_effect=sqlite3.OperationalError("boom")):
            self.store.record("discord-2", "Bob")      # must not raise


class TestDisplayName(_TempDBTestCase):
    def test_returns_the_recorded_name(self):
        self.store.record("discord-1", "Alice", "discord")
        self.assertEqual(self.store.display_name("discord-1"), "Alice")

    def test_unknown_id_falls_back_to_the_supplied_default(self):
        self.assertEqual(self.store.display_name("discord-9", default="Stranger"), "Stranger")

    def test_unknown_id_without_a_default_falls_back_to_the_bare_id(self):
        # Better that Fritz says "123456789" than "discord-123456789".
        self.assertEqual(self.store.display_name("discord-123456789"), "123456789")

    def test_empty_id(self):
        self.assertEqual(self.store.display_name(""), "")
        self.assertEqual(self.store.display_name("", default="x"), "x")

    def test_a_db_error_falls_back_rather_than_raising(self):
        with patch("identity_store.sqlite3.connect", side_effect=sqlite3.OperationalError("boom")):
            self.assertEqual(self.store.display_name("discord-7", default="Fallback"), "Fallback")


class TestListAllAndForget(_TempDBTestCase):
    def test_list_all_is_sorted_case_insensitively(self):
        self.store.record("discord-1", "zoe", "discord")
        self.store.record("discord-2", "Adam", "discord")
        names = [r["display_name"] for r in self.store.list_all()]
        self.assertEqual(names, ["Adam", "zoe"])

    def test_list_all_on_an_empty_db_returns_empty(self):
        self.assertEqual(self.store.list_all(), [])

    def test_forget_removes_the_row_and_the_cache_entry(self):
        self.store.record("discord-1", "Alice", "discord")
        self.assertTrue(self.store.forget("discord-1"))
        self.assertEqual(self.store.list_all(), [])
        # Not served from the in-process cache either.
        self.assertEqual(self.store.display_name("discord-1"), "1")

    def test_forget_an_unknown_id_returns_false(self):
        self.store.record("discord-1", "Alice", "discord")
        self.assertFalse(self.store.forget("discord-999"))


if __name__ == "__main__":
    unittest.main()
