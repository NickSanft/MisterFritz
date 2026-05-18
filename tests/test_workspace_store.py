"""
Tests for workspace_store.py — the SQLite-backed per-user workspace registry.

Uses a temp dir for both the DB and the workspaces parent so tests don't
pollute the real fritz.db.
"""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestWorkspaceStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db_path = self.tmp / "test_fritz.db"
        self.workspaces_root = self.tmp / "workspaces"

        # Patch the env vars BEFORE importing/reloading workspace_store so
        # module-level constants see the test values.
        self._env_patcher = patch.dict(os.environ, {
            "SCHEDULE_DB": str(self.db_path),
            "DB_NAME": str(self.db_path),
            "WORKSPACES_ROOT": str(self.workspaces_root),
        })
        self._env_patcher.start()

        import fritz_utils
        importlib.reload(fritz_utils)
        import workspace_store
        importlib.reload(workspace_store)
        self.store = workspace_store

    def tearDown(self):
        self._env_patcher.stop()

    def test_get_returns_none_when_no_workspace_registered(self):
        self.assertIsNone(self.store.get("alice"))

    def test_get_returns_none_for_empty_user_id(self):
        self.assertIsNone(self.store.get(""))

    def test_set_path_persists_workspace(self):
        target = self.tmp / "alice_dir"
        target.mkdir()
        self.store.set_path("alice", str(target))
        self.assertEqual(self.store.get("alice"), str(target))

    def test_set_path_rejects_nonexistent_directory(self):
        with self.assertRaises(ValueError):
            self.store.set_path("alice", str(self.tmp / "does_not_exist"))

    def test_set_path_rejects_empty_user_id(self):
        with self.assertRaises(ValueError):
            self.store.set_path("", str(self.tmp))

    def test_set_path_upserts_existing_user(self):
        first = self.tmp / "first"
        first.mkdir()
        second = self.tmp / "second"
        second.mkdir()
        self.store.set_path("alice", str(first))
        self.store.set_path("alice", str(second))
        self.assertEqual(self.store.get("alice"), str(second))

    def test_enable_sandboxed_creates_dir_and_registers(self):
        path = self.store.enable_sandboxed("alice")
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(self.store.get("alice"), path)
        # Path should be under WORKSPACES_ROOT/<sanitised-user-id>/
        self.assertIn("alice", path)

    def test_enable_sandboxed_sanitises_dangerous_user_id(self):
        # A user_id like "../escape" must not produce a path outside WORKSPACES_ROOT.
        path = self.store.enable_sandboxed("../escape")
        resolved = Path(path).resolve()
        root_resolved = self.workspaces_root.resolve()
        # The created directory must live inside WORKSPACES_ROOT.
        self.assertTrue(
            str(resolved).startswith(str(root_resolved)),
            f"{resolved} is not inside {root_resolved}",
        )

    def test_enable_sandboxed_is_idempotent(self):
        first = self.store.enable_sandboxed("alice")
        second = self.store.enable_sandboxed("alice")
        self.assertEqual(first, second)

    def test_remove_drops_row_returns_true(self):
        self.store.enable_sandboxed("alice")
        self.assertTrue(self.store.remove("alice"))
        self.assertIsNone(self.store.get("alice"))

    def test_remove_returns_false_for_unknown_user(self):
        self.assertFalse(self.store.remove("unknown"))

    def test_remove_does_not_delete_files_on_disk(self):
        path = self.store.enable_sandboxed("alice")
        marker = Path(path) / "important.txt"
        marker.write_text("keep me", encoding="utf-8")
        self.store.remove("alice")
        # Workspace row is gone, but the files survive — user can re-enable later.
        self.assertTrue(marker.exists())

    def test_list_all_returns_every_registered_user(self):
        self.store.enable_sandboxed("alice")
        self.store.enable_sandboxed("bob")
        rows = self.store.list_all()
        users = {row["user_id"] for row in rows}
        self.assertEqual(users, {"alice", "bob"})
        for row in rows:
            self.assertIn("path", row)
            self.assertIn("enabled_at", row)

    def test_users_get_separate_workspace_dirs(self):
        alice_path = self.store.enable_sandboxed("alice")
        bob_path = self.store.enable_sandboxed("bob")
        self.assertNotEqual(alice_path, bob_path)


if __name__ == "__main__":
    unittest.main()
