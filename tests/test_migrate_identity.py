"""Tests for the one-time identity migration.

This script runs once, against the owner's live data, with no undo other than
--reverse. So the tests care most about the things that would be unrecoverable:
mutating during a dry run, writing the mapping file after the mutation instead
of before, and half-migrating on an unmapped key.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import migrate_identity


def _build_db(path, *, with_workspaces=False):
    """A fritz.db shaped like the real one — which notably has NO workspaces
    table, because workspace_store._init_db only runs on first /workspace use.
    """
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE schedules (id TEXT PRIMARY KEY, user_id TEXT, prompt TEXT)")
        c.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
        c.execute("CREATE TABLE writes (thread_id TEXT, blob TEXT)")
        c.execute("CREATE TABLE store (namespace TEXT, key TEXT)")
        if with_workspaces:
            c.execute("CREATE TABLE workspaces (user_id TEXT PRIMARY KEY, path TEXT)")
            c.execute("INSERT INTO workspaces VALUES ('divora', '/tmp/ws')")
        c.executemany("INSERT INTO schedules VALUES (?,?,?)",
                      [("s1", "divora", "p1"), ("s2", "someone_else", "p2")])
        c.executemany("INSERT INTO checkpoints VALUES (?,?)",
                      [("divora", "a"), ("divora#999", "b"), ("someone_else", "c")])
        c.execute("INSERT INTO writes VALUES ('divora', 'w')")
        c.execute("INSERT INTO store VALUES ('divora', 'k')")
        c.commit()


class _MigrationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = str(self.tmp / "fritz.db")
        _build_db(self.db)
        self.cwd = os.getcwd()
        os.chdir(self.tmp)          # mapping files land here, not in the repo

    def tearDown(self):
        os.chdir(self.cwd)

    def _run(self, *argv):
        return migrate_identity.main([*argv, "--db", self.db, "--skip-chroma"])

    def _keys(self, table, column):
        with sqlite3.connect(self.db) as c:
            return sorted(r[0] for r in c.execute(f"SELECT {column} FROM {table}"))


class TestDryRun(_MigrationTestCase):
    def test_reports_and_mutates_nothing(self):
        before = self._keys("schedules", "user_id")
        rc = self._run("--dry-run", "--map", "divora=discord-1")
        self.assertEqual(rc, 0)
        self.assertEqual(self._keys("schedules", "user_id"), before)

    def test_dry_run_is_the_default(self):
        # No --apply and no --dry-run must still be safe.
        rc = self._run("--map", "divora=discord-1")
        self.assertEqual(rc, 0)
        self.assertIn("divora", self._keys("schedules", "user_id"))

    def test_dry_run_writes_no_mapping_file(self):
        self._run("--dry-run", "--map", "divora=discord-1")
        self.assertEqual(list(self.tmp.glob("identity_migration_*.json")), [])

    def test_dry_run_tolerates_an_unmapped_key(self):
        # Reporting is the whole point of a dry run; it must not refuse.
        self.assertEqual(self._run("--dry-run"), 0)

    def test_missing_workspaces_table_is_not_an_error(self):
        # The live fritz.db genuinely has no workspaces table.
        with sqlite3.connect(self.db) as c:
            self.assertIsNone(c.execute(
                "SELECT 1 FROM sqlite_master WHERE name='workspaces'").fetchone())
        self.assertEqual(self._run("--dry-run", "--map", "divora=discord-1"), 0)

    def test_survey_collapses_channel_suffixes(self):
        # The operator maps identities, not one entry per channel.
        found = migrate_identity.survey(self.db)
        self.assertIn("divora", found["checkpoints"])
        self.assertNotIn("divora#999", found["checkpoints"])


class TestApply(_MigrationTestCase):
    MAP = ("--map", "divora=discord-1", "--map", "someone_else=discord-2")

    def test_rewrites_every_store(self):
        rc = self._run("--apply", *self.MAP)
        self.assertEqual(rc, 0)
        self.assertEqual(self._keys("schedules", "user_id"), ["discord-1", "discord-2"])
        self.assertEqual(self._keys("store", "namespace"), ["discord-1"])
        self.assertEqual(self._keys("writes", "thread_id"), ["discord-1"])

    def test_channel_suffix_survives_the_rewrite(self):
        self._run("--apply", *self.MAP)
        self.assertEqual(
            self._keys("checkpoints", "thread_id"),
            ["discord-1", "discord-1#999", "discord-2"],
        )

    def test_is_idempotent(self):
        self._run("--apply", *self.MAP)
        after_first = self._keys("checkpoints", "thread_id")
        self._run("--apply", *self.MAP)
        self.assertEqual(self._keys("checkpoints", "thread_id"), after_first)

    def test_workspaces_table_is_migrated_when_present(self):
        self.db = str(self.tmp / "with_ws.db")
        _build_db(self.db, with_workspaces=True)
        self._run("--apply", *self.MAP)
        self.assertEqual(self._keys("workspaces", "user_id"), ["discord-1"])

    def test_mapping_file_is_written(self):
        self._run("--apply", *self.MAP)
        files = list(self.tmp.glob("identity_migration_*.json"))
        self.assertEqual(len(files), 1)
        saved = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(saved["mapping"]["divora"], "discord-1")

    def test_mapping_file_is_written_before_mutating(self):
        # If the run dies partway through, --reverse must still be possible.
        # Proven by making the rewrite blow up and checking the file survives.
        original = migrate_identity.rewrite_sqlite
        migrate_identity.rewrite_sqlite = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("died partway"))
        try:
            with self.assertRaises(RuntimeError):
                self._run("--apply", *self.MAP)
        finally:
            migrate_identity.rewrite_sqlite = original
        self.assertEqual(len(list(self.tmp.glob("identity_migration_*.json"))), 1)

    def test_unmapped_key_refuses_to_half_migrate(self):
        rc = self._run("--apply", "--map", "divora=discord-1")   # someone_else unmapped
        self.assertEqual(rc, 2)
        self.assertIn("divora", self._keys("schedules", "user_id"))   # untouched

    def test_identity_mapping_is_a_no_op(self):
        rc = self._run("--apply", "--map", "divora=divora",
                       "--map", "someone_else=someone_else")
        self.assertEqual(rc, 0)
        self.assertEqual(self._keys("schedules", "user_id"), ["divora", "someone_else"])

    def test_map_file_is_accepted(self):
        path = self.tmp / "m.json"
        path.write_text(json.dumps({"divora": "discord-1",
                                    "someone_else": "discord-2"}), encoding="utf-8")
        self.assertEqual(self._run("--apply", "--map-file", str(path)), 0)
        self.assertEqual(self._keys("store", "namespace"), ["discord-1"])


class TestReverse(_MigrationTestCase):
    MAP = ("--map", "divora=discord-1", "--map", "someone_else=discord-2")

    def test_restores_the_original_state_exactly(self):
        before = {t: self._keys(t, c) for t, c in
                  (("schedules", "user_id"), ("checkpoints", "thread_id"),
                   ("writes", "thread_id"), ("store", "namespace"))}
        self._run("--apply", *self.MAP)
        mapping_file = str(next(self.tmp.glob("identity_migration_*.json")))
        rc = self._run("--apply", "--reverse", mapping_file)
        self.assertEqual(rc, 0)
        after = {t: self._keys(t, c) for t, c in
                 (("schedules", "user_id"), ("checkpoints", "thread_id"),
                  ("writes", "thread_id"), ("store", "namespace"))}
        self.assertEqual(after, before)

    def test_reverse_does_not_write_a_new_mapping_file(self):
        self._run("--apply", *self.MAP)
        mapping_file = str(next(self.tmp.glob("identity_migration_*.json")))
        self._run("--apply", "--reverse", mapping_file)
        self.assertEqual(len(list(self.tmp.glob("identity_migration_*.json"))), 1)


class TestArgumentValidation(_MigrationTestCase):
    def test_malformed_map_is_rejected(self):
        self.assertEqual(self._run("--apply", "--map", "no-equals-sign"), 2)

    def test_missing_database_is_rejected(self):
        rc = migrate_identity.main(
            ["--dry-run", "--db", str(self.tmp / "nope.db"), "--skip-chroma"])
        self.assertEqual(rc, 2)


class TestChroma(unittest.TestCase):
    """Metadata is rewritten in place; nothing is re-embedded. That is what
    lets the migration run with Ollama stopped."""

    def setUp(self):
        try:
            import chromadb                        # noqa: F401
        except ImportError:
            self.skipTest("chromadb not installed")
        self.tmp = Path(tempfile.mkdtemp())

    def _collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.tmp))
        return client, client.get_or_create_collection(
            migrate_identity.CHROMA_COLLECTION)

    def test_namespace_is_rewritten_and_the_vector_is_untouched(self):
        client, col = self._collection()
        col.add(ids=["mem1"], embeddings=[[0.1, 0.2, 0.3]],
                documents=["a memory"], metadatas=[{"namespace": "divora"}])
        before = col.get(ids=["mem1"], include=["embeddings"])["embeddings"][0]

        n = migrate_identity.rewrite_chroma(str(self.tmp), {"divora": "discord-1"})
        self.assertEqual(n, 1)

        _client, col = self._collection()
        got = col.get(ids=["mem1"], include=["metadatas", "embeddings"])
        self.assertEqual(got["metadatas"][0]["namespace"], "discord-1")
        self.assertEqual(list(got["embeddings"][0]), list(before))

    def test_profile_document_id_is_renamed_carrying_its_embedding(self):
        # The only identity-derived document id in the store.
        _client, col = self._collection()
        col.add(ids=["profile_divora"], embeddings=[[0.4, 0.5, 0.6]],
                documents=["profile blob"], metadatas=[{"namespace": "divora"}])

        migrate_identity.rewrite_chroma(str(self.tmp), {"divora": "discord-1"})

        _client, col = self._collection()
        got = col.get(ids=["profile_discord-1"], include=["metadatas", "embeddings"])
        self.assertEqual(got["ids"], ["profile_discord-1"])
        self.assertEqual(got["metadatas"][0]["namespace"], "discord-1")
        self.assertEqual([round(v, 4) for v in got["embeddings"][0]], [0.4, 0.5, 0.6])
        self.assertEqual(col.get(ids=["profile_divora"])["ids"], [])

    def test_unmapped_namespace_is_left_alone(self):
        _client, col = self._collection()
        col.add(ids=["mem1"], embeddings=[[0.1, 0.2, 0.3]],
                documents=["x"], metadatas=[{"namespace": "stranger"}])
        self.assertEqual(
            migrate_identity.rewrite_chroma(str(self.tmp), {"divora": "discord-1"}), 0)
        _client, col = self._collection()
        self.assertEqual(
            col.get(ids=["mem1"], include=["metadatas"])["metadatas"][0]["namespace"],
            "stranger")

    def test_missing_store_is_not_an_error(self):
        self.assertEqual(
            migrate_identity.rewrite_chroma(str(self.tmp / "nope"), {"a": "b"}), 0)
        self.assertEqual(migrate_identity.survey_chroma(str(self.tmp / "nope")), {})


if __name__ == "__main__":
    unittest.main()
