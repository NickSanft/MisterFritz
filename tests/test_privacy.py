"""
Tests for the /forget and /export deletion + export helpers in privacy.py.

We exercise each forget_* function with mocked back-end stores so the tests
don't require a live Chroma or an APScheduler instance.
"""
import importlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ddgs (kept out of the import chain so privacy → fritz_utils stays fast) is
# stubbed in tests/conftest.py before any test module is collected.


class TestSanitiseThreadId(unittest.TestCase):
    def setUp(self):
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def test_canonical_ids_pass_through_untouched(self):
        # It used to strip to alphanumerics, matching what ask_stuff did. Now
        # ask_stuff does nothing to the id, so stripping here would MISS the
        # thread this is trying to delete — the same class of write/delete
        # divergence that broke /forget memories.
        self.assertEqual(self.privacy._sanitise_thread_id("discord-1234"), "discord-1234")
        self.assertEqual(self.privacy._sanitise_thread_id("web-alice_smith"), "web-alice_smith")

    def test_empty_input_returns_empty(self):
        self.assertEqual(self.privacy._sanitise_thread_id(""), "")
        self.assertEqual(self.privacy._sanitise_thread_id(None), "")


class TestForgetMemories(unittest.TestCase):
    def setUp(self):
        import privacy
        import storage
        importlib.reload(privacy)
        # Phase 13: the Chroma client is a process-wide singleton — reset
        # between tests so each one gets a clean ChromaStore() call.
        storage.reset_default_chroma_store_for_tests()
        self.privacy = privacy
        self.storage = storage

    def tearDown(self):
        self.storage.reset_default_chroma_store_for_tests()

    def test_delegates_to_chroma_store(self):
        fake_store = MagicMock()
        fake_store.delete_namespace.return_value = 7
        with patch("storage.ChromaStore", return_value=fake_store):
            result = self.privacy.forget_memories("alice")
        self.assertEqual(result, 7)
        fake_store.delete_namespace.assert_called_once_with(("alice",))

    def test_empty_user_returns_zero_without_touching_store(self):
        with patch("storage.ChromaStore") as ctor:
            result = self.privacy.forget_memories("")
        self.assertEqual(result, 0)
        ctor.assert_not_called()

    def test_swallows_store_exceptions_returns_zero(self):
        with patch("storage.ChromaStore", side_effect=RuntimeError("boom")):
            result = self.privacy.forget_memories("alice")
        self.assertEqual(result, 0)


class TestForgetMemoriesPunctuatedId(unittest.TestCase):
    """THE /forget BUG, pinned.

    forget_memories deleted the namespace for the RAW id while
    agent_tools.add_memory wrote to a STRIPPED one, because ask_stuff stripped
    the id before putting it in metadata. For any id containing punctuation the
    two never matched: /forget memories reported success and deleted nothing.
    """

    def setUp(self):
        import privacy
        import storage
        importlib.reload(privacy)
        storage.reset_default_chroma_store_for_tests()
        self.privacy = privacy
        self.storage = storage

    def tearDown(self):
        self.storage.reset_default_chroma_store_for_tests()

    def _namespace_used_by(self, fn, user_id):
        # The store is a process-wide singleton, so a second call in the same
        # test would reuse the first fake and record nothing.
        self.storage.reset_default_chroma_store_for_tests()
        fake_store = MagicMock()
        fake_store.delete_namespace.return_value = 1
        fake_store.export_namespace.return_value = []
        with patch("storage.ChromaStore", return_value=fake_store):
            fn(user_id)
        call = (fake_store.delete_namespace.call_args
                or fake_store.export_namespace.call_args)
        self.assertIsNotNone(call, "neither namespace method was called")
        return call.args[0]

    def test_delete_uses_the_id_verbatim(self):
        # Byte-identical to what agent_tools.add_memory would have written,
        # which is `(str(user_id),)` with the id straight out of metadata.
        uid = "web-alice_smith"
        self.assertEqual(
            self._namespace_used_by(self.privacy.forget_memories, uid), (uid,))

    def test_export_uses_the_same_namespace_as_delete(self):
        # The read path had the identical mismatch, which is why the admin
        # panel's per-user memory counts read 0 for punctuated names.
        uid = "web-alice_smith"
        self.assertEqual(
            self._namespace_used_by(self.privacy.export_memories, uid),
            self._namespace_used_by(self.privacy.forget_memories, uid),
        )

    def test_write_and_delete_use_byte_identical_namespaces(self):
        """The invariant the bug violated, asserted directly.

        agent_tools.add_memory is the write path; privacy.forget_memories is
        the delete path. If these two tuples are not identical, /forget reports
        a cheerful success and removes nothing.
        """
        import agent_tools
        uid = "web-alice_smith-42"

        write_store = MagicMock()
        with patch.object(agent_tools, "_get_chroma_store", return_value=write_store):
            agent_tools.add_memory(uid, "likes_pie", "Alice likes pie")
        written_ns = write_store.put.call_args.args[0]

        deleted_ns = self._namespace_used_by(self.privacy.forget_memories, uid)
        self.assertEqual(written_ns, deleted_ns)

    def test_a_linked_identity_is_resolved_before_deleting(self):
        # IDENTITY_LINKS folds web-alice into discord-1, so her memories live
        # under discord-1 and that is what must be deleted.
        import fritz_utils
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": "discord-1"}):
            self.assertEqual(
                self._namespace_used_by(self.privacy.forget_memories, "web-alice"),
                ("discord-1",),
            )


class TestForgetConversation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db_path = self.tmp / "chat.db"
        # Build the minimal SqliteSaver schema by hand.
        with sqlite3.connect(self.db_path) as c:
            c.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
            c.execute("CREATE TABLE writes (thread_id TEXT, blob TEXT)")
            c.executemany(
                "INSERT INTO checkpoints VALUES (?, ?)",
                [("alice", "a"), ("alice", "b"), ("bob", "c")],
            )
            c.executemany(
                "INSERT INTO writes VALUES (?, ?)",
                [("alice", "w1"), ("bob", "w2")],
            )
            c.commit()

        import fritz_utils
        self._patcher = patch.object(fritz_utils, "CHAT_DB_NAME", str(self.db_path))
        self._patcher.start()
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def tearDown(self):
        self._patcher.stop()

    def test_deletes_users_checkpoints_and_writes(self):
        count = self.privacy.forget_conversation("alice")
        # 2 checkpoint rows + 1 writes row = 3 total
        self.assertEqual(count, 3)
        with sqlite3.connect(self.db_path) as c:
            alice_remaining = c.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'alice'"
            ).fetchone()[0]
            bob_remaining = c.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'bob'"
            ).fetchone()[0]
        self.assertEqual(alice_remaining, 0)
        self.assertEqual(bob_remaining, 1)  # Other users untouched.

    def test_returns_zero_when_no_rows_match(self):
        count = self.privacy.forget_conversation("nobody")
        self.assertEqual(count, 0)

    def test_explicit_thread_id_targets_that_thread_only(self):
        # The web chat lives on "web-alice"; clearing it must not touch the
        # Discord thread "alice".
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO checkpoints VALUES ('web-alice', 'x')")
            c.execute("INSERT INTO writes VALUES ('web-alice', 'wx')")
            c.commit()

        count = self.privacy.forget_conversation("alice", thread_id="web-alice")
        self.assertEqual(count, 2)
        with sqlite3.connect(self.db_path) as c:
            web_left = c.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'web-alice'"
            ).fetchone()[0]
            discord_left = c.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'alice'"
            ).fetchone()[0]
        self.assertEqual(web_left, 0)
        self.assertEqual(discord_left, 2)  # Discord thread untouched.

    def test_explicit_thread_id_keeps_dash_and_underscore(self):
        # _sanitise_thread_id strips these; an explicit thread id must not be
        # put through it, or "web-alice" would collapse to "webalice".
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO checkpoints VALUES ('web-a_b', 'y')")
            c.commit()
        self.assertEqual(self.privacy.forget_conversation("a_b", thread_id="web-a_b"), 1)

    def test_thread_id_alone_is_enough(self):
        self.assertEqual(
            self.privacy.forget_conversation("", thread_id="alice"), 3,
        )

    def test_count_conversation_checkpoints(self):
        self.assertEqual(self.privacy.count_conversation_checkpoints("alice"), 2)
        self.assertEqual(self.privacy.count_conversation_checkpoints("bob"), 1)
        self.assertEqual(self.privacy.count_conversation_checkpoints("nobody"), 0)

    def test_missing_tables_handled_gracefully(self):
        # Drop the tables and confirm we get 0 instead of an exception.
        with sqlite3.connect(self.db_path) as c:
            c.execute("DROP TABLE checkpoints")
            c.execute("DROP TABLE writes")
            c.commit()
        self.assertEqual(self.privacy.forget_conversation("alice"), 0)


class TestForgetConversationPerChannel(unittest.TestCase):
    """With THREADS_PER_CHANNEL on, one identity owns several threads. Clearing
    the identity must sweep all of them — and only them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db_path = self.tmp / "chat.db"
        with sqlite3.connect(self.db_path) as c:
            c.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
            c.execute("CREATE TABLE writes (thread_id TEXT, blob TEXT)")
            c.executemany(
                "INSERT INTO checkpoints VALUES (?, ?)",
                [
                    ("discord-1", "identity-thread"),
                    ("discord-1#111", "general"),
                    ("discord-1#222", "dms"),
                    # The trap: a sibling whose id has `discord-1` as a string
                    # prefix. A naive LIKE 'discord-1%' would swallow it.
                    ("discord-10", "someone-else"),
                    ("discord-10#111", "someone-else-in-general"),
                ],
            )
            c.execute("INSERT INTO writes VALUES (?, ?)", ("discord-1#111", "w"))
            c.commit()

        import fritz_utils
        self._patcher = patch.object(fritz_utils, "CHAT_DB_NAME", str(self.db_path))
        self._patcher.start()
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def tearDown(self):
        self._patcher.stop()

    def _remaining(self):
        with sqlite3.connect(self.db_path) as c:
            return sorted(r[0] for r in c.execute("SELECT thread_id FROM checkpoints"))

    def test_sweeps_every_channel_thread_for_the_identity(self):
        removed = self.privacy.forget_conversation("discord-1")
        self.assertEqual(removed, 4)      # 3 checkpoints + 1 writes row
        self.assertEqual(self._remaining(), ["discord-10", "discord-10#111"])

    def test_does_not_touch_a_prefix_sharing_sibling(self):
        self.privacy.forget_conversation("discord-1")
        self.assertIn("discord-10", self._remaining())

    def test_an_explicit_thread_id_targets_exactly_one(self):
        # What /chat's "New conversation" does — it must not clear the user's
        # other surfaces.
        removed = self.privacy.forget_conversation("discord-1", thread_id="discord-1#111")
        self.assertEqual(removed, 2)      # 1 checkpoint + 1 writes row
        self.assertEqual(
            self._remaining(),
            ["discord-1", "discord-1#222", "discord-10", "discord-10#111"],
        )

    def test_like_wildcards_in_an_identity_are_escaped(self):
        # `_` is legal in an identity and means "any character" to LIKE.
        with sqlite3.connect(self.db_path) as c:
            c.executemany(
                "INSERT INTO checkpoints VALUES (?, ?)",
                [("web-a_b#1", "mine"), ("web-aXb#1", "not mine")],
            )
            c.commit()
        self.privacy.forget_conversation("web-a_b")
        self.assertIn("web-aXb#1", self._remaining())

    def test_count_matches_what_forget_would_remove(self):
        # /export reported a count from a narrower predicate than /forget used,
        # so the two disagreed about how much history existed.
        count = self.privacy.count_conversation_checkpoints("discord-1")
        self.assertEqual(count, 3)


class TestForgetSchedules(unittest.TestCase):
    def setUp(self):
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def test_delegates_to_schedule_manager(self):
        manager = MagicMock()
        manager.remove_all_for_user.return_value = 4
        result = self.privacy.forget_schedules("alice", manager)
        self.assertEqual(result, 4)
        manager.remove_all_for_user.assert_called_once_with("alice")

    def test_no_manager_returns_zero(self):
        self.assertEqual(self.privacy.forget_schedules("alice", None), 0)


class TestForgetWorkspace(unittest.TestCase):
    def setUp(self):
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def test_delegates_to_workspace_store(self):
        with patch("workspace_store.remove", return_value=True) as remove_mock:
            result = self.privacy.forget_workspace("alice")
        self.assertTrue(result)
        remove_mock.assert_called_once_with("alice")


class TestForgetAll(unittest.TestCase):
    def setUp(self):
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def test_runs_every_sub_op_and_returns_counts(self):
        manager = MagicMock()
        manager.remove_all_for_user.return_value = 3

        with patch.object(self.privacy, "forget_memories", return_value=10), \
             patch.object(self.privacy, "forget_conversation", return_value=5), \
             patch.object(self.privacy, "forget_workspace", return_value=True), \
             patch.object(self.privacy, "forget_alias", return_value=True):
            result = self.privacy.forget_all("alice", manager)

        self.assertEqual(result, {
            "memories": 10,
            "conversation_rows": 5,
            "schedules": 3,
            "workspace_dropped": True,
            # The display name is personal data too — "forget me" that leaves
            # behind what you are called has not forgotten you.
            "alias_dropped": True,
        })


class TestExportUserData(unittest.TestCase):
    def setUp(self):
        import privacy
        importlib.reload(privacy)
        self.privacy = privacy

    def test_aggregates_everything_into_one_dict(self):
        manager = MagicMock()
        manager.list_schedules.return_value = [{"id": "s1"}]

        with patch.object(self.privacy, "export_memories",
                          return_value=[{"id": "m1"}]), \
             patch.object(self.privacy, "count_conversation_checkpoints",
                          return_value=12), \
             patch.object(self.privacy, "get_workspace_for_export",
                          return_value="/tmp/workspaces/alice"):
            result = self.privacy.export_user_data("alice", manager)

        self.assertEqual(result["user_id"], "alice")
        self.assertEqual(result["memories"], [{"id": "m1"}])
        self.assertEqual(result["schedules"], [{"id": "s1"}])
        self.assertEqual(result["conversation_checkpoint_count"], 12)
        self.assertEqual(result["workspace_path"], "/tmp/workspaces/alice")


class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log_path = self.tmp / "audit.log"
        import observability
        self._patcher = patch.object(observability, "AUDIT_LOG_PATH", str(self.log_path))
        self._patcher.start()
        self.observability = observability

    def tearDown(self):
        self._patcher.stop()

    def test_appends_json_line_per_event(self):
        self.observability.audit_log("forget", user_id="alice", scope="memories", removed=3)
        self.observability.audit_log("export", user_id="alice", bytes=512)
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        import json
        first = json.loads(lines[0])
        self.assertEqual(first["event"], "forget")
        self.assertEqual(first["user_id"], "alice")
        self.assertEqual(first["scope"], "memories")
        self.assertEqual(first["removed"], 3)
        self.assertIn("ts", first)

    def test_does_not_raise_on_unwritable_path(self):
        # Should swallow OSError silently — we don't want audit failures to
        # abort the user's deletion.
        with patch.object(self.observability, "AUDIT_LOG_PATH", "/nonexistent_dir/audit.log"):
            self.observability.audit_log("forget", user_id="alice")  # must not raise


if __name__ == "__main__":
    unittest.main()
