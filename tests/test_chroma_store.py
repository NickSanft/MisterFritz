"""ChromaStore against a real Chroma, not a MagicMock.

Not one of its data methods had ever been executed. It is the only store
production code constructs — agent_tools and privacy both reach it through
storage.get_default_chroma_store — so `/forget memories`, `/export`, and every
memory the assistant recalls rested on tests that asserted what a mock was
asked, never what came back.

That is a particular problem for the pair that matter most. `put` writes with
`namespace` and `original_key` folded into the metadata, and `search` and
`delete_namespace` filter on that `namespace` field: the write path and the
delete path agree only if those strings match, and a mock agrees with anything.
The same class of divergence is what made `/forget memories` report success and
delete nothing once already (see privacy.py's module docstring).

Everything here runs against a real PersistentClient over a temp directory with
FakeEmbeddings, so no Ollama and no network: seconds, not minutes.
"""
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage  # noqa: E402


def _fake_embeddings():
    """Deterministic vectors, so similarity_search is reproducible.

    FakeEmbeddings hashes the text, which is enough for "does the filter work"
    questions. Nothing here asserts an ordering that depends on real semantics.
    """
    from langchain_core.embeddings import DeterministicFakeEmbedding
    return DeterministicFakeEmbedding(size=32)


class ChromaStoreCase(unittest.TestCase):
    def setUp(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:                           # pragma: no cover
            self.skipTest("chromadb not installed")
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        self.addCleanup(SharedSystemClient.clear_system_cache)
        self.tmp = tempfile.mkdtemp()
        patcher = unittest.mock.patch.object(
            storage, "_get_embeddings", return_value=_fake_embeddings())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = storage.ChromaStore(persist_directory=self.tmp)

    def reopen(self):
        """A second ChromaStore over the same directory: what a restart sees."""
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        return storage.ChromaStore(persist_directory=self.tmp)


class TestOneMemoryGoesInAndComesBack(ChromaStoreCase):
    def test_put_then_get_returns_the_metadata(self):
        self.store.put(("discord-1",), "key-1", {"likes": "strong tea"})
        got = self.store.get("key-1")
        self.assertEqual(got["likes"], "strong tea")

    def test_what_put_writes_is_what_the_filters_read(self):
        """The invariant the mocks could not check: `put` folds `namespace` and
        `original_key` into the metadata, and `search`, `delete_namespace` and
        `export_namespace` all key off `namespace`."""
        self.store.put(("discord-1",), "key-1", {"likes": "strong tea"})
        got = self.store.get("key-1")
        self.assertEqual(got["namespace"], "discord-1")
        self.assertEqual(got["original_key"], "key-1")

    def test_an_absent_key_is_none_not_an_error(self):
        self.assertIsNone(self.store.get("never-written"))

    def test_it_survives_a_restart(self):
        self.store.put(("discord-1",), "key-1", {"likes": "strong tea"})
        self.assertEqual(self.reopen().get("key-1")["likes"], "strong tea")

    def test_the_namespace_is_a_path_when_it_has_several_parts(self):
        self.store.put(("discord-1", "profile"), "key-1", {"a": "b"})
        self.assertEqual(self.store.get("key-1")["namespace"], "discord-1/profile")


class TestTheBatchMethods(ChromaStoreCase):
    def setUp(self):
        super().setUp()
        self.store.mset([
            ("discord-1", "a", {"text": "first"}),
            ("discord-1", "b", {"text": "second"}),
            ("discord-2", "c", {"text": "somebody else's"}),
        ])

    def test_mget_returns_one_entry_per_key_in_order(self):
        got = self.store.mget(["b", "a"])
        self.assertEqual([g["original_key"] for g in got], ["b", "a"])

    def test_mget_keeps_a_none_in_place_for_a_missing_key(self):
        """The contract BaseStore callers rely on: positions line up with the
        keys asked for, so a miss cannot shift the others."""
        got = self.store.mget(["a", "absent", "b"])
        self.assertEqual(len(got), 3)
        self.assertIsNone(got[1])
        self.assertEqual(got[0]["original_key"], "a")
        self.assertEqual(got[2]["original_key"], "b")

    def test_mget_of_nothing_is_an_empty_list(self):
        self.assertEqual(self.store.mget([]), [])

    def test_yield_keys_lists_everything(self):
        self.assertEqual(sorted(self.store.yield_keys()), ["a", "b", "c"])

    def test_yield_keys_filters_by_prefix(self):
        self.store.put(("discord-1",), "profile_discord-1", {"x": "y"})
        self.assertEqual(list(self.store.yield_keys("profile_")),
                         ["profile_discord-1"])

    def test_delete_removes_one_and_leaves_the_rest(self):
        self.store.delete("a")
        self.assertIsNone(self.store.get("a"))
        self.assertIsNotNone(self.store.get("b"))

    def test_mdelete_removes_several(self):
        self.store.mdelete(["a", "b"])
        self.assertEqual(sorted(self.store.yield_keys()), ["c"])

    def test_mdelete_of_nothing_deletes_nothing(self):
        """Chroma's delete with an empty id list is not a no-op in every
        version, which is why the guard is there."""
        self.store.mdelete([])
        self.assertEqual(sorted(self.store.yield_keys()), ["a", "b", "c"])


class TestTheContentChromaIndexes(ChromaStoreCase):
    """`mset` picks the document text out of the value, in a fixed order, and
    falls back to JSON. That choice decides what similarity_search can match
    on, so it is behaviour rather than formatting."""

    CASES = (
        ({"page_content": "from page_content", "text": "ignored"}, "from page_content"),
        ({"text": "from text", "content": "ignored"}, "from text"),
        ({"content": "from content"}, "from content"),
    )

    def test_the_documented_precedence_holds(self):
        for index, (value, expected) in enumerate(self.CASES):
            with self.subTest(value=value):
                key = f"key-{index}"
                self.store.put(("discord-1",), key, value)
                [entry] = [e for e in self.store.export_namespace(("discord-1",))
                           if e["id"] == key]
                self.assertEqual(entry["content"], expected)

    def test_a_value_with_no_text_field_is_stored_as_json(self):
        self.store.put(("discord-1",), "key-json", {"likes": "strong tea"})
        [entry] = self.store.export_namespace(("discord-1",))
        self.assertIn("strong tea", entry["content"])
        self.assertIn("likes", entry["content"])


class TestANamespaceIsOnePersonsData(ChromaStoreCase):
    """What `/forget memories` and `/export` are: everything under one
    identity, and nothing under anybody else's."""

    def setUp(self):
        super().setUp()
        self.store.mset([
            ("discord-1", "mine-1", {"text": "my first memory"}),
            ("discord-1", "mine-2", {"text": "my second memory"}),
            ("discord-2", "theirs", {"text": "somebody else's memory"}),
        ])

    def test_export_returns_only_that_namespace(self):
        exported = self.store.export_namespace(("discord-1",))
        self.assertEqual(sorted(e["id"] for e in exported), ["mine-1", "mine-2"])

    def test_export_carries_the_content_and_the_metadata(self):
        [entry] = [e for e in self.store.export_namespace(("discord-1",))
                   if e["id"] == "mine-1"]
        self.assertEqual(entry["content"], "my first memory")
        self.assertEqual(entry["metadata"]["namespace"], "discord-1")

    def test_export_of_an_unknown_namespace_is_empty(self):
        self.assertEqual(self.store.export_namespace(("discord-999",)), [])

    def test_delete_namespace_returns_what_it_removed(self):
        self.assertEqual(self.store.delete_namespace(("discord-1",)), 2)

    def test_delete_namespace_leaves_the_other_person_alone(self):
        """The assertion a mock cannot make, and the one that matters: this is
        /forget memories, and it must not reach into anyone else's."""
        self.store.delete_namespace(("discord-1",))
        self.assertEqual(sorted(self.store.yield_keys()), ["theirs"])
        self.assertIsNotNone(self.store.get("theirs"))

    def test_deleting_twice_removes_nothing_the_second_time(self):
        self.store.delete_namespace(("discord-1",))
        self.assertEqual(self.store.delete_namespace(("discord-1",)), 0)

    def test_the_deletion_survives_a_restart(self):
        self.store.delete_namespace(("discord-1",))
        self.assertEqual(sorted(self.reopen().yield_keys()), ["theirs"])


class TestSearchStaysInsideTheNamespace(ChromaStoreCase):
    """search() is how the agent recalls anything, and it filters on the same
    `namespace` field `put` writes. If those ever diverge it returns either
    nothing or somebody else's memories, and a mock would report neither."""

    def setUp(self):
        super().setUp()
        self.store.mset([
            ("discord-1", "mine", {"tea": "strong, no sugar"}),
            ("discord-2", "theirs", {"tea": "strong, no sugar"}),
        ])

    def test_it_returns_the_key_and_the_metadata(self):
        [(key, metadata)] = self.store.search("tea", ("discord-1",), limit=5)
        self.assertEqual(key, "mine")
        self.assertEqual(metadata["tea"], "strong, no sugar")

    def test_it_never_returns_another_persons_memory(self):
        found = self.store.search("tea", ("discord-1",), limit=50)
        self.assertEqual([key for key, _ in found], ["mine"])

    def test_an_unknown_namespace_finds_nothing(self):
        self.assertEqual(self.store.search("tea", ("discord-999",), limit=5), [])

    def test_the_limit_is_respected(self):
        self.store.mset([("discord-1", f"extra-{i}", {"tea": "more tea"})
                         for i in range(5)])
        self.assertLessEqual(len(self.store.search("tea", ("discord-1",), limit=2)), 2)


class TestTheProductionCallPathsWork(ChromaStoreCase):
    """The two real entry points, driven as the app drives them, because the
    namespace is built differently in each: agent_tools.add_memory wraps the id
    in a one-tuple, privacy resolves it first.

    Both bindings have to be redirected, and that is the point of the assertion
    in `_use_this_store`. agent_tools does `from storage import
    get_default_chroma_store` at module scope, so patching the attribute on
    `storage` leaves agent_tools holding the original; privacy imports it
    inside the function, so patching `storage` is exactly what reaches it.
    Patching only `storage` made the first version of the round-trip test pass
    while both halves quietly used the real singleton — which is the failure
    this whole file exists to make impossible.
    """

    def _use_this_store(self):
        import agent_tools
        import contextlib
        stack = contextlib.ExitStack()
        stack.enter_context(unittest.mock.patch.object(
            agent_tools, "get_default_chroma_store", return_value=self.store))
        stack.enter_context(unittest.mock.patch.object(
            storage, "get_default_chroma_store", return_value=self.store))
        self.addCleanup(stack.close)
        return stack

    def test_add_memory_then_search_memories_internal_round_trips(self):
        import agent_tools
        with self._use_this_store():
            agent_tools.add_memory("discord-1", "tea", "strong, no sugar")
            # Proof the write landed HERE and not in the shared singleton.
            self.assertEqual(len(list(self.store.yield_keys())), 1)
            found = agent_tools.search_memories_internal(
                {"metadata": {"user_id": "discord-1"}}, "tea")
        self.assertIn("strong, no sugar", found)

    def test_forget_memories_removes_what_add_memory_wrote(self):
        """The write path and the delete path agreeing, end to end. They did
        not once: privacy.py's docstring records /forget memories reporting
        success and deleting nothing, because the two sides keyed the namespace
        differently."""
        import agent_tools
        import privacy
        with self._use_this_store():
            agent_tools.add_memory("discord-1", "tea", "strong, no sugar")
            agent_tools.add_memory("discord-1", "biscuits", "digestives")
            self.assertEqual(len(list(self.store.yield_keys())), 2)
            self.assertEqual(privacy.forget_memories("discord-1"), 2)
            self.assertEqual(privacy.export_memories("discord-1"), [])

    def test_forget_memories_is_one_persons_only(self):
        import agent_tools
        import privacy
        with self._use_this_store():
            agent_tools.add_memory("discord-1", "tea", "strong")
            agent_tools.add_memory("discord-2", "tea", "weak")
            privacy.forget_memories("discord-1")
            self.assertEqual(len(privacy.export_memories("discord-2")), 1)
            self.assertEqual(privacy.export_memories("discord-1"), [])


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
