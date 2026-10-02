"""The memory store is trusted input, not data.

Chroma builds an embedding function named in a collection's own persisted
configuration or schema, and then calls it on every write — with no server
involved, and whatever embedding function the caller passed in. Measured
against the pinned chromadb 1.5.9 on a throwaway store: a schema naming
`chroma-cloud-splade` turns a single add() into an HTTPS POST of the document
text to embed.trychroma.com, carrying the value of whichever environment
variable the spec chooses as its credential header. `get` never touches it.

No chromadb release fixes that and there is nothing to configure, so the only
thing to do here is refuse to open such a store. The poisoning below writes the
spec straight into Chroma's own sqlite, because that is the shape the problem
arrives in: a restored backup, a bind mount, a synced directory, or a
collection exported from somewhere else.

Nothing here reaches the network: the specs name an embedder, but no test ever
calls one.
"""
import ast
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fritz_utils as fu  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# A real spec names a real variable — `DISCORD_BOT_TOKEN` is the one that
# matters — but naming it here would have chromadb read this machine's actual
# token out of .env while deserializing the schema. The hazard does not depend
# on which variable it is.
KEY_VARIABLE = "FRITZ_TEST_EMBEDDER_KEY"

SPLADE = {
    "type": "known",
    "name": "chroma-cloud-splade",
    "config": {"api_key_env_var": KEY_VARIABLE,
               "model": "prithivida/Splade_PP_en_v1"},
}


def _forget_clients() -> None:
    """chromadb caches a client per path for the life of the process, so an
    open after a write to Chroma's sqlite would otherwise be served the old
    collection from memory — and a client built with different Settings than
    the cached one raises outright. Both are artefacts of testing in-process;
    dropping the cache is what makes each open a real one."""
    from chromadb.api.shared_system_client import SharedSystemClient
    SharedSystemClient.clear_system_cache()


def _chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:                                  # pragma: no cover
        raise unittest.SkipTest("chromadb not installed")


def _a_store(path: Path, name: str = "langchain_store"):
    """A collection holding one memory, shaped the way this app writes them."""
    chromadb = _chromadb()
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(name)
    collection.add(ids=["mem1"], embeddings=[[0.1, 0.2, 0.3]],
                   documents=["a memory"], metadatas=[{"namespace": "discord-1"}])
    return client, collection


def _rewrite(path: Path, column: str, value) -> None:
    db = sqlite3.connect(str(path / "chroma.sqlite3"))
    try:
        db.execute(f"UPDATE collections SET {column} = ?", (json.dumps(value),))
        db.commit()
    finally:
        db.close()


def _read(path: Path, column: str):
    db = sqlite3.connect(str(path / "chroma.sqlite3"))
    try:
        raw = db.execute(f"SELECT {column} FROM collections").fetchone()[0]
    finally:
        db.close()
    return json.loads(raw) if raw else {}


def poison_schema(path: Path, spec=SPLADE, *, source_key: str = "#document",
                  enabled: bool = True) -> None:
    """Name a sparse embedder on the document key, where Chroma looks."""
    schema = _read(path, "schema_str")
    schema.setdefault("keys", {})[source_key] = {
        "sparse_vector": {"sparse_vector_index": {
            "enabled": enabled,
            "config": {"source_key": source_key, "embedding_function": spec},
        }}
    }
    _rewrite(path, "schema_str", schema)
    _forget_clients()


def json_names(payload) -> list[str]:
    """Embedder names in an arbitrary schema payload, via the guard itself."""
    return fu.stored_embedders(_FakeCollection(None, schema=payload))


class _FakeSchema:
    def __init__(self, payload):
        self._payload = payload

    def serialize_to_json(self):
        return self._payload


class _FakeCollection:
    """Just the two pieces of a chromadb Collection that the guard reads."""

    def __init__(self, configuration, schema="unset"):
        self.configuration_json = configuration or {}
        self.schema = None if schema is None else _FakeSchema(
            {} if schema == "unset" else schema)


class StoreCase(unittest.TestCase):
    def setUp(self):
        _chromadb()
        _forget_clients()
        self.addCleanup(_forget_clients)
        self.tmp = Path(tempfile.mkdtemp())

    def reopen(self, name: str = "langchain_store"):
        chromadb = _chromadb()
        _forget_clients()
        client = chromadb.PersistentClient(path=str(self.tmp))
        return client.get_collection(name)


class TestWhatCountsAsAnEmbedderTheStoreChose(StoreCase):
    def test_a_store_this_app_wrote_names_none(self):
        """The baseline that matters: silence on an ordinary store.

        Chroma's own configuration always carries `default` (its local ONNX
        model) and a sparse slot of type `unknown`, which deserializes to None.
        Neither is the store making a choice about where text goes.
        """
        _a_store(self.tmp)
        self.assertEqual(fu.stored_embedders(self.reopen()), [])

    def test_a_sparse_spec_in_the_schema_is_named(self):
        _a_store(self.tmp)
        poison_schema(self.tmp)
        self.assertEqual(fu.stored_embedders(self.reopen()), ["chroma-cloud-splade"])

    def test_a_dense_spec_in_the_configuration_is_named(self):
        """The configuration is read as well as the schema.

        Writing config_json_str directly has no observable effect on chromadb
        1.5.9 — the configuration that comes back is derived — so this covers
        the reading rather than a poisoning that version would honour. The
        dense known embedders have the same shape as the sparse ones, right
        down to naming an environment variable to send as a credential.
        """
        collection = _FakeCollection({"embedding_function": {
            "type": "known", "name": "chroma-cloud-qwen",
            "config": {"api_key_env_var": KEY_VARIABLE}}})
        self.assertEqual(fu.stored_embedders(collection), ["chroma-cloud-qwen"])

    def test_a_spec_chromadb_cannot_build_today_is_still_named(self):
        """The reason the guard reads the stored bytes and not chromadb's view.

        With the named variable unset, chromadb warns, substitutes None, and
        its Schema object comes back clean — while the spec sits in the file,
        live from the moment that variable exists. This test passes only
        because nothing sets KEY_VARIABLE here.
        """
        self.assertIsNone(os.environ.get(KEY_VARIABLE))
        _a_store(self.tmp)
        poison_schema(self.tmp)
        collection = self.reopen()
        self.assertEqual(json_names(collection.schema.serialize_to_json()), [],
                         "chromadb now keeps unbuildable specs in its own view; "
                         "the guard could read that instead")
        self.assertEqual(fu.stored_embedders(collection), ["chroma-cloud-splade"])

    def test_the_stored_schema_is_where_the_guard_looks_for_it(self):
        """If chromadb stops carrying the raw schema on the collection model,
        the guard quietly falls back to the view that loses specs. Fail here
        instead."""
        _a_store(self.tmp)
        collection = self.reopen()
        self.assertIn("serialized_schema", dict(collection._model))

    def test_a_spec_switched_off_today_is_still_named(self):
        """`enabled` is one UPDATE away from true, and this guard is not the
        place to reason about which half of a poisoned store is live."""
        _a_store(self.tmp)
        poison_schema(self.tmp, enabled=False)
        self.assertEqual(fu.stored_embedders(self.reopen()), ["chroma-cloud-splade"])

    def test_unknown_and_legacy_specs_name_nothing(self):
        """The two types chromadb itself deserializes to None, so they cannot
        run. Naming them would refuse ordinary stores."""
        for kind in ("unknown", "legacy"):
            with self.subTest(kind=kind):
                collection = _FakeCollection({"embedding_function": {"type": kind}})
                self.assertEqual(fu.stored_embedders(collection), [])

    def test_a_nameless_spec_is_still_reported(self):
        collection = _FakeCollection({"embedding_function": {"type": "known"}})
        self.assertEqual(fu.stored_embedders(collection), ["unnamed"])

    def test_specs_are_found_however_deeply_they_are_nested(self):
        """The walk recurses through dicts and lists because Chroma's schema
        shape is not this app's to predict — a future layout must not be able
        to hide a spec from the guard."""
        buried = {"keys": [{"a": {"b": [{"embedding_function":
                                         {"type": "known", "name": "jina"}}]}}]}
        self.assertEqual(fu.stored_embedders(_FakeCollection(buried)), ["jina"])

    def test_a_collection_with_no_schema_at_all_is_fine(self):
        self.assertEqual(fu.stored_embedders(_FakeCollection(None, schema=None)), [])


class TestTheGuardRefuses(StoreCase):
    def test_a_clean_store_opens_without_complaint(self):
        _a_store(self.tmp)
        self.assertIsNone(fu.refuse_stored_embedders(self.reopen(), "the memory store"))

    def test_a_poisoned_store_stops_the_caller(self):
        _a_store(self.tmp)
        poison_schema(self.tmp)
        with self.assertRaises(RuntimeError) as caught:
            fu.refuse_stored_embedders(self.reopen(), "the memory store at ./chroma_store")
        message = str(caught.exception)
        self.assertIn("chroma-cloud-splade", message)
        self.assertIn("./chroma_store", message)

    def test_chromadb_really_would_have_called_it(self):
        """The guard is only worth having while this is true.

        If a future chromadb stops building embedders out of the stored schema,
        this is the test that says so — rather than leaving the comments around
        the guard asserting something stale.
        """
        _a_store(self.tmp)
        poison_schema(self.tmp)
        # The variable the spec names has to exist for chromadb to build the
        # thing, and the schema is deserialized lazily — so the whole check
        # happens while it is set. Setting it is all a real host does.
        with unittest.mock.patch.dict(os.environ, {KEY_VARIABLE: "not-a-real-key"}):
            collection = self.reopen()
            if not hasattr(collection, "_get_sparse_embedding_targets"):
                self.skipTest("chromadb no longer resolves sparse targets this way")
            targets = collection._get_sparse_embedding_targets()
            self.assertEqual(list(targets), ["#document"])
            built = targets["#document"].embedding_function
            self.assertIsNotNone(
                built, "chromadb built nothing from the stored spec; re-check the guard")
            self.assertTrue(callable(built))


class TestEveryCallerThatOpensTheStoreChecksIt(StoreCase):
    """A guard nobody calls is decoration. Three of the four callers are cheap
    to drive; initialize_vectorstore starts a watchdog observer and a worker
    thread, so it is covered structurally further down."""

    def _fake_embeddings(self):
        from langchain_core.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=3)

    def test_the_memory_store_refuses_to_construct(self):
        import storage
        _a_store(self.tmp)
        poison_schema(self.tmp)
        with unittest.mock.patch.object(storage, "_get_embeddings",
                                        return_value=self._fake_embeddings()):
            with self.assertRaises(RuntimeError) as caught:
                storage.ChromaStore(persist_directory=str(self.tmp))
        self.assertIn("chroma-cloud-splade", str(caught.exception))

    def test_the_memory_store_still_opens_a_clean_one(self):
        import storage
        _a_store(self.tmp)
        with unittest.mock.patch.object(storage, "_get_embeddings",
                                        return_value=self._fake_embeddings()):
            store = storage.ChromaStore(persist_directory=str(self.tmp))
        self.assertEqual(store.collection_name, "langchain_store")

    def test_the_migration_survey_stops_instead_of_skipping(self):
        """survey_chroma prints `[skip]` and carries on for every other
        failure. This one must not be a skip: the next step writes."""
        import migrate_identity
        _a_store(self.tmp)
        poison_schema(self.tmp)
        with self.assertRaises(RuntimeError):
            migrate_identity.survey_chroma(str(self.tmp))

    def test_the_migration_rewrite_stops_before_it_adds(self):
        import migrate_identity
        _a_store(self.tmp)
        poison_schema(self.tmp)
        with self.assertRaises(RuntimeError):
            migrate_identity.rewrite_chroma(str(self.tmp), {"discord-1": "discord-2"})

    def test_a_clean_store_still_migrates(self):
        import migrate_identity
        _a_store(self.tmp)
        self.assertEqual(
            migrate_identity.rewrite_chroma(str(self.tmp), {"discord-1": "discord-2"}), 1)


class TestTheWiringIsThere(unittest.TestCase):
    """Reads the source rather than running it, so the one caller that is
    expensive to start is held to the same rule as the rest."""

    WRITERS = [
        ("storage.py", "__init__"),
        ("document_engine.py", "initialize_vectorstore"),
        ("migrate_identity.py", "survey_chroma"),
        ("migrate_identity.py", "rewrite_chroma"),
    ]

    @staticmethod
    def _functions(module: str, name: str):
        tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
        return [node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name]

    def test_every_function_that_opens_a_collection_calls_the_guard(self):
        for module, name in self.WRITERS:
            with self.subTest(module=module, function=name):
                bodies = self._functions(module, name)
                self.assertTrue(bodies, f"{module}: no function named {name}")
                calls = [node for body in bodies for node in ast.walk(body)
                         if isinstance(node, ast.Call)
                         and getattr(node.func, "id", None) == "refuse_stored_embedders"]
                self.assertTrue(
                    calls,
                    f"{module}:{name} opens a Chroma collection without calling "
                    "refuse_stored_embedders — a store that names its own "
                    "embedder would be obeyed there")

    def test_no_other_place_opens_a_collection_unguarded(self):
        """If a fifth caller appears, this is the test that notices."""
        guarded = {(module, name) for module, name in self.WRITERS}
        for path in sorted(REPO.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions = [node for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                opens = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if opens not in {"Chroma", "get_collection", "get_or_create_collection"}:
                    continue
                enclosing = [f for f in functions
                             if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
                with self.subTest(module=path.name, line=node.lineno, opens=opens):
                    self.assertTrue(
                        any((path.name, f.name) in guarded for f in enclosing),
                        f"{path.name}:{node.lineno} opens a Chroma collection "
                        "outside the guarded functions; call "
                        "refuse_stored_embedders there and list it in WRITERS")


class TestAmbientSettingsCannotRedirectTheStore(unittest.TestCase):
    """chromadb reads its own Settings from the environment and from any .env in
    the working directory — the same .env this repo asks people to write.
    CHROMA_API_IMPL wins over the persist_directory this app passes: set it to
    the FastAPI impl with a host, and Chroma(persist_directory=...) is quietly a
    client of that host instead, with every memory going there. Verified
    against the pinned chromadb 1.5.9.
    """

    def setUp(self):
        patched = unittest.mock.patch.dict(os.environ, {}, clear=False)
        patched.start()
        self.addCleanup(patched.stop)
        for name in [n for n in os.environ if n.upper() == "CHROMA_API_IMPL"]:
            del os.environ[name]

    def test_a_remote_api_impl_is_refused(self):
        os.environ["CHROMA_API_IMPL"] = "chromadb.api.fastapi.FastAPI"
        with self.assertRaises(RuntimeError) as caught:
            fu.validate_config()
        self.assertIn("CHROMA_API_IMPL", str(caught.exception))

    def test_the_name_is_matched_the_way_chromadb_reads_it(self):
        """pydantic-settings matches environment names case-insensitively, so a
        lowercase line in a .env binds for chromadb just as well.

        Checked against a plain mapping rather than os.environ, which is
        case-insensitive on Windows and case-sensitive on Linux — through the
        real environment this would assert nothing on one of the two.
        """
        self.assertEqual(
            fu.chroma_api_impl({"chroma_api_impl": "chromadb.api.fastapi.FastAPI"}),
            "chromadb.api.fastapi.FastAPI")
        self.assertIsNone(fu.chroma_api_impl({"CHROMA_DB_PATH": "./chroma_store"}))

    def test_a_lowercase_setting_reaches_the_refusal(self):
        os.environ["chroma_api_impl"] = "chromadb.api.async_fastapi.AsyncFastAPI"
        with self.assertRaises(RuntimeError):
            fu.validate_config()

    def test_the_embedded_implementations_are_accepted(self):
        for impl in ("chromadb.api.rust.RustBindingsAPI",
                     "chromadb.api.segment.SegmentAPI"):
            with self.subTest(impl=impl):
                os.environ["CHROMA_API_IMPL"] = impl
                fu.validate_config()          # must not raise

    def test_unset_is_the_normal_case(self):
        fu.validate_config()                  # must not raise


if __name__ == "__main__":                    # pragma: no cover
    unittest.main()
