import os
import tempfile
import unittest

from sqlite_store import SQLiteStore


class TestSQLiteStoreCRUD(unittest.TestCase):
    def setUp(self):
        # SQLiteStore re-opens a new connection per operation, so ":memory:"
        # won't work (each connect() gets a fresh empty database).
        # Use a real temp file so all operations share the same on-disk DB.
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = SQLiteStore(self.db_path)

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    # ── put / get ──────────────────────────────────────────────────────────

    def test_put_and_get_round_trip(self):
        self.store.put(("ns",), "key1", {"data": "hello"})
        result = self.store.get("key1")
        self.assertEqual(result, {"data": "hello"})

    def test_get_missing_key_returns_none(self):
        result = self.store.get("nonexistent")
        self.assertIsNone(result)

    def test_put_overwrites_existing_key(self):
        self.store.put(("ns",), "key1", {"v": 1})
        self.store.put(("ns",), "key1", {"v": 2})
        self.assertEqual(self.store.get("key1"), {"v": 2})

    # ── mget ───────────────────────────────────────────────────────────────

    def test_mget_returns_all_values_in_order(self):
        self.store.put(("ns",), "a", {"n": 1})
        self.store.put(("ns",), "b", {"n": 2})
        results = self.store.mget(["b", "a"])
        self.assertEqual(results[0], {"n": 2})
        self.assertEqual(results[1], {"n": 1})

    def test_mget_missing_key_returns_none_in_slot(self):
        self.store.put(("ns",), "exists", {"x": 1})
        results = self.store.mget(["exists", "missing"])
        self.assertEqual(results[0], {"x": 1})
        self.assertIsNone(results[1])

    def test_mget_empty_list_returns_empty(self):
        self.assertEqual(self.store.mget([]), [])

    # ── delete / mdelete ──────────────────────────────────────────────────

    def test_delete_removes_key(self):
        self.store.put(("ns",), "key1", {"v": 1})
        self.store.delete("key1")
        self.assertIsNone(self.store.get("key1"))

    def test_mdelete_removes_multiple_keys(self):
        for k in ("a", "b", "c"):
            self.store.put(("ns",), k, {"v": k})
        self.store.mdelete(["a", "c"])
        self.assertIsNone(self.store.get("a"))
        self.assertIsNone(self.store.get("c"))
        self.assertIsNotNone(self.store.get("b"))

    def test_delete_nonexistent_key_is_silent(self):
        self.store.delete("ghost")  # should not raise

    # ── yield_keys ────────────────────────────────────────────────────────

    def test_yield_keys_with_prefix(self):
        self.store.put(("ns",), "apple", {"v": 1})
        self.store.put(("ns",), "apricot", {"v": 2})
        self.store.put(("ns",), "banana", {"v": 3})
        keys = list(self.store.yield_keys("ap"))
        self.assertIn("apple", keys)
        self.assertIn("apricot", keys)
        self.assertNotIn("banana", keys)

    def test_yield_keys_empty_prefix_returns_all(self):
        self.store.put(("ns",), "x", {"v": 1})
        self.store.put(("ns",), "y", {"v": 2})
        keys = list(self.store.yield_keys(""))
        self.assertIn("x", keys)
        self.assertIn("y", keys)

    # ── search (namespace) ────────────────────────────────────────────────

    def test_search_returns_entries_in_namespace(self):
        self.store.put(("ns1",), "k1", {"val": "a"})
        self.store.put(("ns2",), "k2", {"val": "b"})
        results = self.store.search(("ns1",), limit=10)
        keys = [r[0] for r in results]
        self.assertIn("k1", keys)
        self.assertNotIn("k2", keys)

    def test_search_respects_limit(self):
        for i in range(10):
            self.store.put(("ns",), f"k{i}", {"i": i})
        results = self.store.search(("ns",), limit=3)
        self.assertLessEqual(len(results), 3)

    def test_search_empty_namespace_returns_empty(self):
        results = self.store.search(("empty_ns",), limit=10)
        self.assertEqual(results, [])

    # ── namespace isolation ───────────────────────────────────────────────

    def test_same_key_different_namespace_stored_separately(self):
        self.store.put(("ns1",), "shared_key", {"val": "ns1_val"})
        self.store.put(("ns2",), "shared_key", {"val": "ns2_val"})
        ns1_results = self.store.search(("ns1",), limit=10)
        ns2_results = self.store.search(("ns2",), limit=10)
        ns1_vals = [r[1].get("val") for r in ns1_results]
        ns2_vals = [r[1].get("val") for r in ns2_results]
        self.assertIn("ns1_val", ns1_vals)
        self.assertIn("ns2_val", ns2_vals)
        self.assertNotIn("ns2_val", ns1_vals)


if __name__ == "__main__":
    unittest.main()
