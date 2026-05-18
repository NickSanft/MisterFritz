"""
Tests for the Phase 1 thread-safety work in document_engine.py:

  - VECTORSTORE_LOCK is held when reading/writing GLOBAL_VECTORSTORE
  - ingestion_worker exits cleanly on the _SHUTDOWN_SENTINEL
  - shutdown() is idempotent and safe to call when nothing started

document_engine has a heavy import chain (chroma, langchain, easyocr,
PyMuPDF, watchdog). We stub the few that aren't installable in this
environment so the import itself succeeds.
"""
import importlib.machinery
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch


def _real_module_stub(name: str) -> types.ModuleType:
    """Insert a real (importable) stub module — needed when downstream code
    calls importlib.util.find_spec(name) which rejects MagicMock-based stubs.
    """
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__path__ = []  # mark as a package so submodule imports succeed
    sys.modules[name] = mod
    return mod


def _ensure_mock(name: str) -> MagicMock:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# msoffcrypto + the OCR stack are optional in this codebase; the source code
# guards them with try/except at import time, so stubs only need to exist if
# something downstream imports them eagerly.
_ensure_mock("easyocr")
# PyMuPDF (fitz) — guarded by try/except in the source.
_ensure_mock("fitz")
# spacy / thinc get pulled in by langchain_text_splitters and transformers on
# some installs and can crash with numpy ABI mismatches in dev environments.
# We never use them directly, so real-module stubs are safe.
_real_module_stub("spacy")
_real_module_stub("thinc")
_real_module_stub("thinc.api")
_real_module_stub("transformers")

# If a prior test (e.g. test_bot_commands.py) stubbed document_engine in
# sys.modules, evict the stub so we import the real module — we actually
# need real shutdown(), real ingestion_worker, real VECTORSTORE_LOCK.
if "document_engine" in sys.modules and isinstance(sys.modules["document_engine"], MagicMock):
    del sys.modules["document_engine"]

import document_engine  # noqa: E402


class TestVectorStoreLockHeldDuringIngest(unittest.TestCase):
    """Verify the worker reads GLOBAL_VECTORSTORE under VECTORSTORE_LOCK.

    Strategy: replace the lock with a tracker that records every acquire,
    push one item through the queue, and confirm the worker took the lock
    before touching the vectorstore.
    """

    def test_worker_acquires_vectorstore_lock_before_use(self):
        acquired = threading.Event()

        class TrackingLock:
            def __init__(self, real):
                self._real = real
                self.acquired = False

            def __enter__(self):
                self._real.__enter__()
                self.acquired = True
                acquired.set()
                return self

            def __exit__(self, *exc):
                return self._real.__exit__(*exc)

        fake_vs = MagicMock()
        fake_vs.add_documents = MagicMock()
        fake_vs.delete = MagicMock()

        # Swap state with our tracker for the duration of the test.
        original_lock = document_engine.VECTORSTORE_LOCK
        tracker = TrackingLock(threading.RLock())

        with patch.object(document_engine, "VECTORSTORE_LOCK", tracker), \
             patch.object(document_engine, "GLOBAL_VECTORSTORE", fake_vs), \
             patch.object(document_engine, "load_document_by_extension", return_value=[]):

            worker = threading.Thread(
                target=document_engine.ingestion_worker, daemon=True
            )
            worker.start()
            try:
                # Push a delete (cheapest path — no doc loading needed).
                document_engine.INGESTION_QUEUE.put(("delete", "/tmp/whatever.txt"))
                acquired_in_time = acquired.wait(timeout=5.0)
                self.assertTrue(acquired_in_time, "Worker never acquired lock")
            finally:
                document_engine.INGESTION_QUEUE.put(document_engine._SHUTDOWN_SENTINEL)
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive(), "Worker did not exit on sentinel")

        # Restore original lock so other tests don't see the patch.
        document_engine.VECTORSTORE_LOCK = original_lock


class TestShutdownSentinelStopsWorker(unittest.TestCase):
    def test_worker_exits_cleanly_on_sentinel(self):
        worker = threading.Thread(target=document_engine.ingestion_worker, daemon=True)
        worker.start()
        document_engine.INGESTION_QUEUE.put(document_engine._SHUTDOWN_SENTINEL)
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())


class TestShutdownIdempotent(unittest.TestCase):
    def test_shutdown_safe_when_nothing_started(self):
        # No initialize_vectorstore() means _OBSERVER and _WORKER_THREAD are
        # both None. shutdown() must not raise.
        with patch.object(document_engine, "_OBSERVER", None), \
             patch.object(document_engine, "_WORKER_THREAD", None):
            document_engine.shutdown()  # should be a no-op

    def test_shutdown_calls_observer_stop_and_join(self):
        fake_observer = MagicMock()
        fake_observer.stop = MagicMock()
        fake_observer.join = MagicMock()

        # Use a real thread that is_alive() returns False for — a never-started
        # thread has is_alive() == False, so the sentinel push branch is skipped.
        idle_thread = threading.Thread(target=lambda: None)

        with patch.object(document_engine, "_OBSERVER", fake_observer), \
             patch.object(document_engine, "_WORKER_THREAD", idle_thread):
            document_engine.shutdown(timeout=0.1)

        fake_observer.stop.assert_called_once()
        fake_observer.join.assert_called_once()

    def test_shutdown_pushes_sentinel_to_live_worker(self):
        # Spin up a real worker, then call shutdown — it should push the
        # sentinel and the worker should exit.
        worker = threading.Thread(target=document_engine.ingestion_worker, daemon=True)
        worker.start()
        # Give the worker a moment to actually block on queue.get().
        time.sleep(0.1)

        with patch.object(document_engine, "_OBSERVER", None), \
             patch.object(document_engine, "_WORKER_THREAD", worker):
            document_engine.shutdown(timeout=5.0)

        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
