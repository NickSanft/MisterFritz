import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from observability import (
    Metrics, _format_duration,
    get_health_snapshot, format_health_text,
)


class TestMetricsCounters(unittest.TestCase):
    def setUp(self):
        self.m = Metrics()

    def test_increment_creates_counter(self):
        self.m.increment("foo")
        self.assertEqual(self.m.snapshot()["counters"]["foo"], 1)

    def test_increment_accumulates(self):
        self.m.increment("bar", 3)
        self.m.increment("bar", 2)
        self.assertEqual(self.m.snapshot()["counters"]["bar"], 5)

    def test_multiple_distinct_counters(self):
        self.m.increment("a")
        self.m.increment("b")
        snap = self.m.snapshot()["counters"]
        self.assertEqual(snap["a"], 1)
        self.assertEqual(snap["b"], 1)

    def test_default_value_is_one(self):
        self.m.increment("x")
        self.assertEqual(self.m.snapshot()["counters"]["x"], 1)


class TestMetricsLatency(unittest.TestCase):
    def setUp(self):
        self.m = Metrics()

    def test_record_latency_stored(self):
        self.m.record_latency("op", 0.5)
        snap = self.m.snapshot()
        self.assertIn("op", snap["latencies"])
        self.assertIn(0.5, snap["latencies"]["op"])

    def test_multiple_latencies_accumulated(self):
        for v in (0.1, 0.2, 0.3):
            self.m.record_latency("op", v)
        samples = self.m.snapshot()["latencies"]["op"]
        self.assertEqual(len(samples), 3)

    def test_rolling_window_caps_at_max_samples(self):
        # Record 210 samples with max_samples=200 → only last 200 retained
        for i in range(210):
            self.m.record_latency("op", float(i), max_samples=200)
        samples = self.m.snapshot()["latencies"]["op"]
        self.assertEqual(len(samples), 200)
        # The oldest 10 samples (0.0 – 9.0) should have been evicted
        self.assertNotIn(0.0, samples)

    def test_latency_thread_safety(self):
        import threading
        errors = []

        def worker():
            try:
                for _ in range(50):
                    self.m.record_latency("concurrent", 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestMetricsErrors(unittest.TestCase):
    def setUp(self):
        self.m = Metrics()

    def test_record_error_increments_count(self):
        self.m.record_error("scrape_web", ValueError("oops"))
        self.assertEqual(self.m.snapshot()["errors"]["scrape_web"], 1)

    def test_record_error_stores_last_error(self):
        self.m.record_error("tool", Exception("test error"))
        snap = self.m.snapshot()
        name, ts, msg = snap["last_error"]
        self.assertEqual(name, "tool")
        self.assertIn("test error", msg)

    def test_record_error_accepts_string(self):
        self.m.record_error("cmd", "something went wrong")
        self.assertEqual(self.m.snapshot()["errors"]["cmd"], 1)

    def test_multiple_errors_for_same_key_accumulated(self):
        self.m.record_error("op", "err1")
        self.m.record_error("op", "err2")
        self.assertEqual(self.m.snapshot()["errors"]["op"], 2)


class TestMetricsTimeBlock(unittest.TestCase):
    """Phase 11: time_block context manager records counter + latency
    + errors atomically around a block."""

    def setUp(self):
        self.m = Metrics()

    def test_records_counter_and_latency_on_success(self):
        with self.m.time_block("op"):
            time.sleep(0.01)
        snap = self.m.snapshot()
        self.assertEqual(snap["counters"]["op"], 1)
        self.assertEqual(len(snap["latencies"]["op"]), 1)
        self.assertGreater(snap["latencies"]["op"][0], 0)

    def test_records_error_and_latency_when_block_raises(self):
        with self.assertRaises(ValueError):
            with self.m.time_block("op"):
                raise ValueError("boom")
        snap = self.m.snapshot()
        self.assertEqual(snap["counters"]["op"], 1)        # still incremented
        self.assertEqual(snap["errors"]["op"], 1)          # error recorded
        self.assertEqual(len(snap["latencies"]["op"]), 1)  # latency still observed

    def test_nested_blocks_record_independently(self):
        with self.m.time_block("outer"):
            with self.m.time_block("inner"):
                pass
        snap = self.m.snapshot()
        self.assertEqual(snap["counters"]["outer"], 1)
        self.assertEqual(snap["counters"]["inner"], 1)


class TestTimeToolHelper(unittest.TestCase):
    """Phase 11: time_tool() auto-prefixes the metric name with 'tool.'."""

    def test_prefixes_with_tool_namespace(self):
        from observability import METRICS, time_tool
        # Stash a baseline so other tests don't pollute the count assertion.
        before = METRICS.snapshot()["counters"].get("tool.unit_test_sample", 0)
        with time_tool("unit_test_sample"):
            pass
        after = METRICS.snapshot()["counters"]["tool.unit_test_sample"]
        self.assertEqual(after - before, 1)


class TestFormatDuration(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(_format_duration(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(_format_duration(125), "2m 5s")

    def test_hours_minutes_seconds(self):
        self.assertEqual(_format_duration(3661), "1h 1m 1s")

    def test_days(self):
        result = _format_duration(86400 + 3600 + 60 + 1)
        self.assertIn("1d", result)

    def test_zero_seconds(self):
        self.assertEqual(_format_duration(0), "0s")

    def test_negative_clamped_to_zero(self):
        self.assertEqual(_format_duration(-100), "0s")


class TestHealthSnapshot(unittest.TestCase):
    def test_snapshot_has_required_keys(self):
        snap = get_health_snapshot()
        for key in ("uptime_sec", "counters", "errors", "latencies", "last_error"):
            with self.subTest(key=key):
                self.assertIn(key, snap)

    def test_uptime_is_positive(self):
        snap = get_health_snapshot()
        self.assertGreater(snap["uptime_sec"], 0)


class TestFormatHealthText(unittest.TestCase):
    def test_contains_status_ok(self):
        snap = get_health_snapshot()
        text = format_health_text(snap)
        self.assertIn("Status: OK", text)

    def test_contains_uptime(self):
        snap = get_health_snapshot()
        text = format_health_text(snap)
        self.assertIn("Uptime:", text)

    def test_contains_error_count(self):
        snap = get_health_snapshot()
        text = format_health_text(snap)
        self.assertIn("Errors:", text)

    def test_latency_section_present_when_recorded(self):
        m = Metrics()
        m.record_latency("ask_stuff", 1.23)
        import observability
        original = observability.METRICS
        observability.METRICS = m
        try:
            snap = get_health_snapshot()
            text = format_health_text(snap)
            self.assertIn("ask_stuff", text)
        finally:
            observability.METRICS = original


class TestAuditLogRotation(unittest.TestCase):
    """audit_log was a bare append with no cap. Relay traffic makes that a
    disk-fill bug whose contents are who-messaged-whom."""

    def setUp(self):
        import observability
        self.obs = observability
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "audit.log"
        self._patch = patch.multiple(observability, AUDIT_LOG_PATH=str(self.path),
                                     AUDIT_LOG_MAX_BYTES=400, AUDIT_LOG_BACKUPS=2)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def events(self, path):
        return [json.loads(ln)["n"] for ln in Path(path).read_text(encoding="utf-8").splitlines()]

    def test_writes_under_the_cap_do_not_rotate(self):
        self.obs.audit_log("e", n=1)
        self.obs.audit_log("e", n=2)
        self.assertEqual(self.events(self.path), [1, 2])
        self.assertFalse(Path(f"{self.path}.1").exists())

    def test_crossing_the_cap_moves_the_old_file_aside(self):
        for n in range(20):
            self.obs.audit_log("e", n=n, pad="x" * 40)
        self.assertTrue(Path(f"{self.path}.1").exists())
        self.assertLessEqual(self.path.stat().st_size, 400 + 100)
        # Nothing lost across the boundary: the newest file continues exactly
        # where the backup stops.
        self.assertEqual(self.events(f"{self.path}.1")[-1] + 1, self.events(self.path)[0])

    def test_oldest_backup_falls_off_the_end(self):
        for n in range(200):
            self.obs.audit_log("e", n=n, pad="x" * 40)
        self.assertTrue(Path(f"{self.path}.2").exists())
        self.assertFalse(Path(f"{self.path}.3").exists())
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["audit.log", "audit.log.1", "audit.log.2"])

    def test_one_line_bigger_than_the_cap_is_still_written(self):
        # The file must EXIST and be empty: a missing file returns before the
        # size check, so a test that skips this line cannot tell the guard is
        # there at all.
        self.path.touch()
        self.obs.audit_log("e", n=1, pad="x" * 1000)
        self.assertEqual(self.events(self.path), [1])
        self.assertFalse(Path(f"{self.path}.1").exists(), "rotated an empty file")

    def test_a_failed_rotation_keeps_the_event_and_does_not_raise(self):
        """Windows refuses to rename a file another process has open."""
        for n in range(8):
            self.obs.audit_log("e", n=n, pad="x" * 40)
        with patch.object(self.obs.os, "replace", side_effect=PermissionError("locked")),              self.assertLogs("observability", level="WARNING"):
            self.obs.audit_log("e", n=99, pad="x" * 400)
        self.assertEqual(self.events(self.path)[-1], 99)


class TestAuditKnobParsing(unittest.TestCase):
    def parse(self, raw):
        import observability
        env = {} if raw is None else {"_KNOB": raw}
        with patch.dict(os.environ, env, clear=False):
            if raw is None:
                os.environ.pop("_KNOB", None)
            return observability._positive_int_env("_KNOB", 7)

    def test_unset_uses_the_default(self):
        self.assertEqual(self.parse(None), 7)

    def test_a_real_value_is_used(self):
        self.assertEqual(self.parse("123"), 123)

    def test_zero_is_not_a_way_to_switch_rotation_off(self):
        """A 0-byte cap would rotate every write; 0 backups deletes history."""
        for raw in ("0", "-5", "lots"):
            with self.subTest(raw=raw), self.assertLogs("observability", level="WARNING"):
                self.assertEqual(self.parse(raw), 7)


class TestTestsDoNotWriteTheRealAuditLog(unittest.TestCase):
    def test_audit_log_path_is_sandboxed(self):
        """Every test run used to append to the working tree's audit.log."""
        import observability
        repo = Path(__file__).resolve().parents[1]
        target = Path(observability.AUDIT_LOG_PATH).resolve()
        self.assertNotEqual(target.parent, repo)
        self.assertFalse(str(target).startswith(str(repo) + os.sep))


if __name__ == "__main__":
    unittest.main()
