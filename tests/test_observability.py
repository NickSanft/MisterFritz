import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
