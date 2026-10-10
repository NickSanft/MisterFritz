import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import observability
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


class TestTheMetricsListenerStaysOffTheNetwork(unittest.TestCase):
    """/metrics is the whole counter set and /health the status snapshot, both
    unauthenticated. This used to bind 0.0.0.0 with no knob, so on a laptop or
    a home server it was readable by anyone on the network.

    HTTPServer is patched rather than really bound: the question is which
    address it is asked for, and binding for real would leave a serve_forever
    thread behind in every test.
    """

    def _started(self, env=None, **kwargs):
        env = env or {}
        with patch.object(observability, "HTTPServer") as server:
            with patch.dict(os.environ, env, clear=False):
                for name in ("METRICS_HOST", "METRICS_PORT"):
                    if name not in env:
                        os.environ.pop(name, None)
                with patch.object(observability.threading, "Thread"):
                    observability.start_metrics_server(**kwargs)
        return server.call_args.args[0]

    def test_the_default_is_localhost(self):
        host, _port = self._started()
        self.assertEqual(host, "127.0.0.1")

    def test_the_default_port_is_unchanged(self):
        _host, port = self._started()
        self.assertEqual(port, 8000)

    def test_the_environment_can_open_it_up(self):
        host, _port = self._started(env={"METRICS_HOST": "0.0.0.0"})
        self.assertEqual(host, "0.0.0.0")

    def test_an_explicit_argument_wins_over_the_environment(self):
        host, port = self._started(env={"METRICS_HOST": "0.0.0.0",
                                        "METRICS_PORT": "9999"},
                                   host="10.0.0.5", port=1234)
        self.assertEqual((host, port), ("10.0.0.5", 1234))

    def test_the_log_says_which_interface_it_took(self):
        """An operator who set the knob should be able to confirm it, and one
        who did not should be able to see that it is local."""
        with patch.object(observability, "HTTPServer"):
            with patch.object(observability.threading, "Thread"):
                with self.assertLogs("observability", level="INFO") as caught:
                    observability.start_metrics_server(host="127.0.0.1", port=8123)
        self.assertIn("127.0.0.1:8123", "; ".join(caught.output))


class TestTheContainerOpensItDeliberately(unittest.TestCase):
    """A container has to bind every interface, and the reason belongs beside
    the line that does it — otherwise it reads as the default being
    overridden for nothing, and gets "fixed"."""

    REPO = Path(__file__).resolve().parents[1]

    def _dockerfile(self) -> str:
        return (self.REPO / "Dockerfile").read_text(encoding="utf-8")

    def test_the_image_binds_every_interface(self):
        self.assertIn("ENV METRICS_HOST=0.0.0.0", self._dockerfile())

    def test_the_image_says_why(self):
        dockerfile = self._dockerfile()
        above = dockerfile[:dockerfile.index("ENV METRICS_HOST")]
        reason = above.rsplit(chr(10) + chr(10), 1)[-1]
        self.assertIn("unauthenticated", reason)
        for who in ("Prometheus", "probes"):
            self.assertIn(who, reason)

    def test_prometheus_still_scrapes_across_the_network(self):
        """If the scrape target were loopback, the localhost default would be
        enough and the override unnecessary. It is not."""
        scrape = (self.REPO / "infra/prometheus/prometheus.yml").read_text(encoding="utf-8")
        self.assertIn("misterfritz:8000", scrape)
        self.assertNotIn("127.0.0.1:8000", scrape)

    def test_the_kubernetes_probes_still_use_the_metrics_port(self):
        import yaml
        for document in yaml.safe_load_all(
                (self.REPO / "infra/k8s/deployment.yaml").read_text(encoding="utf-8")):
            if not document or document.get("kind") != "Deployment":
                continue
            for container in document["spec"]["template"]["spec"]["containers"]:
                for probe in ("livenessProbe", "readinessProbe"):
                    with self.subTest(probe=probe):
                        self.assertEqual(container[probe]["httpGet"]["port"], 8000)


class TestTheEndpointsAnOperatorActuallyHits(unittest.TestCase):
    """_MetricsHandler had never served a request. README documents both
    endpoints, Prometheus scrapes one and the Kubernetes probes hit the other,
    and nothing had ever asked either of them for anything.

    A real HTTPServer on port 0 in a thread, shut down afterwards: the handler
    is HTTP machinery, and asserting against a fake request object would be
    asserting against my own idea of one.
    """

    @classmethod
    def setUpClass(cls):
        import http.server
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), observability._MetricsHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path: str):
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_health_answers_with_the_snapshot_the_probes_read(self):
        status, headers, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        for key in ("uptime_sec", "discord_messages", "total_errors"):
            self.assertIn(key, payload)

    def test_healthz_is_the_same_endpoint(self):
        """Kubernetes conventions vary and the manifests could use either."""
        self.assertEqual(self._get("/healthz")[0], 200)

    def test_health_counts_the_errors_it_has_been_told_about(self):
        before = json.loads(self._get("/health")[2])["total_errors"]
        observability.METRICS.record_error("e3_probe", RuntimeError("boom"))
        after = json.loads(self._get("/health")[2])["total_errors"]
        self.assertEqual(after, before + 1)

    def test_metrics_answers_in_the_format_prometheus_parses(self):
        status, headers, body = self._get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", headers["Content-Type"])
        self.assertIn(b"misterfritz", body)

    def test_metrics_says_so_rather_than_lying_when_prometheus_is_absent(self):
        """A 200 with an empty body would read to a scrape as "no metrics",
        which is indistinguishable from a healthy, idle bot."""
        with patch.object(observability, "_PROMETHEUS_AVAILABLE", False):
            status, _headers, body = self._get("/metrics")
        self.assertEqual(status, 503)
        self.assertIn(b"prometheus_client not installed", body)

    def test_anything_else_is_a_404(self):
        status, _headers, body = self._get("/../secrets")
        self.assertEqual(status, 404)
        self.assertEqual(body, b"not found")

    def test_the_content_length_matches_the_body(self):
        """_respond sets it by hand, and a wrong one makes a client hang
        waiting for bytes that never arrive."""
        for path in ("/health", "/metrics"):
            with self.subTest(path=path):
                _status, headers, body = self._get(path)
                self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_requests_are_not_written_to_the_access_log(self):
        """log_message is overridden to silence BaseHTTPRequestHandler, which
        otherwise writes a line to stderr per scrape — every fifteen seconds,
        forever."""
        import io
        import contextlib
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self._get("/health")
        self.assertEqual(stderr.getvalue(), "")


class TestTheServerThreadIsStarted(unittest.TestCase):
    """start_metrics_server's own body: the serving thread, and the side thread
    that keeps the uptime gauge current. Both are closures, so the targets are
    taken from where production hands them to Thread()."""

    class _FakeThread:
        """Runs the server thread inline and records the rest."""

        made = []

        def __init__(self, target=None, name=None, daemon=None):
            self.target, self.name = target, name
            type(self).made.append(self)

        def start(self):
            if self.name == "metrics-server":
                self.target()

    def setUp(self):
        self._FakeThread.made = []
        self.server = MagicMock()
        patcher = patch.object(observability, "HTTPServer", return_value=self.server)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(observability.threading, "Thread", self._FakeThread)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_server_is_told_to_serve(self):
        observability.start_metrics_server(host="127.0.0.1", port=8321)
        self.server.serve_forever.assert_called_once()

    def test_the_serving_thread_is_a_daemon_named_for_itself(self):
        """A non-daemon thread here would keep the process alive after the bot
        has stopped."""
        observability.start_metrics_server(host="127.0.0.1", port=8321)
        names = [t.name for t in self._FakeThread.made]
        self.assertIn("metrics-server", names)

    def test_the_uptime_gauge_is_kept_current(self):
        """The gauge is a snapshot, so without this side thread it reports the
        uptime at the moment the server started, forever."""
        observability.start_metrics_server(host="127.0.0.1", port=8321)
        updaters = [t for t in self._FakeThread.made if t.name != "metrics-server"]
        self.assertTrue(updaters, "no uptime updater was started")

        # Run one iteration of the updater's loop, taking the target from where
        # start_metrics_server handed it over, and break out of the `while
        # True` the way nothing in production does.
        class _Stop(Exception):
            pass

        with patch.object(observability, "_PROM_UPTIME") as gauge:
            with patch.object(observability.time, "sleep", side_effect=_Stop):
                with self.assertRaises(_Stop):
                    updaters[0].target()
        gauge.set.assert_called_once()
        self.assertGreaterEqual(gauge.set.call_args.args[0], 0)

    def test_no_uptime_thread_without_prometheus(self):
        with patch.object(observability, "_PROMETHEUS_AVAILABLE", False):
            observability.start_metrics_server(host="127.0.0.1", port=8321)
        self.assertEqual([t.name for t in self._FakeThread.made], ["metrics-server"])


class TestItDegradesWithoutPrometheus(unittest.TestCase):
    """prometheus-client is a core dependency, so this branch cannot run in
    this process - and reloading observability to force it would hand every
    other module a different METRICS object than the one it imported. A
    subprocess instead, which means the two lines stay off the coverage report
    while the property they carry is still asserted.
    """

    def test_the_module_still_imports_and_says_so(self):
        import subprocess
        program = (
            "import sys; sys.modules['prometheus_client'] = None; "
            "import observability; "
            "print(observability._PROMETHEUS_AVAILABLE)"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False")


class TestTheJsonLogFormat(unittest.TestCase):
    """LOG_FORMAT=json is what the container runs on, and the formatter had
    never formatted anything. A broken one is not a cosmetic problem: it is a
    log pipeline that drops or mangles every line it is given."""

    def _record(self, **kwargs):
        return logging.LogRecord(
            name=kwargs.get("name", "fritz.test"),
            level=kwargs.get("level", logging.INFO),
            pathname=__file__, lineno=1,
            msg=kwargs.get("msg", "hello %s"), args=kwargs.get("args", ("world",)),
            exc_info=kwargs.get("exc_info"),
        )

    def test_a_record_is_one_line_of_json(self):
        formatted = observability._JsonFormatter().format(self._record())
        self.assertNotIn(chr(10), formatted)
        payload = json.loads(formatted)
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "fritz.test")
        self.assertEqual(payload["msg"], "hello world")
        self.assertIn("ts", payload)

    def test_the_message_is_interpolated_not_left_as_a_template(self):
        """%-style args are the whole house style of this codebase's logging."""
        payload = json.loads(observability._JsonFormatter().format(
            self._record(msg="removed %d rows for %s", args=(3, "discord-1"))))
        self.assertEqual(payload["msg"], "removed 3 rows for discord-1")

    def test_a_traceback_travels_with_the_record(self):
        try:
            raise ValueError("the actual cause")
        except ValueError:
            record = self._record(exc_info=sys.exc_info(), level=logging.ERROR)
        payload = json.loads(observability._JsonFormatter().format(record))
        self.assertIn("the actual cause", payload["exc"])
        self.assertIn("ValueError", payload["exc"])

    def test_stack_info_travels_too(self):
        record = self._record()
        record.stack_info = "Stack (most recent call last):\n  fake frame"
        payload = json.loads(observability._JsonFormatter().format(record))
        self.assertIn("fake frame", payload["stack"])


class TestLoggingIsInitialisedOnce(unittest.TestCase):
    """init_logging touches the ROOT logger, so each test here puts it back."""

    def setUp(self):
        root = logging.getLogger()
        self.saved = (root.handlers[:], root.level)
        self.addCleanup(self._restore)
        root.handlers = []

    def _restore(self):
        root = logging.getLogger()
        root.handlers, root.level = self.saved

    def test_json_format_installs_the_json_formatter(self):
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            observability.init_logging()
        [handler] = logging.getLogger().handlers
        self.assertIsInstance(handler.formatter, observability._JsonFormatter)

    def test_the_default_is_the_human_readable_format(self):
        with patch.dict(os.environ, {"LOG_FORMAT": ""}):
            observability.init_logging()
        [handler] = logging.getLogger().handlers
        self.assertNotIsInstance(handler.formatter, observability._JsonFormatter)

    def test_the_level_comes_from_the_environment(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            observability.init_logging()
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_a_nonsense_level_falls_back_to_info(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "LOUD"}):
            observability.init_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_calling_it_again_does_not_double_every_line(self):
        """Every module calls it, so without the guard one log line would be
        emitted once per import that got there first."""
        observability.init_logging()
        observability.init_logging()
        self.assertEqual(len(logging.getLogger().handlers), 1)


class TestAnAuditLineThatCannotBeEncoded(unittest.TestCase):
    """audit_log is the record /forget and /export leave behind, and it is
    best-effort by design: it must never raise into a deletion that already
    happened."""

    def test_an_unencodable_field_is_logged_and_dropped(self):
        circular: dict = {}
        circular["self"] = circular
        with self.assertLogs("observability", level="WARNING") as caught:
            observability.audit_log("forget", user_id="discord-1", detail=circular)
        self.assertIn("audit_log JSON encode failed", "; ".join(caught.output))

    def test_it_does_not_raise(self):
        circular: dict = {}
        circular["self"] = circular
        observability.audit_log("forget", detail=circular)      # must not raise
