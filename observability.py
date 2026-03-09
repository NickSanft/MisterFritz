import json
import logging
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

logger = logging.getLogger(__name__)

# ── Optional Prometheus integration ──────────────────────────────────────────
# prometheus-client is listed in requirements.txt. If somehow absent, metrics
# are still tracked internally; Prometheus scraping just won't be available.
try:
    import prometheus_client as _prom
    from prometheus_client import (
        Counter, Gauge, Histogram,
        CONTENT_TYPE_LATEST, generate_latest,
        REGISTRY,
    )

    _PROM_TOOL_CALLS = Counter(
        "misterfritz_tool_calls_total",
        "LangChain tool invocations",
        ["tool"],
    )
    _PROM_DISCORD_MESSAGES = Counter(
        "misterfritz_discord_messages_total",
        "Discord messages processed by the bot",
    )
    _PROM_ERRORS = Counter(
        "misterfritz_errors_total",
        "Errors recorded by the bot",
        ["operation"],
    )
    _PROM_LATENCY = Histogram(
        "misterfritz_request_duration_seconds",
        "End-to-end request latency",
        ["operation"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _PROM_UPTIME = Gauge(
        "misterfritz_uptime_seconds",
        "Bot uptime in seconds",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ── JSON log formatter ────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON — compatible with Loki/ELK ingest."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload)


# ── Logging initialisation ────────────────────────────────────────────────────

def init_logging() -> None:
    """Initialise application-wide logging once.

    Set LOG_LEVEL env var to control verbosity (default INFO).
    Set LOG_FORMAT=json to emit structured JSON — useful in containers / Loki.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


# ── Internal metrics store ────────────────────────────────────────────────────

class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._errors: dict[str, int] = {}
        self._last_error: tuple[str, float, str] | None = None
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
        # Mirror into Prometheus
        if _PROMETHEUS_AVAILABLE:
            if name == "discord_messages":
                _PROM_DISCORD_MESSAGES.inc(value)
            elif name.startswith("tool."):
                _PROM_TOOL_CALLS.labels(tool=name[5:]).inc(value)

    def record_latency(self, name: str, duration_sec: float, max_samples: int = 200) -> None:
        with self._lock:
            samples = self._latencies.get(name)
            if samples is None:
                samples = deque(maxlen=max_samples)
                self._latencies[name] = samples
            samples.append(duration_sec)
        if _PROMETHEUS_AVAILABLE:
            _PROM_LATENCY.labels(operation=name).observe(duration_sec)

    def record_error(self, name: str, error: Exception | str) -> None:
        message = str(error)
        with self._lock:
            self._errors[name] = self._errors.get(name, 0) + 1
            self._last_error = (name, time.time(), message)
        if _PROMETHEUS_AVAILABLE:
            _PROM_ERRORS.labels(operation=name).inc()

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            errors = dict(self._errors)
            latencies = {k: list(v) for k, v in self._latencies.items()}
            last_error = self._last_error
            start_time = self._start_time
        return {
            "start_time": start_time,
            "counters": counters,
            "errors": errors,
            "latencies": latencies,
            "last_error": last_error,
        }


METRICS = Metrics()


# ── HTTP servers (metrics + health) ──────────────────────────────────────────

class _MetricsHandler(BaseHTTPRequestHandler):
    """Serves /metrics (Prometheus) and /health (JSON)."""

    def do_GET(self) -> None:
        if self.path == "/metrics":
            if _PROMETHEUS_AVAILABLE:
                body = generate_latest()
                self._respond(200, CONTENT_TYPE_LATEST, body)
            else:
                self._respond(503, "text/plain", b"prometheus_client not installed")
        elif self.path in ("/health", "/healthz"):
            snapshot = get_health_snapshot()
            body = json.dumps({
                "status": "ok",
                "uptime_sec": round(snapshot["uptime_sec"], 1),
                "discord_messages": snapshot["counters"].get("discord_messages", 0),
                "total_errors": sum(snapshot["errors"].values()),
            }).encode()
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:  # silence default access log
        pass


def start_metrics_server(port: int | None = None) -> int:
    """Start a background HTTP server for Prometheus scraping and health checks.

    Serves:
      GET /metrics  — Prometheus text format
      GET /health   — JSON health payload

    Returns the port the server is listening on.
    """
    port = port or int(os.getenv("METRICS_PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), _MetricsHandler)

    def _run() -> None:
        if _PROMETHEUS_AVAILABLE:
            # Keep uptime gauge current in a side-thread loop
            def _uptime_updater():
                while True:
                    _PROM_UPTIME.set(time.time() - METRICS._start_time)
                    time.sleep(15)
            threading.Thread(target=_uptime_updater, daemon=True).start()
        server.serve_forever()

    t = threading.Thread(target=_run, name="metrics-server", daemon=True)
    t.start()
    logger.info(
        "Metrics server started on port %d  "
        "(/metrics for Prometheus, /health for status)",
        port,
    )
    return port


# ── Health snapshot helpers ───────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d {hrs}h {mins}m {sec}s"
    if hrs:
        return f"{hrs}h {mins}m {sec}s"
    if mins:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def get_health_snapshot() -> dict:
    snapshot = METRICS.snapshot()
    now = time.time()
    uptime = now - snapshot["start_time"]
    latency_summary = {}
    for name, samples in snapshot["latencies"].items():
        if samples:
            avg = sum(samples) / len(samples)
            latency_summary[name] = {"avg_sec": avg, "count": len(samples)}
    return {
        "uptime_sec": uptime,
        "counters": snapshot["counters"],
        "errors": snapshot["errors"],
        "latencies": latency_summary,
        "last_error": snapshot["last_error"],
    }


def format_health_text(snapshot: dict) -> str:
    uptime = _format_duration(snapshot["uptime_sec"])
    counters = snapshot["counters"]
    errors = snapshot["errors"]
    latencies = snapshot["latencies"]
    last_error = snapshot["last_error"]

    total_errors = sum(errors.values()) if errors else 0
    last_error_text = "none"
    if last_error:
        name, ts, message = last_error
        age = _format_duration(time.time() - ts)
        last_error_text = f"{name} {age} ago: {message}"

    lines = [
        "Status: OK",
        f"Uptime: {uptime}",
        f"Discord messages: {counters.get('discord_messages', 0)}",
        f"Errors: {total_errors} (last: {last_error_text})",
    ]
    for name, summary in latencies.items():
        lines.append(f"Latency {name}: {summary['avg_sec']:.2f}s (n={summary['count']})")
    return "\n".join(lines)
