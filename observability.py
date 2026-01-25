import logging
import os
import threading
import time
from collections import deque

_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def init_logging() -> None:
    """Initialize application-wide logging once."""
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


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

    def record_latency(self, name: str, duration_sec: float, max_samples: int = 200) -> None:
        with self._lock:
            samples = self._latencies.get(name)
            if samples is None:
                samples = deque(maxlen=max_samples)
                self._latencies[name] = samples
            samples.append(duration_sec)

    def record_error(self, name: str, error: Exception | str) -> None:
        message = str(error)
        with self._lock:
            self._errors[name] = self._errors.get(name, 0) + 1
            self._last_error = (name, time.time(), message)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            errors = dict(self._errors)
            latencies = {key: list(values) for key, values in self._latencies.items()}
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
    counters = snapshot["counters"]
    errors = snapshot["errors"]
    latencies = snapshot["latencies"]
    last_error = snapshot["last_error"]

    latency_summary = {}
    for name, samples in latencies.items():
        if samples:
            avg = sum(samples) / len(samples)
            latency_summary[name] = {"avg_sec": avg, "count": len(samples)}

    return {
        "uptime_sec": uptime,
        "counters": counters,
        "errors": errors,
        "latencies": latency_summary,
        "last_error": last_error,
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
        avg = summary["avg_sec"]
        count = summary["count"]
        lines.append(f"Latency {name}: {avg:.2f}s (n={count})")

    return "\n".join(lines)
