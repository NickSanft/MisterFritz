"""
mock_fritz.py — Local simulation of MisterFritz for DevOps demonstrations.

Runs a lightweight HTTP server (no Discord connection, no Ollama needed) that:
  • Generates realistic synthetic traffic metrics
  • Exposes /metrics (Prometheus) and /health (JSON)
  • Simulates configurable error rates and latency profiles

Usage:
  FRITZ_VERSION=stable  FRITZ_ERROR_RATE=0.02 python mock_fritz.py
  FRITZ_VERSION=canary  FRITZ_ERROR_RATE=0.12 python mock_fritz.py --port 8001
"""

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Config from environment ──────────────────────────────────────────────────
VERSION = os.environ.get("FRITZ_VERSION", "stable")
# Fraction of requests that result in an error (0.0–1.0)
ERROR_RATE = float(os.environ.get("FRITZ_ERROR_RATE", "0.02"))
# Base latency in seconds (real responses average 1–3 s)
LATENCY_BASE = float(os.environ.get("FRITZ_LATENCY_BASE", "1.5"))
# Interval between simulated user messages (seconds)
MSG_INTERVAL = float(os.environ.get("FRITZ_MSG_INTERVAL", "1.0"))

START_TIME = time.time()

# ── Prometheus metrics ───────────────────────────────────────────────────────
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
    )
    _messages = Counter(
        "misterfritz_discord_messages_total",
        "Discord messages processed",
        ["version"],
    )
    _errors = Counter(
        "misterfritz_errors_total",
        "Errors",
        ["operation", "version"],
    )
    _latency = Histogram(
        "misterfritz_request_duration_seconds",
        "End-to-end request latency",
        ["operation", "version"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    _uptime = Gauge("misterfritz_uptime_seconds", "Uptime", ["version"])
    _tool_calls = Counter(
        "misterfritz_tool_calls_total",
        "Tool invocations",
        ["tool", "version"],
    )
    PROM_OK = True
except ImportError:
    PROM_OK = False
    print("[mock_fritz] WARNING: prometheus_client not installed — /metrics will be empty")

TOOLS = ["search_web", "search_documents", "get_current_time", "roll_dice", "generate_image"]

# ── Simple in-process counters (for /health without prometheus) ──────────────
_state = {"messages": 0, "errors": 0}
_state_lock = threading.Lock()


def _simulate_traffic() -> None:
    """Background thread: emit synthetic metrics at MSG_INTERVAL pace."""
    while True:
        time.sleep(MSG_INTERVAL * random.uniform(0.5, 1.5))

        if PROM_OK:
            _uptime.labels(version=VERSION).set(time.time() - START_TIME)

        is_error = random.random() < ERROR_RATE

        with _state_lock:
            if is_error:
                _state["errors"] += 1
            else:
                _state["messages"] += 1

        if is_error:
            if PROM_OK:
                _errors.labels(operation="ask_stuff", version=VERSION).inc()
        else:
            duration = max(0.05, random.gauss(LATENCY_BASE, LATENCY_BASE * 0.3))
            if PROM_OK:
                _messages.labels(version=VERSION).inc()
                _latency.labels(operation="ask_stuff", version=VERSION).observe(duration)
                # ~30% chance a tool was called
                if random.random() < 0.30:
                    tool = random.choice(TOOLS)
                    _tool_calls.labels(tool=tool, version=VERSION).inc()


# ── HTTP handler ─────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/metrics":
            if PROM_OK:
                body = generate_latest()
                self._send(200, CONTENT_TYPE_LATEST, body)
            else:
                self._send(503, "text/plain", b"prometheus_client not available")

        elif self.path in ("/health", "/healthz"):
            with _state_lock:
                msgs = _state["messages"]
                errs = _state["errors"]
            payload = {
                "status": "ok",
                "version": VERSION,
                "uptime_sec": round(time.time() - START_TIME, 1),
                "messages_processed": msgs,
                "errors": errs,
                "error_rate": round(errs / max(1, msgs + errs), 4),
                "config": {
                    "error_rate_target": ERROR_RATE,
                    "latency_base_sec": LATENCY_BASE,
                },
            }
            self._send(200, "application/json", json.dumps(payload, indent=2).encode())

        elif self.path == "/":
            body = (
                f"<h1>MisterFritz Mock [{VERSION}]</h1>"
                f"<p>error_rate={ERROR_RATE:.0%} | latency_base={LATENCY_BASE}s</p>"
                f"<ul><li><a href='/metrics'>/metrics</a></li>"
                f"<li><a href='/health'>/health</a></li></ul>"
            ).encode()
            self._send(200, "text/html", body)

        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        pass  # suppress per-request noise


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MisterFritz mock server")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("METRICS_PORT", "8000")))
    args = parser.parse_args()

    print(f"[mock_fritz:{VERSION}] error_rate={ERROR_RATE:.0%}  "
          f"latency={LATENCY_BASE}s  port={args.port}")

    # Start traffic simulator
    t = threading.Thread(target=_simulate_traffic, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"[mock_fritz:{VERSION}] serving on http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[mock_fritz:{VERSION}] shutting down")
