#!/usr/bin/env python3
"""
canary_controller.py — Automated canary deployment controller.

Queries Prometheus to decide whether to promote or roll back the canary.

Usage:
  # Watch-only (print metrics every 30s, no action):
  python canary_controller.py

  # Fully automated (promote or rollback without confirmation):
  python canary_controller.py --auto

  # Single check then exit:
  python canary_controller.py --once
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
ERROR_RATE_THRESHOLD = float(os.environ.get("CANARY_ERROR_THRESHOLD", "0.05"))   # 5%
LATENCY_P99_THRESHOLD = float(os.environ.get("CANARY_LATENCY_THRESHOLD", "8.0")) # 8 s
ANALYSIS_INTERVAL = int(os.environ.get("CANARY_INTERVAL", "20"))    # seconds
PROMOTE_AFTER_CLEAN = int(os.environ.get("CANARY_CLEAN_WINDOWS", "3"))  # windows


# ── Prometheus query helpers ─────────────────────────────────────────────────

def _query(promql: str) -> float | None:
    url = f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(promql)}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
    except Exception as exc:
        print(f"  [prometheus] query failed: {exc}")
    return None


def error_rate(version: str) -> float:
    """Fraction of requests that resulted in an error over the last minute."""
    errors = _query(
        f'sum(rate(misterfritz_errors_total{{version="{version}"}}[1m]))'
    ) or 0.0
    msgs = _query(
        f'sum(rate(misterfritz_discord_messages_total{{version="{version}"}}[1m]))'
    ) or 0.0
    total = errors + msgs
    return errors / total if total > 0 else 0.0


def latency_p99(version: str) -> float:
    """P99 ask_stuff latency over the last minute (seconds)."""
    return _query(
        f'histogram_quantile(0.99, '
        f'rate(misterfritz_request_duration_seconds_bucket'
        f'{{operation="ask_stuff",version="{version}"}}[1m]))'
    ) or 0.0


def throughput(version: str) -> float:
    """Messages per second over the last minute."""
    return _query(
        f'sum(rate(misterfritz_discord_messages_total{{version="{version}"}}[1m]))'
    ) or 0.0


# ── Docker Compose actions ───────────────────────────────────────────────────

def _make(target: str) -> None:
    subprocess.run(["make", target], cwd=os.path.dirname(__file__), check=True)


def promote() -> None:
    print("\n✅  PROMOTING canary → stable")
    _make("promote")


def rollback() -> None:
    print("\n❌  ROLLING BACK canary → stable")
    _make("rollback")


# ── Main controller loop ─────────────────────────────────────────────────────

def _print_metrics(version: str) -> tuple[float, float, float]:
    er = error_rate(version)
    p99 = latency_p99(version)
    thr = throughput(version)
    er_ok = "✅" if er < ERROR_RATE_THRESHOLD else "❌"
    p99_ok = "✅" if p99 < LATENCY_P99_THRESHOLD else "❌"
    print(f"  {er_ok} error_rate  = {er:6.2%}   (threshold {ERROR_RATE_THRESHOLD:.0%})")
    print(f"  {p99_ok} P99 latency = {p99:6.2f}s  (threshold {LATENCY_P99_THRESHOLD:.0f}s)")
    print(f"     throughput  = {thr:.2f} msg/s")
    return er, p99, thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Canary deployment controller")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically promote or roll back based on thresholds")
    parser.add_argument("--once", action="store_true",
                        help="Run one analysis window then exit")
    parser.add_argument("--version", default="canary",
                        help="Version label to analyse (default: canary)")
    args = parser.parse_args()

    print("=" * 60)
    print(" MisterFritz Canary Controller")
    print(f"  Prometheus : {PROMETHEUS_URL}")
    print(f"  Target     : {args.version}")
    print(f"  Thresholds : error_rate < {ERROR_RATE_THRESHOLD:.0%}  "
          f"| P99 < {LATENCY_P99_THRESHOLD:.0f}s")
    print(f"  Auto       : {args.auto}")
    print("=" * 60)

    clean_windows = 0
    window = 0

    while True:
        window += 1
        now = time.strftime("%H:%M:%S")
        print(f"\n[{now}] Window {window} — analysing '{args.version}' ...")

        er, p99, _ = _print_metrics(args.version)

        unhealthy = er >= ERROR_RATE_THRESHOLD or p99 >= LATENCY_P99_THRESHOLD

        if unhealthy:
            clean_windows = 0
            print(f"  ⚠️  Unhealthy. Clean windows reset to 0.")
            if args.auto:
                rollback()
                sys.exit(1)
        else:
            clean_windows += 1
            remaining = PROMOTE_AFTER_CLEAN - clean_windows
            print(f"  Clean window {clean_windows}/{PROMOTE_AFTER_CLEAN}"
                  + (f" — {remaining} more before promotion" if remaining > 0 else ""))
            if args.auto and clean_windows >= PROMOTE_AFTER_CLEAN:
                promote()
                sys.exit(0)

        if args.once:
            break

        print(f"  (next check in {ANALYSIS_INTERVAL}s — Ctrl-C to stop)")
        time.sleep(ANALYSIS_INTERVAL)


if __name__ == "__main__":
    main()
