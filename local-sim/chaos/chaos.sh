#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# chaos.sh — Fault injection experiments for local simulation
#
# Requires: Docker, docker-compose running (make up / make deploy-canary)
#
# Experiments:
#   ./chaos.sh pod-failure     Kill the canary container; watch it restart
#   ./chaos.sh cpu-stress      Spike canary CPU; observe HPA-like behaviour
#   ./chaos.sh network-delay   Add 500 ms latency on canary; watch P99 climb
#   ./chaos.sh high-errors     Restart canary with a high error rate
#   ./chaos.sh all             Run all experiments sequentially (demo mode)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CANARY_CONTAINER="${CANARY_CONTAINER:-fritz-canary}"
STABLE_CONTAINER="${STABLE_CONTAINER:-fritz-stable}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.canary.yml}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
PROM_URL="${PROM_URL:-http://localhost:9090}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[chaos]${NC} $*"; }
success() { echo -e "${GREEN}[chaos]${NC} $*"; }
warn()    { echo -e "${YELLOW}[chaos]${NC} $*"; }
error()   { echo -e "${RED}[chaos]${NC} $*"; }

wait_seconds() {
    local secs=$1 msg=${2:-"Waiting..."}
    info "$msg ($secs s)"
    sleep "$secs"
}

announce() {
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  CHAOS EXPERIMENT: $1${NC}"
    echo -e "${RED}╚══════════════════════════════════════════╝${NC}"
    echo ""
}

# ── Helpers ──────────────────────────────────────────────────────────────────

container_running() {
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true
}

require_canary() {
    if ! container_running "$CANARY_CONTAINER"; then
        error "Canary container '$CANARY_CONTAINER' is not running."
        error "Run 'make deploy-canary' first."
        exit 1
    fi
}

# ── Experiments ──────────────────────────────────────────────────────────────

experiment_pod_failure() {
    announce "POD FAILURE"
    require_canary

    info "Pausing canary container (simulates pod crash)..."
    docker pause "$CANARY_CONTAINER"

    warn "Canary is DOWN. Check Grafana for error spike: $GRAFANA_URL"
    wait_seconds 20 "Canary paused — observe the error rate spike"

    info "Resuming canary (simulates pod restart / self-heal)..."
    docker unpause "$CANARY_CONTAINER"

    wait_seconds 10 "Canary recovered — observe error rate drop"
    success "Pod failure experiment complete. Metrics should show recovery."
}

experiment_cpu_stress() {
    announce "CPU STRESS"
    require_canary

    info "Running CPU stress inside canary container for 30 s..."
    # Start stress in background inside the container
    docker exec -d "$CANARY_CONTAINER" sh -c \
        "python3 -c 'import time; [sum(range(10**6)) for _ in iter(int, 1)]' &" 2>/dev/null \
        || warn "python3 stress fallback not available — using yes loop"

    docker exec -d "$CANARY_CONTAINER" sh -c "yes > /dev/null &" 2>/dev/null || true

    warn "CPU stress running. Watch latency climb in Grafana: $GRAFANA_URL"
    wait_seconds 30 "CPU stress active"

    info "Killing stress processes in canary..."
    docker exec "$CANARY_CONTAINER" sh -c "pkill -f 'yes\|python3' 2>/dev/null || true"

    wait_seconds 10 "Canary recovering from CPU stress"
    success "CPU stress experiment complete."
}

experiment_network_delay() {
    announce "NETWORK DELAY"
    require_canary

    # pumba is the cleanest tool for network emulation in Docker
    if command -v pumba &>/dev/null; then
        info "Using pumba to inject 500ms network delay on canary for 30 s..."
        pumba netem --duration 30s delay --time 500 "$CANARY_CONTAINER" &
        PUMBA_PID=$!
        wait_seconds 35 "Network delay active — watch P99 latency in Grafana"
        wait $PUMBA_PID 2>/dev/null || true
        success "Network delay experiment complete."
    else
        warn "pumba not found — simulating latency via FRITZ_LATENCY_BASE env var instead"
        info "Restarting canary with 5 s base latency..."
        docker compose -f "$COMPOSE_FILE" stop fritz-canary
        CANARY_LATENCY=5.0 docker compose -f "$COMPOSE_FILE" --profile canary up -d fritz-canary
        wait_seconds 30 "High-latency canary running — watch P99 spike in Grafana"
        info "Restoring normal latency..."
        docker compose -f "$COMPOSE_FILE" stop fritz-canary
        docker compose -f "$COMPOSE_FILE" --profile canary up -d fritz-canary
        success "Network delay experiment complete."
    fi
}

experiment_high_errors() {
    announce "HIGH ERROR RATE"
    require_canary

    info "Restarting canary with 25% error rate (above 5% threshold)..."
    docker compose -f "$COMPOSE_FILE" stop fritz-canary
    CANARY_ERROR_RATE=0.25 docker compose -f "$COMPOSE_FILE" --profile canary up -d fritz-canary

    warn "High error rate active. The canary controller should detect this."
    warn "Run 'make watch-canary' in another terminal to see auto-rollback."
    wait_seconds 40 "High-error canary running"

    info "Restoring normal error rate..."
    docker compose -f "$COMPOSE_FILE" stop fritz-canary
    docker compose -f "$COMPOSE_FILE" --profile canary up -d fritz-canary
    success "High error experiment complete."
}

# ── Main ─────────────────────────────────────────────────────────────────────

experiment="${1:-help}"

case "$experiment" in
    pod-failure)     experiment_pod_failure ;;
    cpu-stress)      experiment_cpu_stress ;;
    network-delay)   experiment_network_delay ;;
    high-errors)     experiment_high_errors ;;
    all)
        experiment_pod_failure
        wait_seconds 15 "Cooldown between experiments"
        experiment_high_errors
        wait_seconds 15 "Cooldown between experiments"
        experiment_cpu_stress
        success "All chaos experiments finished. Check Grafana for the full story."
        ;;
    *)
        echo ""
        echo "Usage: $0 <experiment>"
        echo ""
        echo "  pod-failure     Pause canary container; observe restart & recovery"
        echo "  cpu-stress      Spike CPU inside canary; watch latency climb"
        echo "  network-delay   Add 500 ms delay (requires pumba, falls back to env)"
        echo "  high-errors     Restart canary with 25% error rate; triggers rollback"
        echo "  all             Run all experiments with cooldowns (demo mode)"
        echo ""
        ;;
esac
