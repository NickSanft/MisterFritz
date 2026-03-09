# ─────────────────────────────────────────────────────────────────────────────
# MisterFritz Local Canary Simulation — Makefile
#
# All commands run from the local-sim/ directory.
# ─────────────────────────────────────────────────────────────────────────────

COMPOSE      := docker compose -f docker-compose.canary.yml
CONTROLLER   := python canary_controller.py
CHAOS        := bash chaos/chaos.sh

# Traffic weights (must sum to any value; nginx uses relative weights)
STABLE_WEIGHT  ?= 10
CANARY_WEIGHT  ?= 0

# Error rates for the mock services
STABLE_ERROR_RATE  ?= 0.02   # 2%  — healthy stable
CANARY_ERROR_RATE  ?= 0.02   # 2%  — healthy canary (override to inject faults)
CANARY_LATENCY     ?= 1.5

.DEFAULT_GOAL := help

.PHONY: help build up deploy-canary promote rollback down \
        watch-canary watch-canary-auto \
        chaos-pod chaos-errors chaos-cpu chaos-all \
        logs metrics open-grafana open-prometheus status

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  MisterFritz Canary Simulation"
	@echo "  ════════════════════════════"
	@echo ""
	@echo "  Setup"
	@echo "    make build          Build the mock_fritz image"
	@echo ""
	@echo "  Deployment lifecycle"
	@echo "    make up             Start stable-only stack (100% stable)"
	@echo "    make deploy-canary  Launch canary at 10% traffic"
	@echo "    make promote        Shift to 100% canary (stable ← canary)"
	@echo "    make rollback       Revert to 100% stable, stop canary"
	@echo "    make down           Tear down everything"
	@echo ""
	@echo "  Canary analysis"
	@echo "    make watch-canary       Print metrics every 20 s (no action)"
	@echo "    make watch-canary-auto  Auto promote or rollback"
	@echo ""
	@echo "  Chaos experiments"
	@echo "    make chaos-pod      Kill canary; watch self-healing"
	@echo "    make chaos-errors   Inject 25% error rate; triggers rollback"
	@echo "    make chaos-cpu      Spike canary CPU; watch latency"
	@echo "    make chaos-all      Full demo (all experiments with cooldowns)"
	@echo ""
	@echo "  Observability"
	@echo "    make logs           Stream logs from all containers"
	@echo "    make metrics        Print raw Prometheus metrics"
	@echo "    make status         Show container status + quick health check"
	@echo "    make open-grafana   Open Grafana in browser (port 3000)"
	@echo "    make open-prometheus Open Prometheus in browser (port 9090)"
	@echo ""

# ── Build ────────────────────────────────────────────────────────────────────
build:
	@echo "Building mock_fritz image..."
	$(COMPOSE) build

# ── Deployment lifecycle ─────────────────────────────────────────────────────
up: build
	@echo "▶  Starting stable-only stack (100% traffic to stable)..."
	STABLE_WEIGHT=10 CANARY_WEIGHT=0 \
	STABLE_ERROR_RATE=$(STABLE_ERROR_RATE) \
	$(COMPOSE) --profile stable up -d
	@echo ""
	@echo "  Stable  → http://localhost:8001/health"
	@echo "  Proxy   → http://localhost:8080/health"
	@echo "  Grafana → http://localhost:3000  (admin/admin)"
	@echo "  Prom    → http://localhost:9090"
	@echo ""
	@echo "Run 'make deploy-canary' when ready to introduce the canary."

deploy-canary: build
	@echo "▶  Deploying canary at 10% traffic..."
	STABLE_WEIGHT=9 CANARY_WEIGHT=1 \
	STABLE_ERROR_RATE=$(STABLE_ERROR_RATE) \
	CANARY_ERROR_RATE=$(CANARY_ERROR_RATE) \
	CANARY_LATENCY=$(CANARY_LATENCY) \
	$(COMPOSE) --profile canary up -d
	@echo ""
	@echo "  Stable  → http://localhost:8001/health  (90% traffic)"
	@echo "  Canary  → http://localhost:8002/health  (10% traffic)"
	@echo "  Proxy   → http://localhost:8080         (load-balanced)"
	@echo ""
	@echo "Watch:  make watch-canary"
	@echo "Auto:   make watch-canary-auto"

promote:
	@echo "▶  Promoting canary → new stable (100% traffic)..."
	STABLE_WEIGHT=0 CANARY_WEIGHT=10 \
	$(COMPOSE) --profile canary up -d nginx
	$(COMPOSE) stop fritz-stable
	@echo "✅  Canary is now serving 100% of traffic."
	@echo "    Rename the canary image to 'stable' for the next cycle."

rollback:
	@echo "▶  Rolling back: killing canary, restoring stable (100%)..."
	$(COMPOSE) stop fritz-canary 2>/dev/null || true
	STABLE_WEIGHT=10 CANARY_WEIGHT=0 \
	$(COMPOSE) --profile stable up -d nginx fritz-stable
	@echo "✅  Rolled back. Stable is serving 100% of traffic."

down:
	$(COMPOSE) --profile full down --remove-orphans
	@echo "Stack stopped."

# ── Canary analysis ──────────────────────────────────────────────────────────
watch-canary:
	@echo "Watching canary metrics (Ctrl-C to stop)..."
	$(CONTROLLER) --watch --version canary

watch-canary-auto:
	@echo "Auto canary controller — will promote or rollback automatically..."
	$(CONTROLLER) --auto --version canary

# ── Chaos experiments ────────────────────────────────────────────────────────
chaos-pod:
	$(CHAOS) pod-failure

chaos-errors:
	@echo "Injecting high error rate into canary (25%)..."
	CANARY_ERROR_RATE=0.25 $(MAKE) deploy-canary
	@echo "Monitor with:  make watch-canary-auto"

chaos-cpu:
	$(CHAOS) cpu-stress

chaos-all:
	@echo "Running full chaos demo..."
	$(CHAOS) all

# ── Observability ────────────────────────────────────────────────────────────
logs:
	$(COMPOSE) --profile full logs -f --tail=50

metrics:
	@echo "=== Stable metrics ==="
	@curl -s http://localhost:8001/metrics | grep misterfritz || echo "(not running)"
	@echo ""
	@echo "=== Canary metrics ==="
	@curl -s http://localhost:8002/metrics | grep misterfritz || echo "(not running)"

status:
	@echo "=== Container status ==="
	@docker ps --filter "name=fritz" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== Health checks ==="
	@curl -s http://localhost:8001/health 2>/dev/null | python3 -m json.tool || echo "Stable: offline"
	@curl -s http://localhost:8002/health 2>/dev/null | python3 -m json.tool || echo "Canary: offline"

open-grafana:
	@python3 -c "import webbrowser; webbrowser.open('http://localhost:3000')" 2>/dev/null \
		|| echo "Open http://localhost:3000 in your browser"

open-prometheus:
	@python3 -c "import webbrowser; webbrowser.open('http://localhost:9090')" 2>/dev/null \
		|| echo "Open http://localhost:9090 in your browser"
