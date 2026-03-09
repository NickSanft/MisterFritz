# MisterFritz — Local DevOps Simulation

Simulate a production-grade canary deployment pipeline **without a Discord token, Ollama instance, or cloud cluster**.

`mock_fritz.py` replaces the real bot with a lightweight HTTP server that generates synthetic Prometheus metrics. Every DevOps pattern — traffic splitting, automated promotion/rollback, chaos engineering, and Grafana dashboards — works identically to production.

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Docker + Docker Compose v2 | Run all services | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Python 3.10+ | Canary controller, chaos scripts | [python.org](https://www.python.org/downloads/) |
| `prometheus-client` (pip) | Controller queries Prometheus | `pip install prometheus-client requests` |
| `kind` + `kubectl` + `helm` | K8s simulation (optional) | See [Kind setup](#kubernetes-simulation-optional) |

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │       nginx :8080             │
                        │  (weighted upstream)          │
                        └────────┬─────────────┬────────┘
                     90% weight  │             │  10% weight
                    ┌────────────▼──┐     ┌────▼──────────────┐
                    │ fritz-stable  │     │  fritz-canary     │
                    │  :8001/metrics│     │  :8002/metrics    │
                    └──────┬────────┘     └────────┬──────────┘
                           │                        │
                    ┌──────▼────────────────────────▼──────┐
                    │         Prometheus :9090              │
                    │   (scrapes both every 5 s)            │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │         Grafana :3000                 │
                    │   "MisterFritz — Canary Deployment"  │
                    └──────────────────────────────────────┘
```

---

## Quick Start

All commands run from the `local-sim/` directory.

### 1. Start the stable stack

```bash
cd local-sim
make up
```

Services started:

| Service | URL | Description |
|---------|-----|-------------|
| fritz-stable | http://localhost:8001/health | Stable mock bot (100% traffic) |
| nginx proxy | http://localhost:8080 | Load-balanced entry point |
| Prometheus | http://localhost:9090 | Metrics scraper |
| Grafana | http://localhost:3000 | Dashboard (admin / admin) |

### 2. Open Grafana

```bash
make open-grafana
```

Navigate to **Dashboards → MisterFritz → MisterFritz — Canary Deployment**.

You should see steady green metrics for the stable version.

### 3. Deploy the canary

```bash
make deploy-canary
```

This starts `fritz-canary` at **10% traffic** (nginx weight 1:9). Both versions now appear as separate series in the Grafana panels.

```bash
make status   # container health check
make metrics  # raw Prometheus output
```

### 4. Watch or auto-control the canary

**Watch mode** — print metrics every 20 s, no action taken:
```bash
make watch-canary
```

**Auto mode** — promote if healthy for 3 consecutive windows, rollback on any threshold breach:
```bash
make watch-canary-auto
```

Default promotion thresholds (set in `canary_controller.py`):
- Error rate < 5%
- P99 latency < 8 s

### 5. Promote or rollback manually

```bash
make promote    # canary → 100%, stable stopped
make rollback   # stable → 100%, canary stopped
```

### 6. Tear down

```bash
make down
```

---

## Chaos Experiments

Inject faults to validate that the canary controller correctly rolls back.

### Pod failure (self-healing)

```bash
make chaos-pod
```

Pauses `fritz-canary` for 30 s, then resumes it. Watch Grafana for the gap in the canary series and recovery.

### High error rate (automatic rollback)

```bash
make chaos-errors
```

Restarts the canary with `FRITZ_ERROR_RATE=0.25` (25%). Run `make watch-canary-auto` in a second terminal to see the controller trigger a rollback automatically.

### CPU stress (latency spike)

```bash
make chaos-cpu
```

Runs `stress-ng` inside the canary container for 60 s. Watch the P99 latency panel spike past the threshold.

### Full demo sequence

```bash
make chaos-all
```

Runs all experiments sequentially with cooldown periods between them. Good for demos and CI validation.

---

## Configuring Fault Injection

Override error rates and latency at deploy time:

```bash
# Canary with 10% errors and higher latency
CANARY_ERROR_RATE=0.10 CANARY_LATENCY=4.0 make deploy-canary
```

| Variable | Default | Description |
|----------|---------|-------------|
| `STABLE_ERROR_RATE` | `0.02` | Stable error fraction (0–1) |
| `CANARY_ERROR_RATE` | `0.02` | Canary error fraction (0–1) |
| `CANARY_LATENCY` | `1.5` | Canary base latency in seconds |
| `STABLE_WEIGHT` | `10` | Relative nginx weight for stable |
| `CANARY_WEIGHT` | `0` | Relative nginx weight for canary |

---

## Canary Controller Reference

`canary_controller.py` queries Prometheus directly and acts on the results.

```
python canary_controller.py [OPTIONS]

  --version     Version label to evaluate  (default: canary)
  --watch       Print metrics every 20 s, no action
  --once        Run a single analysis window and exit
  --auto        Automatically promote or rollback
  --prometheus  Prometheus URL  (default: http://localhost:9090)
  --windows     Consecutive healthy windows before promoting  (default: 3)
  --interval    Seconds between windows  (default: 20)
```

Example — evaluate a single window:
```bash
python canary_controller.py --once --version canary
```

---

## Grafana Dashboard Panels

| Panel | Metric | Threshold |
|-------|--------|-----------|
| Error Rate by Version | `misterfritz_errors_total` rate | Yellow > 2%, Red > 5% |
| P99 / P50 Latency | `misterfritz_request_duration_seconds` histogram | Yellow > 3 s, Red > 8 s |
| Throughput | `misterfritz_discord_messages_total` rate | — |
| Tool Calls / sec | `misterfritz_tool_calls_total` rate by tool | — |
| Uptime | `misterfritz_uptime_seconds` | — |
| Total Errors | `misterfritz_errors_total` cumulative | Red > 0 |
| Error Rate Bargauge | Canary vs stable comparison | Yellow > 2%, Red > 5% |

---

## Kubernetes Simulation (Optional)

For a full local Kubernetes cluster using [kind](https://kind.sigs.k8s.io/):

### Prerequisites

```bash
# macOS / Linux
brew install kind kubectl helm
# or follow https://kind.sigs.k8s.io/docs/user/quick-start/#installation
```

### Bootstrap the cluster

```bash
bash local-sim/kind/setup.sh
```

This installs (in order):

1. kind cluster `misterfritz` (1 control-plane + 1 worker)
2. NGINX Ingress Controller
3. kube-prometheus-stack (Prometheus + Grafana via Helm)
4. Argo Rollouts + dashboard
5. ArgoCD
6. MisterFritz K8s manifests from `infra/k8s/`

### Create the bot secret

```bash
kubectl create secret generic misterfritz-secrets \
  --from-literal=DISCORD_BOT_TOKEN=<your-token> \
  --from-literal=ROOT_USER=<your-username> \
  -n misterfritz
```

### Access services

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| ArgoCD | http://localhost (NodePort) | admin / (printed by setup.sh) |

### Trigger a canary rollout

```bash
# Load a new image
docker build -t misterfritz:v2 ..
kind load docker-image misterfritz:v2 --name misterfritz

# Update the rollout image
kubectl argo rollouts set image misterfritz \
  misterfritz=misterfritz:v2 -n misterfritz

# Watch the canary steps (10% → analysis → 50% → analysis → 100%)
kubectl argo rollouts status misterfritz -n misterfritz --watch
```

### Tear down

```bash
kind delete cluster --name misterfritz
```

---

## File Reference

```
local-sim/
├── mock_fritz.py               Synthetic metrics server (no Discord/Ollama needed)
├── Dockerfile.mock             Minimal image for mock_fritz
├── docker-compose.canary.yml   Multi-profile stack (stable / canary / full)
├── prometheus.yml              Scrapes fritz-stable and fritz-canary with version labels
├── canary_controller.py        CLI — watch/auto promote/rollback based on Prometheus data
├── Makefile                    All workflow targets (run `make` for help)
├── nginx/
│   └── nginx.conf.template     Weighted upstream with envsubst substitution
├── grafana/
│   └── dashboards/
│       ├── misterfritz.json    7-panel canary deployment dashboard
│       └── provisioning.yml    Grafana dashboard provisioning config
├── chaos/
│   └── chaos.sh                Chaos experiments: pod-failure, cpu-stress, network-delay, high-errors
└── kind/
    ├── cluster.yaml            kind cluster config (1 control-plane + 1 worker)
    └── setup.sh                Full K8s bootstrap script
```
