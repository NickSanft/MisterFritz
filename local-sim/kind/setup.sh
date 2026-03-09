#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Bootstrap a full local Kubernetes environment using kind.
#
# Installs:
#   - kind cluster (misterfritz)
#   - NGINX Ingress Controller
#   - Prometheus + Grafana via kube-prometheus-stack (Helm)
#   - Argo Rollouts
#   - ArgoCD
#   - MisterFritz manifests
#
# Prerequisites: docker, kind, kubectl, helm
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CLUSTER_NAME="misterfritz"
NAMESPACE="misterfritz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="$SCRIPT_DIR/../../infra/k8s"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup]${NC} $*"; exit 1; }

check_prereq() {
    command -v "$1" &>/dev/null || error "$1 is not installed. Please install it first."
    info "$1 ✓"
}

# ── 0. Check prerequisites ───────────────────────────────────────────────────
info "Checking prerequisites..."
for cmd in docker kind kubectl helm; do
    check_prereq "$cmd"
done

# ── 1. Create kind cluster ───────────────────────────────────────────────────
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    info "Cluster '$CLUSTER_NAME' already exists — skipping creation."
else
    info "Creating kind cluster '$CLUSTER_NAME'..."
    kind create cluster --config "$SCRIPT_DIR/cluster.yaml" --name "$CLUSTER_NAME"
    success "Cluster created."
fi

kubectl cluster-info --context "kind-${CLUSTER_NAME}"

# ── 2. NGINX Ingress Controller ──────────────────────────────────────────────
info "Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/component=controller \
    --timeout=90s
success "NGINX Ingress ready."

# ── 3. Prometheus + Grafana (kube-prometheus-stack) ─────────────────────────
info "Adding Helm repos..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm repo add argo https://argoproj.github.io/argo-helm --force-update
helm repo update

info "Installing kube-prometheus-stack..."
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace \
    --set grafana.adminPassword=admin \
    --set grafana.service.type=NodePort \
    --set grafana.service.nodePort=30000 \
    --set prometheus.service.type=NodePort \
    --set prometheus.service.nodePort=30090 \
    --wait --timeout 5m
success "Prometheus + Grafana ready."
info "  Grafana  → http://localhost:3000  (admin / admin)"
info "  Prom     → http://localhost:9090"

# ── 4. Argo Rollouts ─────────────────────────────────────────────────────────
info "Installing Argo Rollouts..."
kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install argo-rollouts argo/argo-rollouts \
    --namespace argo-rollouts \
    --set dashboard.enabled=true \
    --wait --timeout 3m
success "Argo Rollouts ready."

# ── 5. ArgoCD ────────────────────────────────────────────────────────────────
info "Installing ArgoCD..."
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install argocd argo/argo-cd \
    --namespace argocd \
    --set server.service.type=NodePort \
    --wait --timeout 5m

ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" | base64 -d)
success "ArgoCD ready."
info "  ArgoCD password: $ARGOCD_PASSWORD"

# ── 6. Apply MisterFritz manifests ───────────────────────────────────────────
info "Applying MisterFritz K8s manifests..."
kubectl apply -f "$MANIFESTS_DIR/namespace.yaml"
kubectl apply -f "$MANIFESTS_DIR/configmap.yaml"

echo ""
echo -e "${RED}⚠️  You must create the secret before the bot will start:${NC}"
echo ""
echo "  kubectl create secret generic misterfritz-secrets \\"
echo "    --from-literal=DISCORD_BOT_TOKEN=<your-token> \\"
echo "    --from-literal=ROOT_USER=<your-username> \\"
echo "    -n misterfritz"
echo ""

kubectl apply -f "$MANIFESTS_DIR/service.yaml"
kubectl apply -f "$MANIFESTS_DIR/hpa.yaml"
# Apply Rollout instead of plain Deployment when Argo Rollouts is ready
kubectl apply -f "$MANIFESTS_DIR/rollout.yaml"
kubectl apply -f "$MANIFESTS_DIR/analysistemplate.yaml"

success "MisterFritz manifests applied."

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Local K8s cluster is ready!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Grafana   → http://localhost:3000  (admin/admin)"
echo "  Prometheus→ http://localhost:9090"
echo ""
echo "Next steps:"
echo "  1. Create the secret (see above)"
echo "  2. Build & load the image:"
echo "       docker build -t misterfritz:latest .."
echo "       kind load docker-image misterfritz:latest --name $CLUSTER_NAME"
echo "  3. Update infra/k8s/rollout.yaml image to 'misterfritz:latest'"
echo "  4. kubectl apply -f ../../infra/k8s/rollout.yaml"
echo ""
echo "To tear down: kind delete cluster --name $CLUSTER_NAME"
