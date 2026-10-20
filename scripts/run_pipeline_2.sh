#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$HOME/workspace/MLOps Lab/heart-disease-mlops"
NAMESPACE="heart-risk"
MONITORING_NAMESPACE="monitoring"
RELEASE_NAME="monitoring"
CHART="prometheus-community/kube-prometheus-stack"
SERVICE_NAME="heart-risk-api"
SERVICEMONITOR_FILE="/tmp/heart-risk-servicemonitor.yaml"

k() {
    minikube kubectl -- "$@"
}

echo "============================================================"
echo "Heart Risk API - Setup started"
echo "============================================================"

# 1. Correct project directory
cd "$PROJECT_DIR"
echo "[1/10] Project directory: $PWD"

sudo dnf install -y curl tar gzip
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash


# 2. Check prerequisites
for cmd in curl helm minikube; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Required command '$cmd' is missing."
        exit 1
    fi
done




echo "[2/10] Required commands found."
helm version
minikube status

# Verify Minikube is running
k get nodes

# 3. Check API and smoke tests
echo "[3/10] Checking API health..."

curl --fail --silent --show-error http://127.0.0.1:8000/health
echo
curl --fail --silent --show-error http://127.0.0.1:8000/ready
echo

if [[ ! -f scripts/smoke_test.sh ]]; then
    echo "ERROR: scripts/smoke_test.sh not found in $PROJECT_DIR"
    exit 1
fi

echo "Running smoke tests..."
BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh

# 4. Configure Helm
echo "[4/10] Configuring Prometheus Helm repository..."

helm repo add prometheus-community \
    https://prometheus-community.github.io/helm-charts \
    --force-update

helm repo update

# 5. Install or upgrade monitoring
echo "[5/10] Installing/upgrading monitoring stack..."

helm upgrade --install "$RELEASE_NAME" "$CHART" \
    --namespace "$MONITORING_NAMESPACE" \
    --create-namespace \
    --set grafana.service.type=ClusterIP \
    --wait \
    --timeout 15m

echo "Monitoring Helm release installed successfully."

# 6. Wait for ServiceMonitor CRD
echo "[6/10] Waiting for ServiceMonitor CRD..."

k wait \
    --for=condition=Established \
    crd/servicemonitors.monitoring.coreos.com \
    --timeout=180s

# 7. Create and apply ServiceMonitor
echo "[7/10] Creating ServiceMonitor manifest..."

cat > "$SERVICEMONITOR_FILE" <<'YAML_EOF'
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: heart-risk-api
  namespace: monitoring
  labels:
    release: monitoring
spec:
  namespaceSelector:
    matchNames:
      - heart-risk
  selector:
    matchLabels:
      app.kubernetes.io/name: heart-risk-api
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
YAML_EOF

k apply -f "$SERVICEMONITOR_FILE"

# 8. Verify service, endpoints, and monitoring
echo "[8/10] Checking API service..."

k get svc "$SERVICE_NAME" -n "$NAMESPACE" --show-labels
k get endpoints "$SERVICE_NAME" -n "$NAMESPACE"
k get servicemonitor -n "$MONITORING_NAMESPACE"

echo
echo "Monitoring Pods:"
k get pods -n "$MONITORING_NAMESPACE" -o wide

# 9. Print access instructions
echo "[9/10] Setup complete. Access commands:"

cat <<'INFO_EOF'

Run each port-forward in a separate terminal.

Heart Risk API:
  minikube kubectl -- port-forward -n heart-risk svc/heart-risk-api 8000:80

Prometheus:
  minikube kubectl -- port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090

Grafana:
  minikube kubectl -- port-forward -n monitoring svc/monitoring-grafana 3000:80

URLs on the machine running port-forward:
  API:        http://127.0.0.1:8000/health
  Prometheus: http://127.0.0.1:9090
  Grafana:    http://127.0.0.1:3000

Grafana username:
  minikube kubectl -- get secret monitoring-grafana -n monitoring -o jsonpath='{.data.admin-user}' | base64 --decode
  echo

Grafana password:
  minikube kubectl -- get secret monitoring-grafana -n monitoring -o jsonpath='{.data.admin-password}' | base64 --decode
  echo

INFO_EOF

# 10. Final status
echo "[10/10] Final cluster status..."

k get pods -n "$NAMESPACE" -o wide
k get pods -n "$MONITORING_NAMESPACE" -o wide

echo
echo "============================================================"
echo "Setup script finished successfully."
echo "Verify Prometheus Targets to confirm API scraping."
echo "============================================================"



