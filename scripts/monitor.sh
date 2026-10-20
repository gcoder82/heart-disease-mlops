#URLs on the machine running port-forward:
#  API:        http://127.0.0.1:8000/health
# Prometheus: http://127.0.0.1:9090
# Grafana:    http://127.0.0.1:3000

NAMESPACE="heart-risk"
MONITORING_NAMESPACE="monitoring"
RELEASE_NAME="monitoring"
CHART="prometheus-community/kube-prometheus-stack"
SERVICE_NAME="heart-risk-api"
SERVICEMONITOR_FILE="/tmp/heart-risk-servicemonitor.yaml"

k() {
    minikube kubectl -- "$@"
}

#Grafana username:
  minikube kubectl -- get secret monitoring-grafana -n monitoring -o jsonpath='{.data.admin-user}' | base64 --decode
  echo

#Grafana password:
  minikube kubectl -- get secret monitoring-grafana -n monitoring -o jsonpath='{.data.admin-password}' | base64 --decode
  echo

#INFO_EOF

# 10. Final status
echo " Final cluster status..."

k get pods -n "$NAMESPACE" -o wide
k get pods -n "$MONITORING_NAMESPACE" -o wide

echo
echo "============================================================"
echo "Setup script finished successfully."
echo "Verify Prometheus Targets to confirm API scraping."
echo "============================================================"
