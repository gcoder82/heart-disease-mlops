
cd "$HOME/workspace/MLOps Lab/heart-disease-mlops"
pwd
ls -l Dockerfile
ls
ls k8s
ls scripts
command -v docker
command -v minikube
helm version
minikube status
minikube start --driver=docker
minikube status
cd "$HOME/workspace/MLOps Lab/heart-disease-mlops"
minikube image build -t heart-risk-api:1.0.0 .
minikube image ls | grep heart-risk-api
minikube kubectl -- apply -k k8s/
minikube kubectl -- rollout status deployment/heart-risk-api -n heart-risk --timeout=180s
minikube kubectl -- get pods -n heart-risk
minikube kubectl -- get svc -n heart-risk
ls k8s
minikube kubectl -- apply -f k8s/
minikube kubectl -- port-forward -n heart-risk svc/heart-risk-api 8000:80
cd "$HOME/workspace/MLOps Lab/heart-disease-mlops"
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh
helm version
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.service.type=ClusterIP \
  --timeout 10m
  minikube kubectl -- get pods -n monitoring
  minikube kubectl -- get svc -n monitoring
  minikube kubectl -- get pods -n monitoring
minikube kubectl -- get events -n monitoring --sort-by=.lastTimestamp
cat > /tmp/heart-risk-servicemonitor.yaml <<'YAML'
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
YAML
minikube kubectl -- apply -f /tmp/heart-risk-servicemonitor.yaml
minikube kubectl -- get servicemonitor -n monitoring
minikube kubectl -- get svc heart-risk-api -n heart-risk --show-labels
minikube kubectl -- get svc heart-risk-api -n heart-risk -o yaml
minikube kubectl -- port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
minikube kubectl -- port-forward -n monitoring svc/monitoring-grafana 3000:80
minikube kubectl -- get secret monitoring-grafana -n monitoring \
  -o jsonpath='{.data.admin-user}' | base64 --decode; echo
  minikube kubectl -- get secret monitoring-grafana -n monitoring \
  -o jsonpath='{.data.admin-password}' | base64 --decode; echo
  