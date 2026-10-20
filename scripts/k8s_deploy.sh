#!/usr/bin/env bash
# Deploy API (+ Prometheus & Grafana) to the *current* kubectl context.
# Supports Docker Desktop Kubernetes and Minikube; for cloud clusters push the image to a
# registry first and edit `image:` in k8s/deployment.yaml (or helm values).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-heart-risk-api:1.0.0}"
CTX="$(kubectl config current-context)"
echo ">> kubectl context: $CTX"

[ -f models/model.joblib ] || { echo "models/model.joblib missing - run 'make train' first" >&2; exit 1; }

case "$CTX" in
  minikube*)
    echo ">> building image inside Minikube's Docker daemon"
    eval "$(minikube -p minikube docker-env)"
    docker build -t "$IMAGE" .
    ;;
  docker-desktop*)
    echo ">> building image (Docker Desktop shares its image store with Kubernetes)"
    docker build -t "$IMAGE" .
    ;;
  *)
    echo "!! Unrecognised context '$CTX'. Assuming the image is reachable from the cluster." >&2
    ;;
esac

minikube kubectl apply -k k8s
minikube kubectl apply -k k8s/monitoring
minikube kubectl -n heart-risk rollout status deploy/heart-risk-api --timeout=180s
minikube kubectl -n heart-risk rollout status deploy/prometheus --timeout=180s
minikube kubectl -n heart-risk rollout status deploy/grafana --timeout=180s

echo
kubectl -n heart-risk get pods,svc,ingress,hpa
cat <<MSG

Access
  API (Docker Desktop LoadBalancer): http://localhost/docs      -> BASE_URL=http://localhost
  API (Minikube): run 'minikube tunnel' in another terminal, then use the EXTERNAL-IP above
  Anything else: kubectl -n heart-risk port-forward svc/heart-risk-api 8080:80
  Prometheus: kubectl -n heart-risk port-forward svc/prometheus 9090:9090
  Grafana:    kubectl -n heart-risk port-forward svc/grafana 3000:3000   (admin / admin)

Verify:  BASE_URL=http://localhost bash scripts/smoke_test.sh
MSG
