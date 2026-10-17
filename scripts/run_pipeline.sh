#!/usr/bin/env bash
# One-command local verification of the whole pipeline (everything except Kubernetes).
# Produces the evidence you need for the report: logs, MLflow runs, container proof.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3.11}"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

step "1/8 clean virtualenv + install from requirements-dev.txt"
rm -rf .venv && $PY -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements-dev.txt

step "2/8 lint";                     .venv/bin/ruff check .
step "3/8 unit tests";               .venv/bin/pytest --cov=heartrisk --cov-report=term-missing
step "4/8 download + clean data";    .venv/bin/python scripts/download_data.py --force --prepare
step "5/8 EDA";                      .venv/bin/python -m heartrisk.eda
step "6/8 train + MLflow tracking";  .venv/bin/python -m heartrisk.train

step "7/8 docker build + run + smoke test"
docker build -t heart-risk-api:1.0.0 .
docker rm -f heart-risk-api >/dev/null 2>&1 || true
docker run -d --rm --name heart-risk-api -p 8000:8000 heart-risk-api:1.0.0
trap 'docker logs heart-risk-api > reports/container.log 2>&1 || true; docker stop heart-risk-api >/dev/null 2>&1 || true' EXIT
bash scripts/smoke_test.sh

step "8/8 done"
echo "MLflow UI:   make mlflow-ui     (http://localhost:5000)"
echo "Kubernetes:  make k8s-deploy"
echo "Report:      make report"
