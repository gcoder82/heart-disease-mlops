# Usage: `make help`
PY ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
IMAGE ?= heart-risk-api:1.0.0
PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup data eda train train-quick test lint format mlflow-ui serve notebooks report \
        docker-build docker-run docker-test docker-stop compose-up compose-down traffic \
        k8s-deploy k8s-delete helm-install all clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup:  ## create venv and install everything (clean-room install from requirements)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

data:  ## download UCI data and build the cleaned CSV
	$(BIN)/python scripts/download_data.py --prepare

eda:  ## EDA figures -> reports/figures
	$(BIN)/python -m heartrisk.eda

train:  ## tune + track (MLflow) + save the model -> models/
	$(BIN)/python -m heartrisk.train

train-quick:  ## tiny grids, no MLflow (smoke test only)
	$(BIN)/python -m heartrisk.train --quick --no-mlflow --cv-folds 3

test:  ## unit tests with coverage
	$(BIN)/pytest --cov=heartrisk --cov-report=term-missing

lint:  ## static analysis
	$(BIN)/ruff check .

format:  ## auto-fix lint issues
	$(BIN)/ruff check . --fix

mlflow-ui:  ## browse experiments at http://localhost:5000
	$(BIN)/mlflow ui --backend-store-uri ./mlruns --port 5000

serve:  ## run the API without Docker (hot reload)
	$(BIN)/uvicorn heartrisk.api.app:app --reload --port $(PORT)

notebooks:  ## execute the three notebooks in place (stores outputs)
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/*.ipynb

report:  ## build the written report (docx + md [+ pdf]) -> reports/
	$(BIN)/python scripts/build_report.py --pdf

docker-build:  ## build the serving image (needs models/model.joblib)
	@test -f models/model.joblib || { echo "models/model.joblib missing - run 'make train' first"; exit 1; }
	docker build -t $(IMAGE) .

docker-run:  ## run the container on :8000
	docker run -d --rm --name heart-risk-api -p $(PORT):8000 $(IMAGE)

docker-test:  ## smoke-test the running container
	BASE_URL=http://localhost:$(PORT) ./scripts/smoke_test.sh

docker-stop:
	-docker stop heart-risk-api

compose-up:  ## API + Prometheus + Grafana
	docker compose up -d --build

compose-down:
	docker compose down

traffic:  ## generate load for the Grafana dashboard
	python3 scripts/generate_traffic.py --url http://localhost:$(PORT) --requests 300 --rps 10

k8s-deploy:  ## deploy to Docker Desktop / Minikube (+ monitoring)
	./scripts/k8s_deploy.sh

k8s-delete:
	kubectl delete namespace heart-risk --ignore-not-found

helm-install:  ## alternative to k8s-deploy: install via Helm
	helm upgrade --install hr helm/heart-risk-api -n heart-risk --create-namespace

all: setup data eda train test docker-build  ## full local build

clean:
	rm -rf $(VENV) mlruns mlflow.db .pytest_cache .ruff_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
