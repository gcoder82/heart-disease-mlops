# Environment setup - everything you need to run the whole package

## 1. Hardware / accounts
| Need | Minimum | Why |
|---|---|---|
| CPU / RAM / disk | 4 cores, **8 GB RAM** (16 GB nicer with Kubernetes + Grafana), ~10 GB free | Docker Desktop + a local K8s node + Python envs |
| Internet | required once | UCI download, pip, Docker base images |
| GitHub account | free | repository + GitHub Actions (CI/CD) |
| Cloud account | *optional* | only for GKE / EKS / AKS instead of local Kubernetes |

## 2. Software (macOS - Apple Silicon or Intel)
| Tool | Version tested against | Install |
|---|---|---|
| Homebrew | any | https://brew.sh |
| Python | **3.11** (3.12 also works: `make setup PY=python3.12`) | `brew install python@3.11` |
| Git | >= 2.30 | `brew install git` |
| Docker Desktop | >= 4.30 | `brew install --cask docker` (then launch it) |
| Kubernetes | Docker Desktop built-in **or** Minikube >= 1.33 | Docker Desktop -> Settings -> Kubernetes -> Enable; or `brew install minikube` |
| kubectl | >= 1.28 | `brew install kubernetes-cli` |
| Helm | >= 3.12 (only for the Helm route) | `brew install helm` |
| make | Xcode CLT | `xcode-select --install` |
| LibreOffice | optional, for PDF export of the report | `brew install --cask libreoffice` |

Shortcut for the Homebrew items: `bash scripts/bootstrap_mac.sh`.

Python libraries are **not** installed globally - `make setup` creates `.venv` and installs the pinned
`requirements-dev.txt` (numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, MLflow 2.17.2, FastAPI 0.115.5,
prometheus-client 0.21.0, pytest 8.3.3, ruff 0.7.4, ...).

## 3. Ubuntu 22.04 / 24.04 VM variant
```bash
sudo apt-get update && sudo apt-get install -y git make curl python3-venv python3-pip docker.io jq
sudo usermod -aG docker "$USER" && newgrp docker
# kubectl / minikube / helm
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && sudo install kubectl /usr/local/bin/
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && sudo install minikube-linux-amd64 /usr/local/bin/minikube
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
minikube start --driver=docker --cpus=2 --memory=4096
```
Ubuntu 24.04 ships Python 3.12 -> use `make setup PY=python3.12`. (Ubuntu 22.04 has 3.10; install 3.11 from the
deadsnakes PPA.) Training and the Docker image may use different 3.11/3.12 minors; scikit-learn / numpy versions
are pinned identically so the joblib model loads fine.

## 4. Ports used
`8000` API (local/Docker) - `80` K8s LoadBalancer (Docker Desktop) - `5000` MLflow UI - `9090` Prometheus -
`3000` Grafana. Free them or override (`PORT=8001 make docker-run`).

## 5. First-run sequence
```bash
make setup && make data && make eda && make train && make test && make lint
make docker-build docker-run docker-test        # Docker proof
make k8s-deploy                                  # Kubernetes + Prometheus + Grafana
make report                                      # after capturing screenshots
```
Or run the first eight steps in one go: `bash scripts/run_pipeline.sh`.

## 6. GitHub setup (CI/CD)
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main && git remote add origin https://github.com/<you>/heart-disease-mlops.git
git push -u origin main
```
Actions run automatically. For the GHCR `publish` job: repo -> Settings -> Actions -> General ->
Workflow permissions -> *Read and write*. Nothing else (no secrets) is required.

## 7. Cloud variant (GKE / EKS / AKS)
1. Push the image: `docker tag heart-risk-api:1.0.0 ghcr.io/<you>/heart-risk-api:1.0.0 && docker push ...`
2. Edit `image:` in `k8s/deployment.yaml` (or `--set image.repository=...,image.tag=...` for Helm).
3. `kubectl apply -k k8s && kubectl apply -k k8s/monitoring`; `kubectl -n heart-risk get svc heart-risk-api -w`
   waits for the cloud load balancer's EXTERNAL-IP -> that is the public API URL.

## 8. Troubleshooting
| Symptom | Fix |
|---|---|
| `models/model.joblib missing` | run `make train` before `docker build` / `k8s-deploy` |
| UCI download fails | `python scripts/download_data.py --file <manually downloaded file> --prepare` |
| `ImagePullBackOff` on Minikube | image not in Minikube's daemon: use `make k8s-deploy` (it does `minikube docker-env`) |
| LoadBalancer `<pending>` on Minikube | run `minikube tunnel` in a second terminal |
| Port 80 busy on Docker Desktop | `kubectl -n heart-risk port-forward svc/heart-risk-api 8080:80` |
| HPA shows `<unknown>` | `minikube addons enable metrics-server` (Docker Desktop: install metrics-server) |
| Apple-Silicon image warnings | none expected: all base images are multi-arch |
