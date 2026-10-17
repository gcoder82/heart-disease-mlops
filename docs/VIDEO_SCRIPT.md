# Pipeline video - storyboard (about 5-6 minutes)

Record with macOS `Cmd+Shift+5` (screen + microphone). Keep terminal font large (>= 16 pt).

| Time | Show | Say / prove |
|---|---|---|
| 0:00 | Repo tree in VS Code | Problem, dataset, what the repo contains |
| 0:30 | `make data` then `make eda` | Reproducible download + cleaning; open 2-3 EDA figures |
| 1:15 | `make train` (let it run, then skip ahead) | Three models, grid search, CV, test metrics, selected model |
| 2:00 | `make mlflow-ui` | Parent run + child runs, params/metrics/plots/model artifacts, compare view |
| 2:45 | `make test` and `make lint` | Unit tests green, coverage |
| 3:15 | GitHub Actions run page | lint -> test -> train -> docker jobs; artifacts; (flash the red demo run) |
| 3:50 | `make docker-build docker-run docker-test` | Container builds, serves `/predict`, returns prediction + confidence |
| 4:30 | `make k8s-deploy` then `kubectl get pods,svc,hpa` | 2 replicas, LoadBalancer; curl `/predict` via the LB |
| 5:00 | `make traffic` + Grafana dashboard + `kubectl logs` | Metrics and structured logs reacting live |
| 5:40 | Close | Link to repo + report; limitations in one sentence |
