# Screenshots needed for the report

Save PNGs in this folder with **exactly** these names, then run `make report`
(`python scripts/build_report.py --author ... --student-id ... --repo-url ... --pdf`).
Missing files show up as orange "pending" notes in the report, so you can see what is left.

| File | How to capture it |
|---|---|
| `mlflow_experiments.png` | `make mlflow-ui` -> http://localhost:5000 -> experiment *heart-disease-risk*, expand `training-session` |
| `mlflow_compare.png` | tick the 3 child runs -> **Compare** |
| `mlflow_run_detail.png` | open one child run (parameters, metrics, artifacts panel visible) |
| `ci_workflow_overview.png` | GitHub -> Actions -> a green run, graph showing lint -> test -> train -> docker (-> publish) |
| `ci_workflow_artifacts.png` | bottom of the same run page: Artifacts list |
| `ci_failure_demo.png` | on a throw-away branch change an assertion in `tests/test_predictor.py` (or add `import os` unused), push, screenshot the red run; then delete the branch |
| `docker_build.png` | terminal: `make docker-build` (final lines with "naming to docker.io/library/heart-risk-api:1.0.0") |
| `docker_run_predict.png` | terminal: `make docker-run docker-test` (shows PASS lines and the JSON response) |
| `swagger_ui.png` | http://localhost:8000/docs -> *POST /predict* -> Try it out -> Execute |
| `k8s_pods_services.png` | `kubectl -n heart-risk get pods,svc,hpa,ingress -o wide` |
| `k8s_predict_via_lb.png` | `curl -s -X POST http://localhost/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json` (Docker Desktop) |
| `k8s_ingress.png` | optional: `curl -H 'Host: heart-risk.local' http://localhost/health`, or the port-forward command + response |
| `prometheus_targets.png` | `kubectl -n heart-risk port-forward svc/prometheus 9090:9090` -> http://localhost:9090/targets |
| `grafana_dashboard.png` | port-forward Grafana :3000 (admin/admin), run `make traffic` in another terminal, open *Heart Risk API* dashboard after ~1 min |
| `api_logs.png` | `kubectl -n heart-risk logs deploy/heart-risk-api --tail=20` (or `docker logs heart-risk-api`) |

Tip: macOS `Cmd+Shift+4` then `Space` captures a single window; `Cmd+Shift+5` records the screen (use it for the video).
