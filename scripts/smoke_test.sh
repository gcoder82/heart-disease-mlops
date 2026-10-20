#!/usr/bin/env bash
# Black-box checks against a running API.   BASE_URL=http://localhost:8000 ./scripts/smoke_test.sh
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$HERE/sample_request.json"
fail() { echo "FAIL: $1" >&2; exit 1; }
pass() { echo "PASS: $1"; }

# wait up to 60 s for readiness
for i in $(seq 1 30); do
  if curl -fsS "$BASE_URL/ready" >/dev/null 2>&1; then break; fi
  [ "$i" -eq 30 ] && fail "API not ready at $BASE_URL"
  sleep 2
done
pass "GET /ready"

curl -fsS "$BASE_URL/health" | grep -q '"ok"' || fail "/health"
pass "GET /health"

RESP=$(curl -fsS -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d @"$PAYLOAD")
echo "      response: $RESP"
for key in prediction confidence probability_disease risk_level model_version request_id; do
  echo "$RESP" | grep -q "\"$key\"" || fail "/predict response missing '$key'"
done
pass "POST /predict returns prediction + confidence"

CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' -d '{"age": 400}')
[ "$CODE" = "422" ] || fail "invalid payload should give 422, got $CODE"
pass "invalid payload rejected with 422"

curl -fsS "$BASE_URL/metrics" -o /tmp/heart-risk-metrics.txt || fail "/metrics request"
grep -q "heart_api_requests_total" /tmp/heart-risk-metrics.txt || fail "/metrics content"
pass "GET /metrics exposes Prometheus counters"
echo "All smoke tests passed."
