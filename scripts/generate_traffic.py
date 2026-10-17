#!/usr/bin/env python3
"""Send synthetic traffic to the API so the Grafana dashboard has something to show.

    python scripts/generate_traffic.py --url http://localhost:8000 --requests 300 --rps 10
Roughly 8 % of requests are deliberately invalid (to populate the 4xx panel).
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter


def random_patient(rng: random.Random) -> dict:
    sick = rng.random() < 0.45
    return {
        "age": rng.randint(35, 75), "sex": rng.choice([0, 1]),
        "cp": rng.choice([4, 4, 3, 2, 1]) if sick else rng.choice([1, 2, 3, 3, 4]),
        "trestbps": rng.randint(100, 180), "chol": rng.randint(150, 330),
        "fbs": rng.choice([0, 0, 0, 1]), "restecg": rng.choice([0, 0, 2]),
        "thalach": rng.randint(100, 150) if sick else rng.randint(130, 195),
        "exang": int(sick and rng.random() < 0.6),
        "oldpeak": round(rng.uniform(0.5, 4.0) if sick else rng.uniform(0, 1.5), 1),
        "slope": rng.choice([2, 2, 3]) if sick else rng.choice([1, 1, 2]),
        "ca": rng.choice([0, 1, 2, 3]) if sick else rng.choice([0, 0, 0, 1]),
        "thal": rng.choice([7, 7, 6]) if sick else rng.choice([3, 3, 3, 7]),
    }  # fmt: skip


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    codes: Counter = Counter()
    for i in range(a.requests):
        body = random_patient(rng)
        if rng.random() < 0.08:
            body["age"] = 999  # invalid on purpose
        codes[post(f"{a.url}/predict", body)] += 1
        if (i + 1) % 50 == 0:
            print(f"  sent {i + 1}/{a.requests}  {dict(codes)}")
        time.sleep(1.0 / a.rps)
    print("done:", dict(codes))


if __name__ == "__main__":
    main()
