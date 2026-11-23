"""Latency / throughput benchmark of the deployed API (results for the paper).

  docker run -p 7860:7860 -e MLFLOW_TRACKING_URI=... -e MLFLOW_TRACKING_USERNAME=... -e MLFLOW_TRACKING_PASSWORD=... ghcr.io/afifnet70/bairops
  python scripts/benchmark.py --url http://localhost:7860 --n 200 --concurrency 8
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--concurrency", type=int, default=8)
a = ap.parse_args()
base = a.url.rstrip("/")

cities = requests.get(f"{base}/cities", timeout=60).json()
requests.get(f"{base}/forecast", params={"city": cities[0]}, timeout=120)  # warm-up


def call(i):
    t0 = time.perf_counter()
    r = requests.get(f"{base}/forecast", params={"city": cities[i % len(cities)]}, timeout=60)
    return (time.perf_counter() - t0) * 1000, r.status_code, r.json().get("model_latency_ms")


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(a.concurrency) as ex:
    res = list(ex.map(call, range(a.n)))
wall = time.perf_counter() - t0
lat = sorted(r[0] for r in res)
model_lat = [r[2] for r in res if r[2] is not None]
out = {"requests": a.n, "concurrency": a.concurrency, "success_rate_%": 100 * sum(r[1] == 200 for r in res) / a.n,
       "throughput_req_per_s": round(a.n / wall, 2), "latency_ms_p50": round(statistics.median(lat), 1),
       "latency_ms_p95": round(lat[int(0.95 * len(lat)) - 1], 1), "latency_ms_max": round(lat[-1], 1),
       "model_inference_ms_median": round(statistics.median(model_lat), 3) if model_lat else None}
print(json.dumps(out, indent=2))
