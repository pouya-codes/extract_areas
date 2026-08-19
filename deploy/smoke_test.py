#!/usr/bin/env python
"""
Exercise every loaded model through the inference API.

Sends a real tissue ROI plus a polygon annotation to /process_region_annotation
for each loaded model, mirroring exactly what aimviewer's Django proxy sends.

Usage:
    python deploy/smoke_test.py [--url http://127.0.0.1:8000] [--concurrent]

--concurrent fires one request per replica at once to confirm the load balancer
spreads work across all GPUs.
"""

import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "module" / "DeepLiff" / "Sample_Large_Tissues" / "ROI_1.png"


def build_region(max_side=512):
    """Return (jpeg_bytes, polygon_points) for a downscaled sample ROI."""
    image = Image.open(SAMPLE).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)

    w, h = image.size
    inset_x, inset_y = int(w * 0.05), int(h * 0.05)
    polygon = [
        [inset_x, inset_y],
        [w - inset_x, inset_y],
        [w - inset_x, h - inset_y],
        [inset_x, h - inset_y],
    ]
    return buf.getvalue(), polygon, (w, h)


def call_model(url, model, jpeg, polygon, region_id, hyperparameters=None):
    files = {"region": ("region.jpg", jpeg, "image/jpeg")}
    data = {
        "mask": json.dumps(polygon),
        "region_id": str(region_id),
        "model_name": model,
    }
    if hyperparameters:
        data["hyperparameters"] = json.dumps(hyperparameters)

    started = time.time()
    resp = requests.post(
        f"{url}/process_region_annotation", files=files, data=data, timeout=1800
    )
    elapsed = time.time() - started
    return resp, elapsed


def describe(model, resp, elapsed):
    if resp.status_code != 200:
        print(f"  FAIL {model}: HTTP {resp.status_code} {resp.text[:300]}")
        return False

    payload = resp.json()
    if payload.get("status") != "success":
        print(f"  FAIL {model}: {payload.get('message')}")
        return False

    overlay = base64.b64decode(payload["processed_image_base64"])
    size = Image.open(io.BytesIO(overlay)).size
    print(f"  OK   {model} in {elapsed:.1f}s")
    print(f"       overlay {size[0]}x{size[1]} ({len(overlay) / 1024:.0f} KB)")
    print(f"       result  {payload.get('str_result')}")

    scores = payload.get("score") or {}
    if isinstance(scores, dict) and scores:
        preview = {k: scores[k] for k in list(scores)[:6]}
        print(f"       scores  {preview}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()

    print(f"Endpoint: {args.url}")
    health = requests.get(f"{args.url}/health", timeout=30).json()
    print(f"Health:   {health['loaded_models']} on {health['gpu']['name']}")

    models = requests.get(f"{args.url}/models/list", timeout=30).json()
    loaded = models["loaded_models"]
    print(f"Loaded:   {loaded}\n")

    jpeg, polygon, (w, h) = build_region()
    print(f"Region:   {w}x{h} JPEG, {len(polygon)}-point polygon\n")

    ok = True
    for model in loaded:
        print(f"[{model}]")
        info = requests.get(f"{args.url}/models/{model}/info", timeout=30).json()
        print(f"  version {info.get('version')} | requires_mask={info.get('requires_mask')}")
        schema = requests.get(
            f"{args.url}/models/{model}/hyperparameters", timeout=30
        ).json()
        print(f"  hyperparameters: {list(schema.get('hyperparameters', {}))}")

        resp, elapsed = call_model(args.url, model, jpeg, polygon, region_id=1)
        ok &= describe(model, resp, elapsed)
        print()

    if args.concurrent and loaded:
        model = loaded[0]
        n = 4
        print(f"[concurrency] {n} simultaneous '{model}' requests across replicas")
        started = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(call_model, args.url, model, jpeg, polygon, i)
                for i in range(n)
            ]
            results = [f.result() for f in futures]
        wall = time.time() - started
        times = [round(e, 1) for _, e in results]
        serial = sum(e for _, e in results)
        print(f"  per-request {times}")
        print(f"  wall {wall:.1f}s vs {serial:.1f}s serial -> {serial / wall:.2f}x speedup")
        ok &= all(r.status_code == 200 for r, _ in results)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
