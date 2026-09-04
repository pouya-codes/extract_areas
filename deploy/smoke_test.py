#!/usr/bin/env python
"""
Exercise loaded models through the canonical inference APIs.

Uses ``/process_region_from_slide`` (sync) and optionally ``/inference/jobs``
(async). Upload-based ``/process_region*`` endpoints must return 410.

Usage:
    python deploy/smoke_test.py [--url http://127.0.0.1:8000] \\
        [--slide /data/omero/data/.../slide.svs] \\
        [--async-jobs] [--concurrent]

Without --slide, only health / catalog / 410 checks run.
"""

from __future__ import annotations

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


def check_gone(url: str) -> bool:
    ok = True
    for path in ("/process_region", "/process_region_annotation"):
        resp = requests.post(f"{url}{path}", timeout=30)
        if resp.status_code != 410:
            print(f"  FAIL {path}: expected 410, got {resp.status_code}")
            ok = False
        else:
            print(f"  OK   {path} -> 410")
    return ok


def call_from_slide(url, model, slide_path, region_id=1, hyperparameters=None):
    # Small level-0 box near the origin; real slides usually have tissue there
    # or the model returns an empty/low score — either exercises the path.
    payload = {
        "slide_path": slide_path,
        "roi_bbox": [0, 0, 1024, 1024],
        "polygon": [[0, 0], [1024, 0], [1024, 1024], [0, 1024]],
        "downsample": 4.0,
        "overlay_downsample": 8.0,
        "model_name": model,
        "region_id": str(region_id),
    }
    if hyperparameters:
        payload["hyperparameters"] = hyperparameters

    started = time.time()
    resp = requests.post(
        f"{url}/process_region_from_slide", json=payload, timeout=1800
    )
    return resp, time.time() - started


def call_async_job(url, model, slide_path, region_id=1):
    payload = {
        "slide_path": slide_path,
        "roi_bbox": [0, 0, 1024, 1024],
        "polygon": [[0, 0], [1024, 0], [1024, 1024], [0, 1024]],
        "downsample": 4.0,
        "overlay_downsample": 8.0,
        "model_name": model,
        "region_id": str(region_id),
    }
    started = time.time()
    submit = requests.post(f"{url}/inference/jobs", json=payload, timeout=60)
    if submit.status_code != 200:
        return submit, time.time() - started
    job_id = submit.json()["job_id"]
    deadline = time.time() + 1800
    while time.time() < deadline:
        poll = requests.get(f"{url}/inference/jobs/{job_id}", timeout=30)
        data = poll.json()
        if data.get("status") in ("succeeded", "failed", "cancelled"):
            # Mimic sync response shape for describe()
            if data.get("status") == "succeeded" and data.get("has_overlay"):
                ov = requests.get(f"{url}/inference/jobs/{job_id}/overlay", timeout=60)
                data["processed_image_base64"] = base64.b64encode(ov.content).decode()
                data["status"] = "success"
            elif data.get("status") != "succeeded":
                data["status"] = "error"
                data["message"] = data.get("error") or data.get("status")
            else:
                data["status"] = "success"
            class _R:
                status_code = 200 if data.get("status") == "success" else 500

                def json(self_inner):
                    return data

            return _R(), time.time() - started
        time.sleep(1)
    class _T:
        status_code = 504

        def json(self_inner):
            return {"status": "error", "message": "timeout"}

    return _T(), time.time() - started


def describe(model, resp, elapsed):
    if resp.status_code != 200:
        body = getattr(resp, "text", None)
        try:
            payload = resp.json()
            msg = payload.get("message") or ""
        except Exception:
            payload, msg = {}, body or ""
        # Domain outcome from PDAC on non-tumor ROIs still proves the heavy path.
        if model == "pdac" and "no tumor patches" in str(msg):
            print(f"  OK   {model} in {elapsed:.1f}s (domain: no tumor patches)")
            return True
        print(f"  FAIL {model}: HTTP {resp.status_code} {str(body)[:300]}")
        return False

    payload = resp.json()
    if payload.get("status") != "success":
        msg = payload.get("message") or ""
        if model == "pdac" and "no tumor patches" in str(msg):
            print(f"  OK   {model} in {elapsed:.1f}s (domain: no tumor patches)")
            return True
        print(f"  FAIL {model}: {msg}")
        return False

    overlay_b64 = payload.get("processed_image_base64")
    if overlay_b64:
        overlay = base64.b64decode(overlay_b64)
        size = Image.open(io.BytesIO(overlay)).size
        print(f"  OK   {model} in {elapsed:.1f}s")
        print(f"       overlay {size[0]}x{size[1]} ({len(overlay) / 1024:.0f} KB)")
    else:
        print(f"  OK   {model} in {elapsed:.1f}s (no overlay bytes)")
    print(f"       result  {payload.get('str_result')}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--heavy-url", default="http://127.0.0.1:8005")
    parser.add_argument("--slide", default="")
    parser.add_argument("--async-jobs", action="store_true")
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()

    print(f"General:  {args.url}")
    print(f"Heavy:    {args.heavy_url}")
    ok = True

    for label, url in (("general", args.url), ("heavy", args.heavy_url)):
        try:
            health = requests.get(f"{url}/health", timeout=30).json()
        except Exception as exc:
            print(f"Health {label}: UNREACHABLE ({exc})")
            ok = False
            continue
        print(
            f"Health {label}: pool={health.get('pool')} "
            f"models={health.get('loaded_models')} "
            f"gpu={((health.get('gpu') or {}).get('name'))}"
        )

    print("\n[deprecated upload APIs]")
    ok &= check_gone(args.url)

    if not args.slide:
        print("\nNo --slide given; skipping inference smoke.")
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    slide = args.slide
    print(f"\nSlide: {slide}")

    for label, url in (("general", args.url), ("heavy", args.heavy_url)):
        try:
            loaded = requests.get(f"{url}/models/list", timeout=30).json()["loaded_models"]
        except Exception:
            continue
        print(f"\n=== {label} ({url}) models={loaded} ===")
        for model in loaded:
            print(f"[{model}]")
            caller = call_async_job if args.async_jobs else call_from_slide
            resp, elapsed = caller(url, model, slide)
            ok &= describe(model, resp, elapsed)
            print()

        if args.concurrent and loaded and not args.async_jobs:
            model = loaded[0]
            n = min(3, len(loaded) + 2)
            print(f"[concurrency] {n}x '{model}' on {label}")
            started = time.time()
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [
                    pool.submit(call_from_slide, url, model, slide, i)
                    for i in range(n)
                ]
                results = [f.result() for f in futures]
            wall = time.time() - started
            times = [round(e, 1) for _, e in results]
            print(f"  per-request {times} wall {wall:.1f}s")
            ok &= all(r.status_code == 200 for r, _ in results)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
