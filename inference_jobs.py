"""Async inference jobs for extract_areas (Redis-backed, per-replica worker).

Aimviewer submits ``POST /inference/jobs`` with a SlideRegionRequest body,
polls ``GET /inference/jobs/{id}``, then fetches the overlay PNG. Each replica
runs one background worker that serialises GPU work via ``_inference_lock``.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_JOB_PREFIX = "extract_areas:inference_job:"
_QUEUE_KEY = "extract_areas:inference_queue:{pool}"
_JOB_TTL = 6 * 60 * 60
_OVERLAY_TTL = 60 * 60

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)

_redis_client = None
_redis_lock = threading.Lock()
_worker_started = False
_worker_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis_cfg(config: dict) -> tuple[str, int, int]:
    block = config.get("redis") or {}
    host = os.environ.get("EXTRACT_AREAS_REDIS_HOST") or block.get("host") or "localhost"
    port = int(os.environ.get("EXTRACT_AREAS_REDIS_PORT") or block.get("port") or 6379)
    db = int(os.environ.get("EXTRACT_AREAS_REDIS_DB") or block.get("db") or 0)
    return host, port, db


def get_redis(config: dict):
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                import redis

                host, port, db = _redis_cfg(config)
                _redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
    return _redis_client


def _job_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}{job_id}"


def _overlay_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}{job_id}:overlay"


def _queue_key(pool: str) -> str:
    return _QUEUE_KEY.format(pool=pool)


def create_job(config: dict, pool: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = get_redis(config)
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": STATUS_QUEUED,
        "pool": pool,
        "created_at": _now(),
        "updated_at": _now(),
        "payload": payload,
        "cancel_requested": False,
        "error": None,
        "score": None,
        "str_result": None,
        "timing": None,
        "metadata": None,
        "model_used": payload.get("model_name"),
        "model_version": None,
        "overlay_bbox": None,
        "overlay_width": None,
        "overlay_height": None,
        "overlay_downsample": None,
        "processed_downsample": None,
        "slide_level_used": None,
        "region_id": payload.get("region_id") or "",
        "has_overlay": False,
    }
    r.set(_job_key(job_id), json.dumps(job).encode("utf-8"), ex=_JOB_TTL)
    r.rpush(_queue_key(pool), job_id.encode("utf-8"))
    return {"job_id": job_id, "status": STATUS_QUEUED, "pool": pool}


def get_job(config: dict, job_id: str) -> Optional[dict[str, Any]]:
    raw = get_redis(config).get(_job_key(job_id))
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def request_cancel(config: dict, job_id: str) -> Optional[dict[str, Any]]:
    job = get_job(config, job_id)
    if not job:
        return None
    if job["status"] in TERMINAL:
        return job
    job["cancel_requested"] = True
    job["updated_at"] = _now()
    if job["status"] == STATUS_QUEUED:
        job["status"] = STATUS_CANCELLED
    get_redis(config).set(_job_key(job_id), json.dumps(job).encode("utf-8"), ex=_JOB_TTL)
    return job


def get_overlay(config: dict, job_id: str) -> Optional[bytes]:
    return get_redis(config).get(_overlay_key(job_id))


def _save_job(config: dict, job: dict[str, Any]) -> None:
    job["updated_at"] = _now()
    get_redis(config).set(_job_key(job["job_id"]), json.dumps(job).encode("utf-8"), ex=_JOB_TTL)


def start_worker(
    config: dict,
    pool: str,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
    inference_lock: threading.Lock,
) -> None:
    """Start a daemon worker once per process."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True

    def _loop():
        r = get_redis(config)
        qkey = _queue_key(pool)
        logger.info("inference worker started for pool=%s queue=%s", pool, qkey)
        while True:
            try:
                item = r.blpop(qkey, timeout=5)
                if not item:
                    continue
                _, job_id_b = item
                job_id = job_id_b.decode("utf-8") if isinstance(job_id_b, bytes) else str(job_id_b)
                job = get_job(config, job_id)
                if not job:
                    continue
                if job.get("cancel_requested") or job["status"] == STATUS_CANCELLED:
                    job["status"] = STATUS_CANCELLED
                    _save_job(config, job)
                    continue

                job["status"] = STATUS_RUNNING
                job["started_at"] = _now()
                _save_job(config, job)

                try:
                    with inference_lock:
                        result = process_fn(job["payload"])
                except Exception as exc:
                    logger.exception("inference job %s failed", job_id)
                    job["status"] = STATUS_FAILED
                    job["error"] = str(exc)
                    job["completed_at"] = _now()
                    _save_job(config, job)
                    continue

                if result.get("status") == "error":
                    job["status"] = STATUS_FAILED
                    job["error"] = result.get("message") or "inference failed"
                    job["completed_at"] = _now()
                    _save_job(config, job)
                    continue

                overlay_b64 = result.get("processed_image_base64")
                if overlay_b64:
                    get_redis(config).set(
                        _overlay_key(job_id),
                        base64.b64decode(overlay_b64),
                        ex=_OVERLAY_TTL,
                    )
                    job["has_overlay"] = True

                for key in (
                    "score",
                    "str_result",
                    "timing",
                    "metadata",
                    "model_used",
                    "model_version",
                    "overlay_bbox",
                    "overlay_width",
                    "overlay_height",
                    "overlay_downsample",
                    "processed_downsample",
                    "slide_level_used",
                    "region_id",
                ):
                    if key in result:
                        job[key] = result[key]

                job["status"] = STATUS_SUCCEEDED
                job["completed_at"] = _now()
                _save_job(config, job)
            except Exception:
                logger.exception("inference worker loop error")
                time.sleep(1)

    t = threading.Thread(target=_loop, name=f"inference-worker-{pool}", daemon=True)
    t.start()
