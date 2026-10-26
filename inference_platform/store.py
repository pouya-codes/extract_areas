"""Job and task records in Redis.

A *job* is one submission: one model, one slide, many regions. Each region is
a *task*, and a task is the unit of work a worker takes off the queue. Tasks
are stored under their own keys so that the worker running one only ever
writes to that one key; nothing in the hot path needs a lock. The job-level
status, counters and message are derived from the tasks when the job is read
(:meth:`JobStore.job_view`) rather than kept up to date by the workers.

Everything expires with ``JOB_TTL_SECONDS``. The durable record of a run is
the AI-run history AIMViewer writes once it has collected the overlays.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from inference_platform.contracts import JobStatus, JobSubmission, TaskStatus
from inference_platform.overlays import OverlayStore, RedisOverlayStore
from inference_platform.settings import Settings

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _loads(raw: Optional[bytes]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    return json.loads(raw)


class JobStore:
    def __init__(self, redis, settings: Settings, overlays: Optional[OverlayStore] = None):
        self._redis = redis
        self._settings = settings
        self._overlays = overlays or RedisOverlayStore(redis, settings)

    # ------------------------------------------------------------------ #
    # Creation
    # ------------------------------------------------------------------ #
    def create_job(self, submission: JobSubmission) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        created = _now()
        owner_key = submission.owner.key if submission.owner else None
        job = {
            "job_id": job_id,
            "model_name": submission.model_name,
            "slide_path": submission.slide_path,
            "image_id": submission.image_id,
            "owner": submission.owner.model_dump() if submission.owner else None,
            "owner_key": owner_key,
            "downsample": submission.downsample,
            "overlay_downsample": submission.overlay_downsample,
            "hyperparameters": submission.hyperparameters or {},
            "priority": submission.priority,
            "client_ref": submission.client_ref,
            "regions_total": len(submission.regions),
            "task_keys": [r.key for r in submission.regions],
            "status": JobStatus.QUEUED,
            "cancel_requested": False,
            "created_at": created,
            "updated_at": created,
            "completed_at": None,
        }
        tasks = [
            {
                "key": region.key,
                "region_id": region.region_id,
                "label_slug": region.label_slug,
                "roi_bbox": region.roi_bbox,
                "polygon": region.polygon,
                "status": TaskStatus.PENDING,
                "attempt": 0,
                "worker": None,
                "error": None,
                "str_result": None,
                "score": None,
                "timing": None,
                "metadata": None,
                "bbox": None,
                "overlay_width": None,
                "overlay_height": None,
                "has_overlay": False,
                "patch_done": None,
                "patch_total": None,
                "progress_percent": None,
                "progress_stage": None,
                "progress_label": None,
                "created_at": created,
                "started_at": None,
                "completed_at": None,
            }
            for region in submission.regions
        ]

        ttl = self._settings.job_ttl_seconds
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(self._settings.job_key(job_id), _dumps(job), ex=ttl)
        for task in tasks:
            pipe.set(self._settings.task_key(job_id, task["key"]), _dumps(task), ex=ttl)
        pipe.zadd(self._settings.recent_jobs_key(), {job_id: time.time()})
        pipe.expire(self._settings.recent_jobs_key(), ttl)
        if owner_key:
            pipe.zadd(self._settings.owner_jobs_key(owner_key), {job_id: time.time()})
            pipe.expire(self._settings.owner_jobs_key(owner_key), ttl)
        pipe.execute()
        return job

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return _loads(self._redis.get(self._settings.job_key(job_id)))

    def get_task(self, job_id: str, task_key: str) -> Optional[dict[str, Any]]:
        return _loads(self._redis.get(self._settings.task_key(job_id, task_key)))

    def list_tasks(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        keys = job.get("task_keys") or []
        if not keys:
            return []
        raws = self._redis.mget([self._settings.task_key(job["job_id"], k) for k in keys])
        return [t for t in (_loads(r) for r in raws) if t is not None]

    def is_cancel_requested(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and job.get("cancel_requested"))

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def update_job(self, job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.update(fields)
        job["updated_at"] = _now()
        self._redis.set(self._settings.job_key(job_id), _dumps(job), ex=self._settings.job_ttl_seconds)
        return job

    def update_task(self, job_id: str, task_key: str, **fields: Any) -> Optional[dict[str, Any]]:
        task = self.get_task(job_id, task_key)
        if task is None:
            return None
        task.update(fields)
        task["updated_at"] = _now()
        self._redis.set(
            self._settings.task_key(job_id, task_key),
            _dumps(task),
            ex=self._settings.job_ttl_seconds,
        )
        return task

    def request_cancel(self, job_id: str) -> Optional[dict[str, Any]]:
        """Flag the job and cancel every task that has not started yet.

        Running tasks are not touched here: their worker notices the flag at
        its next progress report and stops on its own.
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.get("status") in JobStatus.TERMINAL:
            return job
        now = _now()
        for task in self.list_tasks(job):
            if task["status"] == TaskStatus.PENDING:
                self.update_task(
                    job_id, task["key"], status=TaskStatus.CANCELLED, completed_at=now
                )
        return self.update_job(job_id, cancel_requested=True)

    # ------------------------------------------------------------------ #
    # Overlays
    # ------------------------------------------------------------------ #
    def put_overlay(self, job_id: str, task_key: str, data: bytes) -> None:
        self._overlays.put(job_id, task_key, data)

    def get_overlay(self, job_id: str, task_key: str) -> Optional[bytes]:
        return self._overlays.get(job_id, task_key)

    # ------------------------------------------------------------------ #
    # Quotas / listings
    # ------------------------------------------------------------------ #
    def active_job_count(self, owner_key: str) -> int:
        """Jobs this owner has that are not finished; finished ones are pruned."""
        key = self._settings.owner_jobs_key(owner_key)
        job_ids = [j.decode("utf-8") if isinstance(j, bytes) else str(j) for j in self._redis.zrange(key, 0, -1)]
        if not job_ids:
            return 0
        active = 0
        finished: list[str] = []
        for job_id in job_ids:
            view = self.job_view(job_id)
            if view is None or view["status"] in JobStatus.TERMINAL:
                finished.append(job_id)
            else:
                active += 1
        if finished:
            self._redis.zrem(key, *finished)
        return active

    def recent_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        job_ids = self._redis.zrevrange(self._settings.recent_jobs_key(), 0, max(0, limit - 1))
        views = []
        for raw_id in job_ids:
            job_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            view = self.job_view(job_id, include_regions=False)
            if view is not None:
                views.append(view)
        return views

    # ------------------------------------------------------------------ #
    # Aggregated view
    # ------------------------------------------------------------------ #
    def job_view(self, job_id: str, *, include_regions: bool = True) -> Optional[dict[str, Any]]:
        """The job as the client sees it, with status derived from its tasks."""
        job = self.get_job(job_id)
        if job is None:
            return None
        tasks = self.list_tasks(job)

        counts = {status: 0 for status in (
            TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.DONE,
            TaskStatus.FAILED, TaskStatus.CANCELLED,
        )}
        for task in tasks:
            counts[task.get("status", TaskStatus.PENDING)] = counts.get(task.get("status"), 0) + 1

        total = job.get("regions_total") or len(tasks)
        done, failed, cancelled = counts[TaskStatus.DONE], counts[TaskStatus.FAILED], counts[TaskStatus.CANCELLED]
        running, pending = counts[TaskStatus.RUNNING], counts[TaskStatus.PENDING]
        # Tasks whose records expired count as finished so the job can close.
        missing = max(0, total - len(tasks))
        all_terminal = running == 0 and pending == 0

        if all_terminal:
            if cancelled and job.get("cancel_requested"):
                status = JobStatus.CANCELLED
                message = f"Cancelled after {done} of {total} region(s)"
            elif done == 0 and (failed or missing):
                status = JobStatus.FAILED
                message = f"All {failed + missing} region(s) failed"
            else:
                status = JobStatus.SUCCEEDED
                message = (
                    f"{done} of {total} region(s) processed" if failed or cancelled or missing
                    else f"{done} region(s) processed"
                )
            stage = "done"
        elif running or done or failed:
            status = JobStatus.RUNNING
            stage = "processing"
            message = "Cancelling..." if job.get("cancel_requested") else (
                f"Processing {total} region(s)... {done + failed + cancelled} finished"
            )
        else:
            status = JobStatus.QUEUED
            stage = "queued"
            message = "Cancelling..." if job.get("cancel_requested") else "Waiting for a worker..."

        error = next(
            (t["error"] for t in tasks if t.get("status") == TaskStatus.FAILED and t.get("error")),
            None,
        )

        if all_terminal and not job.get("completed_at"):
            job = self.update_job(job_id, status=status, completed_at=_now()) or job
        elif job.get("status") != status and not all_terminal:
            job = self.update_job(job_id, status=status) or job

        view = {
            "job_id": job_id,
            "image_id": job.get("image_id"),
            "model_name": job.get("model_name"),
            "slide_path": job.get("slide_path"),
            "owner": job.get("owner"),
            "user_id": (job.get("owner") or {}).get("user_id"),
            "username": (job.get("owner") or {}).get("username", ""),
            "client_ref": job.get("client_ref"),
            "status": status,
            "stage": stage,
            "message": message,
            "error": error,
            "cancel_requested": bool(job.get("cancel_requested")),
            "downsample": job.get("downsample"),
            "overlay_downsample": job.get("overlay_downsample"),
            "hyperparameters": job.get("hyperparameters") or {},
            "regions_total": total,
            "regions_done": done,
            "regions_failed": failed,
            "regions_cancelled": cancelled,
            "regions_running": running,
            "regions_pending": pending,
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "completed_at": job.get("completed_at"),
        }
        if include_regions:
            view["regions"] = [self._region_view(t) for t in tasks]
        return view

    @staticmethod
    def _region_view(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": task["key"],
            "region_id": task.get("region_id"),
            "label_slug": task.get("label_slug", ""),
            "status": task.get("status"),
            "attempt": task.get("attempt", 0),
            "worker": task.get("worker"),
            "error": task.get("error"),
            "str_result": task.get("str_result"),
            "score": task.get("score"),
            "timing": task.get("timing"),
            "bbox": task.get("bbox"),
            "overlay_width": task.get("overlay_width"),
            "overlay_height": task.get("overlay_height"),
            "has_overlay": bool(task.get("has_overlay")),
            "patch_done": task.get("patch_done"),
            "patch_total": task.get("patch_total"),
            "progress_percent": task.get("progress_percent"),
            "progress_stage": task.get("progress_stage"),
            "progress_label": task.get("progress_label"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
        }
