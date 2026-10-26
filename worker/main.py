"""Model worker main loop.

    python -m worker.main                 serve MODEL_NAME until SIGTERM
    python -m worker.main --describe      publish the model descriptor and exit
    python -m worker.main --max-tasks N   process N tasks then exit (tests, batch)

Life of a task
--------------
1. ``XREADGROUP`` hands the worker one stream entry (job_id, task_key).
2. The task record is read. If the job was cancelled, or the task already
   finished (a duplicate delivery), the entry is acknowledged and skipped.
3. The task is marked ``running`` with this worker's name and attempt count.
   A lease thread re-claims the entry every few seconds so other workers can
   tell it is still alive.
4. The model runs. Its progress callbacks are written to the task record
   (rate-limited) and double as cancellation checks.
5. The overlay is stored, the task is marked ``done``/``failed``/``cancelled``,
   and only then is the entry acknowledged and deleted.

If the process dies between 3 and 5 the entry stays pending; once its lease
is stale another worker reclaims it and the attempt counter decides whether
to retry or to fail the task for good.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from inference_platform import metrics
from inference_platform.broker import Message, TaskBroker, default_consumer_name
from inference_platform.contracts import TaskStatus
from inference_platform.logging_setup import configure_logging
from inference_platform.overlays import build_overlay_store
from inference_platform.registry import ModelRegistry
from inference_platform.settings import Settings, connect_redis
from inference_platform.store import JobStore
from progress_report import InferenceCancelled, reset_progress_reporter, set_progress_reporter
from worker import loaders
from worker.core import InferenceCore

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Worker:
    def __init__(
        self,
        settings: Settings,
        *,
        redis_client=None,
        model=None,
        core: Optional[InferenceCore] = None,
        consumer: Optional[str] = None,
        serve_http: bool = True,
        raw_config: Optional[dict[str, Any]] = None,
    ):
        if not settings.model_name:
            raise ValueError("MODEL_NAME is required")
        self.settings = settings
        self.model_name = settings.model_name
        self.redis = redis_client if redis_client is not None else connect_redis(settings)
        self.store = JobStore(self.redis, settings, build_overlay_store(settings, self.redis))
        self.broker = TaskBroker(self.redis, settings)
        self.registry = ModelRegistry(self.redis, settings)
        self.core = core or InferenceCore(settings)
        self.consumer = consumer or default_consumer_name()
        self.model = model
        self._raw_config = raw_config
        self._serve_http = serve_http

        self._ready = False
        self._stopping = threading.Event()
        self._busy = False
        self._current: Optional[Message] = None
        self._tasks_done = 0
        self._tasks_failed = 0
        self._started_at = _now()
        self._http = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        started = time.perf_counter()
        if self.model is None:
            self.model = loaders.build_model(self.model_name, self.settings, self._raw_config)
            if self.settings.worker_warmup:
                loaders.warmup(self.model, self.model_name)
        metrics.MODEL_LOAD_SECONDS.labels(self.model_name).set(time.perf_counter() - started)
        metrics.MODEL_LOADED.labels(self.model_name).set(1)
        self.registry.publish(loaders.describe(self.model, self.model_name, self.settings, self._raw_config))
        self.broker.ensure_group(self.model_name)
        self._ready = True
        logger.info("worker %s ready for model %s", self.consumer, self.model_name)

    def describe_only(self) -> dict[str, Any]:
        if self.model is None:
            self.model = loaders.build_model(self.model_name, self.settings, self._raw_config)
        descriptor = loaders.describe(self.model, self.model_name, self.settings, self._raw_config)
        self.registry.publish(descriptor)
        return descriptor

    def stop(self, *_args) -> None:
        if not self._stopping.is_set():
            logger.info("stop requested; finishing current task then exiting")
        self._stopping.set()

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "consumer": self.consumer,
            "ready": self._ready and not self._stopping.is_set(),
            "busy": self._busy,
            "current_job": self._current.job_id if self._current else None,
            "current_task": self._current.task_key if self._current else None,
            "tasks_done": self._tasks_done,
            "tasks_failed": self._tasks_failed,
            "started_at": self._started_at,
            "hostname": socket.gethostname(),
            "device": self.settings.device,
            "pid": os.getpid(),
        }

    def _heartbeat_loop(self) -> None:
        while not self._stopping.wait(self.settings.heartbeat_seconds):
            try:
                self.registry.heartbeat(self.model_name, self.consumer, self.status())
            except Exception:  # noqa: BLE001
                logger.exception("heartbeat failed")

    def run(self, max_tasks: Optional[int] = None) -> int:
        """Serve tasks until stopped; returns the number of tasks handled."""
        if not self._ready:
            self.load()
        if self._serve_http:
            from worker import health

            self._http = health.serve(self.settings.worker_http_port, self.status, lambda: self._ready and not self._stopping.is_set())
        self.registry.heartbeat(self.model_name, self.consumer, self.status())
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        self._heartbeat_thread.start()

        handled = 0
        last_reclaim = 0.0
        try:
            while not self._stopping.is_set():
                if max_tasks is not None and handled >= max_tasks:
                    break
                try:
                    messages = self.broker.consume(
                        self.model_name, self.consumer, block_ms=self.settings.worker_consume_block_ms
                    )
                    if not messages and time.monotonic() - last_reclaim >= self.settings.worker_reclaim_interval:
                        last_reclaim = time.monotonic()
                        messages = self.broker.reclaim_stale(
                            self.model_name,
                            self.consumer,
                            min_idle_ms=self.settings.task_lease_seconds * 1000,
                        )
                except Exception:  # noqa: BLE001 - Redis hiccup: back off, keep serving
                    logger.exception("queue read failed; retrying in 2s")
                    self._stopping.wait(2.0)
                    continue
                for message in messages:
                    self.handle(message)
                    handled += 1
        finally:
            self._ready = False
            try:
                self.registry.deregister(self.model_name, self.consumer)
            except Exception:  # noqa: BLE001
                logger.exception("deregister failed")
            if self._http is not None:
                self._http.shutdown()
            if self.model is not None and hasattr(self.model, "cleanup"):
                try:
                    self.model.cleanup()
                except Exception:  # noqa: BLE001
                    logger.exception("model cleanup failed")
        return handled

    # ------------------------------------------------------------------ #
    # One task
    # ------------------------------------------------------------------ #
    def handle(self, message: Message) -> None:
        job_id, key = message.job_id, message.task_key
        log = logging.LoggerAdapter(logger, {"job_id": job_id, "task_key": key, "model": self.model_name})

        task = self.store.get_task(job_id, key)
        job = self.store.get_job(job_id)
        if task is None or job is None:
            log.warning("task record missing (expired?); dropping entry %s", message.msg_id)
            self.broker.ack(self.model_name, message.msg_id)
            return
        if task["status"] in TaskStatus.TERMINAL:
            log.info("task already %s; dropping duplicate delivery", task["status"])
            self.broker.ack(self.model_name, message.msg_id)
            return
        if job.get("cancel_requested"):
            self.store.update_task(job_id, key, status=TaskStatus.CANCELLED, completed_at=_now())
            self.broker.ack(self.model_name, message.msg_id)
            metrics.TASKS_PROCESSED.labels(self.model_name, TaskStatus.CANCELLED).inc()
            return

        attempt = int(task.get("attempt") or 0) + 1
        if attempt > self.settings.task_max_attempts:
            log.error("giving up after %d attempts", attempt - 1)
            self.store.update_task(
                job_id, key,
                status=TaskStatus.FAILED,
                error=(
                    f"The worker processing this region stopped unexpectedly "
                    f"{attempt - 1} time(s); giving up."
                ),
                completed_at=_now(),
            )
            self.broker.ack(self.model_name, message.msg_id)
            metrics.TASKS_PROCESSED.labels(self.model_name, TaskStatus.FAILED).inc()
            self._tasks_failed += 1
            return

        self.store.update_task(
            job_id, key,
            status=TaskStatus.RUNNING,
            attempt=attempt,
            worker=self.consumer,
            started_at=_now(),
            error=None,
            progress_stage="starting",
            progress_label="Reading slide region…",
        )
        if attempt > 1:
            log.warning("retrying (attempt %d of %d)", attempt, self.settings.task_max_attempts)

        self._busy = True
        self._current = message
        metrics.WORKER_BUSY.labels(self.model_name).set(1)
        lease_stop = threading.Event()
        lease_thread = threading.Thread(
            target=self._lease_loop, args=(message, lease_stop), name="lease", daemon=True
        )
        lease_thread.start()

        wall_started = time.perf_counter()
        status = TaskStatus.FAILED
        try:
            payload = {
                "slide_path": job["slide_path"],
                "roi_bbox": task["roi_bbox"],
                "polygon": task.get("polygon") or [],
                "downsample": job.get("downsample") or 1.0,
                "overlay_downsample": job.get("overlay_downsample"),
                "model_name": self.model_name,
                "hyperparameters": job.get("hyperparameters") or {},
                "region_id": str(task.get("region_id") or ""),
            }
            token = set_progress_reporter(self._progress_reporter(job_id, key))
            try:
                result = self.core.run(self.model, payload)
            finally:
                reset_progress_reporter(token)

            if result.get("status") != "success":
                self.store.update_task(
                    job_id, key,
                    status=TaskStatus.FAILED,
                    error=result.get("message") or "inference failed",
                    completed_at=_now(),
                )
                log.error("task failed: %s", result.get("message"))
            else:
                overlay = result.pop("overlay_png", None)
                has_overlay = False
                if overlay:
                    self.store.put_overlay(job_id, key, overlay)
                    has_overlay = True
                timing = dict(result.get("timing") or {})
                timing["wall_seconds"] = round(time.perf_counter() - wall_started, 3)
                self.store.update_task(
                    job_id, key,
                    status=TaskStatus.DONE,
                    error=None,
                    str_result=result.get("str_result"),
                    score=result.get("score"),
                    timing=timing,
                    metadata=result.get("metadata"),
                    bbox=result.get("overlay_bbox"),
                    overlay_width=result.get("overlay_width"),
                    overlay_height=result.get("overlay_height"),
                    has_overlay=has_overlay,
                    progress_percent=100,
                    progress_stage="done",
                    progress_label="Complete",
                    completed_at=_now(),
                )
                status = TaskStatus.DONE
                if timing.get("model_seconds") is not None:
                    metrics.MODEL_SECONDS.labels(self.model_name).observe(timing["model_seconds"])
                if timing.get("region_pixels"):
                    metrics.REGION_PIXELS.labels(self.model_name).observe(timing["region_pixels"])
                log.info("task done in %.1fs: %s", timing["wall_seconds"], result.get("str_result"))
        except InferenceCancelled:
            status = TaskStatus.CANCELLED
            self.store.update_task(job_id, key, status=TaskStatus.CANCELLED, completed_at=_now())
            log.info("task cancelled by request")
        except Exception as exc:  # noqa: BLE001
            log.exception("task crashed")
            self.store.update_task(
                job_id, key, status=TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}", completed_at=_now()
            )
        finally:
            lease_stop.set()
            lease_thread.join(timeout=5)
            self.broker.ack(self.model_name, message.msg_id)
            self._busy = False
            self._current = None
            metrics.WORKER_BUSY.labels(self.model_name).set(0)
            metrics.TASKS_PROCESSED.labels(self.model_name, status).inc()
            metrics.TASK_SECONDS.labels(self.model_name).observe(time.perf_counter() - wall_started)
            if status == TaskStatus.DONE:
                self._tasks_done += 1
            elif status == TaskStatus.FAILED:
                self._tasks_failed += 1
            _release_gpu_memory()

    def _lease_loop(self, message: Message, stop: threading.Event) -> None:
        interval = max(1.0, self.settings.task_lease_seconds / 3)
        while not stop.wait(interval):
            try:
                self.broker.renew(self.model_name, message.msg_id, self.consumer)
            except Exception:  # noqa: BLE001
                logger.exception("lease renewal failed for %s", message.msg_id)

    def _progress_reporter(self, job_id: str, key: str):
        last_write = 0.0
        last_cancel_check = 0.0
        allowed = ("patch_done", "patch_total", "progress_percent", "progress_stage", "progress_label")

        def report(**fields: Any) -> None:
            nonlocal last_write, last_cancel_check
            now = time.monotonic()
            if now - last_cancel_check >= 1.0:
                last_cancel_check = now
                if self.store.is_cancel_requested(job_id):
                    raise InferenceCancelled(job_id)
            if now - last_write < self.settings.progress_min_interval:
                return
            update = {k: v for k, v in fields.items() if k in allowed and v is not None}
            if update:
                last_write = now
                self.store.update_task(job_id, key, **update)

        return report


def _release_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AIM inference model worker")
    parser.add_argument("--describe", action="store_true", help="publish the descriptor and exit")
    parser.add_argument("--max-tasks", type=int, default=None, help="exit after this many tasks")
    parser.add_argument("--no-http", action="store_true", help="do not start the probe/metrics server")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    configure_logging(f"worker-{settings.model_name or 'unset'}")
    worker = Worker(settings, serve_http=not args.no_http)

    if args.describe:
        descriptor = worker.describe_only()
        logger.info("published descriptor for %s (%d hyperparameters)", descriptor["name"], len(descriptor["hyperparameters"]))
        return 0

    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    handled = worker.run(max_tasks=args.max_tasks)
    logger.info("worker exiting after %d task(s)", handled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
