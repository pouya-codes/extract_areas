"""Inference gateway.

Stateless HTTP API in front of the per-model task queues. It validates a job,
writes its records, puts one task per region on the model's stream and then
answers polls from those records. Scale it horizontally behind a Service; no
request depends on which replica served the previous one.

Routes
------
GET  /healthz, /readyz, /metrics
GET  /v1/models                       catalogue + live replica / queue stats
GET  /v1/models/{name}
POST /v1/jobs                         submit a batch of regions   -> 202 job view
GET  /v1/jobs                         recent jobs (operators)
GET  /v1/jobs/{id}                    poll
POST /v1/jobs/{id}/cancel
GET  /v1/jobs/{id}/regions/{key}/overlay   PNG
GET  /v1/jobs/{id}/report.pdf         results of a finished job as a PDF
GET  /v1/queues                       per-model queue depth / lag / pending
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from inference_platform import __version__, metrics
from inference_platform.broker import TaskBroker
from inference_platform.catalog import load_catalog
from gateway.report import build_job_report, report_filename
from inference_platform.contracts import JobStatus, JobSubmission, TaskStatus
from inference_platform.overlays import build_overlay_store
from inference_platform.registry import ModelRegistry
from inference_platform.settings import Settings, connect_redis
from inference_platform.store import JobStore

logger = logging.getLogger(__name__)


class GatewayState:
    def __init__(self, settings: Settings, redis_client=None):
        self.settings = settings
        self.redis = redis_client if redis_client is not None else connect_redis(settings)
        self.store = JobStore(self.redis, settings, build_overlay_store(settings, self.redis))
        self.broker = TaskBroker(self.redis, settings)
        self.registry = ModelRegistry(self.redis, settings)
        self.catalog = load_catalog(settings.model_catalog_path)

    # ------------------------------------------------------------------ #
    def known_models(self) -> list[str]:
        return sorted(set(self.catalog) | set(self.registry.names()))

    def model_summary(self, name: str) -> Optional[dict[str, Any]]:
        descriptor = self.registry.get(name)
        catalog_entry = self.catalog.get(name)
        if descriptor is None and catalog_entry is None:
            return None
        replicas = self.registry.replicas(name)
        queue = self.broker.queue_stats(name)
        busy = sum(1 for r in replicas if r.get("busy"))
        summary: dict[str, Any] = {
            "name": name,
            "version": (descriptor or {}).get("version"),
            "description": (descriptor or {}).get("description") or (catalog_entry or {}).get("description", ""),
            "requires_mask": bool((descriptor or {}).get("requires_mask", False)),
            "supported_formats": (descriptor or {}).get("supported_formats") or [],
            "hyperparameters": (descriptor or {}).get("hyperparameters") or {},
            "pool": (descriptor or {}).get("pool") or (catalog_entry or {}).get("pool") or "general",
            "image": (descriptor or {}).get("image"),
            "descriptor_published_at": (descriptor or {}).get("published_at"),
            "in_catalog": catalog_entry is not None,
            "replicas": len(replicas),
            "replicas_busy": busy,
            "queue": queue,
            # Available means a job can be accepted: either a worker is up or
            # the deployment is declared and will be scaled up on demand.
            "available": bool(replicas) or catalog_entry is not None,
            "scale_to_zero": bool((catalog_entry or {}).get("scale_to_zero")),
        }
        metrics.QUEUE_DEPTH.labels(name).set(queue["depth"])
        metrics.QUEUE_LAG.labels(name).set(queue["lag"])
        metrics.QUEUE_PENDING.labels(name).set(queue["pending"])
        metrics.MODEL_REPLICAS.labels(name).set(len(replicas))
        return summary

    def refresh_stats(self) -> None:
        for name in self.known_models():
            try:
                self.model_summary(name)
            except Exception:  # noqa: BLE001
                logger.exception("stats refresh failed for %s", name)


def create_app(settings: Optional[Settings] = None, redis_client=None) -> FastAPI:
    settings = settings or Settings.from_env()
    state = GatewayState(settings, redis_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _stats_loop():
            while True:
                await asyncio.sleep(settings.gateway_stats_interval)
                await asyncio.get_running_loop().run_in_executor(None, state.refresh_stats)

        task = asyncio.create_task(_stats_loop()) if settings.gateway_stats_interval > 0 else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(
        title="AIM inference gateway",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.gateway = state
    app.mount("/metrics", make_asgi_app())

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    def require_token(request: Request) -> None:
        token = settings.api_token
        if not token:
            return
        header = request.headers.get("authorization", "")
        if header.startswith("Bearer ") and header[7:].strip() == token:
            return
        if request.headers.get("x-api-token", "") == token:
            return
        raise HTTPException(status_code=401, detail="missing or invalid API token")

    authed = [Depends(require_token)]

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> Response:
        try:
            state.redis.ping()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"status": "unavailable", "redis": str(exc)}, status_code=503)
        return JSONResponse({"status": "ok"})

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    @app.get("/v1/models", dependencies=authed)
    def list_models() -> dict[str, Any]:
        models = [m for m in (state.model_summary(n) for n in state.known_models()) if m]
        return {"models": models}

    @app.get("/v1/models/{name}", dependencies=authed)
    def get_model(name: str) -> dict[str, Any]:
        summary = state.model_summary(name)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown model '{name}'")
        return summary

    @app.get("/v1/queues", dependencies=authed)
    def list_queues() -> dict[str, Any]:
        return {
            "queues": {
                name: {
                    **state.broker.queue_stats(name),
                    "replicas": len(state.registry.replicas(name)),
                }
                for name in state.known_models()
            }
        }

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #
    @app.post("/v1/jobs", status_code=202, dependencies=authed)
    def submit_job(submission: JobSubmission) -> dict[str, Any]:
        name = submission.model_name
        if name not in state.known_models():
            metrics.JOBS_REJECTED.labels("unknown_model").inc()
            raise HTTPException(status_code=404, detail=f"unknown model '{name}'")
        if len(submission.regions) > settings.max_regions_per_job:
            metrics.JOBS_REJECTED.labels("too_many_regions").inc()
            raise HTTPException(
                status_code=413,
                detail=f"a job may contain at most {settings.max_regions_per_job} regions",
            )
        owner_key = submission.owner.key if submission.owner else None
        if owner_key and settings.max_active_jobs_per_owner > 0:
            active = state.store.active_job_count(owner_key)
            if active >= settings.max_active_jobs_per_owner:
                metrics.JOBS_REJECTED.labels("owner_quota").inc()
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"you already have {active} jobs in progress; wait for some to "
                        f"finish before submitting more"
                    ),
                )

        job = state.store.create_job(submission)
        state.broker.enqueue_many(name, job["job_id"], job["task_keys"])
        metrics.JOBS_SUBMITTED.labels(name).inc()
        metrics.TASKS_SUBMITTED.labels(name).inc(len(job["task_keys"]))
        logger.info(
            "job %s queued: model=%s regions=%d owner=%s image=%s",
            job["job_id"], name, len(job["task_keys"]), owner_key, submission.image_id,
        )
        view = state.store.job_view(job["job_id"])
        return {"job": view}

    @app.get("/v1/jobs", dependencies=authed)
    def list_jobs(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        return {"jobs": state.store.recent_jobs(limit)}

    @app.get("/v1/jobs/{job_id}", dependencies=authed)
    def get_job(job_id: str) -> dict[str, Any]:
        view = state.store.job_view(job_id)
        if view is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        return {"job": view}

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=authed)
    def cancel_job(job_id: str) -> dict[str, Any]:
        if state.store.request_cancel(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        return {"job": state.store.job_view(job_id)}

    @app.get("/v1/jobs/{job_id}/regions/{task_key}/overlay", dependencies=authed)
    def get_overlay(job_id: str, task_key: str) -> Response:
        task = state.store.get_task(job_id, task_key)
        if task is None:
            raise HTTPException(status_code=404, detail="job or region not found")
        if task.get("status") != TaskStatus.DONE:
            raise HTTPException(status_code=409, detail=f"region status is {task.get('status')}")
        overlay = state.store.get_overlay(job_id, task_key)
        if overlay is None:
            raise HTTPException(status_code=404, detail="overlay has expired")
        return Response(content=overlay, media_type="image/png")

    @app.get("/v1/jobs/{job_id}/report.pdf", dependencies=authed)
    def get_job_report(job_id: str, overlays: bool = Query(default=True)) -> Response:
        view = state.store.job_view(job_id)
        if view is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        if view["status"] not in JobStatus.TERMINAL:
            raise HTTPException(
                status_code=409,
                detail=f"job is {view['status']}; the report is available once it finishes",
            )
        pdf = build_job_report(view, state.store.get_overlay if overlays else None)
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{report_filename(view)}"'},
        )

    return app


# ``uvicorn gateway.app:app`` entry point.
app = create_app()
