"""Environment-driven configuration shared by the gateway and the workers.

Everything is read from the environment so that the same image runs unchanged
under docker-compose, systemd and Kubernetes; only the environment differs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


def _get(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _get(env, name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_list(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _get(env, name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(":") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Runtime settings. See ``Settings.from_env`` for the variable names."""

    # --- Redis / queue ---------------------------------------------------- #
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "inference"
    consumer_group: str = "workers"

    # --- Job lifecycle ---------------------------------------------------- #
    job_ttl_seconds: int = 6 * 60 * 60
    overlay_ttl_seconds: int = 60 * 60
    # A running task whose worker has not renewed its lease for this long is
    # assumed orphaned (the pod died) and is handed to another worker.
    task_lease_seconds: int = 120
    task_max_attempts: int = 2
    max_regions_per_job: int = 500
    max_active_jobs_per_owner: int = 20

    # --- Overlay storage -------------------------------------------------- #
    overlay_store: str = "redis"  # redis | s3
    s3_bucket: Optional[str] = None
    s3_prefix: str = "overlays"
    s3_endpoint_url: Optional[str] = None
    s3_region: Optional[str] = None

    # --- Slides / inference ----------------------------------------------- #
    slide_roots: tuple[str, ...] = ("/slides",)
    max_region_pixels: int = 64_000_000

    # --- Gateway ---------------------------------------------------------- #
    api_token: Optional[str] = None
    model_catalog_path: Optional[str] = None
    gateway_stats_interval: float = 10.0

    # --- Worker ----------------------------------------------------------- #
    model_name: Optional[str] = None
    device: str = "auto"  # auto | cuda | cpu
    worker_http_port: int = 9100
    heartbeat_seconds: int = 10
    replica_ttl_seconds: int = 30
    progress_min_interval: float = 0.5
    worker_warmup: bool = True
    worker_eager_load: bool = True
    worker_consume_block_ms: int = 5000
    worker_reclaim_interval: float = 30.0
    worker_poll_once: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            redis_url=_get(env, "REDIS_URL", cls.redis_url) or cls.redis_url,
            key_prefix=_get(env, "INFERENCE_KEY_PREFIX", cls.key_prefix) or cls.key_prefix,
            consumer_group=_get(env, "INFERENCE_CONSUMER_GROUP", cls.consumer_group)
            or cls.consumer_group,
            job_ttl_seconds=_get_int(env, "JOB_TTL_SECONDS", cls.job_ttl_seconds),
            overlay_ttl_seconds=_get_int(env, "OVERLAY_TTL_SECONDS", cls.overlay_ttl_seconds),
            task_lease_seconds=_get_int(env, "TASK_LEASE_SECONDS", cls.task_lease_seconds),
            task_max_attempts=_get_int(env, "TASK_MAX_ATTEMPTS", cls.task_max_attempts),
            max_regions_per_job=_get_int(env, "MAX_REGIONS_PER_JOB", cls.max_regions_per_job),
            max_active_jobs_per_owner=_get_int(
                env, "MAX_ACTIVE_JOBS_PER_OWNER", cls.max_active_jobs_per_owner
            ),
            overlay_store=(_get(env, "OVERLAY_STORE", cls.overlay_store) or cls.overlay_store).lower(),
            s3_bucket=_get(env, "OVERLAY_S3_BUCKET"),
            s3_prefix=_get(env, "OVERLAY_S3_PREFIX", cls.s3_prefix) or cls.s3_prefix,
            s3_endpoint_url=_get(env, "OVERLAY_S3_ENDPOINT_URL"),
            s3_region=_get(env, "OVERLAY_S3_REGION"),
            slide_roots=_get_list(env, "SLIDE_ROOTS", cls.slide_roots),
            max_region_pixels=_get_int(env, "MAX_REGION_PIXELS", cls.max_region_pixels),
            api_token=_get(env, "GATEWAY_API_TOKEN"),
            model_catalog_path=_get(env, "MODEL_CATALOG_PATH"),
            gateway_stats_interval=_get_float(
                env, "GATEWAY_STATS_INTERVAL", cls.gateway_stats_interval
            ),
            model_name=_get(env, "MODEL_NAME"),
            device=(_get(env, "DEVICE", cls.device) or cls.device).lower(),
            worker_http_port=_get_int(env, "WORKER_HTTP_PORT", cls.worker_http_port),
            heartbeat_seconds=_get_int(env, "HEARTBEAT_SECONDS", cls.heartbeat_seconds),
            replica_ttl_seconds=_get_int(env, "REPLICA_TTL_SECONDS", cls.replica_ttl_seconds),
            progress_min_interval=_get_float(
                env, "PROGRESS_MIN_INTERVAL", cls.progress_min_interval
            ),
            worker_warmup=_get_bool(env, "WORKER_WARMUP", cls.worker_warmup),
            worker_eager_load=_get_bool(env, "WORKER_EAGER_LOAD", cls.worker_eager_load),
            worker_consume_block_ms=_get_int(
                env, "WORKER_CONSUME_BLOCK_MS", cls.worker_consume_block_ms
            ),
            worker_reclaim_interval=_get_float(
                env, "WORKER_RECLAIM_INTERVAL", cls.worker_reclaim_interval
            ),
        )

    # Key helpers are here so every component names Redis keys the same way.
    def job_key(self, job_id: str) -> str:
        return f"{self.key_prefix}:job:{job_id}"

    def task_key(self, job_id: str, task_key: str) -> str:
        return f"{self.key_prefix}:job:{job_id}:task:{task_key}"

    def task_index_key(self, job_id: str) -> str:
        return f"{self.key_prefix}:job:{job_id}:tasks"

    def overlay_key(self, job_id: str, task_key: str) -> str:
        return f"{self.key_prefix}:job:{job_id}:overlay:{task_key}"

    def owner_jobs_key(self, owner: str) -> str:
        return f"{self.key_prefix}:owner:{owner}:jobs"

    def recent_jobs_key(self) -> str:
        return f"{self.key_prefix}:jobs:recent"

    def stream_key(self, model_name: str) -> str:
        return f"{self.key_prefix}:tasks:{model_name}"

    def model_key(self, model_name: str) -> str:
        return f"{self.key_prefix}:model:{model_name}"

    def models_set_key(self) -> str:
        return f"{self.key_prefix}:models"

    def replica_key(self, model_name: str, replica_id: str) -> str:
        return f"{self.key_prefix}:model:{model_name}:replica:{replica_id}"

    def replicas_set_key(self, model_name: str) -> str:
        return f"{self.key_prefix}:model:{model_name}:replicas"


def connect_redis(settings: Settings):
    """Open a Redis client for ``settings.redis_url`` (bytes responses)."""
    import redis

    return redis.Redis.from_url(settings.redis_url, decode_responses=False)
