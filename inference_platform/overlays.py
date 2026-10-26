"""Where finished overlay PNGs are kept until the client collects them.

Redis is the default and is right for a single cluster with a modest number of
in-flight jobs. S3 (or any S3-compatible store such as MinIO) is for larger
deployments where overlay bytes would otherwise crowd job state out of Redis.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol

from inference_platform.settings import Settings

logger = logging.getLogger(__name__)


class OverlayStore(Protocol):
    def put(self, job_id: str, task_key: str, data: bytes) -> None: ...

    def get(self, job_id: str, task_key: str) -> Optional[bytes]: ...

    def delete(self, job_id: str, task_key: str) -> None: ...


class RedisOverlayStore:
    def __init__(self, redis, settings: Settings):
        self._redis = redis
        self._settings = settings

    def put(self, job_id: str, task_key: str, data: bytes) -> None:
        self._redis.set(
            self._settings.overlay_key(job_id, task_key),
            data,
            ex=self._settings.overlay_ttl_seconds,
        )

    def get(self, job_id: str, task_key: str) -> Optional[bytes]:
        return self._redis.get(self._settings.overlay_key(job_id, task_key))

    def delete(self, job_id: str, task_key: str) -> None:
        self._redis.delete(self._settings.overlay_key(job_id, task_key))


class S3OverlayStore:
    """Overlays as ``<prefix>/<job_id>/<task_key>.png`` objects.

    Expiry is left to a bucket lifecycle rule (set it to a day or so); the
    store itself never lists or sweeps objects.
    """

    def __init__(self, settings: Settings, client=None):
        if not settings.s3_bucket:
            raise ValueError("OVERLAY_S3_BUCKET is required when OVERLAY_STORE=s3")
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                region_name=settings.s3_region,
            )
        self._client = client
        self._bucket = settings.s3_bucket
        self._prefix = settings.s3_prefix.strip("/")

    def _key(self, job_id: str, task_key: str) -> str:
        return f"{self._prefix}/{job_id}/{task_key}.png"

    def put(self, job_id: str, task_key: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key(job_id, task_key),
            Body=data,
            ContentType="image/png",
        )

    def get(self, job_id: str, task_key: str) -> Optional[bytes]:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._key(job_id, task_key))
        except Exception as exc:  # noqa: BLE001 - botocore error classes are dynamic
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return response["Body"].read()

    def delete(self, job_id: str, task_key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self._key(job_id, task_key))


def build_overlay_store(settings: Settings, redis) -> OverlayStore:
    if settings.overlay_store == "s3":
        return S3OverlayStore(settings)
    if settings.overlay_store != "redis":
        raise ValueError(f"unknown OVERLAY_STORE {settings.overlay_store!r}; use redis or s3")
    return RedisOverlayStore(redis, settings)
