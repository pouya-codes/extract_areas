"""Model registry: which models exist and which workers are serving them.

Workers publish a *descriptor* (name, version, hyperparameter schema) once at
start-up and a *heartbeat* every few seconds while alive. Descriptors never
expire, so a model whose deployment has scaled to zero still shows up in the
catalogue with its parameters; heartbeats expire, so replica counts are live.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from inference_platform.settings import Settings

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, redis, settings: Settings):
        self._redis = redis
        self._settings = settings

    def publish(self, descriptor: dict[str, Any]) -> None:
        name = descriptor["name"]
        record = dict(descriptor)
        record["published_at"] = _now()
        pipe = self._redis.pipeline(transaction=False)
        pipe.set(self._settings.model_key(name), json.dumps(record).encode("utf-8"))
        pipe.sadd(self._settings.models_set_key(), name)
        pipe.execute()

    def get(self, name: str) -> Optional[dict[str, Any]]:
        raw = self._redis.get(self._settings.model_key(name))
        return json.loads(raw) if raw else None

    def names(self) -> list[str]:
        members = self._redis.smembers(self._settings.models_set_key()) or []
        return sorted(m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members)

    def heartbeat(self, name: str, replica_id: str, info: dict[str, Any]) -> None:
        record = dict(info)
        record["replica_id"] = replica_id
        record["seen_at"] = _now()
        pipe = self._redis.pipeline(transaction=False)
        pipe.set(
            self._settings.replica_key(name, replica_id),
            json.dumps(record).encode("utf-8"),
            ex=self._settings.replica_ttl_seconds,
        )
        pipe.sadd(self._settings.replicas_set_key(name), replica_id)
        pipe.execute()

    def deregister(self, name: str, replica_id: str) -> None:
        pipe = self._redis.pipeline(transaction=False)
        pipe.delete(self._settings.replica_key(name, replica_id))
        pipe.srem(self._settings.replicas_set_key(name), replica_id)
        pipe.execute()

    def replicas(self, name: str) -> list[dict[str, Any]]:
        """Live replicas of ``name``; stale ids are pruned as they are found."""
        ids = self._redis.smembers(self._settings.replicas_set_key(name)) or []
        ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i) for i in ids]
        if not ids:
            return []
        raws = self._redis.mget([self._settings.replica_key(name, i) for i in ids])
        live: list[dict[str, Any]] = []
        stale: list[str] = []
        for replica_id, raw in zip(ids, raws):
            if raw:
                live.append(json.loads(raw))
            else:
                stale.append(replica_id)
        if stale:
            self._redis.srem(self._settings.replicas_set_key(name), *stale)
        return sorted(live, key=lambda r: r.get("replica_id", ""))
