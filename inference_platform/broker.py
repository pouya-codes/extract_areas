"""Per-model task queue on Redis Streams.

Each model has its own stream (``inference:tasks:<model>``) and one consumer
group. Worker pods for that model are the consumers. A task stays *pending*
in the group until the worker that took it acknowledges it, which happens
only after the task's terminal state has been written, so a worker that dies
mid-task leaves the entry pending and another worker reclaims it once its
lease has gone stale.

The queues are also what the autoscaler watches: KEDA's ``redis-streams``
scaler reads the group lag of each stream and sizes that model's deployment
to match.
"""
from __future__ import annotations

import logging
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from inference_platform.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    msg_id: str
    job_id: str
    task_key: str
    # How many times the group has delivered this entry; >1 means a reclaim.
    deliveries: int = 1


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def default_consumer_name() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"


class TaskBroker:
    def __init__(self, redis, settings: Settings):
        self._redis = redis
        self._settings = settings
        self._group = settings.consumer_group

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def stream(self, model_name: str) -> str:
        return self._settings.stream_key(model_name)

    def ensure_group(self, model_name: str) -> None:
        """Create the stream and consumer group if they do not exist yet."""
        try:
            self._redis.xgroup_create(self.stream(model_name), self._group, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------ #
    # Producer side
    # ------------------------------------------------------------------ #
    def enqueue_many(self, model_name: str, job_id: str, task_keys: list[str]) -> list[str]:
        self.ensure_group(model_name)
        stream = self.stream(model_name)
        pipe = self._redis.pipeline(transaction=False)
        for key in task_keys:
            pipe.xadd(stream, {"job_id": job_id, "task_key": key})
        return [_text(msg_id) for msg_id in pipe.execute()]

    def enqueue(self, model_name: str, job_id: str, task_key: str) -> str:
        return self.enqueue_many(model_name, job_id, [task_key])[0]

    # ------------------------------------------------------------------ #
    # Consumer side
    # ------------------------------------------------------------------ #
    def consume(
        self, model_name: str, consumer: str, *, block_ms: int = 5000, count: int = 1
    ) -> list[Message]:
        """Take up to ``count`` new entries, waiting at most ``block_ms``."""
        try:
            response = self._redis.xreadgroup(
                self._group, consumer, {self.stream(model_name): ">"}, count=count, block=block_ms
            )
        except _redis_timeout_error() as exc:
            # redis-py arms a client-side timer for blocking reads; when the
            # server's "nothing arrived" reply lands at the same instant the
            # timer fires, the client raises instead of returning []. Either
            # way the queue was empty for ``block_ms``.
            logger.debug("blocking read timed out without entries: %s", exc)
            return []
        messages: list[Message] = []
        for _stream, entries in response or []:
            for msg_id, fields in entries:
                messages.append(self._message(msg_id, fields))
        return messages

    def reclaim_stale(
        self, model_name: str, consumer: str, *, min_idle_ms: int, count: int = 1
    ) -> list[Message]:
        """Take over entries another consumer has held longer than ``min_idle_ms``.

        Uses XPENDING + XCLAIM rather than XAUTOCLAIM so it works on Redis 6.0
        as well as on the fake used in tests.
        """
        stream = self.stream(model_name)
        try:
            pending = self._redis.xpending_range(
                stream, self._group, min="-", max="+", count=count * 4, idle=min_idle_ms
            )
        except TypeError:
            # Older clients lack the ``idle`` filter; filter client-side.
            pending = [
                entry
                for entry in self._redis.xpending_range(stream, self._group, min="-", max="+", count=count * 4)
                if int(entry.get("time_since_delivered", 0)) >= min_idle_ms
            ]
        if not pending:
            return []
        ids = [entry["message_id"] for entry in pending[:count]]
        claimed = self._redis.xclaim(stream, self._group, consumer, min_idle_ms, ids)
        messages: list[Message] = []
        for msg_id, fields in claimed:
            if fields is None:
                # The entry was deleted from the stream but still pending; drop it.
                self._redis.xack(stream, self._group, msg_id)
                continue
            # XPENDING reports deliveries so far; this claim is one more.
            deliveries = next(
                (int(e["times_delivered"]) + 1 for e in pending if _text(e["message_id"]) == _text(msg_id)),
                2,
            )
            messages.append(self._message(msg_id, fields, deliveries=deliveries))
        if messages:
            logger.warning(
                "reclaimed %d stale task(s) on %s for consumer %s", len(messages), stream, consumer
            )
        return messages

    def renew(self, model_name: str, msg_id: str, consumer: str) -> None:
        """Reset the idle clock on an in-flight entry (lease heartbeat)."""
        self._redis.xclaim(self.stream(model_name), self._group, consumer, 0, [msg_id], justid=True)

    def ack(self, model_name: str, msg_id: str) -> None:
        stream = self.stream(model_name)
        pipe = self._redis.pipeline(transaction=False)
        pipe.xack(stream, self._group, msg_id)
        pipe.xdel(stream, msg_id)
        pipe.execute()

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    def queue_stats(self, model_name: str) -> dict:
        """Depth and group lag of one model's queue, as the autoscaler sees it."""
        stream = self.stream(model_name)
        depth = int(self._redis.xlen(stream) or 0)
        pending = 0
        lag: Optional[int] = None
        try:
            for group in self._redis.xinfo_groups(stream):
                if _text(group.get("name", b"")) == self._group:
                    pending = int(group.get("pending", 0) or 0)
                    raw_lag = group.get("lag")
                    lag = int(raw_lag) if raw_lag is not None else None
        except Exception:  # noqa: BLE001 - stream may not exist yet
            pass
        if lag is None:
            # Redis < 7 does not report lag; approximate as entries not yet delivered.
            lag = max(0, depth - pending)
        return {"depth": depth, "pending": pending, "lag": lag}

    def _message(self, msg_id, fields, *, deliveries: int = 1) -> Message:
        decoded = {_text(k): _text(v) for k, v in fields.items()}
        return Message(
            msg_id=_text(msg_id),
            job_id=decoded.get("job_id", ""),
            task_key=decoded.get("task_key", ""),
            deliveries=deliveries,
        )


def _redis_timeout_error():
    try:
        from redis.exceptions import TimeoutError as RedisTimeout

        return (RedisTimeout, TimeoutError)
    except ImportError:  # pragma: no cover
        return (TimeoutError,)


def now_ms() -> int:
    return int(time.time() * 1000)
