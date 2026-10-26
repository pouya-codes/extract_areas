"""Prometheus metrics shared by the gateway and the workers."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Gateway ------------------------------------------------------------- #
JOBS_SUBMITTED = Counter(
    "inference_jobs_submitted_total", "Jobs accepted by the gateway", ["model"]
)
TASKS_SUBMITTED = Counter(
    "inference_tasks_submitted_total", "Regions (tasks) accepted by the gateway", ["model"]
)
JOBS_REJECTED = Counter(
    "inference_jobs_rejected_total", "Jobs the gateway refused", ["reason"]
)
QUEUE_DEPTH = Gauge("inference_queue_depth", "Entries in the model's task stream", ["model"])
QUEUE_LAG = Gauge("inference_queue_lag", "Tasks not yet delivered to any worker", ["model"])
QUEUE_PENDING = Gauge("inference_queue_pending", "Tasks delivered but not acknowledged", ["model"])
MODEL_REPLICAS = Gauge("inference_model_replicas", "Live workers serving the model", ["model"])

# --- Worker -------------------------------------------------------------- #
TASKS_PROCESSED = Counter(
    "inference_tasks_processed_total", "Tasks finished by this worker", ["model", "status"]
)
TASK_SECONDS = Histogram(
    "inference_task_seconds",
    "Wall time per task including slide read and overlay encoding",
    ["model"],
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600),
)
MODEL_SECONDS = Histogram(
    "inference_model_seconds",
    "Time spent inside the model's process() call",
    ["model"],
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600),
)
REGION_PIXELS = Histogram(
    "inference_region_pixels",
    "Pixels in the region actually handed to the model",
    ["model"],
    buckets=(1e5, 5e5, 1e6, 4e6, 1.6e7, 6.4e7, 2.56e8),
)
WORKER_BUSY = Gauge("inference_worker_busy", "1 while a task is running", ["model"])
MODEL_LOADED = Gauge("inference_model_loaded", "1 once the model is ready", ["model"])
MODEL_LOAD_SECONDS = Gauge("inference_model_load_seconds", "How long loading took", ["model"])
