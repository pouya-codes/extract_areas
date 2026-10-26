# Region-processing microservices

This document describes how AI region processing (DeepLIIF, HoVer-Net, EC
cancer, PDAC) is split into independently scaled services, how a request
flows through them, and how to operate and grow the deployment.

## Why

The previous design ran every model inside each of four `extract_areas`
replicas, one per Tesla T4, behind an nginx split into a "general" pool
(three GPUs) and a "heavy" pool (one GPU for PDAC). AIMViewer fanned regions
out from a daemon thread inside the web process. That had four limits:

1. **Capacity was fixed at deploy time.** Adding a GPU meant editing nginx,
   systemd units and AIMViewer's pool map by hand.
2. **Every replica loaded every model** (about 19 GB RSS each), so GPU memory
   was spent on models that replica would rarely run.
3. **Job state lived in the web process.** A gunicorn restart lost the
   thread driving a job; the browser spun until its two-hour timeout.
4. **One slow model starved the others.** A PDAC region on the heavy pool
   queued behind nothing, while three fast models shared three cards.

## Shape

```
                       ┌──────────────────────────────────────────────────────┐
  browser ──► Django ──►  inference gateway  (stateless, 2..N replicas, HPA)  │
  (poll)     (aimviewer) │  POST /v1/jobs  GET /v1/jobs/{id}  …/overlay        │
                         └───────────────┬──────────────────────────────────────┘
                                         │ job + task records, per-model streams
                                   ┌─────▼─────┐
                                   │   Redis   │   inference:tasks:deepliif
                                   │ (streams) │   inference:tasks:hovernet
                                   └─────┬─────┘   inference:tasks:ec_cancer
                                         │         inference:tasks:pdac
        ┌───────────────┬────────────────┼────────────────┬──────────────────┐
  ┌─────▼─────┐   ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐           │
  │ worker-   │   │ worker-   │    │ worker-   │    │ worker-   │   KEDA sizes each
  │ deepliif  │   │ hovernet  │    │ ec-cancer │    │ pdac      │   deployment from
  │ 1 GPU/pod │   │ 1 GPU/pod │    │ 1 GPU/pod │    │ 1 GPU/pod │   its stream's lag
  │ min 1     │   │ min 1     │    │ min 1     │    │ min 0     │
  └─────┬─────┘   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
        └───────────────┴────────┬───────┴────────────────┘
                        /slides (RO, shared)   /models (RO, shared)
```

| Component | Lives in | Image | Scales on |
|---|---|---|---|
| Django (`aimviewer`) | OMERO.web | unchanged | n/a |
| Gateway | `gateway/` | `aim-inference-gateway` | CPU (HPA) |
| Redis | managed or `deploy/k8s/base/redis.yaml` | `redis:7` | n/a |
| Worker, one per model | `worker/` + `models/<model>_model.py` | `aim-inference-worker-<model>` | queue lag (KEDA) |

Shared code for all of them is `inference_platform/` (settings, job store,
stream broker, model registry, overlay store, metrics).

## Request flow

1. **Submit.** Django validates the regions while it still holds the OMERO
   connection, resolves the slide path, then `POST /v1/jobs` with the model,
   slide path, regions (level-0 bbox + polygon), downsample factors,
   hyperparameters and the owner.
2. **Queue.** The gateway writes one *job* record and one *task* record per
   region, then `XADD`s one entry per task onto `inference:tasks:<model>`.
   It answers 202 with the job view immediately.
3. **Scale.** KEDA reads the stream's consumer-group lag every 10 s and sets
   the model's deployment to `ceil(lag / 1)` pods, between its min and max.
   With no GPU free, pods stay Pending until the cluster autoscaler adds a
   node (or an operator does).
4. **Run.** A worker takes an entry with `XREADGROUP`, marks the task
   `running`, reads the region from the slide at the nearest pyramid level,
   rasterises the polygon into a mask, calls `model.process()`, and writes
   progress fields (patch counters, stage label) as the model reports them.
   A lease thread re-claims the entry every ~40 s to prove the worker is
   alive.
5. **Finish.** The overlay PNG goes to the overlay store (Redis by default,
   S3 optional), the task becomes `done`/`failed`/`cancelled`, and only then
   is the entry `XACK`ed and deleted.
6. **Poll.** Django's `GET /region_jobs/{id}` proxies `GET /v1/jobs/{id}`;
   the gateway derives the job status, counters and message from the task
   records on every read, so there is no job-level state to keep consistent.
7. **Collect.** The browser fetches each region's overlay, merges them into
   one heatmap per label, uploads it to OMERO and records the AI run. Job
   records expire six hours later; the AI-run history is the durable record.

### Failure handling

| Failure | What happens |
|---|---|
| Worker pod dies mid-task | Entry stays pending. After `TASK_LEASE_SECONDS` (120 s) without renewal another worker reclaims it. Attempt 2 runs; a second death fails the task with a clear message. The job completes with that region marked failed. |
| Slide unreadable / outside `SLIDE_ROOTS` | Task fails with the reason; other regions unaffected. |
| Model raises | Task fails with the exception text; GPU cache is cleared; worker continues. |
| User cancels | Pending tasks are cancelled at once; a running task stops at its next progress report (`InferenceCancelled`). |
| Gateway pod restarts | Nothing is lost: all state is in Redis. |
| Redis restarts | AOF persistence keeps jobs and streams; workers reconnect. |
| Gateway unreachable | Django returns 503 with a human message; the panel offers retry. |
| User floods the system | 429 after `MAX_ACTIVE_JOBS_PER_OWNER` (20) unfinished jobs. |

Delivery is at-least-once; the task record's terminal status makes the
second delivery of the same entry a no-op.

## Capacity and scaling

**Unit of parallelism = one region.** A job of 40 regions on a model with
four workers finishes in roughly the time of its ten slowest regions, not
the sum of all forty.

**One pod, one GPU, one task.** The models were written for exclusive GPU
use; concurrency comes from replicas, not threads. Every worker Deployment
requests `nvidia.com/gpu: 1`.

**KEDA trigger.** `redis-streams` with `lagCount: "1"`: target one pod per
queued region, so the backlog is what drives scale-out. `cooldownPeriod`
(10–15 min) keeps a loaded model around between bursts because loading
takes 1–2 min. PDAC idles at zero: GigaPath is ~10 GB of VRAM and is used
rarely; its first job of the day waits for a pod to start.

**Adding GPU capacity later** is the point of the design. Any of these
works without code changes:

* Add a node to the GPU pool (label `aim.inference/gpu=true`, taint
  `nvidia.com/gpu`). Pending pods schedule onto it.
* Raise `maxReplicaCount` in `deploy/k8s/overlays/prod-gpu/scaling.yaml`.
* Let the cluster autoscaler / Karpenter create GPU nodes when pods are
  Pending with an unsatisfied `nvidia.com/gpu` request.
* Mix GPU types: a node pool of larger cards for `worker-pdac` only, via a
  per-model `nodeSelector` patch.

**Sizing on the current 4×T4 host** (`overlays/k3s-local`): each model's
`maxReplicaCount` is 2 and the sum of minimums is 3, leaving one card for
whichever queue is deepest. Replace the limits once more cards exist.

**Throughput ceiling without more GPUs.** Time-slicing (NVIDIA device plugin
`sharing.timeSlicing`) lets two small-model pods share one T4; it is a
cluster-level setting and needs no change here.

## Operating

### Prerequisites in the cluster

```bash
# NVIDIA device plugin (exposes nvidia.com/gpu); needs the nvidia container runtime on GPU nodes
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
# KEDA (ScaledObject CRD + metrics adapter)
kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.16.1/keda-2.16.1.yaml
```

### Deploy

```bash
make images REGISTRY=registry.example.org/aimlab TAG=1.0.0 && docker/build.sh --push
kubectl apply -k deploy/k8s/overlays/prod-gpu          # or k3s-local / dev-cpu
kubectl -n aim-inference get deploy,scaledobject,pods
```

Secrets (`HF_TOKEN` for GigaPath, `GATEWAY_API_TOKEN`) go in the
`inference-secrets` Secret; the committed one is an empty placeholder.

Model weights are **not** in the images. Mount the directory that contains
`deepliif/`, `hovernet_checkpoints/`, `EC_model/` and
`PDAC_GigaPath_1024_FT/` at `/models` (PVC `model-weights`), and the OMERO
data directory at `/slides` (PVC `slides`). Paths inside the containers are
set in `deploy/k8s/base/config.container.json`.

### Point AIMViewer at it

```yaml
# aimviewer/config.yaml
production:
  AI_GATEWAY:
    URL: http://inference-gateway.aim-inference.svc   # or the NodePort / ingress
    TOKEN: <GATEWAY_API_TOKEN>
```

### Without Kubernetes

* `deploy/docker-compose.yml` runs the same images on one GPU host.
* `deploy/systemd/` runs the gateway and workers straight from the conda
  environment, one unit per `<model>:<gpu>`, for the current bare-metal
  server. The same code, the same queue, no containers.

### Observability

* `GET /v1/queues` on the gateway: depth / lag / pending / replicas per model.
* `GET /v1/models`: catalogue with live replica counts.
* Prometheus: gateway `/metrics` (`inference_queue_lag`, `inference_jobs_submitted_total`, …);
  every worker `:9100/metrics` (`inference_task_seconds`, `inference_tasks_processed_total{status}`,
  `inference_worker_busy`, `inference_model_loaded`). Pods carry
  `prometheus.io/scrape` annotations.
* Logs are JSON (`LOG_FORMAT=json`) with `job_id` / `task_key` / `model` fields.
* Worker `:9100/status` shows the current task for one pod.

### Runbook

| Symptom | Check |
|---|---|
| Panel says the service is unreachable | `kubectl -n aim-inference get pods -l app.kubernetes.io/name=inference-gateway`; `/readyz` fails if Redis is down. |
| Jobs stay queued | `GET /v1/queues`: lag > 0 and replicas = 0 → pods Pending; `kubectl describe pod` shows the GPU/affinity reason. |
| A region "stopped unexpectedly" | The worker crashed twice on it; `kubectl logs --previous` on the pod named in the task's `worker` field. Usually OOM on a huge region: lower `MAX_REGION_PIXELS` or ask for downscaling. |
| Models missing from the panel | The catalogue comes from `config.container.json`; a model also needs a published descriptor, which any worker start (or `python -m worker.main --describe`) writes. |
| Slow first region after idle | Expected for scale-to-zero models (PDAC). Set `minReplicaCount: 1` if that matters more than the idle GPU. |

## API summary (gateway)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/jobs` | body: `model_name, slide_path, regions[], downsample, overlay_downsample, hyperparameters, image_id, owner` → 202 `{job}` |
| `GET` | `/v1/jobs/{id}` | `{job}` with `status`, `regions[]`, counters, `message`, `error` |
| `POST` | `/v1/jobs/{id}/cancel` | |
| `GET` | `/v1/jobs/{id}/regions/{key}/overlay` | PNG; 409 while not done |
| `GET` | `/v1/jobs?limit=` | recent jobs (operators) |
| `GET` | `/v1/models`, `/v1/models/{name}` | catalogue + replicas + queue |
| `GET` | `/v1/queues` | per-model stream stats |
| `GET` | `/healthz`, `/readyz`, `/metrics` | |

Authentication: `Authorization: Bearer <GATEWAY_API_TOKEN>` when the token is
set. The API is internal; do not expose it on the public internet.

## Adding a model

1. Implement `models/<name>_model.py` on `BaseAIModel` (see
   `ADDING_NEW_AI_MODELS.md`); call `progress_report.report_progress()` from
   batch loops so progress and cancellation work.
2. Add its config under `models.<name>` in `config.json` and
   `deploy/k8s/base/config.container.json`.
3. Add a loader branch in `worker/loaders.py` and a requirements file under
   `worker/requirements/`.
4. Add a `worker-<name>` stage to `docker/Dockerfile`, a Deployment +
   `ScaledObject` under `deploy/k8s/base/workers/`, and the entry in
   `docker/build.sh`'s `TARGETS`.

No gateway or AIMViewer change is needed: the model appears in the panel as
soon as the catalogue lists it and a worker has published its descriptor.

## Testing

```bash
make test        # 19 tests: store/broker, gateway API, worker loop (in-memory Redis, stub model)
```

The stub model (`worker/stub_model.py`) runs anywhere, reports fake patch
progress and draws the region outline; `deploy/docker-compose.dev.yml` and
`overlays/dev-cpu` use it to prove the queue, probes and scaling on a
machine with no GPU.
