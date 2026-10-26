# Region-processing microservices

This document describes how AI region processing (DeepLIIF, HoVer-Net, EC
cancer, PDAC) and the slide-mask helpers run as independently scaled
services, how a request flows through them, and how to operate and grow the
deployment.

**Current deployment (October 2026):** k3s on the single 4×Tesla T4 host
(`ai-cpathportal`), overlay `deploy/k8s/overlays/k3s-local`. Every model
worker scales to zero when idle and is started by KEDA when a region is
queued for it. AIMViewer reaches the gateway on NodePort 30080 and the
tissue API on NodePort 30800.

## Why

The previous design ran every model inside each of four `extract_areas`
replicas, one per T4, behind an nginx split into a "general" pool (three
GPUs) and a "heavy" pool (one GPU for PDAC). AIMViewer fanned regions out
from a daemon thread inside the web process. That had four limits:

1. **Capacity was fixed at deploy time.** Adding a GPU meant editing nginx,
   systemd units and AIMViewer's pool map by hand.
2. **Every replica loaded every model** (about 19 GB RSS each), so GPU memory
   was spent on models that replica would rarely run.
3. **Job state lived in the web process.** A gunicorn restart lost the
   thread driving a job; the browser spun until its two-hour timeout.
4. **One slow model starved the others.** PDAC had a card to itself while
   three fast models shared three cards, whatever the actual demand.

## Shape

```
                     ┌─────────────────────────────────────────────────────┐
 browser ─► Django ─►│ inference-gateway   stateless, HPA 1..10            │
 (polls)  (aimviewer)│ POST /v1/jobs  GET /v1/jobs/{id}  …/overlay  :30080 │
    │                └──────────────────────────┬──────────────────────────┘
    │                                           │ job + task records,
    │                                           │ one stream per model
    │                                     ┌─────▼─────┐
    │                                     │   Redis   │  inference:tasks:deepliif
    │                                     │ (streams) │  inference:tasks:hovernet
    │                                     └─────┬─────┘  inference:tasks:ec_cancer
    │                                           │        inference:tasks:pdac
    │         ┌──────────────┬──────────────────┼───────────────┬──────────────┐
    │   ┌─────▼─────┐  ┌─────▼─────┐      ┌─────▼─────┐   ┌─────▼─────┐        │
    │   │ worker-   │  │ worker-   │      │ worker-   │   │ worker-   │  KEDA sizes each
    │   │ deepliif  │  │ hovernet  │      │ ec-cancer │   │ pdac      │  deployment from
    │   │ 0..2 pods │  │ 0..2 pods │      │ 0..2 pods │   │ 0..2 pods │  its stream length
    │   └─────┬─────┘  └─────┬─────┘      └─────┬─────┘   └─────┬─────┘
    │         └──────────────┴────────┬─────────┴───────────────┘
    │                    /slides (RO)   /models (RO)   one GPU slot per pod
    │
    └─ mask / tissue helpers (synchronous) ─► tissue-api (SAM3, 1 pod) :30800
```

| Component | Code | Image | Replicas | Scales on |
|---|---|---|---|---|
| Django (`aimviewer`) | OMERO.web | n/a | n/a | n/a |
| Gateway | `gateway/` | `aim-inference-gateway` | 1..10 | CPU (HPA) |
| Redis | `deploy/k8s/base/redis.yaml` | `redis:7.4-alpine` | 1 (AOF) | n/a |
| Model worker, one per model | `worker/` + `models/<model>_model.py` | `aim-inference-worker-<model>` | 0..2 | stream length (KEDA) |
| Tissue API | `app_refactored.py`, region models disabled | `aim-inference-tissue-api` | 1 | fixed (synchronous calls) |

Shared code for the gateway and the workers is `inference_platform/`:
settings, job store, stream broker, model registry, overlay store, catalogue
and metrics.

## Request flow

1. **Submit.** Django validates the regions while it still holds the OMERO
   connection, resolves the slide path, then calls `POST /v1/jobs` with the
   model, slide path, regions (level-0 bbox and polygon), downsample factors,
   hyperparameters and the owner.
2. **Queue.** The gateway writes one *job* record and one *task* record per
   region, then `XADD`s one entry per task onto `inference:tasks:<model>`.
   It answers 202 with the job view immediately.
3. **Scale.** KEDA reads the stream's length every 10 s and sets the model's
   Deployment to `ceil(length / 1)` pods, between its min (0) and max (2).
   One queued region is enough to start a pool that is at zero.
4. **Run.** A worker takes an entry with `XREADGROUP` and marks the task
   `running`. It maps the slide path into the container (`SLIDE_PATH_MAP`),
   reads the region at the nearest pyramid level, rasterises the polygon into
   a mask and calls `model.process()`. Progress fields (patch counters, stage
   label) are written as the model reports them. A lease thread re-claims
   the entry every ~40 s to prove the worker is alive.
5. **Finish.** The overlay PNG goes to the overlay store (Redis by default,
   S3 optional), the task becomes `done`, `failed` or `cancelled`, and only
   then is the entry acknowledged and deleted from the stream.
6. **Poll.** Django's `GET /region_jobs/{id}` proxies `GET /v1/jobs/{id}`.
   The gateway derives the job status, counters and message from the task
   records on every read, so there is no job-level state to keep consistent.
7. **Collect.** The browser fetches each region's overlay, merges them into
   one heatmap per label, uploads it to OMERO and records the AI run. Job
   records expire after six hours; the AI-run history is the durable record.

The slide-mask and tissue-region helpers (`generate_mask`,
`generate_tissue_regions`) are short synchronous calls from Django to the
tissue API and do not use the queue.

### Failure handling

| Failure | What happens |
|---|---|
| Worker pod dies mid-task | The entry stays pending. After `TASK_LEASE_SECONDS` (120 s) without renewal another worker reclaims it. Attempt 2 runs; a second death fails the task with a clear message, and the job completes with that region marked failed. |
| Slide unreadable or outside `SLIDE_ROOTS` | The task fails with the reason; other regions are unaffected. |
| Model raises | The task fails with the exception text, the GPU cache is cleared, and the worker continues. |
| User cancels | Pending tasks are cancelled at once; a running task stops at its next progress report (`InferenceCancelled`). |
| Gateway pod restarts | Nothing is lost: all state is in Redis. |
| Redis restarts | AOF persistence keeps jobs and streams; workers reconnect. |
| Gateway unreachable | Django returns 503 with a readable message; the panel offers retry. |
| User floods the system | 429 after `MAX_ACTIVE_JOBS_PER_OWNER` (20) unfinished jobs. |

Delivery is at-least-once. The task record's terminal status makes a second
delivery of the same entry a no-op.

## Capacity and scaling

**Unit of parallelism is one region.** A job of 40 regions on a model with
four workers finishes in roughly the time of its ten slowest regions, not
the sum of all forty.

**One pod, one GPU slot, one task.** The models were written for exclusive
GPU use, so concurrency comes from replicas, not threads. Every worker
requests `nvidia.com/gpu: 1`.

**GPU time-slicing.** The NVIDIA device plugin advertises each physical T4
as two `nvidia.com/gpu` slots, so the host offers 8 slots. Two pods can
share one card; memory is not partitioned, so the workers' automatic batch
sizing is what keeps a shared card within its 16 GB. On a cluster with
larger or more cards, set `replicas: 1` in the device plugin ConfigMap for
exclusive GPUs.

**Scale to zero.** Every model worker idles at zero replicas, so an idle
model holds no GPU memory. After its queue empties, a pool stays up for its
cooldown, then KEDA removes it.

| Model | Min | Max (k3s-local) | Cooldown | Memory limit |
|---|---|---|---|---|
| DeepLIIF | 0 | 2 | 10 min | 16 Gi |
| HoVer-Net | 0 | 2 | 10 min | 12 Gi |
| EC cancer | 0 | 2 | 10 min | 12 Gi |
| PDAC | 0 | 2 | 15 min | 24 Gi |
| Tissue API | 1 | 1 | n/a | 12 Gi |

The cost is a cold start on the first job after an idle period. Measured on
this host with a warm page cache: a HoVer-Net job on an idle pool finished
in 30 s end to end; a PDAC pod was ready 50 s after the job was queued, and
the region itself took another 60 s. Set `minReplicaCount: 1` in
`deploy/k8s/base/workers/worker-<model>.yaml` for a model that must answer
immediately.

**KEDA trigger.** Each worker has a `redis-streams` ScaledObject in
stream-length mode (`streamLength: "1"`, `activationStreamLength: "0"`).
Workers delete an entry once its task finishes, so the stream length is the
number of regions queued plus running. Two details matter:

* Lag mode looks more natural but cannot wake a pool from zero on a single
  queued region: activation needs lag strictly greater than
  `activationLagCount`, and KEDA rejects `activationLagCount: 0`.
* The Redis address in each trigger is fully qualified
  (`redis.aim-inference.svc.cluster.local:6379`). The KEDA operator runs in
  the `keda` namespace, where the short name `redis` does not resolve.

**Adding GPU capacity** needs no code change. Any of these works:

* Add a node to the GPU pool (label `aim.inference/gpu=true`, taint
  `nvidia.com/gpu`). Pending pods schedule onto it.
* Raise `maxReplicaCount` (`deploy/k8s/overlays/prod-gpu/scaling.yaml`, or
  the patch in `overlays/k3s-local/kustomization.yaml` on this host).
* Let the cluster autoscaler or Karpenter create GPU nodes when pods are
  Pending on an unsatisfied `nvidia.com/gpu` request.
* Mix GPU types: a node pool of larger cards for `worker-pdac` only, via a
  per-model `nodeSelector` patch.

## Operating

### Cluster prerequisites

Both are installed on this host. Pinned copies live in `deploy/k8s/addons/`.

| Add-on | File | Provides |
|---|---|---|
| NVIDIA container toolkit | apt package on every GPU node | the `nvidia` RuntimeClass (k3s detects the runtime on restart) |
| NVIDIA device plugin v0.17.1 | `addons/nvidia-device-plugin.yaml` | `nvidia.com/gpu` with 2-way time-slicing |
| KEDA 2.16.1 | `addons/keda-2.16.1.yaml` | `ScaledObject` CRD, operator, metrics adapter |

```bash
make k8s-addons        # needs cluster-admin
```

### Volumes, configuration and secrets

| Item | Inside the pods | On this host |
|---|---|---|
| Slides (read-only) | `/slides` | hostPath `/data/omero/data` |
| Model weights (read-only) | `/models` | hostPath `/data/models` (`deepliif/`, `hovernet_checkpoints/`, `EC_model/`, `PDAC_GigaPath_1024_FT/`) |
| Hugging Face cache (GigaPath, SAM3) | `/tmp/hf` | hostPath `/data/models/hf-cache`, owned by uid 10001 |
| Model paths and catalogue | `/app/config.json` | ConfigMap `inference-config` from `deploy/k8s/base/config.container.json` |
| Runtime settings | env | ConfigMap `inference-env` (`deploy/k8s/base/configmap.yaml`) |
| `HF_TOKEN`, `GATEWAY_API_TOKEN` | env | Secret `inference-secrets`, created from `deploy/compose.env` by `overlays/k3s-local/create-secret.sh` |

Weights are never baked into images. The committed Secret manifest is an
empty placeholder; `deploy/compose.env` is gitignored.

AIMViewer submits slide paths as OMERO.web sees them. `SLIDE_PATH_MAP`
(`/data/omero/data=/slides`) rewrites that prefix to the pods' mount before
a worker opens the file, and `SLIDE_ROOTS` still confines reads to the mount.

### Build and deploy

```bash
make images                                    # build all seven images with Docker
make k8s-import-images                         # load them into k3s containerd (no registry)
deploy/k8s/overlays/k3s-local/create-secret.sh # namespace + tokens
make k8s-apply-local                           # kubectl apply -k overlays/k3s-local
kubectl -n aim-inference get deploy,scaledobject,pods
```

To roll out a code change to one model, rebuild its image, import it, and
restart its Deployment:

```bash
TARGETS=worker-hovernet docker/build.sh
docker save aimlab/aim-inference-worker-hovernet:dev | sudo k3s ctr images import -
kubectl -n aim-inference rollout restart deploy/worker-hovernet
```

For a multi-node cluster, push to a registry and use the `prod-gpu`
overlay, which adds GPU node selectors, NFS volumes and registry image tags:

```bash
REGISTRY=registry.example.org/aimlab TAG=1.0.0 docker/build.sh --push
kubectl apply -k deploy/k8s/overlays/prod-gpu
```

### Point AIMViewer at it

```yaml
# aimviewer/aimviewer/config.yaml (per environment)
production:
  AI_GATEWAY:
    URL: http://127.0.0.1:30080      # in-cluster: http://inference-gateway.aim-inference.svc
    TOKEN: <GATEWAY_API_TOKEN>
  AI_BACKENDS:
    general: http://127.0.0.1:30800  # tissue-api; in-cluster: http://tissue-api.aim-inference.svc
```

AIMViewer reads these from `aimviewer.settings`, not from
`django.conf.settings`: under OMERO.web the Django settings module is
`omeroweb.settings`, which carries none of the app's `config.yaml` values.
Restart OMERO.web after changing them (`omero web restart`).

### Other ways to run the same services

Do not run two of these at once: each would claim the GPUs.

* **Docker Compose** (`deploy/docker-compose.yml`, `make compose-up`): the
  same images on one GPU host without autoscaling. Each service is pinned to
  a card in `deploy/compose.env`; the gateway is published on 8090 (8080 is
  nginx on this host) and the tissue API on 8010.
* **systemd** (`deploy/systemd/`): the gateway and workers straight from the
  conda environment, one unit per `<model>:<gpu>`.
* **dev-cpu** (`overlays/dev-cpu`, `deploy/docker-compose.dev.yml`): gateway,
  Redis and the stub worker on a machine with no GPU.

`overlays/k3s-local/switch-from-compose.sh` moves the host from Compose to
k3s in one step: add-ons, images, secret, `compose down`, apply, smoke
check, then it repoints `config.yaml` and restarts OMERO.web. If the
shell predates the user's `docker` group membership, it falls back to
`sudo docker`.

**Rollback** to Compose: `make compose-up` in this repo, restore
`aimviewer/config.yaml.bak-pre-k3s`, and `omero web restart`. Scale the k3s
workloads to zero first, or delete the namespace, so the GPUs are free. The
original `extract-areas-api@N` systemd units are disabled but still
installed; `config.yaml.bak-pre-gateway` is the configuration from before
the gateway existed.

### Observability

* `GET /v1/models` on the gateway: catalogue with live replicas, busy
  replicas, queue stats and `scale_to_zero`. The AIMViewer panel shows this
  as a capacity line under the selected model.
* `GET /v1/queues`: depth, lag, pending and replicas per model.
* `kubectl -n aim-inference get scaledobject,hpa`: whether KEDA is Ready and
  Active per model.
* Prometheus: the gateway's `/metrics` (`inference_queue_depth`,
  `inference_jobs_submitted_total`, …) and every worker's `:9100/metrics`
  (`inference_task_seconds`, `inference_tasks_processed_total{status}`,
  `inference_worker_busy`, `inference_model_loaded`). Pods carry
  `prometheus.io/scrape` annotations.
* Logs are JSON (`LOG_FORMAT=json`) with `job_id`, `task_key` and `model`
  fields. A worker's `:9100/status` shows its current task.

### Runbook

| Symptom | Check |
|---|---|
| Panel shows "Request failed with status code 503" | Gateway down or misaddressed. `curl 127.0.0.1:30080/readyz` (fails if Redis is down). Check `AI_GATEWAY.URL` in `config.yaml` and the OMERO.web log for the URL actually used. |
| A model has no pod | Normal when idle: every worker scales to zero. Queue a region and a pod appears within ~10 s. |
| Jobs stay queued and no pod appears | `kubectl -n aim-inference get scaledobject`. If READY is False, read `kubectl describe scaledobject worker-<model>` and the `keda-operator` log; an unresolvable Redis address and invalid trigger metadata both show there. |
| Pod stays Pending | No free GPU slot. `kubectl describe pod` gives the reason; lower another model's max or add capacity. |
| A region "stopped unexpectedly" | The worker crashed twice on it. Run `kubectl logs --previous` on the pod named in the task's `worker` field. Usually an OOM on a huge region: lower `MAX_REGION_PIXELS` or use downscaling. |
| "slide path is outside the permitted roots" | The path AIMViewer sent does not map under `/slides`. Check `SLIDE_PATH_MAP` in `inference-env`. |
| Model missing from the panel | The catalogue comes from `config.container.json`; parameters come from a descriptor that a worker publishes on start (or `python -m worker.main --describe`). |
| Slow first region after idle | Expected cold start. Set `minReplicaCount: 1` for that model if instant response matters more than the idle GPU. |

## API summary (gateway)

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/jobs` | Body: `model_name, slide_path, regions[], downsample, overlay_downsample, hyperparameters, image_id, owner`. Returns 202 `{job}`. |
| `GET` | `/v1/jobs/{id}` | `{job}` with `status`, `regions[]`, counters, `message`, `error`. |
| `POST` | `/v1/jobs/{id}/cancel` | |
| `GET` | `/v1/jobs/{id}/regions/{key}/overlay` | PNG; 409 while the region is not done. |
| `GET` | `/v1/jobs?limit=` | Recent jobs, for operators. |
| `GET` | `/v1/models`, `/v1/models/{name}` | Catalogue with replicas and queue stats. |
| `GET` | `/v1/queues` | Per-model stream stats. |
| `GET` | `/healthz`, `/readyz`, `/metrics` | |

Authentication: `Authorization: Bearer <GATEWAY_API_TOKEN>` when the token
is set, which it is on this host. The API is internal; do not expose it on
the public internet.

## Adding a model

```bash
python scripts/new_model.py tumor_grader --min 0 --max 4
```

That scaffolds, from `worker/templates/`:

| Created | Purpose |
|---|---|
| `models/tumor_grader_model.py` | A `BaseAIModel` subclass with tiling, progress reporting and a `build_model(section, device)` factory; three TODOs to fill. |
| `worker/requirements/tumor_grader.txt` | This model's extra pip dependencies. |
| `docker/Dockerfile` stage `worker-tumor-grader`, `docker/build.sh` target | Its image. |
| `deploy/k8s/base/workers/worker-tumor-grader.yaml` and a kustomization entry | Deployment plus KEDA `ScaledObject` on stream `inference:tasks:tumor_grader`, with the same trigger settings as the existing workers. |
| `models.tumor_grader` in `config.json` and `config.container.json` | `weights_path`, pool, scale-to-zero. |

`worker/loaders.py` finds any model that exposes `build_model()` without
being edited. No gateway or AIMViewer change is needed either: the model
appears in the panel once the catalogue lists it and a worker has published
its descriptor. Then build, import and apply as in *Build and deploy*.
`tests/test_template.py` runs the unmodified template through the worker
loop, so the scaffold is known to work before any model code is written.

## Testing

```bash
make test        # 23 tests: store/broker, gateway API, worker loop, path mapping, model template
```

The tests use an in-memory Redis and need no GPU. The stub model
(`worker/stub_model.py`) runs anywhere, reports fake patch progress and
draws the region outline; `deploy/docker-compose.dev.yml` and
`overlays/dev-cpu` use it to prove the queue, probes and scaling on a
machine without a GPU.
