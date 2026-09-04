# How to Add a New AI Processing Model

Guide for coworkers adding models to **extract_areas** so they appear in Aimviewer’s
**AI Region Process** panel. Covers architecture, `BaseAIModel` contracts, HTTP APIs,
config/pools, and end-to-end examples.

| | |
|---|---|
| **Repos** | `/home/ubuntu/code/extract_areas` |
| **Runtime** | conda env `extract_areas` |
| **App entry** | `app_refactored.py` (uvicorn via `deploy/run_replica.sh`) |
| **Config** | `config.json` (+ Aimviewer `AI_BACKENDS` in `config.yaml`) |

You usually **do not** need Vue changes: models and hyperparameters are discovered
via `/models/list` and `/models/{name}/hyperparameters`. Only add a pool badge /
routing entry when the model belongs on a non-default GPU pool.

---

## Architecture

```
Annotator UI (MenuRegionProcess)
    → Aimviewer Django Redis batch jobs  (browser polls Aimviewer; Cloudflare-safe)
        → POST /inference/jobs + short poll  (per region)
            → backend_url_for_model(name)
                ├─ :8000 general  → GPUs 0–2 (:8001–8003)
                │     deepliif, hovernet, ec_cancer, SAM3 mask
                └─ :8005 heavy    → GPU 3 (:8004)
                      pdac
```

- **Async GPU work** lives in extract_areas (`/inference/jobs`).
- Aimviewer keeps its own Redis **batch** job for multi-region UX.
- Sync `POST /process_region_from_slide` remains for debug/smoke only.
- Upload-based `/process_region` and `/process_region_annotation` return **410 Gone**.

---

## Contract map

| Layer | Contract you implement | Who calls it |
|-------|------------------------|--------------|
| **A.** `BaseAIModel.process` | Python method | All process paths |
| **B.** HTTP inference APIs | JSON `SlideRegionRequest` | Aimviewer / curl |
| **C.** Discovery APIs | GET | Aimviewer model picker |
| **D.** Tissue / mask APIs | multipart | Auto region tools (general pool only) |

Most authors only implement **A**. Aimviewer’s live path uses **B** (async jobs) + **C**.

---

## A. Model contract — `BaseAIModel`

Subclass `models/base_model.py`. Reference template: `models/example_model.py`.

### Required methods

| Method | Purpose |
|--------|---------|
| `initialize(config)` | Load weights once at replica boot |
| `process(image, mask=None, annotation_points=None, hyperparameters=None)` | Run inference |
| `get_hyperparameters_schema()` | UI/API schema for tunable params |

Optional: set `self.accepts_mpp = True` if the model should receive slide `mpp` /
`downsample` from the from-slide path.

### `process` inputs

| Arg | Type | Notes |
|-----|------|--------|
| `image` | `PIL.Image` RGB | Region pixels (already read at requested downsample) |
| `mask` | `PIL.Image` L or `None` | White = keep; built from polygon when present |
| `annotation_points` | list of `(x,y)` or `None` | From-slide path passes `None` (mask already built) |
| `hyperparameters` | `dict` or `None` | Overrides merged over model defaults |

### `process` return value

**Success:**

```python
{
    "success": True,
    "processed_image": <PIL.Image>,   # overlay / visualization (required for UI)
    "scores": { ... },                # JSON-serializable metrics
    "str_result": "12 cells · 3 pos / 9 neg · 25% positive",  # short UI string
    "metadata": { ... },              # optional
}
```

**Failure:**

```python
{
    "success": False,
    "error": "human-readable reason",
    "str_result": "Error: ...",       # optional but useful
}
```

Do **not** put huge base64 in `scores`/`metadata`. Overlays travel as PNG separately.

### Hyperparameter schema

Returned by `get_hyperparameters_schema()` and exposed at
`GET /models/{name}/hyperparameters`. The annotator builds form controls from this.

```python
{
    "threshold": {
        "type": "float",          # int | float | bool | string | choice
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "description": "Detection threshold",
        "required": False
    },
    "model_variant": {
        "type": "choice",
        "default": "pannuke",
        "choices": ["pannuke", "consep"],
        "description": "Checkpoint variant"
    }
}
```

### Minimal example model

```python
# models/toy_model.py
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
from models.base_model import BaseAIModel


class ToyModel(BaseAIModel):
    def __init__(self):
        super().__init__(model_name="toy", model_version="0.1.0")
        self.description = "Demo overlay model"
        self.requires_mask = False
        self.hyperparameters = {"threshold": 0.5}

    def initialize(self, config: Dict[str, Any]) -> None:
        self.device = config.get("device", "cpu")
        self._is_initialized = True

    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ok, err = self.validate_input(image, mask)
        if not ok:
            return {"success": False, "error": err}

        params = {**self.hyperparameters, **(hyperparameters or {})}
        out = image.copy()
        draw = ImageDraw.Draw(out, "RGBA")
        w, h = out.size
        draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4],
                       fill=(255, 0, 0, 64), outline=(255, 0, 0, 255), width=3)
        if mask is not None:
            out = self.postprocess_output(out, mask)

        return {
            "success": True,
            "processed_image": out,
            "scores": {"threshold_used": params["threshold"]},
            "str_result": f"toy overlay @ threshold={params['threshold']}",
        }

    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        return {
            "threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Demo threshold",
            }
        }
```

---

## B. HTTP inference APIs

Base URLs (nginx):

| Pool | URL | Replicas |
|------|-----|----------|
| general | `http://127.0.0.1:8000` | `:8001`–`:8003` |
| heavy | `http://127.0.0.1:8005` | `:8004` |

Aimviewer picks the base with `backend_url_for_model(model_name)`.

### Shared request body — `SlideRegionRequest`

Used by both `POST /process_region_from_slide` and `POST /inference/jobs`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `slide_path` | string | yes | Absolute path under configured `slide_roots` (symlink targets must also resolve under a root) |
| `roi_bbox` | `[x, y, w, h]` | yes | Level-0 slide coordinates |
| `polygon` | `[[x,y], ...]` | no | Level-0 polygon; fills the binary mask |
| `downsample` | float | no (default `1.0`) | Read region at this downsample (≥1) |
| `overlay_downsample` | float | no | Output overlay scale; defaults to processed downsample |
| `model_name` | string | no (default `deepliif`) | Must be loaded on that pool’s replicas |
| `hyperparameters` | object | no | Model-specific overrides |
| `region_id` | string | no | Echoed back for caller correlation |

Example:

```json
{
  "slide_path": "/data/omero/data/ManagedRepository/.../slide.svs",
  "roi_bbox": [5000, 5000, 2048, 2048],
  "polygon": [[5000, 5000], [7048, 5000], [7048, 7048], [5000, 7048]],
  "downsample": 4.0,
  "overlay_downsample": 8.0,
  "model_name": "deepliif",
  "hyperparameters": { "tile_size": 0 },
  "region_id": "42"
}
```

---

### `POST /process_region_from_slide` (sync debug)

Blocks until inference finishes. Prefer `/inference/jobs` in production.

**Success `200`:**

```json
{
  "status": "success",
  "processed_image_base64": "<PNG base64>",
  "overlay_bbox": [5000, 5000, 2048, 2048],
  "overlay_width": 256,
  "overlay_height": 256,
  "overlay_downsample": 8.0,
  "processed_downsample": 4.0,
  "slide_level_used": 2,
  "score": { "num_total": 0, "num_pos": 0, "num_neg": 0 },
  "str_result": "0 cells · 0 pos / 0 neg · 0% positive",
  "region_id": "42",
  "model_used": "deepliif",
  "model_version": "1.0.0",
  "metadata": {},
  "timing": {
    "queue_seconds": 0.0,
    "model_seconds": 0.16,
    "region_pixels": 262144
  }
}
```

**Error `4xx/5xx`:**

```json
{ "status": "error", "message": "Model 'pdac' not found on this replica (pool=general). Available: [...]" }
```

```bash
curl -sS http://127.0.0.1:8000/process_region_from_slide \
  -H 'Content-Type: application/json' \
  -d @payload.json | jq '{status, str_result, timing}'
```

---

### Async jobs (canonical for Aimviewer)

#### `POST /inference/jobs`

Enqueues work; returns immediately.

**Success `200`:**

```json
{
  "job_id": "d4cadc2a-6910-4cd6-a2b0-86212b80bec9",
  "status": "queued",
  "pool": "general"
}
```

**Error `400`** if the model is not loaded on that pool:

```json
{
  "status": "error",
  "message": "Model 'pdac' not loaded on pool=general. Available: ['deepliif', 'ec_cancer', 'hovernet']"
}
```

#### `GET /inference/jobs/{job_id}`

Poll every ~1s. **No** huge base64 in this body.

Statuses: `queued` | `running` | `succeeded` | `failed` | `cancelled`

**Succeeded example:**

```json
{
  "job_id": "d4cadc2a-...",
  "status": "succeeded",
  "pool": "general",
  "model_used": "deepliif",
  "model_version": "1.0.0",
  "score": { "num_total": 0 },
  "str_result": "0 cells · 0 pos / 0 neg · 0% positive",
  "timing": { "queue_seconds": 0.0, "model_seconds": 0.16, "region_pixels": 262144 },
  "overlay_bbox": [5000, 5000, 2048, 2048],
  "overlay_width": 256,
  "overlay_height": 256,
  "has_overlay": true,
  "region_id": "42",
  "error": null
}
```

**Failed example:**

```json
{
  "job_id": "8b842195-...",
  "status": "failed",
  "pool": "heavy",
  "error": "no tumor patches passed stage-1 filter",
  "has_overlay": false
}
```

#### `GET /inference/jobs/{job_id}/overlay`

Raw **PNG bytes** (`Content-Type: image/png`) when `has_overlay` is true.
`404` / `409` if missing or job not succeeded.

#### `POST /inference/jobs/{job_id}/cancel`

Best-effort. Queued jobs become `cancelled`; in-flight GPU work finishes.

```json
{ "job_id": "...", "status": "cancelled", "cancel_requested": true }
```

#### Full async curl flow

```bash
BASE=http://127.0.0.1:8000   # or :8005 for pdac

JOB=$(curl -sS "$BASE/inference/jobs" -H 'Content-Type: application/json' -d @payload.json)
echo "$JOB"
ID=$(echo "$JOB" | jq -r .job_id)

while true; do
  ST=$(curl -sS "$BASE/inference/jobs/$ID")
  echo "$ST" | jq '{status, str_result, error, has_overlay}'
  case $(echo "$ST" | jq -r .status) in
    succeeded|failed|cancelled) break ;;
  esac
  sleep 1
done

curl -sS "$BASE/inference/jobs/$ID/overlay" -o overlay.png
```

---

### Deprecated upload APIs (410)

| Endpoint | Status |
|----------|--------|
| `POST /process_region` | **410 Gone** |
| `POST /process_region_annotation` | **410 Gone** |

Response body:

```json
{
  "status": "error",
  "message": "Upload-based process APIs are retired. Use POST /process_region_from_slide (sync debug) or POST /inference/jobs (async).",
  "canonical": ["/process_region_from_slide", "/inference/jobs"]
}
```

---

## C. Discovery & health

### `GET /health`

```json
{
  "status": "ok",
  "port": "8001",
  "pool": "general",
  "gpu": {
    "name": "Tesla T4",
    "physical_index": "0",
    "memory_allocated_mb": 2799.2,
    "memory_reserved_mb": 2832.0
  },
  "mask_generator": true,
  "mask_generator_backend": "sam3",
  "loaded_models": ["deepliif", "ec_cancer", "hovernet"]
}
```

Heavy replica example: `pool=heavy`, `loaded_models=["pdac"]`, `mask_generator=false`.

### `GET /models/list`

```json
{
  "registered_models": ["deepliif", "ec_cancer", "hovernet"],
  "loaded_models": ["deepliif", "ec_cancer", "hovernet"]
}
```

### `GET /models/{name}/info`

Includes `pool` (from config):

```json
{
  "name": "deepliif",
  "version": "1.0.0",
  "description": "...",
  "requires_mask": false,
  "supported_formats": ["PNG", "JPEG", "JPG"],
  "hyperparameters": { "...": "..." },
  "initialized": true,
  "pool": "general"
}
```

### `GET /models/{name}/hyperparameters`

```json
{
  "model": "deepliif",
  "hyperparameters": {
    "tile_size": { "type": "int", "default": 0, "min": 0, "max": 1024, "description": "..." }
  }
}
```

Aimviewer `GET /aimviewer/.../models/available` queries **both** backends and merges
by name, attaching `pool` and `backend` for the UI.

---

## D. Tissue / mask APIs (general pool only)

These run only where the mask generator is enabled (`AI_REPLICA_POOL=general`).
Aimviewer always calls the **general** URL (`:8000`).

| Endpoint | Input | Output |
|----------|--------|--------|
| `POST /generate_mask` | multipart PNG `file`, form `is_tma` | `{ status, mask_base64 }` |
| `POST /generate_tissue_regions` | PNG + `slide_width`, `slide_height`, `is_tma` | regions + optional mask |
| `POST /extract_regions` | mask PNG + slide size | polygon regions |
| `POST /auto_name_regions` | JSON regions | named regions |

Example:

```bash
curl -sS http://127.0.0.1:8000/generate_mask \
  -F 'file=@thumb.png;type=image/png' \
  -F 'is_tma=false' | jq '{status}'
```

---

## Config & pools

### `config.json` (extract_areas)

```json
{
  "slide_roots": ["/data/omero/data"],
  "max_region_pixels": 64000000,
  "redis": { "host": "localhost", "port": 6379, "db": 0 },
  "mask_generator": {
    "enabled": true,
    "backend": "sam3",
    "sam3_model_id": "facebook/sam3"
  },
  "pools": {
    "general": { "gpus": [0, 1, 2], "base_port": 8001 },
    "heavy": { "gpus": [3], "base_port": 8004 }
  },
  "models": {
    "toy": {
      "enabled": true,
      "pool": "general",
      "lazy_load": false,
      "path": "models/toy/"
    },
    "pdac": {
      "enabled": true,
      "pool": "heavy",
      "lazy_load": true,
      "package_root": "/data/models/PDAC_GigaPath_1024_FT"
    }
  }
}
```

Rules:

- Only models whose `pool` matches `AI_REPLICA_POOL` load on a replica.
- Put VRAM-heavy foundation models on **`heavy`** (GPU 3).
- Mask generator loads only on **`general`**.
- `validate_model_paths()` fails fast when enabled weights are missing (for models wired in `config.py`).

### Aimviewer `config.yaml`

```yaml
AI_BACKENDS:
  general: http://127.0.0.1:8000
  heavy: http://127.0.0.1:8005
  model_pools:
    deepliif: general
    hovernet: general
    ec_cancer: general
    pdac: heavy
    toy: general          # ← add your model here
```

---

## Checklist: register a new model

1. **Implement** `models/<name>_model.py` subclassing `BaseAIModel`.
2. **Weights / deps** under `models/` or `/data/models/...`; add pip deps to
   `requirements-linux-gpu.txt` if needed.
3. **`config.json`** — add under `models` with `enabled`, `pool`, paths.
4. **`config.py`** — extend `validate_model_paths()` if the model has custom path keys.
5. **`app_refactored.py` → `initialize_models()`** — register class + `load_model` when
   `model_enabled_for_pool(...)` is true (copy an existing block).
6. **Aimviewer** — add `model_pools.<name>` in `AI_BACKENDS` (both `production` and
   `development` if you use both).
7. **Restart** replicas; reload nginx only if ports changed; reload OMERO.web so
   Django picks up `AI_BACKENDS`.

```bash
sudo systemctl restart extract-areas-api.target
# after Aimviewer config change:
cd /home/ubuntu/prog/omero/OMERO.server && \
  /home/ubuntu/prog/omero/venv_server/bin/omero web restart
```

8. **Verify**

```bash
curl -sS http://127.0.0.1:8000/health | jq '{pool, loaded_models}'
curl -sS http://127.0.0.1:8005/health | jq '{pool, loaded_models}'
# Expect GPU0–2: no pdac; GPU3: only pdac (when testing that split)
```

### Example `initialize_models` snippet

```python
mcfg = models_cfg.get("toy") or {}
if model_enabled_for_pool("toy", mcfg, _REPLICA_POOL):
    from models.toy_model import ToyModel
    model_registry.register_model_class("toy", ToyModel)
    model_registry.load_model("toy", {"device": mcfg.get("device", "cuda")})
    print("✓ Loaded toy model")
```

---

## Smoke test

```bash
python deploy/smoke_test.py \
  --url http://127.0.0.1:8000 \
  --heavy-url http://127.0.0.1:8005 \
  --slide /data/omero/data/.../slide.svs \
  --async-jobs
```

Without `--slide`, health + **410** checks still run. Prefer a real file under
`slide_roots` (not a ManagedRepository symlink whose target is outside the root).

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Model missing from UI | Not loaded on any pool, or Aimviewer `model_pools` / web not restarted |
| `Model 'x' not loaded on pool=general` | Model is on `heavy` — Aimviewer must route to `:8005` |
| `slide path is outside the permitted roots` | Symlink target outside `slide_roots`; use a real file under `/data/omero/data` |
| OOM / CUDA errors on GPU 3 | Don’t put SAM3/DeepLIIF/HoVer-Net on `heavy`; keep PDAC alone |
| Job stuck `queued` | Redis down, or worker crashed — check `journalctl -u extract-areas-api@*` |
| `ModuleNotFoundError: redis` | `pip install redis` in the `extract_areas` conda env |

---

## Reference files

| Path | Role |
|------|------|
| `models/base_model.py` | Abstract contract |
| `models/example_model.py` | Template |
| `models/deepliif_model.py` | Production example (`str_result`, schema) |
| `models/pdac_model.py` | Heavy / lazy-load example |
| `inference_jobs.py` | Redis queue + worker |
| `app_refactored.py` | Endpoints + `initialize_models` |
| `config.json` / `config.py` | Structured config + validation |
| `deploy/run_replica.sh` | Sets `AI_REPLICA_POOL` |
| `deploy/nginx-extract-areas-api.conf` | `:8000` / `:8005` upstreams |
| `aimviewer/api/ai.py` | `backend_url_for_model`, merged catalog |
| `aimviewer/services/region_process.py` | Submit + poll inference jobs |
