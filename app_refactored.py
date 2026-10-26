# app_refactored.py
"""
Refactored FastAPI application using class-based AI models.

This version demonstrates how to use the model registry system.
Replace your current app.py with this code, or merge the changes.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import json
import threading
import time
import numpy as np
import tempfile
import cv2
from io import BytesIO
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
from pathlib import Path

# Import model system. Concrete model classes are imported inside
# initialize_models() so a container that only serves the mask generator
# (tissue-api) does not need every model's dependencies installed.
from models.model_registry import model_registry

# Import existing utilities (keep your existing imports)
from mask_generator import MaskGenerator
from config import abs_path, config, model_enabled_for_pool, replica_pool, validate_model_paths
from slide_region import (
    SlideAccessError,
    polygon_mask,
    read_region,
    resolve_slide_path,
)
from src.utils import extract_regions_from_mask
import inference_jobs

# Each replica owns one GPU. Endpoints declared with `def` rather than `async
# def` run in FastAPI's threadpool, so without this two requests landing on the
# same replica would enter the models concurrently and contend for that card.
# Serialising inference keeps GPU memory bounded and matches the one-at-a-time
# behaviour the models were written for; concurrency comes from the four
# replicas behind the load balancer, not from threads inside one of them.
_inference_lock = threading.Lock()
_REPLICA_POOL = replica_pool()


def get_absolute_path(relative_path):
    """Helper function to get absolute path from relative path."""
    resolved = abs_path(relative_path)
    if resolved:
        return resolved
    return str(Path(relative_path).absolute())


def init_mask_generator():
    """Initialize mask generator (SAM / SAM2 / SAM3). Only on the general pool."""
    mg = config.get("mask_generator") or {}
    backend = mg.get("backend", "sam3")
    if backend == "sam":
        return MaskGenerator(model_path=mg.get("sam_model_path", "TBD"), model_name="sam")
    if backend == "sam2":
        return MaskGenerator(
            model_path=get_absolute_path(mg["sam2_model_path"]),
            model_config=mg["sam2_config_path"],
            model_name="sam2",
        )
    if backend == "sam3":
        return MaskGenerator(
            model_name="sam3",
            hf_model_id=mg.get("sam3_model_id", "facebook/sam3"),
            points_per_batch=int(mg.get("sam3_points_per_batch", 64)),
            crop_overlap_ratio=float(mg.get("sam3_crop_overlap_ratio", 0.4)),
        )
    raise ValueError(f"Unknown mask generator backend: {backend}")


# Initialize FastAPI app
app = FastAPI()

# Mask generator: general pool only (heavy GPUs keep VRAM for PDAC).
_mg_cfg = config.get("mask_generator") or {}
if _mg_cfg.get("enabled") and _REPLICA_POOL == "general":
    mask_gen = init_mask_generator()
else:
    mask_gen = None
    if _mg_cfg.get("enabled") and _REPLICA_POOL != "general":
        print(f"↷ Mask generator skipped on pool={_REPLICA_POOL}")

# ============================================================================
# Initialize AI Models using Registry System (pool-filtered)
# ============================================================================


def initialize_models():
    """Load only models whose ``pool`` matches ``AI_REPLICA_POOL``."""
    models_cfg = config.get("models") or {}
    print(f"Initializing models for pool={_REPLICA_POOL}")

    # deepliif
    mcfg = models_cfg.get("deepliif") or {}
    if model_enabled_for_pool("deepliif", mcfg, _REPLICA_POOL):
        from models.deepliif_model import DeepLIIFModel

        model_registry.register_model_class("deepliif", DeepLIIFModel)
        errors = validate_model_paths("deepliif", mcfg)
        if errors:
            for e in errors:
                print(f"✗ deepliif: {e}")
            raise RuntimeError(f"deepliif enabled but paths missing: {errors}")
        deepliif_config = {
            "model_dir": get_absolute_path(mcfg["path"]),
            "tile_size": mcfg.get("tile_size", 256),
            "post_processing": mcfg.get("post_processing", True),
            "gpu_ids": [],
        }
        model_registry.load_model("deepliif", deepliif_config)
        print("✓ Loaded DeepLIIF model")

    # patch_classifier
    mcfg = models_cfg.get("patch_classifier") or {}
    if model_enabled_for_pool("patch_classifier", mcfg, _REPLICA_POOL):
        from models.patch_classifier_model import PatchClassifierModel

        model_registry.register_model_class("patch_classifier", PatchClassifierModel)
        errors = validate_model_paths("patch_classifier", mcfg)
        if errors:
            for e in errors:
                print(f"⚠ patch_classifier: {e}")
            print("  Model registered but not loaded.")
        else:
            model_registry.load_model(
                "patch_classifier",
                {
                    "model_path": get_absolute_path(mcfg["path"]),
                    "patch_size": mcfg.get("patch_size", 64),
                    "batch_size": mcfg.get("batch_size", 32),
                    "classifier_threshold": mcfg.get("classifier_threshold", 0.8),
                    "generate_gradcam": mcfg.get("generate_gradcam", True),
                    "device": mcfg.get("device", "auto"),
                },
            )
            print("✓ Loaded PatchClassifier model")

    # ec_cancer
    mcfg = models_cfg.get("ec_cancer") or {}
    if model_enabled_for_pool("ec_cancer", mcfg, _REPLICA_POOL):
        from models.ec_cancer_model import ECCancerModel

        model_registry.register_model_class("ec_cancer", ECCancerModel)
        errors = validate_model_paths("ec_cancer", mcfg)
        if errors:
            for e in errors:
                print(f"⚠ ec_cancer: {e}")
            print("  Model registered but not loaded.")
        else:
            model_registry.load_model(
                "ec_cancer",
                {
                    "patch_classifier_model_path": get_absolute_path(mcfg["patch_classifier_path"]),
                    "representation_generator_model_path": get_absolute_path(mcfg["representation_path"]),
                    "varmil_model_path": get_absolute_path(mcfg["varmil_path"]),
                    "device": mcfg.get("device", "auto"),
                },
            )
            print("✓ Loaded EC Cancer model")

    # hovernet
    mcfg = models_cfg.get("hovernet") or {}
    if model_enabled_for_pool("hovernet", mcfg, _REPLICA_POOL):
        from models.hovernet_model import HoVerNetModel

        model_registry.register_model_class("hovernet", HoVerNetModel)
        errors = validate_model_paths("hovernet", mcfg)
        if errors:
            for e in errors:
                print(f"⚠ hovernet: {e}")
            print("  Model registered but not loaded.")
        else:
            default_variant = mcfg.get("default_variant", "pannuke")
            model_registry.load_model(
                "hovernet",
                {
                    "model_variant": default_variant,
                    "device": mcfg.get("device", "cuda"),
                    "gpu_ids": mcfg.get("gpu_ids", [0]),
                    "batch_size": mcfg.get("batch_size", 8),
                },
            )
            print(f"✓ Loaded HoVer-Net model (variant: {default_variant})")

    # pdac
    mcfg = models_cfg.get("pdac") or {}
    if model_enabled_for_pool("pdac", mcfg, _REPLICA_POOL):
        from models.pdac_model import PDACModel

        model_registry.register_model_class("pdac", PDACModel)
        errors = validate_model_paths("pdac", mcfg)
        if errors:
            for e in errors:
                print(f"✗ pdac: {e}")
            raise RuntimeError(f"pdac enabled but paths missing: {errors}")
        pdac_config = {
            "package_root": abs_path(mcfg["package_root"]),
            "device": mcfg.get("device", "auto"),
            "batch_size": mcfg.get("batch_size", 16),
            "lazy_load": mcfg.get("lazy_load", True),
        }
        model_registry.load_model("pdac", pdac_config)
        print(f"✓ Registered PDAC model (lazy={pdac_config['lazy_load']}) from {pdac_config['package_root']}")

    loaded = model_registry.list_loaded_models()
    print(f"Pool {_REPLICA_POOL} loaded models: {loaded}")


def warmup_models():
    """Run a throwaway region through each model so the first user does not.

    Several postprocessing steps are numba-jitted, which costs about seven
    seconds the first time a replica runs them. Paying that here means it
    happens once at boot instead of on somebody's first region.
    """
    rng = np.random.default_rng(0)
    # Structured noise rather than a flat colour: models skip blank tiles, and
    # a skipped tile compiles nothing.
    noise = rng.integers(90, 220, size=(512, 512, 3), dtype=np.uint8)
    probe = Image.fromarray(noise)
    mask = Image.new("L", probe.size, 255)

    for name in model_registry.list_loaded_models():
        model = model_registry.get_model(name)
        if model is None:
            continue
        # PDAC GigaPath is ~10 GB; keep it lazy until a real request.
        if name == "pdac" or getattr(model, "_lazy", False):
            print(f"↷ Skipping warmup for {name} (lazy / heavy model)")
            continue
        started = time.perf_counter()
        try:
            model.process(image=probe, mask=mask, annotation_points=None,
                          hyperparameters=None)
            print(f"✓ Warmed up {name} in {time.perf_counter() - started:.1f}s")
        except Exception as e:  # a failed warmup must not stop the service
            print(f"⚠ Warmup for {name} failed ({e}); first request will be slower")

    probe.close()
    mask.close()


# Initialize models at startup
initialize_models()
warmup_models()


# ============================================================================
# Existing Endpoints (Keep from original app.py)
# ============================================================================

@app.post("/generate_mask")
async def generate_mask_api(
    file: UploadFile = File(...),
    is_tma: bool = Form(False)
):
    if mask_gen is None:
        return JSONResponse(
            {"status": "error", "message": "Mask generator is not enabled."},
            status_code=500
        )
    """Generate mask from slide image."""
    # Check if it's a PNG file
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
        is_png = True
    
    if not is_png:
        return JSONResponse(
            {"status": "error", "message": "Only PNG files are supported."},
            status_code=400
        )

    # Process uploaded PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        image_bytes = BytesIO(contents)

    # Generate mask
    final_mask = mask_gen.generate_mask(image_bytes, is_tma=is_tma)

    if final_mask is None:
        return JSONResponse({"status": "no_masks_found"})

    # Convert mask to base64 PNG
    _, mask_png = cv2.imencode(".png", final_mask.astype(np.uint8))
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    return {
        "status": "success",
        "is_tma": is_tma,
        "mask_base64": mask_b64,
        "mask_shape": final_mask.shape
    }


@app.post("/extract_regions")
async def extract_regions_api(
    file: UploadFile = File(...),
    slide_width: int = Form(...),
    slide_height: int = Form(...),
):
    """Extract regions from mask."""
    # Check if it's a PNG file
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
        is_png = True
    
    if not is_png:
        return JSONResponse(
            {"status": "error", "message": "Only PNG files are supported."},
            status_code=400
        )

    # Process uploaded mask PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        image_bytes = BytesIO(contents)
    
    # Extract regions from the mask
    regions = extract_regions_from_mask(
        image_bytes, (slide_width, slide_height)
    )

    return {
        "status": "success",
        "regions": regions
    }


@app.post("/generate_tissue_regions")
async def generate_tissue_regions_api(
    file: UploadFile = File(...),
    slide_width: int = Form(...),
    slide_height: int = Form(...),
    is_tma: bool = Form(False),
):
    """Generate mask from a thumbnail and extract tissue regions in one step.

    Combines /generate_mask and /extract_regions so the caller does not need
    two round-trips.  Returns the same region format as /extract_regions plus
    the intermediate mask as base64 PNG.
    """
    if mask_gen is None:
        return JSONResponse(
            {"status": "error", "message": "Mask generator is not enabled."},
            status_code=500,
        )

    # Validate PNG input
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
        is_png = True

    if not is_png:
        return JSONResponse(
            {"status": "error", "message": "Only PNG files are supported."},
            status_code=400,
        )

    contents = await file.read()
    image_bytes = BytesIO(contents)

    # Step 1 – generate binary mask
    final_mask = mask_gen.generate_mask(image_bytes, is_tma=is_tma)
    if final_mask is None:
        return JSONResponse({"status": "no_masks_found"})

    # Encode mask to PNG bytes for region extraction
    _, mask_png = cv2.imencode(".png", final_mask.astype(np.uint8))
    mask_bytes_io = BytesIO(mask_png.tobytes())

    # Step 2 – extract regions from the mask
    regions = extract_regions_from_mask(
        mask_bytes_io, (slide_width, slide_height)
    )

    # Also return the mask as base64 for optional client-side use
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    return {
        "status": "success",
        "mask_base64": mask_b64,
        "mask_shape": list(final_mask.shape),
        "regions": regions,
    }


@app.post("/auto_name_regions")
async def auto_name_regions_api(request: dict):
    """Auto-name regions using intelligent spatial ordering."""
    try:
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        
        slide_width = request.get('slide_width')
        slide_height = request.get('slide_height')
        regions = request.get('regions', [])
        
        if not slide_width or not slide_height:
            return JSONResponse({
                "success": False,
                "error": "slide_width and slide_height are required"
            }, status_code=400)
        
        if not regions:
            return {
                "success": True,
                "ordered_indices": [],
                "metadata": {"message": "No regions to order"}
            }
        
        # Convert to numpy arrays
        centers = np.array([
            [r['center_x'], r['center_y']] for r in regions
        ])
        original_indices = [r['original_index'] for r in regions]
        
        # Detect orientation
        orientation_info = detect_slide_orientation(
            centers, slide_width, slide_height
        )
        
        # Apply snake pattern ordering
        if orientation_info['is_grid_like']:
            ordered_indices = grid_snake_ordering(
                centers, original_indices, orientation_info
            )
        else:
            ordered_indices = simple_spatial_ordering(
                centers, original_indices
            )
        
        # Convert to native Python types
        safe_ordered_indices = [int(idx) for idx in ordered_indices]
        safe_orientation_info = {}
        for key, value in orientation_info.items():
            if hasattr(value, 'item'):
                safe_orientation_info[key] = value.item()
            elif isinstance(value, np.bool_):
                safe_orientation_info[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                safe_orientation_info[key] = float(value)
            else:
                safe_orientation_info[key] = value
        
        return {
            "success": True,
            "ordered_indices": safe_ordered_indices,
            "metadata": {
                "total_regions": len(regions),
                "orientation": safe_orientation_info,
                "ordering_method": "grid_snake" if orientation_info['is_grid_like'] else "spatial_fallback"
            }
        }
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Internal error: {str(e)}"
        }, status_code=500)


# ============================================================================
# REFACTORED: Processing Endpoints Using Model Registry
# ============================================================================

_UPLOAD_GONE = {
    "status": "error",
    "message": (
        "Upload-based process APIs are retired. Use POST /process_region_from_slide "
        "(sync debug) or POST /inference/jobs (async)."
    ),
    "canonical": ["/process_region_from_slide", "/inference/jobs"],
}


@app.post("/process_region_annotation")
async def process_region_annotation_api(
    region: UploadFile = File(None),
    mask: str = Form(None),
    region_id: str = Form(""),
    model_name: str = Form("deepliif"),
    hyperparameters: str = Form(None),
):
    """Deprecated — use /process_region_from_slide or /inference/jobs."""
    return JSONResponse(_UPLOAD_GONE, status_code=410)


class SlideRegionRequest(BaseModel):
    """A region to process, described by location rather than by pixels."""

    slide_path: str
    roi_bbox: List[float]
    polygon: Optional[List[Any]] = None
    downsample: float = 1.0
    overlay_downsample: Optional[float] = None
    model_name: str = "deepliif"
    hyperparameters: Optional[Dict[str, Any]] = None
    region_id: str = ""


def run_slide_region_inference(
    request_data: Dict[str, Any],
    *,
    hold_lock: bool = True,
) -> Dict[str, Any]:
    """Shared core for sync /process_region_from_slide and async inference jobs.

    ``hold_lock=False`` when the caller (async worker) already holds
    ``_inference_lock``.
    """
    model_name = request_data.get("model_name") or "deepliif"
    model = model_registry.get_model(model_name)
    if not model:
        return {
            "status": "error",
            "message": (
                f"Model '{model_name}' not found on this replica "
                f"(pool={_REPLICA_POOL}). Available: {model_registry.list_loaded_models()}"
            ),
        }

    try:
        slide_path = resolve_slide_path(
            request_data["slide_path"], config.get("slide_roots", [])
        )
        region_image, info = read_region(
            slide_path,
            request_data["roi_bbox"],
            downsample=float(request_data.get("downsample") or 1.0),
            max_pixels=config.get("max_region_pixels") or None,
        )
    except SlideAccessError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print(f"ERROR: failed to read region from {request_data.get('slide_path')}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to read slide region: {e}"}

    bbox = info["bbox"]
    mask_image = polygon_mask(request_data.get("polygon"), bbox, info["size"])

    hyperparameters = dict(request_data.get("hyperparameters") or {})
    if getattr(model, "accepts_mpp", False):
        if info.get("mpp"):
            hyperparameters.setdefault("mpp", info["mpp"])
        if info.get("downsample"):
            hyperparameters.setdefault("downsample", info["downsample"])

    queued_at = time.perf_counter()
    started_at = queued_at
    elapsed = 0.0
    try:
        def _run():
            nonlocal started_at, elapsed
            started_at = time.perf_counter()
            result = model.process(
                image=region_image,
                mask=mask_image,
                annotation_points=None,
                hyperparameters=hyperparameters,
            )
            elapsed = time.perf_counter() - started_at
            return result

        if hold_lock:
            with _inference_lock:
                result = _run()
        else:
            result = _run()
    except Exception as e:
        region_image.close()
        mask_image.close()
        print(f"ERROR: Model {model_name} failed to process region: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Model processing failed: {e}"}

    processed = result.get("processed_image") if result.get("success") else None
    try:
        if not result.get("success"):
            return {
                "status": "error",
                "message": result.get("error", "Unknown error"),
            }

        x0, y0, w, h = bbox
        overlay_downsample = float(
            request_data.get("overlay_downsample") or info["downsample"]
        )
        overlay_downsample = max(1.0, overlay_downsample)
        overlay_w = max(1, int(round(w / overlay_downsample)))
        overlay_h = max(1, int(round(h / overlay_downsample)))
        if processed.size != (overlay_w, overlay_h):
            processed = processed.resize(
                (overlay_w, overlay_h), Image.Resampling.LANCZOS
            )

        buffer = BytesIO()
        processed.save(buffer, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        for image in (processed, region_image, mask_image):
            if image is not None:
                image.close()

    return {
        "status": "success",
        "processed_image_base64": img_base64,
        "overlay_bbox": [x0, y0, w, h],
        "overlay_width": overlay_w,
        "overlay_height": overlay_h,
        "overlay_downsample": w / overlay_w if overlay_w else overlay_downsample,
        "processed_downsample": info["downsample"],
        "slide_level_used": info["level"],
        "score": result["scores"],
        "str_result": result.get("str_result", "Processing complete"),
        "region_id": request_data.get("region_id") or "",
        "model_used": model_name,
        "model_version": model.model_version,
        "metadata": result.get("metadata") or {},
        "timing": {
            "queue_seconds": round(started_at - queued_at, 3),
            "model_seconds": round(elapsed, 3),
            "region_pixels": info["size"][0] * info["size"][1],
        },
    }


@app.post("/process_region_from_slide")
def process_region_from_slide_api(request: SlideRegionRequest):
    """Sync debug/smoke path — same core as async ``/inference/jobs``."""
    result = run_slide_region_inference(request.model_dump())
    if result.get("status") == "error":
        return JSONResponse(result, status_code=400 if "not found" in (result.get("message") or "") else 500)
    return result


@app.post("/inference/jobs")
def create_inference_job(request: SlideRegionRequest):
    """Enqueue region inference; returns immediately with a job id."""
    model_name = request.model_name
    if model_name not in model_registry.list_loaded_models():
        # Still enqueue only if this replica owns the model; Aimviewer routes by pool.
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    f"Model '{model_name}' not loaded on pool={_REPLICA_POOL}. "
                    f"Available: {model_registry.list_loaded_models()}"
                ),
            },
            status_code=400,
        )
    created = inference_jobs.create_job(config, _REPLICA_POOL, request.model_dump())
    return created


@app.get("/inference/jobs/{job_id}")
def get_inference_job(job_id: str):
    job = inference_jobs.get_job(config, job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "Job not found"}, status_code=404)
    # Never return huge base64 in the poll body.
    public = {k: v for k, v in job.items() if k not in ("payload",)}
    public.pop("processed_image_base64", None)
    return public


@app.get("/inference/jobs/{job_id}/overlay")
def get_inference_job_overlay(job_id: str):
    from fastapi.responses import Response

    overlay = inference_jobs.get_overlay(config, job_id)
    if overlay is None:
        job = inference_jobs.get_job(config, job_id)
        if not job:
            return JSONResponse({"status": "error", "message": "Job not found"}, status_code=404)
        if job.get("status") != inference_jobs.STATUS_SUCCEEDED:
            return JSONResponse(
                {"status": "error", "message": f"Job status is {job.get('status')}"},
                status_code=409,
            )
        return JSONResponse({"status": "error", "message": "Overlay not available"}, status_code=404)
    return Response(content=overlay, media_type="image/png")


@app.post("/inference/jobs/{job_id}/cancel")
def cancel_inference_job(job_id: str):
    job = inference_jobs.request_cancel(config, job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "Job not found"}, status_code=404)
    return {"job_id": job_id, "status": job["status"], "cancel_requested": job.get("cancel_requested")}


@app.post("/process_region")
async def process_region_api(
    region: UploadFile = File(None),
    mask: UploadFile = File(None),
    model_name: str = Form("deepliif"),
):
    """Deprecated — use /process_region_from_slide or /inference/jobs."""
    return JSONResponse(_UPLOAD_GONE, status_code=410)


# Per-replica async inference worker (must start after run_slide_region_inference exists).
inference_jobs.start_worker(
    config,
    _REPLICA_POOL,
    process_fn=lambda payload: run_slide_region_inference(payload, hold_lock=False),
    inference_lock=_inference_lock,
)


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """
    Report replica readiness and which GPU this process is pinned to.

    Each replica runs with CUDA_VISIBLE_DEVICES set to a single GPU, so the
    device seen here as index 0 is the physical card named by AI_REPLICA_GPU.
    """
    import os
    import torch

    gpu = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "physical_index": os.environ.get("AI_REPLICA_GPU"),
            "memory_allocated_mb": round(
                torch.cuda.memory_allocated(0) / 1024 ** 2, 1
            ),
            "memory_reserved_mb": round(
                torch.cuda.memory_reserved(0) / 1024 ** 2, 1
            ),
        }

    return {
        "status": "ok",
        "port": os.environ.get("AI_REPLICA_PORT"),
        "pool": _REPLICA_POOL,
        "gpu": gpu,
        "mask_generator": mask_gen is not None,
        "mask_generator_backend": getattr(mask_gen, "model_name", None),
        "loaded_models": model_registry.list_loaded_models(),
    }


@app.get("/models/list")
async def list_models():
    """
    List all available models.
    
    Returns information about registered and loaded models.
    """
    return {
        "registered_models": model_registry.list_registered_models(),
        "loaded_models": model_registry.list_loaded_models()
    }


@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Returns model configuration, hyperparameters schema, etc.
    """
    info = model_registry.get_model_info(model_name)
    if info:
        mcfg = (config.get("models") or {}).get(model_name) or {}
        info = dict(info)
        info["pool"] = mcfg.get("pool") or _REPLICA_POOL
        return info
    else:
        return JSONResponse(
            {
                "error": f"Model '{model_name}' not loaded. "
                        f"Available: {model_registry.list_loaded_models()}"
            },
            status_code=404
        )


@app.get("/models/{model_name}/hyperparameters")
async def get_model_hyperparameters(model_name: str):
    """
    Get the hyperparameters schema for a model.
    
    Useful for building dynamic UIs for model configuration.
    """
    model = model_registry.get_model(model_name)
    if model:
        return {
            "model": model_name,
            "hyperparameters": model.get_hyperparameters_schema()
        }
    else:
        return JSONResponse(
            {"error": f"Model '{model_name}' not found"},
            status_code=404
        )


# ============================================================================
# Helper Functions for Auto-Naming Regions
# ============================================================================

def detect_slide_orientation(centers, slide_width, slide_height):
    """Detect if regions form a grid pattern and determine slide orientation."""
    try:
        if len(centers) < 4:
            return {
                'is_grid_like': False,
                'rotation_angle': 0,
                'rows': 1,
                'cols': len(centers)
            }
        
        # Calculate distances between all pairs of points
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(centers))
        
        # Find the most common distances (indicating grid spacing)
        non_zero_distances = distances[distances > 0]
        
        if len(non_zero_distances) == 0:
            return {'is_grid_like': False, 'rotation_angle': 0, 'rows': 1, 'cols': len(centers)}
        
        # Use histogram to find common distances
        hist, bins = np.histogram(non_zero_distances, bins=20)
        most_common_distance = bins[np.argmax(hist)]
        
        # Tolerance for grid detection
        tolerance = most_common_distance * 0.3
        
        # Count how many point pairs have approximately the common distance
        grid_connections = np.sum((np.abs(distances - most_common_distance) < tolerance) & (distances > 0))
        total_possible_connections = len(centers) * (len(centers) - 1) / 2
        
        # If a significant portion of connections match the grid pattern
        grid_ratio = grid_connections / total_possible_connections
        is_grid_like = grid_ratio > 0.3  # At least 30% of connections suggest grid
        
        # Estimate grid dimensions
        if is_grid_like:
            # Sort by Y coordinate to find rows
            y_sorted_indices = np.argsort(centers[:, 1])
            y_coords = centers[y_sorted_indices, 1]
            
            # Find row breaks (significant Y jumps)
            y_diffs = np.diff(y_coords)
            row_threshold = np.median(y_diffs) * 1.5 if len(y_diffs) > 0 else 0
            
            rows = 1 + np.sum(y_diffs > row_threshold) if row_threshold > 0 else 1
            cols = len(centers) // rows if rows > 0 else len(centers)
            
            # Estimate rotation by looking at the angle of the most common vector
            vectors = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if abs(dist - most_common_distance) < tolerance:
                        vectors.append(centers[j] - centers[i])
            
            if vectors:
                # Find the dominant direction
                vectors = np.array(vectors)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                # Normalize angles to [0, π/2] since we care about grid alignment
                angles = np.abs(angles) % (np.pi / 2)
                rotation_angle = np.median(angles)
            else:
                rotation_angle = 0
        else:
            rows = 1
            cols = len(centers)
            rotation_angle = 0
        
        return {
            'is_grid_like': is_grid_like,
            'rotation_angle': rotation_angle,
            'rows': max(1, rows),
            'cols': max(1, cols),
            'grid_ratio': grid_ratio,
            'common_distance': most_common_distance
        }
        
    except Exception:
        # Fallback if scipy is not available or other errors
        return {
            'is_grid_like': False,
            'rotation_angle': 0,
            'rows': 1,
            'cols': len(centers)
        }


def grid_snake_ordering(centers, original_indices, orientation_info):
    """Order regions in a snake pattern considering grid layout and rotation."""
    try:
        if len(centers) <= 1:
            return original_indices
        
        # Rotate points to align with grid if needed
        rotation_angle = orientation_info.get('rotation_angle', 0)
        
        if abs(rotation_angle) > 0.1:  # Only rotate if significant rotation detected
            cos_a, sin_a = np.cos(-rotation_angle), np.sin(-rotation_angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_centers = centers @ rotation_matrix.T
        else:
            rotated_centers = centers.copy()
        
        # Sort by Y coordinate first to identify rows
        y_sorted_indices = np.argsort(rotated_centers[:, 1])
        y_coords = rotated_centers[y_sorted_indices, 1]
        
        # Group points into rows using clustering approach
        if len(y_coords) > 1:
            y_diffs = np.diff(y_coords)
            if len(y_diffs) > 0:
                row_threshold = np.median(y_diffs) * 1.2
            else:
                row_threshold = 0
        else:
            row_threshold = 0
        
        # Group into rows
        rows = []
        current_row = [y_sorted_indices[0]]
        current_y = y_coords[0]
        
        for i in range(1, len(y_coords)):
            if y_coords[i] - current_y > row_threshold:
                # Start new row
                rows.append(current_row)
                current_row = [y_sorted_indices[i]]
                current_y = y_coords[i]
            else:
                # Add to current row
                current_row.append(y_sorted_indices[i])
        rows.append(current_row)
        
        # Sort each row by X coordinate and apply snake pattern
        ordered_indices = []
        for row_idx, row_indices in enumerate(rows):
            row_centers = rotated_centers[row_indices]
            x_sorted = np.argsort(row_centers[:, 0])
            
            if row_idx % 2 == 0:
                # Even rows: left to right
                row_order = [row_indices[x_sorted[i]] for i in range(len(x_sorted))]
            else:
                # Odd rows: right to left (snake pattern)
                row_order = [row_indices[x_sorted[i]] for i in range(len(x_sorted) - 1, -1, -1)]
            
            ordered_indices.extend(row_order)
        
        # Map back to original indices
        result = [original_indices[i] for i in ordered_indices]
        return result
        
    except Exception:
        # Fallback to simple ordering
        return simple_spatial_ordering(centers, original_indices)


def simple_spatial_ordering(centers, original_indices):
    """Simple fallback ordering: top-to-bottom, left-to-right."""
    try:
        # Sort by Y first (top to bottom), then by X (left to right)
        # Normalize coordinates to handle different scales
        y_coords = centers[:, 1]
        x_coords = centers[:, 0]
        
        # Create a composite sorting key
        # Primary: Y coordinate (top to bottom)
        # Secondary: X coordinate (left to right)
        sort_indices = np.lexsort((x_coords, y_coords))
        
        return [original_indices[i] for i in sort_indices]
        
    except Exception:
        # Ultimate fallback: return original order
        return original_indices


# ============================================================================
# Cleanup on Shutdown
# ============================================================================

@app.on_event("shutdown")
def shutdown_event():
    """Clean up models on shutdown."""
    print("Shutting down and cleaning up models...")
    model_registry.unload_all_models()
    print("Cleanup complete")


if __name__ == "__main__":
    import uvicorn
    import os
    import sys
    
    # Ensure current directory is in Python path for reload subprocess
    current_dir = str(Path(__file__).parent.resolve())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Set PYTHONPATH environment variable for uvicorn subprocess
    python_path = os.environ.get('PYTHONPATH', '')
    paths_to_add = [current_dir]
    for path in paths_to_add:
        if path not in python_path:
            python_path = f"{path}:{python_path}" if python_path else path
    os.environ['PYTHONPATH'] = python_path
    
    # Run server
    # Note: reload=False to avoid subprocess import issues with HoVer-Net
    # Use: uvicorn app_refactored:app --reload for development with manual restart
    uvicorn.run(
        app,  # Direct app reference (no string) when reload=False
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled to avoid subprocess issues
        log_level="info"
    )
