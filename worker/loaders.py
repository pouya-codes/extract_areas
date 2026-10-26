"""Build and initialise the one model a worker serves.

The configuration is the ``models`` section of ``config.json`` -- the same
file the monolith used -- so the paths and defaults stay in one place. Two
environment variables adjust it per container:

``MODEL_CONFIG_JSON``  a JSON object merged over the model's section
``DEVICE``             ``cuda`` / ``cpu`` / ``auto``; overrides ``device``
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from inference_platform.settings import Settings

logger = logging.getLogger(__name__)


def model_config(name: str, raw_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if raw_config is None:
        from config import config as repo_config

        raw_config = repo_config
    section = dict((raw_config.get("models") or {}).get(name) or {})
    override = os.environ.get("MODEL_CONFIG_JSON")
    if override:
        try:
            section.update(json.loads(override))
        except ValueError as exc:
            raise ValueError(f"MODEL_CONFIG_JSON is not valid JSON: {exc}") from exc
    return section


def _device(settings: Settings, default: str) -> str:
    return default if settings.device == "auto" else settings.device


def build_model(name: str, settings: Settings, raw_config: Optional[dict[str, Any]] = None):
    """Instantiate + initialise the model ``name``; returns the model."""
    from config import abs_path, validate_model_paths

    section = model_config(name, raw_config)
    started = time.perf_counter()

    if name == "stub":
        from worker.stub_model import StubModel

        model = StubModel()
        model.initialize(section)

    elif name == "deepliif":
        from models.deepliif_model import DeepLIIFModel

        _check(validate_model_paths(name, section))
        model = DeepLIIFModel()
        model.initialize({
            "model_dir": abs_path(section["path"]),
            "tile_size": section.get("tile_size", 256),
            "post_processing": section.get("post_processing", True),
            # DeepLIIF picks the visible GPU itself; an empty list means
            # "whatever CUDA exposes", which in a pod is exactly one card.
            "gpu_ids": [] if _device(settings, "cuda") != "cpu" else [],
        })

    elif name == "hovernet":
        from models.hovernet_model import HoVerNetModel

        _check(validate_model_paths(name, section))
        model = HoVerNetModel()
        model.initialize({
            "model_variant": section.get("default_variant", "pannuke"),
            "device": _device(settings, section.get("device", "cuda")),
            "gpu_ids": section.get("gpu_ids", [0]),
            "batch_size": section.get("batch_size", 8),
        })

    elif name == "ec_cancer":
        from models.ec_cancer_model import ECCancerModel

        _check(validate_model_paths(name, section))
        model = ECCancerModel()
        model.initialize({
            "patch_classifier_model_path": abs_path(section["patch_classifier_path"]),
            "representation_generator_model_path": abs_path(section["representation_path"]),
            "varmil_model_path": abs_path(section["varmil_path"]),
            "device": _device(settings, section.get("device", "auto")),
        })

    elif name == "pdac":
        from models.pdac_model import PDACModel

        _check(validate_model_paths(name, section))
        model = PDACModel()
        model.initialize({
            "package_root": abs_path(section["package_root"]),
            "device": _device(settings, section.get("device", "auto")),
            "batch_size": section.get("batch_size", 0),
            # A dedicated worker should be ready when it says it is ready;
            # lazy loading only made sense when PDAC shared a replica.
            "lazy_load": False if settings.worker_eager_load else section.get("lazy_load", True),
        })

    elif name == "patch_classifier":
        from models.patch_classifier_model import PatchClassifierModel

        _check(validate_model_paths(name, section))
        model = PatchClassifierModel()
        model.initialize({
            "model_path": abs_path(section["path"]),
            "patch_size": section.get("patch_size", 64),
            "batch_size": section.get("batch_size", 32),
            "classifier_threshold": section.get("classifier_threshold", 0.8),
            "generate_gradcam": section.get("generate_gradcam", True),
            "device": _device(settings, section.get("device", "auto")),
        })

    else:
        raise ValueError(f"unknown model '{name}'")

    model._is_initialized = True
    logger.info("model %s loaded in %.1fs", name, time.perf_counter() - started)
    return model


def _check(errors: list[str]) -> None:
    if errors:
        raise RuntimeError("model files missing: " + "; ".join(errors))


def describe(model, name: str, settings: Settings, raw_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The descriptor a worker publishes to the registry."""
    section = model_config(name, raw_config)
    try:
        schema = model.get_hyperparameters_schema()
    except Exception:  # noqa: BLE001
        logger.exception("could not read hyperparameter schema for %s", name)
        schema = {}
    return {
        "name": name,
        "version": getattr(model, "model_version", "") or "",
        "description": getattr(model, "description", "") or "",
        "requires_mask": bool(getattr(model, "requires_mask", False)),
        "supported_formats": list(getattr(model, "supported_input_formats", []) or []),
        "hyperparameters": schema,
        "pool": section.get("pool") or "general",
        "image": os.environ.get("WORKER_IMAGE"),
        "device": settings.device,
    }


def warmup(model, name: str) -> None:
    """Run a throwaway region so numba/cuDNN compile before the first user."""
    import numpy as np
    from PIL import Image

    if name == "pdac" or getattr(model, "_lazy", False):
        logger.info("skipping warmup for %s (heavy model)", name)
        return
    rng = np.random.default_rng(0)
    noise = rng.integers(90, 220, size=(512, 512, 3), dtype=np.uint8)
    probe = Image.fromarray(noise)
    mask = Image.new("L", probe.size, 255)
    started = time.perf_counter()
    try:
        model.process(image=probe, mask=mask, annotation_points=None, hyperparameters=None)
        logger.info("warmed up %s in %.1fs", name, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - a failed warmup must not stop the worker
        logger.warning("warmup for %s failed (%s); first task will be slower", name, exc)
    finally:
        probe.close()
        mask.close()
