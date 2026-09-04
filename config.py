# config.py — load and normalize extract_areas configuration.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def load_config() -> dict[str, Any]:
    with open(CONFIG_FILE, "r") as f:
        raw = json.load(f)
    return normalize_config(raw)


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept the structured schema; soft-migrate legacy flat keys if present."""
    cfg = dict(raw)

    # Legacy flat → structured migration (one-shot compatibility).
    if "models" not in cfg and (
        "deepliif_enabled" in cfg or "pdac_enabled" in cfg or "hovernet_enabled" in cfg
    ):
        cfg["models"] = _legacy_models(cfg)
    if "mask_generator" not in cfg and "maskgenerator_enabled" in cfg:
        cfg["mask_generator"] = {
            "enabled": bool(cfg.get("maskgenerator_enabled")),
            "backend": cfg.get("default_model", "sam3"),
            "sam_model_path": cfg.get("sam_model_path", "TBD"),
            "sam2_model_path": cfg.get("sam2_model_path"),
            "sam2_config_path": cfg.get("sam2_config_path"),
            "sam3_model_id": cfg.get("sam3_model_id", "facebook/sam3"),
            "sam3_points_per_batch": cfg.get("sam3_points_per_batch", 64),
            "sam3_crop_overlap_ratio": cfg.get("sam3_crop_overlap_ratio", 0.4),
        }
    if "pools" not in cfg:
        cfg["pools"] = {
            "general": {"gpus": [0, 1, 2], "base_port": 8001},
            "heavy": {"gpus": [3], "base_port": 8004},
        }
    if "redis" not in cfg:
        cfg["redis"] = {"host": "localhost", "port": 6379, "db": 0}

    cfg.setdefault("slide_roots", ["/data/omero/data"])
    cfg.setdefault("max_region_pixels", 64000000)
    cfg.setdefault("models", {})
    cfg.setdefault("mask_generator", {"enabled": False, "backend": "sam3"})
    return cfg


def _legacy_models(cfg: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if cfg.get("deepliif_enabled"):
        models["deepliif"] = {
            "enabled": True,
            "pool": "general",
            "path": cfg.get("deepliif_model_path", "models/deepliif/"),
        }
    if cfg.get("patch_classifier_enabled"):
        models["patch_classifier"] = {
            "enabled": True,
            "pool": "general",
            "path": cfg.get("patch_classifier_model_path"),
        }
    if cfg.get("ec_cancer_enabled"):
        models["ec_cancer"] = {
            "enabled": True,
            "pool": "general",
            "patch_classifier_path": cfg.get("ec_cancer_patch_classifier_path"),
            "representation_path": cfg.get("ec_cancer_representation_path"),
            "varmil_path": cfg.get("ec_cancer_varmil_path"),
        }
    if cfg.get("hovernet_enabled"):
        models["hovernet"] = {
            "enabled": True,
            "pool": "general",
            "default_variant": cfg.get("hovernet_default_variant", "pannuke"),
            "device": cfg.get("hovernet_device", "cuda"),
            "gpu_ids": cfg.get("hovernet_gpu_ids", [0]),
            "batch_size": cfg.get("hovernet_batch_size", 8),
            "variants": cfg.get("hovernet_models", {}),
        }
    if cfg.get("pdac_enabled"):
        models["pdac"] = {
            "enabled": True,
            "pool": "heavy",
            "lazy_load": cfg.get("pdac_lazy_load", True),
            "package_root": cfg.get("pdac_package_root"),
            "device": cfg.get("pdac_device", "auto"),
            "batch_size": cfg.get("pdac_batch_size", 16),
        }
    return models


def replica_pool() -> str:
    return os.environ.get("AI_REPLICA_POOL", "general").strip().lower() or "general"


def model_enabled_for_pool(name: str, model_cfg: dict[str, Any], pool: str | None = None) -> bool:
    if not model_cfg.get("enabled", False):
        return False
    pool = pool or replica_pool()
    return (model_cfg.get("pool") or "general").lower() == pool


def abs_path(relative_or_absolute: str | None) -> str | None:
    if not relative_or_absolute:
        return None
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return str(p)
    return str((Path(__file__).resolve().parent / p).resolve())


def validate_model_paths(name: str, model_cfg: dict[str, Any]) -> list[str]:
    """Return list of missing-path error strings (empty = OK)."""
    errors: list[str] = []
    if name == "deepliif":
        path = abs_path(model_cfg.get("path"))
        if not path or not Path(path).exists():
            errors.append(f"deepliif path missing: {model_cfg.get('path')}")
    elif name == "patch_classifier":
        path = abs_path(model_cfg.get("path"))
        if not path or not Path(path).is_file():
            errors.append(f"patch_classifier weights missing: {model_cfg.get('path')}")
    elif name == "ec_cancer":
        for key in ("patch_classifier_path", "representation_path", "varmil_path"):
            path = abs_path(model_cfg.get(key))
            if not path or not Path(path).is_file():
                errors.append(f"ec_cancer {key} missing: {model_cfg.get(key)}")
    elif name == "hovernet":
        variants = model_cfg.get("variants") or {}
        default = model_cfg.get("default_variant", "pannuke")
        variant = variants.get(default) or {}
        ckpt = abs_path(variant.get("checkpoint"))
        if not ckpt or not Path(ckpt).is_file():
            errors.append(f"hovernet checkpoint missing for variant {default}: {variant.get('checkpoint')}")
    elif name == "pdac":
        root = abs_path(model_cfg.get("package_root"))
        stage1 = Path(root or "") / "stage1" / "best_model.pth"
        stage2 = Path(root or "") / "stage2" / "best_mil_model.pth"
        if not stage1.is_file():
            errors.append(f"pdac stage1 missing: {stage1}")
        if not stage2.is_file():
            errors.append(f"pdac stage2 missing: {stage2}")
    return errors


config = load_config()
