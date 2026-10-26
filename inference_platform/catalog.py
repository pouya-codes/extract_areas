"""Static model catalogue for the gateway.

The registry only knows about a model once a worker for it has started. The
catalogue fills the gap: it lists the models the deployment is *meant* to
serve, so the gateway can accept a job for a model whose workers are
currently scaled to zero and let the autoscaler bring one up.

By default it is derived from the ``models`` section of ``config.json``;
``MODEL_CATALOG_PATH`` points at a JSON file of the same shape to override it.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def load_catalog(path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Return ``{model_name: {"pool": ..., "enabled": ...}}`` for enabled models."""
    source = Path(path) if path else _REPO_CONFIG
    if not source.is_file():
        logger.warning("model catalogue %s not found; catalogue is empty", source)
        return {}
    with open(source) as fh:
        raw = json.load(fh)
    models = raw.get("models") if isinstance(raw, dict) and "models" in raw else raw
    catalog: dict[str, dict[str, Any]] = {}
    for name, cfg in (models or {}).items():
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        catalog[name] = {
            "pool": cfg.get("pool") or "general",
            "description": cfg.get("description") or "",
            "scale_to_zero": bool(cfg.get("scale_to_zero", False)),
        }
    return catalog


def catalog_from_env() -> dict[str, dict[str, Any]]:
    return load_catalog(os.environ.get("MODEL_CATALOG_PATH"))
