"""
{{Title}} model -- template for a new region-processing model.

Copy this file to ``models/{{name}}_model.py`` (``scripts/new_model.py`` does
that and the rest of the wiring for you), then fill in the three TODOs:
``initialize`` (load weights), ``process`` (run on one region) and
``get_hyperparameters_schema`` (what the panel lets users tune).

How a worker uses this file
---------------------------
* ``build_model(section, device)`` is what the worker calls. ``section`` is
  this model's entry in ``config.json`` (``models.{{name}}``) and ``device``
  is ``"cuda"`` or ``"cpu"``. No change to ``worker/loaders.py`` is needed.
* ``process()`` receives the region already cut from the slide as an RGB
  image, with a binary mask of the annotation polygon. ``hyperparameters``
  holds the user's choices plus, for models with ``accepts_mpp = True``,
  ``mpp`` (microns per pixel of the image handed in) and ``downsample``.
* Call ``report_progress()`` from your batch loop. It drives the progress
  bar in the panel and is where a cancel request stops the task.
* Return a ``processed_image`` the size of the input; the worker rescales it
  to the overlay resolution the user asked for.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from models.base_model import BaseAIModel
from progress_report import report_progress


class {{Class}}(BaseAIModel):
    def __init__(self):
        super().__init__(model_name="{{name}}", model_version="0.1.0")
        self.description = "TODO: one sentence shown under the model in the panel."
        self.requires_mask = True
        # Set to True if tile size / thresholds depend on physical scale.
        self.accepts_mpp = False
        self.supported_input_formats = ["PNG", "JPEG", "JPG", "TIFF"]
        # Defaults; the panel's values override these per job.
        self.hyperparameters = {
            "patch_size": 512,
            "batch_size": 8,
            "threshold": 0.5,
        }
        self.device = "cpu"
        self.net = None

    # ------------------------------------------------------------------ #
    def initialize(self, config: Dict[str, Any]) -> None:
        """Load weights once per worker process.

        ``config`` is built by :func:`build_model` below from the model's
        ``config.json`` section; put paths there, not in code.
        """
        self.device = config.get("device", "cpu")
        weights = config.get("weights_path")
        if not weights:
            raise ValueError("weights_path is required in config.json models.{{name}}")
        # TODO: load the network, e.g.
        #   import torch
        #   self.net = torch.load(weights, map_location=self.device).eval()
        self.net = object()
        self._is_initialized = True

    # ------------------------------------------------------------------ #
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = {**self.hyperparameters, **(hyperparameters or {})}
        patch = int(params["patch_size"])
        width, height = image.size

        # Tile the region; skip tiles outside the annotation.
        mask_arr = np.asarray(mask) > 0 if mask is not None else np.ones((height, width), bool)
        tiles = [
            (x, y)
            for y in range(0, height, patch)
            for x in range(0, width, patch)
            if mask_arr[y:y + patch, x:x + patch].mean() > 0.5
        ]
        total = len(tiles)
        if total == 0:
            return {"success": False, "error": "No tissue inside the annotation."}

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        positives = 0
        for index, (x, y) in enumerate(tiles, start=1):
            tile = image.crop((x, y, x + patch, y + patch))
            # TODO: replace with real inference on ``tile`` (batch them!).
            score = float(np.asarray(tile.convert("L")).mean() < 128)
            if score >= float(params["threshold"]):
                positives += 1
                draw.rectangle([x, y, x + patch, y + patch], fill=(255, 0, 0, 90))
            report_progress(
                patch_done=index,
                patch_total=total,
                progress_percent=int(100 * index / total),
                progress_stage="inference",
                progress_label=f"Tiles {index} / {total}",
            )

        fraction = positives / total
        return {
            "processed_image": overlay,
            "scores": {"tiles": total, "positive_tiles": positives, "positive_fraction": round(fraction, 4)},
            "str_result": f"Positive tiles: {positives}/{total} ({100 * fraction:.1f}%)",
            "metadata": {"patch_size": patch},
            "success": True,
        }

    # ------------------------------------------------------------------ #
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """What the panel renders. Types: int, float, bool, string, choice."""
        return {
            "patch_size": {"type": "int", "default": 512, "min": 64, "max": 4096,
                           "description": "Tile size in pixels"},
            "batch_size": {"type": "int", "default": 8, "min": 0, "max": 128,
                           "description": "Tiles per forward pass (0 = auto)"},
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                          "description": "Score above which a tile counts as positive"},
        }

    def cleanup(self) -> None:
        self.net = None


def build_model(section: Dict[str, Any], device: str) -> BaseAIModel:
    """Called by the worker. ``section`` = config.json ``models.{{name}}``."""
    from config import abs_path

    model = {{Class}}()
    model.initialize({
        "weights_path": abs_path(section.get("weights_path")),
        "device": device,
        **{k: v for k, v in section.items() if k not in ("enabled", "pool", "weights_path")},
    })
    return model
