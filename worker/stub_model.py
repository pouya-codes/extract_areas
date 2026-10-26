"""A model that needs no weights and no GPU.

It outlines the annotation mask and reports fake patch progress. It exists so
the queue, the gateway and the Kubernetes wiring can be exercised on a laptop
or in CI without any of the real models, and so a cluster can be smoke-tested
before GPU nodes are attached.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from models.base_model import BaseAIModel
from progress_report import report_progress


class StubModel(BaseAIModel):
    def __init__(self):
        super().__init__(model_name="stub", model_version="0.1.0")
        self.description = "Diagnostic model: outlines the region and counts tissue pixels."
        self.requires_mask = False
        self.hyperparameters = {"steps": 4, "step_seconds": 0.0, "line_width": 4}

    def initialize(self, config: Dict[str, Any]) -> None:
        self.hyperparameters.update({k: v for k, v in config.items() if k in self.hyperparameters})
        self._is_initialized = True

    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = {**self.hyperparameters, **(hyperparameters or {})}
        steps = max(1, int(params.get("steps", 4)))
        pause = float(params.get("step_seconds", 0.0))

        for step in range(1, steps + 1):
            if pause:
                time.sleep(pause)
            report_progress(
                patch_done=step,
                patch_total=steps,
                progress_percent=int(100 * step / steps),
                progress_stage="stub",
                progress_label=f"Stub step {step} / {steps}",
            )

        arr = np.asarray(image.convert("L"))
        mask_arr = np.asarray(mask) > 0 if mask is not None else np.ones_like(arr, dtype=bool)
        tissue = int(((arr < 220) & mask_arr).sum())
        total = int(mask_arr.sum())

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if mask is not None:
            edge = np.asarray(mask) > 0
            ys, xs = np.where(edge)
            if len(xs):
                draw.rectangle(
                    [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    outline=(255, 0, 0, 255),
                    width=int(params.get("line_width", 4)),
                )
        return {
            "processed_image": overlay,
            "scores": {"tissue_pixels": tissue, "region_pixels": total,
                       "tissue_fraction": round(tissue / total, 4) if total else 0.0},
            "str_result": f"Tissue {tissue}/{total} px",
            "metadata": {"steps": steps},
            "success": True,
        }

    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        return {
            "steps": {"type": "int", "default": 4, "min": 1, "max": 100,
                      "description": "Progress steps to report"},
            "step_seconds": {"type": "float", "default": 0.0, "min": 0.0, "max": 60.0,
                             "description": "Pause per step, to simulate slow inference"},
        }
