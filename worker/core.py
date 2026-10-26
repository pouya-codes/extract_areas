"""The inference step itself, independent of how the task arrived.

This is ``run_slide_region_inference`` from ``app_refactored.py`` lifted out
of the FastAPI process so a worker can call it with a model it loaded
itself. Slide access is injectable so the loop can be tested with an
in-memory image instead of a real pyramid file.
"""
from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Any, Callable, Optional

from PIL import Image

from inference_platform.settings import Settings

logger = logging.getLogger(__name__)

ReadRegion = Callable[..., tuple[Image.Image, dict[str, Any]]]
ResolvePath = Callable[[str, list[str]], Any]
PolygonMask = Callable[[Any, Any, Any], Image.Image]


class InferenceCore:
    def __init__(
        self,
        settings: Settings,
        *,
        read_region: Optional[ReadRegion] = None,
        resolve_slide_path: Optional[ResolvePath] = None,
        polygon_mask: Optional[PolygonMask] = None,
    ):
        self._settings = settings
        if read_region is None or resolve_slide_path is None or polygon_mask is None:
            # Imported lazily: openslide is only needed by real workers.
            import slide_region

            read_region = read_region or slide_region.read_region
            resolve_slide_path = resolve_slide_path or slide_region.resolve_slide_path
            polygon_mask = polygon_mask or slide_region.polygon_mask
        self._read_region = read_region
        self._resolve = resolve_slide_path
        self._polygon_mask = polygon_mask

    def run(self, model, payload: dict[str, Any]) -> dict[str, Any]:
        """Process one region. Returns the same dict shape the old API did."""
        from slide_region import SlideAccessError  # light import, no openslide

        try:
            slide_path = self._resolve(payload["slide_path"], list(self._settings.slide_roots))
            region_image, info = self._read_region(
                slide_path,
                payload["roi_bbox"],
                downsample=float(payload.get("downsample") or 1.0),
                max_pixels=self._settings.max_region_pixels or None,
            )
        except SlideAccessError as exc:
            return {"status": "error", "message": str(exc), "retryable": False}
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to read region from %s", payload.get("slide_path"))
            return {"status": "error", "message": f"Failed to read slide region: {exc}"}

        bbox = info["bbox"]
        mask_image = self._polygon_mask(payload.get("polygon"), bbox, info["size"])

        hyperparameters = dict(payload.get("hyperparameters") or {})
        if getattr(model, "accepts_mpp", False):
            if info.get("mpp"):
                hyperparameters.setdefault("mpp", info["mpp"])
            if info.get("downsample"):
                hyperparameters.setdefault("downsample", info["downsample"])

        started = time.perf_counter()
        try:
            result = model.process(
                image=region_image,
                mask=mask_image,
                annotation_points=None,
                hyperparameters=hyperparameters,
            )
        except Exception:
            region_image.close()
            mask_image.close()
            raise
        elapsed = time.perf_counter() - started

        processed = result.get("processed_image") if result.get("success") else None
        try:
            if not result.get("success"):
                return {"status": "error", "message": result.get("error", "Unknown error")}

            x0, y0, w, h = bbox
            overlay_downsample = max(
                1.0, float(payload.get("overlay_downsample") or info["downsample"])
            )
            overlay_w = max(1, int(round(w / overlay_downsample)))
            overlay_h = max(1, int(round(h / overlay_downsample)))
            overlay_png: Optional[bytes] = None
            if processed is not None:
                if processed.size != (overlay_w, overlay_h):
                    processed = processed.resize((overlay_w, overlay_h), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                processed.save(buffer, format="PNG", optimize=True)
                overlay_png = buffer.getvalue()
        finally:
            for image in (processed, region_image, mask_image):
                if image is not None:
                    image.close()

        return {
            "status": "success",
            "overlay_png": overlay_png,
            "overlay_bbox": [x0, y0, w, h],
            "overlay_width": overlay_w,
            "overlay_height": overlay_h,
            "overlay_downsample": w / overlay_w if overlay_w else overlay_downsample,
            "processed_downsample": info["downsample"],
            "slide_level_used": info.get("level"),
            "score": result.get("scores"),
            "str_result": result.get("str_result", "Processing complete"),
            "model_used": getattr(model, "model_name", payload.get("model_name")),
            "model_version": getattr(model, "model_version", None),
            "metadata": result.get("metadata") or {},
            "timing": {
                "model_seconds": round(elapsed, 3),
                "region_pixels": info["size"][0] * info["size"][1],
            },
        }


def overlay_to_base64(png: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(png).decode("utf-8") if png else None
