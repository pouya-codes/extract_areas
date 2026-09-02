# mask_generator.py
"""Tissue mask generation via SAM / SAM2 / SAM3 automatic mask generators."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from collections import Counter

logger = logging.getLogger(__name__)

# Device selection (respects CUDA_VISIBLE_DEVICES set by deploy/run_replica.sh).
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class MaskGenerator:
    """Automatic tissue mask generator backed by SAM, SAM2, or SAM3."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_config: Optional[str] = None,
        model_name: Optional[str] = None,
        hf_model_id: str = "facebook/sam3",
        points_per_batch: int = 64,
        crop_overlap_ratio: float = 0.4,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
    ):
        """
        Args:
            model_path: Local checkpoint for SAM / SAM2.
            model_config: Hydra YAML for SAM2 (required when model_name is sam2).
            model_name: Explicit backend: ``sam``, ``sam2``, or ``sam3``.
                If omitted, inferred from ``model_config`` (sam2) else sam.
            hf_model_id: Hugging Face model id for SAM3 (default facebook/sam3).
            points_per_batch: HF mask-generation mini-batch size (SAM3).
            crop_overlap_ratio: Crop overlap used by SAM2 / SAM3 AMG.
            pred_iou_thresh / stability_score_thresh: Quality filters where supported.
        """
        if model_name is None:
            model_name = "sam2" if model_config is not None else "sam"
        self.model_name = model_name.lower()
        self.hf_model_id = hf_model_id
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._pipeline = None  # SAM3 HF pipeline
        self.mask_generator = None  # SAM / SAM2 AMG
        self.sam = None
        self.predictor = None

        if self.model_name == "sam":
            self._init_sam(model_path)
        elif self.model_name == "sam2":
            self._init_sam2(model_path, model_config, crop_overlap_ratio)
        elif self.model_name == "sam3":
            self._init_sam3(hf_model_id, points_per_batch, crop_overlap_ratio)
        else:
            raise ValueError(
                f"Unknown model_name={self.model_name!r}; expected sam, sam2, or sam3"
            )

        logger.info(
            "MaskGenerator ready: backend=%s device=%s",
            self.model_name,
            device,
        )

    def _init_sam(self, model_path: Optional[str]) -> None:
        from segment_anything import (
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def _init_sam2(
        self,
        model_path: Optional[str],
        model_config: Optional[str],
        crop_overlap_ratio: float,
    ) -> None:
        from hydra.core.global_hydra import GlobalHydra
        import hydra
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        if model_config is None:
            raise ValueError("Model config must be provided for SAM2")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra.initialize(config_path=os.path.dirname(model_config) or ".")
        self.sam = build_sam2(
            os.path.basename(model_config),
            model_path,
            device=device,
            apply_postprocessing=True,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam,
            stability_score_thresh=self.stability_score_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
        )

    def _init_sam3(
        self,
        hf_model_id: str,
        points_per_batch: int,
        crop_overlap_ratio: float,
    ) -> None:
        """Load facebook/sam3 via Hugging Face mask-generation pipeline (GPU)."""
        from transformers import pipeline

        # Prefer HF_TOKEN / HUGGING_FACE_HUB_TOKEN from the environment; the
        # gated facebook/sam3 checkpoint requires an authenticated download.
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or True  # fall back to cached `huggingface-cli login`
        )
        pipe_device = 0 if device.type == "cuda" else -1
        # Must be float32: HF post_process casts boxes to .float() but leaves
        # scores as-is; torchvision NMS then fails if scores are bf16/fp16.
        dtype = torch.float32
        logger.info(
            "Loading SAM3 mask-generation pipeline model=%s device=%s dtype=%s",
            hf_model_id,
            pipe_device,
            dtype,
        )
        self._pipeline = pipeline(
            "mask-generation",
            model=hf_model_id,
            device=pipe_device,
            dtype=dtype,
            token=token,
        )
        # Keep knobs used at generate() time.
        self.points_per_batch = points_per_batch
        self._crop_overlap_ratio = crop_overlap_ratio

    def generate_mask(self, image_bytes, is_tma=False):
        # Read image from BytesIO object
        image_bytes.seek(0)
        file_bytes = np.frombuffer(image_bytes.read(), np.uint8)
        thumb_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if thumb_np is None:
            raise ValueError("Could not decode uploaded PNG thumbnail")
        thumb_np = cv2.cvtColor(thumb_np, cv2.COLOR_BGR2RGB)

        if self.model_name == "sam3":
            masks = self._generate_masks_sam3(thumb_np)
        else:
            masks = self.mask_generator.generate(thumb_np)

        final_mask = self.process_masks(thumb_np, masks, is_tma=is_tma)
        return final_mask

    def _generate_masks_sam3(self, thumb_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Run HF SAM3 automatic mask generation and normalize to SAM dicts."""
        from PIL import Image

        assert self._pipeline is not None
        image = Image.fromarray(thumb_rgb)
        outputs = self._pipeline(
            image,
            points_per_batch=self.points_per_batch,
            crop_overlap_ratio=getattr(self, "_crop_overlap_ratio", 0.4),
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        return self._normalize_hf_masks(outputs)

    @staticmethod
    def _normalize_hf_masks(outputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert HF pipeline output into SAM AutomaticMaskGenerator dicts.

        Expected keys used by ``process_masks``: segmentation, area, bbox,
        predicted_iou.
        """
        raw_masks = outputs.get("masks") or []
        scores = outputs.get("scores")
        if scores is None:
            score_list: list[float] = []
        elif hasattr(scores, "detach"):
            score_list = scores.detach().float().cpu().tolist()
        else:
            score_list = list(scores)

        normalized: list[dict[str, Any]] = []

        for i, mask in enumerate(raw_masks):
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            seg = np.asarray(mask).astype(bool)
            if seg.ndim == 3:
                seg = seg.squeeze()
            if seg.ndim != 2:
                continue
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            score = float(score_list[i]) if i < len(score_list) else 1.0
            normalized.append(
                {
                    "segmentation": seg,
                    "area": int(seg.sum()),
                    "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],  # XYWH
                    "predicted_iou": score,
                }
            )
        return normalized

    def is_circle(self, mask, tolerance_ratio=0.5):
        segmentation = mask["segmentation"]
        y, x = np.nonzero(segmentation)
        centroid = [np.mean(x), np.mean(y)]
        distances = distance.cdist([centroid], list(zip(x, y)), "euclidean")[0]
        average_distance = np.mean(distances)
        tolerance = tolerance_ratio * average_distance
        return np.std(distances) < tolerance

    def estimate_background_color(self, img):
        pixels = img.reshape(-1, img.shape[-1])
        color_counts = Counter(map(tuple, pixels))
        background_color = color_counts.most_common(1)[0][0]
        return np.array(background_color)

    def process_masks(
        self,
        img,
        masks,
        median_ratio=0.7,
        edge_margin=0,
        variance_threshold=100,
        score_threshold=0.8,
        is_tma=False,
    ):
        if len(masks) == 0:
            return None
        areas_median = np.median([mask["area"] for mask in masks])
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        img_height, img_width = img.shape[:2]

        for mask in masks:
            x, y, w, h = map(int, mask["bbox"])
            if mask["predicted_iou"] < score_threshold:
                continue
            if (
                x > edge_margin
                and y > edge_margin
                and x + w < img_width - edge_margin
                and y + h < img_height - edge_margin
                # and abs(mask["area"] - areas_median) < areas_median * median_ratio
                and (not is_tma or self.is_circle(mask))
            ):
                roi = img[y : y + h, x : x + w]
                variance = np.var(roi)
                if variance > variance_threshold:
                    final_mask[np.where(mask["segmentation"] != 0)] = 255
        return final_mask
