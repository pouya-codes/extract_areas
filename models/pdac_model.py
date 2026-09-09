"""
PDAC GigaPath FT model — PurIST Basal vs Classical subtype classification.

Wraps the portable PDAC_GigaPath_1024_FT package (Stage-1 tumor/stroma filter +
Stage-2 Attention MIL) so it can run on annotated regions in the aimviewer /
extract_areas AI processing pipeline.

The region image is tiled into patches (default 1024 px @ 50% stride); patches
outside the annotation mask are skipped. Returns an attention heatmap overlay
plus classical/basal probabilities.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from models.base_model import BaseAIModel

# Vendored inference package lives under module/pdac_inference
_MODULE_DIR = Path(__file__).resolve().parent.parent / "module"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from pdac_inference.models import (  # noqa: E402
    GigaPathFeatureEncoder,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
)
from pdac_inference.wsi import get_eval_transform  # noqa: E402

LABEL_STAGE2 = {0: "classical", 1: "basal"}


class PDACModel(BaseAIModel):
    """Two-stage PDAC PurIST subtype classifier (GigaPath FT, 1024 px)."""

    def __init__(self):
        super().__init__(model_name="pdac", model_version="1.0.0")
        self.description = (
            "PDAC PanSubNet — GigaPath fine-tune (1024px). "
            "Stage 1 filters tumor vs stroma patches; Stage 2 Attention MIL "
            "predicts PurIST subtype (Basal vs Classical)."
        )
        self.requires_mask = True
        self.supported_input_formats = ["PNG", "JPEG", "JPG", "TIFF"]
        self.accepts_mpp = True  # region downsample/mpp can refine patch sizing

        self.hyperparameters = {
            "patch_size": 1024,
            "stride": 512,
            "tumor_threshold": 0.7,
            "tissue_threshold": 0.5,
            "bag_size": 512,
            # 0 = auto: largest batch that fits in VRAM for each stage.
            "batch_size": 0,
            "model_input_size": 224,
            "heatmap_mode": "attention",  # attention | tumor
        }

        self.package_root: Optional[Path] = None
        self.device: Optional[torch.device] = None
        self.stage1_model = None
        self.mil_model = None
        self.feature_encoder = None
        self.transform = None
        self.defaults: Dict[str, Any] = {}
        self._lazy = False
        self._gigapath_on_gpu = None
        self._hf_token = None
        self._stage1_path = None
        self._stage2_path = None
        self._auto_batch_cache: Dict[str, int] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        package_root = config.get("package_root") or config.get("model_dir")
        if not package_root:
            raise ValueError("package_root (PDAC_GigaPath_1024_FT directory) is required")
        self.package_root = Path(package_root)
        stage1 = self.package_root / "stage1" / "best_model.pth"
        stage2 = self.package_root / "stage2" / "best_mil_model.pth"
        if not stage1.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint missing: {stage1}")
        if not stage2.is_file():
            raise FileNotFoundError(f"Stage-2 checkpoint missing: {stage2}")

        defaults_path = self.package_root / "configs" / "inference_defaults.json"
        if defaults_path.is_file():
            with open(defaults_path) as fh:
                self.defaults = json.load(fh)
            for key in (
                "patch_size",
                "stride",
                "tumor_threshold",
                "tissue_threshold",
                "bag_size",
                "batch_size",
                "model_input_size",
            ):
                if key in self.defaults:
                    self.hyperparameters[key] = self.defaults[key]
        # Prefer config; 0 / "auto" means probe the largest batch that fits.
        raw_bs = config.get("batch_size", self.hyperparameters.get("batch_size", 0))
        if isinstance(raw_bs, str) and raw_bs.strip().lower() in ("auto", ""):
            raw_bs = 0
        self.hyperparameters["batch_size"] = int(raw_bs)

        device_name = config.get("device", "auto")
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        hf_token = (
            config.get("hf_token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._lazy = bool(config.get("lazy_load", True))
        self._hf_token = hf_token
        self._stage1_path = stage1
        self._stage2_path = stage2

        if not self._lazy:
            self._load_weights()
        else:
            print(
                f"PDAC model registered (lazy load) from {self.package_root} "
                f"on {self.device}"
            )

        self._is_initialized = True

    def _load_weights(self) -> None:
        """Load MIL always; GigaPath backbones are swapped onto the GPU sequentially
        during ``process`` so a 16 GB T4 can share the card with other models."""
        if self.mil_model is not None and self.transform is not None:
            return
        assert self.device is not None
        cpu = torch.device("cpu")
        print(f"Loading PDAC Stage-1 (CPU) from {self._stage1_path}")
        self.stage1_model = load_stage1_checkpoint(str(self._stage1_path), cpu)
        print(f"Loading PDAC Stage-2 MIL from {self._stage2_path}")
        self.mil_model, _meta = load_stage2_checkpoint(str(self._stage2_path), self.device)
        print("Loading Prov-GigaPath feature encoder (CPU) from Hugging Face...")
        self.feature_encoder = GigaPathFeatureEncoder(hf_token=self._hf_token).to(cpu).eval()
        self.transform = get_eval_transform(
            int(self.hyperparameters.get("model_input_size", 224))
        )
        self._gigapath_on_gpu = None  # "stage1" | "encoder" | None
        print(f"✓ PDAC weights ready (sequential GPU) on {self.device}")

    def _place_gigapath(self, which: str) -> None:
        """Move one GigaPath backbone to GPU and the other back to CPU."""
        assert self.device is not None
        if self._gigapath_on_gpu == which:
            return
        cpu = torch.device("cpu")
        if which == "stage1":
            if self.feature_encoder is not None:
                self.feature_encoder.to(cpu)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.stage1_model.to(self.device).eval()
        elif which == "encoder":
            if self.stage1_model is not None:
                self.stage1_model.to(cpu)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.feature_encoder.to(self.device).eval()
        else:
            raise ValueError(which)
        self._gigapath_on_gpu = which

    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._load_weights()
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to load PDAC weights: {exc}",
                "str_result": f"PDAC load error: {exc}",
                "processed_image": image.convert("RGB"),
                "scores": {},
            }

        params = {**self.hyperparameters, **(hyperparameters or {})}
        # Region read at downsample D means a model patch of P native pixels
        # should be P/D pixels in the region image.
        downsample = float(params.get("downsample") or 1.0)
        if downsample < 1.0:
            downsample = 1.0 / downsample if 0 < downsample < 1 else 1.0
        patch_size = int(params.get("patch_size", 1024))
        stride = int(params.get("stride", 512))
        region_patch = max(32, int(round(patch_size / downsample)))
        region_stride = max(16, int(round(stride / downsample)))

        rgb = image.convert("RGB")
        mask_img = self._resolve_mask(rgb.size, mask, annotation_points)
        if mask_img is None:
            return {
                "success": False,
                "error": "PDAC requires a region mask/annotation",
                "str_result": "Error: mask required",
                "processed_image": rgb,
                "scores": {},
            }

        coords = self._tile_coords(
            rgb.size,
            mask_img,
            region_patch,
            region_stride,
            float(params.get("tissue_threshold", 0.5)),
        )
        if not coords:
            return {
                "success": False,
                "error": "no tissue patches found in region",
                "str_result": "No tissue patches in region",
                "processed_image": rgb,
                "scores": {"n_tissue_patches": 0, "n_tumor_patches": 0},
            }

        try:
            result = self._infer_region(rgb, coords, region_patch, params)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error": str(exc),
                "str_result": f"PDAC error: {exc}",
                "processed_image": rgb,
                "scores": {},
            }

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "str_result": result["error"],
                "processed_image": rgb,
                "scores": {
                    "n_tissue_patches": result.get("n_tissue_patches", 0),
                    "n_tumor_patches": result.get("n_tumor_patches", 0),
                },
            }

        heatmap_mode = str(params.get("heatmap_mode", "attention")).lower()
        overlay = self._render_overlay(
            rgb,
            result["mil_coords"],
            result["attention"]
            if heatmap_mode != "tumor"
            else result["mil_tumor_prob"],
            region_patch,
        )

        pred_name = result["pred_name"]
        p_basal = result["prob_basal"]
        p_classical = result["prob_classical"]
        str_result = (
            f"PDAC {pred_name} · "
            f"basal {p_basal:.0%} / classical {p_classical:.0%} · "
            f"{result['n_tumor_patches']} tumor / {result['n_tissue_patches']} tissue patches"
        )
        return {
            "success": True,
            "processed_image": overlay,
            "scores": {
                "pred": result["pred"],
                "pred_name": pred_name,
                "prob_basal": round(p_basal, 4),
                "prob_classical": round(p_classical, 4),
                "n_tissue_patches": result["n_tissue_patches"],
                "n_tumor_patches": result["n_tumor_patches"],
                "n_mil_patches": result["n_mil_patches"],
                "patch_size_region": region_patch,
                "stride_region": region_stride,
                "downsample": downsample,
                "batch_size_used": result.get("batch_size_used"),
            },
            "str_result": str_result,
            "metadata": {
                "model": self.model_name,
                "version": self.model_version,
                "encoder": "gigapath",
                "package_root": str(self.package_root),
            },
        }

    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        return {
            "patch_size": {
                "type": "int",
                "default": int(self.hyperparameters.get("patch_size", 1024)),
                "min": 256,
                "max": 2048,
                "description": "Native-resolution patch size (px at 20×)",
            },
            "stride": {
                "type": "int",
                "default": int(self.hyperparameters.get("stride", 512)),
                "min": 128,
                "max": 2048,
                "description": "Native-resolution patch stride (px)",
            },
            "tumor_threshold": {
                "type": "float",
                "default": float(self.hyperparameters.get("tumor_threshold", 0.7)),
                "min": 0.0,
                "max": 1.0,
                "description": "Min Stage-1 P(tumor) to keep a patch",
            },
            "tissue_threshold": {
                "type": "float",
                "default": float(self.hyperparameters.get("tissue_threshold", 0.5)),
                "min": 0.0,
                "max": 1.0,
                "description": "Min mask tissue fraction inside a patch",
            },
            "bag_size": {
                "type": "int",
                "default": int(self.hyperparameters.get("bag_size", 512)),
                "min": 16,
                "max": 2048,
                "description": "Max tumor patches for Stage-2 MIL",
            },
            "batch_size": {
                "type": "int",
                "default": int(self.hyperparameters.get("batch_size", 0)),
                "min": 0,
                "max": 128,
                "description": (
                    "GPU batch size for patch encoding. "
                    "0 = auto (largest batch that fits in VRAM)."
                ),
            },
            "heatmap_mode": {
                "type": "choice",
                "default": "attention",
                "choices": ["attention", "tumor"],
                "description": "Overlay MIL attention or Stage-1 tumor probability",
            },
        }

    def cleanup(self) -> None:
        self.stage1_model = None
        self.mil_model = None
        self.feature_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mask(
        size: Tuple[int, int],
        mask: Optional[Image.Image],
        annotation_points: Optional[List[Tuple[float, float]]],
    ) -> Optional[Image.Image]:
        if mask is not None:
            return mask.convert("L").resize(size, Image.NEAREST)
        if annotation_points and len(annotation_points) >= 3:
            from PIL import ImageDraw

            m = Image.new("L", size, 0)
            draw = ImageDraw.Draw(m)
            w, h = size
            pts = [
                (max(0, min(w - 1, float(x))), max(0, min(h - 1, float(y))))
                for x, y in annotation_points
            ]
            draw.polygon(pts, outline=255, fill=255)
            return m
        return None

    @staticmethod
    def _tile_coords(
        size: Tuple[int, int],
        mask: Image.Image,
        patch_size: int,
        stride: int,
        tissue_threshold: float,
    ) -> List[Tuple[int, int]]:
        w, h = size
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.shape[0] != h or mask_arr.shape[1] != w:
            mask_arr = np.asarray(mask.resize((w, h), Image.NEAREST), dtype=np.uint8)

        if w < patch_size or h < patch_size:
            # Single patch covering the region when ROI is smaller than patch.
            if np.mean(mask_arr > 0) >= tissue_threshold:
                return [(0, 0)]
            return []

        coords: List[Tuple[int, int]] = []
        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                region = mask_arr[y0 : y0 + patch_size, x0 : x0 + patch_size]
                if np.mean(region > 0) >= tissue_threshold:
                    coords.append((x0, y0))
        return coords

    def _resolve_batch_size(self, stage: str, sample: torch.Tensor, requested: int) -> int:
        """Return a fixed batch size, or the largest that fits for this stage."""
        if requested and requested > 0:
            return max(1, int(requested))
        if stage in self._auto_batch_cache:
            return self._auto_batch_cache[stage]

        from auto_batch import find_max_batch_size

        assert self.device is not None
        channels, height, width = sample.shape
        model = self.stage1_model if stage == "stage1" else self.feature_encoder
        assert model is not None

        def try_forward(n: int) -> None:
            batch = sample.unsqueeze(0).expand(n, channels, height, width).contiguous()
            batch = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                if stage == "encoder" and self.device.type == "cuda":
                    with torch.autocast("cuda"):
                        model(batch)
                else:
                    model(batch)
            del batch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        best = find_max_batch_size(try_forward, max_size=64, min_size=1)
        self._auto_batch_cache[stage] = best
        print(f"PDAC auto batch_size[{stage}] = {best}")
        return best

    def _infer_region(
        self,
        rgb: Image.Image,
        coords: List[Tuple[int, int]],
        region_patch: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        from progress_report import report_progress

        assert self.device is not None
        assert self.stage1_model is not None
        assert self.feature_encoder is not None
        assert self.mil_model is not None
        assert self.transform is not None

        requested_bs = int(params.get("batch_size", 0) or 0)
        tumor_threshold = float(params.get("tumor_threshold", 0.7))
        model_input = int(params.get("model_input_size", 224))
        # Re-bind transform if input size overridden
        if model_input != int(self.hyperparameters.get("model_input_size", 224)):
            self.transform = get_eval_transform(model_input)

        n_coords = len(coords)
        report_progress(
            progress_stage="tiling",
            progress_label=f"Preparing {n_coords} patches…",
            patch_done=0,
            patch_total=n_coords,
            progress_percent=2,
        )

        # Build all transformed patches once (CPU), then run Stage-1 and the
        # feature encoder in two GPU passes so only one ViT-G is resident.
        tensors = []
        tissue_coords: List[Tuple[int, int]] = []
        for i, (x, y) in enumerate(coords):
            crop = rgb.crop((x, y, x + region_patch, y + region_patch))
            if crop.size != (region_patch, region_patch):
                padded = Image.new("RGB", (region_patch, region_patch), (255, 255, 255))
                padded.paste(crop, (0, 0))
                crop = padded
            crop = crop.resize((model_input, model_input), Image.LANCZOS)
            tensors.append(self.transform(crop))
            tissue_coords.append((x, y))
            if (i + 1) % 32 == 0 or i + 1 == n_coords:
                report_progress(
                    progress_stage="tiling",
                    progress_label=f"Preparing patches {i + 1} / {n_coords}",
                    patch_done=i + 1,
                    patch_total=n_coords,
                    progress_percent=max(2, int(5 * (i + 1) / max(1, n_coords))),
                )
        if not tensors:
            return {
                "error": "no tissue patches found in region",
                "n_tissue_patches": 0,
                "n_tumor_patches": 0,
            }

        n_before = len(tensors)
        tumor_probs: List[torch.Tensor] = []
        keep_flags: List[bool] = []
        self._place_gigapath("stage1")
        batch_size = self._resolve_batch_size("stage1", tensors[0], requested_bs)
        report_progress(
            progress_stage="stage1",
            progress_label=f"Stage-1 tumor filter · 0 / {n_before} patches (batch {batch_size})",
            patch_done=0,
            patch_total=n_before,
            progress_percent=5,
        )
        for start in range(0, n_before, batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                logits = self.stage1_model(batch)
                probs = torch.softmax(logits, dim=1)[:, 0]  # P(tumor)
            tumor_probs.append(probs.cpu())
            keep_flags.extend((probs >= tumor_threshold).cpu().tolist())
            done = min(start + batch_size, n_before)
            # Stage-1 occupies ~5–50% of the bar.
            pct = 5 + int(45 * done / max(1, n_before))
            report_progress(
                progress_stage="stage1",
                progress_label=f"Stage-1 tumor filter · {done} / {n_before} patches",
                patch_done=done,
                patch_total=n_before,
                progress_percent=pct,
            )

        tissue_tp = torch.cat(tumor_probs)
        kept_coords = [c for c, k in zip(tissue_coords, keep_flags) if k]
        kept_tensors = [t for t, k in zip(tensors, keep_flags) if k]
        n_after = len(kept_coords)
        if n_after == 0:
            return {
                "error": "no tumor patches passed stage-1 filter",
                "n_tissue_patches": n_before,
                "n_tumor_patches": 0,
            }

        all_feats = []
        self._place_gigapath("encoder")
        batch_size = self._resolve_batch_size("encoder", kept_tensors[0], requested_bs)
        report_progress(
            progress_stage="encoder",
            progress_label=(
                f"Encoding tumor patches · 0 / {n_after} "
                f"(kept {n_after}/{n_before}, batch {batch_size})"
            ),
            patch_done=0,
            patch_total=n_after,
            progress_percent=50,
        )
        for start in range(0, n_after, batch_size):
            batch = torch.stack(kept_tensors[start : start + batch_size]).to(
                self.device, non_blocking=True
            )
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.autocast("cuda"):
                        feats = self.feature_encoder(batch)
                else:
                    feats = self.feature_encoder(batch)
            all_feats.append(feats.float().cpu())
            done = min(start + batch_size, n_after)
            pct = 50 + int(45 * done / max(1, n_after))
            report_progress(
                progress_stage="encoder",
                progress_label=f"Encoding tumor patches · {done} / {n_after}",
                patch_done=done,
                patch_total=n_after,
                progress_percent=pct,
            )

        feats = torch.cat(all_feats, dim=0)
        # Free encoder VRAM before MIL (tiny) — optional but keeps headroom.
        if self.device.type == "cuda":
            self.feature_encoder.to(torch.device("cpu"))
            self._gigapath_on_gpu = None
            torch.cuda.empty_cache()

        tissue_tp_np = tissue_tp.numpy()
        tissue_index = {tuple(c): i for i, c in enumerate(tissue_coords)}
        kept_tumor_prob = np.array(
            [tissue_tp_np[tissue_index[tuple(c)]] for c in kept_coords],
            dtype=np.float32,
        )

        bag_size = int(params.get("bag_size", 512))
        if feats.size(0) > bag_size:
            idx = torch.randperm(feats.size(0))[:bag_size]
            feats_mil = feats[idx]
            mil_coords = [kept_coords[i] for i in idx.tolist()]
            mil_tumor_prob = kept_tumor_prob[idx.numpy()]
        else:
            feats_mil = feats
            mil_coords = list(kept_coords)
            mil_tumor_prob = kept_tumor_prob

        report_progress(
            progress_stage="mil",
            progress_label=f"MIL subtype · {feats_mil.size(0)} patches",
            patch_done=n_after,
            patch_total=n_after,
            progress_percent=96,
        )

        feats_batch = feats_mil.unsqueeze(0).to(self.device).float()
        mil_mask = torch.ones(1, feats_mil.size(0), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, attn = self.mil_model(feats_batch, mil_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(logits.argmax(1).item())
            attn_np = attn.squeeze(0).cpu().numpy().astype(np.float32)

        report_progress(
            progress_stage="done",
            progress_label="Complete",
            patch_done=n_after,
            patch_total=n_after,
            progress_percent=100,
        )

        return {
            "pred": pred,
            "pred_name": LABEL_STAGE2[pred],
            "prob_classical": float(probs[0]),
            "prob_basal": float(probs[1]),
            "n_tissue_patches": n_before,
            "n_tumor_patches": n_after,
            "n_mil_patches": int(feats_mil.size(0)),
            "mil_coords": mil_coords,
            "attention": attn_np,
            "mil_tumor_prob": mil_tumor_prob,
            "batch_size_used": batch_size,
        }

    @staticmethod
    def _render_overlay(
        rgb: Image.Image,
        coords: List[Tuple[int, int]],
        scores: np.ndarray,
        patch_size: int,
        alpha: int = 140,
    ) -> Image.Image:
        base = rgb.convert("RGBA")
        w, h = base.size
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_arr.size == 0 or not coords:
            return rgb.convert("RGB")
        lo, hi = float(scores_arr.min()), float(scores_arr.max())
        if hi > lo:
            scores_n = (scores_arr - lo) / (hi - lo)
        else:
            scores_n = np.full_like(scores_arr, 0.5)

        for (x, y), a in zip(coords, scores_n):
            r = int(255 * a)
            b = int(255 * (1.0 - a))
            g = int(255 * min(a, 1.0 - a) * 2)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + patch_size), min(h, y + patch_size)
            overlay[y1:y2, x1:x2] = [r, g, b, alpha]

        composite = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
        return composite.convert("RGB")
