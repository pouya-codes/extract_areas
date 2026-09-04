"""WSI patch extraction and overlay utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import openslide
except ImportError as exc:
    raise ImportError("openslide-python is required. Install openslide system library first.") from exc


def get_eval_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_slide_magnification(slide) -> float:
    for key in (openslide.PROPERTY_NAME_OBJECTIVE_POWER, "aperio.AppMag"):
        val = slide.properties.get(key)
        if val is not None:
            return float(val)
    return 40.0


def find_mask_for_slide(slide_path: str, masks_dir: str) -> str | None:
    """Locate a tissue mask for a slide (flat or residual HistoQC naming)."""
    stem = Path(slide_path).stem
    name = Path(slide_path).name  # e.g. slide.svs
    masks = Path(masks_dir)
    candidates = [
        masks / f"{stem}_mask.png",
        masks / f"{stem}.png",
        masks / f"{stem}_mask.jpg",
        masks / f"{stem}.jpg",
        # HistoQC-style leftovers (after prepare-masks these should be rare)
        masks / f"{name}_mask_use.png",
        masks / f"{stem}_mask_use.png",
        masks / name / f"{name}_mask_use.png",
        masks / name / f"{stem}_mask_use.png",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def extract_patch_coords_from_mask(
    slide_path: str,
    mask_path: str,
    patch_size: int = 1024,
    stride: int = 512,
    target_mag: float = 20.0,
    tissue_threshold: float = 0.5,
):
    slide = openslide.OpenSlide(slide_path)
    slide_w, slide_h = slide.dimensions
    native_mag = get_slide_magnification(slide)
    scale = native_mag / target_mag
    read_size = int(patch_size * scale)

    mask_arr = np.array(Image.open(mask_path).convert("L"))
    mask_h, mask_w = mask_arr.shape
    sx = slide_w / mask_w
    sy = slide_h / mask_h
    stride_level0 = int(stride * scale)
    coords = []

    for y0 in range(0, slide_h - read_size + 1, stride_level0):
        for x0 in range(0, slide_w - read_size + 1, stride_level0):
            mx_start = int(x0 / sx)
            my_start = int(y0 / sy)
            mx_end = min(int((x0 + read_size) / sx) + 1, mask_w)
            my_end = min(int((y0 + read_size) / sy) + 1, mask_h)
            if mx_end <= mx_start or my_end <= my_start:
                continue
            region = mask_arr[my_start:my_end, mx_start:mx_end]
            if np.mean(region > 0) >= tissue_threshold:
                coords.append((x0, y0))

    slide.close()
    return coords, read_size


def extract_slide_features(
    slide_path: str,
    coords: list,
    read_size: int,
    patch_size: int,
    feature_encoder,
    device: torch.device,
    transform,
    batch_size: int = 64,
    use_amp: bool = True,
    stage1_model=None,
    tumor_threshold: float = 0.7,
):
    slide = openslide.OpenSlide(slide_path)
    all_feats = []
    n_before = 0
    n_after = 0
    tissue_coords = []
    tumor_probs = []
    kept_coords = []

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        imgs = []
        valid_coords = []
        for x, y in batch_coords:
            try:
                region = slide.read_region((x, y), 0, (read_size, read_size)).convert("RGB")
                region = region.resize((patch_size, patch_size), Image.LANCZOS)
                imgs.append(transform(region))
                valid_coords.append((x, y))
            except Exception:
                continue
        if not imgs:
            continue

        batch = torch.stack(imgs).to(device, non_blocking=True)
        n_before += batch.size(0)
        tissue_coords.extend(valid_coords)

        with torch.no_grad():
            if stage1_model is not None:
                logits = stage1_model(batch)
                probs = torch.softmax(logits, dim=1)[:, 0]
                tumor_probs.append(probs.cpu())
                keep = probs >= tumor_threshold
                if not keep.any():
                    continue
                batch = batch[keep]
                batch_kept = [c for c, k in zip(valid_coords, keep.cpu().tolist()) if k]
            else:
                tumor_probs.append(torch.ones(batch.size(0)))
                batch_kept = valid_coords

            n_after += batch.size(0)
            kept_coords.extend(batch_kept)
            if use_amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    feats = feature_encoder(batch)
            else:
                feats = feature_encoder(batch)
        all_feats.append(feats.cpu())

    slide.close()
    tumor_probs_cat = torch.cat(tumor_probs) if tumor_probs else torch.tensor([])
    if not all_feats:
        return {
            "feats": None,
            "n_before": n_before,
            "n_after": n_after,
            "tissue_coords": tissue_coords,
            "tumor_probs": tumor_probs_cat,
            "kept_coords": [],
        }
    return {
        "feats": torch.cat(all_feats, dim=0),
        "n_before": n_before,
        "n_after": n_after,
        "tissue_coords": tissue_coords,
        "tumor_probs": tumor_probs_cat,
        "kept_coords": kept_coords,
    }
