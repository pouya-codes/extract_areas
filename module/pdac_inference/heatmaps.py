"""Patch-level heatmap writers: HDF5 scores + PNG overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required for heatmap HDF5 output. pip install h5py") from exc

try:
    import openslide
except ImportError as exc:
    raise ImportError("openslide-python is required.") from exc


def _as_xy_array(coords: Sequence) -> np.ndarray:
    if coords is None or len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64).reshape(-1, 2)


def save_patch_heatmap_h5(
    path: str | Path,
    *,
    slide_id: str,
    coords: Sequence,
    scores: np.ndarray,
    read_size: int,
    patch_size: int,
    score_name: str = "attention",
    tumor_prob: np.ndarray | None = None,
    tissue_coords: Sequence | None = None,
    tissue_tumor_prob: np.ndarray | None = None,
    slide_pred: int | None = None,
    pred_name: str | None = None,
    prob_classical: float | None = None,
    prob_basal: float | None = None,
    sample_id: str | None = None,
) -> Path:
    """Write patch coordinates and prediction scores to an HDF5 file.

    Datasets
    --------
    coords : (N, 2) int64
        Level-0 (x, y) for patches that entered Stage-2 MIL.
    scores / ``score_name`` : (N,) float32
        Per-patch score (default: MIL attention).
    tumor_prob : (N,) float32, optional
        Stage-1 P(tumor) for the same patches.
    tissue_coords / tissue_tumor_prob : optional
        All tissue patches before the tumor filter (Stage-1 map).
    Attributes store slide-level prediction metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coords_arr = _as_xy_array(coords)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if coords_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"coords ({coords_arr.shape[0]}) and scores ({scores_arr.shape[0]}) length mismatch"
        )

    with h5py.File(path, "w") as h5:
        h5.create_dataset("coords", data=coords_arr, compression="gzip")
        h5.create_dataset(score_name, data=scores_arr, compression="gzip")
        # Alias for generic consumers
        if score_name != "scores":
            h5.create_dataset("scores", data=scores_arr, compression="gzip")

        if tumor_prob is not None:
            tp = np.asarray(tumor_prob, dtype=np.float32).reshape(-1)
            if tp.shape[0] == coords_arr.shape[0]:
                h5.create_dataset("tumor_prob", data=tp, compression="gzip")

        if tissue_coords is not None:
            tc = _as_xy_array(tissue_coords)
            h5.create_dataset("tissue_coords", data=tc, compression="gzip")
            if tissue_tumor_prob is not None:
                ttp = np.asarray(tissue_tumor_prob, dtype=np.float32).reshape(-1)
                if ttp.shape[0] == tc.shape[0]:
                    h5.create_dataset("tissue_tumor_prob", data=ttp, compression="gzip")

        h5.attrs["slide_id"] = slide_id
        h5.attrs["read_size"] = int(read_size)
        h5.attrs["patch_size"] = int(patch_size)
        h5.attrs["n_patches"] = int(coords_arr.shape[0])
        h5.attrs["score_name"] = score_name
        if sample_id is not None:
            h5.attrs["sample_id"] = sample_id
        if slide_pred is not None:
            h5.attrs["slide_pred"] = int(slide_pred)
        if pred_name is not None:
            h5.attrs["pred_name"] = pred_name
        if prob_classical is not None:
            h5.attrs["prob_classical"] = float(prob_classical)
        if prob_basal is not None:
            h5.attrs["prob_basal"] = float(prob_basal)

    return path


def _thumbnail_rgba(slide_path: str, max_dim: int = 2048):
    slide = openslide.OpenSlide(slide_path)
    slide_w, slide_h = slide.dimensions
    scale = min(max_dim / slide_w, max_dim / slide_h, 1.0)
    thumb_w = max(1, int(slide_w * scale))
    thumb_h = max(1, int(slide_h * scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGBA")
    slide.close()
    return thumb, slide_w, slide_h, scale


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi > lo:
        return (s - lo) / (hi - lo)
    return np.full_like(s, 0.5)


def render_patch_heatmap_png(
    slide_path: str,
    coords: Sequence,
    scores: np.ndarray,
    read_size: int,
    output_path: str | Path,
    max_dim: int = 2048,
    alpha: int = 150,
) -> Path:
    """Render a patch-score heatmap over a slide thumbnail (blue→red)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords_arr = _as_xy_array(coords)
    scores_n = _normalize_scores(scores)
    if coords_arr.shape[0] == 0:
        # Still write a plain thumbnail so downstream tools find a file
        thumb, *_ = _thumbnail_rgba(slide_path, max_dim=max_dim)
        thumb.convert("RGB").save(output_path)
        return output_path

    thumb, slide_w, slide_h, _ = _thumbnail_rgba(slide_path, max_dim=max_dim)
    thumb_w, thumb_h = thumb.size
    sx = thumb_w / slide_w
    sy = thumb_h / slide_h
    pw = max(1, int(read_size * sx))
    ph = max(1, int(read_size * sy))

    overlay = np.zeros((thumb_h, thumb_w, 4), dtype=np.uint8)
    for (x, y), a in zip(coords_arr, scores_n):
        tx, ty = int(x * sx), int(y * sy)
        r = int(255 * a)
        b = int(255 * (1.0 - a))
        g = int(255 * min(a, 1.0 - a) * 2)
        overlay[ty : min(ty + ph, thumb_h), tx : min(tx + pw, thumb_w)] = [r, g, b, alpha]

    composite = Image.alpha_composite(thumb, Image.fromarray(overlay, "RGBA"))
    composite.convert("RGB").save(output_path)
    return output_path


def write_slide_heatmaps(
    *,
    slide_path: str,
    slide_id: str,
    output_dir: str | Path,
    coords: Sequence,
    attention: np.ndarray,
    read_size: int,
    patch_size: int,
    tumor_prob: np.ndarray | None = None,
    tissue_coords: Sequence | None = None,
    tissue_tumor_prob: np.ndarray | None = None,
    slide_pred: int | None = None,
    pred_name: str | None = None,
    prob_classical: float | None = None,
    prob_basal: float | None = None,
    sample_id: str | None = None,
    max_dim: int = 2048,
) -> dict:
    """Write ``{slide_id}.h5`` and ``{slide_id}_heatmap.png`` under output_dir/heatmaps."""
    out = Path(output_dir)
    h5_dir = out / "heatmaps" / "h5"
    png_dir = out / "heatmaps" / "png"
    h5_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    h5_path = h5_dir / f"{slide_id}.h5"
    png_path = png_dir / f"{slide_id}_heatmap.png"

    save_patch_heatmap_h5(
        h5_path,
        slide_id=slide_id,
        coords=coords,
        scores=attention,
        read_size=read_size,
        patch_size=patch_size,
        score_name="attention",
        tumor_prob=tumor_prob,
        tissue_coords=tissue_coords,
        tissue_tumor_prob=tissue_tumor_prob,
        slide_pred=slide_pred,
        pred_name=pred_name,
        prob_classical=prob_classical,
        prob_basal=prob_basal,
        sample_id=sample_id,
    )
    render_patch_heatmap_png(
        slide_path,
        coords,
        attention,
        read_size,
        png_path,
        max_dim=max_dim,
    )

    # Optional Stage-1 tumor map PNG when tissue-level scores are available
    tumor_png = None
    if tissue_coords is not None and tissue_tumor_prob is not None:
        tumor_png = png_dir / f"{slide_id}_tumor.png"
        render_patch_heatmap_png(
            slide_path,
            tissue_coords,
            tissue_tumor_prob,
            read_size,
            tumor_png,
            max_dim=max_dim,
        )

    return {
        "h5": str(h5_path),
        "png": str(png_path),
        "tumor_png": str(tumor_png) if tumor_png else None,
    }
