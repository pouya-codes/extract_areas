"""End-to-end inference pipeline for PDAC subtype prediction."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .heatmaps import write_slide_heatmaps
from .models import (
    GigaPathFeatureEncoder,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
)
from .wsi import (
    extract_patch_coords_from_mask,
    extract_slide_features,
    find_mask_for_slide,
    get_eval_transform,
)


LABEL_STAGE1 = {0: "tumor", 1: "stroma"}
LABEL_STAGE2 = {0: "classical", 1: "basal"}


def load_optional_labels(labels_path: str | None) -> dict[str, int]:
    """Load optional labels TSV/CSV with columns: sample_id (or sample), purist_call."""
    if not labels_path:
        return {}
    labels = {}
    with open(labels_path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            sid = (row.get("sample_id") or row.get("sample") or "").strip()
            call = (row.get("purist_call") or row.get("label") or "").strip().lower()
            if sid and call in ("classical", "basal"):
                labels[sid] = 0 if call == "classical" else 1
    return labels


def extract_sample_id(slide_filename: str) -> str:
    """CPTAC-style: C3L-00017-21.svs -> C3L-00017. Otherwise use stem."""
    stem = Path(slide_filename).stem
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0].startswith("C3"):
        return "-".join(parts[:2])
    return stem


class PDACInferencePipeline:
    def __init__(
        self,
        package_root: str | Path,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        self.package_root = Path(package_root)
        self.stage1_ckpt = self.package_root / "stage1" / "best_model.pth"
        self.stage2_ckpt = self.package_root / "stage2" / "best_mil_model.pth"
        self.defaults = self._load_defaults()

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading Stage-1 checkpoint (offline): {self.stage1_ckpt}")
        self.stage1_model = load_stage1_checkpoint(str(self.stage1_ckpt), self.device)
        print(f"Loading Stage-2 MIL checkpoint: {self.stage2_ckpt}")
        self.mil_model, self.stage2_meta = load_stage2_checkpoint(str(self.stage2_ckpt), self.device)
        print("Loading GigaPath feature encoder (HuggingFace prov-gigapath/prov-gigapath)...")
        self.feature_encoder = GigaPathFeatureEncoder(hf_token=hf_token).to(self.device).eval()
        self.transform = get_eval_transform(self.defaults.get("model_input_size", 224))
        print(f"Ready on {self.device}")

    def _load_defaults(self) -> dict:
        cfg = self.package_root / "configs" / "inference_defaults.json"
        if cfg.is_file():
            with open(cfg) as fh:
                return json.load(fh)
        return {}

    def discover_slides(self, slides_dir: str, masks_dir: str, labels_path: str | None = None):
        labels = load_optional_labels(labels_path)
        entries = []
        slides = sorted(
            f for f in os.listdir(slides_dir)
            if f.lower().endswith((".svs", ".tif", ".tiff", ".ndpi"))
        )
        for sf in slides:
            slide_path = os.path.join(slides_dir, sf)
            mask_path = find_mask_for_slide(slide_path, masks_dir)
            if mask_path is None:
                continue
            sample_id = extract_sample_id(sf)
            entry = {
                "slide_file": sf,
                "slide_path": slide_path,
                "mask_path": mask_path,
                "slide_id": Path(sf).stem,
                "sample_id": sample_id,
            }
            if sample_id in labels:
                entry["label"] = labels[sample_id]
                entry["label_name"] = LABEL_STAGE2[labels[sample_id]]
            entries.append(entry)
        return entries

    def predict_slide(
        self,
        slide_path: str,
        mask_path: str,
        slide_id: str | None = None,
        sample_id: str | None = None,
        output_dir: str | None = None,
        save_heatmaps: bool = True,
        **kwargs,
    ) -> dict:
        cfg = {**self.defaults, **kwargs}
        coords, read_size = extract_patch_coords_from_mask(
            slide_path,
            mask_path,
            patch_size=cfg.get("patch_size", 1024),
            stride=cfg.get("stride", 512),
            target_mag=cfg.get("magnification", 20.0),
            tissue_threshold=cfg.get("tissue_threshold", 0.5),
        )
        if not coords:
            return {"error": "no tissue patches found", "slide_path": slide_path}

        result = extract_slide_features(
            slide_path,
            coords,
            read_size,
            cfg.get("patch_size", 1024),
            self.feature_encoder,
            self.device,
            self.transform,
            batch_size=cfg.get("batch_size", 64),
            stage1_model=self.stage1_model,
            tumor_threshold=cfg.get("tumor_threshold", 0.7),
        )
        feats = result["feats"]
        kept_coords = result["kept_coords"]
        tissue_coords = result["tissue_coords"]
        tissue_tumor_prob = result["tumor_probs"]

        if feats is None or feats.size(0) == 0:
            return {
                "error": "no tumor patches passed stage-1 filter",
                "slide_path": slide_path,
                "n_tissue_patches": result["n_before"],
            }

        # Align Stage-1 tumor probs with kept (tumor) patches
        if torch.is_tensor(tissue_tumor_prob) and tissue_tumor_prob.numel() == len(tissue_coords):
            tissue_tp_np = tissue_tumor_prob.numpy()
            # Map kept coords back to tissue tumor probs
            tissue_index = {tuple(c): i for i, c in enumerate(tissue_coords)}
            kept_tumor_prob = np.array(
                [tissue_tp_np[tissue_index[tuple(c)]] for c in kept_coords],
                dtype=np.float32,
            )
        else:
            tissue_tp_np = (
                tissue_tumor_prob.numpy()
                if torch.is_tensor(tissue_tumor_prob)
                else np.asarray(tissue_tumor_prob, dtype=np.float32)
            )
            kept_tumor_prob = np.ones(len(kept_coords), dtype=np.float32)

        bag_size = cfg.get("bag_size", 512)
        if feats.size(0) > bag_size:
            idx = torch.randperm(feats.size(0))[:bag_size]
            feats_mil = feats[idx]
            mil_coords = [kept_coords[i] for i in idx.tolist()]
            mil_tumor_prob = kept_tumor_prob[idx.numpy()]
        else:
            feats_mil = feats
            mil_coords = list(kept_coords)
            mil_tumor_prob = kept_tumor_prob

        feats_batch = feats_mil.unsqueeze(0).to(self.device).float()
        mask = torch.ones(1, feats_mil.size(0), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, attn = self.mil_model(feats_batch, mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(logits.argmax(1).item())
            attn_np = attn.squeeze(0).cpu().numpy().astype(np.float32)

        sid = slide_id or Path(slide_path).stem
        out = {
            "slide_path": slide_path,
            "slide_id": sid,
            "sample_id": sample_id or sid,
            "pred": pred,
            "pred_name": LABEL_STAGE2[pred],
            "prob_classical": float(probs[0]),
            "prob_basal": float(probs[1]),
            "n_tissue_patches": result["n_before"],
            "n_tumor_patches": result["n_after"],
            "n_mil_patches": int(feats_mil.size(0)),
            "attention": attn_np.tolist(),
        }

        if save_heatmaps and output_dir:
            try:
                hm = write_slide_heatmaps(
                    slide_path=slide_path,
                    slide_id=sid,
                    output_dir=output_dir,
                    coords=mil_coords,
                    attention=attn_np,
                    read_size=read_size,
                    patch_size=cfg.get("patch_size", 1024),
                    tumor_prob=mil_tumor_prob,
                    tissue_coords=tissue_coords,
                    tissue_tumor_prob=tissue_tp_np,
                    slide_pred=pred,
                    pred_name=LABEL_STAGE2[pred],
                    prob_classical=float(probs[0]),
                    prob_basal=float(probs[1]),
                    sample_id=out["sample_id"],
                )
                out["heatmap_h5"] = hm["h5"]
                out["heatmap_png"] = hm["png"]
                if hm.get("tumor_png"):
                    out["tumor_heatmap_png"] = hm["tumor_png"]
            except Exception as exc:
                out["heatmap_error"] = str(exc)
                print(f"  WARNING: heatmap failed for {sid}: {exc}")

        return out

    def run_cohort(
        self,
        slides_dir: str,
        masks_dir: str,
        output_dir: str,
        labels_path: str | None = None,
        save_heatmaps: bool = True,
        **kwargs,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)

        entries = self.discover_slides(slides_dir, masks_dir, labels_path)
        if not entries:
            raise RuntimeError("No slides matched. Provide .svs slides and matching tissue masks.")

        cfg = {**self.defaults, **kwargs}
        predictions = []
        t0 = time.time()

        for i, entry in enumerate(entries):
            pred = self.predict_slide(
                entry["slide_path"],
                entry["mask_path"],
                slide_id=entry["slide_id"],
                sample_id=entry["sample_id"],
                output_dir=output_dir,
                save_heatmaps=save_heatmaps,
                **cfg,
            )
            if "label" in entry:
                pred["label"] = entry["label"]
                pred["label_name_true"] = entry["label_name"]
                pred["correct"] = pred.get("pred") == entry["label"]
            # Drop bulky attention arrays from the JSON summary
            summary = {k: v for k, v in pred.items() if k != "attention"}
            predictions.append(summary)
            status = pred.get("pred_name", pred.get("error", "?"))
            hm = " +heatmap" if pred.get("heatmap_h5") else ""
            print(f"[{i+1}/{len(entries)}] {entry['slide_id']} -> {status}{hm}")

        out = {
            "model": "GigaPath FT 1024",
            "n_slides": len(predictions),
            "predictions": predictions,
            "elapsed_s": round(time.time() - t0, 1),
            "heatmaps_enabled": bool(save_heatmaps),
        }

        with open(os.path.join(output_dir, "slide_predictions.json"), "w") as fh:
            json.dump(predictions, fh, indent=2)

        labeled = [p for p in predictions if "label" in p and "error" not in p]
        if labeled:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

            y_true = [p["label"] for p in labeled]
            y_pred = [p["pred"] for p in labeled]
            y_prob = [[p["prob_classical"], p["prob_basal"]] for p in labeled]
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "auc_roc": float(roc_auc_score(y_true, np.array(y_prob)[:, 1])),
                "n_evaluated": len(labeled),
            }
            out["metrics"] = metrics
            with open(os.path.join(output_dir, "eval_metrics.json"), "w") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"Metrics: {metrics}")

        with open(os.path.join(output_dir, "run_summary.json"), "w") as fh:
            json.dump(out, fh, indent=2)

        if save_heatmaps:
            print(f"Heatmaps → {os.path.join(output_dir, 'heatmaps')}/{{h5,png}}/")
        return out
