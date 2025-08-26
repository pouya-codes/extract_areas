# mask_generator.py
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import torch
import numpy as np
from scipy.spatial import distance
from collections import Counter
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import hydra
from hydra.core.global_hydra import GlobalHydra

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class MaskGenerator:
    def __init__(self, model_path, model_config=None):
        self.model_name = "sam"
        if model_config is not None:
            self.model_name = "sam2"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        if self.model_name == "sam":
            self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
            self.predictor = SamPredictor(self.sam)
            self.sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        elif self.model_name == "sam2":
            if model_config is None:
                raise ValueError("Model config must be provided for SAM2")
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            hydra.initialize(config_path=os.path.dirname(model_config))
            self.sam = build_sam2(
                os.path.basename(model_config), model_path, device=device,
                apply_postprocessing=True
            )
            self.mask_generator = SAM2AutomaticMaskGenerator(
                self.sam,
                # points_per_side=64,
                # pred_iou_thresh=0.85,
                stability_score_thresh=0.9,
                # crop_n_layers=1,
                crop_overlap_ratio=0.4,
                # min_mask_region_area=100  # especially important for small tissue
            )

    def generate_mask(self, image_bytes, is_tma=False):
        # Read image from BytesIO object
        image_bytes.seek(0)
        file_bytes = np.frombuffer(image_bytes.read(), np.uint8)
        thumb_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        thumb_np = cv2.cvtColor(thumb_np, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(thumb_np)
        final_mask = self.process_masks(thumb_np, masks, is_tma=is_tma)
        return final_mask

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
        edge_margin=5,
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
                roi = img[y:y+h, x:x+w]
                variance = np.var(roi)
                if variance > variance_threshold:
                    final_mask[np.where(mask["segmentation"] != 0)] = 255
        return final_mask
