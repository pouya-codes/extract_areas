import pyvips
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import Counter
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


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
            self.sam.to(device="cuda")
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        elif self.model_name == "sam2":
            if model_config is None:
                raise ValueError("Model config must be provided for SAM2")
            self.sam = build_sam2(
                model_config, model_path, device=device,
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

    def generate_mask(self, slide_path, thumbnail_width=2000, is_tma=False):
        thumb = pyvips.Image.thumbnail(slide_path, thumbnail_width)
        thumb = thumb.colourspace("srgb")
        thumb_np = np.ndarray(
            buffer=thumb.write_to_memory(),
            dtype=np.uint8,
            shape=[thumb.height, thumb.width, thumb.bands],
        )[:, :, :3]
        masks = self.mask_generator.generate(thumb_np)
        final_mask = self.process_masks(thumb_np, masks, is_tma=is_tma)
        return final_mask, thumb

    def is_circle(self, mask, tolerance_ratio=0.5):
        segmentation = mask["segmentation"]
        # Get the indices of the non-zero elements
        y, x = np.nonzero(segmentation)
        # Calculate the centroid of the non-zero elements
        centroid = [np.mean(x), np.mean(y)]
        # Calculate the distances from the centroid to all non-zero elements
        distances = distance.cdist([centroid], list(zip(x, y)), "euclidean")[0]
        average_distance = np.mean(distances)
        tolerance = tolerance_ratio * average_distance  # 5% of the average distance
        # Check if the distances are approximately equal
        # Here, np.std computes the standard deviation, which measures how spread out the distances are
        # If the standard deviation is small, the distances are approximately equal
        is_circle = np.std(distances) < tolerance  # Set tolerance as needed
        return is_circle

    def estimate_background_color(self, img):
        # Reshape the image to a list of pixels
        pixels = img.reshape(-1, img.shape[-1])
        # Count the frequency of each color
        color_counts = Counter(map(tuple, pixels))
        # Get the most common color
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
            return
        areas_median = np.median([mask["area"] for mask in masks])
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        img_height, img_width = img.shape[:2]

        # background_color = self.estimate_background_color(img)

        for mask in masks:
            x, y, w, h = map(int, mask["bbox"])
            if mask["predicted_iou"] < score_threshold:
                continue
            # Check if the mask is within the image bounds with a margin
            if (
                x > edge_margin
                and y > edge_margin
                and x + w < img_width - edge_margin
                and y + h < img_height - edge_margin
                # and abs(mask["area"] - areas_median) < areas_median * median_ratio
                and (not is_tma or self.is_circle(mask))
            ):
                roi = img[y:y+h, x:x+w]
                # avg_color = np.mean(roi, axis=(0, 1))
                variance = np.var(roi)

                # if not np.allclose(avg_color, background_color, atol=300):
                if variance > variance_threshold:
                    # if mask['area'] > 20000 and mask['area'] < 500000:
                    final_mask[np.where(mask["segmentation"] != 0)] = 255
        return final_mask
