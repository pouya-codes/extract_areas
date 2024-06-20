import pyvips
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import glob, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
if torch.cuda.is_available():
    print("CUDA is available")

class MaskGenerator:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.predictor = SamPredictor(self.sam)
        self.sam.to(device = "cuda")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

  
    def generate_mask(self, slide_path, thumbnail_width = 2000):
        thumb = pyvips.Image.thumbnail(slide_path, thumbnail_width)
        thumb = thumb.colourspace('srgb')
        thumb_np = np.ndarray(buffer=thumb.write_to_memory(),
                            dtype=np.uint8,
                            shape=[thumb.height, thumb.width, thumb.bands])[:, :, :3]
        masks = self.mask_generator.generate(thumb_np)
        final_mask = self.process_masks(thumb_np, masks)
        return final_mask, thumb

    
    def is_circle(self, mask, tolerance_ratio=0.5):
        segmentation = mask['segmentation']
        # Get the indices of the non-zero elements
        y, x = np.nonzero(segmentation)
        # Calculate the centroid of the non-zero elements
        centroid = [np.mean(x), np.mean(y)]
        # Calculate the distances from the centroid to all non-zero elements
        distances = distance.cdist([centroid], list(zip(x, y)), 'euclidean')[0]
        average_distance = np.mean(distances)
        tolerance = tolerance_ratio * average_distance  # 5% of the average distance
        # Check if the distances are approximately equal
        # Here, np.std computes the standard deviation, which measures how spread out the distances are
        # If the standard deviation is small, the distances are approximately equal
        is_circle = np.std(distances) < tolerance  # Set tolerance as needed
        return is_circle

    def process_masks(self, img, masks, median_ratio=0.6):
        if len(masks) == 0:
            return
        areas_median = np.median([mask['area'] for mask in masks])
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        for mask in masks:
            if abs(mask['area'] - areas_median) < areas_median * median_ratio: # and self.is_circle(mask):
            # if mask['area'] > 20000 and mask['area'] < 500000:
                final_mask[np.where(mask['segmentation']!=0)] = 255
        return final_mask
        


    

