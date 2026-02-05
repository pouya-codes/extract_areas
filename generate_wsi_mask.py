#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to generate tissue masks from WSI slides using SAM2 model.

Usage:
    python generate_wsi_mask.py /path/to/slide.svs
    python generate_wsi_mask.py /path/to/slides_folder --batch
    python generate_wsi_mask.py /path/to/slide.svs --output /path/to/output.png
    python generate_wsi_mask.py /path/to/slide.svs --thumbnail-width 3000 --is-tma
"""

import argparse
import os
import sys
import glob
import logging

import pyvips
import cv2
import torch
import numpy as np
from scipy.spatial import distance
from collections import Counter

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# Default paths for SAM2 model (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/sam2/sam2.1_hiera_large.pt")
# Note: SAM2 uses Hydra config - config path relative to sam2 package's configs directory
# For SAM2.1 models, the path must include the subdirectory: configs/sam2.1/
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Supported slide extensions
SLIDE_EXTENSIONS = ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif', '.svslide']


class SAM2MaskGenerator:
    """Generate tissue masks from WSI slides using SAM2 model."""
    
    def __init__(self, model_path, model_config):
        """
        Initialize the SAM2 mask generator.
        
        Parameters
        ----------
        model_path : str
            Path to the SAM2 model checkpoint file.
        model_config : str
            SAM2 model configuration name (e.g., 'sam2.1_hiera_l.yaml').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        logger.info(f"Loading SAM2 model from {model_path}")
        self.sam = build_sam2(
            model_config, model_path, device=device,
            apply_postprocessing=True
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam,
            stability_score_thresh=0.9,
            crop_overlap_ratio=0.4,
        )
        logger.info("SAM2 model loaded successfully")

    def generate_mask(self, slide_path, thumbnail_width=2000, is_tma=False):
        """
        Generate tissue mask from a WSI slide.
        
        Parameters
        ----------
        slide_path : str
            Path to the WSI slide file.
        thumbnail_width : int
            Width of the thumbnail to generate for mask processing.
        is_tma : bool
            Whether the slide is a Tissue Microarray (TMA).
            
        Returns
        -------
        tuple
            (final_mask, thumbnail) - numpy array mask and pyvips thumbnail image
        """
        logger.info(f"Processing slide: {slide_path}")
        
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide not found at {slide_path}")
        
        # Load thumbnail using pyvips
        thumb = pyvips.Image.thumbnail(slide_path, thumbnail_width)
        thumb = thumb.colourspace("srgb")
        thumb_np = np.ndarray(
            buffer=thumb.write_to_memory(),
            dtype=np.uint8,
            shape=[thumb.height, thumb.width, thumb.bands],
        )[:, :, :3]
        
        logger.info(f"Thumbnail size: {thumb_np.shape}")
        
        # Generate masks using SAM2
        logger.info("Generating masks with SAM2...")
        masks = self.mask_generator.generate(thumb_np)
        logger.info(f"Generated {len(masks)} masks")
        
        # Process masks to create final tissue mask
        final_mask = self._process_masks(thumb_np, masks, is_tma=is_tma)
        
        return final_mask, thumb

    def _is_circle(self, mask, tolerance_ratio=0.5):
        """Check if a mask region is approximately circular (for TMA cores)."""
        segmentation = mask["segmentation"]
        y, x = np.nonzero(segmentation)
        if len(x) == 0 or len(y) == 0:
            return False
        centroid = [np.mean(x), np.mean(y)]
        distances = distance.cdist([centroid], list(zip(x, y)), "euclidean")[0]
        average_distance = np.mean(distances)
        tolerance = tolerance_ratio * average_distance
        is_circle = np.std(distances) < tolerance
        return is_circle

    def _process_masks(
        self,
        img,
        masks,
        median_ratio=0.7,
        edge_margin=5,
        variance_threshold=100,
        score_threshold=0.8,
        is_tma=False,
    ):
        """
        Process SAM2 masks to create a final tissue segmentation mask.
        
        Parameters
        ----------
        img : np.ndarray
            Input thumbnail image.
        masks : list
            List of mask dictionaries from SAM2.
        median_ratio : float
            Ratio for filtering masks by area relative to median.
        edge_margin : int
            Margin from image edges to exclude masks.
        variance_threshold : float
            Minimum variance threshold for tissue regions.
        score_threshold : float
            Minimum predicted IoU score for masks.
        is_tma : bool
            Whether to apply circular filtering for TMA cores.
            
        Returns
        -------
        np.ndarray
            Binary mask where 255 indicates tissue regions.
        """
        if len(masks) == 0:
            logger.warning("No masks generated")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        areas_median = np.median([mask["area"] for mask in masks])
        final_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        img_height, img_width = img.shape[:2]

        accepted_count = 0
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
                and (not is_tma or self._is_circle(mask))
            ):
                roi = img[y:y+h, x:x+w]
                variance = np.var(roi)

                if variance > variance_threshold:
                    final_mask[np.where(mask["segmentation"] != 0)] = 255
                    accepted_count += 1
        
        logger.info(f"Accepted {accepted_count} masks out of {len(masks)}")
        return final_mask


def save_mask(mask, output_path):
    """Save mask as PNG image."""
    cv2.imwrite(output_path, mask)
    logger.info(f"Mask saved to: {output_path}")


def save_thumbnail(thumb, output_path):
    """Save pyvips thumbnail as image."""
    thumb.write_to_file(output_path)
    logger.info(f"Thumbnail saved to: {output_path}")


def get_output_path(slide_path, output_dir=None, suffix="_mask"):
    """Generate output path for mask file."""
    base_name = os.path.splitext(os.path.basename(slide_path))[0]
    output_name = f"{base_name}{suffix}.png"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, output_name)
    else:
        return os.path.join(os.path.dirname(slide_path), output_name)


def find_slides(path):
    """Find all slide files in a directory or return single slide path."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        slides = []
        for ext in SLIDE_EXTENSIONS:
            slides.extend(glob.glob(os.path.join(path, f"*{ext}")))
            slides.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        return sorted(slides)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tissue masks from WSI slides using SAM2 model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single slide
    python generate_wsi_mask.py /path/to/slide.svs
    
    # Process with custom output path
    python generate_wsi_mask.py /path/to/slide.svs --output /path/to/mask.png
    
    # Process all slides in a folder
    python generate_wsi_mask.py /path/to/slides_folder --batch --output-dir /path/to/output
    
    # Process TMA slide with larger thumbnail
    python generate_wsi_mask.py /path/to/tma.svs --is-tma --thumbnail-width 3000
    
    # Save thumbnail alongside mask
    python generate_wsi_mask.py /path/to/slide.svs --save-thumbnail
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to WSI slide file or directory containing slides"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for mask (single file mode only)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for masks (batch mode)"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all slides in the input directory"
    )
    parser.add_argument(
        "--thumbnail-width", "-w",
        type=int,
        default=2000,
        help="Width of thumbnail for mask generation (default: 2000)"
    )
    parser.add_argument(
        "--is-tma",
        action="store_true",
        help="Enable TMA (Tissue Microarray) mode - filters for circular regions"
    )
    parser.add_argument(
        "--save-thumbnail",
        action="store_true",
        help="Also save the thumbnail image"
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to SAM2 model checkpoint (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help=f"SAM2 model config name (default: {DEFAULT_MODEL_CONFIG})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find slides to process
    try:
        slides = find_slides(args.input_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not slides:
        logger.error(f"No slide files found at {args.input_path}")
        sys.exit(1)
    
    if not args.batch and len(slides) > 1:
        logger.error("Multiple slides found. Use --batch flag to process all.")
        sys.exit(1)
    
    logger.info(f"Found {len(slides)} slide(s) to process")
    
    # Initialize mask generator
    try:
        generator = SAM2MaskGenerator(args.model_path, args.model_config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize SAM2 model: {e}")
        sys.exit(1)
    
    # Process slides
    success_count = 0
    for slide_path in slides:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {slide_path}")
            
            # Generate mask
            mask, thumb = generator.generate_mask(
                slide_path,
                thumbnail_width=args.thumbnail_width,
                is_tma=args.is_tma
            )
            
            if mask is None:
                logger.warning(f"No mask generated for {slide_path}")
                continue
            
            # Determine output path
            if args.output and not args.batch:
                mask_path = args.output
            else:
                mask_path = get_output_path(slide_path, args.output_dir, "_mask")
            
            # Save mask
            save_mask(mask, mask_path)
            
            # Optionally save thumbnail
            if args.save_thumbnail:
                thumb_path = get_output_path(slide_path, args.output_dir, "_thumbnail")
                save_thumbnail(thumb, thumb_path)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {slide_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete: {success_count}/{len(slides)} slides successful")


if __name__ == "__main__":
    main()
