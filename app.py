# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import os
import sys
sys.path.append("module/DeepLiff")
import numpy as np
import tempfile
import base64
import json
from mask_generator import MaskGenerator
from config import config
from pathlib import Path
from src.utils import extract_regions_from_mask
from io import BytesIO
from src.process_file import ImageProcessor
from PIL import Image, ImageDraw


def get_absolute_path(relative_path):
    """Helper function to get absolute path from relative path."""
    return str(Path(relative_path).absolute())

def init_image_processor(
    model_dir, tile_size=256, post_processing=True, gpu_ids=[]
    ):
    return ImageProcessor(
            model_dir, tile_size, post_processing, gpu_ids
        )

def init_mask_generator():
    if config["default_model"] == "sam":
        return MaskGenerator(model_path=config["sam_model_path"])
    elif config["default_model"] == "sam2":
        return MaskGenerator(
            model_path=get_absolute_path(config["sam2_model_path"]),
            model_config=config["sam2_config_path"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['default_model']}")

app = FastAPI()
# Initialize the image processor
model_dir = get_absolute_path(config["deepliif_model_path"])
image_processor = init_image_processor(model_dir)
# Initialize mask generator
mask_gen = init_mask_generator()

@app.post("/generate_mask")
async def generate_mask_api(
    file: UploadFile = File(...),
    is_tma: bool = Form(False)
    ):
    # Check if it's a PNG file (either by filename or content type)
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
        # If no filename provided, assume it's PNG data (common with BytesIO uploads)
        is_png = True
    
    if not is_png:
        return JSONResponse(
            {"status": "error", "message": "Only PNG files are supported."},
            status_code=400
        )

    # Process uploaded PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        image_bytes = BytesIO(contents)

    # Generate mask
    final_mask = mask_gen.generate_mask(image_bytes, is_tma=is_tma)

    if final_mask is None:
        return JSONResponse({"status": "no_masks_found"})

    # Convert mask to base64 PNG
    _, mask_png = cv2.imencode(".png", final_mask.astype(np.uint8))
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    return {
        "status": "success",
        "is_tma": is_tma,
        "mask_base64": mask_b64,
        "mask_shape": final_mask.shape
    }

# get regions from the mask
# input: mask image, slide width, slide height
# output: list of regions with coordinates and properties
@app.post("/extract_regions")
async def extract_regions_api(
    file: UploadFile = File(...),
    slide_width: int = Form(...),
    slide_height: int = Form(...),
):
    # Check if it's a PNG file (either by filename or content type)
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
        # If no filename provided, assume it's PNG data (common with BytesIO uploads)
        is_png = True
    
    if not is_png:
        return JSONResponse(
            {"status": "error", "message": "Only PNG files are supported."},
            status_code=400
        )

    # Process uploaded mask PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        image_bytes = BytesIO(contents)
    # Extract regions from the mask
    regions = extract_regions_from_mask(image_bytes, (slide_width, slide_height))

    return {
        "status": "success",
        "regions": regions
    }

# Process a region with a mask
# input: region image, mask image
# return: region overlaid with activation map
@app.post("/process_region")
async def process_region_api(
    region: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    # Check if both files are PNG
    is_region_png = region.filename.lower().endswith(".png") or region.content_type == "image/png"
    is_mask_png = mask.filename.lower().endswith(".png") or mask.content_type == "image/png"

    if not is_region_png or not is_mask_png:
        return JSONResponse(
            {"status": "error", "message": "Both files must be PNG."},
            status_code=400
        )

    # Read the region and mask images
    region_bytes = await region.read()
    mask_bytes = await mask.read()
    # covert region_bytes to pil image
    region_image = Image.open(BytesIO(region_bytes)) 
    mask_image = Image.open(BytesIO(mask_bytes))
    # Apply mask (white=keep, black=white out)
    white_bg = Image.new("RGB", region_image.size, (255, 255, 255))
    masked_region = Image.composite(region_image, white_bg, mask_image)
    # Process the region with the mask
    processed_image, score = image_processor.test_img(
        masked_region,
        eager_mode=False,  # non-eager mode enables DP across all available GPUs
        color_dapi=False,
        color_marker=False,
        tissue_mask=mask_image,
    )
    if "cell_coords" in score:
        del score["cell_coords"]
    overlay_image = processed_image["SegRefined"]
    # white out the masked region
    white_bg = Image.new("RGB", overlay_image.size, (255, 255, 255))
    masked_result = Image.composite(overlay_image, white_bg, mask_image)
    # Convert processed image to base64
    buffer = BytesIO()
    masked_result.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # _, processed_png = cv2.imencode(".png", overlay_image)
    # processed_b64 = base64.b64encode(processed_png).decode("utf-8")

    return {
        "status": "success",
        "processed_image_base64": img_base64,
        "score": score
    }


# Process a region with a annotation
# input: region image, annoatation str
# return: region overlaid with activation map
@app.post("/process_region_annotation")
async def process_region_annotation_api(
    region: UploadFile = File(...),
    mask: str = Form(...),
    region_id: str = Form("")
):
        # Validate region file is JPEG
    region_ct = (region.content_type or "").lower()
    region_fn = (region.filename or "").lower()
    is_region_jpeg = region_ct in {"image/jpeg", "image/jpg"} or region_fn.endswith((".jpg", ".jpeg"))

    if not is_region_jpeg:
        return JSONResponse(
            {"status": "error", "message": "Region file must be JPEG (.jpg/.jpeg)."},
            status_code=400
        )

    # Read the region image
    region_bytes = await region.read()
    region_image = Image.open(BytesIO(region_bytes)).convert("RGB")

    # Parse JSON mask points from the 'mask' form field
    try:
        points_payload = json.loads(mask)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON in 'mask' field."},
            status_code=400
        )

    # Normalize to list of (x, y)
    def to_points(obj):
        pts = []
        if isinstance(obj, dict) and "points" in obj:
            obj = obj["points"]
        if isinstance(obj, list):
            for p in obj:
                if isinstance(p, dict):
                    x = p.get("x", p.get("X"))
                    y = p.get("y", p.get("Y"))
                    if x is None or y is None:
                        continue
                    pts.append((int(round(float(x))), int(round(float(y)))))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((int(round(float(p[0]))), int(round(float(p[1])))))
        return pts

    polygon_points = to_points(points_payload)
    if len(polygon_points) < 3:
        return JSONResponse(
            {"status": "error", "message": "'mask' must contain at least three points."},
            status_code=400
        )

    # Build a binary mask from polygon points
    w, h = region_image.size
    mask_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_image)
    clamped = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in polygon_points]
    draw.polygon(clamped, outline=255, fill=255)

    # Apply mask (white=keep, black=white out)
    white_bg = Image.new("RGB", region_image.size, (255, 255, 255))
    masked_region = Image.composite(region_image, white_bg, mask_image)
    # Process the region with the mask
    processed_image, score = image_processor.test_img(
        masked_region,
        eager_mode=False,  # non-eager mode enables DP across all available GPUs
        color_dapi=False,
        color_marker=False,
        tissue_mask=mask_image,
    )
    # save mask and region to disk for testing
    mask_image.save(f"mask_{region_id}.png")
    masked_region.save(f"region_{region_id}.png")
    if "cell_coords" in score:
        del score["cell_coords"]
    overlay_image = processed_image["SegRefined"]
    # white out the masked region
    white_bg = Image.new("RGB", overlay_image.size, (255, 255, 255))
    masked_result = Image.composite(overlay_image, white_bg, mask_image)
    # Convert processed image to base64
    buffer = BytesIO()
    masked_result.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # _, processed_png = cv2.imencode(".png", overlay_image)
    # processed_b64 = base64.b64encode(processed_png).decode("utf-8")

    return {
        "status": "success",
        "processed_image_base64": img_base64,
    "score": score,
    "region_id": region_id,
    }