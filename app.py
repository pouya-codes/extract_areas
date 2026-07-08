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
    white_bg = Image.new("RGB", region_image.size, (240, 240, 240))
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
    # save mask, region and overlay to disk for testing
    # masked_result.save(f"tests/overlay_{region_id}.png")
    # mask_image.save(f"tests/mask_{region_id}.png")
    # masked_region.save(f"tests/region_{region_id}.png")
    
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


@app.post("/auto_name_regions")
async def auto_name_regions_api(request: dict):
    """Auto-name regions using intelligent spatial ordering in snake pattern.
    
    This endpoint receives region data and returns an intelligent ordering that:
    1. Analyzes spatial distribution of regions
    2. Detects slide rotation and orientation
    3. Orders regions in snake pattern (left-right, then right-left alternating)
    4. Handles missing regions and irregular layouts
    
    Input:
    - slide_width: Width of the slide in pixels
    - slide_height: Height of the slide in pixels  
    - regions: List of region objects with center coordinates and bounding boxes
    
    Returns:
    - success: Boolean indicating if ordering was successful
    - ordered_indices: List of original indices in the new snake pattern order
    - metadata: Additional information about the ordering process
    """
    try:
        slide_width = request.get('slide_width')
        slide_height = request.get('slide_height')
        regions = request.get('regions', [])
        
        if not slide_width or not slide_height:
            return JSONResponse({
                "success": False,
                "error": "slide_width and slide_height are required"
            }, status_code=400)
        
        if not regions:
            return {
                "success": True,
                "ordered_indices": [],
                "metadata": {"message": "No regions to order"}
            }
        
        # Convert to numpy arrays for easier processing
        centers = np.array([[r['center_x'], r['center_y']] for r in regions])
        original_indices = [r['original_index'] for r in regions]
        
        # Detect slide orientation and rotation
        orientation_info = detect_slide_orientation(centers, slide_width, slide_height)
        
        # Apply intelligent snake pattern ordering
        if orientation_info['is_grid_like']:
            ordered_indices = grid_snake_ordering(
                centers, original_indices, orientation_info
            )
        else:
            # Fallback to simple spatial ordering for irregular layouts
            ordered_indices = simple_spatial_ordering(centers, original_indices)
        
        # Convert numpy types to native Python types for JSON serialization
        safe_ordered_indices = [int(idx) for idx in ordered_indices]
        safe_orientation_info = {}
        for key, value in orientation_info.items():
            if hasattr(value, 'item'):  # numpy scalar
                safe_orientation_info[key] = value.item()
            elif isinstance(value, np.bool_):
                safe_orientation_info[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                safe_orientation_info[key] = float(value)
            else:
                safe_orientation_info[key] = value
        print({
            "success": True,
            "ordered_indices": safe_ordered_indices,
            "metadata": {
                "total_regions": len(regions),
                "orientation": safe_orientation_info,
                "ordering_method": "grid_snake" if orientation_info['is_grid_like'] else "spatial_fallback"
            }
        })
        return {
            "success": True,
            "ordered_indices": safe_ordered_indices,
            "metadata": {
                "total_regions": len(regions),
                "orientation": safe_orientation_info,
                "ordering_method": "grid_snake" if orientation_info['is_grid_like'] else "spatial_fallback"
            }
        }
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Internal error: {str(e)}"
        }, status_code=500)


def detect_slide_orientation(centers, slide_width, slide_height):
    """Detect if regions form a grid pattern and determine slide orientation."""
    try:
        if len(centers) < 4:
            return {
                'is_grid_like': False,
                'rotation_angle': 0,
                'rows': 1,
                'cols': len(centers)
            }
        
        # Calculate distances between all pairs of points
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(centers))
        
        # Find the most common distances (indicating grid spacing)
        non_zero_distances = distances[distances > 0]
        
        if len(non_zero_distances) == 0:
            return {'is_grid_like': False, 'rotation_angle': 0, 'rows': 1, 'cols': len(centers)}
        
        # Use histogram to find common distances
        hist, bins = np.histogram(non_zero_distances, bins=20)
        most_common_distance = bins[np.argmax(hist)]
        
        # Tolerance for grid detection
        tolerance = most_common_distance * 0.3
        
        # Count how many point pairs have approximately the common distance
        grid_connections = np.sum((np.abs(distances - most_common_distance) < tolerance) & (distances > 0))
        total_possible_connections = len(centers) * (len(centers) - 1) / 2
        
        # If a significant portion of connections match the grid pattern
        grid_ratio = grid_connections / total_possible_connections
        is_grid_like = grid_ratio > 0.3  # At least 30% of connections suggest grid
        
        # Estimate grid dimensions
        if is_grid_like:
            # Sort by Y coordinate to find rows
            y_sorted_indices = np.argsort(centers[:, 1])
            y_coords = centers[y_sorted_indices, 1]
            
            # Find row breaks (significant Y jumps)
            y_diffs = np.diff(y_coords)
            row_threshold = np.median(y_diffs) * 1.5 if len(y_diffs) > 0 else 0
            
            rows = 1 + np.sum(y_diffs > row_threshold) if row_threshold > 0 else 1
            cols = len(centers) // rows if rows > 0 else len(centers)
            
            # Estimate rotation by looking at the angle of the most common vector
            vectors = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if abs(dist - most_common_distance) < tolerance:
                        vectors.append(centers[j] - centers[i])
            
            if vectors:
                # Find the dominant direction
                vectors = np.array(vectors)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                # Normalize angles to [0, π/2] since we care about grid alignment
                angles = np.abs(angles) % (np.pi / 2)
                rotation_angle = np.median(angles)
            else:
                rotation_angle = 0
        else:
            rows = 1
            cols = len(centers)
            rotation_angle = 0
        
        return {
            'is_grid_like': is_grid_like,
            'rotation_angle': rotation_angle,
            'rows': max(1, rows),
            'cols': max(1, cols),
            'grid_ratio': grid_ratio,
            'common_distance': most_common_distance
        }
        
    except Exception:
        # Fallback if scipy is not available or other errors
        return {
            'is_grid_like': False,
            'rotation_angle': 0,
            'rows': 1,
            'cols': len(centers)
        }


def grid_snake_ordering(centers, original_indices, orientation_info):
    """Order regions in a snake pattern considering grid layout and rotation."""
    try:
        if len(centers) <= 1:
            return original_indices
        
        # Rotate points to align with grid if needed
        rotation_angle = orientation_info.get('rotation_angle', 0)
        
        if abs(rotation_angle) > 0.1:  # Only rotate if significant rotation detected
            cos_a, sin_a = np.cos(-rotation_angle), np.sin(-rotation_angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_centers = centers @ rotation_matrix.T
        else:
            rotated_centers = centers.copy()
        
        # Sort by Y coordinate first to identify rows
        y_sorted_indices = np.argsort(rotated_centers[:, 1])
        y_coords = rotated_centers[y_sorted_indices, 1]
        
        # Group points into rows using clustering approach
        if len(y_coords) > 1:
            y_diffs = np.diff(y_coords)
            if len(y_diffs) > 0:
                row_threshold = np.median(y_diffs) * 1.2
            else:
                row_threshold = 0
        else:
            row_threshold = 0
        
        # Group into rows
        rows = []
        current_row = [y_sorted_indices[0]]
        current_y = y_coords[0]
        
        for i in range(1, len(y_coords)):
            if y_coords[i] - current_y > row_threshold:
                # Start new row
                rows.append(current_row)
                current_row = [y_sorted_indices[i]]
                current_y = y_coords[i]
            else:
                # Add to current row
                current_row.append(y_sorted_indices[i])
        rows.append(current_row)
        
        # Sort each row by X coordinate and apply snake pattern
        ordered_indices = []
        for row_idx, row_indices in enumerate(rows):
            row_centers = rotated_centers[row_indices]
            x_sorted = np.argsort(row_centers[:, 0])
            
            if row_idx % 2 == 0:
                # Even rows: left to right
                row_order = [row_indices[x_sorted[i]] for i in range(len(x_sorted))]
            else:
                # Odd rows: right to left (snake pattern)
                row_order = [row_indices[x_sorted[i]] for i in range(len(x_sorted) - 1, -1, -1)]
            
            ordered_indices.extend(row_order)
        
        # Map back to original indices
        result = [original_indices[i] for i in ordered_indices]
        return result
        
    except Exception:
        # Fallback to simple ordering
        return simple_spatial_ordering(centers, original_indices)


def simple_spatial_ordering(centers, original_indices):
    """Simple fallback ordering: top-to-bottom, left-to-right."""
    try:
        # Sort by Y first (top to bottom), then by X (left to right)
        # Normalize coordinates to handle different scales
        y_coords = centers[:, 1]
        x_coords = centers[:, 0]
        
        # Create a composite sorting key
        # Primary: Y coordinate (top to bottom)
        # Secondary: X coordinate (left to right)
        sort_indices = np.lexsort((x_coords, y_coords))
        
        return [original_indices[i] for i in sort_indices]
        
    except Exception:
        # Ultimate fallback: return original order
        return original_indices