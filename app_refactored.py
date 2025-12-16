# app_refactored.py
"""
Refactored FastAPI application using class-based AI models.

This version demonstrates how to use the model registry system.
Replace your current app.py with this code, or merge the changes.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import base64
import json
import numpy as np
import tempfile
import cv2
from io import BytesIO
from PIL import Image, ImageDraw
from pathlib import Path

# Import model system
from models.model_registry import model_registry
from models.deepliif_model import DeepLIIFModel
from models.example_model import ExampleModel
from models.patch_classifier_model import PatchClassifierModel
from models.ec_cancer_model import ECCancerModel
from models.hovernet_model import HoVerNetModel

# Import existing utilities (keep your existing imports)
from mask_generator import MaskGenerator
from config import config
from src.utils import extract_regions_from_mask


def get_absolute_path(relative_path):
    """Helper function to get absolute path from relative path."""
    return str(Path(relative_path).absolute())


def init_mask_generator():
    """Initialize mask generator (unchanged)."""
    if config["default_model"] == "sam":
        return MaskGenerator(model_path=config["sam_model_path"])
    elif config["default_model"] == "sam2":
        return MaskGenerator(
            model_path=get_absolute_path(config["sam2_model_path"]),
            model_config=config["sam2_config_path"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['default_model']}")


# Initialize FastAPI app
app = FastAPI()

# Initialize mask generator (existing functionality)
if config.get("maskgenerator_enabled", False):
    mask_gen = init_mask_generator()
else:
    mask_gen = None 

# ============================================================================
# NEW: Initialize AI Models using Registry System
# ============================================================================


def initialize_models():
    """
    Initialize all AI processing models.
    
    Add your models here following the pattern:
    1. Register the model class
    2. Load the model with configuration
    """
    if config.get("deepliif_enabled", False):
        # Register DeepLIIF model
        model_registry.register_model_class("deepliif", DeepLIIFModel)
        
        # Load DeepLIIF with configuration
        deepliif_config = {
            'model_dir': get_absolute_path(config["deepliif_model_path"]),
            'tile_size': 256,
            'post_processing': True,
            'gpu_ids': []
        }
        model_registry.load_model("deepliif", deepliif_config)
        print("✓ Loaded DeepLIIF model")
    
    # Register and load PatchClassifier model (if enabled and weights exist)
    if config.get("patch_classifier_enabled", False):
        model_registry.register_model_class("patch_classifier", PatchClassifierModel)
        
        patch_classifier_path = config.get("patch_classifier_model_path")
        if patch_classifier_path and Path(patch_classifier_path).exists():
            patch_classifier_config = {
                'model_path': get_absolute_path(patch_classifier_path),
                'patch_size': 64,
                'batch_size': 32,
                'classifier_threshold': 0.8,
                'generate_gradcam': True,
                'device': 'auto'
            }
            model_registry.load_model("patch_classifier", patch_classifier_config)
            print("✓ Loaded PatchClassifier model")
        else:
            print(f"⚠ PatchClassifier enabled but weights not found at: {patch_classifier_path}")
            print("  Model registered but not loaded. Provide weights to use it.")
    
    # Register and load EC Cancer model (if enabled)
    if config.get("ec_cancer_enabled", False):
        model_registry.register_model_class("ec_cancer", ECCancerModel)
        
        # Check if all required model paths exist
        ec_paths = {
            'patch_classifier_model_path': config.get("ec_cancer_patch_classifier_path"),
            'representation_generator_model_path': config.get("ec_cancer_representation_path"),
            'varmil_model_path': config.get("ec_cancer_varmil_path")
        }
        
        all_paths_exist = all(
            path and Path(get_absolute_path(path)).exists() 
            for path in ec_paths.values()
        )
        
        if all_paths_exist:
            ec_cancer_config = {
                'patch_classifier_model_path': get_absolute_path(ec_paths['patch_classifier_model_path']),
                'representation_generator_model_path': get_absolute_path(ec_paths['representation_generator_model_path']),
                'varmil_model_path': get_absolute_path(ec_paths['varmil_model_path']),
                'device': 'auto'
            }
            model_registry.load_model("ec_cancer", ec_cancer_config)
            print("✓ Loaded EC Cancer model")
        else:
            missing = [k for k, v in ec_paths.items() if not v or not Path(get_absolute_path(v)).exists()]
            print(f"⚠ EC Cancer enabled but some model weights not found:")
            for m in missing:
                print(f"  - {m}: {ec_paths.get(m, 'not specified')}")
            print("  Model registered but not loaded. Provide all weights to use it.")
    
    # Register and load HoVer-Net model (if enabled)
    if config.get("hovernet_enabled", False):
        model_registry.register_model_class("hovernet", HoVerNetModel)
        
        # Use new config structure with model_variant
        default_variant = config.get("hovernet_default_variant", "pannuke")
        hovernet_config = {
            'model_variant': default_variant,
            'device': config.get("hovernet_device", "cuda"),
            'gpu_ids': config.get("hovernet_gpu_ids", [0]),
            'batch_size': config.get("hovernet_batch_size", 8)
        }
        
        # Check if the default variant's checkpoint exists
        hovernet_models = config.get("hovernet_models", {})
        if default_variant in hovernet_models:
            checkpoint_path = hovernet_models[default_variant].get('checkpoint')
            if checkpoint_path and Path(get_absolute_path(checkpoint_path)).exists():
                model_registry.load_model("hovernet", hovernet_config)
                print(f"✓ Loaded HoVer-Net model (variant: {default_variant})")
            else:
                print(f"⚠ HoVer-Net enabled but checkpoint not found: {checkpoint_path}")
                print("  Model registered but not loaded. Provide weights to use it.")
        else:
            print(f"⚠ HoVer-Net enabled but variant '{default_variant}' not found in config")
            print("  Model registered but not loaded. Configure hovernet_models in config.json.")


# Initialize models at startup
initialize_models()


# ============================================================================
# Existing Endpoints (Keep from original app.py)
# ============================================================================

@app.post("/generate_mask")
async def generate_mask_api(
    file: UploadFile = File(...),
    is_tma: bool = Form(False)
):
    if mask_gen is None:
        return JSONResponse(
            {"status": "error", "message": "Mask generator is not enabled."},
            status_code=500
        )
    """Generate mask from slide image."""
    # Check if it's a PNG file
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
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


@app.post("/extract_regions")
async def extract_regions_api(
    file: UploadFile = File(...),
    slide_width: int = Form(...),
    slide_height: int = Form(...),
):
    """Extract regions from mask."""
    # Check if it's a PNG file
    is_png = False
    if file.filename and file.filename.lower().endswith(".png"):
        is_png = True
    elif file.content_type and file.content_type == "image/png":
        is_png = True
    elif not file.filename or file.filename == "":
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
    regions = extract_regions_from_mask(
        image_bytes, (slide_width, slide_height)
    )

    return {
        "status": "success",
        "regions": regions
    }


@app.post("/auto_name_regions")
async def auto_name_regions_api(request: dict):
    """Auto-name regions using intelligent spatial ordering."""
    try:
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        
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
        
        # Convert to numpy arrays
        centers = np.array([
            [r['center_x'], r['center_y']] for r in regions
        ])
        original_indices = [r['original_index'] for r in regions]
        
        # Detect orientation
        orientation_info = detect_slide_orientation(
            centers, slide_width, slide_height
        )
        
        # Apply snake pattern ordering
        if orientation_info['is_grid_like']:
            ordered_indices = grid_snake_ordering(
                centers, original_indices, orientation_info
            )
        else:
            ordered_indices = simple_spatial_ordering(
                centers, original_indices
            )
        
        # Convert to native Python types
        safe_ordered_indices = [int(idx) for idx in ordered_indices]
        safe_orientation_info = {}
        for key, value in orientation_info.items():
            if hasattr(value, 'item'):
                safe_orientation_info[key] = value.item()
            elif isinstance(value, np.bool_):
                safe_orientation_info[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                safe_orientation_info[key] = float(value)
            else:
                safe_orientation_info[key] = value
        
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


# ============================================================================
# REFACTORED: Processing Endpoints Using Model Registry
# ============================================================================

@app.post("/process_region_annotation")
async def process_region_annotation_api(
    region: UploadFile = File(...),
    mask: str = Form(...),
    region_id: str = Form(""),
    model_name: str = Form("deepliif"),  # NEW: Allow model selection
    hyperparameters: str = Form(None)  # NEW: Model hyperparameters as JSON string
):
    """
    Process a region with annotation using specified AI model.
    
    NEW: Now supports multiple models via model_name parameter.
    NEW: Accepts hyperparameters as JSON string.
    """
    # Validate region file is JPEG
    region_ct = (region.content_type or "").lower()
    region_fn = (region.filename or "").lower()
    is_region_jpeg = region_ct in {"image/jpeg", "image/jpg"} or \
                     region_fn.endswith((".jpg", ".jpeg"))
    
    if not is_region_jpeg:
        return JSONResponse(
            {"status": "error", "message": "Region file must be JPEG."},
            status_code=400
        )
    
    # Get the model from registry
    model = model_registry.get_model(model_name)
    if not model:
        return JSONResponse(
            {
                "status": "error",
                "message": f"Model '{model_name}' not found. "
                          f"Available: {model_registry.list_loaded_models()}"
            },
            status_code=400
        )
    
    # Read and process region image
    region_bytes = await region.read()
    region_image = Image.open(BytesIO(region_bytes)).convert("RGB")
    
    # Parse annotation points
    try:
        points_payload = json.loads(mask)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON in 'mask' field."},
            status_code=400
        )
    
    # Convert points to list of tuples
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
    
    annotation_points = to_points(points_payload)
    if len(annotation_points) < 3:
        return JSONResponse(
            {"status": "error", 
             "message": "'mask' must contain at least three points."},
            status_code=400
        )
    
    # Build a binary mask from polygon points
    w, h = region_image.size
    mask_image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_image)
    clamped = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in annotation_points]
    draw.polygon(clamped, outline=255, fill=255)
    
    # Parse hyperparameters if provided
    model_hyperparameters = None
    if hyperparameters:
        try:
            model_hyperparameters = json.loads(hyperparameters)
            print(f"Received hyperparameters: {model_hyperparameters}")
        except json.JSONDecodeError as e:
            return JSONResponse(
                {"status": "error", "message": f"Invalid hyperparameters JSON: {e}"},
                status_code=400
            )
    
    # NEW: Use model's process method with generated mask
    try:
        result = model.process(
            image=region_image,
            mask=mask_image,
            annotation_points=None,
            hyperparameters=model_hyperparameters
        )
    except Exception as e:
        print(f"ERROR: Model {model_name} failed to process region: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"status": "error", "message": f"Model processing failed: {str(e)}"},
            status_code=500
        )
    
    if not result['success']:
        return JSONResponse(
            {"status": "error", "message": result.get('error', 'Unknown error')},
            status_code=500
        )
    
    # Convert processed image to base64
    buffer = BytesIO()
    result['processed_image'].save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # save image to disk for debugging
    # result['processed_image'].save(f"debug_{model_name}_processed.png")
    print(f"Model {model_name} processed region successfully.")
    return {
        "status": "success",
        "processed_image_base64": img_base64,
        "score": result['scores'],
        "str_result": result.get('str_result', 'Processing complete'),
        "region_id": region_id,
        "model_used": model_name,
        "model_version": model.model_version
    }


@app.post("/process_region")
async def process_region_api(
    region: UploadFile = File(...),
    mask: UploadFile = File(...),
    model_name: str = Form("deepliif")  # NEW: Model selection
):
    """
    Process a region with a mask file using specified AI model.
    
    NEW: Now supports multiple models via model_name parameter.
    """
    # Validate file types
    is_region_png = region.filename.lower().endswith(".png") or \
                    region.content_type == "image/png"
    is_mask_png = mask.filename.lower().endswith(".png") or \
                  mask.content_type == "image/png"
    
    if not is_region_png or not is_mask_png:
        return JSONResponse(
            {"status": "error", "message": "Both files must be PNG."},
            status_code=400
        )
    
    # Get model
    model = model_registry.get_model(model_name)
    if not model:
        return JSONResponse(
            {
                "status": "error",
                "message": f"Model '{model_name}' not found. "
                          f"Available: {model_registry.list_loaded_models()}"
            },
            status_code=400
        )
    
    # Read images
    region_bytes = await region.read()
    mask_bytes = await mask.read()
    
    region_image = Image.open(BytesIO(region_bytes)).convert("RGB")
    mask_image = Image.open(BytesIO(mask_bytes)).convert("L")
    
    # NEW: Use model's process method
    result = model.process(
        image=region_image,
        mask=mask_image,
        hyperparameters=None
    )
    
    if not result['success']:
        return JSONResponse(
            {"status": "error", "message": result.get('error', 'Unknown error')},
            status_code=500
        )
    
    # Convert to base64
    buffer = BytesIO()
    result['processed_image'].save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {
        "status": "success",
        "processed_image_base64": img_base64,
        "score": result['scores'],
        "str_result": result.get('str_result', 'Processing complete'),
        "model_used": model_name
    }


# ============================================================================
# NEW: Model Management Endpoints
# ============================================================================

@app.get("/models/list")
async def list_models():
    """
    List all available models.
    
    Returns information about registered and loaded models.
    """
    return {
        "registered_models": model_registry.list_registered_models(),
        "loaded_models": model_registry.list_loaded_models()
    }


@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Returns model configuration, hyperparameters schema, etc.
    """
    info = model_registry.get_model_info(model_name)
    if info:
        return info
    else:
        return JSONResponse(
            {
                "error": f"Model '{model_name}' not loaded. "
                        f"Available: {model_registry.list_loaded_models()}"
            },
            status_code=404
        )


@app.get("/models/{model_name}/hyperparameters")
async def get_model_hyperparameters(model_name: str):
    """
    Get the hyperparameters schema for a model.
    
    Useful for building dynamic UIs for model configuration.
    """
    model = model_registry.get_model(model_name)
    if model:
        return {
            "model": model_name,
            "hyperparameters": model.get_hyperparameters_schema()
        }
    else:
        return JSONResponse(
            {"error": f"Model '{model_name}' not found"},
            status_code=404
        )


# ============================================================================
# Helper Functions for Auto-Naming Regions
# ============================================================================

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


# ============================================================================
# Cleanup on Shutdown
# ============================================================================

@app.on_event("shutdown")
def shutdown_event():
    """Clean up models on shutdown."""
    print("Shutting down and cleaning up models...")
    model_registry.unload_all_models()
    print("Cleanup complete")


if __name__ == "__main__":
    import uvicorn
    import os
    import sys
    
    # Ensure current directory is in Python path for reload subprocess
    current_dir = str(Path(__file__).parent.resolve())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Set PYTHONPATH environment variable for uvicorn subprocess
    python_path = os.environ.get('PYTHONPATH', '')
    paths_to_add = [current_dir]
    for path in paths_to_add:
        if path not in python_path:
            python_path = f"{path}:{python_path}" if python_path else path
    os.environ['PYTHONPATH'] = python_path
    
    # Run server
    # Note: reload=False to avoid subprocess import issues with HoVer-Net
    # Use: uvicorn app_refactored:app --reload for development with manual restart
    uvicorn.run(
        app,  # Direct app reference (no string) when reload=False
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled to avoid subprocess issues
        log_level="info"
    )
