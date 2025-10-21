# Migration Guide: From Monolithic to Class-Based AI Models

This guide helps you migrate from the current `app.py` to the new class-based model architecture.

## Overview

The refactoring separates AI model logic from API endpoints, making it easy to:
- Add new models without modifying API code
- Switch between models dynamically
- Manage model lifecycle (loading/unloading)
- Share models across multiple endpoints

## Step-by-Step Migration

### Step 1: Keep Existing Code Working

**Before making changes:**
1. Backup your current `app.py`:
   ```bash
   cp app.py app.py.backup
   ```

2. Test that everything works:
   ```bash
   python app.py
   # Test your endpoints
   ```

### Step 2: Add Model System Imports

At the top of `app.py`, add:

```python
# NEW: Import model system
from models.model_registry import model_registry
from models.deepliif_model import DeepLIIFModel
```

### Step 3: Replace Model Initialization

**OLD CODE (remove this):**
```python
# Initialize the image processor
model_dir = get_absolute_path(config["deepliif_model_path"])
image_processor = init_image_processor(model_dir)

def init_image_processor(model_dir, tile_size=256, post_processing=True, gpu_ids=[]):
    return ImageProcessor(model_dir, tile_size, post_processing, gpu_ids)
```

**NEW CODE (add this):**
```python
def initialize_models():
    """Initialize all AI processing models."""
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

# Call at startup
initialize_models()
```

### Step 4: Update process_region_annotation Endpoint

**OLD CODE (your current implementation):**
```python
@app.post("/process_region_annotation")
async def process_region_annotation_api(
    region: UploadFile = File(...),
    mask: str = Form(...),
    region_id: str = Form("")
):
    # ... validation code ...
    
    # OLD: Direct call to image_processor
    processed_image, score = image_processor.test_img(
        masked_region,
        eager_mode=False,
        color_dapi=False,
        color_marker=False,
        tissue_mask=mask_image,
    )
    
    # ... response code ...
```

**NEW CODE (refactored):**
```python
@app.post("/process_region_annotation")
async def process_region_annotation_api(
    region: UploadFile = File(...),
    mask: str = Form(...),
    region_id: str = Form(""),
    model_name: str = Form("deepliif")  # NEW: Allow model selection
):
    # ... keep validation code ...
    
    # NEW: Get model from registry
    model = model_registry.get_model(model_name)
    if not model:
        return JSONResponse(
            {"status": "error", "message": f"Model '{model_name}' not found"},
            status_code=400
        )
    
    # NEW: Use model's process method
    result = model.process(
        image=region_image,
        annotation_points=annotation_points
    )
    
    if not result['success']:
        return JSONResponse(
            {"status": "error", "message": result.get('error')},
            status_code=500
        )
    
    # Convert to base64 (same as before)
    buffer = BytesIO()
    result['processed_image'].save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {
        "status": "success",
        "processed_image_base64": img_base64,
        "score": result['scores'],
        "region_id": region_id,
        "model_used": model_name  # NEW: Show which model was used
    }
```

### Step 5: Update process_region Endpoint

**Apply the same pattern:**

```python
@app.post("/process_region")
async def process_region_api(
    region: UploadFile = File(...),
    mask: UploadFile = File(...),
    model_name: str = Form("deepliif")  # NEW
):
    # ... validation ...
    
    # NEW: Get and use model
    model = model_registry.get_model(model_name)
    result = model.process(image=region_image, mask=mask_image)
    
    # ... handle result ...
```

### Step 6: Add New Model Management Endpoints (Optional)

Add these new endpoints for better model management:

```python
@app.get("/models/list")
async def list_models():
    return {
        "registered": model_registry.list_registered_models(),
        "loaded": model_registry.list_loaded_models()
    }

@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    info = model_registry.get_model_info(model_name)
    if info:
        return info
    return JSONResponse({"error": "Model not found"}, status_code=404)
```

### Step 7: Add Cleanup on Shutdown

```python
@app.on_event("shutdown")
def shutdown_event():
    """Clean up models on shutdown."""
    model_registry.unload_all_models()
```

### Step 8: Test the Migration

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Test existing functionality:**
   ```bash
   curl -X POST "http://localhost:8000/process_region_annotation" \
     -F "region=@test.jpg" \
     -F "mask={\"points\": [[0,0], [100,0], [100,100]]}"
   ```

3. **Test new endpoints:**
   ```bash
   curl http://localhost:8000/models/list
   curl http://localhost:8000/models/deepliif/info
   ```

## What Changes for Users?

### For API Users

**Minimal breaking changes:**
- Existing endpoints work the same way
- NEW optional parameter: `model_name` to select which model to use
- Response includes `model_used` field

**Before (still works):**
```bash
curl -X POST "/process_region_annotation" \
  -F "region=@image.jpg" \
  -F "mask={...}"
```

**After (with model selection):**
```bash
curl -X POST "/process_region_annotation" \
  -F "region=@image.jpg" \
  -F "mask={...}" \
  -F "model_name=deepliif"
```

### For Developers

**Adding a new model is now easy:**

1. Create model file: `models/my_model.py`
2. Register in `app.py`: 
   ```python
   model_registry.register_model_class("my_model", MyModel)
   model_registry.load_model("my_model", config)
   ```
3. Done! No need to modify endpoints

## Complete Example

Here's a minimal example showing the before/after:

### Before (Monolithic)

```python
# app.py
image_processor = ImageProcessor(model_dir)

@app.post("/process")
async def process(image: UploadFile):
    img = Image.open(BytesIO(await image.read()))
    result, scores = image_processor.test_img(img, ...)
    return {"result": result, "scores": scores}
```

### After (Modular)

```python
# app.py
model_registry.register_model_class("deepliif", DeepLIIFModel)
model_registry.load_model("deepliif", config)

@app.post("/process")
async def process(
    image: UploadFile,
    model_name: str = "deepliif"
):
    img = Image.open(BytesIO(await image.read()))
    model = model_registry.get_model(model_name)
    result = model.process(img)
    return result
```

## Rollback Plan

If something goes wrong:

1. **Restore backup:**
   ```bash
   cp app.py.backup app.py
   ```

2. **Restart server:**
   ```bash
   python app.py
   ```

3. **Report issues** and we'll help fix them

## Benefits After Migration

1. **Easy to add models:** Just inherit from `BaseAIModel`
2. **Model switching:** Users can select models via API
3. **Better resource management:** Load/unload models as needed
4. **Testability:** Each model can be tested independently
5. **Documentation:** Clear interface for all models

## Need Help?

- Check `app_refactored.py` for a complete working example
- Review `CONTRIBUTING_AI_MODELS.md` for detailed docs
- Look at `models/example_model.py` for a template

## Next Steps

After migration:
1. Add more models following the template
2. Implement model-specific hyperparameter controls
3. Add model performance monitoring
4. Create model comparison endpoints
