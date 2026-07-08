# AI Models Directory

This directory contains all AI processing models for the platform using a modular, class-based architecture.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
  - [DeepLIIF](#deepliif)
  - [HoVer-Net](#hover-net)
  - [EC Cancer Model](#ec-cancer-model)
- [Architecture](#architecture)
- [API Integration](#api-integration)
- [HoVer-Net Setup Guide](#hover-net-setup-guide)
- [EC Cancer Quick Reference](#ec-cancer-quick-reference)
- [Testing](#testing)
- [Support](#support)

## Overview

The models directory provides a modular, extensible framework for AI-powered histology image analysis. All models inherit from `BaseAIModel` and are managed through the `ModelRegistry`.

### Structure

```
models/
├── __init__.py              # Package initialization
├── base_model.py            # Abstract base class for all models
├── model_registry.py        # Model registry and lifecycle management
├── deepliif_model.py        # DeepLIIF implementation
├── hovernet_model.py        # HoVer-Net implementation
├── ec_cancer_model.py       # EC Cancer classifier
├── example_model.py         # Example/template model
└── README.md                # This file
```

## Quick Start

### Using an Existing Model

```python
from models.model_registry import model_registry

# Get a loaded model
model = model_registry.get_model("deepliif")

# Process an image
result = model.process(
    image=pil_image,
    mask=mask_image,
    hyperparameters={'threshold': 0.7}
)

if result['success']:
    output = result['processed_image']
    scores = result['scores']
```

### Adding a New Model

1. **Copy the template:**
   ```bash
   cp models/example_model.py models/your_model_model.py
   ```

2. **Implement your model:**
   ```python
   from models.base_model import BaseAIModel
   
   class YourModel(BaseAIModel):
       def initialize(self, config):
           # Load your model
           pass
       
       def process(self, image, mask=None, hyperparameters=None):
           # Process image
           pass
       
       def get_hyperparameters_schema(self):
           # Define parameters
           pass
   ```

3. **Register in app.py:**
   ```python
   from models.your_model_model import YourModel
   from models.model_registry import model_registry
   
   model_registry.register_model_class("your_model", YourModel)
   model_registry.load_model("your_model", config)
   ```

## Available Models

### DeepLIIF
- **File:** `deepliif_model.py`
- **Description:** Deep-Learning Inferred Multiplex ImmunoFluorescence for IHC analysis
- **Requires Mask:** Yes
- **Use Cases:** Cell segmentation, DAPI/marker coloring, cell counting
- **Hyperparameters:**
  - `eager_mode` (bool): Single GPU mode
  - `color_dapi` (bool): Apply DAPI coloring
  - `color_marker` (bool): Apply marker coloring

### HoVer-Net

**Purpose:** Nucleus segmentation and classification in H&E-stained tissue  
**File:** `hovernet_model.py`  
**Requires Mask:** No (optional ROI mask supported)

**Features:**
- Instance segmentation of individual nuclei
- Nucleus type classification (multiple variants available)
- Centroid and boundary detection
- Area statistics

**Model Variants:**

| Variant | Types | Mode | Dataset | Use Case |
|---------|-------|------|---------|----------|
| **PanNuke** | 6 | fast | Multi-organ | General nucleus classification |
| **CoNSeP** | 4 | original | Colorectal | Colorectal tissue analysis |
| **MoNuSAC** | 4 | fast | Multi-organ | Multi-organ segmentation |
| **Kumar** | 0 | original | Multi-organ | Segmentation only (no classification) |

**Quick Usage:**

```python
from PIL import Image

# Get model
hovernet = registry.get_model("hovernet")

# Process image
image = Image.open("tissue.png")
result = hovernet.process(
    image=image,
    hyperparameters={
        'model_variant': 'pannuke',  # Select variant
        'batch_size': 32,
        'draw_centroids': True
    }
)

# Get results
if result['success']:
    print(result['str_result'])  # "Detected 1234 nuclei ..."
    overlay = result['processed_image']
    total = result['scores']['total_nuclei']
```

**Hyperparameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | choice | 'pannuke' | Model variant to use |
| `batch_size` | int | 32 | Patches to process in parallel |
| `draw_centroids` | bool | True | Draw nucleus centers |
| `min_nucleus_size` | int | 10 | Min nucleus area (px) |

**Output Format:**

```python
{
    'success': True,
    'processed_image': PIL.Image,  # Overlay visualization
    'scores': {
        'total_nuclei': 1234,
        'nuclei_by_type': {'Neoplastic': 450, ...},
        'mean_nucleus_area': 85.3,
        'median_nucleus_area': 72.0,
        'std_nucleus_area': 28.5
    },
    'str_result': 'Detected 1234 nuclei ...',
    'metadata': {'model_mode': 'fast', 'nr_types': 6, ...}
}
```

**Use Cases:**

| ✅ Use HoVer-Net | ❌ Don't Use |
|-----------------|--------------|
| H&E tissue | IHC tissue (use DeepLIIF) |
| Nucleus segmentation | Tissue segmentation |
| Nucleus classification | Cell type classification |
| Routine histology | Fluorescence images |

See [HoVer-Net Setup Guide](#hover-net-setup-guide) below for detailed installation instructions.

### EC Cancer Model

**Purpose:** Endometrial cancer classification (NSMP vs p53)  
**File:** `ec_cancer_model.py`  
**Requires Mask:** Yes (user-provided annotation points or mask)

**Features:**
- Patch-based tumor detection
- NSMP vs p53 classification
- Confidence scoring
- Visualization overlay

**Quick Usage:**

```python
import requests
import json
from PIL import Image

# Process image with annotation points
response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files={'region': ('tissue.jpg', image_buffer, 'image/jpeg')},
    data={
        'mask': json.dumps([
            {"x": 100, "y": 100},
            {"x": 500, "y": 500}
        ]),
        'model_name': 'ec_cancer',
        'hyperparameters': json.dumps({
            'tumor_threshold': 0.85
        })
    }
)
```

**Hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tumor_threshold` | 0.9 | Tumor detection threshold (lower=more patches) |
| `batch_size` | 32 | Processing batch size |
| `patch_size` | 1024 | Patch extraction size |
| `stride` | 1 | Stride for patch extraction (increase for speed) |
| `generate_visualization` | true | Generate overlay image |

**Output Scores:**

```json
{
  "nsmp_probability": 0.2345,
  "p53_probability": 0.7655,
  "classification": "p53",
  "confidence": 0.7655,
  "tumor_patches_found": 142
}
```

**Visualization:**
- 🟢 **Green Overlay**: NSMP classification
- 🔴 **Red Overlay**: p53 classification

See [EC Cancer Quick Reference](#ec-cancer-quick-reference) below for detailed configuration.

### Example Model
- **File:** `example_model.py`
- **Description:** Template for creating new models
- **Requires Mask:** No
- **Use:** Copy this as a starting point for new models

## Architecture

### BaseAIModel
Abstract base class that all models must inherit from.

**Key Methods:**
- `initialize(config)`: Load model and set up resources
- `process(image, mask, hyperparameters, ...)`: Process an image and return results
- `get_hyperparameters_schema()`: Define tunable parameters
- `cleanup()`: Free resources

### ModelRegistry
Singleton that manages model lifecycle.

**Key Methods:**
- `register_model_class(name, class)`: Register a model class
- `load_model(name, config)`: Load and initialize a model
- `get_model(name)`: Get a loaded model instance
- `list_loaded_models()`: List all loaded models
- `unload_model(name)`: Unload a model and free resources

## API Integration

### Process with Model

```python
@app.post("/process")
async def process_image(
    image: UploadFile,
    model_name: str = "deepliif"
):
    model = model_registry.get_model(model_name)
    result = model.process(image)
    return result
```

### List Available Models

```python
@app.get("/models/list")
async def list_models():
    return {
        "loaded": model_registry.list_loaded_models()
    }
```

### Get Model Info

```python
@app.get("/models/{name}/info")
async def model_info(name: str):
    return model_registry.get_model_info(name)
```

---

## HoVer-Net Setup Guide

### Installation

#### 1. Clone HoVer-Net Repository

```bash
cd /home/pouya/Develop/UBC/cpathportal/extract_areas/models
git clone https://github.com/vqdang/hover_net.git hovernet
```

#### 2. Download Pre-trained Weights

Download one of the pre-trained checkpoints:

**For Nucleus Segmentation + Classification:**
- [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view?usp=sharing) (fast mode, 6 types)
- [CoNSeP checkpoint](https://drive.google.com/file/d/1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P/view?usp=sharing) (original mode, 4 types)
- [MoNuSAC checkpoint](https://drive.google.com/file/d/13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV/view?usp=sharing) (fast mode, 4 types)

**For Nucleus Segmentation Only:**
- [Kumar checkpoint](https://drive.google.com/file/d/1NUnO4oQRGL-b0fyzlT8LKZzo6KJD0_6X/view?usp=sharing) (original mode)

Place checkpoints in:
```
/home/pouya/Develop/UBC/cpathportal/extract_areas/models/hovernet_checkpoints/
```

#### 3. Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-image scipy
```

### Configuration

#### Add to config.json

```json
{
  "hovernet_models": {
    "pannuke": {
      "checkpoint": "models/hovernet_checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar",
      "type_info": "models/hovernet_checkpoints/type_info.json",
      "nr_types": 6,
      "mode": "fast",
      "description": "PanNuke (6 types): Neoplastic, Inflammatory, Connective, Dead, Epithelial"
    },
    "consep": {
      "checkpoint": "models/hovernet_checkpoints/hovernet_original_consep_type_tf2pytorch.tar",
      "type_info": "models/hovernet_checkpoints/consep_type_info.json",
      "nr_types": 4,
      "mode": "original",
      "description": "CoNSeP (4 types): Inflammatory, Epithelial, Spindle, Miscellaneous"
    },
    "monusac": {
      "checkpoint": "models/hovernet_checkpoints/hovernet_fast_monusac_type_tf2pytorch.tar",
      "type_info": "models/hovernet_checkpoints/monusac_type_info.json",
      "nr_types": 4,
      "mode": "fast",
      "description": "MoNuSAC (4 types): Epithelial, Lymphocyte, Neutrophil, Macrophage"
    },
    "kumar": {
      "checkpoint": "models/hovernet_checkpoints/hovernet_original_kumar_notype_tf2pytorch.tar",
      "type_info": null,
      "nr_types": 0,
      "mode": "original",
      "description": "Kumar (segmentation only, no classification)"
    }
  },
  "hovernet_default_variant": "pannuke"
}
```

#### Type Information File (Optional)

Create `type_info.json` for custom nucleus type colors:

```json
{
    "0": ["Background", [0, 0, 0]],
    "1": ["Neoplastic", [255, 0, 0]],
    "2": ["Inflammatory", [0, 255, 0]],
    "3": ["Connective", [0, 0, 255]],
    "4": ["Dead", [255, 255, 0]],
    "5": ["Epithelial", [255, 0, 255]]
}
```

### Model Modes

| Mode | Input Size | Output Size | Checkpoints |
|------|-----------|-------------|-------------|
| **fast** | 256×256 | 164×164 | PanNuke, MoNuSAC |
| **original** | 270×270 | 80×80 | CoNSeP, Kumar |

⚠️ **Important:** The model mode must match the checkpoint you're using!

### Troubleshooting

**Issue: "Model checkpoint not found"**  
Solution: Verify the checkpoint path in config.json points to the correct `.tar` file

**Issue: "CUDA out of memory"**  
Solution: Reduce `batch_size` hyperparameter or switch to CPU mode

**Issue: "Wrong number of types"**  
Solution: Ensure `nr_types` matches your checkpoint (PanNuke=6, CoNSeP=4, MoNuSAC=4, Kumar=0)

**Issue: "Model mode mismatch"**  
Solution: Set correct mode for your checkpoint (PanNuke/MoNuSAC → fast, CoNSeP/Kumar → original)

### Performance Tips

1. **GPU Usage**: Always use GPU if available
2. **Batch Size**: Increase for faster processing (limited by GPU memory)
3. **Image Size**: Process smaller regions if memory is limited
4. **Preprocessing**: Apply tissue detection to avoid processing background

### Processing Time (GPU)

- Small (1K×1K): 2-5 sec
- Medium (2K×2K): 8-15 sec
- Large (4K×4K): 30-60 sec

### References

- **Paper**: Graham et al., "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images", Medical Image Analysis, 2019
- **GitHub**: https://github.com/vqdang/hover_net

---

## EC Cancer Quick Reference

### Enable the Model

Edit `config.json`:
```json
{
  "ec_cancer_enabled": true,
  "ec_cancer_patch_classifier_path": "models/EC_model/tumor_normal.pt",
  "ec_cancer_representation_path": "models/EC_model/representation.pth",
  "ec_cancer_varmil_path": "models/EC_model/VarMIL.pth"
}
```

### Start Server

```bash
cd extract_areas
python app_refactored.py
```

### Verify Model Loaded

```bash
curl http://localhost:8000/models/list
# Should show "ec_cancer" in loaded_models
```

### Process an Image

```bash
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@tissue.jpg" \
  -F 'mask=[{"x":100,"y":100},{"x":500,"y":500}]' \
  -F "model_name=ec_cancer"
```

### Python Example

```python
import requests
import json
from PIL import Image
from io import BytesIO

# Load image
image = Image.open("tissue.jpg")
buffer = BytesIO()
image.save(buffer, format='JPEG')
buffer.seek(0)

# Define region
polygon = [
    {"x": 100, "y": 100},
    {"x": 500, "y": 100},
    {"x": 500, "y": 500},
    {"x": 100, "y": 500}
]

# Process
response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files={'region': ('tissue.jpg', buffer, 'image/jpeg')},
    data={
        'mask': json.dumps(polygon),
        'model_name': 'ec_cancer',
        'hyperparameters': json.dumps({
            'tumor_threshold': 0.85,
            'generate_visualization': True
        })
    }
)

result = response.json()
print(f"{result['score']['classification']}: {result['score']['confidence']:.1%}")
```

### Common Issues

**❌ Model Not Loading**  
Check: 
1. `ec_cancer_enabled: true` in config.json
2. All 3 model files exist at specified paths
3. Console output for error messages

**❌ No Tumor Patches Found**  
Fix: Lower `tumor_threshold` to 0.8 or 0.7

**❌ Out of Memory**  
Fix: Reduce `batch_size` to 16 or 8

**❌ Slow Processing**  
Fix: Increase `stride` to 2 or 3

### Performance Tuning

**For Speed:**
```json
{
  "stride": 3,
  "batch_size": 64,
  "patch_size": 512,
  "generate_visualization": false
}
```

**For Accuracy:**
```json
{
  "stride": 1,
  "batch_size": 32,
  "patch_size": 1024,
  "tumor_threshold": 0.85
}
```

### Resource Requirements

- **RAM**: 8-16 GB
- **VRAM**: 6-8 GB (GPU mode)
- **Disk**: ~250 MB (3 model weights)

---

## Testing

Run model tests:
```bash
pytest tests/test_models.py
```

Test a specific model:
```bash
python -m models.deepliif_model
python -m models.hovernet_model
```

## Documentation

For detailed instructions on adding new models:
- **[../ADDING_NEW_AI_MODELS.md](../ADDING_NEW_AI_MODELS.md)** - Complete guide for model integration

## Support

- Check existing model implementations for examples
- Review the `BaseAIModel` documentation in `base_model.py`
- See test files for usage examples
- Check GitHub repositories for model-specific issues:
  - HoVer-Net: https://github.com/vqdang/hover_net/issues

---

**Last Updated:** 2024  
**Maintained By:** UBC cpathportal Team
