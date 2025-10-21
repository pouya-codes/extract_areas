# Architecture Overview - Refactored App

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Aimviewer Frontend                        │
│                  (Django/OMERO Web Interface)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP REST API
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (app_refactored.py)           │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                      │
│  • /generate_mask               • /process_region_annotation    │
│  • /extract_regions             • /process_region               │
│  • /auto_name_regions           • /models/*                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Model Registry System                       │
│                   (models/model_registry.py)                     │
├─────────────────────────────────────────────────────────────────┤
│  Manages:                                                        │
│  • Model registration and loading                               │
│  • Model lifecycle (init, process, cleanup)                     │
│  • Multiple model instances                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  DeepLIIF Model  │  │   Your Model     │  │  Future Models   │
    │  (deepliif_      │  │   (your_model_   │  │  (new_model_     │
    │   model.py)      │  │    model.py)     │  │   model.py)      │
    ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
    │ Implements:      │  │ Implements:      │  │ Implements:      │
    │ • BaseAIModel    │  │ • BaseAIModel    │  │ • BaseAIModel    │
    │ • initialize()   │  │ • initialize()   │  │ • initialize()   │
    │ • process()      │  │ • process()      │  │ • process()      │
    │ • cleanup()      │  │ • cleanup()      │  │ • cleanup()      │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
            │                     │                     │
            ↓                     ↓                     ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ DeepLIIF Engine  │  │ Your AI Engine   │  │ Future AI Engine │
    │ (module/         │  │ (your code)      │  │ (future code)    │
    │  DeepLiff/)      │  │                  │  │                  │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Request Flow: Process Tissue Region

### Step-by-Step Flow

```
1. User draws polygon in Aimviewer
   ├─ Polygon points: [(x1,y1), (x2,y2), ...]
   └─ Region bounding box: [x, y, w, h]

2. Aimviewer extracts region image
   ├─ Calls OMERO API to read region
   └─ Gets JPEG image data

3. Aimviewer sends HTTP POST to FastAPI
   ├─ URL: http://127.0.0.1:8000/process_region_annotation
   ├─ Files: region.jpg (JPEG image)
   ├─ Form data:
   │   ├─ mask: '[{"x":50,"y":50},{"x":200,"y":50},...]' (JSON)
   │   ├─ region_id: "tissue_region_1"
   │   └─ model_name: "deepliif"
   └─ Timeout: 300 seconds

4. FastAPI validates request
   ├─ Check file type (must be JPEG)
   ├─ Check model exists
   └─ Parse JSON annotation points

5. FastAPI creates binary mask ← THE FIX!
   ├─ Create blank mask: Image.new("L", (w,h), 0)
   ├─ Draw polygon: draw.polygon(points, fill=255)
   └─ Result: White polygon on black background

6. FastAPI calls model.process()
   ├─ Input: region_image, mask_image
   └─ Passes to DeepLIIF model

7. DeepLIIF processes image
   ├─ Preprocess: Apply mask (white out non-annotated areas)
   ├─ Run inference: Cell segmentation and classification
   ├─ Postprocess: Apply mask to output
   └─ Return: Segmentation overlay + cell counts

8. FastAPI prepares response
   ├─ Convert processed image to PNG
   ├─ Encode as base64
   └─ Include scores (cell counts, etc.)

9. Aimviewer receives response
   ├─ Decode base64 image
   ├─ Display overlay on slide
   └─ Show scores to user
```

## Data Flow Diagram

```
┌──────────────┐
│ Polygon      │
│ Points       │ ─────┐
└──────────────┘      │
                      ├─→ to_points() ──→ List[(x,y)]
┌──────────────┐      │                        │
│ Region       │ ─────┘                        │
│ Image (JPEG) │                               │
└──────────────┘                               ↓
       │                              ┌─────────────────┐
       │                              │ ImageDraw.      │
       │                              │ polygon()       │
       │                              └────────┬────────┘
       │                                       │
       │                                       ↓
       │                              ┌─────────────────┐
       │                              │ Binary Mask     │
       │                              │ (PIL Image "L") │
       │                              └────────┬────────┘
       │                                       │
       ↓                                       ↓
┌──────────────────────────────────────────────────────┐
│           model.process(image, mask)                 │
└──────────────────┬───────────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────────┐
│           DeepLIIF Processing                        │
├──────────────────────────────────────────────────────┤
│ 1. Preprocess:  masked_region = composite(img,mask) │
│ 2. Inference:   run neural network                  │
│ 3. Postprocess: apply mask to output                │
└──────────────────┬───────────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────────┐
│ Result: {'processed_image': PIL.Image,              │
│          'scores': {'total_cells': 150, ...},       │
│          'success': True}                            │
└──────────────────┬───────────────────────────────────┘
                   │
                   ↓
            Convert to Base64
                   │
                   ↓
            Return to Frontend
```

## Model Interface (BaseAIModel)

```python
class BaseAIModel(ABC):
    """All AI models must implement this interface"""
    
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        """Load model weights and initialize"""
        
    @abstractmethod
    def process(
        self,
        image: PIL.Image,          # Input region (RGB)
        mask: PIL.Image = None,    # Binary mask (L mode)
        annotation_points: List = None,  # Alternative input
        hyperparameters: Dict = None     # Runtime config
    ) -> Dict:
        """
        Process image and return results
        
        Returns:
            {
                'processed_image': PIL.Image,  # Overlay/result
                'scores': Dict,                # Metrics
                'success': bool,               # Status
                'error': str                   # If failed
            }
        """
        
    @abstractmethod
    def get_hyperparameters_schema(self) -> Dict:
        """Return schema for configurable parameters"""
        
    def cleanup(self) -> None:
        """Clean up resources (optional)"""
```

## Key Components

### 1. Model Registry (`model_registry.py`)
- Singleton pattern
- Manages model lifecycle
- Provides model lookup
- Handles initialization and cleanup

### 2. Base Model (`base_model.py`)
- Abstract interface for all models
- Common preprocessing/postprocessing
- Input validation
- Consistent return format

### 3. DeepLIIF Model (`deepliif_model.py`)
- Implements BaseAIModel
- Wraps existing DeepLIIF code
- Handles mask application
- Returns segmentation results

### 4. FastAPI Endpoints (`app_refactored.py`)
- RESTful API
- Model-agnostic processing
- Flexible model selection
- Comprehensive error handling

## Benefits of Refactored Architecture

✓ **Extensibility:** Add new AI models easily  
✓ **Maintainability:** Clean separation of concerns  
✓ **Flexibility:** Switch models at runtime  
✓ **Consistency:** Standardized interface  
✓ **Testability:** Mock models for testing  
✓ **Documentation:** Self-documenting API  

---

## How to Develop and Deploy a New Model

### Step 1: Create Your Model Class

Create a new file in the `models/` directory (e.g., `models/your_model.py`):

```python
"""
Your Model Implementation

Description of what your model does.
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np

from models.base_model import BaseAIModel


class YourModel(BaseAIModel):
    """
    Your custom AI model for image analysis.
    
    Describe what your model does, what it detects, etc.
    """
    
    def __init__(self):
        super().__init__(
            model_name="your_model",  # Unique identifier
            model_version="1.0.0"     # Version string
        )
        self.description = "Description of your model"
        self.requires_mask = True  # Set to True if mask is required
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG']
        
        # Initialize your model-specific attributes
        self.model = None
        self.device = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'threshold': 0.5,
            'batch_size': 1,
            'your_param': 'default_value'
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize your model with configuration.
        
        Args:
            config: Dict with keys like:
                - model_path: Path to model weights
                - device: 'cpu' or 'cuda'
                - any other configuration you need
        """
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("model_path is required in config")
        
        device = config.get('device', 'cpu')
        self.device = device
        
        # Load your model
        # Example: self.model = torch.load(model_path)
        # Example: self.model.to(self.device)
        # Example: self.model.eval()
        
        print(f"Loaded {self.model_name} from {model_path}")
        
        self._is_initialized = True
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an image with your model.
        
        Args:
            image: Input image (RGB)
            mask: Optional binary mask (L mode, 0=background, 255=foreground)
            annotation_points: Optional polygon points (if mask not provided)
            hyperparameters: Optional runtime parameters
        
        Returns:
            Dict with:
                - processed_image: PIL Image with your model's output/overlay
                - scores: Dict of metrics (e.g., {'object_count': 42})
                - success: True if successful
                - error: Error message if failed
        """
        try:
            # Validate inputs
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Merge hyperparameters
            params = {**self.hyperparameters}
            if hyperparameters:
                params.update(hyperparameters)
            
            # If annotation_points provided but no mask, create mask
            if annotation_points and not mask:
                from PIL import ImageDraw
                w, h = image.size
                mask = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(annotation_points, outline=255, fill=255)
            
            # Preprocess: apply mask if provided
            if mask:
                processed_input = self.preprocess_image(image, mask)
            else:
                processed_input = image
            
            # YOUR MODEL INFERENCE HERE
            # Example:
            # input_tensor = self.preprocess_for_model(processed_input)
            # output = self.model(input_tensor)
            # result_image = self.postprocess_model_output(output)
            
            # For now, placeholder:
            result_image = processed_input.copy()
            scores = {
                'detected_objects': 0,
                'confidence': 0.0
            }
            
            # Postprocess: apply mask to output if provided
            if mask:
                result_image = self.postprocess_output(result_image, mask)
            
            return {
                'processed_image': result_image,
                'scores': scores,
                'success': True,
                'metadata': {
                    'model': self.model_name,
                    'version': self.model_version,
                    'hyperparameters': params
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Define configurable hyperparameters for your model.
        
        Returns:
            Dictionary describing each hyperparameter
        """
        return {
            'threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Detection confidence threshold'
            },
            'batch_size': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 32,
                'description': 'Batch size for processing'
            },
            'your_param': {
                'type': 'string',
                'default': 'default_value',
                'description': 'Description of your parameter'
            }
        }
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            # Clean up GPU memory, close files, etc.
            # Example: del self.model
            # Example: torch.cuda.empty_cache()
            pass
        self.model = None
```

### Step 2: Add Model Configuration

Add your model's configuration to `config.yaml` or `config.json`:

```yaml
# config.yaml
your_model_path: "models/your_model/weights.pth"
your_model_device: "cuda"  # or "cpu"
```

Or in `config.py`:

```python
config = {
    # ... existing config ...
    "your_model_path": "models/your_model/weights.pth",
    "your_model_device": "cuda",
}
```

### Step 3: Register and Load Your Model

Edit `app_refactored.py` in the `initialize_models()` function:

```python
def initialize_models():
    """
    Initialize all AI processing models.
    """
    # Existing models
    model_registry.register_model_class("deepliif", DeepLIIFModel)
    deepliif_config = {
        'model_dir': get_absolute_path(config["deepliif_model_path"]),
        'tile_size': 256,
        'post_processing': True,
        'gpu_ids': []
    }
    model_registry.load_model("deepliif", deepliif_config)
    print("✓ Loaded DeepLIIF model")
    
    # ADD YOUR MODEL HERE
    from models.your_model import YourModel
    model_registry.register_model_class("your_model", YourModel)
    your_model_config = {
        'model_path': get_absolute_path(config["your_model_path"]),
        'device': config.get("your_model_device", "cpu")
    }
    model_registry.load_model("your_model", your_model_config)
    print("✓ Loaded your model")
```

### Step 4: Test Your Model

Create a test script (e.g., `test_your_model.py`):

```python
"""Test script for your model."""
import requests
import json
from io import BytesIO
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

# Create test image
img = Image.new('RGB', (256, 256), color='white')
buffer = BytesIO()
img.save(buffer, format='JPEG')
buffer.seek(0)

# Test annotation points
points = [[50, 50], [200, 50], [200, 200], [50, 200]]

# Call the API with your model
files = {'region': ('test.jpg', buffer, 'image/jpeg')}
data = {
    'mask': json.dumps(points),
    'region_id': 'test_1',
    'model_name': 'your_model'  # ← Use your model name
}

response = requests.post(
    f"{BASE_URL}/process_region_annotation",
    files=files,
    data=data,
    timeout=60
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Success: {result.get('status')}")
    print(f"Model: {result.get('model_used')}")
    print(f"Scores: {result.get('score')}")
else:
    print(f"Error: {response.text}")
```

### Step 5: Deploy to Production

#### Option A: Direct Replacement

```bash
# Backup current app
cp app.py app.py.backup

# Use refactored app
cp app_refactored.py app.py

# Restart service
systemctl restart your-service-name
```

#### Option B: Gradual Migration

Run both apps side-by-side:

```bash
# Terminal 1: Old app
python app.py --port 8000

# Terminal 2: New app (with your model)
python app_refactored.py --port 8001

# Update aimviewer to use port 8001 for testing
```

#### Option C: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "app_refactored.py"]
```

Build and run:

```bash
docker build -t your-model-api .
docker run -p 8000:8000 your-model-api
```

### Step 6: Verify Deployment

Check that your model is loaded:

```bash
curl http://localhost:8000/models/list
```

Expected output:
```json
{
  "registered_models": ["deepliif", "your_model"],
  "loaded_models": ["deepliif", "your_model"]
}
```

Get model info:

```bash
curl http://localhost:8000/models/your_model/info
```

### Step 7: Use Your Model from Frontend

The frontend can now select your model:

```javascript
// In aimviewer or your frontend
const formData = new FormData();
formData.append('region', regionImageBlob);
formData.append('mask', JSON.stringify(annotationPoints));
formData.append('model_name', 'your_model');  // ← Select your model

fetch('http://localhost:8000/process_region_annotation', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Model used:', data.model_used);
    console.log('Scores:', data.score);
    // Display processed_image_base64
});
```

---

## Model Development Checklist

- [ ] Create model class inheriting from `BaseAIModel`
- [ ] Implement required methods: `initialize()`, `process()`, `get_hyperparameters_schema()`
- [ ] Add model configuration to `config.py` or `config.yaml`
- [ ] Register model in `app_refactored.py:initialize_models()`
- [ ] Test model with test script
- [ ] Verify model loads: `GET /models/list`
- [ ] Check hyperparameters: `GET /models/{name}/hyperparameters`
- [ ] Test processing: `POST /process_region_annotation`
- [ ] Document model usage and requirements
- [ ] Deploy to production environment
- [ ] Update frontend to support new model selection

---

## Common Patterns and Best Practices

### 1. GPU Memory Management

```python
def cleanup(self) -> None:
    """Clean up GPU memory."""
    if hasattr(self, 'model') and self.model:
        del self.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 2. Batch Processing

```python
def process(self, image, mask=None, **kwargs):
    # Split large images into tiles
    tiles = self.split_into_tiles(image, tile_size=512)
    
    results = []
    for tile in tiles:
        result = self.process_tile(tile)
        results.append(result)
    
    # Merge results
    final_result = self.merge_tiles(results)
    return {'processed_image': final_result, 'success': True}
```

### 3. Progress Reporting

```python
def process(self, image, **kwargs):
    steps = ['preprocess', 'inference', 'postprocess']
    for i, step in enumerate(steps):
        progress = (i + 1) / len(steps) * 100
        # Log or emit progress
        print(f"{step}: {progress}%")
```

### 4. Error Recovery

```python
def process(self, image, **kwargs):
    try:
        result = self.run_inference(image)
    except OutOfMemoryError:
        # Fallback to CPU or smaller batch
        result = self.run_inference_cpu(image)
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed: {str(e)}",
            'fallback_available': True
        }
```

### 5. Caching Results

```python
import hashlib
from functools import lru_cache

def get_image_hash(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()

def process(self, image, **kwargs):
    cache_key = get_image_hash(image)
    if cache_key in self.cache:
        return self.cache[cache_key]
    
    result = self.run_inference(image)
    self.cache[cache_key] = result
    return result
```

---

## Troubleshooting

### Model Not Loading

**Problem:** Model doesn't appear in `/models/list`

**Solutions:**
1. Check model class is imported in `app_refactored.py`
2. Verify `register_model_class()` is called
3. Check `load_model()` is called with correct config
4. Look for errors during `initialize()`

### Out of Memory Errors

**Problem:** GPU runs out of memory during processing

**Solutions:**
1. Reduce batch size in hyperparameters
2. Process image in tiles/patches
3. Use `torch.no_grad()` during inference
4. Clear cache with `torch.cuda.empty_cache()`
5. Fallback to CPU processing

### Slow Processing

**Problem:** Model takes too long to process images

**Solutions:**
1. Use GPU if available
2. Optimize model (quantization, pruning)
3. Cache repeated computations
4. Process multiple images in parallel
5. Use smaller input resolution

### Incorrect Results

**Problem:** Model produces wrong output

**Solutions:**
1. Verify input preprocessing matches training
2. Check mask is applied correctly
3. Validate hyperparameters
4. Test with known good examples
5. Check model weights loaded correctly
