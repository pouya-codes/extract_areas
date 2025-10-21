# Contributing AI Models to the Platform

This guide explains how to add new AI processing models to the platform. The platform uses a modular, class-based architecture that makes it easy to integrate new models while maintaining consistency.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Detailed Implementation Guide](#detailed-implementation-guide)
4. [Testing Your Model](#testing-your-model)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Architecture Overview

The platform uses a three-layer architecture:

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Endpoints                  │
│                    (app.py)                          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Model Registry                      │
│            (models/model_registry.py)                │
│  - Registers models                                  │
│  - Manages model lifecycle                           │
│  - Provides unified access                           │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              Your AI Model Class                     │
│         (inherits from BaseAIModel)                  │
│  - Implements processing logic                       │
│  - Defines hyperparameters                           │
│  - Handles initialization & cleanup                  │
└─────────────────────────────────────────────────────┘
```

### Key Components

- **BaseAIModel**: Abstract base class that defines the interface all models must implement
- **ModelRegistry**: Singleton that manages model registration and lifecycle
- **Individual Model Classes**: Your model implementation (e.g., DeepLIIFModel, YourModel)
- **FastAPI Endpoints**: HTTP API that routes requests to the appropriate model

## Quick Start

### 1. Copy the Template

```bash
cp models/example_model.py models/your_model_name_model.py
```

### 2. Implement Your Model

Edit `models/your_model_name_model.py`:

```python
from models.base_model import BaseAIModel
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

class YourModel(BaseAIModel):
    def __init__(self):
        super().__init__(
            model_name="your_model",
            model_version="1.0.0"
        )
        self.description = "What your model does"
        self.requires_mask = True  # or False
        
    def initialize(self, config: Dict[str, Any]) -> None:
        # Load your model here
        pass
    
    def process(self, image, mask=None, annotation_points=None, 
                hyperparameters=None) -> Dict[str, Any]:
        # Process the image and return results
        pass
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        # Define your hyperparameters
        pass
```

### 3. Register Your Model

Add to `app.py`:

```python
from models.your_model_name_model import YourModel
from models.model_registry import model_registry

# Register the model
model_registry.register_model_class("your_model", YourModel)

# Load the model
model_registry.load_model("your_model", {
    'model_path': '/path/to/weights',
    'device': 'cuda:0'
})
```

### 4. Use in Endpoints

```python
@app.post("/process_with_your_model")
async def process_with_your_model(
    region: UploadFile = File(...),
    model_name: str = Form("your_model")
):
    model = model_registry.get_model(model_name)
    result = model.process(image, mask=mask)
    return result
```

## Detailed Implementation Guide

### Step 1: Understanding the BaseAIModel Interface

Your model must implement these abstract methods:

#### `initialize(config: Dict[str, Any]) -> None`

Called once when the model is loaded. Use this to:
- Load model weights from disk
- Initialize GPU/CPU resources
- Set up preprocessing pipelines
- Validate configuration

```python
def initialize(self, config: Dict[str, Any]) -> None:
    model_path = config.get('model_path')
    if not model_path:
        raise ValueError("model_path is required")
    
    # Load your model
    self.model = load_your_model(model_path)
    self.model.to(config.get('device', 'cpu'))
    self.model.eval()
    
    self._is_initialized = True
```

#### `process(...) -> Dict[str, Any]`

The main processing method. Must return a dict with these keys:

```python
{
    'success': True/False,
    'processed_image': PIL.Image,  # Output visualization
    'scores': {                     # Quantitative metrics
        'metric1': value1,
        'metric2': value2
    },
    'metadata': {...},             # Optional additional info
    'error': "error message"       # If success=False
}
```

Example implementation:

```python
def process(self, image, mask=None, annotation_points=None, 
            hyperparameters=None) -> Dict[str, Any]:
    try:
        # 1. Validate
        is_valid, error = self.validate_input(image, mask)
        if not is_valid:
            return {'success': False, 'error': error}
        
        # 2. Merge hyperparameters
        params = {**self.hyperparameters}
        if hyperparameters:
            params.update(hyperparameters)
        
        # 3. Preprocess
        processed = self.preprocess_image(image, mask)
        
        # 4. Run inference
        output = self.model(processed)
        
        # 5. Convert to PIL Image
        result_image = tensor_to_pil(output)
        
        # 6. Calculate metrics
        scores = self.calculate_metrics(output)
        
        # 7. Return
        return {
            'success': True,
            'processed_image': result_image,
            'scores': scores
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

#### `get_hyperparameters_schema() -> Dict[str, Any]`

Define all tunable parameters for your model:

```python
def get_hyperparameters_schema(self) -> Dict[str, Any]:
    return {
        'threshold': {
            'type': 'float',
            'default': 0.5,
            'min': 0.0,
            'max': 1.0,
            'description': 'Detection confidence threshold',
            'required': False
        },
        'batch_size': {
            'type': 'int',
            'default': 8,
            'min': 1,
            'max': 64,
            'description': 'Batch size for inference'
        },
        'mode': {
            'type': 'choice',
            'default': 'fast',
            'choices': ['fast', 'accurate'],
            'description': 'Processing mode'
        }
    }
```

Supported types:
- `'int'`: Integer values (supports min/max)
- `'float'`: Float values (supports min/max)
- `'bool'`: Boolean values
- `'string'`: Text strings
- `'choice'`: One of predefined choices

### Step 2: Handling Input Data

Your model receives three types of input:

#### 1. Image (Required)
Always provided as a PIL Image in RGB mode:

```python
image = Image.open(file).convert('RGB')
```

#### 2. Mask (Optional)
Binary mask as PIL Image in 'L' mode (grayscale):
- 255 = foreground (region of interest)
- 0 = background (ignore)

```python
if mask:
    # Apply mask to image
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    masked_image = Image.composite(image, white_bg, mask)
```

#### 3. Annotation Points (Optional)
List of (x, y) coordinates defining a polygon:

```python
annotation_points = [(x1, y1), (x2, y2), (x3, y3), ...]

# Convert to mask
mask = Image.new('L', image.size, 0)
draw = ImageDraw.Draw(mask)
draw.polygon(annotation_points, fill=255)
```

### Step 3: Providing Output

Your `process()` method must return a dictionary with these required keys:

#### Required Keys

1. **success** (bool): Whether processing succeeded
2. **processed_image** (PIL.Image): Visualization/overlay to display to user
3. **scores** (dict): Quantitative metrics

#### Optional Keys

4. **metadata** (dict): Additional information
5. **error** (str): Error message if success=False

#### Example Outputs

Successful processing:

```python
{
    'success': True,
    'processed_image': <PIL.Image of overlay>,
    'scores': {
        'total_cells': 150,
        'positive_cells': 42,
        'negative_cells': 108,
        'positivity_rate': 0.28,
        'processing_time': 2.34
    },
    'metadata': {
        'model': 'your_model',
        'version': '1.0.0',
        'confidence': 0.95
    }
}
```

Failed processing:

```python
{
    'success': False,
    'error': 'GPU out of memory. Try reducing image size.'
}
```

### Step 4: Model Registration and Loading

#### In app.py

```python
from models.your_model_name_model import YourModel
from models.model_registry import model_registry
from config import config

# 1. Register model class (at startup)
model_registry.register_model_class("your_model", YourModel)

# 2. Load and initialize model
model_config = {
    'model_path': config["your_model_path"],
    'device': 'cuda:0',
    'batch_size': 8
}
model_registry.load_model("your_model", model_config)
```

#### Using the Model

```python
# Get model instance
model = model_registry.get_model("your_model")

# Process an image
result = model.process(
    image=pil_image,
    mask=mask_image,
    hyperparameters={'threshold': 0.7}
)

if result['success']:
    output_image = result['processed_image']
    scores = result['scores']
else:
    error = result['error']
```

## Testing Your Model

### 1. Unit Testing

Create `tests/test_your_model.py`:

```python
import pytest
from PIL import Image
from models.your_model_name_model import YourModel

def test_model_initialization():
    model = YourModel()
    config = {'model_path': '/path/to/test/weights'}
    model.initialize(config)
    assert model._is_initialized

def test_model_process():
    model = YourModel()
    model.initialize({'model_path': '/path/to/weights'})
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # Process
    result = model.process(test_image)
    
    assert result['success'] == True
    assert 'processed_image' in result
    assert 'scores' in result
```

### 2. API Testing

Test via the endpoint:

```bash
# Test with curl
curl -X POST "http://localhost:8000/process_region_annotation" \
  -F "region=@test_image.jpg" \
  -F "mask={\"points\": [[0,0], [100,0], [100,100], [0,100]]}" \
  -F "model_name=your_model"
```

Or with Python:

```python
import requests

files = {'region': open('test_image.jpg', 'rb')}
data = {
    'mask': '{"points": [[0,0], [100,0], [100,100], [0,100]]}',
    'model_name': 'your_model'
}

response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files=files,
    data=data
)
print(response.json())
```

### 3. Integration Testing

```python
from models.model_registry import model_registry

# Test model info
info = model_registry.get_model_info("your_model")
print(info)

# Test hyperparameters schema
model = model_registry.get_model("your_model")
schema = model.get_hyperparameters_schema()
print(schema)

# Test processing
result = model.process(test_image, mask=test_mask)
assert result['success']
```

## Best Practices

### 1. Error Handling

Always catch exceptions and return meaningful errors:

```python
try:
    output = self.model(image)
except torch.cuda.OutOfMemoryError:
    return {
        'success': False,
        'error': 'GPU out of memory. Try reducing image size or batch size.'
    }
except Exception as e:
    return {
        'success': False,
        'error': f'Unexpected error: {str(e)}'
    }
```

### 2. Resource Management

Clean up properly in `cleanup()`:

```python
def cleanup(self) -> None:
    if self.model:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    self.model = None
```

### 3. Validation

Use `validate_input()` to check inputs:

```python
def validate_input(self, image, mask=None):
    is_valid, error = super().validate_input(image, mask)
    if not is_valid:
        return is_valid, error
    
    # Custom validation
    if image.size[0] > 8000 or image.size[1] > 8000:
        return False, "Image too large. Max 8000x8000 pixels."
    
    return True, None
```

### 4. Preprocessing & Postprocessing

Use the inherited methods:

```python
# Preprocessing: apply mask
masked_image = self.preprocess_image(image, mask)

# Postprocessing: white out masked regions
final_image = self.postprocess_output(result, mask)
```

### 5. Configuration

Store paths and settings in `config.yaml`:

```yaml
your_model:
  model_path: "models/your_model/weights.pth"
  device: "cuda:0"
  default_threshold: 0.5
```

Load in app.py:

```python
model_config = config["your_model"]
model_registry.load_model("your_model", model_config)
```

### 6. Logging

Use Python's logging:

```python
import logging

logger = logging.getLogger(__name__)

class YourModel(BaseAIModel):
    def process(self, image, ...):
        logger.info(f"Processing image of size {image.size}")
        try:
            ...
            logger.info("Processing successful")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
```

## Troubleshooting

### Common Issues

#### 1. "Model class must inherit from BaseAIModel"

Make sure your class inherits from `BaseAIModel`:

```python
from models.base_model import BaseAIModel

class YourModel(BaseAIModel):  # ← Must inherit
    ...
```

#### 2. "Model 'your_model' not registered"

Register before loading:

```python
# First register
model_registry.register_model_class("your_model", YourModel)

# Then load
model_registry.load_model("your_model", config)
```

#### 3. "Image and mask must have the same dimensions"

Ensure mask matches image size:

```python
if mask and mask.size != image.size:
    mask = mask.resize(image.size, Image.NEAREST)
```

#### 4. GPU Out of Memory

- Reduce batch size
- Process smaller tiles
- Use CPU instead of GPU
- Enable gradient checkpointing

```python
config = {
    'model_path': path,
    'device': 'cpu',  # Use CPU
    'batch_size': 1   # Smaller batch
}
```

#### 5. Model Takes Too Long

- Use `eager_mode=True` for faster single-image processing
- Reduce image resolution before processing
- Use model quantization
- Profile your code to find bottlenecks

### Debugging Tips

1. **Test initialization separately**:
   ```python
   model = YourModel()
   model.initialize(config)
   print(model._is_initialized)
   ```

2. **Test with small images first**:
   ```python
   test_image = Image.new('RGB', (256, 256))
   result = model.process(test_image)
   ```

3. **Check model info**:
   ```python
   info = model.get_model_info()
   print(json.dumps(info, indent=2))
   ```

4. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Example: Complete Model Implementation

See `models/example_model.py` for a complete working example with:
- Full interface implementation
- Input validation
- Hyperparameter handling
- Error handling
- Documentation

Also see `models/deepliif_model.py` for a real-world implementation.

## Questions?

If you have questions or need help:
1. Check the existing model implementations
2. Review the `BaseAIModel` documentation
3. Look at the test files for examples
4. Open an issue on the project repository

Happy model building! 🚀
