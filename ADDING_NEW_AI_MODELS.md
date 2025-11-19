# Adding New AI Models to the Platform

This guide walks you through the process of integrating a new AI model into the platform using the `BaseAIModel` interface.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [BaseAIModel Interface](#baseaimodel-interface)
- [Model Registration](#model-registration)
- [Frontend Integration](#frontend-integration)
- [Testing Your Model](#testing-your-model)
- [Best Practices](#best-practices)
- [Example: EC Cancer Model](#example-ec-cancer-model)

## Overview

The platform uses a consistent interface (`BaseAIModel`) for all AI models. This allows:
- Easy integration of new models without modifying the core platform
- Consistent API for frontend and backend communication
- Automatic hyperparameter exposure to the UI
- Standardized input/output formats

## Quick Start

**5-Step Process:**

1. **Create your model file**: `models/your_model_name_model.py`
2. **Inherit from BaseAIModel**: Implement required abstract methods
3. **Register your model**: Add to `model_registry` in your app initialization
4. **Configure model paths**: Add configuration in your config file
5. **Test**: Use API endpoints to verify functionality

## Step-by-Step Guide

### Step 1: Create Your Model File

Create a new Python file in the `models/` directory:

```bash
cd models/
cp example_model.py my_model_model.py
```

### Step 2: Implement BaseAIModel Interface

Your model must inherit from `BaseAIModel` and implement these methods:

```python
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from models.base_model import BaseAIModel

class MyModel(BaseAIModel):
    """Your model description."""
    
    def __init__(self):
        super().__init__(
            model_name="my_model",  # Unique identifier
            model_version="1.0.0"
        )
        self.description = "What your model does"
        self.requires_mask = False  # True if you need mask/annotations
        self.hyperparameters = {
            'threshold': 0.5,
            'batch_size': 32
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Load model weights and initialize resources."""
        # Load your model here
        pass
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an image and return results."""
        # Your processing logic here
        return {
            'success': True,
            'processed_image': image,  # Your output image
            'scores': {'accuracy': 0.95},
            'str_result': 'Processing complete: 95% confidence',
            'metadata': {'processing_time': 1.23}
        }
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """Define configurable parameters for frontend UI."""
        return {
            'threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Detection threshold'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 1,
                'max': 128,
                'description': 'Number of samples per batch'
            }
        }
    
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        # Release resources here
        pass
```

### Step 3: Register Your Model

In your application initialization (e.g., `app_refactored.py`):

```python
from models.model_registry import ModelRegistry
from models.my_model_model import MyModel

# Get registry instance
model_registry = ModelRegistry()

# Register your model class
model_registry.register_model_class("my_model", MyModel)

# Configure and load your model
my_model_config = {
    'model_path': '/path/to/model/weights.pth',
    'device': 'cuda'  # or 'cpu'
}

model_registry.load_model("my_model", my_model_config)
```

### Step 4: Add Configuration

Add your model configuration to `config.json`:

```json
{
    "models": {
        "my_model": {
            "enabled": true,
            "model_path": "/absolute/path/to/weights.pth",
            "device": "cuda",
            "custom_param": "value"
        }
    }
}
```

### Step 5: Test Your Model

Use the API endpoint to test:

```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my_model",
    "image_id": 123,
    "region_id": 456,
    "hyperparameters": {
      "threshold": 0.7
    }
  }'
```

## BaseAIModel Interface

### Required Methods

#### `__init__(self)`
Initialize your model with metadata:
- `model_name`: Unique identifier (lowercase, underscores)
- `model_version`: Semantic version string
- `description`: User-friendly description
- `requires_mask`: Boolean indicating if mask is required
- `hyperparameters`: Default values for configurable parameters

#### `initialize(self, config: Dict[str, Any]) -> None`
Called once when the model is loaded. Use this to:
- Load model weights from disk
- Initialize GPU/CPU resources
- Set up preprocessing pipelines
- Validate configuration

**Raises:**
- `ValueError`: If configuration is invalid
- `RuntimeError`: If initialization fails

#### `process(self, image, mask, annotation_points, hyperparameters) -> Dict`
Main processing method. Must return a dictionary with:

**Required keys:**
- `success` (bool): Whether processing succeeded
- `processed_image` (PIL.Image): Output image with overlays
- `str_result` (str): Human-readable summary for UI display
- `scores` (dict): Quantitative metrics

**Optional keys:**
- `metadata` (dict): Additional information
- `error` (str): Error message if `success=False`

**Example return:**
```python
{
    'success': True,
    'processed_image': output_image,
    'scores': {
        'total_cells': 150,
        'positive_cells': 42,
        'confidence': 0.95
    },
    'str_result': 'Detected 42/150 positive cells (28.0%)',
    'metadata': {
        'processing_time': 2.34,
        'model_version': '1.0.0'
    }
}
```

#### `get_hyperparameters_schema(self) -> Dict`
Define configurable parameters for the frontend UI.

**Supported types:**
- `int`: Integer values with min/max
- `float`: Floating-point values with min/max
- `bool`: True/False checkbox
- `string`: Text input
- `choice`: Dropdown selection

**Example:**
```python
{
    'patch_size': {
        'type': 'int',
        'default': 1024,
        'min': 256,
        'max': 2048,
        'description': 'Size of patches to extract (pixels)'
    },
    'enable_visualization': {
        'type': 'bool',
        'default': True,
        'description': 'Generate visualization overlay'
    },
    'color_space': {
        'type': 'choice',
        'default': 'RGB',
        'choices': ['RGB', 'LAB', 'HSV'],
        'description': 'Color space for processing'
    }
}
```

### Optional Methods

#### `validate_input(self, image, mask) -> Tuple[bool, Optional[str]]`
Custom input validation. Returns `(is_valid, error_message)`.

#### `preprocess_image(self, image, mask) -> Image`
Custom preprocessing before model inference.

#### `postprocess_output(self, output_image, mask) -> Image`
Custom postprocessing after model inference.

#### `cleanup(self) -> None`
Release resources (GPU memory, file handles, etc.).

## Model Registration

### Using ModelRegistry

The `ModelRegistry` is a singleton that manages all models:

```python
from models.model_registry import ModelRegistry

registry = ModelRegistry()

# Register model class (doesn't load it yet)
registry.register_model_class("my_model", MyModelClass)

# Load and initialize model
config = {'model_path': '/path/to/weights.pth'}
model = registry.load_model("my_model", config)

# Get loaded model
model = registry.get_model("my_model")

# List all registered models
available = registry.list_models()

# Unload model
registry.unload_model("my_model")
```

### Model Lifecycle

1. **Registration**: `register_model_class()` - Registers class without instantiating
2. **Loading**: `load_model()` - Instantiates and calls `initialize()`
3. **Usage**: `process()` - Called for each processing request
4. **Cleanup**: `unload_model()` - Calls `cleanup()` and removes from memory

## Frontend Integration

### Automatic UI Generation

Hyperparameters defined in `get_hyperparameters_schema()` automatically appear in the frontend UI:

- **int/float**: Number input with slider
- **bool**: Checkbox
- **choice**: Dropdown menu
- **string**: Text input field

### API Endpoint Structure

```
POST /api/process
{
    "model": "my_model",
    "image_id": 123,
    "region_id": 456,
    "hyperparameters": {
        "threshold": 0.7,
        "enable_visualization": true
    }
}
```

Response:
```json
{
    "success": true,
    "result": {
        "str_result": "Detected 42 cells",
        "scores": {
            "total": 42
        }
    },
    "overlay_url": "/api/overlay/abc123.png"
}
```

## Testing Your Model

### Unit Testing

Create tests in `tests/test_my_model.py`:

```python
import pytest
from PIL import Image
from models.my_model_model import MyModel

def test_model_initialization():
    model = MyModel()
    config = {'model_path': 'path/to/weights.pth'}
    model.initialize(config)
    assert model._is_initialized

def test_model_processing():
    model = MyModel()
    model.initialize({'model_path': 'test_weights.pth'})
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # Process
    result = model.process(test_image)
    
    # Validate
    assert result['success']
    assert 'processed_image' in result
    assert 'str_result' in result
```

### Integration Testing

Test via API:

```python
import requests

response = requests.post(
    'http://localhost:5000/api/process',
    json={
        'model': 'my_model',
        'image_id': 123,
        'hyperparameters': {'threshold': 0.5}
    }
)

assert response.status_code == 200
result = response.json()
assert result['success']
```

## Best Practices

### 1. Resource Management

```python
def initialize(self, config):
    """Load model efficiently."""
    # Set device first
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model to device
    self.model = load_model(config['model_path'])
    self.model = self.model.to(self.device)
    self.model.eval()

def cleanup(self):
    """Always clean up properly."""
    if hasattr(self, 'model'):
        del self.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 2. Error Handling

```python
def process(self, image, mask=None, **kwargs):
    """Always handle errors gracefully."""
    try:
        # Validate input
        is_valid, error = self.validate_input(image, mask)
        if not is_valid:
            return {
                'success': False,
                'error': error,
                'processed_image': None,
                'scores': {},
                'str_result': f'Error: {error}'
            }
        
        # Process image
        result = self._internal_process(image)
        
        return {
            'success': True,
            'processed_image': result,
            'scores': self._compute_scores(result),
            'str_result': self._format_result(result)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'processed_image': None,
            'scores': {},
            'str_result': f'Processing Error: {str(e)}'
        }
```

### 3. Hyperparameter Handling

```python
def process(self, image, mask=None, annotation_points=None, 
            hyperparameters=None):
    """Merge user hyperparameters with defaults."""
    # Start with defaults
    params = self.hyperparameters.copy()
    
    # Override with user values
    if hyperparameters:
        params.update(hyperparameters)
    
    # Use params in processing
    threshold = params['threshold']
    batch_size = params['batch_size']
```

### 4. Visualization

```python
def _create_visualization(self, image, detections):
    """Create clear, informative overlays."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis, 'RGBA')
    
    # Use transparent overlays
    for detection in detections:
        # Semi-transparent bounding box
        draw.rectangle(
            detection['bbox'],
            outline=(255, 0, 0, 255),
            fill=(255, 0, 0, 50)
        )
    
    return vis
```

### 5. Documentation

```python
class MyModel(BaseAIModel):
    """
    One-line summary of what the model does.
    
    Detailed description:
    - What type of images does it process?
    - What does it detect/classify?
    - What are the outputs?
    
    Example:
        model = MyModel()
        model.initialize({'model_path': 'weights.pth'})
        result = model.process(image)
    
    References:
        Paper: https://arxiv.org/abs/...
        Code: https://github.com/...
    """
```

## Example: EC Cancer Model

See `models/ec_cancer_model.py` for a complete, production-ready example:

### Key Features:
- Multi-stage pipeline (SAM2 → ResNet50 → VarMIL)
- Patch-based processing with sliding window
- Grad-CAM visualization
- Comprehensive error handling
- GPU memory management
- Progress tracking

### Notable Implementation Details:

```python
class ECCancerModel(BaseAIModel):
    def __init__(self):
        super().__init__("ec_cancer", "1.0.0")
        self.description = "EC subtype classifier using VarMIL"
        self.requires_mask = True  # Needs tissue annotation
        
        # Multiple model components
        self.patch_classifier = None
        self.representation_generator = None
        self.varmil_model = None
        self.gradient_cam = None
    
    def initialize(self, config):
        """Load three model components."""
        # Load patch classifier (ResNet50)
        self.patch_classifier = torch.load(
            config['patch_classifier_model_path']
        )
        
        # Load feature extractor (ResNet34)
        self.representation_generator = VanillaModel("resnet34")
        self.representation_generator.load_state_dict(...)
        
        # Load VarMIL aggregator
        self.varmil_model = VarMIL("resnet34", 2)
        self.varmil_model.load_state_dict(...)
        
        # Initialize Grad-CAM for visualization
        self.gradient_cam = GradCAM(
            model=self.patch_classifier,
            target_layers=[self.patch_classifier.layer4[-1]]
        )
    
    def process(self, image, mask, annotation_points, hyperparameters):
        """Three-stage processing pipeline."""
        # Stage 1: Extract patches from tissue region
        patches = self._extract_patches(image, mask, hyperparameters)
        
        # Stage 2: Classify patches as tumor/normal
        tumor_patches = self._classify_patches(patches, hyperparameters)
        
        # Stage 3: Aggregate patch predictions
        classification = self._aggregate_predictions(tumor_patches)
        
        # Create visualization
        if hyperparameters.get('generate_gradcam'):
            overlay = self._generate_gradcam_overlay(tumor_patches)
        
        return {
            'success': True,
            'processed_image': overlay,
            'scores': {
                'classification': classification,
                'confidence': confidence,
                'tumor_patches': len(tumor_patches)
            },
            'str_result': f'{classification}: {confidence:.1%} confidence'
        }
```

## Troubleshooting

### Common Issues

**Issue**: Model not appearing in frontend
- Check model is registered: `registry.list_models()`
- Verify model name matches in registration and API calls
- Check `get_hyperparameters_schema()` returns valid dict

**Issue**: GPU out of memory
- Implement batch processing
- Use `torch.cuda.empty_cache()` in cleanup
- Reduce batch size in hyperparameters

**Issue**: Slow processing
- Profile your code to find bottlenecks
- Use batch processing instead of loops
- Move preprocessing outside model inference
- Consider model quantization/optimization

**Issue**: Inconsistent results
- Ensure model is in eval mode: `model.eval()`
- Disable dropout: `torch.no_grad()`
- Check random seed is set if needed
- Verify preprocessing is deterministic

## Additional Resources

- **Base Model**: `models/base_model.py`
- **Example Template**: `models/example_model.py`
- **Production Example**: `models/ec_cancer_model.py`
- **Registry**: `models/model_registry.py`
- **API Docs**: `API_DOCUMENTATION.md`

## Support

For questions or issues:
1. Check existing model implementations
2. Review API documentation
3. Test with `example_model.py` first
4. Open an issue with logs and code samples
