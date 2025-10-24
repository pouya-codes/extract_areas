# AI Model System - Complete Guide

> **A modular, extensible architecture for integrating multiple AI models into the pathology image processing platform.**

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Adding Your Own Model](#adding-your-own-model)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What Is This?

This system provides a **class-based architecture** for integrating AI processing models into the platform. It allows you to:

✅ Add new AI models without modifying existing code  
✅ Switch between models at runtime via API  
✅ Standardize model interfaces for consistency  
✅ Manage multiple models simultaneously  
✅ Configure models with hyperparameters  
✅ Test and deploy models independently  

### Key Features

- **Abstract Base Class**: All models implement a common interface (`BaseAIModel`)
- **Model Registry**: Centralized management of model lifecycle
- **FastAPI Integration**: RESTful endpoints for all operations
- **Mask Support**: Built-in support for tissue region masking
- **Hyperparameter Configuration**: Dynamic model configuration
- **Type Safety**: Full type hints for better development experience

### Files Overview

```
models/
├── __init__.py                 # Package exports
├── base_model.py              # Abstract base class (265 lines)
├── model_registry.py          # Model management system (159 lines)
├── deepliif_model.py          # DeepLIIF implementation (217 lines)
├── example_model.py           # Template for new models (324 lines)
└── README.md                  # Models directory documentation

app_refactored.py              # FastAPI app with model integration (692 lines)
```

---

## Quick Start

### Step 1: Create Your Model (5 minutes)

```bash
# Copy the template
cd models/
cp example_model.py my_model.py
```

Edit `models/my_model.py`:

```python
from models.base_model import BaseAIModel
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

class MyModel(BaseAIModel):
    """Your custom AI model."""
    
    def __init__(self):
        super().__init__(
            model_name="my_model",
            model_version="1.0.0"
        )
        self.description = "My awesome AI model"
        self.requires_mask = True  # Set to False if mask not needed
        self.model = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Load your model weights."""
        model_path = config['model_path']
        # TODO: Load your model
        # self.model = YourModelLoader.load(model_path)
        self._is_initialized = True
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an image and return results."""
        try:
            # Validate
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Apply mask if provided
            if mask:
                processed_image = self.preprocess_image(image, mask)
            else:
                processed_image = image
            
            # TODO: Run your model inference
            # output = self.model.predict(processed_image)
            # result_image = self.convert_output_to_image(output)
            
            result_image = processed_image  # Placeholder
            scores = {'confidence': 0.95}
            
            return {
                'success': True,
                'processed_image': result_image,
                'scores': scores
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """Define configurable parameters."""
        return {
            'threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Detection threshold'
            }
        }
```

### Step 2: Register Your Model (2 minutes)

Edit `app_refactored.py` in the `initialize_models()` function:

```python
def initialize_models():
    # ... existing models ...
    
    # Add your model
    from models.my_model import MyModel
    model_registry.register_model_class("my_model", MyModel)
    
    my_model_config = {
        'model_path': '/path/to/your/model/weights.pth'
    }
    model_registry.load_model("my_model", my_model_config)
    print("✓ Loaded MyModel")
```

### Step 3: Test Your Model (2 minutes)

```bash
# Start the server
python app_refactored.py

# In another terminal, test it
curl http://localhost:8000/models/list

# Should show your model:
# {"registered_models": ["deepliif", "my_model"], ...}
```

### Step 4: Use Your Model

```python
import requests
import json
from io import BytesIO
from PIL import Image

# Create test image
img = Image.new('RGB', (256, 256), 'white')
buffer = BytesIO()
img.save(buffer, format='JPEG')
buffer.seek(0)

# Call API with your model
files = {'region': ('test.jpg', buffer, 'image/jpeg')}
data = {
    'mask': json.dumps([[50,50], [200,50], [200,200], [50,200]]),
    'model_name': 'my_model'  # ← Your model!
}

response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files=files,
    data=data
)

result = response.json()
print(f"Success: {result['status']}")
print(f"Model used: {result['model_used']}")
```

**That's it! Your model is now integrated.** 🎉

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Aimviewer Frontend (Django)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP REST API
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (app_refactored.py)                 │
│                                                                  │
│  Endpoints:                                                      │
│  • /generate_mask          • /process_region_annotation         │
│  • /extract_regions        • /process_region                    │
│  • /auto_name_regions      • /models/*                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│               Model Registry (model_registry.py)                 │
│                                                                  │
│  • Register models          • Load/unload models                │
│  • Model lookup             • Lifecycle management              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  DeepLIIF Model  │  │   Your Model     │  │  Future Models   │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
           ↓                      ↓                      ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ DeepLIIF Engine  │  │ Your AI Engine   │  │ Future AI Engine │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Request Flow

```
1. Frontend draws annotation polygon on tissue image
   └─ Points: [(x₁,y₁), (x₂,y₂), ...]

2. Frontend sends POST to /process_region_annotation
   ├─ region: JPEG image
   ├─ mask: JSON points
   └─ model_name: "my_model"

3. Backend parses request
   ├─ Validates JPEG format
   ├─ Parses annotation points
   └─ Creates binary mask from points

4. Backend calls model.process()
   └─ Passes image + mask to selected model

5. Model processes image
   ├─ Applies mask (white out non-annotated areas)
   ├─ Runs inference
   └─ Returns overlay + scores

6. Backend converts result to base64
   └─ Returns JSON with processed image

7. Frontend displays result
   ├─ Shows overlay on slide
   └─ Displays scores/metrics
```

### BaseAIModel Interface

All models must implement this interface:

```python
class BaseAIModel(ABC):
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Load model weights and initialize resources."""
        pass
    
    @abstractmethod
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process image and return results.
        
        Returns:
            {
                'success': bool,
                'processed_image': PIL.Image,
                'scores': Dict[str, Any],
                'error': Optional[str]
            }
        """
        pass
    
    @abstractmethod
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """Return schema for configurable parameters."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method."""
        pass
```

---

## Adding Your Own Model

### Complete Example: Object Detection Model

Here's a complete example of integrating a PyTorch object detection model:

```python
"""
Object Detection Model

Detects objects in pathology images using a custom PyTorch model.
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import numpy as np

from models.base_model import BaseAIModel


class ObjectDetectionModel(BaseAIModel):
    """Object detection model for pathology images."""
    
    def __init__(self):
        super().__init__(
            model_name="object_detector",
            model_version="1.0.0"
        )
        self.description = "Detects objects in pathology images"
        self.requires_mask = False  # Can work with or without mask
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG']
        
        self.model = None
        self.device = None
        self.transform = None
        
        # Default hyperparameters
        self.hyperparameters = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'max_detections': 100
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the object detection model."""
        model_path = config.get('model_path')
        if not model_path:
            raise ValueError("model_path is required")
        
        # Setup device
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Loaded {self.model_name} on {self.device}")
        self._is_initialized = True
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect objects in the image."""
        try:
            # Validate
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Merge hyperparameters
            params = {**self.hyperparameters}
            if hyperparameters:
                params.update(hyperparameters)
            
            # Apply mask if provided
            if mask:
                processed_input = self.preprocess_image(image, mask)
            else:
                processed_input = image
            
            # Run inference
            with torch.no_grad():
                # Transform image
                input_tensor = self.transform(processed_input).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                
                # Predict
                predictions = self.model(input_tensor)
                
                # Post-process predictions
                boxes, labels, scores = self._postprocess_predictions(
                    predictions,
                    params['confidence_threshold'],
                    params['nms_threshold'],
                    params['max_detections']
                )
            
            # Draw results on image
            result_image = self._draw_detections(
                processed_input.copy(),
                boxes,
                labels,
                scores
            )
            
            # Calculate scores
            scores_dict = {
                'total_detections': len(boxes),
                'average_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0,
                'detections_by_class': self._count_by_class(labels)
            }
            
            # Apply mask to output if provided
            if mask:
                result_image = self.postprocess_output(result_image, mask)
            
            return {
                'success': True,
                'processed_image': result_image,
                'scores': scores_dict,
                'metadata': {
                    'model': self.model_name,
                    'version': self.model_version,
                    'device': str(self.device),
                    'hyperparameters': params
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Detection failed: {str(e)}"
            }
    
    def _postprocess_predictions(self, predictions, conf_thresh, nms_thresh, max_det):
        """Post-process model predictions."""
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= conf_thresh
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Apply NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            from torchvision.ops import nms
            keep_indices = nms(
                torch.tensor(boxes),
                torch.tensor(scores),
                nms_thresh
            )
            boxes = boxes[keep_indices]
            labels = labels[keep_indices]
            scores = scores[keep_indices]
        
        # Limit detections
        if len(boxes) > max_det:
            boxes = boxes[:max_det]
            labels = labels[:max_det]
            scores = scores[:max_det]
        
        return boxes, labels, scores
    
    def _draw_detections(self, image, boxes, labels, scores):
        """Draw bounding boxes on image."""
        draw = ImageDraw.Draw(image)
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Draw label
            text = f"Class {label}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill='red')
        
        return image
    
    def _count_by_class(self, labels):
        """Count detections by class."""
        unique, counts = np.unique(labels, return_counts=True)
        return {f"class_{int(cls)}": int(count) 
                for cls, count in zip(unique, counts)}
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """Get hyperparameters schema."""
        return {
            'confidence_threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Minimum confidence for detections'
            },
            'nms_threshold': {
                'type': 'float',
                'default': 0.4,
                'min': 0.0,
                'max': 1.0,
                'description': 'Non-maximum suppression threshold'
            },
            'max_detections': {
                'type': 'int',
                'default': 100,
                'min': 1,
                'max': 1000,
                'description': 'Maximum number of detections to return'
            }
        }
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
```

### Register and Use

```python
# In app_refactored.py
from models.object_detection_model import ObjectDetectionModel

def initialize_models():
    # ... existing models ...
    
    model_registry.register_model_class("object_detector", ObjectDetectionModel)
    detector_config = {
        'model_path': 'models/detector/weights.pth',
        'device': 'cuda'
    }
    model_registry.load_model("object_detector", detector_config)
    print("✓ Loaded Object Detector")
```

---

## Deployment

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python app_refactored.py

# Server runs on http://localhost:8000
```

### Production Deployment

#### Option 1: Direct Deployment

```bash
# Install production server
pip install uvicorn[standard] gunicorn

# Run with Gunicorn
gunicorn app_refactored:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

#### Option 2: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app_refactored.py"]
```

Build and run:

```bash
docker build -t pathology-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models pathology-api
```

#### Option 3: Systemd Service

Create `/etc/systemd/system/pathology-api.service`:

```ini
[Unit]
Description=Pathology Image Processing API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/pathology-api
Environment="PATH=/opt/pathology-api/venv/bin"
ExecStart=/opt/pathology-api/venv/bin/python app_refactored.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable pathology-api
sudo systemctl start pathology-api
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

All endpoints are relative to this base URL.

---

### Model Discovery & Information

#### 1. List All Models

**Endpoint:** `GET /models/list`

Returns all registered and loaded models in the system.

**Request:**
```bash
curl http://localhost:8000/models/list
```

**Response:**
```json
{
  "registered_models": ["deepliif", "example_model"],
  "loaded_models": ["deepliif"]
}
```

**Fields:**
- `registered_models`: Models that have been registered but may not be initialized
- `loaded_models`: Models that are loaded, initialized, and ready to use

---

#### 2. Get Model Information

**Endpoint:** `GET /models/{model_name}/info`

Returns detailed information about a specific model.

**Request:**
```bash
curl http://localhost:8000/models/deepliif/info
```

**Response:**
```json
{
  "name": "deepliif",
  "version": "1.0.0",
  "description": "DeepLIIF: Deep-Learning Inferred Multiplex ImmunoFluorescence for IHC Image Analysis",
  "requires_mask": true,
  "supported_formats": ["PNG", "JPEG", "JPG"],
  "initialized": true,
  "hyperparameters": {
    "eager_mode": {
      "type": "bool",
      "default": false,
      "description": "Use eager mode for single GPU processing"
    },
    "color_dapi": {
      "type": "bool",
      "default": false,
      "description": "Whether to color DAPI in output"
    },
    "positive_threshold": {
      "type": "float",
      "default": 0.5,
      "min": 0.0,
      "max": 1.0,
      "description": "Threshold for positive cell detection"
    }
  }
}
```

**Fields:**
- `name`: Unique identifier for the model
- `version`: Model version (semantic versioning)
- `description`: Human-readable description
- `requires_mask`: Whether model needs tissue mask for processing
- `supported_formats`: List of supported image formats
- `initialized`: Whether model is loaded and ready
- `hyperparameters`: Available configuration options (see below)

**Status Codes:**
- `200 OK`: Model found and information returned
- `404 Not Found`: Model not registered in the system

---

#### 3. Get Model Hyperparameters

**Endpoint:** `GET /models/{model_name}/hyperparameters`

Returns schema for model-specific hyperparameters.

**Request:**
```bash
curl http://localhost:8000/models/deepliif/hyperparameters
```

**Response:**
```json
{
  "model": "deepliif",
  "hyperparameters": {
    "eager_mode": {
      "type": "bool",
      "default": false,
      "description": "Use eager mode for single GPU processing. Set to True for single GPU, False for multi-GPU"
    },
    "color_dapi": {
      "type": "bool",
      "default": false,
      "description": "Whether to color DAPI (nucleus) channel in output images"
    },
    "color_marker": {
      "type": "bool",
      "default": false,
      "description": "Whether to color marker channel in output images"
    },
    "positive_threshold": {
      "type": "float",
      "default": 0.5,
      "min": 0.0,
      "max": 1.0,
      "description": "Threshold value for classifying cells as positive (0.0-1.0)"
    }
  }
}
```

**Hyperparameter Types:**

| Type | Description | UI Control | Example |
|------|-------------|------------|---------|
| `bool` | Boolean flag | Checkbox | `{"eager_mode": true}` |
| `int` | Integer number | Number input | `{"batch_size": 8}` |
| `float` | Floating point | Number input | `{"threshold": 0.5}` |
| `string` | Text value | Text input | `{"output_format": "png"}` |
| `choice` | Enum selection | Dropdown | `{"device": "cuda"}` |

**Status Codes:**
- `200 OK`: Hyperparameters returned
- `404 Not Found`: Model not found

---

### Image Processing

#### 4. Process Region with Annotation Points

**Endpoint:** `POST /process_region_annotation`

Process a tissue region with polygon annotation points. The backend converts points to a binary mask automatically.

**Request:**
```bash
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@tissue_region.jpg" \
  -F 'mask=[{"x":100,"y":100},{"x":500,"y":100},{"x":500,"y":400},{"x":100,"y":400}]' \
  -F "region_id=tissue_001" \
  -F "model_name=deepliif" \
  -F 'hyperparameters={"eager_mode": false, "positive_threshold": 0.5}'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `region` | File | Yes | Tissue region image (JPEG, PNG) |
| `mask` | JSON string | Yes | Array of polygon points `[{"x":x1,"y":y1}, ...]` |
| `region_id` | String | No | Identifier for this region (default: "unknown") |
| `model_name` | String | No | Model to use (default: "deepliif") |
| `hyperparameters` | JSON string | No | Model-specific parameters (default: `{}`) |

**Response:**
```json
{
  "status": "success",
  "processed_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...(base64 encoded PNG)",
  "overlay_base64": "iVBORw0KGgoAAAANSUhEUgAA...(base64 encoded overlay)",
  "score": {
    "total_cells": 1247,
    "positive_cells": 342,
    "positive_percentage": 27.42,
    "region_area_pixels": 250000
  },
  "region_id": "tissue_001",
  "model_used": "deepliif",
  "model_version": "1.0.0",
  "processing_time_seconds": 3.42,
  "hyperparameters_used": {
    "eager_mode": false,
    "positive_threshold": 0.5
  }
}
```

**Response Fields:**
- `status`: "success" or "error"
- `processed_image_base64`: Base64-encoded result image (PNG)
- `overlay_base64`: Base64-encoded overlay visualization
- `score`: Model-specific quantitative results
- `region_id`: Echo of the input region ID
- `model_used`: Model that processed the image
- `model_version`: Version of the model used
- `processing_time_seconds`: Time taken to process
- `hyperparameters_used`: Parameters used for this processing

**Error Response:**
```json
{
  "status": "error",
  "error": "Model not found: unknown_model",
  "region_id": "tissue_001"
}
```

**Status Codes:**
- `200 OK`: Processing successful
- `400 Bad Request`: Invalid parameters or image format
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Processing failed

---

#### 5. Process Region with Mask Image

**Endpoint:** `POST /process_region`

Process a tissue region with a separate mask image file.

**Request:**
```bash
curl -X POST http://localhost:8000/process_region \
  -F "region=@tissue.png" \
  -F "mask=@mask.png" \
  -F "model_name=deepliif" \
  -F 'hyperparameters={"color_dapi": true}'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `region` | File | Yes | Tissue region image |
| `mask` | File | Yes | Binary mask image (white=tissue, black=background) |
| `model_name` | String | No | Model to use (default: "deepliif") |
| `hyperparameters` | JSON string | No | Model-specific parameters |

**Response:** Same format as `/process_region_annotation`

**Status Codes:** Same as `/process_region_annotation`

---

### Other Endpoints

#### 6. Generate Tissue Mask

**Endpoint:** `POST /generate_mask`

Generate binary tissue mask from whole slide image using SAM2.

**Request:**
```bash
curl -X POST http://localhost:8000/generate_mask \
  -F "file=@slide.tif"
```

**Response:**
```json
{
  "status": "success",
  "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mask_shape": [1024, 1024],
  "tissue_percentage": 42.5
}
```

---

#### 7. Extract Tissue Regions

**Endpoint:** `POST /extract_regions`

Extract individual tissue regions from slide using mask.

**Request:**
```bash
curl -X POST http://localhost:8000/extract_regions \
  -F "slide=@slide.tif" \
  -F "mask=@mask.png"
```

**Response:**
```json
{
  "status": "success",
  "regions": [
    {
      "region_id": 0,
      "bbox": [100, 200, 300, 400],
      "image_base64": "iVBORw0KGgo..."
    }
  ],
  "total_regions": 12
}
```

---

#### 8. Auto-Name Regions

**Endpoint:** `POST /auto_name_regions`

Generate descriptive names for tissue regions using AI.

**Request:**
```bash
curl -X POST http://localhost:8000/auto_name_regions \
  -F "file=@region.jpg"
```

**Response:**
```json
{
  "status": "success",
  "name": "Dense lymphocytic infiltrate",
  "confidence": 0.87
}
```

---

### Integration Examples

#### Python Client

```python
import requests
import json
from io import BytesIO
from PIL import Image

class ModelAPIClient:
    """Client for model processing API."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def list_models(self):
        """Get available models."""
        response = requests.get(f"{self.base_url}/models/list")
        return response.json()
    
    def get_model_info(self, model_name):
        """Get model information."""
        response = requests.get(f"{self.base_url}/models/{model_name}/info")
        return response.json()
    
    def process_region(
        self,
        image: Image.Image,
        annotation_points: list,
        model_name: str = "deepliif",
        hyperparameters: dict = None,
        region_id: str = "unknown"
    ):
        """Process tissue region with annotation."""
        # Convert image to bytes
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Prepare files and data
        files = {'region': ('region.jpg', buffer, 'image/jpeg')}
        data = {
            'mask': json.dumps(annotation_points),
            'region_id': region_id,
            'model_name': model_name
        }
        
        if hyperparameters:
            data['hyperparameters'] = json.dumps(hyperparameters)
        
        # Send request
        response = requests.post(
            f"{self.base_url}/process_region_annotation",
            files=files,
            data=data
        )
        
        return response.json()

# Usage example
client = ModelAPIClient()

# List available models
models = client.list_models()
print(f"Available models: {models['loaded_models']}")

# Get model info
info = client.get_model_info("deepliif")
print(f"Model: {info['name']} v{info['version']}")
print(f"Description: {info['description']}")

# Process a region
image = Image.open("tissue_region.jpg")
points = [
    {"x": 100, "y": 100},
    {"x": 500, "y": 100},
    {"x": 500, "y": 400},
    {"x": 100, "y": 400}
]

result = client.process_region(
    image=image,
    annotation_points=points,
    model_name="deepliif",
    hyperparameters={
        "eager_mode": False,
        "positive_threshold": 0.6
    },
    region_id="tissue_001"
)

print(f"Status: {result['status']}")
print(f"Total cells: {result['score']['total_cells']}")
print(f"Positive cells: {result['score']['positive_cells']}")
```

#### JavaScript/Fetch

```javascript
// Get model list
async function listModels() {
  const response = await fetch('http://localhost:8000/models/list');
  const data = await response.json();
  return data.loaded_models;
}

// Process region
async function processRegion(imageFile, annotationPoints, modelName = 'deepliif') {
  const formData = new FormData();
  formData.append('region', imageFile);
  formData.append('mask', JSON.stringify(annotationPoints));
  formData.append('model_name', modelName);
  formData.append('hyperparameters', JSON.stringify({
    eager_mode: false,
    positive_threshold: 0.5
  }));
  
  const response = await fetch('http://localhost:8000/process_region_annotation', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Usage
const models = await listModels();
console.log('Available models:', models);

const fileInput = document.getElementById('imageInput');
const result = await processRegion(
  fileInput.files[0],
  [{x: 100, y: 100}, {x: 500, y: 100}, {x: 500, y: 400}, {x: 100, y: 400}],
  'deepliif'
);

console.log('Processing result:', result);
```

#### cURL Examples

```bash
# Get all models
curl http://localhost:8000/models/list

# Get DeepLIIF info
curl http://localhost:8000/models/deepliif/info

# Process with default settings
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@tissue.jpg" \
  -F 'mask=[{"x":100,"y":100},{"x":500,"y":100},{"x":500,"y":400},{"x":100,"y":400}]'

# Process with custom hyperparameters
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@tissue.jpg" \
  -F 'mask=[{"x":100,"y":100},{"x":500,"y":100},{"x":500,"y":400},{"x":100,"y":400}]' \
  -F "model_name=deepliif" \
  -F 'hyperparameters={"eager_mode": false, "color_dapi": true, "positive_threshold": 0.6}'
```

---

## Best Practices

### 1. Model Initialization

```python
def initialize(self, config: Dict[str, Any]) -> None:
    """Best practices for initialization."""
    # Validate required config
    required = ['model_path']
    for key in required:
        if key not in config:
            raise ValueError(f"{key} is required in config")
    
    # Use absolute paths
    model_path = Path(config['model_path']).absolute()
    
    # Handle GPU/CPU gracefully
    self.device = torch.device(
        config.get('device', 
                  'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Log initialization
    print(f"Initializing {self.model_name}")
    print(f"  Device: {self.device}")
    print(f"  Model path: {model_path}")
    
    # Set initialized flag
    self._is_initialized = True
```

### 2. Error Handling

```python
def process(self, image, **kwargs) -> Dict[str, Any]:
    """Comprehensive error handling."""
    try:
        # Validate inputs
        is_valid, error = self.validate_input(image, kwargs.get('mask'))
        if not is_valid:
            return {'success': False, 'error': error}
        
        # Process image
        result = self._run_inference(image)
        
        return {
            'success': True,
            'processed_image': result,
            'scores': {}
        }
        
    except torch.cuda.OutOfMemoryError:
        return {
            'success': False,
            'error': 'GPU out of memory. Try reducing image size or batch size.'
        }
    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': f"Processing failed: {str(e)}"
        }
```

### 3. Memory Management

```python
def process(self, image, **kwargs):
    """Efficient memory management."""
    # Use context manager for GPU operations
    with torch.no_grad():
        input_tensor = self.preprocess(image)
        
        # Move to GPU only when needed
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        output = self.model(input_tensor)
        
        # Move back to CPU immediately
        output = output.cpu()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output

def cleanup(self) -> None:
    """Cleanup resources."""
    if hasattr(self, 'model') and self.model:
        del self.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 4. Input Validation

```python
def validate_input(
    self,
    image: Image.Image,
    mask: Optional[Image.Image] = None
) -> Tuple[bool, Optional[str]]:
    """Custom validation."""
    # Call parent validation
    is_valid, error = super().validate_input(image, mask)
    if not is_valid:
        return False, error
    
    # Additional custom validation
    if image.size[0] > 10000 or image.size[1] > 10000:
        return False, "Image too large. Max size: 10000x10000"
    
    if image.mode not in ['RGB', 'RGBA']:
        return False, f"Unsupported image mode: {image.mode}"
    
    return True, None
```

### 5. Hyperparameter Handling

```python
def process(self, image, hyperparameters=None, **kwargs):
    """Merge and validate hyperparameters."""
    # Start with defaults
    params = self.hyperparameters.copy()
    
    # Merge user-provided parameters
    if hyperparameters:
        params.update(hyperparameters)
    
    # Validate parameters
    schema = self.get_hyperparameters_schema()
    for key, value in params.items():
        if key in schema:
            param_def = schema[key]
            
            # Type checking
            if param_def['type'] == 'float':
                params[key] = float(value)
                # Range checking
                if 'min' in param_def:
                    params[key] = max(param_def['min'], params[key])
                if 'max' in param_def:
                    params[key] = min(param_def['max'], params[key])
    
    return params
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Symptom:** Model doesn't appear in `/models/list`

**Solutions:**
- Check import statement in `app_refactored.py`
- Verify `register_model_class()` is called
- Check `load_model()` is called with correct config
- Look for errors in terminal during startup
- Check model file path exists

#### 2. Out of Memory Errors

**Symptom:** `torch.cuda.OutOfMemoryError` or slow processing

**Solutions:**
```python
# Process in tiles
def process_large_image(self, image):
    tiles = self.split_into_tiles(image, tile_size=512)
    results = []
    for tile in tiles:
        with torch.no_grad():
            result = self.model(tile)
        results.append(result.cpu())  # Move to CPU immediately
        torch.cuda.empty_cache()
    return self.merge_tiles(results)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = self.model(input)

# Reduce batch size
self.hyperparameters['batch_size'] = 1
```

#### 3. Slow Inference

**Symptom:** Processing takes too long

**Solutions:**
- Use GPU: `device='cuda'`
- Enable model optimization: `model.eval()`
- Use TorchScript: `torch.jit.script(model)`
- Reduce input resolution
- Use model quantization
- Enable async processing

#### 4. Incorrect Results

**Symptom:** Model produces wrong output

**Solutions:**
- Verify preprocessing matches training
- Check normalization values
- Validate mask application
- Test with known good examples
- Check model version/weights
- Verify hyperparameters

#### 5. API Errors

**Symptom:** HTTP 400/500 errors

**Solutions:**
```bash
# Check logs
tail -f /var/log/pathology-api.log

# Test endpoints individually
curl -X GET http://localhost:8000/models/list
curl -X GET http://localhost:8000/models/my_model/info

# Validate JSON
echo '{"test": "value"}' | python -m json.tool

# Check file formats
file test.jpg  # Should be JPEG
```

### Debug Mode

Enable debug mode in `app_refactored.py`:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add to your model
def process(self, image, **kwargs):
    logging.debug(f"Processing image: {image.size}, mode: {image.mode}")
    logging.debug(f"Hyperparameters: {kwargs.get('hyperparameters')}")
    
    try:
        result = self._run_inference(image)
        logging.debug(f"Processing successful")
        return result
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        raise
```

---

## Summary

This system provides a **robust, extensible architecture** for integrating AI models into your pathology image processing platform. Key benefits:

✅ **Easy Integration**: Copy template, implement 3 methods, register model  
✅ **Standardized Interface**: All models work the same way  
✅ **Runtime Flexibility**: Switch models via API parameter  
✅ **Production Ready**: Error handling, logging, resource management  
✅ **Well Documented**: Complete examples and API reference  

**Need Help?**
- Check `models/example_model.py` for complete template
- Review `models/deepliif_model.py` for real implementation
- Test with provided examples
- Enable debug logging for troubleshooting

**Ready to add your model?** Start with the [Quick Start](#quick-start) section above! 🚀
