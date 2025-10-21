# Quick Start: Adding Your First AI Model

This is a 10-minute guide to add your first AI model to the platform.

## Prerequisites

✓ Python 3.7+
✓ FastAPI installed
✓ Your AI model code/weights ready

## Step 1: Copy the Template (30 seconds)

```bash
cd models/
cp example_model.py my_awesome_model.py
```

## Step 2: Customize Your Model (5 minutes)

Edit `models/my_awesome_model.py`:

```python
from models.base_model import BaseAIModel
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

class MyAwesomeModel(BaseAIModel):
    
    def __init__(self):
        super().__init__(
            model_name="my_awesome_model",
            model_version="1.0.0"
        )
        self.description = "My awesome AI model for image analysis"
        self.requires_mask = True  # Set to False if no mask needed
        self.model = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Load your model here"""
        model_path = config['model_path']
        
        # TODO: Replace with your model loading code
        # self.model = YourModelLoader.load(model_path)
        # self.model.eval()
        
        self._is_initialized = True
        print(f"Loaded {self.model_name} from {model_path}")
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an image and return results"""
        try:
            # Validate inputs
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Apply mask if provided
            if mask:
                processed_image = self.preprocess_image(image, mask)
            else:
                processed_image = image
            
            # TODO: Replace with your model inference
            # output = self.model(processed_image)
            # result_image = convert_to_pil(output)
            # scores = calculate_scores(output)
            
            # For now, return the input image
            result_image = processed_image
            scores = {
                'confidence': 0.95,
                'objects_detected': 10
            }
            
            return {
                'success': True,
                'processed_image': result_image,
                'scores': scores,
                'metadata': {
                    'model': self.model_name,
                    'version': self.model_version
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """Define your model's parameters"""
        return {
            'threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Detection confidence threshold'
            },
            'max_detections': {
                'type': 'int',
                'default': 100,
                'min': 1,
                'max': 1000,
                'description': 'Maximum number of objects to detect'
            }
        }
```

## Step 3: Register Your Model (2 minutes)

Add to `app.py` (or `app_refactored.py`):

```python
# At the top, add import
from models.my_awesome_model import MyAwesomeModel

# In initialize_models() function, add:
def initialize_models():
    # ... existing models ...
    
    # Register your model
    model_registry.register_model_class("my_awesome_model", MyAwesomeModel)
    
    # Load your model
    my_model_config = {
        'model_path': '/path/to/your/model/weights.pth',
        'device': 'cuda:0'  # or 'cpu'
    }
    model_registry.load_model("my_awesome_model", my_model_config)
    print("✓ Loaded my_awesome_model")
```

## Step 4: Test It! (2 minutes)

### Start the server:
```bash
python app.py
```

### Test via curl:
```bash
# List models (should see your model)
curl http://localhost:8000/models/list

# Get model info
curl http://localhost:8000/models/my_awesome_model/info

# Process an image
curl -X POST "http://localhost:8000/process_region_annotation" \
  -F "region=@test_image.jpg" \
  -F "mask={\"points\": [[0,0], [512,0], [512,512], [0,512]]}" \
  -F "model_name=my_awesome_model"
```

### Or test with Python:
```python
import requests
import json

# Test model list
response = requests.get('http://localhost:8000/models/list')
print("Available models:", response.json())

# Test processing
files = {'region': open('test_image.jpg', 'rb')}
data = {
    'mask': json.dumps({
        'points': [[0, 0], [512, 0], [512, 512], [0, 512]]
    }),
    'model_name': 'my_awesome_model'
}

response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files=files,
    data=data
)

result = response.json()
print("Processing result:", result['status'])
if result['status'] == 'success':
    print("Scores:", result['score'])
```

## Step 5: Implement Your Logic

Now go back to Step 2 and replace the TODO sections with your actual model code:

### In `initialize()`:
```python
def initialize(self, config: Dict[str, Any]) -> None:
    import torch
    from your_model import YourModelClass
    
    model_path = config['model_path']
    device = config.get('device', 'cpu')
    
    # Load your model
    self.model = YourModelClass()
    self.model.load_state_dict(torch.load(model_path))
    self.model.to(device)
    self.model.eval()
    
    self._is_initialized = True
```

### In `process()`:
```python
def process(self, image, mask=None, **kwargs):
    try:
        # Preprocess
        if mask:
            processed = self.preprocess_image(image, mask)
        else:
            processed = image
        
        # Convert to tensor
        tensor = your_transform(processed)
        
        # Run model
        with torch.no_grad():
            output = self.model(tensor)
        
        # Convert back to PIL
        result_image = tensor_to_pil(output)
        
        # Calculate metrics
        scores = {
            'metric1': calculate_metric1(output),
            'metric2': calculate_metric2(output)
        }
        
        return {
            'success': True,
            'processed_image': result_image,
            'scores': scores
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

## Common Patterns

### Pattern 1: Using PyTorch Model
```python
import torch

def initialize(self, config):
    self.model = torch.load(config['model_path'])
    self.model.eval()
    self.device = config.get('device', 'cpu')
    self.model.to(self.device)
```

### Pattern 2: Using TensorFlow Model
```python
import tensorflow as tf

def initialize(self, config):
    self.model = tf.keras.models.load_model(config['model_path'])
```

### Pattern 3: Using ONNX Model
```python
import onnxruntime as ort

def initialize(self, config):
    self.session = ort.InferenceSession(config['model_path'])
```

### Pattern 4: Converting Between PIL and NumPy
```python
import numpy as np
from PIL import Image

# PIL to NumPy
np_array = np.array(pil_image)

# NumPy to PIL
pil_image = Image.fromarray(np_array.astype('uint8'))
```

## Troubleshooting

### "Model not found"
- Check that you called `register_model_class()` before `load_model()`
- Verify the model name matches in both places

### "Model failed to initialize"
- Check the model path is correct
- Verify GPU is available if using CUDA
- Check all dependencies are installed

### "Processing failed"
- Add try-except blocks with detailed error messages
- Test with a simple image first
- Check input/output dimensions

## Next Steps

1. ✅ Read the full documentation: `CONTRIBUTING_AI_MODELS.md`
2. ✅ Look at real example: `models/deepliif_model.py`
3. ✅ Add tests for your model
4. ✅ Configure hyperparameters
5. ✅ Optimize performance

## Get Help

- Check the example: `models/example_model.py`
- Read the docs: `CONTRIBUTING_AI_MODELS.md`
- See working code: `models/deepliif_model.py`

Congratulations! You've added your first AI model! 🎉
