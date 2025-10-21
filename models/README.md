# AI Models Directory

This directory contains all AI processing models for the platform using a modular, class-based architecture.

## Structure

```
models/
├── __init__.py              # Package initialization
├── base_model.py            # Abstract base class for all models
├── model_registry.py        # Model registry and lifecycle management
├── deepliif_model.py        # DeepLIIF implementation
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
       
       def process(self, image, mask=None, ...):
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
- **Hyperparameters:**
  - `eager_mode` (bool): Single GPU mode
  - `color_dapi` (bool): Apply DAPI coloring
  - `color_marker` (bool): Apply marker coloring

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
- `process(image, mask, ...)`: Process an image and return results
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

## Testing

Run model tests:
```bash
pytest tests/test_models.py
```

Test a specific model:
```bash
python -m models.deepliif_model
```

## Documentation

For detailed instructions on adding new models, see:
**[CONTRIBUTING_AI_MODELS.md](../CONTRIBUTING_AI_MODELS.md)**

## Support

- Check existing model implementations for examples
- Review the BaseAIModel documentation
- See test files for usage examples
