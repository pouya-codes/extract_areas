"""
Example AI Model Template

This is a template for creating new AI models. Copy this file and implement
the required methods for your specific model.

Steps to create a new model:
1. Copy this file to models/<your_model_name>_model.py
2. Rename the class to match your model
3. Implement all abstract methods
4. Register your model in app.py
5. Test your model with the provided endpoints

For detailed instructions, see CONTRIBUTING_AI_MODELS.md
"""

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
from models.base_model import BaseAIModel


class ExampleModel(BaseAIModel):
    """
    Example AI model implementation.
    
    Replace this with your model's description.
    Explain what your model does, what type of images it processes,
    and what outputs it produces.
    """
    
    def __init__(self):
        """
        Initialize your model.
        
        Set model name, version, and default configuration here.
        """
        super().__init__(
            model_name="example_model",
            model_version="1.0.0"
        )
        
        # Set model description
        self.description = (
            "Example model that demonstrates the interface. "
            "Replace with your model's description."
        )
        
        # Does your model require a mask/annotation?
        self.requires_mask = False
        
        # Set supported input formats
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG']
        
        # Default hyperparameters
        self.hyperparameters = {
            'threshold': 0.5,
            'min_size': 10,
            'max_size': 1000
        }
        
        # Your model's internal state
        self.model = None
        self.device = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize your model with configuration.
        
        This is called once when the model is loaded.
        Use this to:
        - Load model weights
        - Set up GPU/CPU
        - Initialize preprocessing pipelines
        - Validate configuration
        
        Args:
            config: Configuration dictionary with model-specific params
                Example: {
                    'model_path': '/path/to/weights.pth',
                    'device': 'cuda:0',
                    'batch_size': 8
                }
        
        Raises:
            ValueError: If config is invalid
            RuntimeError: If initialization fails
        """
        # Example: Validate required config
        if 'model_path' not in config:
            raise ValueError("model_path is required in config")
        
        model_path = config['model_path']
        device = config.get('device', 'cpu')
        
        # Example: Load your model
        # self.model = YourModelClass.load(model_path)
        # self.model.to(device)
        # self.model.eval()
        
        # For this example, we'll just store the config
        self.device = device
        
        # Mark as initialized
        self._is_initialized = True
        
        print(f"Initialized {self.model_name} on {device}")
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an image and return results.
        
        This is the main method that applies your AI model to an image.
        
        Args:
            image: Input image (PIL Image in RGB mode)
            mask: Optional binary mask (PIL Image in L mode)
            annotation_points: Optional polygon points [(x,y), ...]
            hyperparameters: Optional parameter overrides
        
        Returns:
            Dictionary with:
                - processed_image: Output visualization
                - scores: Quantitative metrics
                - success: Boolean
                - error: Error message (if failed)
                - metadata: Additional info (optional)
        """
        try:
            # Step 1: Validate inputs
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Step 2: Merge hyperparameters
            params = {**self.hyperparameters}
            if hyperparameters:
                params.update(hyperparameters)
            
            # Step 3: Create mask from annotation points if provided
            if annotation_points and len(annotation_points) >= 3:
                mask = self._create_mask_from_points(
                    annotation_points,
                    image.size
                )
            
            # Step 4: Preprocess image
            processed_img = self.preprocess_image(image, mask)
            
            # Step 5: Run your model inference
            # Replace this with your actual model code
            result_image, metrics = self._run_model(
                processed_img,
                params
            )
            
            # Step 6: Postprocess output
            final_image = self.postprocess_output(result_image, mask)
            
            # Step 7: Return results
            return {
                'processed_image': final_image,
                'scores': metrics,
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
    
    def _run_model(
        self,
        image: Image.Image,
        params: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Run your model inference.
        
        Replace this with your actual model code.
        
        Args:
            image: Preprocessed image
            params: Hyperparameters
        
        Returns:
            (result_image, metrics)
        """
        # Example: Dummy processing
        # In a real model, you would:
        # 1. Convert PIL Image to tensor
        # 2. Run model inference
        # 3. Convert output back to PIL Image
        # 4. Calculate metrics
        
        # For this example, just add a colored overlay
        result = image.copy()
        draw = ImageDraw.Draw(result, 'RGBA')
        width, height = result.size
        
        # Draw a semi-transparent rectangle
        draw.rectangle(
            [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            fill=(255, 0, 0, 64),
            outline=(255, 0, 0, 255),
            width=3
        )
        
        # Example metrics
        metrics = {
            'objects_detected': 42,
            'confidence': 0.95,
            'processing_time': 1.23
        }
        
        return result, metrics
    
    def _create_mask_from_points(
        self,
        points: List[Tuple[float, float]],
        size: Tuple[int, int]
    ) -> Image.Image:
        """
        Create a binary mask from polygon points.
        
        Args:
            points: List of (x, y) coordinates
            size: Image size (width, height)
        
        Returns:
            Binary mask as PIL Image
        """
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Clamp points to image bounds
        width, height = size
        clamped = [
            (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
            for x, y in points
        ]
        
        # Draw filled polygon
        draw.polygon(clamped, outline=255, fill=255)
        
        return mask
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Define your model's hyperparameters.
        
        Returns a schema describing all tunable parameters.
        """
        return {
            'threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Detection threshold (0.0 to 1.0)',
                'required': False
            },
            'min_size': {
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 10000,
                'description': 'Minimum object size in pixels',
                'required': False
            },
            'max_size': {
                'type': 'int',
                'default': 1000,
                'min': 1,
                'max': 100000,
                'description': 'Maximum object size in pixels',
                'required': False
            },
            'color_mode': {
                'type': 'choice',
                'default': 'RGB',
                'choices': ['RGB', 'LAB', 'HSV'],
                'description': 'Color space for processing',
                'required': False
            }
        }
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        
        Called when the model is unloaded.
        Use this to free GPU memory, close files, etc.
        """
        if self.model:
            # Example: Free GPU memory
            # del self.model
            # torch.cuda.empty_cache()
            pass
        
        self.model = None
        print(f"Cleaned up {self.model_name}")
