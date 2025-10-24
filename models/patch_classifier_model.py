"""
Patch Classifier Model

This module wraps the PatchClassifier into the BaseAIModel interface,
enabling it to be used within the aimviewer model registry system.

The PatchClassifier performs sliding window analysis with a ResNet50 backbone
and optional GradCAM visualization for tissue region classification.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np
import torch
import io
import base64

# Add src directory to path to import PatchClassifier
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.base_model import BaseAIModel
from patch_classifier import PatchClassifier


class PatchClassifierModel(BaseAIModel):
    """
    Patch-based tissue classifier model using ResNet50.
    
    This model performs sliding window classification on tissue regions,
    identifying positive and negative patches. Optionally generates
    GradCAM heatmaps for visualization.
    
    Features:
    - ResNet50 backbone trained for binary classification
    - Sliding window analysis with configurable patch size
    - Batch processing for efficiency
    - Optional GradCAM visualization
    - Confidence thresholding
    """
    
    def __init__(self):
        """Initialize the PatchClassifier model wrapper."""
        super().__init__(
            model_name="patch_classifier",
            model_version="1.0.0"
        )
        self.description = (
            "Patch-based tissue classifier using ResNet50 backbone. "
            "Performs sliding window analysis to identify positive/negative regions "
            "with optional GradCAM visualization."
        )
        self.requires_mask = True
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG', 'TIFF']
        
        # Define hyperparameters schema
        self.hyperparameters = {
            'patch_size': {
                'type': 'int',
                'default': 64,
                'min': 32,
                'max': 256,
                'description': 'Size of sliding window patches (pixels)'
            },
            'batch_size': {
                'type': 'int',
                'default': 32,
                'min': 1,
                'max': 128,
                'description': 'Number of patches to process in parallel'
            },
            'classifier_threshold': {
                'type': 'float',
                'default': 0.8,
                'min': 0.0,
                'max': 1.0,
                'description': 'Confidence threshold for patch classification (0.0-1.0)'
            },
            'generate_gradcam': {
                'type': 'bool',
                'default': False,
                'description': 'Generate GradCAM heatmap visualization (slower but provides attention maps)'
            },
            'device': {
                'type': 'choice',
                'default': 'auto',
                'choices': ['auto', 'cuda', 'cpu'],
                'description': 'Device to run model on (auto detects GPU availability)'
            }
        }
        
        self.classifier = None
        self.model_path = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the PatchClassifier model.
        
        Args:
            config: Configuration dictionary with keys:
                - model_path (str): Path to trained ResNet50 weights (.pth file)
                - patch_size (int, optional): Patch size for sliding window
                - batch_size (int, optional): Batch size for processing
                - classifier_threshold (float, optional): Classification threshold
                - generate_gradcam (bool, optional): Whether to generate GradCAM
                - device (str, optional): Device to use ('cuda', 'cpu', or 'auto')
        
        Raises:
            ValueError: If model_path is not provided or file doesn't exist
            RuntimeError: If model initialization fails
        """
        # Validate required config
        if 'model_path' not in config:
            raise ValueError("model_path is required in config")
        
        self.model_path = Path(config['model_path'])
        if not self.model_path.exists():
            raise ValueError(f"Model weights not found at: {self.model_path}")
        
        # Extract hyperparameters with defaults
        patch_size = config.get('patch_size', 64)
        batch_size = config.get('batch_size', 32)
        classifier_threshold = config.get('classifier_threshold', 0.8)
        generate_gradcam = config.get('generate_gradcam', False)
        device_config = config.get('device', 'auto')
        
        # Handle device configuration
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        print(f"Initializing PatchClassifier Model")
        print(f"  Model path: {self.model_path}")
        print(f"  Device: {device}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Threshold: {classifier_threshold}")
        print(f"  GradCAM: {generate_gradcam}")
        
        try:
            # Initialize the PatchClassifier
            self.classifier = PatchClassifier(
                model_path=str(self.model_path),
                device=device,
                patch_size=patch_size,
                batch_size=batch_size,
                classifier_threshold=classifier_threshold,
                generate_gradcam=generate_gradcam
            )
            
            self._is_initialized = True
            print("✓ PatchClassifier initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PatchClassifier: {str(e)}")
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process tissue region with patch-based classification.
        
        Args:
            image: RGB tissue region image
            mask: Binary mask (required for this model)
            annotation_points: List of polygon points (converted to mask if provided)
            hyperparameters: Runtime hyperparameters to override defaults
        
        Returns:
            Dictionary with:
                - success (bool): Whether processing succeeded
                - processed_image (PIL.Image): Classifier overlay showing positive/negative patches
                - overlay_image (PIL.Image, optional): GradCAM heatmap if enabled
                - scores (dict): Classification statistics
                - error (str, optional): Error message if failed
        """
        try:
            # Validate initialization
            if not self._is_initialized or self.classifier is None:
                return {
                    'success': False,
                    'error': 'Model not initialized. Call initialize() first.'
                }
            
            # Validate inputs
            is_valid, error = self.validate_input(image, mask)
            if not is_valid:
                return {'success': False, 'error': error}
            
            # Handle hyperparameter overrides
            if hyperparameters:
                # For now, hyperparameters require re-initialization
                # This is a limitation of the current PatchClassifier design
                if any(key in hyperparameters for key in ['patch_size', 'batch_size', 
                                                            'classifier_threshold', 'generate_gradcam']):
                    print("Warning: Runtime hyperparameter changes require re-initialization")
                    print("Using model's default hyperparameters")
            
            # Convert mask to area format required by PatchClassifier
            # PatchClassifier expects: (x, y, width, height, path)
            # where path is a matplotlib.path.Path object
            from matplotlib.path import Path as MplPath
            
            if mask is None and annotation_points is None:
                return {
                    'success': False,
                    'error': 'Either mask or annotation_points is required for PatchClassifier'
                }
            
            # Convert mask to polygon if not provided
            if annotation_points is None:
                # Find contours in mask to get polygon points
                mask_array = np.array(mask)
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]
                
                # Simple approach: use mask bounds as rectangle
                # For production, you'd want proper contour detection
                y_indices, x_indices = np.where(mask_array > 0)
                if len(x_indices) == 0:
                    return {
                        'success': False,
                        'error': 'Mask is empty - no tissue region to process'
                    }
                
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                # Create bounding box polygon
                annotation_points = [
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max)
                ]
            
            # Create matplotlib path from annotation points
            vertices = np.array(annotation_points)
            path = MplPath(vertices)
            
            # Get bounding box
            x_coords = [p[0] for p in annotation_points]
            y_coords = [p[1] for p in annotation_points]
            x, y = int(min(x_coords)), int(min(y_coords))
            width = int(max(x_coords) - x)
            height = int(max(y_coords) - y)
            
            # Create area tuple
            area = (x, y, width, height, path)
            
            # Process image
            print(f"Processing region: {width}x{height} at ({x}, {y})")
            gradcam_overlay, classifier_overlay, scores = self.classifier.process_image_with_sliding_window_batch(
                image, area
            )
            
            # Prepare result
            result = {
                'success': True,
                'processed_image': classifier_overlay,  # Main output
                'scores': scores,
                'metadata': {
                    'model_name': self.model_name,
                    'model_version': self.model_version,
                    'patch_size': self.classifier.patch_size,
                    'batch_size': self.classifier.batch_size,
                    'threshold': self.classifier.classifier_threshold,
                    'gradcam_generated': self.classifier.generate_gradcam
                }
            }
            
            # Add GradCAM overlay if generated
            if gradcam_overlay is not None:
                result['overlay_image'] = gradcam_overlay
            
            print(f"✓ Processing complete: {scores['num_total']} patches analyzed")
            print(f"  Positive: {scores['num_pos']} ({scores['percent_pos']}%)")
            print(f"  Negative: {scores['num_neg']}")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Get schema for configurable hyperparameters.
        
        Returns:
            Dictionary defining all available hyperparameters
        """
        return self.hyperparameters
    
    def validate_input(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input image and mask.
        
        Args:
            image: Input image to validate
            mask: Optional mask to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Call parent validation
        is_valid, error = super().validate_input(image, mask)
        if not is_valid:
            return is_valid, error
        
        # PatchClassifier specific validation
        if self.classifier:
            # Check if image is large enough for at least one patch
            width, height = image.size
            patch_size = self.classifier.patch_size
            
            if width < patch_size or height < patch_size:
                return False, (
                    f"Image too small: {width}x{height}. "
                    f"Minimum size: {patch_size}x{patch_size}"
                )
        
        return True, None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        info = super().get_info()
        
        # Add classifier-specific info
        if self.classifier:
            info['classifier_config'] = {
                'patch_size': self.classifier.patch_size,
                'batch_size': self.classifier.batch_size,
                'threshold': self.classifier.classifier_threshold,
                'gradcam_enabled': self.classifier.generate_gradcam,
                'device': str(self.classifier.device)
            }
        
        return info
    
    def cleanup(self) -> None:
        """
        Clean up model resources.
        
        Releases GPU memory and model resources.
        """
        if self.classifier is not None:
            # Clear model
            if hasattr(self.classifier, 'model'):
                del self.classifier.model
            
            # Clear GradCAM if exists
            if hasattr(self.classifier, 'gradient_cam'):
                del self.classifier.gradient_cam
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.classifier = None
        
        self._is_initialized = False
        print(f"✓ {self.model_name} cleaned up")


# For standalone testing
if __name__ == "__main__":
    import sys
    
    # Example usage
    model = PatchClassifierModel()
    
    # Initialize with config
    config = {
        'model_path': '/path/to/resnet50_weights.pth',
        'patch_size': 64,
        'batch_size': 32,
        'classifier_threshold': 0.8,
        'generate_gradcam': True,
        'device': 'auto'
    }
    
    try:
        model.initialize(config)
        
        # Test with dummy image
        test_image = Image.new('RGB', (512, 512), color='white')
        test_mask = Image.new('L', (512, 512), color=255)
        
        result = model.process(test_image, mask=test_mask)
        
        if result['success']:
            print(f"✓ Test successful!")
            print(f"  Scores: {result['scores']}")
        else:
            print(f"✗ Test failed: {result['error']}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
