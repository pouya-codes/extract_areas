"""
Base AI Model Interface

This module defines the abstract base class that all AI processing models must implement.
This ensures a consistent interface across different models and enables easy integration
of new AI models into the platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np


class BaseAIModel(ABC):
    """
    Abstract base class for all AI processing models.
    
    All AI models in the platform must inherit from this class and implement
    the required abstract methods. This ensures consistency and allows the
    platform to work with any AI model without modification.
    
    Attributes:
        model_name (str): Unique identifier for the model
        model_version (str): Version of the model
        description (str): Human-readable description of what the model does
        supported_input_formats (List[str]): List of supported image formats (e.g., ['PNG', 'JPEG'])
        requires_mask (bool): Whether the model requires a mask/annotation input
        hyperparameters (Dict[str, Any]): Default hyperparameters for the model
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize the base model.
        
        Args:
            model_name: Unique identifier for the model
            model_version: Version string for the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.description = ""
        self.supported_input_formats = ['PNG', 'JPEG', 'JPG']
        self.requires_mask = False
        self.hyperparameters = {}
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the model with configuration parameters.
        
        This method is called once when the model is first loaded. Use it to:
        - Load model weights
        - Initialize GPU/CPU resources
        - Set up any required preprocessing pipelines
        - Validate configuration
        
        Args:
            config: Dictionary containing model configuration
                   (e.g., model_path, device, batch_size)
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model initialization fails
        """
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
        Process an image with optional mask/annotation and return results.
        
        This is the main processing method that applies the AI model to an input image.
        
        Args:
            image: PIL Image object of the region to process (RGB format)
            mask: Optional binary mask as PIL Image (L mode, 0=background, 255=foreground)
            annotation_points: Optional list of (x, y) polygon points defining the region
            hyperparameters: Optional dict of hyperparameters to override defaults
        
        Returns:
            Dictionary containing:
                - 'processed_image': PIL Image with model output/overlay
                - 'scores': Dict of quantitative metrics (e.g., {'positive_cells': 42})
                - 'str_result': String summary for frontend display (REQUIRED)
                - 'metadata': Optional dict with additional information
                - 'success': Boolean indicating if processing succeeded
                - 'error': Optional error message if processing failed
        
        Example return value:
            {
                'processed_image': <PIL.Image>,
                'scores': {
                    'total_cells': 150,
                    'positive_cells': 42,
                    'negative_cells': 108,
                    'positivity_rate': 0.28
                },
                'str_result': 'Positive: 42/150 cells (28.0%)',
                'metadata': {
                    'processing_time': 2.34,
                    'model_version': '1.0.0'
                },
                'success': True
            }
        """
        pass
    
    @abstractmethod
    def get_hyperparameters_schema(self) -> Dict[str, Any]:
        """
        Get the schema defining available hyperparameters.
        
        Returns a JSON-schema-like dictionary describing all hyperparameters
        that can be configured for this model.
        
        Returns:
            Dictionary with hyperparameter definitions:
            {
                'parameter_name': {
                    'type': 'int' | 'float' | 'bool' | 'string' | 'choice',
                    'default': default_value,
                    'description': 'What this parameter does',
                    'min': minimum_value,  # for numeric types
                    'max': maximum_value,  # for numeric types
                    'choices': [val1, val2],  # for choice type
                    'required': True | False
                }
            }
        
        Example:
            {
                'threshold': {
                    'type': 'float',
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'description': 'Detection threshold for positive cells'
                },
                'color_mode': {
                    'type': 'choice',
                    'default': 'RGB',
                    'choices': ['RGB', 'LAB', 'HSV'],
                    'description': 'Color space for processing'
                }
            }
        """
        pass
    
    def validate_input(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input image and mask before processing.
        
        Override this method to add custom validation logic.
        
        Args:
            image: Input image to validate
            mask: Optional mask to validate
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if inputs are valid, False otherwise
            - error_message: Description of validation error, or None if valid
        """
        if image is None:
            return False, "Image is required"
        
        if self.requires_mask and mask is None:
            return False, f"Model '{self.model_name}' requires a mask"
        
        if mask is not None and image.size != mask.size:
            return False, "Image and mask must have the same dimensions"
        
        return True, None
    
    def preprocess_image(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Preprocess the input image before model inference.
        
        Override this method to implement custom preprocessing.
        Default implementation applies mask if provided.
        
        Args:
            image: Input image
            mask: Optional mask to apply
        
        Returns:
            Preprocessed image
        """
        if mask is not None:
            # Apply mask: white=keep, black=white out
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            return Image.composite(image, white_bg, mask)
        return image
    
    def postprocess_output(
        self,
        output_image: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Image.Image:
        """
        Postprocess the model output before returning.
        
        Override this method to implement custom postprocessing.
        Default implementation applies mask to output if provided.
        
        Args:
            output_image: Model output image
            mask: Optional mask to apply
        
        Returns:
            Postprocessed image
        """
        if mask is not None:
            # White out the masked region
            white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
            return Image.composite(output_image, white_bg, mask)
        return output_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.model_name,
            'version': self.model_version,
            'description': self.description,
            'supported_formats': self.supported_input_formats,
            'requires_mask': self.requires_mask,
            'hyperparameters': self.get_hyperparameters_schema(),
            'initialized': self._is_initialized
        }
    
    def cleanup(self) -> None:
        """
        Clean up resources (GPU memory, file handles, etc.).
        
        Override this method to implement custom cleanup logic.
        Called when the model is being unloaded or the application is shutting down.
        """
        pass
