"""
AI Model Registry

This module provides a registry system for managing multiple AI models.
Models can be registered, loaded, and accessed dynamically.
"""

from typing import Dict, Any, Optional, Type
from models.base_model import BaseAIModel
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing AI processing models.
    
    This singleton class maintains a registry of available models and
    provides methods to register, load, and access them.
    """
    
    _instance = None
    _models: Dict[str, BaseAIModel] = {}
    _model_classes: Dict[str, Type[BaseAIModel]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_model_class(
        self,
        model_name: str,
        model_class: Type[BaseAIModel]
    ) -> None:
        """
        Register a model class for lazy initialization.
        
        Args:
            model_name: Unique identifier for the model
            model_class: The model class (not instance)
        """
        if not issubclass(model_class, BaseAIModel):
            raise ValueError(
                f"Model class must inherit from BaseAIModel"
            )
        
        self._model_classes[model_name] = model_class
        logger.info(f"Registered model class: {model_name}")
    
    def load_model(
        self,
        model_name: str,
        config: Dict[str, Any]
    ) -> BaseAIModel:
        """
        Load and initialize a model.
        
        Args:
            model_name: Name of the model to load
            config: Configuration dict for model initialization
        
        Returns:
            Initialized model instance
        
        Raises:
            ValueError: If model is not registered
            RuntimeError: If model initialization fails
        """
        if model_name in self._models:
            logger.info(f"Model '{model_name}' already loaded")
            return self._models[model_name]
        
        if model_name not in self._model_classes:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available: {list(self._model_classes.keys())}"
            )
        
        try:
            model_class = self._model_classes[model_name]
            model = model_class()
            model.initialize(config)
            model._is_initialized = True
            self._models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise RuntimeError(
                f"Failed to initialize model '{model_name}': {e}"
            )
    
    def get_model(self, model_name: str) -> Optional[BaseAIModel]:
        """
        Get a loaded model instance.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model instance if loaded, None otherwise
        """
        return self._models.get(model_name)
    
    def list_registered_models(self) -> list:
        """
        List all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(self._model_classes.keys())
    
    def list_loaded_models(self) -> list:
        """
        List all loaded model names.
        
        Returns:
            List of loaded model names
        """
        return list(self._models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model info dict if model is loaded, None otherwise
        """
        model = self.get_model(model_name)
        if model:
            return model.get_model_info()
        return None
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model and free its resources.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self._models:
            model = self._models[model_name]
            model.cleanup()
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self) -> None:
        """
        Unload all models and free their resources.
        """
        for model_name in list(self._models.keys()):
            self.unload_model(model_name)
        logger.info("Unloaded all models")


# Global registry instance
model_registry = ModelRegistry()
