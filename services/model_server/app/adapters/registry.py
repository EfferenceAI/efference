"""Dynamic model registry for loading external models."""

import logging
from typing import Dict, Type

from .base import BaseAdapter
from .rgbd_adapter import RGBDAdapter

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry of available model adapters."""
    
    # Map model names to their adapter classes
    MODELS: Dict[str, Type[BaseAdapter]] = {
    # Map all RGBD variants to same adapter
    "rgbd": RGBDAdapter,
    "d435": RGBDAdapter,
    "rgbd_d435": RGBDAdapter,
    "d405": RGBDAdapter,
    "rgbd_d405": RGBDAdapter,
}

    @classmethod
    def get_adapter(cls, model_name: str, weights_path: str) -> BaseAdapter:
        """
        Get an adapter instance for a model.
        
        Args:
            model_name: Name of model (e.g., "rgbd")
            weights_path: Path to model weights
            
        Returns:
            Initialized adapter instance
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in cls.MODELS:
            available = ", ".join(cls.MODELS.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        
        logger.info(f"Loading adapter for model: {model_name}")
        AdapterClass = cls.MODELS[model_name]
        
        try:
            adapter = AdapterClass(weights_path)
            logger.info(f"✓ Adapter for '{model_name}' initialized")
            return adapter
        except Exception as e:
            logger.error(f"✗ Failed to initialize adapter: {str(e)}")
            raise
    
    @classmethod
    def list_models(cls) -> list:
        """List all available models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def register_model(cls, model_name: str, adapter_class: Type[BaseAdapter]):
        """
        Register a new model adapter at runtime.
        
        Args:
            model_name: Name of model
            adapter_class: Adapter class
        """
        cls.MODELS[model_name] = adapter_class
        logger.info(f"Registered new model: {model_name}")
