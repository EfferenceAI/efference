"""Base adapter for all models."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseAdapter(ABC):
    """Base adapter for external models."""
    
    def __init__(self, weights_path: str):
        """
        Initialize adapter.
        
        Args:
            weights_path: Path to model weights
        """
        self.weights_path = weights_path
        self.model = None
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        """Load the external model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def infer(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference and return structured result.
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            Structured inference result
        """
        pass
