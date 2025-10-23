"""Adapter for efference-rgbd model."""

import logging
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class RGBDAdapter(BaseAdapter):
    """Wraps efference-rgbd model."""
    
    def load_model(self):
        """Load efference-rgbd model with weights."""
        try:
            # NEW API: Import load_model from utils
            from efference_rgbd.utils import load_model
            
            # Load model (returns RGBDDepth instance)
            self.model = load_model(self.weights_path)
            
            logger.info("✓ RGBD model loaded successfully")
        except ImportError as e:
            logger.error(f"efference_rgbd package not installed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load RGBD model: {str(e)}")
            raise
    
    def infer(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run RGBD inference."""
        try:
            # NEW API: Call infer_image as a method on the model
            # Note: Expects actual depth (not inverse depth)
            depth_placeholder = np.zeros_like(frame[:, :, 0], dtype=np.float32)
            
            # Call model's infer_image method
            # Returns ONLY depth_corrected (not a tuple)
            depth_corrected = self.model.infer_image(
                rgb=frame,
                depth=depth_placeholder,
                input_size=518,
                max_depth=25.0
            )
            
            return {
                "model_type": "rgbd",
                "output": {
                    "shape": list(depth_corrected.shape),
                    "dtype": str(depth_corrected.dtype),
                    "min": float(np.min(depth_corrected)),
                    "max": float(np.max(depth_corrected)),
                    "mean": float(np.mean(depth_corrected)),
                    "has_valid_depth": bool(np.any(depth_corrected > 0))
                }
            }
        except Exception as e:
            logger.error(f"RGBD inference failed: {str(e)}", exc_info=True)
            raise
    def infer_rgbd(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run RGBD inference with RGB image and optional depth.
        
        Args:
            rgb: RGB image (H, W, 3), uint8
            depth: Optional depth image from sensor (H, W), float32
            
        Returns:
            Structured inference result with corrected depth
        """
        try:
            # Check if depth was provided BEFORE we modify it
            depth_was_provided = depth is not None and bool(np.any(depth > 0))
            
            # If no depth provided, use zeros (pure depth estimation mode)
            if depth is None:
                depth = np.zeros_like(rgb[:, :, 0], dtype=np.float32)
            
            # Ensure depth is float32
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            
            logger.info(f"RGBD inference - RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
            
            # Call model's infer_image method
            depth_corrected = self.model.infer_image(
                rgb=rgb,
                depth=depth,
                input_size=518,
                max_depth=25.0
            )
            
            return {
                "model_type": "rgbd",
                "output": {
                    "shape": list(depth_corrected.shape),
                    "dtype": str(depth_corrected.dtype),
                    "min": float(np.min(depth_corrected)),
                    "max": float(np.max(depth_corrected)),
                    "mean": float(np.mean(depth_corrected)),
                    "has_valid_depth": bool(np.any(depth_corrected > 0))
                },
                "input_depth_provided": depth_was_provided  # Already converted to Python bool
            }
        except Exception as e:
            logger.error(f"RGBD inference failed: {str(e)}", exc_info=True)
            raise