"""Adapter for efference-rgbd model """

import logging
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class RGBDAdapter(BaseAdapter):
    """Wraps efference-rgbd model with minimal duplication."""
    
    def load_model(self):
        """Load efference-rgbd model with weights."""
        try:
            from efference_rgbd.utils import load_model
            self.model = load_model(self.weights_path)
            logger.info("✓ RGBD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RGBD model: {str(e)}")
            raise
    
    def infer(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run RGBD inference on video frame."""
        try:
            depth_placeholder = np.zeros_like(frame[:, :, 0], dtype=np.float32)
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
    
    def infer_rgbd(
        self, 
        rgb: np.ndarray, 
        depth: Optional[np.ndarray] = None,
        return_visualization: bool = True,
        image_min: float = 0.1,
        image_max: float = 5.0
    ) -> Dict[str, Any]:
        """
        Run RGBD inference with RGB image and optional depth.
        
        REUSES: efference-rgbd's colorize() and create_visualization()
        
        Args:
            rgb: RGB image (H, W, 3), uint8
            depth: Optional depth image from sensor (H, W), float32
            return_visualization: Whether to return base64-encoded visualization
            image_min: Min depth for colorization (meters)
            image_max: Max depth for colorization (meters)
            
        Returns:
            Structured inference result with optional visualizations
        """
        try:
            from efference_rgbd.utils import colorize, create_visualization
            
            depth_was_provided = depth is not None and bool(np.any(depth > 0))
            depth_orig = depth if depth is not None else np.zeros_like(rgb[:, :, 0], dtype=np.float32)
            
            if depth_orig.dtype != np.float32:
                depth_orig = depth_orig.astype(np.float32)
            
            logger.info(f"RGBD inference - RGB: {rgb.shape}, Depth: {depth_orig.shape}")
            
            # Run inference
            depth_corrected = self.model.infer_image(
                rgb=rgb,
                depth=depth_orig,
                input_size=518,
                max_depth=25.0
            )
            
            result = {
                "model_type": "rgbd",
                "output": {
                    "shape": list(depth_corrected.shape),
                    "dtype": str(depth_corrected.dtype),
                    "min": float(np.min(depth_corrected)),
                    "max": float(np.max(depth_corrected)),
                    "mean": float(np.mean(depth_corrected)),
                    "has_valid_depth": bool(np.any(depth_corrected > 0))
                },
                "input_depth_provided": depth_was_provided
            }
            
            # Generate visualization using efference-rgbd's built-in functions
            if return_visualization:
                # Use ACTUAL depth range for better visualization
                actual_min = float(np.min(depth_corrected))
                actual_max = float(np.max(depth_corrected))
                
                # 3-panel visualization (RGB + Original + Corrected)
                vis_3panel = create_visualization(
                    rgb, depth_orig, depth_corrected, 
                    actual_min, actual_max  # Use actual range!
                )
                result["depth_visualization_3panel"] = self._to_base64_png(vis_3panel)
                
                # Single colorized corrected depth
                depth_colored_single = colorize(
                    depth_corrected,
                    min_depth=actual_min,  # Use actual range!
                    max_depth=actual_max,  # Use actual range!
                    cmap=cv2.COLORMAP_TURBO
                )
                result["depth_visualization"] = self._to_base64_png(depth_colored_single)
            
            return result
            
        except Exception as e:
            logger.error(f"RGBD inference failed: {str(e)}", exc_info=True)
            raise
    
    def _to_base64_png(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 PNG string."""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')