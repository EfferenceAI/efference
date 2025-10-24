"""Efference Python SDK - Official client for the Efference ML API."""

import httpx
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO, Union

# Configure SDK logger only (don't interfere with user's logging)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EfferenceClient:
    """
    Official Python client for the Efference ML API.
    
    Example:
        >>> client = EfferenceClient(api_key="sk_live_your_key")
        >>> result = client.videos.process("path/to/video.mp4")
        >>> print(result)
    """
    
    DEFAULT_TIMEOUT = 300.0  # 5 minutes
    DEFAULT_BASE_URL = os.getenv("EFFERENCE_API_URL", "https://api.efference.ai")

    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,  # Keep for testing, but don't document
        timeout: Optional[float] = None
    ):
        """
        Initialize the Efference API client.
        
        Args:
            api_key: Your API key (starts with sk_live_ or sk_test_)
            timeout: Request timeout in seconds (defaults to 300)
        """
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.videos = self.Videos(self)
        self.images = self.Images(self)
        
        # Only log in debug mode, not base_url
        logger.info("Efference client initialized")
    
    class Videos:
        """Video processing operations."""
        
        def __init__(self, client: "EfferenceClient"):
            """Initialize the Videos namespace."""
            self.client = client
        
        def process(
            self,
            file_path: Union[str, Path, BinaryIO],
            model: str = "rgbd",
            content_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Process a video through the ML model.
            
            Args:
                file_path: Path to the video file or file-like object
                model: Model to use for inference (default: "rgbd")
                content_type: MIME type of the video (e.g., 'video/mp4'). 
                             If not provided, will be inferred from file extension.
            
            Returns:
                Dictionary containing the model's inference results
            
            Raises:
                FileNotFoundError: If the video file does not exist
                ValueError: If the file path is invalid or empty
                httpx.HTTPStatusError: If the API returns an error status
                httpx.TimeoutException: If the request times out
                httpx.RequestError: If there's a connection error
                
            Example:
                >>> result = client.videos.process("video.mp4")
                >>> print(result["status"])
                >>> print(result["inference_result"])
            """
            # Handle file-like objects
            if hasattr(file_path, 'read'):
                return self._process_file_object(file_path, model, content_type)
            
            # Handle file paths
            if not file_path or not str(file_path).strip():
                raise ValueError("file_path cannot be empty")
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"Processing video: {file_path.name} ({file_size_mb:.2f} MB)")
            
            # Determine content type
            if content_type is None:
                content_type = self._get_content_type(file_path)
                logger.debug(f"Inferred content type: {content_type}")
            
            # Prepare request
            url = f"{self.client.base_url}/v1/videos/process"
            headers = {"Authorization": f"Bearer {self.client.api_key}"}
            
            try:
                with open(file_path, "rb") as f:
                    files = {"video": (file_path.name, f, content_type)}
                    
                    logger.debug(f"Sending request to {url}")
                    
                    with httpx.Client(timeout=self.client.timeout) as client:
                        response = client.post(url, headers=headers, files=files)
                        response.raise_for_status()
                
                result = response.json()
                logger.info("Video processed successfully")
                return result
                
            except httpx.TimeoutException:
                logger.error(f"Request timed out after {self.client.timeout}s")
                raise
            except httpx.HTTPStatusError as e:
                self._handle_http_error(e)
                raise
            except httpx.RequestError as e:
                logger.error(f"Request failed: {str(e)}")
                raise
        
        def _process_file_object(
            self,
            file_obj: BinaryIO,
            model: str,
            content_type: Optional[str]
        ) -> Dict[str, Any]:
            """Process a file-like object."""
            if content_type is None:
                content_type = "video/mp4"  # Default
            
            url = f"{self.client.base_url}/v1/videos/process"
            headers = {"Authorization": f"Bearer {self.client.api_key}"}
            
            files = {"video": ("video", file_obj, content_type)}
            
            try:
                with httpx.Client(timeout=self.client.timeout) as client:
                    response = client.post(url, headers=headers, files=files)
                    response.raise_for_status()
                
                return response.json()
            
            except httpx.TimeoutException:
                logger.error(f"Request timed out after {self.client.timeout}s")
                raise
            except httpx.HTTPStatusError as e:
                self._handle_http_error(e)
                raise
        
        def _get_content_type(self, file_path: Path) -> str:
            """Get MIME type from file extension."""
            extension = file_path.suffix.lower()
            content_type_map = {
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
                ".flv": "video/x-flv",
                ".wmv": "video/x-ms-wmv",
                ".m4v": "video/x-m4v",
            }
            return content_type_map.get(extension, "video/mp4")
        
        def _handle_http_error(self, error: httpx.HTTPStatusError):
            """Handle HTTP errors with descriptive messages."""
            status_code = error.response.status_code
            
            if status_code == 401:
                logger.error("Authentication failed. Check your API key.")
            elif status_code == 402:
                logger.error("Insufficient credits.")
            elif status_code == 413:
                logger.error("Video file too large.")
            elif status_code == 429:
                logger.error("Rate limit exceeded. Wait before retrying.")
            elif status_code >= 500:
                logger.error(f"Server error: {status_code}")
            else:
                logger.error(f"HTTP error: {status_code}")
    
    class Images:
        """Image processing operations."""
        
        def __init__(self, client: "EfferenceClient"):
            """Initialize the Images namespace."""
            self.client = client
        
        def process_rgbd(
            self,
            rgb_path: Union[str, Path, BinaryIO],
            depth_path: Optional[Union[str, Path, BinaryIO]] = None,
            depth_scale: float = 1000.0,
            input_size: int = 518,
            max_depth: float = 25.0,
            save_visualization: Optional[Union[str, Path]] = None,
            save_3panel: Optional[Union[str, Path]] = None
        ) -> Dict[str, Any]:
            """
            Process RGB image with optional depth for depth estimation.
            
            Args:
                rgb_path: Path to RGB image or file-like object
                depth_path: Optional path to depth image from sensor
                depth_scale: Depth scale for sensor (default: 1000 for RealSense)
                input_size: Model input size (default: 518)
                max_depth: Maximum depth in meters (default: 25.0)
                save_visualization: Optional path to save single colorized depth PNG
                save_3panel: Optional path to save 3-panel comparison (RGB|Original|Corrected)
                
            Returns:
                Dictionary containing inference results and depth visualizations
                
            Example:
                >>> result = client.images.process_rgbd(
                ...     "color.png",
                ...     save_visualization="depth_colored.png",
                ...     save_3panel="comparison.png"
                ... )
                >>> print(f"Depth range: {result['inference_result']['output']['min']:.2f}-{result['inference_result']['output']['max']:.2f}m")
            """
            url = f"{self.client.base_url}/v1/images/rgbd"
            headers = {"Authorization": f"Bearer {self.client.api_key}"}
            
            # Prepare RGB file
            rgb_file_obj = None
            if hasattr(rgb_path, 'read'):
                rgb_file = ("rgb.png", rgb_path, "image/png")
            else:
                rgb_path = Path(rgb_path)
                if not rgb_path.exists():
                    raise FileNotFoundError(f"RGB image not found: {rgb_path}")
                rgb_file_obj = open(rgb_path, "rb")
                rgb_file = ("rgb.png", rgb_file_obj, "image/png")
            
            files = {"rgb": rgb_file}
            
            # Prepare optional depth file
            depth_file_obj = None
            if depth_path is not None:
                if hasattr(depth_path, 'read'):
                    files["depth"] = ("depth.png", depth_path, "image/png")
                else:
                    depth_path = Path(depth_path)
                    if not depth_path.exists():
                        raise FileNotFoundError(f"Depth image not found: {depth_path}")
                    depth_file_obj = open(depth_path, "rb")
                    files["depth"] = ("depth.png", depth_file_obj, "image/png")
            
            # Prepare form data for configurable parameters
            data = {
                "depth_scale": str(depth_scale),
                "input_size": str(input_size),
                "max_depth": str(max_depth)
            }
            
            try:
                logger.info(f"Processing RGBD image...")
                
                with httpx.Client(timeout=self.client.timeout) as client:
                    response = client.post(url, headers=headers, files=files, data=data)
                    response.raise_for_status()
                
                result = response.json()
                logger.info("RGBD processing completed successfully")
                
                # Save single visualization if requested
                if save_visualization and "inference_result" in result:
                    viz_data = result["inference_result"].get("depth_visualization")
                    if viz_data:
                        self._save_visualization(viz_data, save_visualization)
                        logger.info(f"Saved depth visualization to: {save_visualization}")
                
                # Save 3-panel visualization if requested
                if save_3panel and "inference_result" in result:
                    viz_3panel = result["inference_result"].get("depth_visualization_3panel")
                    if viz_3panel:
                        self._save_visualization(viz_3panel, save_3panel)
                        logger.info(f"Saved 3-panel visualization to: {save_3panel}")
                
                return result
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code}")
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    pass
                raise
            except httpx.RequestError as e:
                logger.error(f"Request failed: {str(e)}")
                raise
            finally:
                # Close files if we opened them
                if rgb_file_obj:
                    rgb_file_obj.close()
                if depth_file_obj:
                    depth_file_obj.close()
        
        def _save_visualization(self, base64_data: str, output_path: Union[str, Path]):
            """Decode base64 PNG and save to file."""
            import base64
            
            output_path = Path(output_path)
            img_data = base64.b64decode(base64_data)
            
            with open(output_path, "wb") as f:
                f.write(img_data)
        
        def visualize_depth(self, result: Dict[str, Any], mode: str = "single", show: bool = True):
            """
            Display depth visualization using matplotlib.
            
            Args:
                result: API response containing depth_visualization
                mode: "single" for colorized depth only, "3panel" for RGB|Original|Corrected comparison
                show: Whether to display the plot immediately
                
            Returns:
                Matplotlib figure object
                
            Example:
                >>> result = client.images.process_rgbd("color.png")
                >>> client.images.visualize_depth(result, mode="3panel")
            """
            try:
                import matplotlib.pyplot as plt
                import base64
                from PIL import Image
                from io import BytesIO
            except ImportError:
                raise ImportError(
                    "matplotlib and Pillow required for visualization. "
                    "Install with: pip install efference[visualization]"
                )
            
            # Extract base64 visualization based on mode
            inference_result = result.get("inference_result", {})
            
            if mode == "3panel":
                depth_viz_b64 = inference_result.get("depth_visualization_3panel")
                if not depth_viz_b64:
                    raise ValueError("No 3-panel visualization found. Falling back to single mode.")
                title = "RGBD Comparison (RGB | Original Depth | Corrected Depth)"
            else:
                depth_viz_b64 = inference_result.get("depth_visualization")
                if not depth_viz_b64:
                    raise ValueError("No depth_visualization found in result.")
                title = "Depth Estimation (Blue=Near, Red=Far)"
            
            # Decode and display
            img_data = base64.b64decode(depth_viz_b64)
            img = Image.open(BytesIO(img_data))
            
            fig = plt.figure(figsize=(15, 8) if mode == "3panel" else (10, 8))
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            
            # Add depth stats for single mode
            if mode != "3panel":
                stats = inference_result.get("output", {})
                stats_text = f"Min: {stats.get('min', 0):.2f}m | Max: {stats.get('max', 0):.2f}m | Mean: {stats.get('mean', 0):.2f}m"
                plt.text(10, 30, stats_text,
                        color='white', fontsize=12, 
                        bbox=dict(facecolor='black', alpha=0.7))
            
            if show:
                plt.show()
            
            return fig


__all__ = ["EfferenceClient"]