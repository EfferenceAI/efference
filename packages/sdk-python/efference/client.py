"""Efference Python SDK - Official client for the Efference ML API."""

import httpx
import logging
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfferenceClient:
    """
    Official Python client for the Efference ML API.
    
    Example:
        >>> client = EfferenceClient(
        ...     api_key="sk_live_your_key",
        ...     base_url="https://api.efference.ai"
        ... )
        >>> result = client.videos.process("path/to/video.mp4")
        >>> print(result)
    """
    
    DEFAULT_TIMEOUT = 300.0  # 5 minutes
    DEFAULT_BASE_URL = "https://api.efference.ai"
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize the Efference API client.
        
        Args:
            api_key: Your API key for authentication (starts with sk_live_ or sk_test_)
            base_url: The base URL of the API (defaults to official API)
            timeout: Request timeout in seconds (defaults to 300)
            
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.videos = self.Videos(self)
        
        logger.info(f"Efference client initialized with base_url: {self.base_url}")
    
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


__all__ = ["EfferenceClient"]