"""Efference Model Server - Dynamic Model Inference Service."""

import os
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status

from .adapters import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model adapter
model_adapter = None
model_name = None


def get_device_info() -> Dict[str, Any]:
    """Get device information (GPU/CPU)."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        device_info = {"has_gpu": has_gpu}
        
        if has_gpu:
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return device_info
    except Exception:
        return {"has_gpu": False}


def load_model_from_s3() -> None:
    """Download and load model from S3."""
    global model_adapter, model_name
    
    model_name = os.getenv("MODEL_NAME", "d435")
    
    # Determine weight filename based on model name
    if model_name in ["rgbd", "d435", "rgbd_d435"]:
        weight_file = "d435.pth"
    elif model_name in ["d405", "rgbd_d405"]:
        weight_file = "d405.pth"
    else:
        logger.warning(f"Unknown MODEL_NAME '{model_name}', defaulting to d435.pth")
        weight_file = "d435.pth"
    
    local_model_path = f"/app/weights/{weight_file}"
    
    logger.info(f"Requested model: {model_name}")
    logger.info(f"Weight file: {weight_file}")
    logger.info(f"Local path: {local_model_path}")
    
    try:
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(
                f"Model weights not found at {local_model_path}. "
                "The entrypoint.sh script may have failed to download them."
            )
        
        logger.info(f"Loading model from: {local_model_path}")
        
        # Load model using registry (handles adapter instantiation)
        model_adapter = ModelRegistry.get_adapter(model_name, local_model_path)
        logger.info(f"✓ Model '{model_name}' loaded successfully!")
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise


def extract_frames_from_video(
    video_data: bytes,
    max_frames: int = 1,
    target_size: tuple = (518, 518)
) -> tuple:
    """
    Extract frames from video data.
    
    Args:
        video_data: Raw video file bytes
        max_frames: Maximum number of frames to extract
        target_size: Target frame size (height, width)
        
    Returns:
        Tuple of (frames, metadata)
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {frame_count} frames @ {fps}fps, {width}x{height}")
        
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "extracted_frames": len(frames)
        }
        
        return frames, metadata
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("=" * 70)
    logger.info("EFFERENCE MODEL SERVER - STARTUP")
    logger.info("=" * 70)
    
    try:
        load_model_from_s3()
        device_info = get_device_info()
        logger.info(f"Device info: {device_info}")
        logger.info("=" * 70)
        logger.info("Server ready to accept requests!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    
    yield
    
    logger.info("Server shutting down...")


app = FastAPI(
    title="Efference Model Server",
    description="Dynamic model inference service with adapter pattern",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=200)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "model-server",
        "model_loaded": model_adapter is not None,
        "model_name": model_name,
        "available_models": ModelRegistry.list_models(),
        **get_device_info()
    }


@app.post("/infer", status_code=200)
async def run_inference(video: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Run inference on video using the loaded model.
    
    Args:
        video: Video file to process
        
    Returns:
        Inference results
    """
    if model_adapter is None:
        logger.error("Model adapter not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Service is not ready."
        )
    
    try:
        # Validate input
        if not video.filename:
            raise ValueError("Video filename is required")
        
        # Read video data
        video_data = await video.read()
        
        if len(video_data) == 0:
            raise ValueError("Video file is empty")
        
        logger.info(f"Processing: {video.filename} ({len(video_data) / 1e6:.1f}MB)")
        
        # Extract frames
        frames, video_metadata = extract_frames_from_video(video_data, max_frames=1)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Run inference using adapter
        logger.info("Running inference...")
        inference_result = model_adapter.infer(frames[0])
        
        # Return response
        response = {
            "status": "success",
            "filename": video.filename,
            "file_size_bytes": len(video_data),
            "model_name": model_name,
            "video_metadata": video_metadata,
            "inference_result": inference_result
        }
        
        logger.info(f"Successfully processed: {video.filename}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
@app.post("/infer-image", status_code=200)
async def run_image_inference(
    rgb: UploadFile = File(..., description="RGB image"),
    depth: UploadFile = File(None, description="Optional depth image from sensor")
) -> Dict[str, Any]:
    """Run inference on a single RGB image with optional depth."""
    if model_adapter is None:
        logger.error("Model adapter not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    try:
        # Read RGB image
        rgb_data = await rgb.read()
        rgb_array = np.frombuffer(rgb_data, dtype=np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        
        if rgb_image is None:
            raise ValueError("Failed to decode RGB image")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Read optional depth image
        depth_image = None
        if depth is not None:
            depth_data = await depth.read()
            depth_array = np.frombuffer(depth_data, dtype=np.uint8)
            depth_image = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            
            if depth_image is None:
                raise ValueError("Failed to decode depth image")
            depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
        
        logger.info(f"Processing RGB image: {rgb.filename}, shape: {rgb_image.shape}")
        if depth_image is not None:
            logger.info(f"With depth image: {depth.filename}, shape: {depth_image.shape}")
        
        # Run inference using adapter
        logger.info("Running RGBD inference...")
        inference_result = model_adapter.infer_rgbd(rgb_image, depth_image)
        
        # Return response
        response = {
            "status": "success",
            "rgb_filename": rgb.filename,
            "depth_filename": depth.filename if depth else None,
            "model_name": model_name,
            "inference_result": inference_result
        }
        
        logger.info(f"Successfully processed: {rgb.filename}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )