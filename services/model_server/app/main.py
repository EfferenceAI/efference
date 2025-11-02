"""Efference Model Server - Dynamic Model Inference Service."""

import os
import logging
import tempfile
import io
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form

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


def stream_video_frames(
    video_data: bytes,
    max_frames: int = 10,
    frame_skip: int = 1,
    target_size: tuple = (256, 256)  # Reduced from 518 to 256 for GPU memory
):
    """
    Generator that yields video frames one at a time to reduce memory usage.
    
    Args:
        video_data: Raw video file bytes
        max_frames: Maximum number of frames to process
        frame_skip: Process every Nth frame
        target_size: Target frame size (height, width)
        
    Yields:
        Tuple of (frame_index, frame_array, metadata)
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
        
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
        
        logger.info(f"Video info: {frame_count} frames @ {fps}fps, {width}x{height}")
        logger.info(f"Processing max {max_frames} frames with skip={frame_skip}")
        
        frame_index = 0
        processed_count = 0
        
        while processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame skipping
            if frame_index % frame_skip == 0:
                frame = cv2.resize(frame, target_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_index, frame_rgb, metadata
                processed_count += 1
            
            frame_index += 1
        
        cap.release()
        
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
        
        # Use streaming to extract just the first frame (memory safe)
        video_metadata = None
        inference_result = None
        
        for frame_index, frame, metadata in stream_video_frames(video_data, max_frames=1):
            if video_metadata is None:
                video_metadata = metadata
            
            logger.info("Running inference on first frame...")
            inference_result = model_adapter.infer(frame)
            break  # Only process the first frame
        
        if inference_result is None:
            raise ValueError("No frames could be extracted from video")
        
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
@app.post("/infer-batch", status_code=200)
async def run_batch_video_inference(
    video: UploadFile = File(..., description="Video file to process"),
    max_frames: int = Form(None, description="Maximum frames to process (None = all frames)"),
    frame_skip: int = Form(1, description="Process every Nth frame (1 = all frames)"),
    target_size: tuple = Form((518, 518), description="Target frame size")
) -> Dict[str, Any]:
    """
    Process ALL frames from a video file (batch processing).
    Unlike /infer which processes only 1 frame, this processes all frames.
    """
    if model_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    try:
        # Read video data
        video_data = await video.read()
        if len(video_data) == 0:
            raise ValueError("Video file is empty")
        
        logger.info(f"Batch processing: {video.filename} ({len(video_data) / 1e6:.1f}MB)")
        
        # Stream process frames to reduce memory usage
        max_frames = max_frames or 10  # Reduced from 9999 to 10 for memory safety
        frame_results = []
        video_metadata = None
        frames_processed = 0
        
        logger.info(f"Starting streaming processing of max {max_frames} frames...")
        
        # Process frames one at a time using generator
        for frame_index, frame, metadata in stream_video_frames(
            video_data, 
            max_frames=max_frames,
            frame_skip=frame_skip,
            target_size=target_size
        ):
            if video_metadata is None:
                video_metadata = metadata
            
            logger.info(f"Processing frame {frames_processed+1}/{max_frames} (original index: {frame_index})")
            
            # Run inference on single frame
            result = model_adapter.infer(frame)
            frame_results.append({
                "frame_index": frame_index,
                "inference_result": result
            })
            
            frames_processed += 1
            
            # Force garbage collection every 5 frames to free memory
            if frames_processed % 5 == 0:
                import gc
                gc.collect()
        
        if not frame_results:
            raise ValueError("No frames could be processed from video")
        
        # Return batch results
        response = {
            "status": "success",
            "filename": video.filename,
            "file_size_bytes": len(video_data),
            "model_name": model_name,
            "video_metadata": video_metadata,
            "frames_processed": frames_processed,
            "frame_skip": frame_skip,
            "batch_results": frame_results,
            "processing_summary": {
                "total_frames_in_video": video_metadata["frame_count"],
                "frames_processed": frames_processed,
                "max_frames_limit": max_frames
            }
        }
        
        logger.info(f"Batch processing complete: {len(frame_results)} frames")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@app.post("/infer/models/switch", status_code=200)
async def switch_model(
    new_model_name: str = Form(..., description="Model to switch to (d435, d405, rgbd)")
) -> Dict[str, Any]:
    """
    Hot-swap the currently loaded model.
    Supports: d435, d405, rgbd variants.
    """
    global model_adapter, model_name
    
    try:
        logger.info(f"Switching model from {model_name} to {new_model_name}")
        
        # Validate model name
        available_models = ModelRegistry.list_models()
        if new_model_name not in available_models:
            raise ValueError(f"Model '{new_model_name}' not available. Options: {available_models}")
        
        # Determine weight file path
        if new_model_name in ["rgbd", "d435", "rgbd_d435"]:
            weight_file = "d435.pth"
        elif new_model_name in ["d405", "rgbd_d405"]:
            weight_file = "d405.pth"
        else:
            raise ValueError(f"Unknown model variant: {new_model_name}")
        
        new_model_path = f"/app/weights/{weight_file}"
        
        # Check if weights exist
        if not os.path.exists(new_model_path):
            raise FileNotFoundError(f"Model weights not found: {new_model_path}")
        
        # Load new model
        old_model_name = model_name
        model_adapter = ModelRegistry.get_adapter(new_model_name, new_model_path)
        model_name = new_model_name
        
        logger.info(f"✓ Successfully switched from {old_model_name} to {new_model_name}")
        
        return {
            "status": "success",
            "message": f"Model switched from {old_model_name} to {new_model_name}",
            "previous_model": old_model_name,
            "current_model": model_name,
            "model_path": new_model_path,
            "available_models": available_models
        }
        
    except Exception as e:
        logger.error(f"Model switch failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to switch model: {str(e)}"
        )


@app.get("/infer/models/list", status_code=200)
async def list_available_models() -> Dict[str, Any]:
    """List all available models and their status."""
    return {
        "available_models": ModelRegistry.list_models(),
        "current_model": model_name,
        "model_loaded": model_adapter is not None,
        "weights_available": {
            "d435": os.path.exists("/app/weights/d435.pth"),
            "d405": os.path.exists("/app/weights/d405.pth")
        }
    }


@app.post("/infer-image", status_code=200)
async def run_image_inference(
    rgb: UploadFile = File(..., description="RGB image"),
    depth: UploadFile = File(None, description="Optional depth image from sensor"),
    depth_scale: float = Form(1000.0, description="Depth scale for sensor (default: 1000 for RealSense)"),
    input_size: int = Form(518, description="Model input size"),
    max_depth: float = Form(25.0, description="Maximum depth in meters"),
    include_pointcloud: bool = Form(False, description="Generate 3D point cloud")
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


@app.post("/infer-advanced", status_code=200)
async def run_advanced_inference(
    # Format 1: RGB video + numpy depth
    video: UploadFile = File(None, description="RGB video file"),
    depth_numpy: UploadFile = File(None, description="Depth data as numpy array"),
    
    # Format 2: RGB video + OpenEXR depth
    depth_exr: UploadFile = File(None, description="Depth data as OpenEXR file"),
    
    # Format 3: RGBD numpy array
    rgbd_numpy: UploadFile = File(None, description="Combined RGBD data as numpy array"),
    
    # Format 4: RGB numpy + Depth numpy arrays
    rgb_numpy: UploadFile = File(None, description="RGB data as numpy array"),
    depth_numpy_separate: UploadFile = File(None, description="Separate depth numpy array"),
    
    # Processing parameters
    max_frames: int = Form(None, description="Maximum frames to process for video inputs"),
    frame_skip: int = Form(1, description="Process every Nth frame"),
    depth_format: str = Form("auto", description="Depth data format (numpy, exr, auto)"),
    data_format: str = Form("auto", description="Overall data format"),
    
    # Array metadata (passed as strings from control plane)
    depth_shape: str = Form(None, description="Depth array shape"),
    depth_dtype: str = Form(None, description="Depth array data type"),
    rgbd_shape: str = Form(None, description="RGBD array shape"),
    rgbd_dtype: str = Form(None, description="RGBD array data type"),
    rgb_shape: str = Form(None, description="RGB array shape"),
    rgb_dtype: str = Form(None, description="RGB array data type")
) -> Dict[str, Any]:
    """
    Advanced inference supporting multiple input formats:
    1. RGB video + numpy depth
    2. RGB video + OpenEXR depth  
    3. RGBD numpy array
    4. RGB numpy array + Depth numpy array
    """
    if model_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    
    try:
        # Determine processing format
        if video and depth_numpy:
            return await _process_video_numpy_depth(video, depth_numpy, max_frames, frame_skip)
        elif video and depth_exr:
            return await _process_video_exr_depth(video, depth_exr, max_frames, frame_skip)
        elif rgbd_numpy:
            return await _process_rgbd_numpy_array(rgbd_numpy, rgbd_shape, rgbd_dtype)
        elif rgb_numpy and depth_numpy_separate:
            return await _process_rgb_depth_numpy_arrays(rgb_numpy, depth_numpy_separate, rgb_shape, rgb_dtype, depth_shape, depth_dtype)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input combination for advanced processing"
            )
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Advanced inference failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def _process_video_numpy_depth(video, depth_numpy, max_frames, frame_skip):
    """Process RGB video + numpy depth format."""
    
    # Read video and extract frames
    video_data = await video.read()
    frames, video_metadata = extract_frames_from_video(
        video_data, 
        max_frames=max_frames or 10,  # Default limit
        target_size=(518, 518)
    )
    
    # Read and load depth numpy data
    depth_data = await depth_numpy.read()
    depth_array = np.load(io.BytesIO(depth_data))
    
    logger.info(f"Processing {len(frames)} video frames with depth array shape: {depth_array.shape}")
    
    # Process frames with corresponding depth data
    frame_results = []
    for i, frame in enumerate(frames[::frame_skip]):
        frame_idx = i * frame_skip
        
        # Get corresponding depth frame (if depth has temporal dimension)
        if len(depth_array.shape) == 3:  # (frames, height, width)
            if frame_idx < depth_array.shape[0]:
                depth_frame = depth_array[frame_idx]
            else:
                depth_frame = depth_array[-1]  # Use last depth frame if video is longer
        else:  # Single depth image for all frames
            depth_frame = depth_array
        
        # Run RGBD inference
        result = model_adapter.infer_rgbd(frame, depth_frame.astype(np.float32))
        
        frame_results.append({
            "frame_index": frame_idx,
            "inference_result": result
        })
    
    return {
        "status": "success",
        "filename": video.filename,
        "model_name": model_name,
        "video_metadata": video_metadata,
        "frames_processed": len(frame_results),
        "frame_skip": frame_skip,
        "batch_results": frame_results,
        "depth_array_shape": list(depth_array.shape),
        "depth_array_dtype": str(depth_array.dtype)
    }


async def _process_video_exr_depth(video, depth_exr, max_frames, frame_skip):
    """Process RGB video + OpenEXR depth format."""
    
    # Read video and extract frames
    video_data = await video.read()
    frames, video_metadata = extract_frames_from_video(
        video_data, 
        max_frames=max_frames or 10,
        target_size=(518, 518)
    )
    
    # Read EXR depth data
    exr_data = await depth_exr.read()
    
    # Use imageio to read EXR (requires imageio-plugin)
    try:
        import imageio
        depth_array = imageio.imread(io.BytesIO(exr_data))
    except Exception as e:
        raise ValueError(f"Failed to read EXR depth data: {e}")
    
    logger.info(f"Processing {len(frames)} video frames with EXR depth shape: {depth_array.shape}")
    
    # Process frames
    frame_results = []
    for i, frame in enumerate(frames[::frame_skip]):
        frame_idx = i * frame_skip
        
        # Use single depth for all frames (EXR typically single image)
        result = model_adapter.infer_rgbd(frame, depth_array.astype(np.float32))
        
        frame_results.append({
            "frame_index": frame_idx,
            "inference_result": result
        })
    
    return {
        "status": "success",
        "filename": video.filename,
        "model_name": model_name,
        "video_metadata": video_metadata,
        "frames_processed": len(frame_results),
        "frame_skip": frame_skip,
        "batch_results": frame_results,
        "depth_format": "exr",
        "depth_array_shape": list(depth_array.shape)
    }


async def _process_rgbd_numpy_array(rgbd_numpy, rgbd_shape, rgbd_dtype):
    """Process RGBD numpy array format."""
    
    # Read and load RGBD numpy data
    rgbd_data = await rgbd_numpy.read()
    rgbd_array = np.load(io.BytesIO(rgbd_data))
    
    logger.info(f"Processing RGBD numpy array with shape: {rgbd_array.shape}")
    
    # Split RGBD array into RGB and Depth
    if len(rgbd_array.shape) == 3 and rgbd_array.shape[2] == 4:
        # Format: (height, width, 4) - RGB + D
        rgb = (rgbd_array[:, :, :3] * 255).astype(np.uint8)  # Denormalize RGB
        depth = rgbd_array[:, :, 3].astype(np.float32)
    else:
        raise ValueError(f"Invalid RGBD array shape: {rgbd_array.shape}. Expected (H, W, 4)")
    
    # Run RGBD inference
    result = model_adapter.infer_rgbd(rgb, depth)
    
    return {
        "status": "success",
        "filename": rgbd_numpy.filename,
        "model_name": model_name,
        "input_format": "rgbd_numpy",
        "rgbd_array_shape": list(rgbd_array.shape),
        "rgbd_array_dtype": str(rgbd_array.dtype),
        "batch_results": [{
            "frame_index": 0,
            "inference_result": result
        }],
        "frames_processed": 1
    }


async def _process_rgb_depth_numpy_arrays(rgb_numpy, depth_numpy_separate, rgb_shape, rgb_dtype, depth_shape, depth_dtype):
    """Process RGB numpy array + Depth numpy array format."""
    
    # Read and load both arrays
    rgb_data = await rgb_numpy.read()
    depth_data = await depth_numpy_separate.read()
    
    rgb_array = np.load(io.BytesIO(rgb_data))
    depth_array = np.load(io.BytesIO(depth_data))
    
    logger.info(f"Processing RGB array: {rgb_array.shape}, Depth array: {depth_array.shape}")
    
    # Validate and prepare arrays
    if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Invalid RGB array shape: {rgb_array.shape}. Expected (H, W, 3)")
    
    if len(depth_array.shape) != 2:
        raise ValueError(f"Invalid depth array shape: {depth_array.shape}. Expected (H, W)")
    
    # Ensure correct data types
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).astype(np.uint8) if rgb_array.max() <= 1.0 else rgb_array.astype(np.uint8)
    
    depth_array = depth_array.astype(np.float32)
    
    # Run RGBD inference
    result = model_adapter.infer_rgbd(rgb_array, depth_array)
    
    return {
        "status": "success",
        "rgb_filename": rgb_numpy.filename,
        "depth_filename": depth_numpy_separate.filename,
        "model_name": model_name,
        "input_format": "rgb_depth_numpy",
        "rgb_array_shape": list(rgb_array.shape),
        "depth_array_shape": list(depth_array.shape),
        "batch_results": [{
            "frame_index": 0,
            "inference_result": result
        }],
        "frames_processed": 1
    }


# ============================================================================
# Live Camera Streaming Endpoints
# ============================================================================

# Global streaming state
streaming_state = {
    "active": False,
    "camera": None,
    "latest_frame": None,
    "latest_result": None,
    "frame_count": 0
}


@app.post("/infer/stream/start", status_code=200)
async def start_camera_stream(
    camera_type: str = Form("realsense", description="Camera type (realsense, webcam)"),
    config_path: str = Form(None, description="Optional camera config file"),
    live_inference: bool = Form(True, description="Run live inference on frames")
) -> Dict[str, Any]:
    """
    Start live camera streaming with optional real-time inference.
    Based on efference-rgbd's RealSenseCamera live mode.
    """
    global streaming_state
    
    try:
        if streaming_state["active"]:
            return {
                "status": "already_active",
                "message": "Stream is already running",
                "frame_count": streaming_state["frame_count"]
            }
        
        logger.info(f"Starting {camera_type} live stream...")
        
        # Import camera class from efference-rgbd
        try:
            if camera_type == "realsense":
                # Check if pyrealsense2 is available first
                try:
                    import pyrealsense2 as rs
                    if rs is None:
                        raise ImportError("pyrealsense2 module is None")
                except ImportError:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail="pyrealsense2 library is not installed. Live RealSense camera streaming is not available. "
                               "To enable it, install: pip install pyrealsense2"
                    )
                
                from efference_rgbd.camera import RealSenseCamera
                # RealSenseCamera expects config_paths as a list, not a single string
                config_paths = [config_path] if config_path else None
                camera = RealSenseCamera(config_paths=config_paths, live=True)
            else:
                raise ValueError(f"Unsupported camera type: {camera_type}")
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Camera support not available: {str(e)}"
            )
        
        # Start camera
        camera.start()
        
        # Update streaming state
        streaming_state.update({
            "active": True,
            "camera": camera,
            "latest_frame": None,
            "latest_result": None,
            "frame_count": 0
        })
        
        logger.info("✓ Live camera stream started")
        
        return {
            "status": "success",
            "message": f"{camera_type} stream started",
            "camera_type": camera_type,
            "live_inference": live_inference,
            "endpoints": {
                "get_frame": "/stream/frame",
                "stop_stream": "/stream/stop",
                "stream_status": "/stream/status"
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (already have proper status codes)
        raise
    except RuntimeError as e:
        # Handle camera-specific errors (e.g., "No RealSense device detected")
        error_msg = str(e)
        logger.error(f"Camera runtime error: {error_msg}")
        
        if "No RealSense device" in error_msg or "device detected" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No RealSense camera detected. Please ensure a RealSense camera is connected via USB. Error: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Camera error: {error_msg}"
            )
    except Exception as e:
        logger.error(f"Failed to start stream: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream start failed: {str(e)}"
        )


@app.get("/infer/stream/frame", status_code=200)
async def get_latest_frame(
    run_inference: bool = False,
    format: str = "json"
) -> Dict[str, Any]:
    """
    Get the latest frame from the live camera stream.
    Optionally run inference on the frame.
    """
    global streaming_state
    
    if not streaming_state["active"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active stream. Start stream first with /stream/start"
        )
    
    try:
        camera = streaming_state["camera"]
        
        # Get next frame from camera
        for rgb_frames, depth_frames in camera.frames():
            # Take first frame (single camera setup)
            rgb_frame = rgb_frames[0] if isinstance(rgb_frames, list) else rgb_frames
            depth_frame = depth_frames[0] if isinstance(depth_frames, list) else depth_frames
            
            streaming_state["frame_count"] += 1
            
            frame_data = {
                "frame_count": streaming_state["frame_count"],
                "rgb_shape": list(rgb_frame.shape) if rgb_frame is not None else None,
                "depth_shape": list(depth_frame.shape) if depth_frame is not None else None,
                "timestamp": "2025-01-26T12:00:00"  # Placeholder timestamp
            }
            
            # Run inference if requested
            if run_inference and model_adapter is not None and rgb_frame is not None:
                logger.info(f"Running inference on frame {streaming_state['frame_count']}")
                
                # Convert depth to placeholder if needed
                if depth_frame is None:
                    depth_frame = np.zeros_like(rgb_frame[:, :, 0], dtype=np.float32)
                
                # Run RGBD inference
                inference_result = model_adapter.infer_rgbd(rgb_frame, depth_frame)
                frame_data["inference_result"] = inference_result
                
                # Store latest result
                streaming_state["latest_result"] = inference_result
            
            # Store latest frame
            streaming_state["latest_frame"] = frame_data
            
            return {
                "status": "success",
                "frame_data": frame_data,
                "stream_info": {
                    "total_frames": streaming_state["frame_count"],
                    "camera_active": True,
                    "model_loaded": model_adapter is not None
                }
            }
            
        # If we get here, stream ended
        return await stop_camera_stream()
        
    except Exception as e:
        logger.error(f"Frame capture failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame capture failed: {str(e)}"
        )


@app.post("/infer/stream/stop", status_code=200)
async def stop_camera_stream() -> Dict[str, Any]:
    """Stop the active camera stream."""
    global streaming_state
    
    try:
        if not streaming_state["active"]:
            return {
                "status": "not_active",
                "message": "No active stream to stop"
            }
        
        # Stop camera
        if streaming_state["camera"]:
            streaming_state["camera"].stop()
        
        total_frames = streaming_state["frame_count"]
        
        # Reset state
        streaming_state.update({
            "active": False,
            "camera": None,
            "latest_frame": None,
            "latest_result": None,
            "frame_count": 0
        })
        
        logger.info(f"✓ Camera stream stopped (processed {total_frames} frames)")
        
        return {
            "status": "success",
            "message": "Stream stopped",
            "total_frames_processed": total_frames
        }
        
    except Exception as e:
        logger.error(f"Failed to stop stream: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream stop failed: {str(e)}"
        )


@app.get("/infer/stream/status", status_code=200)
async def get_stream_status() -> Dict[str, Any]:
    """Get current streaming status and statistics."""
    return {
        "stream_active": streaming_state["active"],
        "frames_captured": streaming_state["frame_count"],
        "has_latest_frame": streaming_state["latest_frame"] is not None,
        "has_latest_result": streaming_state["latest_result"] is not None,
        "model_loaded": model_adapter is not None,
        "current_model": model_name if model_adapter else None
    }