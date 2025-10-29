"""Image processing endpoints."""

import logging
import httpx
import numpy as np
import io
import imageio
from typing import Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, Form
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import UsageType
from ..services.deps import validate_customer_api_key
from ..services.security import deduct_credits
from ..constants import CREDIT_COSTS
from ..config import MODEL_SERVER_URL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/images", tags=["images"])


@router.post("/rgbd")
async def process_rgbd_image(
    rgb: UploadFile = File(..., description="RGB image file"),
    depth: UploadFile = File(None, description="Optional depth image from sensor"),
    auth_data: tuple = Depends(validate_customer_api_key),  # Changed from customer_id
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process RGB image with optional depth for depth estimation/correction.
    
    - **rgb**: Required RGB image (PNG, JPG)
    - **depth**: Optional depth image from sensor (PNG, NPY)
    """
    try:
        # Unpack authentication data
        api_key, customer = auth_data
        customer_id = customer.customer_id
        
        # Read file sizes
        rgb_data = await rgb.read()
        rgb_size_mb = len(rgb_data) / (1024 * 1024)
        
        depth_data = None
        depth_size_mb = 0
        if depth is not None:
            depth_data = await depth.read()
            depth_size_mb = len(depth_data) / (1024 * 1024)
        
        total_size_mb = rgb_size_mb + depth_size_mb
        
        logger.info(
            f"Processing RGBD for {customer_id}: "
            f"RGB={rgb.filename} ({rgb_size_mb:.1f}MB), "
            f"Depth={depth.filename if depth else 'None'} ({depth_size_mb:.1f}MB)"
        )
        
        # Forward to model server
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"rgb": (rgb.filename, rgb_data, rgb.content_type)}
            if depth_data:
                files["depth"] = (depth.filename, depth_data, depth.content_type if depth.content_type else "image/png")
            
            response = await client.post(
                f"{MODEL_SERVER_URL.replace('/infer', '/infer-image')}",
                files=files
            )
            response.raise_for_status()
            result = response.json()
        
        # Calculate credits (base cost + per MB)
        credits_cost = CREDIT_COSTS["image_rgbd_base"] + (total_size_mb * CREDIT_COSTS["per_mb"])
        
        # Deduct credits
        remaining_credits = deduct_credits(
            db=db,
            customer_id=customer_id,
            key_id=api_key.key_id,
            credits_amount=credits_cost,
            usage_type=UsageType.IMAGE_RGBD,
            metadata={
                "rgb_filename": rgb.filename,
                "depth_filename": depth.filename if depth else None,
                "rgb_size_mb": round(rgb_size_mb, 2),
                "depth_size_mb": round(depth_size_mb, 2) if depth else 0,
                "model_name": result.get("model_name", "unknown")
            }
        )
        
        logger.info(
            f"Successfully processed RGBD for {customer_id}. "
            f"Credits deducted: {credits_cost:.2f}, Remaining: {remaining_credits:.2f}"
        )
        
        # Return combined response
        return {
            "status": "success",
            "rgb_filename": rgb.filename,
            "depth_filename": depth.filename if depth else None,
            "rgb_size_mb": round(rgb_size_mb, 2),
            "depth_size_mb": round(depth_size_mb, 2) if depth else 0,
            **result,
            "credits_deducted": round(credits_cost, 2),
            "credits_remaining": round(remaining_credits, 2)
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Model server error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/rgbd-advanced")
async def process_rgbd_advanced(
    # Format 1: RGB video + numpy depth
    rgb_video: Optional[UploadFile] = File(None, description="RGB video file (MP4, AVI)"),
    depth_numpy: Optional[UploadFile] = File(None, description="Depth data as numpy array (.npy)"),
    
    # Format 2: RGB video + OpenEXR depth  
    depth_exr: Optional[UploadFile] = File(None, description="Depth data as OpenEXR file (.exr)"),
    
    # Format 3: RGBD numpy array
    rgbd_numpy: Optional[UploadFile] = File(None, description="Combined RGBD data as numpy array (.npy)"),
    
    # Format 4: RGB numpy array + Depth numpy array
    rgb_numpy: Optional[UploadFile] = File(None, description="RGB data as numpy array (.npy)"),
    depth_numpy_separate: Optional[UploadFile] = File(None, description="Separate depth numpy array (.npy)"),
    
    # Processing parameters
    max_frames: Optional[int] = Form(None, description="Maximum frames to process for video inputs"),
    frame_skip: int = Form(1, description="Process every Nth frame"),
    
    auth_data: tuple = Depends(validate_customer_api_key),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Advanced RGBD processing supporting multiple input formats:
    
    **Format 1**: RGB video + numpy depth
    - `rgb_video`: Video file (MP4, AVI, etc.)
    - `depth_numpy`: Corresponding depth data (.npy file)
    
    **Format 2**: RGB video + OpenEXR depth
    - `rgb_video`: Video file (MP4, AVI, etc.) 
    - `depth_exr`: Depth data in OpenEXR format (.exr file)
    
    **Format 3**: RGBD numpy array
    - `rgbd_numpy`: Combined RGBD data as numpy array (.npy file)
    
    **Format 4**: RGB numpy array + Depth numpy array
    - `rgb_numpy`: RGB data as numpy array (.npy file)
    - `depth_numpy_separate`: Depth data as numpy array (.npy file)
    
    Returns numpy array of cleaned depth values.
    """
    try:
        # Unpack authentication data
        api_key, customer = auth_data
        customer_id = customer.customer_id
        
        # Determine input format based on provided files
        input_format = _determine_input_format(
            rgb_video, depth_numpy, depth_exr, rgbd_numpy, rgb_numpy, depth_numpy_separate
        )
        
        logger.info(f"Processing RGBD advanced for {customer_id} using format: {input_format}")
        
        # Process based on format
        if input_format == "rgb_video_numpy_depth":
            return await _process_rgb_video_numpy_depth(
                rgb_video, depth_numpy, max_frames, frame_skip,
                api_key, customer, customer_id, db
            )
            
        elif input_format == "rgb_video_exr_depth":
            return await _process_rgb_video_exr_depth(
                rgb_video, depth_exr, max_frames, frame_skip,
                api_key, customer, customer_id, db
            )
            
        elif input_format == "rgbd_numpy":
            return await _process_rgbd_numpy(
                rgbd_numpy, api_key, customer, customer_id, db
            )
            
        elif input_format == "rgb_numpy_depth_numpy":
            return await _process_rgb_numpy_depth_numpy(
                rgb_numpy, depth_numpy_separate, api_key, customer, customer_id, db
            )
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input format: {input_format}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced RGBD processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Advanced RGBD processing failed"
        )


def _determine_input_format(rgb_video, depth_numpy, depth_exr, rgbd_numpy, rgb_numpy, depth_numpy_separate) -> str:
    """Determine which input format is being used based on provided files."""
    
    if rgb_video and depth_numpy:
        return "rgb_video_numpy_depth"
    elif rgb_video and depth_exr:
        return "rgb_video_exr_depth"
    elif rgbd_numpy:
        return "rgbd_numpy"
    elif rgb_numpy and depth_numpy_separate:
        return "rgb_numpy_depth_numpy"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input combination. Please provide one of: "
                   "(rgb_video + depth_numpy), (rgb_video + depth_exr), "
                   "(rgbd_numpy), or (rgb_numpy + depth_numpy_separate)"
        )


async def _process_rgb_video_numpy_depth(
    rgb_video, depth_numpy, max_frames, frame_skip, api_key, customer, customer_id, db
) -> Dict[str, Any]:
    """Process Format 1: RGB video + numpy depth."""
    
    # Read files
    video_data = await rgb_video.read()
    depth_data = await depth_numpy.read()
    
    # Load numpy depth data
    depth_array = np.load(io.BytesIO(depth_data))
    
    # Calculate sizes
    video_size_mb = len(video_data) / (1024 * 1024)
    depth_size_mb = len(depth_data) / (1024 * 1024)
    total_size_mb = video_size_mb + depth_size_mb
    
    # Forward to model server with special handling for numpy depth
    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {
            "video": (rgb_video.filename, video_data, rgb_video.content_type or "video/mp4")
        }
        data = {
            "max_frames": str(max_frames) if max_frames else "",
            "frame_skip": str(frame_skip),
            "depth_format": "numpy",
            "depth_shape": str(list(depth_array.shape)),
            "depth_dtype": str(depth_array.dtype)
        }
        
        # Send depth as separate data
        files["depth_numpy"] = ("depth.npy", depth_data, "application/octet-stream")
        
        response = await client.post(
            f"{MODEL_SERVER_URL.replace('/infer', '/infer-advanced')}",
            files=files,
            data=data
        )
        response.raise_for_status()
        result = response.json()
    
    # Calculate credits
    frames_processed = result.get("frames_processed", 1)
    credits_cost = CREDIT_COSTS["video_batch_base"] + (frames_processed * 0.1) + (total_size_mb * 0.5)
    
    # Deduct credits
    remaining_credits = deduct_credits(
        db=db,
        customer_id=customer_id,
        key_id=api_key.key_id,
        credits_amount=credits_cost,
        usage_type=UsageType.VIDEO_BATCH,
        metadata={
            "input_format": "rgb_video_numpy_depth",
            "video_filename": rgb_video.filename,
            "depth_filename": depth_numpy.filename,
            "video_size_mb": round(video_size_mb, 2),
            "depth_size_mb": round(depth_size_mb, 2),
            "frames_processed": frames_processed,
            "depth_shape": list(depth_array.shape)
        }
    )
    
    # Return result with credits
    result.update({
        "input_format": "rgb_video_numpy_depth",
        "credits_deducted": round(credits_cost, 2),
        "credits_remaining": round(remaining_credits, 2)
    })
    
    return result


async def _process_rgb_video_exr_depth(
    rgb_video, depth_exr, max_frames, frame_skip, api_key, customer, customer_id, db
) -> Dict[str, Any]:
    """Process Format 2: RGB video + OpenEXR depth."""
    
    # Read files
    video_data = await rgb_video.read()
    exr_data = await depth_exr.read()
    
    # Calculate sizes
    video_size_mb = len(video_data) / (1024 * 1024)
    exr_size_mb = len(exr_data) / (1024 * 1024)
    total_size_mb = video_size_mb + exr_size_mb
    
    # Forward to model server
    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {
            "video": (rgb_video.filename, video_data, rgb_video.content_type or "video/mp4"),
            "depth_exr": (depth_exr.filename, exr_data, "image/x-exr")
        }
        data = {
            "max_frames": str(max_frames) if max_frames else "",
            "frame_skip": str(frame_skip),
            "depth_format": "exr"
        }
        
        response = await client.post(
            f"{MODEL_SERVER_URL.replace('/infer', '/infer-advanced')}",
            files=files,
            data=data
        )
        response.raise_for_status()
        result = response.json()
    
    # Calculate and deduct credits
    frames_processed = result.get("frames_processed", 1)
    credits_cost = CREDIT_COSTS["video_batch_base"] + (frames_processed * 0.1) + (total_size_mb * 0.5)
    
    remaining_credits = deduct_credits(
        db=db,
        customer_id=customer_id,
        key_id=api_key.key_id,
        credits_amount=credits_cost,
        usage_type=UsageType.VIDEO_BATCH,
        metadata={
            "input_format": "rgb_video_exr_depth",
            "video_filename": rgb_video.filename,
            "depth_filename": depth_exr.filename,
            "video_size_mb": round(video_size_mb, 2),
            "exr_size_mb": round(exr_size_mb, 2),
            "frames_processed": frames_processed
        }
    )
    
    result.update({
        "input_format": "rgb_video_exr_depth",
        "credits_deducted": round(credits_cost, 2),
        "credits_remaining": round(remaining_credits, 2)
    })
    
    return result


async def _process_rgbd_numpy(rgbd_numpy, api_key, customer, customer_id, db) -> Dict[str, Any]:
    """Process Format 3: RGBD numpy array."""
    
    # Read file
    rgbd_data = await rgbd_numpy.read()
    
    # Load numpy RGBD data
    rgbd_array = np.load(io.BytesIO(rgbd_data))
    
    # Calculate size
    file_size_mb = len(rgbd_data) / (1024 * 1024)
    
    # Forward to model server
    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {
            "rgbd_numpy": (rgbd_numpy.filename, rgbd_data, "application/octet-stream")
        }
        data = {
            "data_format": "rgbd_numpy",
            "rgbd_shape": str(list(rgbd_array.shape)),
            "rgbd_dtype": str(rgbd_array.dtype)
        }
        
        response = await client.post(
            f"{MODEL_SERVER_URL.replace('/infer', '/infer-advanced')}",
            files=files,
            data=data
        )
        response.raise_for_status()
        result = response.json()
    
    # Calculate and deduct credits
    credits_cost = CREDIT_COSTS["image_rgbd_base"] + (file_size_mb * CREDIT_COSTS["per_mb"])
    
    remaining_credits = deduct_credits(
        db=db,
        customer_id=customer_id,
        key_id=api_key.key_id,
        credits_amount=credits_cost,
        usage_type=UsageType.IMAGE_RGBD,
        metadata={
            "input_format": "rgbd_numpy",
            "filename": rgbd_numpy.filename,
            "file_size_mb": round(file_size_mb, 2),
            "rgbd_shape": list(rgbd_array.shape)
        }
    )
    
    result.update({
        "input_format": "rgbd_numpy",
        "credits_deducted": round(credits_cost, 2),
        "credits_remaining": round(remaining_credits, 2)
    })
    
    return result


async def _process_rgb_numpy_depth_numpy(
    rgb_numpy, depth_numpy_separate, api_key, customer, customer_id, db
) -> Dict[str, Any]:
    """Process Format 4: RGB numpy array + Depth numpy array."""
    
    # Read files
    rgb_data = await rgb_numpy.read()
    depth_data = await depth_numpy_separate.read()
    
    # Load numpy arrays
    rgb_array = np.load(io.BytesIO(rgb_data))
    depth_array = np.load(io.BytesIO(depth_data))
    
    # Calculate sizes
    rgb_size_mb = len(rgb_data) / (1024 * 1024)
    depth_size_mb = len(depth_data) / (1024 * 1024)
    total_size_mb = rgb_size_mb + depth_size_mb
    
    # Forward to model server
    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {
            "rgb_numpy": (rgb_numpy.filename, rgb_data, "application/octet-stream"),
            "depth_numpy": (depth_numpy_separate.filename, depth_data, "application/octet-stream")
        }
        data = {
            "data_format": "rgb_depth_numpy",
            "rgb_shape": str(list(rgb_array.shape)),
            "rgb_dtype": str(rgb_array.dtype),
            "depth_shape": str(list(depth_array.shape)),
            "depth_dtype": str(depth_array.dtype)
        }
        
        response = await client.post(
            f"{MODEL_SERVER_URL.replace('/infer', '/infer-advanced')}",
            files=files,
            data=data
        )
        response.raise_for_status()
        result = response.json()
    
    # Calculate and deduct credits
    credits_cost = CREDIT_COSTS["image_rgbd_base"] + (total_size_mb * CREDIT_COSTS["per_mb"])
    
    remaining_credits = deduct_credits(
        db=db,
        customer_id=customer_id,
        key_id=api_key.key_id,
        credits_amount=credits_cost,
        usage_type=UsageType.IMAGE_RGBD,
        metadata={
            "input_format": "rgb_numpy_depth_numpy",
            "rgb_filename": rgb_numpy.filename,
            "depth_filename": depth_numpy_separate.filename,
            "rgb_size_mb": round(rgb_size_mb, 2),
            "depth_size_mb": round(depth_size_mb, 2),
            "rgb_shape": list(rgb_array.shape),
            "depth_shape": list(depth_array.shape)
        }
    )
    
    result.update({
        "input_format": "rgb_numpy_depth_numpy",
        "credits_deducted": round(credits_cost, 2),
        "credits_remaining": round(remaining_credits, 2)
    })
    
    return result