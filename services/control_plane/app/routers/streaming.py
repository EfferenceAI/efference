"""Live streaming and batch processing endpoints."""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import httpx
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import UsageType
from ..services.deps import validate_customer_api_key
from ..services.security import deduct_credits

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["streaming", "batch", "models"])

# Configuration
MODEL_SERVER_BASE = os.getenv("MODEL_SERVER_URL", "http://model_server:8000/infer").rstrip('/infer')
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))


# ============================================================================
# Batch Video Processing
# ============================================================================

@router.post("/videos/process-batch", status_code=200)
async def process_video_batch(
    video: UploadFile = File(...),
    max_frames: Optional[int] = Form(None, description="Maximum frames to process"),
    frame_skip: int = Form(1, description="Process every Nth frame"),
    db: Session = Depends(get_db),
    auth_data: Tuple = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """
    Process ALL frames from a video file using batch processing.
    More expensive than single-frame processing but provides complete analysis.
    """
    try:
        key_record, customer = auth_data
        customer_id = customer.customer_id
        logger.info(f"Batch processing request from customer {customer_id}")
        
        # Validate file
        if not video.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video filename is required"
            )
        
        # Read video content
        video_content = await video.read()
        if len(video_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video file is empty"
            )
        
        # Forward to model server
        files = {"video": (video.filename, video_content, "video/mp4")}
        data = {
            "max_frames": str(max_frames) if max_frames else "",
            "frame_skip": str(frame_skip)
        }
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{MODEL_SERVER_BASE}/infer-batch",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        # Calculate credits (batch processing is more expensive)
        base_credits = 2.0  # Higher base cost for batch
        file_size_mb = len(video_content) / (1024 * 1024)
        frames_processed = result.get("frames_processed", 1)
        
        # Credit formula: base + (frames * 0.1) + (size * 0.5)
        credits_cost = base_credits + (frames_processed * 0.1) + (file_size_mb * 0.5)
        
        # Deduct credits
        remaining_credits = deduct_credits(
            db=db,
            customer_id=customer_id,
            key_id=key_record.key_id,
            credits_amount=credits_cost,
            usage_type=UsageType.VIDEO_BATCH,
            metadata={
                "filename": video.filename,
                "frames_processed": frames_processed,
                "frame_skip": frame_skip,
                "file_size_mb": file_size_mb
            }
        )
        
        # Add billing info to response
        result.update({
            "credits_deducted": credits_cost,
            "credits_remaining": remaining_credits,
            "billing_info": {
                "base_cost": base_credits,
                "frame_cost": frames_processed * 0.1,
                "size_cost": file_size_mb * 0.5,
                "total": credits_cost
            }
        })
        
        logger.info(f"Batch processing complete: {frames_processed} frames, {credits_cost} credits")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Model server processing failed"
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch processing failed"
        )


# ============================================================================
# Model Management
# ============================================================================

@router.post("/models/switch", status_code=200)
async def switch_model(
    model_name: str = Form(..., description="Model to switch to (d435, d405)"),
    customer_id: str = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """
    Switch the active model on the model server.
    Allows switching between d435 and d405 variants.
    """
    try:
        logger.info(f"Model switch request from customer {customer_id}: {model_name}")
        
        # Forward to model server
        data = {"new_model_name": model_name}
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{MODEL_SERVER_BASE}/infer/models/switch",
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"Model switched successfully to {model_name}")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Model switch failed"
        )
    except Exception as e:
        logger.error(f"Model switch failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model switch failed"
        )


@router.get("/models/list", status_code=200)
async def list_models(
    customer_id: str = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """List all available models and their status."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{MODEL_SERVER_BASE}/infer/models/list")
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to list models"
        )


# ============================================================================
# Live Camera Streaming
# ============================================================================

@router.post("/stream/start", status_code=200)
async def start_camera_stream(
    camera_type: str = Form("realsense", description="Camera type"),
    customer_id: str = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """
    Start live camera streaming.
    Requires compatible hardware (RealSense camera).
    """
    try:
        logger.info(f"Stream start request from customer {customer_id}")
        
        # Forward to model server
        data = {"camera_type": camera_type}
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{MODEL_SERVER_BASE}/infer/stream/start",
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"Camera stream started for customer {customer_id}")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to start camera stream"
        )
    except Exception as e:
        logger.error(f"Stream start failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stream start failed"
        )


@router.get("/stream/frame", status_code=200)
async def get_stream_frame(
    run_inference: bool = False,
    db: Session = Depends(get_db),
    auth_data: Tuple = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """
    Get latest frame from active camera stream.
    Optionally run inference on the frame (costs credits).
    """
    try:
        key_record, customer = auth_data
        customer_id = customer.customer_id
        
        # Forward to model server
        params = {"run_inference": run_inference}
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{MODEL_SERVER_BASE}/infer/stream/frame",
                params=params
            )
            response.raise_for_status()
            result = response.json()
        
        # Deduct credits if inference was run
        if run_inference and "inference_result" in result.get("frame_data", {}):
            credits_cost = 0.1  # Small cost per frame inference
            
            remaining_credits = deduct_credits(
                db=db,
                customer_id=customer_id,
                key_id=key_record.key_id,
                credits_amount=credits_cost,
                usage_type=UsageType.STREAM_INFERENCE,
                metadata={"frame_count": result.get("frame_data", {}).get("frame_count", 0)}
            )
            
            result["credits_deducted"] = credits_cost
            result["credits_remaining"] = remaining_credits
        
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get frame"
        )


@router.post("/stream/stop", status_code=200)
async def stop_camera_stream(
    customer_id: str = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """Stop the active camera stream."""
    try:
        logger.info(f"Stream stop request from customer {customer_id}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{MODEL_SERVER_BASE}/infer/stream/stop")
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"Camera stream stopped for customer {customer_id}")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to stop stream"
        )


@router.get("/stream/status", status_code=200)
async def get_stream_status(
    customer_id: str = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """Get current streaming status."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{MODEL_SERVER_BASE}/infer/stream/status")
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Model server error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to get stream status"
        )