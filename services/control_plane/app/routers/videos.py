"""Video processing endpoints."""

import os
import logging
from typing import Dict, Any, Tuple

import httpx
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import UsageType
from ..constants import calculate_credits
from ..services.deps import validate_customer_api_key
from ..services.security import deduct_credits, get_customer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/videos", tags=["videos"])

# Configuration
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model_server:8080/infer")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes default
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "500000000"))  # 500MB default


@router.post("/process", status_code=200)
async def process_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    auth_data: Tuple = Depends(validate_customer_api_key)
) -> Dict[str, Any]:
    """
    Process a video through the ML model with credit deduction.
    
    **Authentication**: Requires Bearer token in Authorization header
    
    **Request**:
    - `Authorization: Bearer <API_KEY>`
    - `video`: Video file (multipart form data)
    
    **Returns**:
    - JSON response with inference results + credits consumed
    
    **Errors**:
    - 401: Missing or invalid API key
    - 402: Insufficient credits
    - 400: Invalid video file
    - 413: File too large
    - 503: Model service unavailable
    - 504: Model service timeout
    """
    
    key_record, customer = auth_data
    customer_id = customer.customer_id
    
    # Validate file
    if not video.filename:
        logger.warning("Request received without filename")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video file is required"
        )
    
    try:
        # Read file content with size validation
        video_content = await video.read()
        
        if len(video_content) == 0:
            logger.warning(f"Empty video file: {video.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video file is empty"
            )
        
        if len(video_content) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(video_content)} > {MAX_FILE_SIZE}")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video file exceeds maximum size of {MAX_FILE_SIZE / 1e6:.0f}MB"
            )
        
        file_size_mb = len(video_content) / 1e6
        logger.info(f"Processing video for {customer_id}: {video.filename} ({file_size_mb:.1f}MB)")
        
        # Forward request to model server
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            try:
                files = {
                    "video": (video.filename, video_content, video.content_type or "video/mp4")
                }
                
                logger.debug(f"Forwarding request to: {MODEL_SERVER_URL}")
                response = await client.post(MODEL_SERVER_URL, files=files)
                response.raise_for_status()
                
                inference_result = response.json()
                
                # Extract video metadata to calculate credits
                video_metadata = inference_result.get("video_metadata", {})
                if video_metadata:
                    # Calculate credits needed
                    credits_needed = calculate_credits(video_metadata)
                    
                    # Check if customer has enough credits
                    if customer.credits_available < credits_needed:
                        logger.warning(
                            f"Insufficient credits for {customer_id}. "
                            f"Need: {credits_needed}, Have: {customer.credits_available}"
                        )
                        raise HTTPException(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail=f"Insufficient credits. Need {credits_needed}, have {customer.credits_available}"
                        )
                    
                    # Deduct credits
                    success = deduct_credits(
                        db=db,
                        customer_id=customer_id,
                        credits_amount=credits_needed,
                        key_id=key_record.key_id,
                        usage_type=UsageType.VIDEO_INFERENCE,
                        metadata={
                            "video_file": video.filename,
                            "video_duration": video_metadata.get("frame_count", 0) / max(video_metadata.get("fps", 1), 1),
                            "frame_count": video_metadata.get("frame_count"),
                            "fps": video_metadata.get("fps"),
                            "file_size_bytes": len(video_content)
                        }
                    )
                    
                    if not success:
                        logger.error(f"Failed to deduct credits for {customer_id}")
                        raise HTTPException(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail="Failed to process payment"
                        )
                    
                    # Get updated customer info for response
                    updated_customer = get_customer(db, customer_id)
                    
                    # Add credit info to response
                    inference_result["credits_deducted"] = credits_needed
                    inference_result["credits_remaining"] = updated_customer.credits_available
                    
                    logger.info(
                        f"Successfully processed video for {customer_id}. "
                        f"Credits deducted: {credits_needed}, "
                        f"Remaining: {updated_customer.credits_available}"
                    )
                else:
                    logger.warning("No video_metadata in inference result")
                
                return inference_result
                
            except httpx.TimeoutException:
                logger.error(f"Model server timeout after {REQUEST_TIMEOUT}s")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Model server did not respond within {REQUEST_TIMEOUT}s"
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"Model server error: {e.status_code}")
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except:
                    error_detail = str(e)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Model server error: {error_detail}"
                )
            except httpx.RequestError as e:
                logger.error(f"Cannot reach model server: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model server is unavailable. Please try again later."
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )