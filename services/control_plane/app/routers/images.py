"""Image processing endpoints."""

import logging
import httpx
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from sqlalchemy.orm import Session

from ..db import get_db
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
    customer_id: str = Depends(validate_customer_api_key),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process RGB image with optional depth for depth estimation/correction.
    
    - **rgb**: Required RGB image (PNG, JPG)
    - **depth**: Optional depth image from sensor (PNG, NPY)
    """
    try:
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
            credits_amount=credits_cost,
            usage_type="image_rgbd",
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