"""Health check endpoints."""

import logging
import os
from typing import Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import Customer, APIKey, CreditTransaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model_server:8080/infer")


@router.get("/health", status_code=200)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for the control plane.
    
    Returns:
        Dictionary with status and service information
    """
    return {
        "status": "healthy",
        "service": "control-plane",
        "model_server_url": MODEL_SERVER_URL
    }


@router.get("/health/db", status_code=200)
async def db_health_check(db: Session = Depends(get_db)) -> Dict[str, any]:
    """
    Health check with database connectivity.
    
    Returns:
        Database connection status
    """
    try:
        customers_count = db.query(Customer).count()
        api_keys_count = db.query(APIKey).count()
        transactions_count = db.query(CreditTransaction).count()
        
        return {
            "status": "healthy",
            "database": "connected",
            "customers_count": customers_count,
            "api_keys_count": api_keys_count,
            "transactions_count": transactions_count
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }