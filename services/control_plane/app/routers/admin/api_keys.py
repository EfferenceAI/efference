"""Admin endpoints for API key management."""

import logging
from typing import Dict, Any, Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ...db import get_db
from ...models import APIKeyCreateRequest
from ...services.security import get_customer
from ...services.api import create_api_key, revoke_api_key
from .auth import verify_admin_key, security

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api-keys", tags=["admin-api-keys"])


@router.post("", response_model=Dict[str, Any])
async def create_api_key_endpoint(
    request: APIKeyCreateRequest,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Create a new API key for a customer."""
    verify_admin_key(credentials)
    
    try:
        api_key, key_record = create_api_key(
            db=db,
            customer_id=request.customer_id,
            key_name=request.name,
            expires_in_days=request.expires_in_days,
            rate_limit=request.rate_limit
        )
        
        return {
            "status": "success",
            "message": "API key created. WARNING: Save the key immediately, it won't be shown again!",
            "key_id": key_record.key_id,
            "api_key": api_key,
            "customer_id": request.customer_id,
            "name": request.name,
            "created_at": key_record.created_at
        }
    except Exception as e:
        logger.error(f"Failed to create API key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/{key_id}/revoke", response_model=Dict[str, Any])
async def revoke_api_key_endpoint(
    key_id: str,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Revoke an API key."""
    verify_admin_key(credentials)
    
    success = revoke_api_key(db, key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )
    
    return {
        "status": "success",
        "message": f"API key {key_id} revoked"
    }