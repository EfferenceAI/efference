"""
FastAPI dependencies for authentication and database sessions.
"""
import logging
from typing import Tuple
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from ..db import get_db
from ..db.models import APIKey, Customer
from .api import validate_api_key

logger = logging.getLogger(__name__)

# API Key Header security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def validate_customer_api_key(
    authorization: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> Tuple[APIKey, Customer]:
    """
    Validate customer API key from Authorization header.
    
    Expected format: "Bearer <API_KEY>"
    
    Returns:
        Tuple of (api_key_record, customer_record)
        
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        scheme, _, api_key = authorization.partition(' ')
        if scheme.lower() != 'bearer':
            raise ValueError("Invalid scheme")
        
        if not api_key:
            raise ValueError("Empty API key")
        
        # Validate the API key
        key_record, customer = validate_api_key(db, api_key)
        
        if not key_record or not customer:
            raise ValueError("Invalid API key")
        
        return key_record, customer
        
    except ValueError as e:
        logger.warning(f"API key validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"}
        )