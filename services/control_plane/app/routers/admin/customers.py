"""Admin endpoints for customer management."""

import logging
import secrets
from typing import Dict, Any, Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ...db import get_db
from ...models import CustomerCreateRequest
from ...services.security import create_customer, get_customer
from .auth import verify_admin_key, security

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/customers", tags=["admin-customers"])


@router.post("", response_model=Dict[str, Any])
async def create_customer_endpoint(
    request: CustomerCreateRequest,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Create a new customer account."""
    verify_admin_key(credentials)
    
    try:
        customer_id = f"cust_{secrets.token_hex(6)}"
        
        customer = create_customer(
            db=db,
            customer_id=customer_id,
            name=request.name,
            email=request.email,
            initial_credits=request.initial_credits
        )
        
        return {
            "status": "success",
            "customer_id": customer_id,
            "message": f"Created customer: {request.name}",
            "customer": {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "credits_available": customer.credits_available,
                "credits_total": customer.credits_total,
                "credits_used": customer.credits_used,
                "status": customer.status,
                "created_at": customer.created_at,
                "updated_at": customer.updated_at
            }
        }
    except IntegrityError as e:
        db.rollback()
        if "email" in str(e).lower():
            logger.warning(f"Email already exists: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{request.email}' already exists"
            )
        elif "customer_id" in str(e).lower():
            logger.warning(f"Customer ID already exists: {customer_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer ID already exists (this is very unlikely)"
            )
        else:
            logger.error(f"Integrity error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A customer with these details already exists"
            )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create customer"
        )


@router.get("/{customer_id}", response_model=Dict[str, Any])
async def get_customer_endpoint(
    customer_id: str,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Get customer information."""
    verify_admin_key(credentials)
    
    customer = get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found"
        )
    
    return {
        "status": "success",
        "customer": {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "credits_available": customer.credits_available,
            "credits_total": customer.credits_total,
            "credits_used": customer.credits_used,
            "status": customer.status,
            "created_at": customer.created_at,
            "updated_at": customer.updated_at
        }
    }