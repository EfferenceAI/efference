"""Admin endpoints for credit management."""

import logging
from typing import Dict, Any, Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.models import CreditTransaction
from ...models import CreditPurchaseRequest
from ...services.security import get_customer, add_credits
from .auth import verify_admin_key, security

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/credits", tags=["admin-credits"])


@router.post("/add", response_model=Dict[str, Any])
async def add_credits_endpoint(
    request: CreditPurchaseRequest,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Add credits to a customer account."""
    verify_admin_key(credentials)
    
    try:
        customer = add_credits(
            db=db,
            customer_id=request.customer_id,
            credits_amount=request.credits_amount
        )
        
        return {
            "status": "success",
            "message": f"Added {request.credits_amount} credits to {request.customer_id}",
            "customer_id": request.customer_id,
            "credits_available": customer.credits_available,
            "credits_used": customer.credits_used
        }
    except Exception as e:
        logger.error(f"Failed to add credits: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/transactions/{customer_id}", response_model=Dict[str, Any])
async def get_customer_transactions(
    customer_id: str,
    db: Session = Depends(get_db),
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None
) -> Dict[str, Any]:
    """Get transaction history for a customer."""
    verify_admin_key(credentials)
    
    customer = get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found"
        )
    
    transactions = db.query(CreditTransaction).filter(
        CreditTransaction.customer_id == customer_id
    ).all()
    
    return {
        "status": "success",
        "customer_id": customer_id,
        "transaction_count": len(transactions),
        "transactions": sorted(
            [
                {
                    "transaction_id": t.transaction_id,
                    "credits_deducted": t.credits_deducted,
                    "usage_type": t.usage_type.value,
                    "status": t.status,
                    "created_at": t.created_at,
                    "metadata": t.meta
                }
                for t in transactions
            ],
            key=lambda x: x["created_at"],
            reverse=True
        )
    }