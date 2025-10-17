"""
Security and Authentication for customer API keys with Credit tracking
"""
import os
import logging
import secrets
from datetime import datetime, timedelta

import hashlib
from .config import DATABASE_URL
from sqlalchemy.orm import Session

from typing import Dict, Optional, Tuple, Any
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from .db import SessionLocal, get_db
from .db.models import Customer, APIKey, CreditTransaction, APIKeyStatus, UsageType



logger = logging.getLogger(__name__)

# From my research, we typically want to pass the API key in the request headers
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
#api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False) # In case the above doesn't work

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


### Customer Management
def create_customer(
        db: Session,
        customer_id: str,
        name: str,
        email: str, 
        initial_credits: float = 100.0
) -> Customer:
    """Create a new customer account."""
    # check if customer already exists
    existing = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer ID already exists."
        )

    new_customer = Customer(
        customer_id=customer_id,
        name=name,
        email=email,
        credits_available=initial_credits,
        credits_total=initial_credits,
        credits_used=0.0,
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)
    logger.info(f"Created new customer: {customer_id}")
    return new_customer

def get_customer(db: Session, customer_id: str) -> Optional[Customer]:
    """Retrieve customer information."""
    return db.query(Customer).filter(Customer.customer_id == customer_id).first()


def deduct_credits(
    db: Session,
    customer_id: str,
    credits_amount: float,
    key_id: str,
    usage_type: UsageType = UsageType.VIDEO_INFERENCE,
    metadata: dict = None
) -> bool:
    """Deduct credits from customer account. Returns True if successful, False if insufficient."""
    customer = get_customer(db, customer_id)
    if not customer:
        raise ValueError(f"Customer {customer_id} not found")
    
    if customer.credits_available < credits_amount:
        logger.warning(f"Insufficient credits for {customer_id}. Available: {customer.credits_available}, Needed: {credits_amount}")
        return False
    
    customer.credits_available -= credits_amount
    customer.credits_used += credits_amount
    customer.updated_at = datetime.utcnow()
    
    # Record transaction
    txn_id = f"txn_{secrets.token_hex(8)}"
    transaction = CreditTransaction(
        transaction_id=txn_id,
        customer_id=customer_id,
        key_id=key_id,
        usage_type=usage_type,
        credits_deducted=credits_amount,
        metadata=metadata or {},
        status="completed",
        created_at=datetime.utcnow()
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(customer)
    
    logger.info(f"Deducted {credits_amount} credits from {customer_id}. Remaining: {customer.credits_available}")
    return True