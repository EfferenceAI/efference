"""
Security and Authentication for customer API keys with Credit tracking
"""
import logging
import secrets
from datetime import datetime
import hashlib
from typing import Optional
from sqlalchemy.orm import Session

from ..db.models import Customer, CreditTransaction,APIKey, APIKeyStatus, UsageType

logger = logging.getLogger(__name__)


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================================================
# Customer Management
# ============================================================================

def create_customer(
    db: Session,
    customer_id: str,
    name: str,
    email: str, 
    initial_credits: float = 100.0
) -> Customer:
    """Create a new customer account."""
    existing = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if existing:
        raise ValueError(f"Customer {customer_id} already exists")

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


def add_credits(db: Session, customer_id: str, credits_amount: float) -> Customer:
    """Add credits to customer account."""
    customer = get_customer(db, customer_id)
    if not customer:
        raise ValueError(f"Customer {customer_id} not found")
    
    customer.credits_available += credits_amount
    customer.credits_total += credits_amount
    customer.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(customer)
    logger.info(f"Added {credits_amount} credits to {customer_id}. Total available: {customer.credits_available}")
    return customer


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
        meta=metadata or {},
        status="completed",
        created_at=datetime.utcnow()
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(customer)
    
    logger.info(f"Deducted {credits_amount} credits from {customer_id}. Remaining: {customer.credits_available}")
    return True



# ============================================================================
# Demo Data Setup (for development)
# ============================================================================

def setup_demo_data():
    """Create demo customers and keys for testing."""
    from ..db import SessionLocal
    
    db = SessionLocal()
    try:
        # Create demo customer (if it doesn't exist)
        try:
            create_customer(
                db=db,
                customer_id="demo_customer_1",
                name="Demo Customer",
                email="demo@example.com",
                initial_credits=1000.0
            )
            logger.info("Demo customer created")
        except ValueError as e:
            logger.info(f"Demo customer already exists: {str(e)}")
        
        # Check if demo API key already exists
        from .api import get_api_key
        existing_key = db.query(APIKey).filter(
            APIKey.customer_id == "demo_customer_1",
            APIKey.name == "Demo Key"
        ).first()
        
        if existing_key:
            # API key already exists, just return it (but we can't show the plaintext again)
            logger.info("Demo API key already exists, not creating a new one")
            return None
        
        # Create demo API key only if it doesn't exist
        try:
            from .api import create_api_key
            api_key, key_record = create_api_key(
                db=db,
                customer_id="demo_customer_1",
                key_name="Demo Key",
                expires_in_days=30,
                rate_limit=100
            )
            logger.info(f"Demo API Key created: {api_key}")
            return api_key
        except ValueError as e:
            logger.info(f"Demo API key creation failed: {str(e)}")
            return None
    finally:
        db.close()