"""
API Key management and validation
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from ..db.models import APIKey, Customer, APIKeyStatus
from .security import hash_api_key, get_customer

logger = logging.getLogger(__name__)


# ============================================================================
# API Key Management
# ============================================================================

def generate_api_key() -> str:
    """Generate a secure API key."""
    # Format: sk_live_<random>
    return f"sk_live_{secrets.token_urlsafe(32)}"


def create_api_key(
    db: Session,
    customer_id: str,
    key_name: str,
    expires_in_days: Optional[int] = None,
    rate_limit: int = 100
) -> Tuple[str, APIKey]:
    """
    Create a new API key for a customer.
    
    Returns:
        Tuple of (api_key_plain_text, key_record)
        WARNING: Plain text is only shown once!
    """
    customer = get_customer(db, customer_id)
    if not customer:
        raise ValueError(f"Customer {customer_id} not found")
    
    api_key = generate_api_key()
    key_id = f"key_{secrets.token_hex(8)}"
    
    key_record = APIKey(
        key_id=key_id,
        customer_id=customer_id,
        api_key_hash=hash_api_key(api_key),
        name=key_name,
        status=APIKeyStatus.ACTIVE,
        created_at=datetime.utcnow(),
        last_used=None,
        expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
        rate_limit=rate_limit
    )
    
    db.add(key_record)
    db.commit()
    db.refresh(key_record)
    logger.info(f"Created API key {key_id} for customer {customer_id}")
    
    return api_key, key_record


def get_api_key(db: Session, key_id: str) -> Optional[APIKey]:
    """Retrieve API key record."""
    return db.query(APIKey).filter(APIKey.key_id == key_id).first()


def validate_api_key(db: Session, api_key_plain: str) -> Tuple[Optional[APIKey], Optional[Customer]]:
    """
    Validate an API key and return the key record and customer.
    
    Returns:
        Tuple of (api_key_record, customer_record) or (None, None) if invalid
    """
    api_key_hash = hash_api_key(api_key_plain)
    
    # Search for key by hash
    key_record = db.query(APIKey).filter(APIKey.api_key_hash == api_key_hash).first()
    
    if not key_record:
        logger.warning("API key not found or invalid")
        return None, None
    
    # Check if key is active
    if key_record.status != APIKeyStatus.ACTIVE:
        logger.warning(f"API key {key_record.key_id} is {key_record.status}")
        return None, None
    
    # Check if key is expired
    if key_record.expires_at and key_record.expires_at < datetime.utcnow():
        logger.warning(f"API key {key_record.key_id} has expired")
        return None, None
    
    # Get customer
    customer = get_customer(db, key_record.customer_id)
    if not customer:
        logger.error(f"Customer {key_record.customer_id} not found")
        return None, None
    
    # Check customer status
    if customer.status != "active":
        logger.warning(f"Customer {customer.customer_id} is {customer.status}")
        return None, None
    
    # Update last used
    key_record.last_used = datetime.utcnow()
    db.commit()
    
    return key_record, customer


def revoke_api_key(db: Session, key_id: str) -> bool:
    """Revoke an API key."""
    key_record = get_api_key(db, key_id)
    if not key_record:
        return False
    
    key_record.status = APIKeyStatus.REVOKED
    db.commit()
    logger.info(f"Revoked API key {key_id}")
    return True