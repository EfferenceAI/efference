"""
SQLAlchemy ORM models for database tables.

This defines the actual database schema for:
- Customers
- API Keys
- Credit Transactions
"""
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Enum as SqlEnum, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from enum import Enum
from .base import Base


class APIKeyStatus(str, Enum):
    """Status states for API keys."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"

class UsageType(str, Enum):
    """Types of service usage."""
    VIDEO_INFERENCE = "video_inference"
    FRAME_PROCESSING = "frame_processing"
    VIDEO_STORAGE = "video_storage"
    IMAGE_RGBD = "image_rgbd"
    VIDEO_BATCH = "video_batch"
    STREAM_INFERENCE = "stream_inference"  


class Customer(Base):
    """Customer account table."""
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    
    # Credit tracking
    credits_available = Column(Float, default=0.0, nullable=False)
    credits_total = Column(Float, default=0.0, nullable=False)
    credits_used = Column(Float, default=0.0, nullable=False)
    
    # Account status
    status = Column(String, default="active", nullable=False)  # active, suspended, inactive
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="customer", cascade="all, delete-orphan")
    transactions = relationship("CreditTransaction", back_populates="customer", cascade="all, delete-orphan")


class APIKey(Base):
    """API keys for customer authentication table."""
    __tablename__ = "api_keys"

    key_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False, index=True)
    api_key_hash = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    
    # Status and expiration
    status = Column(SqlEnum(APIKeyStatus), default=APIKeyStatus.ACTIVE, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Rate limiting
    rate_limit = Column(Integer, default=100, nullable=False)
    
    # Usage tracking
    last_used = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="api_keys")


class CreditTransaction(Base):
    """Credit transaction history table."""
    __tablename__ = "credit_transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False, index=True)
    key_id = Column(String, ForeignKey("api_keys.key_id"), nullable=False, index=True)
    
    # Transaction details
    usage_type = Column(SqlEnum(UsageType), nullable=False)
    credits_deducted = Column(Float, nullable=False)
    
    # Metadata for transaction details (JSON)
    meta = Column(JSON, default={})
    
    # Status
    status = Column(String, default="completed", nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="transactions")