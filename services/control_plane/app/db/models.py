"""
This is the Data Models for customer management, API Keys, and Credit Tracking

"""
from pydantic import Field, BaseModel
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Enum as SqlEnum, Integer, JSON, ForeignKey
from typing import Optional
from sqlalchemy.orm import relationship
from enum import Enum
from .db import BaseModel



class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"

class UsageType(str, Enum):
    """ Types of Service Usage. """
    VIDEO_INFERENCE = "video_inference"
    FRAME_PROCESSING = "frame_processing"
    VIDEO_STORAGE = "video_storage"

class Customer(Base):
    """Customer account information."""
    __tablename__ = "customers"

    
    customer_id = Column(String, primary_key=True, index=True, description="Unique identifier for the customer")
    name = Column(String, description="Full name of the customer")
    email = Column(String, description="Email address of the customer")

    # account status
    status = Column(String, default="active", description="Account status: active, suspended, inactive")

    # credit tracking
    credits_available = Column(Float, default=0.0, description="Available credits for the customer")
    # I can essentially calculate this as credits_total - credits_available
    # But I don't want to do that calculation every time I need to access credits used (so I want just one query instead of two)
    credits_used = Column(Float, default=0.0, description="Total credits used by the customer")
    credits_total = Column(Float, default=0.0, description="Total credits allocated to the customer")

    # timestamps
    created_at = Column(DateTime, default_factory=datetime.utcnow, description="Timestamp when the customer was created")
    updated_at = Column(DateTime, default_factory=datetime.utcnow, description="Timestamp when the customer was last updated")

    # relationships
    api_keys = relationship("APIKey", back_populates="customer", cascade="all, delete-orphan")
    credit_transactions = relationship("CreditTransaction", back_populates="customer", cascade="all, delete-orphan")


class APIKey(BaseModel):
    """API key for customer authentication."""
    __tablename__ = "api_keys"

    key_id = Column(String, primary_key=True, index=True, description="Unique key ID")
    customer_id = Column(String, ForeignKey("customers.customer_id"), description="Customer this key belongs to")
    api_key = Column(String, description="The actual API key (hashed in DB)")
    name = Column(String, description="Key name/description")

    # status and expiration
    status = Column(Enum(APIKeyStatus), default=APIKeyStatus.ACTIVE, description="Key status")
    expires_at = Column(DateTime, default=None, description="Expiration time")

    # rate limiting
    rate_limit = Column(Integer, default=100, description="Requests per minute")

    # usage tracking
    last_used = Column(DateTime, default=None, description="Last usage timestamp")

    # timestamps
    created_at = Column(DateTime, default_factory=datetime.utcnow)

    # relationships
    customer = relationship("Customer", back_populates="api_keys")


class CreditTransaction(BaseModel):
    """Record of credit usage."""
    __tablename__ = "credit_transactions"

    transaction_id = Column(String, primary_key=True, index=True, description="Unique transaction ID")
    customer_id = Column(String, ForeignKey("customers.customer_id"), description="Customer ID")
    key_id = Column(String, ForeignKey("api_keys.key_id"), description="API key used")

    # transaction details
    usage_type = Column(Enum(UsageType), description="Type of usage")
    credits_deducted = Column(Float, description="Credits charged")

    # metadata in this context can include details like: videofiles, duration, resolution, inference_time_ms, etc.
    metadata = Column(JSON, default=dict, description="Additional metadata")

    # status and timestamps
    status = Column(String, default="completed", description="Transaction status")
    created_at = Column(DateTime, default_factory=datetime.utcnow)

    # relationships
    customer = relationship("Customer", back_populates="credit_transactions")




class CustomerCreateRequest(BaseModel):
    """Request to create a new customer."""
    customer_id: str = Field(..., description="Unique customer ID")
    name: str = Field(..., description="Customer name")
    email: str = Field(..., description="Customer email")
    initial_credits: float = Field(default=100.0, description="Initial credit allocation")


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key."""
    customer_id: str = Field(..., description="Customer ID")
    name: str = Field(..., description="Key name")
    expires_in_days: Optional[int] = Field(default=None, description="Days until expiration")
    rate_limit: int = Field(default=100, description="Requests per minute")


class CreditPurchaseRequest(BaseModel):
    """Request to add credits to customer account."""
    customer_id: str = Field(..., description="Customer ID")
    credits_amount: float = Field(..., gt=0, description="Credits to add")
    notes: str = Field(default="", description="Purchase notes")


class APIKeyResponse(BaseModel):
    """Response when creating an API key (only shown once)."""
    key_id: str
    api_key: str = Field(..., description="WARNING: Only shown once! Save immediately")
    customer_id: str
    name: str
    created_at: datetime


class CustomerResponse(BaseModel):
    """Response for customer information."""
    customer_id: str
    name: str
    email: str
    credits_available: float
    credits_used: float
    status: str
    created_at: datetime
    updated_at: datetime

