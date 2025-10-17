"""
This is the Data Models for customer management, API Keys, and Credit Tracking

"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"

class UsageType(str, Enum):
    """ Types of Service Usage. """
    VIDEO_INFERENCE = "video_inference"
    FRAME_PROCESSING = "frame_processing"
    VIDEO_STORAGE = "video_storage"

class Customer(BaseModel):
    __table__ = "customers"
    """Customer account information."""
    customer_id: str = Field(..., description="Unique identifier for the customer")
    name: str = Field(..., description="Full name of the customer")
    email: EmailStr = Field(..., description="Email address of the customer")
    status: str = Field(default="active", description="Account status: active, suspended, inactive")
    credits_available: float = Field(default=0.0, description="Available credits for the customer")
    # I can essentially calculate this as credits_total - credits_available
    # But I don't want to do that calculation every time I need to access credits used (so I want just one query instead of two)
    credits_used: float = Field(default=0.0, description="Total credits used by the customer") 
    credits_total: float = Field(default=0.0, description="Total credits allocated to the customer")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the customer was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the customer was last updated")

class APIKey(BaseModel):
    """API key for customer authentication."""
    __table__ = "api_keys"
    key_id: str = Field(..., description="Unique key ID")
    customer_id: str = Field(..., description="Customer this key belongs to")
    api_key: str = Field(..., description="The actual API key (hashed in DB)")
    name: str = Field(..., description="Key name/description")
    status: APIKeyStatus = Field(default=APIKeyStatus.ACTIVE, description="Key status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration time")
    rate_limit: int = Field(default=100, description="Requests per minute")

class CreditTransaction(BaseModel):
    """Record of credit usage."""
    __table__ = "credit_transactions"
    transaction_id: str = Field(..., description="Unique transaction ID")
    customer_id: str = Field(..., description="Customer ID")
    api_key_id: str = Field(..., description="API key used")
    usage_type: UsageType = Field(..., description="Type of usage")
    credits_deducted: float = Field(..., description="Credits charged")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    status: str = Field(default="completed", description="Transaction status")
    created_at: datetime = Field(default_factory=datetime.utcnow)

# metadata in this context can include details like: videofiles, duration, resolution, inference_time_ms, etc.


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

