"""Data models for customer management, API keys, and credit tracking."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class APIKeyStatus(str, Enum):
    """Status of an API key."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class UsageType(str, Enum):
    """Types of service usage."""
    VIDEO_INFERENCE = "video_inference"
    FRAME_PROCESSING = "frame_processing"
    VIDEO_STORAGE = "video_storage"


class Customer(BaseModel):
    """Customer account model."""
    customer_id: str = Field(..., description="Unique customer ID")
    name: str = Field(..., description="Customer name/company")
    email: str = Field(..., description="Customer email")
    credits_available: float = Field(default=0.0, description="Available credits")
    credits_total: float = Field(default=0.0, description="Total credits purchased")
    credits_used: float = Field(default=0.0, description="Total credits consumed")
    status: str = Field(default="active", description="Account status: active, suspended, inactive")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class APIKey(BaseModel):
    """API key for customer authentication."""
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
    transaction_id: str = Field(..., description="Unique transaction ID")
    customer_id: str = Field(..., description="Customer ID")
    api_key_id: str = Field(..., description="API key used")
    usage_type: UsageType = Field(..., description="Type of usage")
    credits_deducted: float = Field(..., description="Credits charged")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    status: str = Field(default="completed", description="Transaction status")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CustomerCreateRequest(BaseModel):
    """Request to create a new customer."""
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