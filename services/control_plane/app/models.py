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
    key_id: str = Field(..., description="Unique key ID")
    customer_id: str = Field(..., description="Customer this key belongs to")
    api_key: str = Field(..., description="The actual API key (hashed in DB)")
    name: str = Field(..., description="Key name/description")
    status: APIKeyStatus = Field(default=APIKeyStatus.ACTIVE, description="Key status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration time")
    rate_limit: int = Field(default=100, description="Requests per minute")




