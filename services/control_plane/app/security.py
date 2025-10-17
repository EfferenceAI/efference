"""
Security and Authentication for customer API keys with Credit tracking
"""
import os
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import hashlib
from .config import DATABASE_URL
logger = logging.getLogger(__name__)

# From my research, we typically want to pass the API key in the request headers
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
#api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False) # In case the above doesn't work

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()