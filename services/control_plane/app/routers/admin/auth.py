"""Admin authentication utilities."""

import os
import logging
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Simple admin key for securing admin endpoints
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin-secret-key-change-me")

security = HTTPBearer()


def verify_admin_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Verify admin authorization."""
    try:
        if credentials.scheme.lower() != 'bearer' or credentials.credentials != ADMIN_KEY:
            raise ValueError("Invalid admin key")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )