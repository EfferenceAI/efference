"""Admin routers module."""
from .customers import router as customers_router
from .api_keys import router as api_keys_router
from .credits import router as credits_router

__all__ = ["customers_router", "api_keys_router", "credits_router"]