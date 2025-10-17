"""Routers module for control plane."""
from .health import router as health_router
from .videos import router as videos_router
from .admin import customers_router, api_keys_router, credits_router

__all__ = [
    "health_router",
    "videos_router", 
    "customers_router",
    "api_keys_router",
    "credits_router"
]