"""Routers module for control plane."""
from .health import router as health_router
from .videos import router as videos_router
from .admin import router as admin_router

__all__ = ["health_router", "videos_router", "admin_router"]