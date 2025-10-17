"""Efference Control Plane - Public API Gateway."""

import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import HTTPException, status

from .db import init_db
from .services.security import setup_demo_data
from .routers import (
    health_router, 
    videos_router, 
    customers_router,
    api_keys_router,
    credits_router
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Efference Control Plane",
    description="Public API gateway with authentication for RGBD model inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include routers
app.include_router(health_router)
app.include_router(videos_router)
app.include_router(customers_router)
app.include_router(api_keys_router)
app.include_router(credits_router)


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Setup database and demo data on application startup."""
    logger.info("=" * 70)
    logger.info("EFFERENCE CONTROL PLANE - STARTUP")
    logger.info("=" * 70)
    
    try:
        # Initialize database tables
        init_db()
        logger.info("Database initialized successfully")
        
        # Setup demo customer and API key
        demo_api_key = setup_demo_data()
        logger.info("")
        logger.info("DEMO CUSTOMER CREATED:")
        logger.info(f"  Customer ID: demo_customer_1")
        logger.info(f"  Name: Demo Customer")
        logger.info(f"  Email: demo@example.com")
        logger.info(f"  Initial Credits: 1000.0")
        logger.info("")
        if demo_api_key:
            logger.info("DEMO API KEY (SAVE THIS!):")
            logger.info(f"  {demo_api_key}")
        logger.info("")
        logger.info("Usage:")
        logger.info('  curl -X POST http://localhost:8080/v1/videos/process \\')
        if demo_api_key:
            logger.info(f'    -H "Authorization: Bearer {demo_api_key}" \\')
        logger.info('    -F "video=@test_video.mp4"')
        logger.info("")
        logger.info("=" * 70)
        logger.info("Control Plane ready!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to setup: {str(e)}")
        # Don't fail startup, just log warning


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path)
            }
        },
        headers=exc.headers if hasattr(exc, 'headers') else {}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "status_code": 500,
                "message": "Internal server error",
                "path": str(request.url.path)
            }
        }
    )