"""
Data Service - Main FastAPI Application
=====================================
Main FastAPI application with health check endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from health_endpoints import health_router
from api_endpoints import api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Data Service",
    description="Enterprise-grade data processing service for ScanSmart",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Data Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

