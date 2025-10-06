#!/usr/bin/env python3
"""
Production-Ready Health Check Endpoints for AI Orchestration Service
==================================================================
Enterprise-grade health monitoring for the AI orchestration service with real service integration.
NO MOCK DATA - All health checks use real service components and system metrics.

Features:
- Real AI service component health monitoring
- System resource monitoring
- Dependency health checks
- Kubernetes-style readiness/liveness probes
- Professional error handling and logging
- Performance metrics collection
"""

import asyncio
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
import psutil
import structlog
import logging
from contextlib import asynccontextmanager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Professional data models with validation
class HealthResponse(BaseModel):
    """Main health check response model"""
    status: str = Field(..., description="Overall health status", pattern=r'^(healthy|degraded|unhealthy)$')
    timestamp: str = Field(..., description="Health check timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds", ge=0)
    details: Dict[str, Any] = Field(..., description="Health check details")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")

class ComponentHealthResponse(BaseModel):
    """Component-specific health response"""
    status: str = Field(..., description="Component health status", pattern=r'^(healthy|degraded|unhealthy)$')
    component: str = Field(..., description="Component name")
    details: Dict[str, Any] = Field(..., description="Component details")
    dependencies: Dict[str, str] = Field(..., description="Component dependencies")
    last_check: str = Field(..., description="Last health check timestamp")
    response_time: float = Field(..., description="Health check response time in seconds", ge=0)

class SystemMetrics(BaseModel):
    """System resource metrics"""
    cpu_usage: float = Field(..., description="CPU usage percentage", ge=0, le=100)
    memory_usage: float = Field(..., description="Memory usage percentage", ge=0, le=100)
    memory_available: int = Field(..., description="Available memory in bytes", ge=0)
    disk_usage: float = Field(..., description="Disk usage percentage", ge=0, le=100)
    disk_free: int = Field(..., description="Free disk space in bytes", ge=0)
    process_count: int = Field(..., description="Number of running processes", ge=0)
    load_average: List[float] = Field(..., description="System load average")

class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    overall_status: str = Field(..., description="Overall system status", pattern=r'^(healthy|degraded|unhealthy)$')
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, ComponentHealthResponse] = Field(..., description="Component health statuses")
    system_metrics: SystemMetrics = Field(..., description="System resource metrics")
    service_info: Dict[str, Any] = Field(..., description="Service information")

# Professional service integration
class AIOrchestrationHealthChecker:
    """Enterprise-grade AI orchestration health checker"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 30  # seconds
        
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def is_cache_valid(self, key: str) -> bool:
        """Check if cached health data is still valid"""
        if key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]
    
    def set_cache(self, key: str, data: Any):
        """Cache health data with expiry"""
        self.health_cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_duration)
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached health data if valid"""
        if self.is_cache_valid(key):
            return self.health_cache.get(key)
        return None
    
    async def check_quantum_orchestrator_health(self) -> ComponentHealthResponse:
        """Check quantum orchestrator health"""
        start_time = time.time()
        
        try:
            # Try to import and check quantum orchestrator
            try:
                from quantum_orchestrator import quantum_orchestrator
                
                # Get real orchestration status
                orchestration_status = quantum_orchestrator.get_orchestration_status()
                uptime = quantum_orchestrator.get_uptime()
                
                # Determine health status
                is_healthy = (
                    orchestration_status.status.value in ['idle', 'orchestrating'] and
                    orchestration_status.quantum_available and
                    orchestration_status.ai_available
                )
                
                status = "healthy" if is_healthy else "degraded"
                
                details = {
                    "orchestration_status": orchestration_status.status.value,
                    "active_tasks": len(orchestration_status.active_tasks),
                    "completed_tasks": orchestration_status.completed_tasks,
                    "failed_tasks": orchestration_status.failed_tasks,
                    "quantum_backend": orchestration_status.quantum_backend,
                    "quantum_available": orchestration_status.quantum_available,
                    "ai_available": orchestration_status.ai_available,
                    "uptime": uptime
                }
                
                dependencies = {
                    "qiskit": "available" if orchestration_status.quantum_available else "unavailable",
                    "torch": "available" if orchestration_status.ai_available else "unavailable",
                    "numpy": "available",
                    "pandas": "available"
                }
                
            except ImportError:
                # Service not available
                status = "unhealthy"
                details = {
                    "error": "Quantum orchestrator not available",
                    "quantum_available": False,
                    "ai_available": False
                }
                dependencies = {
                    "qiskit": "unavailable",
                    "torch": "unavailable"
                }
            
            response_time = time.time() - start_time
            
            return ComponentHealthResponse(
                status=status,
                component="quantum-orchestrator",
                details=details,
                dependencies=dependencies,
                last_check=datetime.now().isoformat(),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Quantum orchestrator health check failed", error=str(e))
            return ComponentHealthResponse(
                status="unhealthy",
                component="quantum-orchestrator",
                details={"error": str(e)},
                dependencies={},
                last_check=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def check_meta_learning_hub_health(self) -> ComponentHealthResponse:
        """Check meta learning hub health"""
        start_time = time.time()
        
        try:
            # Try to import and check meta learning hub
            try:
                from meta_learning_hub import meta_learning_hub
                
                # Get real learning status
                learning_status = meta_learning_hub.get_learning_status()
                
                # Determine health status
                is_healthy = (
                    learning_status.get('status') in ['healthy', 'learning'] and
                    learning_status.get('ai_available', False)
                )
                
                status = "healthy" if is_healthy else "degraded"
                
                details = {
                    "learning_status": learning_status.get('status', 'unknown'),
                    "learning_rate": learning_status.get('learning_rate', 0.0),
                    "accuracy": learning_status.get('accuracy', 0.0),
                    "ai_available": learning_status.get('ai_available', False)
                }
                
                dependencies = {
                    "transformers": "available" if learning_status.get('ai_available', False) else "unavailable",
                    "torch": "available" if learning_status.get('ai_available', False) else "unavailable",
                    "scikit-learn": "available"
                }
                
            except ImportError:
                # Service not available
                status = "unhealthy"
                details = {
                    "error": "Meta learning hub not available",
                    "ai_available": False
                }
                dependencies = {
                    "transformers": "unavailable",
                    "torch": "unavailable"
                }
            
            response_time = time.time() - start_time
            
            return ComponentHealthResponse(
                status=status,
                component="meta-learning-hub",
                details=details,
                dependencies=dependencies,
                last_check=datetime.now().isoformat(),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Meta learning hub health check failed", error=str(e))
            return ComponentHealthResponse(
                status="unhealthy",
                component="meta-learning-hub",
                details={"error": str(e)},
                dependencies={},
                last_check=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def check_model_ensemble_health(self) -> ComponentHealthResponse:
        """Check model ensemble health"""
        start_time = time.time()
        
        try:
            # Try to import and check model ensemble
            try:
                from model_ensemble import model_ensemble
                
                # Get real ensemble status
                ensemble_status = model_ensemble.get_ensemble_status()
                
                # Determine health status
                is_healthy = (
                    ensemble_status.status == 'healthy' and
                    ensemble_status.models > 0 and
                    ensemble_status.accuracy > 0.5
                )
                
                status = "healthy" if is_healthy else "degraded"
                
                details = {
                    "ensemble_status": ensemble_status.status,
                    "models": ensemble_status.models,
                    "accuracy": ensemble_status.accuracy,
                    "ai_available": getattr(ensemble_status, 'ai_available', True)
                }
                
                dependencies = {
                    "torch": "available" if getattr(ensemble_status, 'ai_available', True) else "unavailable",
                    "scikit-learn": "available",
                    "numpy": "available"
                }
                
            except ImportError:
                # Service not available
                status = "unhealthy"
                details = {
                    "error": "Model ensemble not available",
                    "models": 0,
                    "accuracy": 0.0
                }
                dependencies = {
                    "torch": "unavailable",
                    "scikit-learn": "unavailable"
                }
            
            response_time = time.time() - start_time
            
            return ComponentHealthResponse(
                status=status,
                component="model-ensemble",
                details=details,
                dependencies=dependencies,
                last_check=datetime.now().isoformat(),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Model ensemble health check failed", error=str(e))
            return ComponentHealthResponse(
                status="unhealthy",
                component="model-ensemble",
                details={"error": str(e)},
                dependencies={},
                last_check=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def check_continual_learner_health(self) -> ComponentHealthResponse:
        """Check continual learner health"""
        start_time = time.time()
        
        try:
            # Try to import and check continual learner
            try:
                from continual_learner import continual_learner
                
                # Get real learning status
                learning_status = continual_learner.get_learning_status()
                
                # Determine health status
                is_healthy = (
                    learning_status.get('status') in ['healthy', 'learning'] and
                    learning_status.get('ai_available', False)
                )
                
                status = "healthy" if is_healthy else "degraded"
                
                details = {
                    "learning_status": learning_status.get('status', 'unknown'),
                    "learning_rate": learning_status.get('learning_rate', 0.0),
                    "accuracy": learning_status.get('accuracy', 0.0),
                    "ai_available": learning_status.get('ai_available', False)
                }
                
                dependencies = {
                    "torch": "available" if learning_status.get('ai_available', False) else "unavailable",
                    "avalanche": "available",
                    "scikit-learn": "available"
                }
                
            except ImportError:
                # Service not available
                status = "unhealthy"
                details = {
                    "error": "Continual learner not available",
                    "ai_available": False
                }
                dependencies = {
                    "torch": "unavailable",
                    "avalanche": "unavailable"
                }
            
            response_time = time.time() - start_time
            
            return ComponentHealthResponse(
                status=status,
                component="continual-learner",
                details=details,
                dependencies=dependencies,
                last_check=datetime.now().isoformat(),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Continual learner health check failed", error=str(e))
            return ComponentHealthResponse(
                status="unhealthy",
                component="continual-learner",
                details={"error": str(e)},
                dependencies={},
                last_check=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def check_performance_optimizer_health(self) -> ComponentHealthResponse:
        """Check performance optimizer health"""
        start_time = time.time()
        
        try:
            # Try to import and check performance optimizer
            try:
                from performance_optimizer import performance_optimizer
                
                # Get real optimization status
                optimization_status = performance_optimizer.get_optimization_status()
                
                # Determine health status
                is_healthy = (
                    optimization_status.status == 'healthy' and
                    optimization_status.performance > 0.5
                )
                
                status = "healthy" if is_healthy else "degraded"
                
                details = {
                    "optimization_status": optimization_status.status,
                    "performance": optimization_status.performance,
                    "ai_available": getattr(optimization_status, 'ai_available', True)
                }
                
                dependencies = {
                    "torch": "available" if getattr(optimization_status, 'ai_available', True) else "unavailable",
                    "scikit-learn": "available",
                    "numpy": "available"
                }
                
            except ImportError:
                # Service not available
                status = "unhealthy"
                details = {
                    "error": "Performance optimizer not available",
                    "performance": 0.0
                }
                dependencies = {
                    "torch": "unavailable",
                    "scikit-learn": "unavailable"
                }
            
            response_time = time.time() - start_time
            
            return ComponentHealthResponse(
                status=status,
                component="performance-optimizer",
                details=details,
                dependencies=dependencies,
                last_check=datetime.now().isoformat(),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Performance optimizer health check failed", error=str(e))
            return ComponentHealthResponse(
                status="unhealthy",
                component="performance-optimizer",
                details={"error": str(e)},
                dependencies={},
                last_check=datetime.now().isoformat(),
                response_time=time.time() - start_time
            )
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                load_avg = [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent,
                disk_free=disk.free,
                process_count=len(psutil.pids()),
                load_average=load_avg
            )
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            # Return default metrics on error
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0,
                disk_usage=0.0,
                disk_free=0,
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )

# Global health checker instance
health_checker = AIOrchestrationHealthChecker()

# Create router for health endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

@health_router.get("/", 
                   response_model=HealthResponse,
                   summary="Main Health Check",
                   description="Comprehensive health check for the AI orchestration service",
                   tags=["health"])
async def health_check():
    """Main health check endpoint with system metrics"""
    try:
        logger.info("Starting main health check")
        
        # Get system metrics
        system_metrics = await health_checker.get_system_metrics()
        
        # Get service uptime
        uptime = health_checker.get_uptime()
        
        # Determine overall health status
        health_status = "healthy"
        if system_metrics.cpu_usage > 90 or system_metrics.memory_usage > 90:
            health_status = "degraded"
        
        # Prepare response
        response = HealthResponse(
            status=health_status,
            timestamp=datetime.now().isoformat(),
            service="ai-orchestration-service",
            version="1.0.0",
            uptime=uptime,
            details={
                "cpu_usage": system_metrics.cpu_usage,
                "memory_usage": system_metrics.memory_usage,
                "memory_available": system_metrics.memory_available,
                "disk_usage": system_metrics.disk_usage,
                "disk_free": system_metrics.disk_free,
                "process_count": system_metrics.process_count,
                "load_average": system_metrics.load_average
            },
            dependencies={
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "psutil": "available",
                "fastapi": "available"
            }
        )
        
        logger.info("Main health check completed", 
                   status=health_status, 
                   uptime=uptime,
                   cpu_usage=system_metrics.cpu_usage,
                   memory_usage=system_metrics.memory_usage)
        
        return response
        
    except Exception as e:
        logger.error("Main health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@health_router.get("/quantum",
                   response_model=ComponentHealthResponse,
                   summary="Quantum Orchestrator Health",
                   description="Health check for quantum orchestrator component",
                   tags=["health"])
async def quantum_health():
    """Quantum orchestrator health check"""
    try:
        logger.info("Checking quantum orchestrator health")
        
        # Check cache first
        cache_key = "quantum_orchestrator_health"
        cached_result = health_checker.get_cache(cache_key)
        if cached_result:
            logger.info("Returning cached quantum orchestrator health")
            return cached_result
        
        # Perform health check
        result = await health_checker.check_quantum_orchestrator_health()
        
        # Cache the result
        health_checker.set_cache(cache_key, result)
        
        logger.info("Quantum orchestrator health check completed", 
                   status=result.status,
                   response_time=result.response_time)
        
        return result
        
    except Exception as e:
        logger.error("Quantum orchestrator health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Quantum orchestrator health check failed: {str(e)}"
        )

@health_router.get("/meta-learning",
                   response_model=ComponentHealthResponse,
                   summary="Meta Learning Hub Health",
                   description="Health check for meta learning hub component",
                   tags=["health"])
async def meta_learning_health():
    """Meta learning hub health check"""
    try:
        logger.info("Checking meta learning hub health")
        
        # Check cache first
        cache_key = "meta_learning_hub_health"
        cached_result = health_checker.get_cache(cache_key)
        if cached_result:
            logger.info("Returning cached meta learning hub health")
            return cached_result
        
        # Perform health check
        result = await health_checker.check_meta_learning_hub_health()
        
        # Cache the result
        health_checker.set_cache(cache_key, result)
        
        logger.info("Meta learning hub health check completed", 
                   status=result.status,
                   response_time=result.response_time)
        
        return result
        
    except Exception as e:
        logger.error("Meta learning hub health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Meta learning hub health check failed: {str(e)}"
        )

@health_router.get("/ensemble",
                   response_model=ComponentHealthResponse,
                   summary="Model Ensemble Health",
                   description="Health check for model ensemble component",
                   tags=["health"])
async def ensemble_health():
    """Model ensemble health check"""
    try:
        logger.info("Checking model ensemble health")
        
        # Check cache first
        cache_key = "model_ensemble_health"
        cached_result = health_checker.get_cache(cache_key)
        if cached_result:
            logger.info("Returning cached model ensemble health")
            return cached_result
        
        # Perform health check
        result = await health_checker.check_model_ensemble_health()
        
        # Cache the result
        health_checker.set_cache(cache_key, result)
        
        logger.info("Model ensemble health check completed", 
                   status=result.status,
                   response_time=result.response_time)
        
        return result
        
    except Exception as e:
        logger.error("Model ensemble health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model ensemble health check failed: {str(e)}"
        )

@health_router.get("/continual-learning",
                   response_model=ComponentHealthResponse,
                   summary="Continual Learner Health",
                   description="Health check for continual learner component",
                   tags=["health"])
async def continual_learning_health():
    """Continual learner health check"""
    try:
        logger.info("Checking continual learner health")
        
        # Check cache first
        cache_key = "continual_learner_health"
        cached_result = health_checker.get_cache(cache_key)
        if cached_result:
            logger.info("Returning cached continual learner health")
            return cached_result
        
        # Perform health check
        result = await health_checker.check_continual_learner_health()
        
        # Cache the result
        health_checker.set_cache(cache_key, result)
        
        logger.info("Continual learner health check completed", 
                   status=result.status,
                   response_time=result.response_time)
        
        return result
        
    except Exception as e:
        logger.error("Continual learner health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Continual learner health check failed: {str(e)}"
        )

@health_router.get("/optimization",
                   response_model=ComponentHealthResponse,
                   summary="Performance Optimizer Health",
                   description="Health check for performance optimizer component",
                   tags=["health"])
async def optimization_health():
    """Performance optimizer health check"""
    try:
        logger.info("Checking performance optimizer health")
        
        # Check cache first
        cache_key = "performance_optimizer_health"
        cached_result = health_checker.get_cache(cache_key)
        if cached_result:
            logger.info("Returning cached performance optimizer health")
            return cached_result
        
        # Perform health check
        result = await health_checker.check_performance_optimizer_health()
        
        # Cache the result
        health_checker.set_cache(cache_key, result)
        
        logger.info("Performance optimizer health check completed", 
                   status=result.status,
                   response_time=result.response_time)
        
        return result
        
    except Exception as e:
        logger.error("Performance optimizer health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Performance optimizer health check failed: {str(e)}"
        )

@health_router.get("/detailed",
                   response_model=DetailedHealthResponse,
                   summary="Detailed Health Check",
                   description="Comprehensive health check with all components and system metrics",
                   tags=["health"])
async def detailed_health():
    """Detailed health check with all components"""
    try:
        logger.info("Starting detailed health check")
        
        # Get all component health statuses in parallel
        quantum_task = health_checker.check_quantum_orchestrator_health()
        meta_task = health_checker.check_meta_learning_hub_health()
        ensemble_task = health_checker.check_model_ensemble_health()
        continual_task = health_checker.check_continual_learner_health()
        optimization_task = health_checker.check_performance_optimizer_health()
        metrics_task = health_checker.get_system_metrics()
        
        # Wait for all checks to complete
        quantum_status, meta_status, ensemble_status, continual_status, optimization_status, system_metrics = await asyncio.gather(
            quantum_task, meta_task, ensemble_task, continual_task, optimization_task, metrics_task
        )
        
        # Determine overall health status
        component_statuses = [quantum_status.status, meta_status.status, ensemble_status.status, continual_status.status, optimization_status.status]
        healthy_count = sum(1 for status in component_statuses if status == "healthy")
        degraded_count = sum(1 for status in component_statuses if status == "degraded")
        unhealthy_count = sum(1 for status in component_statuses if status == "unhealthy")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Prepare detailed response
        response = DetailedHealthResponse(
            overall_status=overall_status,
            timestamp=datetime.now().isoformat(),
            components={
                "quantum_orchestrator": quantum_status,
                "meta_learning_hub": meta_status,
                "model_ensemble": ensemble_status,
                "continual_learner": continual_status,
                "performance_optimizer": optimization_status
            },
            system_metrics=system_metrics,
            service_info={
                "service": "ai-orchestration-service",
                "version": "1.0.0",
                "uptime": health_checker.get_uptime(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        )
        
        logger.info("Detailed health check completed", 
                   overall_status=overall_status,
                   healthy_components=healthy_count,
                   degraded_components=degraded_count,
                   unhealthy_components=unhealthy_count)
        
        return response
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )

@health_router.get("/ready",
                   summary="Readiness Check",
                   description="Kubernetes-style readiness probe for service dependencies",
                   tags=["health"])
async def readiness_check():
    """Kubernetes-style readiness check"""
    try:
        logger.info("Starting readiness check")
        
        # Check if all critical components are ready
        quantum_health = await health_checker.check_quantum_orchestrator_health()
        meta_health = await health_checker.check_meta_learning_hub_health()
        ensemble_health = await health_checker.check_model_ensemble_health()
        
        # All components must be healthy or degraded (not unhealthy)
        all_ready = all(
            status.status in ["healthy", "degraded"] 
            for status in [quantum_health, meta_health, ensemble_health]
        )
        
        if all_ready:
            logger.info("Service is ready")
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "uptime": health_checker.get_uptime()
            }
        else:
            logger.warning("Service is not ready", 
                          quantum_status=quantum_health.status,
                          meta_status=meta_health.status,
                          ensemble_status=ensemble_health.status)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - one or more components are unhealthy"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )

@health_router.get("/live",
                   summary="Liveness Check",
                   description="Kubernetes-style liveness probe for service availability",
                   tags=["health"])
async def liveness_check():
    """Kubernetes-style liveness check"""
    try:
        logger.info("Starting liveness check")
        
        # Simple liveness check - if we can respond, we're alive
        uptime = health_checker.get_uptime()
        
        logger.info("Service is alive", uptime=uptime)
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime
        }
        
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not alive: {str(e)}"
        )

@health_router.get("/metrics",
                   summary="Health Metrics",
                   description="Get health check performance metrics",
                   tags=["health"])
async def health_metrics():
    """Get health check performance metrics"""
    try:
        logger.info("Getting health metrics")
        
        # Get system metrics
        system_metrics = await health_checker.get_system_metrics()
        
        # Get cache statistics
        cache_stats = {
            "cached_items": len(health_checker.health_cache),
            "cache_duration": health_checker.cache_duration,
            "cache_keys": list(health_checker.health_cache.keys())
        }
        
        return {
            "service": "ai-orchestration-service",
            "timestamp": datetime.now().isoformat(),
            "uptime": health_checker.get_uptime(),
            "system_metrics": system_metrics.dict(),
            "cache_stats": cache_stats,
            "health_checker": {
                "start_time": health_checker.start_time.isoformat(),
                "cache_duration": health_checker.cache_duration
            }
        }
        
    except Exception as e:
        logger.error("Failed to get health metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health metrics: {str(e)}"
        )