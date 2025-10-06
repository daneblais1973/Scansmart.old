"""
Production Configuration for AI Orchestration Service Health Endpoints
==================================================================
Enterprise-grade configuration management for AI orchestration health monitoring.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import timedelta

class HealthCheckConfig(BaseModel):
    """Health check configuration model"""
    cache_duration: int = Field(30, description="Health check cache duration in seconds", ge=1, le=300)
    timeout: int = Field(10, description="Health check timeout in seconds", ge=1, le=60)
    retry_count: int = Field(3, description="Number of retries for failed health checks", ge=0, le=10)
    retry_delay: float = Field(1.0, description="Delay between retries in seconds", ge=0.1, le=10.0)

class SystemMetricsConfig(BaseModel):
    """System metrics configuration"""
    enabled: bool = Field(True, description="Enable system metrics collection")
    cpu_threshold: float = Field(90.0, description="CPU usage threshold for degraded status", ge=0, le=100)
    memory_threshold: float = Field(90.0, description="Memory usage threshold for degraded status", ge=0, le=100)
    disk_threshold: float = Field(90.0, description="Disk usage threshold for degraded status", ge=0, le=100)
    collection_interval: int = Field(5, description="Metrics collection interval in seconds", ge=1, le=60)

class ComponentConfig(BaseModel):
    """Component health check configuration"""
    quantum_orchestrator: bool = Field(True, description="Enable quantum orchestrator health checks")
    meta_learning_hub: bool = Field(True, description="Enable meta learning hub health checks")
    model_ensemble: bool = Field(True, description="Enable model ensemble health checks")
    continual_learner: bool = Field(True, description="Enable continual learner health checks")
    performance_optimizer: bool = Field(True, description="Enable performance optimizer health checks")
    parallel_checks: bool = Field(True, description="Enable parallel health checks")

class LoggingConfig(BaseModel):
    """Logging configuration for health endpoints"""
    level: str = Field("INFO", description="Logging level", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    structured: bool = Field(True, description="Enable structured logging")
    include_metrics: bool = Field(True, description="Include metrics in logs")
    sensitive_data: bool = Field(False, description="Log sensitive data (not recommended for production)")

class AlertingConfig(BaseModel):
    """Alerting configuration"""
    enabled: bool = Field(False, description="Enable health check alerting")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for alerts")
    alert_threshold: int = Field(3, description="Number of consecutive failures before alerting", ge=1, le=10)
    alert_cooldown: int = Field(300, description="Alert cooldown period in seconds", ge=60, le=3600)

class AIOrchestrationHealthConfig(BaseModel):
    """Main AI orchestration health configuration"""
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    system_metrics: SystemMetricsConfig = Field(default_factory=SystemMetricsConfig)
    components: ComponentConfig = Field(default_factory=ComponentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    
    @validator('health_check')
    def validate_health_check(cls, v):
        """Validate health check configuration"""
        if v.cache_duration <= 0:
            raise ValueError("Cache duration must be positive")
        if v.timeout <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @validator('system_metrics')
    def validate_system_metrics(cls, v):
        """Validate system metrics configuration"""
        if v.cpu_threshold < 0 or v.cpu_threshold > 100:
            raise ValueError("CPU threshold must be between 0 and 100")
        if v.memory_threshold < 0 or v.memory_threshold > 100:
            raise ValueError("Memory threshold must be between 0 and 100")
        if v.disk_threshold < 0 or v.disk_threshold > 100:
            raise ValueError("Disk threshold must be between 0 and 100")
        return v

def load_ai_orchestration_health_config() -> AIOrchestrationHealthConfig:
    """Load AI orchestration health configuration from environment variables"""
    return AIOrchestrationHealthConfig(
        health_check=HealthCheckConfig(
            cache_duration=int(os.getenv("AI_HEALTH_CACHE_DURATION", "30")),
            timeout=int(os.getenv("AI_HEALTH_TIMEOUT", "10")),
            retry_count=int(os.getenv("AI_HEALTH_RETRY_COUNT", "3")),
            retry_delay=float(os.getenv("AI_HEALTH_RETRY_DELAY", "1.0"))
        ),
        system_metrics=SystemMetricsConfig(
            enabled=os.getenv("AI_SYSTEM_METRICS_ENABLED", "true").lower() == "true",
            cpu_threshold=float(os.getenv("AI_CPU_THRESHOLD", "90.0")),
            memory_threshold=float(os.getenv("AI_MEMORY_THRESHOLD", "90.0")),
            disk_threshold=float(os.getenv("AI_DISK_THRESHOLD", "90.0")),
            collection_interval=int(os.getenv("AI_METRICS_COLLECTION_INTERVAL", "5"))
        ),
        components=ComponentConfig(
            quantum_orchestrator=os.getenv("QUANTUM_ORCHESTRATOR_HEALTH", "true").lower() == "true",
            meta_learning_hub=os.getenv("META_LEARNING_HUB_HEALTH", "true").lower() == "true",
            model_ensemble=os.getenv("MODEL_ENSEMBLE_HEALTH", "true").lower() == "true",
            continual_learner=os.getenv("CONTINUAL_LEARNER_HEALTH", "true").lower() == "true",
            performance_optimizer=os.getenv("PERFORMANCE_OPTIMIZER_HEALTH", "true").lower() == "true",
            parallel_checks=os.getenv("AI_PARALLEL_HEALTH_CHECKS", "true").lower() == "true"
        ),
        logging=LoggingConfig(
            level=os.getenv("AI_HEALTH_LOG_LEVEL", "INFO"),
            structured=os.getenv("AI_HEALTH_STRUCTURED_LOGGING", "true").lower() == "true",
            include_metrics=os.getenv("AI_HEALTH_INCLUDE_METRICS", "true").lower() == "true",
            sensitive_data=os.getenv("AI_HEALTH_LOG_SENSITIVE_DATA", "false").lower() == "true"
        ),
        alerting=AlertingConfig(
            enabled=os.getenv("AI_HEALTH_ALERTING_ENABLED", "false").lower() == "true",
            webhook_url=os.getenv("AI_HEALTH_WEBHOOK_URL"),
            alert_threshold=int(os.getenv("AI_HEALTH_ALERT_THRESHOLD", "3")),
            alert_cooldown=int(os.getenv("AI_HEALTH_ALERT_COOLDOWN", "300"))
        )
    )

def validate_ai_orchestration_health_config(config: AIOrchestrationHealthConfig) -> bool:
    """Validate AI orchestration health configuration settings"""
    try:
        # Validate health check settings
        if config.health_check.cache_duration <= 0:
            print("Error: Health check cache duration must be positive")
            return False
        
        if config.health_check.timeout <= 0:
            print("Error: Health check timeout must be positive")
            return False
        
        # Validate system metrics
        if config.system_metrics.cpu_threshold < 0 or config.system_metrics.cpu_threshold > 100:
            print("Error: CPU threshold must be between 0 and 100")
            return False
        
        if config.system_metrics.memory_threshold < 0 or config.system_metrics.memory_threshold > 100:
            print("Error: Memory threshold must be between 0 and 100")
            return False
        
        if config.system_metrics.disk_threshold < 0 or config.system_metrics.disk_threshold > 100:
            print("Error: Disk threshold must be between 0 and 100")
            return False
        
        # Validate alerting configuration
        if config.alerting.enabled and not config.alerting.webhook_url:
            print("Warning: Alerting enabled but no webhook URL configured")
        
        # Validate logging configuration
        if config.logging.sensitive_data:
            print("Warning: Sensitive data logging is enabled - not recommended for production")
        
        # Validate component configuration
        if not any([
            config.components.quantum_orchestrator,
            config.components.meta_learning_hub,
            config.components.model_ensemble,
            config.components.continual_learner,
            config.components.performance_optimizer
        ]):
            print("Warning: No health check components are enabled")
        
        return True
        
    except Exception as e:
        print(f"AI orchestration health configuration validation error: {e}")
        return False

# Global configuration instance
ai_orchestration_health_config = load_ai_orchestration_health_config()

def get_ai_orchestration_health_config() -> AIOrchestrationHealthConfig:
    """Get the global AI orchestration health configuration instance"""
    return ai_orchestration_health_config

def reload_ai_orchestration_health_config() -> AIOrchestrationHealthConfig:
    """Reload AI orchestration health configuration from environment"""
    global ai_orchestration_health_config
    ai_orchestration_health_config = load_ai_orchestration_health_config()
    return ai_orchestration_health_config


