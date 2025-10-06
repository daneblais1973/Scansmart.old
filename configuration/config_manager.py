"""
Configuration Management System
==============================
Centralized configuration management for all ScanSmart services
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import structlog

logger = structlog.get_logger()

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "scansmart"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    max_connections: int = 10
    timeout: int = 5

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    health_check_interval: int = 60
    max_retries: int = 3
    retry_delay: int = 5

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_interval: int = 30
    retention_days: int = 30
    prometheus_port: int = 9090
    grafana_port: int = 3000
    alerting_enabled: bool = True
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "response_time": 5.0
    })

@dataclass
class SecurityConfig:
    """Security configuration"""
    authentication_enabled: bool = True
    authorization_enabled: bool = True
    jwt_secret: str = ""
    jwt_expiry: int = 3600
    rate_limiting_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class AIConfig:
    """AI/ML configuration"""
    quantum_enabled: bool = True
    ai_enabled: bool = True
    model_cache_dir: str = "./cache/models"
    max_models: int = 10
    model_timeout: int = 300
    gpu_enabled: bool = False
    mixed_precision: bool = True

class ConfigManager:
    """Main configuration manager"""
    
    def __init__(self, config_dir: str = "config", environment: str = "development"):
        self.config_dir = Path(config_dir)
        self.environment = Environment(environment)
        self.config: Dict[str, Any] = {}
        self.services_config: Dict[str, ServiceConfig] = {}
        self.database_config: Optional[DatabaseConfig] = None
        self.redis_config: Optional[RedisConfig] = None
        self.monitoring_config: Optional[MonitoringConfig] = None
        self.security_config: Optional[SecurityConfig] = None
        self.ai_config: Optional[AIConfig] = None
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from files and environment"""
        try:
            # Load base configuration
            self._load_base_config()
            
            # Load environment-specific configuration
            self._load_environment_config()
            
            # Load service-specific configuration
            self._load_services_config()
            
            # Override with environment variables
            self._load_environment_overrides()
            
            logger.info(f"Configuration loaded for environment: {self.environment.value}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_base_config(self):
        """Load base configuration file"""
        base_config_file = self.config_dir / "base.yaml"
        
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            # Create default configuration
            self.config = self._get_default_config()
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self._merge_config(self.config, env_config)
    
    def _load_services_config(self):
        """Load service-specific configuration"""
        services_dir = self.config_dir / "services"
        
        if services_dir.exists():
            for service_file in services_dir.glob("*.yaml"):
                service_name = service_file.stem
                
                with open(service_file, 'r') as f:
                    service_config = yaml.safe_load(f) or {}
                    
                    # Create ServiceConfig object
                    self.services_config[service_name] = ServiceConfig(
                        name=service_name,
                        **service_config
                    )
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # Database overrides
        if os.getenv("DATABASE_HOST"):
            self.config.setdefault("database", {})["host"] = os.getenv("DATABASE_HOST")
        if os.getenv("DATABASE_PORT"):
            self.config.setdefault("database", {})["port"] = int(os.getenv("DATABASE_PORT"))
        if os.getenv("DATABASE_NAME"):
            self.config.setdefault("database", {})["name"] = os.getenv("DATABASE_NAME")
        
        # Redis overrides
        if os.getenv("REDIS_HOST"):
            self.config.setdefault("redis", {})["host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.config.setdefault("redis", {})["port"] = int(os.getenv("REDIS_PORT"))
        
        # Monitoring overrides
        if os.getenv("MONITORING_ENABLED"):
            self.config.setdefault("monitoring", {})["enabled"] = os.getenv("MONITORING_ENABLED").lower() == "true"
        
        # Security overrides
        if os.getenv("JWT_SECRET"):
            self.config.setdefault("security", {})["jwt_secret"] = os.getenv("JWT_SECRET")
        
        # AI overrides
        if os.getenv("AI_ENABLED"):
            self.config.setdefault("ai", {})["enabled"] = os.getenv("AI_ENABLED").lower() == "true"
        if os.getenv("QUANTUM_ENABLED"):
            self.config.setdefault("ai", {})["quantum_enabled"] = os.getenv("QUANTUM_ENABLED").lower() == "true"
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "environment": self.environment.value,
            "debug": self.environment == Environment.DEVELOPMENT,
            "log_level": "INFO",
            "database": {
                "type": "sqlite",
                "host": "localhost",
                "port": 5432,
                "name": "scansmart",
                "username": "",
                "password": "",
                "pool_size": 10,
                "max_overflow": 20,
                "echo": False
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": "",
                "max_connections": 10,
                "timeout": 5
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 30,
                "retention_days": 30,
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "alerting_enabled": True,
                "thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "disk_usage": 90.0,
                    "response_time": 5.0
                }
            },
            "security": {
                "authentication_enabled": True,
                "authorization_enabled": True,
                "jwt_secret": "",
                "jwt_expiry": 3600,
                "rate_limiting_enabled": True,
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "cors_enabled": True,
                "cors_origins": ["*"]
            },
            "ai": {
                "quantum_enabled": True,
                "ai_enabled": True,
                "model_cache_dir": "./cache/models",
                "max_models": 10,
                "model_timeout": 300,
                "gpu_enabled": False,
                "mixed_precision": True
            }
        }
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        if key is None:
            return self.config
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get service configuration"""
        return self.services_config.get(service_name)
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        if self.database_config is None:
            db_config = self.get_config("database", {})
            self.database_config = DatabaseConfig(**db_config)
        return self.database_config
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        if self.redis_config is None:
            redis_config = self.get_config("redis", {})
            self.redis_config = RedisConfig(**redis_config)
        return self.redis_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        if self.monitoring_config is None:
            monitoring_config = self.get_config("monitoring", {})
            self.monitoring_config = MonitoringConfig(**monitoring_config)
        return self.monitoring_config
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        if self.security_config is None:
            security_config = self.get_config("security", {})
            self.security_config = SecurityConfig(**security_config)
        return self.security_config
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration"""
        if self.ai_config is None:
            ai_config = self.get_config("ai", {})
            self.ai_config = AIConfig(**ai_config)
        return self.ai_config
    
    def save_config(self, filename: str = None):
        """Save configuration to file"""
        if filename is None:
            filename = f"{self.environment.value}.yaml"
        
        config_file = self.config_dir / filename
        
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate required fields
        required_fields = [
            "database.host",
            "database.port",
            "redis.host",
            "redis.port"
        ]
        
        for field in required_fields:
            if self.get_config(field) is None:
                errors.append(f"Required field missing: {field}")
        
        # Validate service configurations
        for service_name, service_config in self.services_config.items():
            if not service_config.host:
                errors.append(f"Service {service_name} missing host")
            if not service_config.port:
                errors.append(f"Service {service_name} missing port")
        
        # Validate security configuration
        security_config = self.get_security_config()
        if security_config.authentication_enabled and not security_config.jwt_secret:
            errors.append("JWT secret required when authentication is enabled")
        
        return errors
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for deployment"""
        env_vars = {}
        
        # Database environment variables
        db_config = self.get_database_config()
        env_vars.update({
            "DATABASE_HOST": db_config.host,
            "DATABASE_PORT": str(db_config.port),
            "DATABASE_NAME": db_config.name,
            "DATABASE_USERNAME": db_config.username,
            "DATABASE_PASSWORD": db_config.password
        })
        
        # Redis environment variables
        redis_config = self.get_redis_config()
        env_vars.update({
            "REDIS_HOST": redis_config.host,
            "REDIS_PORT": str(redis_config.port),
            "REDIS_DB": str(redis_config.db),
            "REDIS_PASSWORD": redis_config.password
        })
        
        # Monitoring environment variables
        monitoring_config = self.get_monitoring_config()
        env_vars.update({
            "MONITORING_ENABLED": str(monitoring_config.enabled).lower(),
            "PROMETHEUS_PORT": str(monitoring_config.prometheus_port),
            "GRAFANA_PORT": str(monitoring_config.grafana_port)
        })
        
        # Security environment variables
        security_config = self.get_security_config()
        env_vars.update({
            "JWT_SECRET": security_config.jwt_secret,
            "JWT_EXPIRY": str(security_config.jwt_expiry),
            "RATE_LIMITING_ENABLED": str(security_config.rate_limiting_enabled).lower()
        })
        
        # AI environment variables
        ai_config = self.get_ai_config()
        env_vars.update({
            "AI_ENABLED": str(ai_config.ai_enabled).lower(),
            "QUANTUM_ENABLED": str(ai_config.quantum_enabled).lower(),
            "GPU_ENABLED": str(ai_config.gpu_enabled).lower()
        })
        
        return env_vars

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get_config(key, default)

def get_service_config(service_name: str) -> Optional[ServiceConfig]:
    """Get service configuration"""
    return config_manager.get_service_config(service_name)

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config_manager.get_database_config()

def get_redis_config() -> RedisConfig:
    """Get Redis configuration"""
    return config_manager.get_redis_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_manager.get_monitoring_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config_manager.get_security_config()

def get_ai_config() -> AIConfig:
    """Get AI configuration"""
    return config_manager.get_ai_config()



