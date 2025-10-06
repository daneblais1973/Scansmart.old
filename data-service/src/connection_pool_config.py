"""
Production Configuration for Advanced Connection Pool
====================================================
Enterprise-grade configuration management for advanced connection pooling.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import timedelta

class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    host: str = Field(..., description="Database host", min_length=1)
    port: int = Field(5432, description="Database port", ge=1, le=65535)
    database: str = Field(..., description="Database name", min_length=1)
    username: str = Field(..., description="Database username", min_length=1)
    password: str = Field(..., description="Database password", min_length=1)
    ssl_mode: str = Field("prefer", description="SSL mode", pattern=r'^(disable|allow|prefer|require|verify-ca|verify-full)$')
    application_name: str = Field("ScanSmart-ConnectionPool", description="Application name for database connections")

class RedisConfig(BaseModel):
    """Redis connection configuration"""
    host: str = Field(..., description="Redis host", min_length=1)
    port: int = Field(6379, description="Redis port", ge=1, le=65535)
    database: int = Field(0, description="Redis database number", ge=0, le=15)
    password: Optional[str] = Field(None, description="Redis password")
    ssl: bool = Field(False, description="Enable SSL for Redis connections")
    decode_responses: bool = Field(True, description="Decode Redis responses")

class MongoDBConfig(BaseModel):
    """MongoDB connection configuration"""
    host: str = Field(..., description="MongoDB host", min_length=1)
    port: int = Field(27017, description="MongoDB port", ge=1, le=65535)
    database: str = Field(..., description="MongoDB database name", min_length=1)
    username: Optional[str] = Field(None, description="MongoDB username")
    password: Optional[str] = Field(None, description="MongoDB password")
    auth_source: str = Field("admin", description="Authentication database")
    ssl: bool = Field(False, description="Enable SSL for MongoDB connections")

class HTTPConfig(BaseModel):
    """HTTP connection configuration"""
    base_url: str = Field(..., description="Base URL for HTTP connections", min_length=1)
    timeout: float = Field(30.0, description="HTTP timeout in seconds", ge=1.0, le=300.0)
    max_redirects: int = Field(10, description="Maximum redirects", ge=0, le=50)
    user_agent: str = Field("ScanSmart-ConnectionPool/1.0", description="User agent string")
    headers: Dict[str, str] = Field(default_factory=dict, description="Default headers")

class PoolConfig(BaseModel):
    """Connection pool configuration"""
    max_connections: int = Field(10, description="Maximum connections in pool", ge=1, le=100)
    min_connections: int = Field(2, description="Minimum connections in pool", ge=1, le=50)
    connection_timeout: float = Field(30.0, description="Connection timeout in seconds", ge=1.0, le=300.0)
    idle_timeout: float = Field(300.0, description="Idle timeout in seconds", ge=60.0, le=3600.0)
    retry_attempts: int = Field(3, description="Number of retry attempts", ge=0, le=10)
    retry_delay: float = Field(1.0, description="Delay between retries in seconds", ge=0.1, le=60.0)
    health_check_interval: float = Field(60.0, description="Health check interval in seconds", ge=10.0, le=3600.0)

class BatchConfig(BaseModel):
    """Batch processing configuration"""
    batch_size: int = Field(100, description="Batch size for operations", ge=1, le=1000)
    batch_timeout: float = Field(1.0, description="Batch timeout in seconds", ge=0.1, le=60.0)
    max_batch_operations: int = Field(1000, description="Maximum batch operations", ge=10, le=10000)
    parallel_workers: int = Field(4, description="Number of parallel workers", ge=1, le=16)

class PerformanceConfig(BaseModel):
    """Performance monitoring configuration"""
    metrics_enabled: bool = Field(True, description="Enable performance metrics")
    response_time_window: int = Field(1000, description="Response time window size", ge=100, le=10000)
    error_rate_window: int = Field(1000, description="Error rate window size", ge=100, le=10000)
    memory_limit_mb: int = Field(1024, description="Memory limit in MB", ge=256, le=8192)
    cpu_limit_percent: float = Field(80.0, description="CPU usage limit", ge=10.0, le=100.0)

class SecurityConfig(BaseModel):
    """Security configuration"""
    encrypt_passwords: bool = Field(True, description="Encrypt passwords in logs")
    mask_sensitive_data: bool = Field(True, description="Mask sensitive data in logs")
    connection_encryption: bool = Field(True, description="Enable connection encryption")
    certificate_validation: bool = Field(True, description="Validate SSL certificates")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    structured: bool = Field(True, description="Enable structured logging")
    include_metrics: bool = Field(True, description="Include metrics in logs")
    log_connections: bool = Field(False, description="Log connection details (security sensitive)")
    log_operations: bool = Field(False, description="Log operation details (security sensitive)")

class AdvancedConnectionPoolConfig(BaseModel):
    """Main advanced connection pool configuration"""
    database: DatabaseConfig = Field(..., description="Database configuration")
    redis: Optional[RedisConfig] = Field(None, description="Redis configuration")
    mongodb: Optional[MongoDBConfig] = Field(None, description="MongoDB configuration")
    http: Optional[HTTPConfig] = Field(None, description="HTTP configuration")
    pool: PoolConfig = Field(default_factory=PoolConfig, description="Pool configuration")
    batch: BatchConfig = Field(default_factory=BatchConfig, description="Batch configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    @validator('database')
    def validate_database_config(cls, v):
        """Validate database configuration"""
        if not v.host or not v.database or not v.username or not v.password:
            raise ValueError("Database host, database, username, and password are required")
        return v
    
    @validator('pool')
    def validate_pool_config(cls, v):
        """Validate pool configuration"""
        if v.min_connections > v.max_connections:
            raise ValueError("Minimum connections cannot be greater than maximum connections")
        if v.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        return v
    
    @validator('batch')
    def validate_batch_config(cls, v):
        """Validate batch configuration"""
        if v.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if v.batch_timeout <= 0:
            raise ValueError("Batch timeout must be positive")
        return v

def load_connection_pool_config() -> AdvancedConnectionPoolConfig:
    """Load connection pool configuration from environment variables"""
    return AdvancedConnectionPoolConfig(
        database=DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "scansmart"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password"),
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer"),
            application_name=os.getenv("DB_APPLICATION_NAME", "ScanSmart-ConnectionPool")
        ),
        redis=RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            decode_responses=os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true"
        ) if os.getenv("REDIS_ENABLED", "false").lower() == "true" else None,
        mongodb=MongoDBConfig(
            host=os.getenv("MONGODB_HOST", "localhost"),
            port=int(os.getenv("MONGODB_PORT", "27017")),
            database=os.getenv("MONGODB_DB", "scansmart"),
            username=os.getenv("MONGODB_USER"),
            password=os.getenv("MONGODB_PASSWORD"),
            auth_source=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
            ssl=os.getenv("MONGODB_SSL", "false").lower() == "true"
        ) if os.getenv("MONGODB_ENABLED", "false").lower() == "true" else None,
        http=HTTPConfig(
            base_url=os.getenv("HTTP_BASE_URL", "http://localhost:8000"),
            timeout=float(os.getenv("HTTP_TIMEOUT", "30.0")),
            max_redirects=int(os.getenv("HTTP_MAX_REDIRECTS", "10")),
            user_agent=os.getenv("HTTP_USER_AGENT", "ScanSmart-ConnectionPool/1.0"),
            headers={k: v for k, v in [pair.split(':', 1) for pair in os.getenv("HTTP_HEADERS", "").split(',') if ':' in pair]}
        ) if os.getenv("HTTP_ENABLED", "false").lower() == "true" else None,
        pool=PoolConfig(
            max_connections=int(os.getenv("POOL_MAX_CONNECTIONS", "10")),
            min_connections=int(os.getenv("POOL_MIN_CONNECTIONS", "2")),
            connection_timeout=float(os.getenv("POOL_CONNECTION_TIMEOUT", "30.0")),
            idle_timeout=float(os.getenv("POOL_IDLE_TIMEOUT", "300.0")),
            retry_attempts=int(os.getenv("POOL_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("POOL_RETRY_DELAY", "1.0")),
            health_check_interval=float(os.getenv("POOL_HEALTH_CHECK_INTERVAL", "60.0"))
        ),
        batch=BatchConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            batch_timeout=float(os.getenv("BATCH_TIMEOUT", "1.0")),
            max_batch_operations=int(os.getenv("BATCH_MAX_OPERATIONS", "1000")),
            parallel_workers=int(os.getenv("BATCH_PARALLEL_WORKERS", "4"))
        ),
        performance=PerformanceConfig(
            metrics_enabled=os.getenv("PERFORMANCE_METRICS_ENABLED", "true").lower() == "true",
            response_time_window=int(os.getenv("PERFORMANCE_RESPONSE_TIME_WINDOW", "1000")),
            error_rate_window=int(os.getenv("PERFORMANCE_ERROR_RATE_WINDOW", "1000")),
            memory_limit_mb=int(os.getenv("PERFORMANCE_MEMORY_LIMIT_MB", "1024")),
            cpu_limit_percent=float(os.getenv("PERFORMANCE_CPU_LIMIT_PERCENT", "80.0"))
        ),
        security=SecurityConfig(
            encrypt_passwords=os.getenv("SECURITY_ENCRYPT_PASSWORDS", "true").lower() == "true",
            mask_sensitive_data=os.getenv("SECURITY_MASK_SENSITIVE_DATA", "true").lower() == "true",
            connection_encryption=os.getenv("SECURITY_CONNECTION_ENCRYPTION", "true").lower() == "true",
            certificate_validation=os.getenv("SECURITY_CERTIFICATE_VALIDATION", "true").lower() == "true"
        ),
        logging=LoggingConfig(
            level=os.getenv("CONNECTION_POOL_LOG_LEVEL", "INFO"),
            structured=os.getenv("CONNECTION_POOL_STRUCTURED_LOGGING", "true").lower() == "true",
            include_metrics=os.getenv("CONNECTION_POOL_INCLUDE_METRICS", "true").lower() == "true",
            log_connections=os.getenv("CONNECTION_POOL_LOG_CONNECTIONS", "false").lower() == "true",
            log_operations=os.getenv("CONNECTION_POOL_LOG_OPERATIONS", "false").lower() == "true"
        )
    )

def validate_connection_pool_config(config: AdvancedConnectionPoolConfig) -> bool:
    """Validate connection pool configuration settings"""
    try:
        # Validate database configuration
        if not config.database.host or not config.database.database:
            print("Error: Database host and database name are required")
            return False
        
        if not config.database.username or not config.database.password:
            print("Error: Database username and password are required")
            return False
        
        # Validate pool configuration
        if config.pool.min_connections > config.pool.max_connections:
            print("Error: Minimum connections cannot be greater than maximum connections")
            return False
        
        if config.pool.connection_timeout <= 0:
            print("Error: Connection timeout must be positive")
            return False
        
        # Validate batch configuration
        if config.batch.batch_size <= 0:
            print("Error: Batch size must be positive")
            return False
        
        if config.batch.batch_timeout <= 0:
            print("Error: Batch timeout must be positive")
            return False
        
        # Validate performance configuration
        if config.performance.memory_limit_mb <= 0:
            print("Error: Memory limit must be positive")
            return False
        
        if config.performance.cpu_limit_percent <= 0 or config.performance.cpu_limit_percent > 100:
            print("Error: CPU limit must be between 0 and 100")
            return False
        
        # Validate security configuration
        if config.security.log_connections:
            print("Warning: Connection logging is enabled - may contain sensitive data")
        
        if config.security.log_operations:
            print("Warning: Operation logging is enabled - may contain sensitive data")
        
        return True
        
    except Exception as e:
        print(f"Connection pool configuration validation error: {e}")
        return False

# Global configuration instance
connection_pool_config = load_connection_pool_config()

def get_connection_pool_config() -> AdvancedConnectionPoolConfig:
    """Get the global connection pool configuration instance"""
    return connection_pool_config

def reload_connection_pool_config() -> AdvancedConnectionPoolConfig:
    """Reload connection pool configuration from environment"""
    global connection_pool_config
    connection_pool_config = load_connection_pool_config()
    return connection_pool_config


