"""
Catalyst Service Configuration
=============================
Configuration settings for catalyst service components
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    enabled: bool = True
    backend: str = "qasm_simulator"
    optimization_level: int = 3
    max_shots: int = 1024
    noise_model: Optional[str] = None
    quantum_advantage_threshold: float = 0.1

@dataclass
class AIConfig:
    """AI/ML configuration"""
    enabled: bool = True
    model_cache_size: int = 1000
    batch_size: int = 32
    max_sequence_length: int = 512
    confidence_threshold: float = 0.7
    fallback_enabled: bool = True

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    max_workers: int = 8
    queue_size: int = 1000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    parallel_processing: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_enabled: bool = True
    log_level: str = "INFO"
    performance_tracking: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'confidence_low': 0.3,
                'processing_time_high': 5.0,
                'error_rate_high': 0.1
            }

@dataclass
class DatabaseConfig:
    """Database configuration"""
    enabled: bool = True
    connection_string: str = "sqlite:///catalyst_service.db"
    max_connections: int = 10
    query_timeout: int = 30
    backup_enabled: bool = True

@dataclass
class CatalystServiceConfig:
    """Main catalyst service configuration"""
    quantum: QuantumConfig
    ai: AIConfig
    processing: ProcessingConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig
    
    # Service-specific settings
    detection_patterns: Dict[str, Any] = None
    ensemble_weights: Dict[str, float] = None
    real_time_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.detection_patterns is None:
            self.detection_patterns = {
                'earnings': {'confidence_threshold': 0.7, 'impact_weight': 0.9},
                'merger': {'confidence_threshold': 0.8, 'impact_weight': 0.95},
                'product': {'confidence_threshold': 0.75, 'impact_weight': 0.85},
                'regulatory': {'confidence_threshold': 0.85, 'impact_weight': 0.9},
                'partnership': {'confidence_threshold': 0.7, 'impact_weight': 0.8}
            }
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'random_forest': 0.2,
                'gradient_boosting': 0.2,
                'svm': 0.15,
                'logistic_regression': 0.15,
                'naive_bayes': 0.1,
                'neural_network': 0.2
            }
        
        if self.real_time_thresholds is None:
            self.real_time_thresholds = {
                'critical_sentiment': 0.8,
                'high_entity_count': 5,
                'anomaly_detection': 0.8,
                'low_confidence': 0.3
            }

def load_config() -> CatalystServiceConfig:
    """Load configuration from environment variables and defaults"""
    
    # Quantum configuration
    quantum_config = QuantumConfig(
        enabled=os.getenv('QUANTUM_ENABLED', 'true').lower() == 'true',
        backend=os.getenv('QUANTUM_BACKEND', 'qasm_simulator'),
        optimization_level=int(os.getenv('QUANTUM_OPTIMIZATION_LEVEL', '3')),
        max_shots=int(os.getenv('QUANTUM_MAX_SHOTS', '1024')),
        noise_model=os.getenv('QUANTUM_NOISE_MODEL'),
        quantum_advantage_threshold=float(os.getenv('QUANTUM_ADVANTAGE_THRESHOLD', '0.1'))
    )
    
    # AI configuration
    ai_config = AIConfig(
        enabled=os.getenv('AI_ENABLED', 'true').lower() == 'true',
        model_cache_size=int(os.getenv('AI_MODEL_CACHE_SIZE', '1000')),
        batch_size=int(os.getenv('AI_BATCH_SIZE', '32')),
        max_sequence_length=int(os.getenv('AI_MAX_SEQUENCE_LENGTH', '512')),
        confidence_threshold=float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.7')),
        fallback_enabled=os.getenv('AI_FALLBACK_ENABLED', 'true').lower() == 'true'
    )
    
    # Processing configuration
    processing_config = ProcessingConfig(
        max_workers=int(os.getenv('PROCESSING_MAX_WORKERS', '8')),
        queue_size=int(os.getenv('PROCESSING_QUEUE_SIZE', '1000')),
        timeout_seconds=int(os.getenv('PROCESSING_TIMEOUT_SECONDS', '30')),
        retry_attempts=int(os.getenv('PROCESSING_RETRY_ATTEMPTS', '3')),
        parallel_processing=os.getenv('PROCESSING_PARALLEL', 'true').lower() == 'true'
    )
    
    # Monitoring configuration
    monitoring_config = MonitoringConfig(
        metrics_enabled=os.getenv('MONITORING_METRICS_ENABLED', 'true').lower() == 'true',
        log_level=os.getenv('MONITORING_LOG_LEVEL', 'INFO'),
        performance_tracking=os.getenv('MONITORING_PERFORMANCE_TRACKING', 'true').lower() == 'true'
    )
    
    # Database configuration
    database_config = DatabaseConfig(
        enabled=os.getenv('DATABASE_ENABLED', 'true').lower() == 'true',
        connection_string=os.getenv('DATABASE_CONNECTION_STRING', 'sqlite:///catalyst_service.db'),
        max_connections=int(os.getenv('DATABASE_MAX_CONNECTIONS', '10')),
        query_timeout=int(os.getenv('DATABASE_QUERY_TIMEOUT', '30')),
        backup_enabled=os.getenv('DATABASE_BACKUP_ENABLED', 'true').lower() == 'true'
    )
    
    return CatalystServiceConfig(
        quantum=quantum_config,
        ai=ai_config,
        processing=processing_config,
        monitoring=monitoring_config,
        database=database_config
    )

# Global configuration instance
config = load_config()




