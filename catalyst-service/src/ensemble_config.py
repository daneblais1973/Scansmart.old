"""
Production Configuration for Ensemble Classifier
================================================
Enterprise-grade configuration management for ensemble catalyst classification.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import timedelta

class ClassifierConfig(BaseModel):
    """Individual classifier configuration"""
    enabled: bool = Field(True, description="Enable this classifier type")
    weight: float = Field(0.2, description="Initial weight for ensemble", ge=0.0, le=1.0)
    max_iterations: int = Field(1000, description="Maximum training iterations", ge=1, le=10000)
    random_state: int = Field(42, description="Random state for reproducibility", ge=0)

class RandomForestConfig(ClassifierConfig):
    """Random Forest classifier configuration"""
    n_estimators: int = Field(100, description="Number of trees", ge=10, le=1000)
    max_depth: int = Field(10, description="Maximum tree depth", ge=1, le=50)
    min_samples_split: int = Field(2, description="Minimum samples to split", ge=2, le=20)
    min_samples_leaf: int = Field(1, description="Minimum samples per leaf", ge=1, le=10)
    n_jobs: int = Field(-1, description="Number of parallel jobs", ge=-1, le=16)

class SVMConfig(ClassifierConfig):
    """SVM classifier configuration"""
    kernel: str = Field("rbf", description="SVM kernel", pattern=r'^(linear|poly|rbf|sigmoid)$')
    C: float = Field(1.0, description="Regularization parameter", ge=0.001, le=1000.0)
    gamma: str = Field("scale", description="Kernel coefficient", pattern=r'^(scale|auto|float)$')
    probability: bool = Field(True, description="Enable probability estimates")

class NeuralNetworkConfig(ClassifierConfig):
    """Neural Network classifier configuration"""
    hidden_layer_sizes: tuple = Field((100, 50), description="Hidden layer sizes")
    activation: str = Field("relu", description="Activation function", pattern=r'^(identity|logistic|tanh|relu)$')
    solver: str = Field("adam", description="Solver algorithm", pattern=r'^(lbfgs|sgd|adam)$')
    alpha: float = Field(0.001, description="L2 regularization", ge=0.0001, le=1.0)
    max_iter: int = Field(500, description="Maximum iterations", ge=100, le=2000)

class EnsembleConfig(BaseModel):
    """Ensemble configuration"""
    voting_type: str = Field("hard", description="Voting type", pattern=r'^(hard|soft)$')
    weight_method: str = Field("performance_based", description="Weight calculation method", pattern=r'^(uniform|performance_based|accuracy_based)$')
    stacking_cv_folds: int = Field(5, description="Cross-validation folds for stacking", ge=2, le=10)
    adaptive_update_frequency: int = Field(100, description="Adaptive ensemble update frequency", ge=10, le=1000)

class TrainingConfig(BaseModel):
    """Training configuration"""
    validation_split: float = Field(0.2, description="Validation data split", ge=0.1, le=0.5)
    cross_validation_folds: int = Field(5, description="Cross-validation folds", ge=2, le=10)
    early_stopping_patience: int = Field(10, description="Early stopping patience", ge=1, le=50)
    batch_size: int = Field(32, description="Training batch size", ge=1, le=1000)
    learning_rate: float = Field(0.01, description="Learning rate", ge=0.001, le=1.0)

class PerformanceConfig(BaseModel):
    """Performance monitoring configuration"""
    metrics_enabled: bool = Field(True, description="Enable performance metrics")
    prediction_timeout: float = Field(30.0, description="Prediction timeout in seconds", ge=1.0, le=300.0)
    training_timeout: float = Field(300.0, description="Training timeout in seconds", ge=10.0, le=3600.0)
    memory_limit_mb: int = Field(2048, description="Memory limit in MB", ge=512, le=16384)
    cpu_limit_percent: float = Field(80.0, description="CPU usage limit", ge=10.0, le=100.0)

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    structured: bool = Field(True, description="Enable structured logging")
    include_metrics: bool = Field(True, description="Include metrics in logs")
    log_predictions: bool = Field(False, description="Log individual predictions (privacy sensitive)")

class EnsembleClassifierConfig(BaseModel):
    """Main ensemble classifier configuration"""
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    svm: SVMConfig = Field(default_factory=SVMConfig)
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @validator('random_forest')
    def validate_random_forest(cls, v):
        """Validate Random Forest configuration"""
        if v.n_estimators <= 0:
            raise ValueError("Number of estimators must be positive")
        if v.max_depth <= 0:
            raise ValueError("Maximum depth must be positive")
        return v
    
    @validator('svm')
    def validate_svm(cls, v):
        """Validate SVM configuration"""
        if v.C <= 0:
            raise ValueError("C parameter must be positive")
        return v
    
    @validator('neural_network')
    def validate_neural_network(cls, v):
        """Validate Neural Network configuration"""
        if v.alpha <= 0:
            raise ValueError("Alpha parameter must be positive")
        if v.max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
        return v

def load_ensemble_classifier_config() -> EnsembleClassifierConfig:
    """Load ensemble classifier configuration from environment variables"""
    return EnsembleClassifierConfig(
        random_forest=RandomForestConfig(
            enabled=os.getenv("RANDOM_FOREST_ENABLED", "true").lower() == "true",
            weight=float(os.getenv("RANDOM_FOREST_WEIGHT", "0.2")),
            n_estimators=int(os.getenv("RANDOM_FOREST_N_ESTIMATORS", "100")),
            max_depth=int(os.getenv("RANDOM_FOREST_MAX_DEPTH", "10")),
            min_samples_split=int(os.getenv("RANDOM_FOREST_MIN_SAMPLES_SPLIT", "2")),
            min_samples_leaf=int(os.getenv("RANDOM_FOREST_MIN_SAMPLES_LEAF", "1")),
            n_jobs=int(os.getenv("RANDOM_FOREST_N_JOBS", "-1")),
            max_iterations=int(os.getenv("RANDOM_FOREST_MAX_ITERATIONS", "1000")),
            random_state=int(os.getenv("RANDOM_FOREST_RANDOM_STATE", "42"))
        ),
        svm=SVMConfig(
            enabled=os.getenv("SVM_ENABLED", "true").lower() == "true",
            weight=float(os.getenv("SVM_WEIGHT", "0.15")),
            kernel=os.getenv("SVM_KERNEL", "rbf"),
            C=float(os.getenv("SVM_C", "1.0")),
            gamma=os.getenv("SVM_GAMMA", "scale"),
            probability=os.getenv("SVM_PROBABILITY", "true").lower() == "true",
            max_iterations=int(os.getenv("SVM_MAX_ITERATIONS", "1000")),
            random_state=int(os.getenv("SVM_RANDOM_STATE", "42"))
        ),
        neural_network=NeuralNetworkConfig(
            enabled=os.getenv("NEURAL_NETWORK_ENABLED", "true").lower() == "true",
            weight=float(os.getenv("NEURAL_NETWORK_WEIGHT", "0.2")),
            hidden_layer_sizes=tuple(map(int, os.getenv("NEURAL_NETWORK_HIDDEN_LAYERS", "100,50").split(','))),
            activation=os.getenv("NEURAL_NETWORK_ACTIVATION", "relu"),
            solver=os.getenv("NEURAL_NETWORK_SOLVER", "adam"),
            alpha=float(os.getenv("NEURAL_NETWORK_ALPHA", "0.001")),
            max_iter=int(os.getenv("NEURAL_NETWORK_MAX_ITER", "500")),
            max_iterations=int(os.getenv("NEURAL_NETWORK_MAX_ITERATIONS", "1000")),
            random_state=int(os.getenv("NEURAL_NETWORK_RANDOM_STATE", "42"))
        ),
        ensemble=EnsembleConfig(
            voting_type=os.getenv("ENSEMBLE_VOTING_TYPE", "hard"),
            weight_method=os.getenv("ENSEMBLE_WEIGHT_METHOD", "performance_based"),
            stacking_cv_folds=int(os.getenv("ENSEMBLE_STACKING_CV_FOLDS", "5")),
            adaptive_update_frequency=int(os.getenv("ENSEMBLE_ADAPTIVE_UPDATE_FREQUENCY", "100"))
        ),
        training=TrainingConfig(
            validation_split=float(os.getenv("TRAINING_VALIDATION_SPLIT", "0.2")),
            cross_validation_folds=int(os.getenv("TRAINING_CV_FOLDS", "5")),
            early_stopping_patience=int(os.getenv("TRAINING_EARLY_STOPPING_PATIENCE", "10")),
            batch_size=int(os.getenv("TRAINING_BATCH_SIZE", "32")),
            learning_rate=float(os.getenv("TRAINING_LEARNING_RATE", "0.01"))
        ),
        performance=PerformanceConfig(
            metrics_enabled=os.getenv("PERFORMANCE_METRICS_ENABLED", "true").lower() == "true",
            prediction_timeout=float(os.getenv("PERFORMANCE_PREDICTION_TIMEOUT", "30.0")),
            training_timeout=float(os.getenv("PERFORMANCE_TRAINING_TIMEOUT", "300.0")),
            memory_limit_mb=int(os.getenv("PERFORMANCE_MEMORY_LIMIT_MB", "2048")),
            cpu_limit_percent=float(os.getenv("PERFORMANCE_CPU_LIMIT_PERCENT", "80.0"))
        ),
        logging=LoggingConfig(
            level=os.getenv("ENSEMBLE_LOG_LEVEL", "INFO"),
            structured=os.getenv("ENSEMBLE_STRUCTURED_LOGGING", "true").lower() == "true",
            include_metrics=os.getenv("ENSEMBLE_INCLUDE_METRICS", "true").lower() == "true",
            log_predictions=os.getenv("ENSEMBLE_LOG_PREDICTIONS", "false").lower() == "true"
        )
    )

def validate_ensemble_classifier_config(config: EnsembleClassifierConfig) -> bool:
    """Validate ensemble classifier configuration settings"""
    try:
        # Validate classifier configurations
        if config.random_forest.n_estimators <= 0:
            print("Error: Random Forest n_estimators must be positive")
            return False
        
        if config.svm.C <= 0:
            print("Error: SVM C parameter must be positive")
            return False
        
        if config.neural_network.alpha <= 0:
            print("Error: Neural Network alpha must be positive")
            return False
        
        # Validate ensemble configuration
        if config.ensemble.stacking_cv_folds < 2:
            print("Error: Stacking CV folds must be at least 2")
            return False
        
        # Validate training configuration
        if config.training.validation_split <= 0 or config.training.validation_split >= 1:
            print("Error: Validation split must be between 0 and 1")
            return False
        
        # Validate performance configuration
        if config.performance.prediction_timeout <= 0:
            print("Error: Prediction timeout must be positive")
            return False
        
        if config.performance.memory_limit_mb <= 0:
            print("Error: Memory limit must be positive")
            return False
        
        # Validate logging configuration
        if config.logging.log_predictions:
            print("Warning: Prediction logging is enabled - may contain sensitive data")
        
        return True
        
    except Exception as e:
        print(f"Ensemble classifier configuration validation error: {e}")
        return False

# Global configuration instance
ensemble_classifier_config = load_ensemble_classifier_config()

def get_ensemble_classifier_config() -> EnsembleClassifierConfig:
    """Get the global ensemble classifier configuration instance"""
    return ensemble_classifier_config

def reload_ensemble_classifier_config() -> EnsembleClassifierConfig:
    """Reload ensemble classifier configuration from environment"""
    global ensemble_classifier_config
    ensemble_classifier_config = load_ensemble_classifier_config()
    return ensemble_classifier_config


