# AI Orchestration Service
# ========================
# Enterprise-grade AI orchestration service for ScanSmart

from .src.quantum_orchestrator import quantum_orchestrator, QuantumOrchestrator
from .src.meta_learning_hub import meta_learning_hub, MetaLearningHub
from .src.model_ensemble import model_ensemble, ModelEnsemble
from .src.continual_learner import continual_learner, ContinualLearner
from .src.performance_optimizer import performance_optimizer, PerformanceOptimizer

__version__ = "1.0.0"
__author__ = "ScanSmart AI Team"
__description__ = "Enterprise-grade AI orchestration service"

__all__ = [
    "quantum_orchestrator",
    "QuantumOrchestrator",
    "meta_learning_hub", 
    "MetaLearningHub",
    "model_ensemble",
    "ModelEnsemble",
    "continual_learner",
    "ContinualLearner",
    "performance_optimizer",
    "PerformanceOptimizer"
]




