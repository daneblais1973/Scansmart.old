"""
AI and Quantum Domain Events
Advanced event-driven architecture with AI orchestration and quantum computing events
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from decimal import Decimal
from enum import Enum

from ..entities.quantum_catalyst import QuantumCatalyst, QuantumCatalystType, QuantumImpactLevel
from ..entities.ai_opportunity import AIOpportunity, AIOpportunityType, AIOpportunityStatus
from ..value_objects import QuantumState, UncertaintyScore, EnsembleConfidence, MetaLearningScore


class EventType(Enum):
    """AI and quantum event types"""
    # Quantum Events
    QUANTUM_STATE_CREATED = "quantum_state_created"
    QUANTUM_STATE_MEASURED = "quantum_state_measured"
    QUANTUM_STATE_ENTANGLED = "quantum_state_entangled"
    QUANTUM_STATE_DECOHERED = "quantum_state_decohered"
    QUANTUM_GATE_APPLIED = "quantum_gate_applied"
    QUANTUM_ERROR_DETECTED = "quantum_error_detected"
    QUANTUM_ERROR_CORRECTED = "quantum_error_corrected"
    
    # AI Events
    AI_MODEL_TRAINED = "ai_model_trained"
    AI_MODEL_PREDICTED = "ai_model_predicted"
    AI_MODEL_UPDATED = "ai_model_updated"
    AI_ENSEMBLE_CREATED = "ai_ensemble_created"
    AI_ENSEMBLE_VOTED = "ai_ensemble_voted"
    AI_UNCERTAINTY_QUANTIFIED = "ai_uncertainty_quantified"
    
    # Meta-Learning Events
    META_LEARNING_TRIGGERED = "meta_learning_triggered"
    META_LEARNING_ADAPTED = "meta_learning_adapted"
    FEW_SHOT_LEARNING_STARTED = "few_shot_learning_started"
    FEW_SHOT_LEARNING_COMPLETED = "few_shot_learning_completed"
    TRANSFER_LEARNING_STARTED = "transfer_learning_started"
    TRANSFER_LEARNING_COMPLETED = "transfer_learning_completed"
    CONTINUAL_LEARNING_STARTED = "continual_learning_started"
    CONTINUAL_LEARNING_COMPLETED = "continual_learning_completed"
    
    # Catalyst Events
    QUANTUM_CATALYST_DETECTED = "quantum_catalyst_detected"
    QUANTUM_CATALYST_ANALYZED = "quantum_catalyst_analyzed"
    QUANTUM_CATALYST_ENTANGLED = "quantum_catalyst_entangled"
    QUANTUM_CATALYST_DECOHERED = "quantum_catalyst_decohered"
    AI_CATALYST_DETECTED = "ai_catalyst_detected"
    AI_CATALYST_ANALYZED = "ai_catalyst_analyzed"
    ENSEMBLE_CATALYST_ANALYZED = "ensemble_catalyst_analyzed"
    
    # Opportunity Events
    AI_OPPORTUNITY_DETECTED = "ai_opportunity_detected"
    AI_OPPORTUNITY_ANALYZED = "ai_opportunity_analyzed"
    AI_OPPORTUNITY_QUANTUM_ENHANCED = "ai_opportunity_quantum_enhanced"
    AI_OPPORTUNITY_ENSEMBLE_VOTED = "ai_opportunity_ensemble_voted"
    AI_OPPORTUNITY_META_LEARNED = "ai_opportunity_meta_learned"
    
    # Infrastructure Events
    GPU_OPERATION_STARTED = "gpu_operation_started"
    GPU_OPERATION_COMPLETED = "gpu_operation_completed"
    GPU_OPERATION_FAILED = "gpu_operation_failed"
    QUANTUM_PROCESSOR_STARTED = "quantum_processor_started"
    QUANTUM_PROCESSOR_COMPLETED = "quantum_processor_completed"
    QUANTUM_PROCESSOR_FAILED = "quantum_processor_failed"
    
    # Performance Events
    PERFORMANCE_METRICS_UPDATED = "performance_metrics_updated"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"
    PERFORMANCE_OPTIMIZATION_TRIGGERED = "performance_optimization_triggered"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class BaseEvent:
    """Base event class"""
    event_id: UUID
    event_type: EventType
    timestamp: datetime
    priority: EventPriority
    source: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate event invariants"""
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        
        if not self.timestamp:
            raise ValueError("Event timestamp cannot be empty")
        
        if not self.source:
            raise ValueError("Event source cannot be empty")


@dataclass
class QuantumStateCreated(BaseEvent):
    """Event when quantum state is created"""
    quantum_state: QuantumState
    qubits: List[int]
    coherence_time: Optional[float] = None
    fidelity: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.qubits:
            raise ValueError("Qubits list cannot be empty")


@dataclass
class QuantumStateMeasured(BaseEvent):
    """Event when quantum state is measured"""
    quantum_state: QuantumState
    measurement_result: str
    measurement_basis: str
    fidelity: Optional[float] = None
    error_rate: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.measurement_result:
            raise ValueError("Measurement result cannot be empty")


@dataclass
class QuantumStateEntangled(BaseEvent):
    """Event when quantum states become entangled"""
    quantum_state: QuantumState
    entangled_qubits: List[int]
    entanglement_strength: Optional[float] = None
    bell_inequality_violation: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.entangled_qubits:
            raise ValueError("Entangled qubits list cannot be empty")


@dataclass
class QuantumStateDecohered(BaseEvent):
    """Event when quantum state decoheres"""
    quantum_state: QuantumState
    decoherence_time: float
    decoherence_rate: Optional[float] = None
    environmental_noise: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.decoherence_time <= 0:
            raise ValueError("Decoherence time must be positive")


@dataclass
class QuantumGateApplied(BaseEvent):
    """Event when quantum gate is applied"""
    quantum_state: QuantumState
    gate_name: str
    target_qubits: List[int]
    gate_fidelity: Optional[float] = None
    processing_time_ns: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.gate_name:
            raise ValueError("Gate name cannot be empty")
        if not self.target_qubits:
            raise ValueError("Target qubits list cannot be empty")


@dataclass
class QuantumErrorDetected(BaseEvent):
    """Event when quantum error is detected"""
    quantum_state: QuantumState
    error_type: str
    error_rate: float
    error_syndrome: Optional[str] = None
    affected_qubits: Optional[List[int]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.error_type:
            raise ValueError("Error type cannot be empty")
        if self.error_rate < 0 or self.error_rate > 1:
            raise ValueError("Error rate must be between 0 and 1")


@dataclass
class QuantumErrorCorrected(BaseEvent):
    """Event when quantum error is corrected"""
    quantum_state: QuantumState
    error_type: str
    correction_applied: str
    correction_fidelity: Optional[float] = None
    correction_time_ns: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.error_type:
            raise ValueError("Error type cannot be empty")
        if not self.correction_applied:
            raise ValueError("Correction applied cannot be empty")


@dataclass
class AIModelTrained(BaseEvent):
    """Event when AI model is trained"""
    model_name: str
    model_type: str
    training_accuracy: float
    validation_accuracy: float
    training_time_ms: Optional[int] = None
    model_parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        if not (0 <= self.training_accuracy <= 1):
            raise ValueError("Training accuracy must be between 0 and 1")
        if not (0 <= self.validation_accuracy <= 1):
            raise ValueError("Validation accuracy must be between 0 and 1")


@dataclass
class AIModelPredicted(BaseEvent):
    """Event when AI model makes prediction"""
    model_name: str
    prediction: float
    confidence: float
    uncertainty: Optional[float] = None
    processing_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class AIEnsembleCreated(BaseEvent):
    """Event when AI ensemble is created"""
    ensemble_name: str
    model_names: List[str]
    ensemble_strategy: str
    model_weights: Optional[List[float]] = None
    ensemble_size: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.ensemble_name:
            raise ValueError("Ensemble name cannot be empty")
        if not self.model_names:
            raise ValueError("Model names list cannot be empty")
        if not self.ensemble_strategy:
            raise ValueError("Ensemble strategy cannot be empty")


@dataclass
class AIEnsembleVoted(BaseEvent):
    """Event when AI ensemble votes"""
    ensemble_name: str
    predictions: List[float]
    ensemble_prediction: float
    ensemble_confidence: float
    voting_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.ensemble_name:
            raise ValueError("Ensemble name cannot be empty")
        if not self.predictions:
            raise ValueError("Predictions list cannot be empty")
        if not (0 <= self.ensemble_confidence <= 1):
            raise ValueError("Ensemble confidence must be between 0 and 1")


@dataclass
class AIUncertaintyQuantified(BaseEvent):
    """Event when AI uncertainty is quantified"""
    model_name: str
    uncertainty_score: UncertaintyScore
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not (0 <= self.epistemic_uncertainty <= 1):
            raise ValueError("Epistemic uncertainty must be between 0 and 1")
        if not (0 <= self.aleatoric_uncertainty <= 1):
            raise ValueError("Aleatoric uncertainty must be between 0 and 1")
        if not (0 <= self.total_uncertainty <= 1):
            raise ValueError("Total uncertainty must be between 0 and 1")


@dataclass
class MetaLearningTriggered(BaseEvent):
    """Event when meta-learning is triggered"""
    meta_learning_type: str
    source_domain: str
    target_domain: str
    adaptation_tasks: List[str]
    meta_learning_score: Optional[MetaLearningScore] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.meta_learning_type:
            raise ValueError("Meta-learning type cannot be empty")
        if not self.source_domain:
            raise ValueError("Source domain cannot be empty")
        if not self.target_domain:
            raise ValueError("Target domain cannot be empty")
        if not self.adaptation_tasks:
            raise ValueError("Adaptation tasks list cannot be empty")


@dataclass
class MetaLearningAdapted(BaseEvent):
    """Event when meta-learning adaptation is completed"""
    meta_learning_type: str
    adaptation_success: bool
    adaptation_accuracy: float
    adaptation_time_ms: Optional[int] = None
    meta_learning_score: Optional[MetaLearningScore] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.meta_learning_type:
            raise ValueError("Meta-learning type cannot be empty")
        if not (0 <= self.adaptation_accuracy <= 1):
            raise ValueError("Adaptation accuracy must be between 0 and 1")


@dataclass
class FewShotLearningStarted(BaseEvent):
    """Event when few-shot learning starts"""
    model_name: str
    num_shots: int
    num_classes: int
    support_set_size: int
    query_set_size: int
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if self.num_shots <= 0:
            raise ValueError("Number of shots must be positive")
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")


@dataclass
class FewShotLearningCompleted(BaseEvent):
    """Event when few-shot learning completes"""
    model_name: str
    support_accuracy: float
    query_accuracy: float
    learning_time_ms: Optional[int] = None
    meta_learning_score: Optional[MetaLearningScore] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not (0 <= self.support_accuracy <= 1):
            raise ValueError("Support accuracy must be between 0 and 1")
        if not (0 <= self.query_accuracy <= 1):
            raise ValueError("Query accuracy must be between 0 and 1")


@dataclass
class TransferLearningStarted(BaseEvent):
    """Event when transfer learning starts"""
    source_model: str
    target_model: str
    source_domain: str
    target_domain: str
    transfer_layers: List[str]
    
    def __post_init__(self):
        super().__post_init__()
        if not self.source_model:
            raise ValueError("Source model cannot be empty")
        if not self.target_model:
            raise ValueError("Target model cannot be empty")
        if not self.source_domain:
            raise ValueError("Source domain cannot be empty")
        if not self.target_domain:
            raise ValueError("Target domain cannot be empty")


@dataclass
class TransferLearningCompleted(BaseEvent):
    """Event when transfer learning completes"""
    source_model: str
    target_model: str
    source_accuracy: float
    target_accuracy: float
    transfer_effectiveness: float
    transfer_time_ms: Optional[int] = None
    meta_learning_score: Optional[MetaLearningScore] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.source_model:
            raise ValueError("Source model cannot be empty")
        if not self.target_model:
            raise ValueError("Target model cannot be empty")
        if not (0 <= self.source_accuracy <= 1):
            raise ValueError("Source accuracy must be between 0 and 1")
        if not (0 <= self.target_accuracy <= 1):
            raise ValueError("Target accuracy must be between 0 and 1")
        if not (0 <= self.transfer_effectiveness <= 1):
            raise ValueError("Transfer effectiveness must be between 0 and 1")


@dataclass
class ContinualLearningStarted(BaseEvent):
    """Event when continual learning starts"""
    model_name: str
    num_tasks: int
    task_sequence: List[str]
    memory_size: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if self.num_tasks <= 0:
            raise ValueError("Number of tasks must be positive")
        if not self.task_sequence:
            raise ValueError("Task sequence cannot be empty")


@dataclass
class ContinualLearningCompleted(BaseEvent):
    """Event when continual learning completes"""
    model_name: str
    task_accuracies: List[float]
    average_accuracy: float
    catastrophic_forgetting: float
    learning_time_ms: Optional[int] = None
    meta_learning_score: Optional[MetaLearningScore] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.task_accuracies:
            raise ValueError("Task accuracies list cannot be empty")
        if not (0 <= self.average_accuracy <= 1):
            raise ValueError("Average accuracy must be between 0 and 1")
        if not (0 <= self.catastrophic_forgetting <= 1):
            raise ValueError("Catastrophic forgetting must be between 0 and 1")


@dataclass
class QuantumCatalystDetected(BaseEvent):
    """Event when quantum catalyst is detected"""
    catalyst: QuantumCatalyst
    detection_confidence: float
    detection_time_ms: Optional[int] = None
    quantum_state: Optional[QuantumState] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.detection_confidence <= 1):
            raise ValueError("Detection confidence must be between 0 and 1")


@dataclass
class QuantumCatalystAnalyzed(BaseEvent):
    """Event when quantum catalyst is analyzed"""
    catalyst: QuantumCatalyst
    analysis_result: str
    quantum_fidelity: Optional[float] = None
    analysis_time_ms: Optional[int] = None
    quantum_state: Optional[QuantumState] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.analysis_result:
            raise ValueError("Analysis result cannot be empty")


@dataclass
class QuantumCatalystEntangled(BaseEvent):
    """Event when quantum catalyst becomes entangled"""
    catalyst: QuantumCatalyst
    entangled_catalyst_id: UUID
    entanglement_strength: float
    quantum_state: Optional[QuantumState] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.entanglement_strength <= 1):
            raise ValueError("Entanglement strength must be between 0 and 1")


@dataclass
class QuantumCatalystDecohered(BaseEvent):
    """Event when quantum catalyst decoheres"""
    catalyst: QuantumCatalyst
    decoherence_time: float
    decoherence_rate: Optional[float] = None
    quantum_state: Optional[QuantumState] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.decoherence_time <= 0:
            raise ValueError("Decoherence time must be positive")


@dataclass
class AIOpportunityDetected(BaseEvent):
    """Event when AI opportunity is detected"""
    opportunity: AIOpportunity
    detection_confidence: float
    detection_time_ms: Optional[int] = None
    ai_models_used: Optional[List[str]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.detection_confidence <= 1):
            raise ValueError("Detection confidence must be between 0 and 1")


@dataclass
class AIOpportunityAnalyzed(BaseEvent):
    """Event when AI opportunity is analyzed"""
    opportunity: AIOpportunity
    analysis_result: str
    ensemble_confidence: Optional[EnsembleConfidence] = None
    uncertainty_score: Optional[UncertaintyScore] = None
    analysis_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.analysis_result:
            raise ValueError("Analysis result cannot be empty")


@dataclass
class AIOpportunityQuantumEnhanced(BaseEvent):
    """Event when AI opportunity is quantum enhanced"""
    opportunity: AIOpportunity
    quantum_enhancement_factor: float
    quantum_state: Optional[QuantumState] = None
    enhancement_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.quantum_enhancement_factor <= 0:
            raise ValueError("Quantum enhancement factor must be positive")


@dataclass
class AIOpportunityEnsembleVoted(BaseEvent):
    """Event when AI opportunity ensemble votes"""
    opportunity: AIOpportunity
    ensemble_prediction: float
    ensemble_confidence: float
    voting_time_ms: Optional[int] = None
    ensemble_models: Optional[List[str]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.ensemble_confidence <= 1):
            raise ValueError("Ensemble confidence must be between 0 and 1")


@dataclass
class AIOpportunityMetaLearned(BaseEvent):
    """Event when AI opportunity meta-learning is applied"""
    opportunity: AIOpportunity
    meta_learning_score: MetaLearningScore
    adaptation_success: bool
    adaptation_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()


@dataclass
class GPUOperationStarted(BaseEvent):
    """Event when GPU operation starts"""
    operation_type: str
    gpu_id: int
    memory_required: int
    estimated_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.operation_type:
            raise ValueError("Operation type cannot be empty")
        if self.gpu_id < 0:
            raise ValueError("GPU ID must be non-negative")
        if self.memory_required < 0:
            raise ValueError("Memory required must be non-negative")


@dataclass
class GPUOperationCompleted(BaseEvent):
    """Event when GPU operation completes"""
    operation_type: str
    gpu_id: int
    processing_time_ms: int
    memory_used: int
    gpu_utilization: float
    success: bool
    
    def __post_init__(self):
        super().__post_init__()
        if not self.operation_type:
            raise ValueError("Operation type cannot be empty")
        if self.gpu_id < 0:
            raise ValueError("GPU ID must be non-negative")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time must be non-negative")
        if not (0 <= self.gpu_utilization <= 1):
            raise ValueError("GPU utilization must be between 0 and 1")


@dataclass
class GPUOperationFailed(BaseEvent):
    """Event when GPU operation fails"""
    operation_type: str
    gpu_id: int
    error_message: str
    error_code: Optional[str] = None
    failure_time_ms: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.operation_type:
            raise ValueError("Operation type cannot be empty")
        if self.gpu_id < 0:
            raise ValueError("GPU ID must be non-negative")
        if not self.error_message:
            raise ValueError("Error message cannot be empty")


@dataclass
class QuantumProcessorStarted(BaseEvent):
    """Event when quantum processor starts"""
    processor_type: str
    num_qubits: int
    coherence_time: float
    gate_fidelity: float
    
    def __post_init__(self):
        super().__post_init__()
        if not self.processor_type:
            raise ValueError("Processor type cannot be empty")
        if self.num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        if self.coherence_time <= 0:
            raise ValueError("Coherence time must be positive")
        if not (0 <= self.gate_fidelity <= 1):
            raise ValueError("Gate fidelity must be between 0 and 1")


@dataclass
class QuantumProcessorCompleted(BaseEvent):
    """Event when quantum processor completes"""
    processor_type: str
    processing_time_ns: int
    quantum_fidelity: float
    error_rate: float
    success: bool
    
    def __post_init__(self):
        super().__post_init__()
        if not self.processor_type:
            raise ValueError("Processor type cannot be empty")
        if self.processing_time_ns < 0:
            raise ValueError("Processing time must be non-negative")
        if not (0 <= self.quantum_fidelity <= 1):
            raise ValueError("Quantum fidelity must be between 0 and 1")
        if not (0 <= self.error_rate <= 1):
            raise ValueError("Error rate must be between 0 and 1")


@dataclass
class QuantumProcessorFailed(BaseEvent):
    """Event when quantum processor fails"""
    processor_type: str
    error_message: str
    error_code: Optional[str] = None
    failure_time_ns: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.processor_type:
            raise ValueError("Processor type cannot be empty")
        if not self.error_message:
            raise ValueError("Error message cannot be empty")


@dataclass
class PerformanceMetricsUpdated(BaseEvent):
    """Event when performance metrics are updated"""
    metrics_type: str
    metrics_data: Dict[str, Any]
    timestamp: datetime
    performance_score: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.metrics_type:
            raise ValueError("Metrics type cannot be empty")
        if not self.metrics_data:
            raise ValueError("Metrics data cannot be empty")


@dataclass
class PerformanceThresholdExceeded(BaseEvent):
    """Event when performance threshold is exceeded"""
    threshold_type: str
    threshold_value: float
    actual_value: float
    severity: str
    recommendations: Optional[List[str]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.threshold_type:
            raise ValueError("Threshold type cannot be empty")
        if not self.severity:
            raise ValueError("Severity cannot be empty")


@dataclass
class PerformanceOptimizationTriggered(BaseEvent):
    """Event when performance optimization is triggered"""
    optimization_type: str
    current_performance: float
    target_performance: float
    optimization_strategy: str
    estimated_improvement: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.optimization_type:
            raise ValueError("Optimization type cannot be empty")
        if not self.optimization_strategy:
            raise ValueError("Optimization strategy cannot be empty")
        if not (0 <= self.current_performance <= 1):
            raise ValueError("Current performance must be between 0 and 1")
        if not (0 <= self.target_performance <= 1):
            raise ValueError("Target performance must be between 0 and 1")

