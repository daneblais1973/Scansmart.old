"""
Quantum-Enhanced Catalyst Entity
Advanced catalyst detection with quantum computing, AI orchestration, and uncertainty quantification
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID, uuid4
import numpy as np
from scipy.stats import entropy
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..value_objects import QuantumState, UncertaintyScore, EnsembleConfidence, MetaLearningScore


class QuantumCatalystType(Enum):
    """Advanced quantum catalyst types with AI classification"""
    # Quantum Financial Catalysts
    QUANTUM_MOMENTUM_SHIFT = "quantum_momentum_shift"
    QUANTUM_VOLATILITY_BREAKOUT = "quantum_volatility_breakout"
    QUANTUM_CORRELATION_BREAKDOWN = "quantum_correlation_breakdown"
    QUANTUM_REGIME_CHANGE = "quantum_regime_change"
    
    # AI-Detected Catalysts
    AI_PATTERN_RECOGNITION = "ai_pattern_recognition"
    AI_ANOMALY_DETECTION = "ai_anomaly_detection"
    AI_SENTIMENT_SHIFT = "ai_sentiment_shift"
    AI_MARKET_MICROSTRUCTURE = "ai_market_microstructure"
    
    # Multi-Modal Catalysts
    MULTIMODAL_NEWS_SENTIMENT = "multimodal_news_sentiment"
    MULTIMODAL_SOCIAL_SIGNALS = "multimodal_social_signals"
    MULTIMODAL_OPTIONS_FLOW = "multimodal_options_flow"
    MULTIMODAL_DARK_POOL_ACTIVITY = "multimodal_dark_pool_activity"
    
    # Quantum Machine Learning Catalysts
    QML_PREDICTION_CONFIDENCE = "qml_prediction_confidence"
    QML_ENSEMBLE_AGREEMENT = "qml_ensemble_agreement"
    QML_MODEL_UNCERTAINTY = "qml_model_uncertainty"
    QML_ADVERSARIAL_DETECTION = "qml_adversarial_detection"
    
    # Meta-Learning Catalysts
    META_LEARNING_ADAPTATION = "meta_learning_adaptation"
    FEW_SHOT_LEARNING_TRIGGER = "few_shot_learning_trigger"
    TRANSFER_LEARNING_OPPORTUNITY = "transfer_learning_opportunity"
    CONTINUAL_LEARNING_UPDATE = "continual_learning_update"
    
    # Traditional Financial Catalysts (Enhanced with Quantum Analysis)
    # Earnings & Financial
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    GUIDANCE_RAISE = "guidance_raise"
    GUIDANCE_CUT = "guidance_cut"
    REVENUE_GROWTH = "revenue_growth"
    PROFIT_MARGIN_EXPANSION = "profit_margin_expansion"
    
    # Regulatory & Approval
    FDA_APPROVAL = "fda_approval"
    FDA_REJECTION = "fda_rejection"
    REGULATORY_APPROVAL = "regulatory_approval"
    REGULATORY_REJECTION = "regulatory_rejection"
    CLINICAL_TRIAL_SUCCESS = "clinical_trial_success"
    CLINICAL_TRIAL_FAILURE = "clinical_trial_failure"
    
    # Product & Innovation
    PRODUCT_LAUNCH = "product_launch"
    PRODUCT_RECALL = "product_recall"
    TECHNOLOGICAL_BREAKTHROUGH = "technological_breakthrough"
    PATENT_APPROVAL = "patent_approval"
    PATENT_LOSS = "patent_loss"
    
    # Corporate Actions
    MERGER_ANNOUNCEMENT = "merger_announcement"
    ACQUISITION_ANNOUNCEMENT = "acquisition_announcement"
    SPINOFF_ANNOUNCEMENT = "spinoff_announcement"
    DIVIDEND_INCREASE = "dividend_increase"
    DIVIDEND_CUT = "dividend_cut"
    STOCK_BUYBACK = "stock_buyback"
    STOCK_SPLIT = "stock_split"
    
    # Management & Leadership
    CEO_APPOINTMENT = "ceo_appointment"
    CEO_DEPARTURE = "ceo_departure"
    BOARD_CHANGES = "board_changes"
    MANAGEMENT_RESTRUCTURE = "management_restructure"
    
    # Market & Sector
    SECTOR_TAILWIND = "sector_tailwind"
    SECTOR_HEADWIND = "sector_headwind"
    MARKET_REGIME_CHANGE = "market_regime_change"
    INTEREST_RATE_CHANGE = "interest_rate_change"
    CURRENCY_MOVEMENT = "currency_movement"
    
    # Geopolitical & Macro
    TRADE_WAR_DEVELOPMENT = "trade_war_development"
    GEOPOLITICAL_TENSION = "geopolitical_tension"
    ECONOMIC_INDICATOR = "economic_indicator"
    CENTRAL_BANK_POLICY = "central_bank_policy"
    
    # Analyst & Institutional
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    PRICE_TARGET_RAISE = "price_target_raise"
    PRICE_TARGET_CUT = "price_target_cut"
    INSTITUTIONAL_BUYING = "institutional_buying"
    INSTITUTIONAL_SELLING = "institutional_selling"
    
    # Social & Media
    VIRAL_SOCIAL_MEDIA = "viral_social_media"
    INFLUENCER_ENDORSEMENT = "influencer_endorsement"
    CONTROVERSY = "controversy"
    BRAND_DAMAGE = "brand_damage"
    
    # Supply Chain & Operations
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    PRODUCTION_INCREASE = "production_increase"
    PRODUCTION_DECREASE = "production_decrease"
    FACILITY_EXPANSION = "facility_expansion"
    FACILITY_CLOSURE = "facility_closure"
    
    # Legal & Compliance
    LEGAL_SETTLEMENT = "legal_settlement"
    LAWSUIT_FILING = "lawsuit_filing"
    REGULATORY_FINE = "regulatory_fine"
    COMPLIANCE_VIOLATION = "compliance_violation"


class QuantumImpactLevel(Enum):
    """Quantum-enhanced impact levels with uncertainty bounds"""
    NEGLIGIBLE = "negligible"  # < 0.1
    LOW = "low"                # 0.1 - 0.3
    MODERATE = "moderate"      # 0.3 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    CRITICAL = "critical"      # 0.8 - 0.95
    CATASTROPHIC = "catastrophic"  # > 0.95


class QuantumConfidenceLevel(Enum):
    """Quantum confidence levels with superposition states"""
    SUPERPOSITION = "superposition"  # Multiple states simultaneously
    ENTANGLED = "entangled"          # Correlated with other catalysts
    COLLAPSED = "collapsed"          # Single definite state
    MEASURED = "measured"            # Observed and confirmed


@dataclass(frozen=True)
class QuantumCatalyst:
    """
    Ultra-advanced quantum-enhanced catalyst entity
    Incorporates quantum computing, AI orchestration, and uncertainty quantification
    """
    
    # Core Properties
    title: str
    content: str
    source: str
    catalyst_type: QuantumCatalystType
    impact_level: QuantumImpactLevel
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    
    # Quantum Properties
    quantum_state: Optional[QuantumState] = None
    superposition_amplitude: Optional[Decimal] = None
    entanglement_correlations: List[UUID] = field(default_factory=list)
    quantum_coherence: Optional[Decimal] = None
    
    # AI Orchestration Properties
    ai_model_ensemble: List[str] = field(default_factory=list)
    meta_learning_score: Optional[MetaLearningScore] = None
    few_shot_confidence: Optional[Decimal] = None
    transfer_learning_effectiveness: Optional[Decimal] = None
    
    # Uncertainty Quantification
    uncertainty_score: Optional[UncertaintyScore] = None
    ensemble_confidence: Optional[EnsembleConfidence] = None
    epistemic_uncertainty: Optional[Decimal] = None
    aleatoric_uncertainty: Optional[Decimal] = None
    
    # Financial Context
    affected_symbols: List[str] = field(default_factory=list)
    sector_impact_vector: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None
    volatility_impact: Optional[Decimal] = None
    
    # Temporal Properties
    detected_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    quantum_decoherence_time: Optional[datetime] = None
    
    # Advanced Analysis
    sentiment_vector: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    transformer_embeddings: Optional[np.ndarray] = None
    graph_neural_features: Optional[np.ndarray] = None
    
    # Metadata
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Metrics
    processing_latency_ns: Optional[int] = None
    quantum_gate_count: Optional[int] = None
    ai_inference_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Validate quantum catalyst invariants"""
        if not self.title or not self.title.strip():
            raise ValueError("Quantum catalyst title cannot be empty")
        
        if not self.content or not self.content.strip():
            raise ValueError("Quantum catalyst content cannot be empty")
        
        if not self.source or not self.source.strip():
            raise ValueError("Quantum catalyst source cannot be empty")
        
        # Validate quantum properties
        if self.superposition_amplitude is not None:
            if not (0 <= self.superposition_amplitude <= 1):
                raise ValueError("Superposition amplitude must be between 0 and 1")
        
        if self.quantum_coherence is not None:
            if not (0 <= self.quantum_coherence <= 1):
                raise ValueError("Quantum coherence must be between 0 and 1")
    
    @property
    def is_quantum_active(self) -> bool:
        """Check if quantum catalyst is in active superposition state"""
        if self.quantum_decoherence_time is None:
            return True
        return datetime.utcnow() < self.quantum_decoherence_time
    
    @property
    def quantum_age_nanoseconds(self) -> int:
        """Age of quantum catalyst in nanoseconds"""
        return int((datetime.utcnow() - self.detected_at).total_seconds() * 1e9)
    
    @property
    def is_entangled(self) -> bool:
        """Check if catalyst is entangled with other catalysts"""
        return len(self.entanglement_correlations) > 0
    
    @property
    def quantum_entropy(self) -> Optional[float]:
        """Calculate quantum entropy of the catalyst state"""
        if self.sentiment_vector is not None:
            # Normalize vector to probability distribution
            prob_dist = np.abs(self.sentiment_vector) ** 2
            prob_dist = prob_dist / np.sum(prob_dist)
            return entropy(prob_dist)
        return None
    
    def add_entanglement(self, other_catalyst_id: UUID) -> 'QuantumCatalyst':
        """Add quantum entanglement with another catalyst"""
        if other_catalyst_id not in self.entanglement_correlations:
            new_correlations = self.entanglement_correlations + [other_catalyst_id]
            return QuantumCatalyst(
                id=self.id,
                title=self.title,
                content=self.content,
                source=self.source,
                catalyst_type=self.catalyst_type,
                impact_level=self.impact_level,
                quantum_state=self.quantum_state,
                superposition_amplitude=self.superposition_amplitude,
                entanglement_correlations=new_correlations,
                quantum_coherence=self.quantum_coherence,
                ai_model_ensemble=self.ai_model_ensemble,
                meta_learning_score=self.meta_learning_score,
                few_shot_confidence=self.few_shot_confidence,
                transfer_learning_effectiveness=self.transfer_learning_effectiveness,
                uncertainty_score=self.uncertainty_score,
                ensemble_confidence=self.ensemble_confidence,
                epistemic_uncertainty=self.epistemic_uncertainty,
                aleatoric_uncertainty=self.aleatoric_uncertainty,
                affected_symbols=self.affected_symbols,
                sector_impact_vector=self.sector_impact_vector,
                correlation_matrix=self.correlation_matrix,
                volatility_impact=self.volatility_impact,
                detected_at=self.detected_at,
                published_at=self.published_at,
                expires_at=self.expires_at,
                quantum_decoherence_time=self.quantum_decoherence_time,
                sentiment_vector=self.sentiment_vector,
                attention_weights=self.attention_weights,
                transformer_embeddings=self.transformer_embeddings,
                graph_neural_features=self.graph_neural_features,
                url=self.url,
                tags=self.tags,
                metadata=self.metadata,
                processing_latency_ns=self.processing_latency_ns,
                quantum_gate_count=self.quantum_gate_count,
                ai_inference_time_ms=self.ai_inference_time_ms
            )
        return self
    
    def collapse_quantum_state(self, measurement_basis: str = "computational") -> 'QuantumCatalyst':
        """Collapse quantum state to classical state"""
        collapsed_confidence = self.ensemble_confidence.value if self.ensemble_confidence else Decimal('0.5')
        
        return QuantumCatalyst(
            id=self.id,
            title=self.title,
            content=self.content,
            source=self.source,
            catalyst_type=self.catalyst_type,
            impact_level=self.impact_level,
            quantum_state=None,  # Collapsed to classical
            superposition_amplitude=Decimal('1.0'),  # Definite state
            entanglement_correlations=self.entanglement_correlations,
            quantum_coherence=Decimal('0.0'),  # No coherence after measurement
            ai_model_ensemble=self.ai_model_ensemble,
            meta_learning_score=self.meta_learning_score,
            few_shot_confidence=self.few_shot_confidence,
            transfer_learning_effectiveness=self.transfer_learning_effectiveness,
            uncertainty_score=self.uncertainty_score,
            ensemble_confidence=self.ensemble_confidence,
            epistemic_uncertainty=self.epistemic_uncertainty,
            aleatoric_uncertainty=self.aleatoric_uncertainty,
            affected_symbols=self.affected_symbols,
            sector_impact_vector=self.sector_impact_vector,
            correlation_matrix=self.correlation_matrix,
            volatility_impact=self.volatility_impact,
            detected_at=self.detected_at,
            published_at=self.published_at,
            expires_at=self.expires_at,
            quantum_decoherence_time=datetime.utcnow(),  # Decoherence occurred
            sentiment_vector=self.sentiment_vector,
            attention_weights=self.attention_weights,
            transformer_embeddings=self.transformer_embeddings,
            graph_neural_features=self.graph_neural_features,
            url=self.url,
            tags=self.tags,
            metadata=self.metadata,
            processing_latency_ns=self.processing_latency_ns,
            quantum_gate_count=self.quantum_gate_count,
            ai_inference_time_ms=self.ai_inference_time_ms
        )
    
    def calculate_quantum_priority_score(self) -> Decimal:
        """Calculate quantum-enhanced priority score"""
        base_score = Decimal('0.5')
        
        # Impact weight
        impact_weights = {
            QuantumImpactLevel.NEGLIGIBLE: Decimal('0.1'),
            QuantumImpactLevel.LOW: Decimal('0.3'),
            QuantumImpactLevel.MODERATE: Decimal('0.5'),
            QuantumImpactLevel.HIGH: Decimal('0.7'),
            QuantumImpactLevel.CRITICAL: Decimal('0.9'),
            QuantumImpactLevel.CATASTROPHIC: Decimal('1.0')
        }
        
        impact_weight = impact_weights[self.impact_level]
        
        # Ensemble confidence weight
        confidence_weight = self.ensemble_confidence.value if self.ensemble_confidence else Decimal('0.5')
        
        # Quantum coherence bonus
        coherence_bonus = self.quantum_coherence if self.quantum_coherence else Decimal('0.0')
        
        # Entanglement bonus
        entanglement_bonus = Decimal(str(len(self.entanglement_correlations))) * Decimal('0.1')
        
        # Calculate final score
        quantum_score = (base_score + impact_weight + confidence_weight + coherence_bonus + entanglement_bonus) / Decimal('4.0')
        
        return min(quantum_score, Decimal('1.0'))
    
    def is_high_quantum_impact(self) -> bool:
        """Check if catalyst has high quantum impact"""
        return self.impact_level in [QuantumImpactLevel.HIGH, QuantumImpactLevel.CRITICAL, QuantumImpactLevel.CATASTROPHIC]
    
    def is_ai_detected(self) -> bool:
        """Check if catalyst was detected by AI models"""
        ai_types = {
            QuantumCatalystType.AI_PATTERN_RECOGNITION,
            QuantumCatalystType.AI_ANOMALY_DETECTION,
            QuantumCatalystType.AI_SENTIMENT_SHIFT,
            QuantumCatalystType.AI_MARKET_MICROSTRUCTURE,
            QuantumCatalystType.QML_PREDICTION_CONFIDENCE,
            QuantumCatalystType.QML_ENSEMBLE_AGREEMENT,
            QuantumCatalystType.QML_MODEL_UNCERTAINTY,
            QuantumCatalystType.QML_ADVERSARIAL_DETECTION
        }
        return self.catalyst_type in ai_types
    
    def is_meta_learning_trigger(self) -> bool:
        """Check if catalyst triggers meta-learning adaptation"""
        meta_types = {
            QuantumCatalystType.META_LEARNING_ADAPTATION,
            QuantumCatalystType.FEW_SHOT_LEARNING_TRIGGER,
            QuantumCatalystType.TRANSFER_LEARNING_OPPORTUNITY,
            QuantumCatalystType.CONTINUAL_LEARNING_UPDATE
        }
        return self.catalyst_type in meta_types
    
    async def process_quantum_analysis(self, quantum_processor) -> 'QuantumCatalyst':
        """Process quantum analysis asynchronously"""
        # This would integrate with actual quantum computing hardware
        # For now, simulate quantum processing
        await asyncio.sleep(0.001)  # Simulate quantum processing time
        
        # Update quantum properties based on analysis
        updated_quantum_state = await quantum_processor.analyze_catalyst(self)
        
        return QuantumCatalyst(
            id=self.id,
            title=self.title,
            content=self.content,
            source=self.source,
            catalyst_type=self.catalyst_type,
            impact_level=self.impact_level,
            quantum_state=updated_quantum_state,
            superposition_amplitude=self.superposition_amplitude,
            entanglement_correlations=self.entanglement_correlations,
            quantum_coherence=self.quantum_coherence,
            ai_model_ensemble=self.ai_model_ensemble,
            meta_learning_score=self.meta_learning_score,
            few_shot_confidence=self.few_shot_confidence,
            transfer_learning_effectiveness=self.transfer_learning_effectiveness,
            uncertainty_score=self.uncertainty_score,
            ensemble_confidence=self.ensemble_confidence,
            epistemic_uncertainty=self.epistemic_uncertainty,
            aleatoric_uncertainty=self.aleatoric_uncertainty,
            affected_symbols=self.affected_symbols,
            sector_impact_vector=self.sector_impact_vector,
            correlation_matrix=self.correlation_matrix,
            volatility_impact=self.volatility_impact,
            detected_at=self.detected_at,
            published_at=self.published_at,
            expires_at=self.expires_at,
            quantum_decoherence_time=self.quantum_decoherence_time,
            sentiment_vector=self.sentiment_vector,
            attention_weights=self.attention_weights,
            transformer_embeddings=self.transformer_embeddings,
            graph_neural_features=self.graph_neural_features,
            url=self.url,
            tags=self.tags,
            metadata=self.metadata,
            processing_latency_ns=self.processing_latency_ns,
            quantum_gate_count=self.quantum_gate_count,
            ai_inference_time_ms=self.ai_inference_time_ms
        )

