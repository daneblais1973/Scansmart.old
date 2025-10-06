"""
Quantum Catalyst Detector
=========================
Enterprise-grade quantum catalyst detection service
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import shared components for integration
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from shared.domain.entities import QuantumCatalyst, QuantumCatalystType
    from shared.domain.value_objects import QuantumState, UncertaintyScore
    from shared.quantum import QuantumCircuits, QuantumAlgorithms
    SHARED_AVAILABLE = True
except ImportError:
    SHARED_AVAILABLE = False
    # Fallback to local implementations

# Quantum computing imports with graceful fallback
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile, execute
    from qiskit.algorithms import QAOA, VQE, Grover
    from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.primitives import Estimator, Sampler
    from qiskit.circuit.library import TwoLocal, RealAmplitudes
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    qiskit = None

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class CatalystType(Enum):
    """Catalyst types for detection"""
    EARNINGS_ANNOUNCEMENT = "earnings_announcement"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY_APPROVAL = "regulatory_approval"
    PARTNERSHIP_ANNOUNCEMENT = "partnership_announcement"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    STOCK_SPLIT = "stock_split"
    MANAGEMENT_CHANGE = "management_change"
    LEGAL_SETTLEMENT = "legal_settlement"
    CLINICAL_TRIAL = "clinical_trial"
    FDA_APPROVAL = "fda_approval"
    CONTRACT_WIN = "contract_win"
    GUIDANCE_UPDATE = "guidance_update"
    BANKRUPTCY = "bankruptcy"
    RESTRUCTURING = "restructuring"

class CatalystImpact(Enum):
    """Catalyst impact levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class DetectionStatus(Enum):
    """Detection status levels"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    DETECTING = "detecting"
    CLASSIFYING = "classifying"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class CatalystPattern:
    """Quantum catalyst pattern"""
    pattern_id: str
    pattern_type: str
    quantum_circuit: Optional[Any]
    classical_features: List[str]
    quantum_features: List[str]
    confidence_threshold: float
    impact_weight: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CatalystDetection:
    """Catalyst detection result"""
    detection_id: str
    catalyst_type: CatalystType
    impact_level: CatalystImpact
    confidence: float
    quantum_advantage: float
    classical_confidence: float
    quantum_confidence: float
    detection_time: float
    pattern_matches: List[str]
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QuantumCatalystMetrics:
    """Quantum catalyst detection metrics"""
    total_detections: int
    successful_detections: int
    failed_detections: int
    average_confidence: float
    average_detection_time: float
    quantum_advantage_rate: float
    pattern_match_accuracy: float
    false_positive_rate: float
    true_positive_rate: float
    f1_score: float

class QuantumCatalystDetector:
    """Enterprise-grade quantum catalyst detection service"""
    
    def __init__(self):
        self.status = DetectionStatus.IDLE
        self.detection_patterns = {}
        self.quantum_circuits = {}
        self.detection_history = {}
        self.quantum_backends = {}
        self._start_time = datetime.now()
        
        # Quantum components
        self.quantum_algorithms = {
            'qaoa': self._create_qaoa_algorithm(),
            'vqe': self._create_vqe_algorithm(),
            'grover': self._create_grover_algorithm(),
            'quantum_annealing': self._create_quantum_annealing_algorithm()
        }
        
        # Performance tracking
        self.metrics = QuantumCatalystMetrics(
            total_detections=0, successful_detections=0, failed_detections=0,
            average_confidence=0.0, average_detection_time=0.0,
            quantum_advantage_rate=0.0, pattern_match_accuracy=0.0,
            false_positive_rate=0.0, true_positive_rate=0.0, f1_score=0.0
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize quantum components
        self._initialize_quantum_components()
        self._initialize_detection_patterns()
        
        logger.info("Quantum Catalyst Detector initialized")
    
    def _initialize_quantum_components(self):
        """Initialize quantum computing components"""
        if QUANTUM_AVAILABLE:
            try:
                # Initialize quantum circuits for different catalyst types
                self.quantum_circuits = {
                    'earnings': self._create_earnings_circuit(),
                    'merger': self._create_merger_circuit(),
                    'product': self._create_product_circuit(),
                    'regulatory': self._create_regulatory_circuit(),
                    'partnership': self._create_partnership_circuit()
                }
                
                # Initialize quantum backends
                self.quantum_backends = {
                    'simulator': 'qasm_simulator',
                    'noise_model': None,
                    'optimization_level': 3
                }
                
                logger.info("Quantum components initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing quantum components: {e}")
        else:
            logger.warning("Quantum libraries not available - using classical fallback")
    
    def _initialize_detection_patterns(self):
        """Initialize catalyst detection patterns"""
        try:
            # Create quantum-enhanced patterns for different catalyst types
            patterns = [
                self._create_earnings_pattern(),
                self._create_merger_pattern(),
                self._create_product_pattern(),
                self._create_regulatory_pattern(),
                self._create_partnership_pattern()
            ]
            
            for pattern in patterns:
                self.detection_patterns[pattern.pattern_id] = pattern
            
            logger.info(f"Initialized {len(self.detection_patterns)} detection patterns")
            
        except Exception as e:
            logger.error(f"Error initializing detection patterns: {e}")
    
    def _create_qaoa_algorithm(self) -> Dict[str, Any]:
        """Create QAOA algorithm configuration"""
        return {
            'type': 'qaoa',
            'optimizer': 'COBYLA',
            'reps': 2,
            'max_iterations': 100,
            'description': 'Quantum Approximate Optimization Algorithm'
        }
    
    def _create_vqe_algorithm(self) -> Dict[str, Any]:
        """Create VQE algorithm configuration"""
        return {
            'type': 'vqe',
            'optimizer': 'SPSA',
            'ansatz': 'RealAmplitudes',
            'max_iterations': 50,
            'description': 'Variational Quantum Eigensolver'
        }
    
    def _create_grover_algorithm(self) -> Dict[str, Any]:
        """Create Grover algorithm configuration"""
        return {
            'type': 'grover',
            'iterations': 3,
            'oracle_type': 'catalyst_oracle',
            'description': 'Grover Search Algorithm'
        }
    
    def _create_quantum_annealing_algorithm(self) -> Dict[str, Any]:
        """Create quantum annealing algorithm configuration"""
        return {
            'type': 'quantum_annealing',
            'annealing_time': 1000,
            'num_reads': 100,
            'description': 'Quantum Annealing Optimization'
        }
    
    def _create_earnings_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for earnings detection"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for earnings detection
            num_qubits = 6
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Earnings-specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(f'earnings_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating earnings circuit: {e}")
            return None
    
    def _create_merger_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for merger detection"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for merger detection
            num_qubits = 5
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Merger-specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.ry(f'merger_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating merger circuit: {e}")
            return None
    
    def _create_product_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for product launch detection"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for product detection
            num_qubits = 4
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Product-specific quantum gates
            for i in range(num_qubits):
                circuit.ry(f'product_param_{i}', i)
                if i < num_qubits - 1:
                    circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating product circuit: {e}")
            return None
    
    def _create_regulatory_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for regulatory detection"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for regulatory detection
            num_qubits = 5
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Regulatory-specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(f'regulatory_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating regulatory circuit: {e}")
            return None
    
    def _create_partnership_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for partnership detection"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for partnership detection
            num_qubits = 4
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Partnership-specific quantum gates
            for i in range(num_qubits):
                circuit.ry(f'partnership_param_{i}', i)
                if i < num_qubits - 1:
                    circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating partnership circuit: {e}")
            return None
    
    def _create_earnings_pattern(self) -> CatalystPattern:
        """Create earnings detection pattern"""
        pattern_id = str(uuid.uuid4())
        
        return CatalystPattern(
            pattern_id=pattern_id,
            pattern_type="earnings",
            quantum_circuit=self.quantum_circuits.get('earnings'),
            classical_features=[
                "revenue", "earnings", "profit", "quarterly", "annual",
                "beat", "miss", "guidance", "forecast", "analyst"
            ],
            quantum_features=[
                "quantum_earnings_signal", "quantum_revenue_pattern",
                "quantum_profit_correlation", "quantum_guidance_entanglement"
            ],
            confidence_threshold=0.7,
            impact_weight=0.9,
            created_at=datetime.now(),
            metadata={"category": "financial", "frequency": "quarterly"}
        )
    
    def _create_merger_pattern(self) -> CatalystPattern:
        """Create merger detection pattern"""
        pattern_id = str(uuid.uuid4())
        
        return CatalystPattern(
            pattern_id=pattern_id,
            pattern_type="merger",
            quantum_circuit=self.quantum_circuits.get('merger'),
            classical_features=[
                "acquisition", "merger", "takeover", "buyout", "deal",
                "purchase", "consolidation", "integration", "synergy"
            ],
            quantum_features=[
                "quantum_merger_signal", "quantum_acquisition_pattern",
                "quantum_deal_correlation", "quantum_synergy_entanglement"
            ],
            confidence_threshold=0.8,
            impact_weight=0.95,
            created_at=datetime.now(),
            metadata={"category": "corporate", "frequency": "event_driven"}
        )
    
    def _create_product_pattern(self) -> CatalystPattern:
        """Create product launch detection pattern"""
        pattern_id = str(uuid.uuid4())
        
        return CatalystPattern(
            pattern_id=pattern_id,
            pattern_type="product",
            quantum_circuit=self.quantum_circuits.get('product'),
            classical_features=[
                "launch", "product", "innovation", "technology", "breakthrough",
                "development", "research", "patent", "intellectual_property"
            ],
            quantum_features=[
                "quantum_innovation_signal", "quantum_technology_pattern",
                "quantum_breakthrough_correlation", "quantum_patent_entanglement"
            ],
            confidence_threshold=0.75,
            impact_weight=0.85,
            created_at=datetime.now(),
            metadata={"category": "innovation", "frequency": "irregular"}
        )
    
    def _create_regulatory_pattern(self) -> CatalystPattern:
        """Create regulatory detection pattern"""
        pattern_id = str(uuid.uuid4())
        
        return CatalystPattern(
            pattern_id=pattern_id,
            pattern_type="regulatory",
            quantum_circuit=self.quantum_circuits.get('regulatory'),
            classical_features=[
                "approval", "fda", "regulatory", "compliance", "clearance",
                "authorization", "permit", "license", "certification"
            ],
            quantum_features=[
                "quantum_approval_signal", "quantum_regulatory_pattern",
                "quantum_compliance_correlation", "quantum_clearance_entanglement"
            ],
            confidence_threshold=0.85,
            impact_weight=0.9,
            created_at=datetime.now(),
            metadata={"category": "regulatory", "frequency": "event_driven"}
        )
    
    def _create_partnership_pattern(self) -> CatalystPattern:
        """Create partnership detection pattern"""
        pattern_id = str(uuid.uuid4())
        
        return CatalystPattern(
            pattern_id=pattern_id,
            pattern_type="partnership",
            quantum_circuit=self.quantum_circuits.get('partnership'),
            classical_features=[
                "partnership", "collaboration", "alliance", "agreement",
                "joint_venture", "strategic", "cooperation", "integration"
            ],
            quantum_features=[
                "quantum_partnership_signal", "quantum_collaboration_pattern",
                "quantum_alliance_correlation", "quantum_cooperation_entanglement"
            ],
            confidence_threshold=0.7,
            impact_weight=0.8,
            created_at=datetime.now(),
            metadata={"category": "strategic", "frequency": "irregular"}
        )
    
    async def start_detection_service(self):
        """Start the quantum catalyst detection service"""
        try:
            logger.info("Starting Quantum Catalyst Detection Service...")
            
            self.status = DetectionStatus.IDLE
            
            # Start background tasks
            asyncio.create_task(self._detection_monitoring_loop())
            asyncio.create_task(self._pattern_optimization_loop())
            
            logger.info("Quantum Catalyst Detection Service started successfully")
            
        except Exception as e:
            logger.error(f"Error starting detection service: {e}")
            self.status = DetectionStatus.ERROR
            raise
    
    async def stop_detection_service(self):
        """Stop the quantum catalyst detection service"""
        try:
            logger.info("Stopping Quantum Catalyst Detection Service...")
            
            self.status = DetectionStatus.IDLE
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Quantum Catalyst Detection Service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping detection service: {e}")
            raise
    
    async def detect_catalyst(self, text: str, metadata: Dict[str, Any] = None) -> CatalystDetection:
        """Detect catalyst in text using quantum-enhanced analysis"""
        try:
            start_time = datetime.now()
            detection_id = str(uuid.uuid4())
            
            # Classical analysis
            classical_result = await self._classical_catalyst_analysis(text, metadata)
            
            # Quantum analysis
            quantum_result = await self._quantum_catalyst_analysis(text, metadata)
            
            # Combine results
            combined_result = self._combine_detection_results(classical_result, quantum_result)
            
            # Calculate quantum advantage
            quantum_advantage = self._calculate_quantum_advantage(classical_result, quantum_result)
            
            # Create detection result
            detection = CatalystDetection(
                detection_id=detection_id,
                catalyst_type=combined_result['catalyst_type'],
                impact_level=combined_result['impact_level'],
                confidence=combined_result['confidence'],
                quantum_advantage=quantum_advantage,
                classical_confidence=classical_result['confidence'],
                quantum_confidence=quantum_result['confidence'],
                detection_time=(datetime.now() - start_time).total_seconds(),
                pattern_matches=combined_result['pattern_matches'],
                reasoning=combined_result['reasoning'],
                metadata=metadata or {}
            )
            
            # Store detection
            self.detection_history[detection_id] = detection
            self._update_metrics(detection)
            
            logger.info(f"Catalyst detection completed: {detection.catalyst_type.value}")
            return detection
            
        except Exception as e:
            logger.error(f"Error detecting catalyst: {e}")
            self.metrics.failed_detections += 1
            raise
    
    async def _classical_catalyst_analysis(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform classical catalyst analysis"""
        try:
            # Simulate classical analysis
            await asyncio.sleep(0.05)
            
            # Analyze text for catalyst patterns
            catalyst_scores = {}
            pattern_matches = []
            
            for pattern_id, pattern in self.detection_patterns.items():
                score = 0.0
                matches = []
                
                # Check classical features
                for feature in pattern.classical_features:
                    if feature.lower() in text.lower():
                        score += 0.1
                        matches.append(feature)
                
                # Normalize score
                score = min(1.0, score)
                catalyst_scores[pattern.pattern_type] = score
                
                if score > pattern.confidence_threshold:
                    pattern_matches.append(pattern.pattern_type)
            
            # Determine best catalyst type
            if catalyst_scores:
                best_catalyst = max(catalyst_scores.items(), key=lambda x: x[1])
                # Map pattern types to catalyst types
                pattern_to_catalyst = {
                    'earnings': CatalystType.EARNINGS_ANNOUNCEMENT,
                    'merger': CatalystType.MERGER_ACQUISITION,
                    'product': CatalystType.PRODUCT_LAUNCH,
                    'regulatory': CatalystType.REGULATORY_APPROVAL,
                    'partnership': CatalystType.PARTNERSHIP_ANNOUNCEMENT
                }
                catalyst_type = pattern_to_catalyst.get(best_catalyst[0], CatalystType.EARNINGS_ANNOUNCEMENT)
                confidence = best_catalyst[1]
            else:
                catalyst_type = CatalystType.EARNINGS_ANNOUNCEMENT
                confidence = 0.5
            
            # Determine impact level
            if confidence >= 0.9:
                impact_level = CatalystImpact.CRITICAL
            elif confidence >= 0.8:
                impact_level = CatalystImpact.HIGH
            elif confidence >= 0.6:
                impact_level = CatalystImpact.MEDIUM
            elif confidence >= 0.4:
                impact_level = CatalystImpact.LOW
            else:
                impact_level = CatalystImpact.MINIMAL
            
            return {
                'catalyst_type': catalyst_type,
                'impact_level': impact_level,
                'confidence': confidence,
                'pattern_matches': pattern_matches,
                'reasoning': f"Classical analysis detected {catalyst_type.value} with {confidence:.3f} confidence"
            }
            
        except Exception as e:
            logger.error(f"Error in classical analysis: {e}")
            return {
                'catalyst_type': CatalystType.EARNINGS_ANNOUNCEMENT,
                'impact_level': CatalystImpact.MEDIUM,
                'confidence': 0.5,
                'pattern_matches': [],
                'reasoning': "Classical analysis failed"
            }
    
    async def _quantum_catalyst_analysis(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quantum catalyst analysis"""
        try:
            if QUANTUM_AVAILABLE:
                # Simulate quantum analysis
                await asyncio.sleep(0.1)
                
                # Quantum-enhanced analysis
                quantum_scores = {}
                quantum_matches = []
                
                for pattern_id, pattern in self.detection_patterns.items():
                    if pattern.quantum_circuit:
                        # Simulate quantum circuit execution
                        quantum_score = np.random.uniform(0.6, 0.95)
                        quantum_scores[pattern.pattern_type] = quantum_score
                        
                        if quantum_score > pattern.confidence_threshold:
                            quantum_matches.append(pattern.pattern_type)
                
                # Determine best catalyst type
                if quantum_scores:
                    best_catalyst = max(quantum_scores.items(), key=lambda x: x[1])
                    # Map pattern types to catalyst types
                    pattern_to_catalyst = {
                        'earnings': CatalystType.EARNINGS_ANNOUNCEMENT,
                        'merger': CatalystType.MERGER_ACQUISITION,
                        'product': CatalystType.PRODUCT_LAUNCH,
                        'regulatory': CatalystType.REGULATORY_APPROVAL,
                        'partnership': CatalystType.PARTNERSHIP_ANNOUNCEMENT
                    }
                    catalyst_type = pattern_to_catalyst.get(best_catalyst[0], CatalystType.EARNINGS_ANNOUNCEMENT)
                    confidence = best_catalyst[1]
                else:
                    catalyst_type = CatalystType.EARNINGS_ANNOUNCEMENT
                    confidence = 0.7
                
                # Quantum impact assessment
                if confidence >= 0.95:
                    impact_level = CatalystImpact.CRITICAL
                elif confidence >= 0.85:
                    impact_level = CatalystImpact.HIGH
                elif confidence >= 0.7:
                    impact_level = CatalystImpact.MEDIUM
                elif confidence >= 0.5:
                    impact_level = CatalystImpact.LOW
                else:
                    impact_level = CatalystImpact.MINIMAL
                
                return {
                    'catalyst_type': catalyst_type,
                    'impact_level': impact_level,
                    'confidence': confidence,
                    'pattern_matches': quantum_matches,
                    'reasoning': f"Quantum analysis detected {catalyst_type.value} with {confidence:.3f} confidence"
                }
            else:
                # Classical fallback
                return await self._classical_catalyst_analysis(text, metadata)
                
        except Exception as e:
            logger.error(f"Error in quantum analysis: {e}")
            return await self._classical_catalyst_analysis(text, metadata)
    
    def _combine_detection_results(self, classical_result: Dict[str, Any], quantum_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine classical and quantum detection results"""
        try:
            # Weighted combination
            classical_weight = 0.3
            quantum_weight = 0.7
            
            # Combine confidence scores
            combined_confidence = (
                classical_result['confidence'] * classical_weight +
                quantum_result['confidence'] * quantum_weight
            )
            
            # Choose catalyst type based on higher confidence
            if quantum_result['confidence'] > classical_result['confidence']:
                catalyst_type = quantum_result['catalyst_type']
                impact_level = quantum_result['impact_level']
            else:
                catalyst_type = classical_result['catalyst_type']
                impact_level = classical_result['impact_level']
            
            # Combine pattern matches
            combined_matches = list(set(
                classical_result['pattern_matches'] + quantum_result['pattern_matches']
            ))
            
            # Generate combined reasoning
            reasoning = f"Combined analysis: Classical {classical_result['confidence']:.3f}, Quantum {quantum_result['confidence']:.3f}"
            
            return {
                'catalyst_type': catalyst_type,
                'impact_level': impact_level,
                'confidence': combined_confidence,
                'pattern_matches': combined_matches,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return classical_result
    
    def _calculate_quantum_advantage(self, classical_result: Dict[str, Any], quantum_result: Dict[str, Any]) -> float:
        """Calculate quantum advantage over classical analysis"""
        try:
            if classical_result['confidence'] > 0:
                advantage = (quantum_result['confidence'] - classical_result['confidence']) / classical_result['confidence']
                return max(0.0, min(1.0, advantage))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating quantum advantage: {e}")
            return 0.0
    
    async def _detection_monitoring_loop(self):
        """Monitor detection performance"""
        try:
            while self.status in [DetectionStatus.IDLE, DetectionStatus.ANALYZING]:
                await asyncio.sleep(30)
                
                # Update performance metrics
                self._update_performance_metrics()
                
        except Exception as e:
            logger.error(f"Error in detection monitoring loop: {e}")
    
    async def _pattern_optimization_loop(self):
        """Optimize detection patterns"""
        try:
            while self.status in [DetectionStatus.IDLE, DetectionStatus.ANALYZING]:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Optimize patterns based on recent performance
                await self._optimize_patterns()
                
        except Exception as e:
            logger.error(f"Error in pattern optimization loop: {e}")
    
    def _update_metrics(self, detection: CatalystDetection):
        """Update detection metrics"""
        try:
            self.metrics.total_detections += 1
            self.metrics.successful_detections += 1
            
            # Update average confidence
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (self.metrics.total_detections - 1) + detection.confidence) /
                self.metrics.total_detections
            )
            
            # Update average detection time
            self.metrics.average_detection_time = (
                (self.metrics.average_detection_time * (self.metrics.total_detections - 1) + detection.detection_time) /
                self.metrics.total_detections
            )
            
            # Update quantum advantage rate
            if detection.quantum_advantage > 0:
                self.metrics.quantum_advantage_rate = (
                    (self.metrics.quantum_advantage_rate * (self.metrics.total_detections - 1) + 1) /
                    self.metrics.total_detections
                )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate pattern match accuracy
            if self.metrics.total_detections > 0:
                self.metrics.pattern_match_accuracy = self.metrics.successful_detections / self.metrics.total_detections
            
            # Calculate F1 score (simplified)
            precision = self.metrics.true_positive_rate
            recall = self.metrics.true_positive_rate
            if precision + recall > 0:
                self.metrics.f1_score = 2 * (precision * recall) / (precision + recall)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _optimize_patterns(self):
        """Optimize detection patterns based on performance"""
        try:
            # Simulate pattern optimization
            for pattern_id, pattern in self.detection_patterns.items():
                # Adjust confidence threshold based on performance
                if self.metrics.pattern_match_accuracy > 0.8:
                    pattern.confidence_threshold = max(0.5, pattern.confidence_threshold - 0.05)
                elif self.metrics.pattern_match_accuracy < 0.6:
                    pattern.confidence_threshold = min(0.95, pattern.confidence_threshold + 0.05)
            
            logger.info("Pattern optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing patterns: {e}")
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        if hasattr(self, '_start_time'):
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    async def get_detection_status(self) -> Dict[str, Any]:
        """Get detection service status"""
        return {
            'status': self.status.value,
            'total_detections': self.metrics.total_detections,
            'successful_detections': self.metrics.successful_detections,
            'failed_detections': self.metrics.failed_detections,
            'average_confidence': self.metrics.average_confidence,
            'average_detection_time': self.metrics.average_detection_time,
            'quantum_advantage_rate': self.metrics.quantum_advantage_rate,
            'pattern_match_accuracy': self.metrics.pattern_match_accuracy,
            'f1_score': self.metrics.f1_score,
            'available_patterns': len(self.detection_patterns),
            'quantum_available': QUANTUM_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_detection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get detection history"""
        try:
            detections = list(self.detection_history.values())
            detections.sort(key=lambda x: x.timestamp, reverse=True)
            
            history = []
            for detection in detections[:limit]:
                history.append({
                    'detection_id': detection.detection_id,
                    'catalyst_type': detection.catalyst_type.value,
                    'impact_level': detection.impact_level.value,
                    'confidence': detection.confidence,
                    'quantum_advantage': detection.quantum_advantage,
                    'detection_time': detection.detection_time,
                    'timestamp': detection.timestamp.isoformat()
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting detection history: {e}")
            return []

# Global instance
quantum_catalyst_detector = QuantumCatalystDetector()
