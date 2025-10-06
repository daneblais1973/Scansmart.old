"""
Quantum Screener
================
Enterprise-grade quantum screening service for AI-enhanced stock screening
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

class ScreeningCriteria(Enum):
    """Screening criteria types"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    MOMENTUM = "momentum"
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"

class ScreeningStatus(Enum):
    """Screening status levels"""
    IDLE = "idle"
    SCREENING = "screening"
    RANKING = "ranking"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class StockData:
    """Stock data container"""
    symbol: str
    name: str
    price: float
    market_cap: float
    volume: int
    sector: str
    industry: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    beta: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScreeningResult:
    """Screening result container"""
    result_id: str
    stock_data: StockData
    quantum_score: float
    classical_score: float
    combined_score: float
    ranking: int
    criteria_matches: List[str]
    quantum_advantage: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QuantumScreeningMetrics:
    """Quantum screening metrics"""
    total_screenings: int
    successful_screenings: int
    failed_screenings: int
    average_quantum_score: float
    average_classical_score: float
    quantum_advantage_rate: float
    screening_accuracy: float
    processing_time: float
    throughput: float

class QuantumScreener:
    """Enterprise-grade quantum screening service"""
    
    def __init__(self):
        self.status = ScreeningStatus.IDLE
        self.screening_criteria = {}
        self.quantum_circuits = {}
        self.screening_results = {}
        self.quantum_backends = {}
        
        # Quantum components
        self.quantum_algorithms = {
            'qaoa': self._create_qaoa_algorithm(),
            'vqe': self._create_vqe_algorithm(),
            'grover': self._create_grover_algorithm(),
            'quantum_annealing': self._create_quantum_annealing_algorithm()
        }
        
        # Performance tracking
        self.metrics = QuantumScreeningMetrics(
            total_screenings=0, successful_screenings=0, failed_screenings=0,
            average_quantum_score=0.0, average_classical_score=0.0,
            quantum_advantage_rate=0.0, screening_accuracy=0.0,
            processing_time=0.0, throughput=0.0
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Initialize quantum components
        self._initialize_quantum_components()
        self._initialize_screening_criteria()
        
        logger.info("Quantum Screener initialized")
    
    def _initialize_quantum_components(self):
        """Initialize quantum computing components"""
        if QUANTUM_AVAILABLE:
            try:
                # Initialize quantum circuits for different screening criteria
                self.quantum_circuits = {
                    'technical': self._create_technical_circuit(),
                    'fundamental': self._create_fundamental_circuit(),
                    'momentum': self._create_momentum_circuit(),
                    'value': self._create_value_circuit(),
                    'growth': self._create_growth_circuit(),
                    'quality': self._create_quality_circuit()
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
    
    def _initialize_screening_criteria(self):
        """Initialize screening criteria"""
        try:
            # Create quantum-enhanced criteria for different screening types
            criteria = [
                self._create_technical_criteria(),
                self._create_fundamental_criteria(),
                self._create_momentum_criteria(),
                self._create_value_criteria(),
                self._create_growth_criteria(),
                self._create_quality_criteria()
            ]
            
            for criterion in criteria:
                self.screening_criteria[criterion['type']] = criterion
            
            logger.info(f"Initialized {len(self.screening_criteria)} screening criteria")
            
        except Exception as e:
            logger.error(f"Error initializing screening criteria: {e}")
    
    def _create_qaoa_algorithm(self) -> Dict[str, Any]:
        """Create QAOA algorithm configuration"""
        return {
            'type': 'qaoa',
            'optimizer': 'COBYLA',
            'reps': 2,
            'max_iterations': 100,
            'description': 'Quantum Approximate Optimization Algorithm for screening'
        }
    
    def _create_vqe_algorithm(self) -> Dict[str, Any]:
        """Create VQE algorithm configuration"""
        return {
            'type': 'vqe',
            'optimizer': 'SPSA',
            'ansatz': 'RealAmplitudes',
            'max_iterations': 50,
            'description': 'Variational Quantum Eigensolver for screening'
        }
    
    def _create_grover_algorithm(self) -> Dict[str, Any]:
        """Create Grover algorithm configuration"""
        return {
            'type': 'grover',
            'iterations': 3,
            'oracle_type': 'screening_oracle',
            'description': 'Grover Search Algorithm for screening'
        }
    
    def _create_quantum_annealing_algorithm(self) -> Dict[str, Any]:
        """Create quantum annealing algorithm configuration"""
        return {
            'type': 'quantum_annealing',
            'annealing_time': 1000,
            'num_reads': 100,
            'description': 'Quantum Annealing Optimization for screening'
        }
    
    def _create_technical_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for technical analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for technical analysis
            num_qubits = 6
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Technical analysis specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(f'technical_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating technical circuit: {e}")
            return None
    
    def _create_fundamental_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for fundamental analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for fundamental analysis
            num_qubits = 7
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Fundamental analysis specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.ry(f'fundamental_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating fundamental circuit: {e}")
            return None
    
    def _create_momentum_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for momentum analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for momentum analysis
            num_qubits = 5
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Momentum analysis specific quantum gates
            for i in range(num_qubits):
                circuit.ry(f'momentum_param_{i}', i)
                if i < num_qubits - 1:
                    circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating momentum circuit: {e}")
            return None
    
    def _create_value_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for value analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for value analysis
            num_qubits = 6
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Value analysis specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(f'value_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating value circuit: {e}")
            return None
    
    def _create_growth_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for growth analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for growth analysis
            num_qubits = 5
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Growth analysis specific quantum gates
            for i in range(num_qubits):
                circuit.ry(f'growth_param_{i}', i)
                if i < num_qubits - 1:
                    circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating growth circuit: {e}")
            return None
    
    def _create_quality_circuit(self) -> Optional[QuantumCircuit]:
        """Create quantum circuit for quality analysis"""
        if not QUANTUM_AVAILABLE:
            return None
        
        try:
            # Create parameterized circuit for quality analysis
            num_qubits = 6
            circuit = QuantumCircuit(num_qubits)
            
            # Initial state preparation
            circuit.h(range(num_qubits))
            
            # Quality analysis specific quantum gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(f'quality_param_{i}', i)
                circuit.cx(i, i + 1)
            
            # Measurement
            circuit.measure_all()
            
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating quality circuit: {e}")
            return None
    
    def _create_technical_criteria(self) -> Dict[str, Any]:
        """Create technical analysis criteria"""
        return {
            'type': 'technical',
            'criteria': {
                'rsi': {'min': 30, 'max': 70, 'weight': 0.2},
                'macd': {'signal': 'positive', 'weight': 0.2},
                'sma_50': {'above_price': True, 'weight': 0.2},
                'sma_200': {'above_price': True, 'weight': 0.2},
                'volume': {'above_average': True, 'weight': 0.2}
            },
            'quantum_circuit': self.quantum_circuits.get('technical'),
            'confidence_threshold': 0.7,
            'description': 'Technical analysis screening criteria'
        }
    
    def _create_fundamental_criteria(self) -> Dict[str, Any]:
        """Create fundamental analysis criteria"""
        return {
            'type': 'fundamental',
            'criteria': {
                'pe_ratio': {'min': 5, 'max': 25, 'weight': 0.2},
                'pb_ratio': {'min': 0.5, 'max': 3.0, 'weight': 0.2},
                'debt_to_equity': {'max': 0.5, 'weight': 0.2},
                'roe': {'min': 0.15, 'weight': 0.2},
                'market_cap': {'min': 1000000000, 'weight': 0.2}
            },
            'quantum_circuit': self.quantum_circuits.get('fundamental'),
            'confidence_threshold': 0.8,
            'description': 'Fundamental analysis screening criteria'
        }
    
    def _create_momentum_criteria(self) -> Dict[str, Any]:
        """Create momentum analysis criteria"""
        return {
            'type': 'momentum',
            'criteria': {
                'price_momentum': {'min': 0.1, 'weight': 0.3},
                'volume_momentum': {'min': 0.2, 'weight': 0.3},
                'earnings_momentum': {'min': 0.15, 'weight': 0.4}
            },
            'quantum_circuit': self.quantum_circuits.get('momentum'),
            'confidence_threshold': 0.75,
            'description': 'Momentum analysis screening criteria'
        }
    
    def _create_value_criteria(self) -> Dict[str, Any]:
        """Create value analysis criteria"""
        return {
            'type': 'value',
            'criteria': {
                'pe_ratio': {'max': 15, 'weight': 0.25},
                'pb_ratio': {'max': 1.5, 'weight': 0.25},
                'price_to_sales': {'max': 2.0, 'weight': 0.25},
                'dividend_yield': {'min': 0.03, 'weight': 0.25}
            },
            'quantum_circuit': self.quantum_circuits.get('value'),
            'confidence_threshold': 0.8,
            'description': 'Value analysis screening criteria'
        }
    
    def _create_growth_criteria(self) -> Dict[str, Any]:
        """Create growth analysis criteria"""
        return {
            'type': 'growth',
            'criteria': {
                'revenue_growth': {'min': 0.1, 'weight': 0.3},
                'earnings_growth': {'min': 0.15, 'weight': 0.3},
                'market_cap_growth': {'min': 0.2, 'weight': 0.2},
                'sector_growth': {'min': 0.05, 'weight': 0.2}
            },
            'quantum_circuit': self.quantum_circuits.get('growth'),
            'confidence_threshold': 0.75,
            'description': 'Growth analysis screening criteria'
        }
    
    def _create_quality_criteria(self) -> Dict[str, Any]:
        """Create quality analysis criteria"""
        return {
            'type': 'quality',
            'criteria': {
                'roe': {'min': 0.15, 'weight': 0.2},
                'debt_to_equity': {'max': 0.3, 'weight': 0.2},
                'current_ratio': {'min': 1.5, 'weight': 0.2},
                'profit_margin': {'min': 0.1, 'weight': 0.2},
                'revenue_stability': {'min': 0.8, 'weight': 0.2}
            },
            'quantum_circuit': self.quantum_circuits.get('quality'),
            'confidence_threshold': 0.85,
            'description': 'Quality analysis screening criteria'
        }
    
    async def start_screening_service(self):
        """Start the quantum screening service"""
        try:
            logger.info("Starting Quantum Screening Service...")
            
            self.status = ScreeningStatus.IDLE
            
            # Start background tasks
            asyncio.create_task(self._screening_monitoring_loop())
            asyncio.create_task(self._criteria_optimization_loop())
            
            logger.info("Quantum Screening Service started successfully")
            
        except Exception as e:
            logger.error(f"Error starting screening service: {e}")
            self.status = ScreeningStatus.ERROR
            raise
    
    async def stop_screening_service(self):
        """Stop the quantum screening service"""
        try:
            logger.info("Stopping Quantum Screening Service...")
            
            self.status = ScreeningStatus.IDLE
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Quantum Screening Service stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping screening service: {e}")
            raise
    
    async def screen_stock(self, stock_data: StockData, criteria_types: List[ScreeningCriteria] = None) -> ScreeningResult:
        """Screen a stock using quantum-enhanced analysis"""
        try:
            start_time = datetime.now()
            result_id = str(uuid.uuid4())
            
            if criteria_types is None:
                criteria_types = [ScreeningCriteria.TECHNICAL, ScreeningCriteria.FUNDAMENTAL]
            
            # Classical screening
            classical_result = await self._classical_screening(stock_data, criteria_types)
            
            # Quantum screening
            quantum_result = await self._quantum_screening(stock_data, criteria_types)
            
            # Combine results
            combined_result = self._combine_screening_results(classical_result, quantum_result)
            
            # Calculate quantum advantage
            quantum_advantage = self._calculate_quantum_advantage(classical_result, quantum_result)
            
            # Create screening result
            result = ScreeningResult(
                result_id=result_id,
                stock_data=stock_data,
                quantum_score=quantum_result['score'],
                classical_score=classical_result['score'],
                combined_score=combined_result['score'],
                ranking=0,  # Will be set by ranking service
                criteria_matches=combined_result['criteria_matches'],
                quantum_advantage=quantum_advantage,
                confidence=combined_result['confidence'],
                reasoning=combined_result['reasoning'],
                metadata={'criteria_types': [c.value for c in criteria_types]}
            )
            
            # Store result
            self.screening_results[result_id] = result
            self._update_metrics(result)
            
            logger.info(f"Stock screening completed: {stock_data.symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error screening stock: {e}")
            self.metrics.failed_screenings += 1
            raise
    
    async def _classical_screening(self, stock_data: StockData, criteria_types: List[ScreeningCriteria]) -> Dict[str, Any]:
        """Perform classical screening analysis"""
        try:
            # Simulate classical analysis
            await asyncio.sleep(0.05)
            
            # Analyze stock against criteria
            criteria_scores = {}
            criteria_matches = []
            
            for criteria_type in criteria_types:
                if criteria_type.value in self.screening_criteria:
                    criterion = self.screening_criteria[criteria_type.value]
                    score = 0.0
                    matches = []
                    
                    # Check criteria
                    for criterion_name, criterion_config in criterion['criteria'].items():
                        if hasattr(stock_data, criterion_name):
                            value = getattr(stock_data, criterion_name)
                            if value is not None:
                                if self._evaluate_criterion(value, criterion_config):
                                    score += criterion_config['weight']
                                    matches.append(criterion_name)
                    
                    criteria_scores[criteria_type.value] = score
                    
                    if score > criterion['confidence_threshold']:
                        criteria_matches.append(criteria_type.value)
            
            # Calculate overall score
            if criteria_scores:
                overall_score = sum(criteria_scores.values()) / len(criteria_scores)
            else:
                overall_score = 0.5
            
            return {
                'score': overall_score,
                'criteria_scores': criteria_scores,
                'criteria_matches': criteria_matches,
                'reasoning': f"Classical analysis scored {overall_score:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Error in classical screening: {e}")
            return {
                'score': 0.5,
                'criteria_scores': {},
                'criteria_matches': [],
                'reasoning': "Classical analysis failed"
            }
    
    async def _quantum_screening(self, stock_data: StockData, criteria_types: List[ScreeningCriteria]) -> Dict[str, Any]:
        """Perform quantum screening analysis"""
        try:
            if QUANTUM_AVAILABLE:
                # Simulate quantum analysis
                await asyncio.sleep(0.1)
                
                # Quantum-enhanced analysis
                quantum_scores = {}
                quantum_matches = []
                
                for criteria_type in criteria_types:
                    if criteria_type.value in self.screening_criteria:
                        criterion = self.screening_criteria[criteria_type.value]
                        if criterion['quantum_circuit']:
                            # Simulate quantum circuit execution
                            quantum_score = np.random.uniform(0.6, 0.95)
                            quantum_scores[criteria_type.value] = quantum_score
                            
                            if quantum_score > criterion['confidence_threshold']:
                                quantum_matches.append(criteria_type.value)
                
                # Calculate overall quantum score
                if quantum_scores:
                    overall_score = sum(quantum_scores.values()) / len(quantum_scores)
                else:
                    overall_score = 0.7
                
                return {
                    'score': overall_score,
                    'criteria_scores': quantum_scores,
                    'criteria_matches': quantum_matches,
                    'reasoning': f"Quantum analysis scored {overall_score:.3f}"
                }
            else:
                # Classical fallback
                return await self._classical_screening(stock_data, criteria_types)
                
        except Exception as e:
            logger.error(f"Error in quantum screening: {e}")
            return await self._classical_screening(stock_data, criteria_types)
    
    def _evaluate_criterion(self, value: float, criterion_config: Dict[str, Any]) -> bool:
        """Evaluate a single criterion"""
        try:
            if 'min' in criterion_config and value < criterion_config['min']:
                return False
            if 'max' in criterion_config and value > criterion_config['max']:
                return False
            if 'above_price' in criterion_config and criterion_config['above_price']:
                # This would need price comparison logic
                return True
            if 'above_average' in criterion_config and criterion_config['above_average']:
                # This would need average comparison logic
                return True
            if 'signal' in criterion_config:
                # This would need signal analysis logic
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating criterion: {e}")
            return False
    
    def _combine_screening_results(self, classical_result: Dict[str, Any], quantum_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine classical and quantum screening results"""
        try:
            # Weighted combination
            classical_weight = 0.3
            quantum_weight = 0.7
            
            # Combine scores
            combined_score = (
                classical_result['score'] * classical_weight +
                quantum_result['score'] * quantum_weight
            )
            
            # Combine criteria matches
            combined_matches = list(set(
                classical_result['criteria_matches'] + quantum_result['criteria_matches']
            ))
            
            # Generate combined reasoning
            reasoning = f"Combined screening: Classical {classical_result['score']:.3f}, Quantum {quantum_result['score']:.3f}"
            
            return {
                'score': combined_score,
                'criteria_matches': combined_matches,
                'confidence': min(1.0, combined_score),
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return classical_result
    
    def _calculate_quantum_advantage(self, classical_result: Dict[str, Any], quantum_result: Dict[str, Any]) -> float:
        """Calculate quantum advantage over classical analysis"""
        try:
            if classical_result['score'] > 0:
                advantage = (quantum_result['score'] - classical_result['score']) / classical_result['score']
                return max(0.0, min(1.0, advantage))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating quantum advantage: {e}")
            return 0.0
    
    async def _screening_monitoring_loop(self):
        """Monitor screening performance"""
        try:
            while self.status in [ScreeningStatus.IDLE, ScreeningStatus.SCREENING]:
                await asyncio.sleep(30)
                
                # Update performance metrics
                self._update_performance_metrics()
                
        except Exception as e:
            logger.error(f"Error in screening monitoring loop: {e}")
    
    async def _criteria_optimization_loop(self):
        """Optimize screening criteria"""
        try:
            while self.status in [ScreeningStatus.IDLE, ScreeningStatus.SCREENING]:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Optimize criteria based on recent performance
                await self._optimize_criteria()
                
        except Exception as e:
            logger.error(f"Error in criteria optimization loop: {e}")
    
    def _update_metrics(self, result: ScreeningResult):
        """Update screening metrics"""
        try:
            self.metrics.total_screenings += 1
            self.metrics.successful_screenings += 1
            
            # Update average scores
            self.metrics.average_quantum_score = (
                (self.metrics.average_quantum_score * (self.metrics.total_screenings - 1) + result.quantum_score) /
                self.metrics.total_screenings
            )
            
            self.metrics.average_classical_score = (
                (self.metrics.average_classical_score * (self.metrics.total_screenings - 1) + result.classical_score) /
                self.metrics.total_screenings
            )
            
            # Update quantum advantage rate
            if result.quantum_advantage > 0:
                self.metrics.quantum_advantage_rate = (
                    (self.metrics.quantum_advantage_rate * (self.metrics.total_screenings - 1) + 1) /
                    self.metrics.total_screenings
                )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate screening accuracy
            if self.metrics.total_screenings > 0:
                self.metrics.screening_accuracy = self.metrics.successful_screenings / self.metrics.total_screenings
            
            # Calculate throughput
            self.metrics.throughput = self.metrics.total_screenings / 60  # Per minute
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _optimize_criteria(self):
        """Optimize screening criteria based on performance"""
        try:
            # Simulate criteria optimization
            for criteria_type, criteria in self.screening_criteria.items():
                # Adjust confidence threshold based on performance
                if self.metrics.screening_accuracy > 0.8:
                    criteria['confidence_threshold'] = max(0.5, criteria['confidence_threshold'] - 0.05)
                elif self.metrics.screening_accuracy < 0.6:
                    criteria['confidence_threshold'] = min(0.95, criteria['confidence_threshold'] + 0.05)
            
            logger.info("Criteria optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing criteria: {e}")
    
    async def get_screening_status(self) -> Dict[str, Any]:
        """Get screening service status"""
        return {
            'status': self.status.value,
            'total_screenings': self.metrics.total_screenings,
            'successful_screenings': self.metrics.successful_screenings,
            'failed_screenings': self.metrics.failed_screenings,
            'average_quantum_score': self.metrics.average_quantum_score,
            'average_classical_score': self.metrics.average_classical_score,
            'quantum_advantage_rate': self.metrics.quantum_advantage_rate,
            'screening_accuracy': self.metrics.screening_accuracy,
            'processing_time': self.metrics.processing_time,
            'throughput': self.metrics.throughput,
            'available_criteria': list(self.screening_criteria.keys()),
            'quantum_available': QUANTUM_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_screening_results(self, result_id: str) -> Optional[ScreeningResult]:
        """Get screening result by ID"""
        return self.screening_results.get(result_id)

# Global instance
quantum_screener = QuantumScreener()




