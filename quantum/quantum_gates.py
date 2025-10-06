"""
Quantum Gates
=============
Enterprise-grade quantum gate operations and management
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Quantum computing imports with graceful fallback
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.circuit import Gate
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    qiskit = None

logger = logging.getLogger(__name__)

class GateType(Enum):
    """Quantum gate type categories"""
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    MULTI_QUBIT = "multi_qubit"
    PARAMETERIZED = "parameterized"
    CUSTOM = "custom"

class GateStatus(Enum):
    """Gate status levels"""
    DEFINED = "defined"
    VALIDATED = "validated"
    OPTIMIZED = "optimized"
    ERROR = "error"

@dataclass
class GateInfo:
    """Quantum gate information container"""
    gate_id: str
    name: str
    gate_type: GateType
    num_qubits: int
    parameters: List[str]
    matrix: Optional[np.ndarray]
    status: GateStatus
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GateMetrics:
    """Gate metrics"""
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    parameterized_gates: int
    custom_gates: int
    average_matrix_size: float

class QuantumGates:
    """Enterprise-grade quantum gate management service"""
    
    def __init__(self):
        self.gates: Dict[str, Gate] = {}
        self.gate_info: Dict[str, GateInfo] = {}
        self.gate_matrices: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.metrics = GateMetrics(
            total_gates=0, single_qubit_gates=0, two_qubit_gates=0,
            parameterized_gates=0, custom_gates=0, average_matrix_size=0.0
        )
        
        # Gate configuration
        self.config = {
            'enable_validation': True,
            'enable_optimization': True,
            'max_matrix_size': 16,
            'precision': 1e-10
        }
        
        # Initialize standard gates
        self._initialize_standard_gates()
        
        logger.info("Quantum Gates initialized")
    
    def _initialize_standard_gates(self):
        """Initialize standard quantum gates"""
        try:
            if not QUANTUM_AVAILABLE:
                return
            
            # Initialize standard gates
            self._create_pauli_gates()
            self._create_hadamard_gates()
            self._create_rotation_gates()
            self._create_phase_gates()
            self._create_entangling_gates()
            
            logger.info("Standard gates initialized")
            
        except Exception as e:
            logger.error(f"Error initializing standard gates: {e}")
    
    def _create_pauli_gates(self):
        """Create Pauli gates"""
        try:
            # Pauli-X gate
            self._register_gate('x', GateType.SINGLE_QUBIT, 1, np.array([[0, 1], [1, 0]]))
            
            # Pauli-Y gate
            self._register_gate('y', GateType.SINGLE_QUBIT, 1, np.array([[0, -1j], [1j, 0]]))
            
            # Pauli-Z gate
            self._register_gate('z', GateType.SINGLE_QUBIT, 1, np.array([[1, 0], [0, -1]]))
            
        except Exception as e:
            logger.error(f"Error creating Pauli gates: {e}")
    
    def _create_hadamard_gates(self):
        """Create Hadamard gates"""
        try:
            # Hadamard gate
            h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            self._register_gate('h', GateType.SINGLE_QUBIT, 1, h_matrix)
            
        except Exception as e:
            logger.error(f"Error creating Hadamard gates: {e}")
    
    def _create_rotation_gates(self):
        """Create rotation gates"""
        try:
            # RX gate (parameterized)
            self._register_parameterized_gate('rx', GateType.PARAMETERIZED, 1, ['theta'])
            
            # RY gate (parameterized)
            self._register_parameterized_gate('ry', GateType.PARAMETERIZED, 1, ['theta'])
            
            # RZ gate (parameterized)
            self._register_parameterized_gate('rz', GateType.PARAMETERIZED, 1, ['theta'])
            
        except Exception as e:
            logger.error(f"Error creating rotation gates: {e}")
    
    def _create_phase_gates(self):
        """Create phase gates"""
        try:
            # S gate
            s_matrix = np.array([[1, 0], [0, 1j]])
            self._register_gate('s', GateType.SINGLE_QUBIT, 1, s_matrix)
            
            # T gate
            t_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
            self._register_gate('t', GateType.SINGLE_QUBIT, 1, t_matrix)
            
            # Phase gate (parameterized)
            self._register_parameterized_gate('p', GateType.PARAMETERIZED, 1, ['lambda'])
            
        except Exception as e:
            logger.error(f"Error creating phase gates: {e}")
    
    def _create_entangling_gates(self):
        """Create entangling gates"""
        try:
            # CNOT gate
            cnot_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]])
            self._register_gate('cx', GateType.TWO_QUBIT, 2, cnot_matrix)
            
            # CZ gate
            cz_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, -1]])
            self._register_gate('cz', GateType.TWO_QUBIT, 2, cz_matrix)
            
            # SWAP gate
            swap_matrix = np.array([[1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]])
            self._register_gate('swap', GateType.TWO_QUBIT, 2, swap_matrix)
            
        except Exception as e:
            logger.error(f"Error creating entangling gates: {e}")
    
    def _register_gate(self, name: str, gate_type: GateType, num_qubits: int, matrix: np.ndarray):
        """Register a quantum gate"""
        try:
            gate_id = str(uuid.uuid4())
            
            # Create gate info
            gate_info = GateInfo(
                gate_id=gate_id,
                name=name,
                gate_type=gate_type,
                num_qubits=num_qubits,
                parameters=[],
                matrix=matrix,
                status=GateStatus.DEFINED,
                created_at=datetime.now()
            )
            
            # Store gate info
            self.gate_info[gate_id] = gate_info
            self.gate_matrices[name] = matrix
            
            # Update metrics
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error registering gate: {e}")
    
    def _register_parameterized_gate(self, name: str, gate_type: GateType, 
                                   num_qubits: int, parameters: List[str]):
        """Register a parameterized quantum gate"""
        try:
            gate_id = str(uuid.uuid4())
            
            # Create gate info
            gate_info = GateInfo(
                gate_id=gate_id,
                name=name,
                gate_type=gate_type,
                num_qubits=num_qubits,
                parameters=parameters,
                matrix=None,  # Will be calculated when parameters are provided
                status=GateStatus.DEFINED,
                created_at=datetime.now()
            )
            
            # Store gate info
            self.gate_info[gate_id] = gate_info
            
            # Update metrics
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error registering parameterized gate: {e}")
    
    async def create_custom_gate(self, name: str, matrix: np.ndarray, 
                                num_qubits: int) -> str:
        """Create a custom quantum gate"""
        try:
            gate_id = str(uuid.uuid4())
            
            # Validate matrix
            if not self._validate_matrix(matrix, num_qubits):
                raise ValueError("Invalid matrix for gate")
            
            # Create gate info
            gate_info = GateInfo(
                gate_id=gate_id,
                name=name,
                gate_type=GateType.CUSTOM,
                num_qubits=num_qubits,
                parameters=[],
                matrix=matrix,
                status=GateStatus.DEFINED,
                created_at=datetime.now(),
                metadata={'custom': True}
            )
            
            # Store gate info
            self.gate_info[gate_id] = gate_info
            self.gate_matrices[name] = matrix
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Custom gate created: {name}")
            return gate_id
            
        except Exception as e:
            logger.error(f"Error creating custom gate: {e}")
            raise
    
    async def get_gate_matrix(self, name: str, parameters: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        """Get gate matrix"""
        try:
            if name in self.gate_matrices:
                return self.gate_matrices[name]
            
            # For parameterized gates, calculate matrix
            if parameters:
                return await self._calculate_parameterized_matrix(name, parameters)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting gate matrix: {e}")
            return None
    
    async def _calculate_parameterized_matrix(self, name: str, parameters: Dict[str, float]) -> Optional[np.ndarray]:
        """Calculate matrix for parameterized gate"""
        try:
            if name == 'rx':
                theta = parameters.get('theta', 0)
                return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                               [-1j*np.sin(theta/2), np.cos(theta/2)]])
            
            elif name == 'ry':
                theta = parameters.get('theta', 0)
                return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                               [np.sin(theta/2), np.cos(theta/2)]])
            
            elif name == 'rz':
                theta = parameters.get('theta', 0)
                return np.array([[np.exp(-1j*theta/2), 0],
                               [0, np.exp(1j*theta/2)]])
            
            elif name == 'p':
                lam = parameters.get('lambda', 0)
                return np.array([[1, 0],
                               [0, np.exp(1j*lam)]])
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating parameterized matrix: {e}")
            return None
    
    def _validate_matrix(self, matrix: np.ndarray, num_qubits: int) -> bool:
        """Validate quantum gate matrix"""
        try:
            # Check dimensions
            expected_size = 2 ** num_qubits
            if matrix.shape != (expected_size, expected_size):
                return False
            
            # Check if matrix is unitary
            if not self._is_unitary(matrix):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False
    
    def _is_unitary(self, matrix: np.ndarray) -> bool:
        """Check if matrix is unitary"""
        try:
            # Check if U†U = I
            conjugate_transpose = np.conj(matrix.T)
            product = np.dot(conjugate_transpose, matrix)
            identity = np.eye(matrix.shape[0])
            
            return np.allclose(product, identity, atol=self.config['precision'])
            
        except Exception as e:
            logger.error(f"Error checking unitarity: {e}")
            return False
    
    async def compose_gates(self, gate_names: List[str], 
                          parameters: Optional[List[Dict[str, float]]] = None) -> np.ndarray:
        """Compose multiple gates"""
        try:
            if not gate_names:
                return np.eye(1)
            
            # Get matrices for all gates
            matrices = []
            for i, gate_name in enumerate(gate_names):
                gate_params = parameters[i] if parameters and i < len(parameters) else None
                matrix = await self.get_gate_matrix(gate_name, gate_params)
                if matrix is not None:
                    matrices.append(matrix)
            
            if not matrices:
                return np.eye(1)
            
            # Compose matrices
            result = matrices[0]
            for matrix in matrices[1:]:
                result = np.dot(result, matrix)
            
            return result
            
        except Exception as e:
            logger.error(f"Error composing gates: {e}")
            return np.eye(1)
    
    async def get_gate_info(self, name: str) -> Optional[GateInfo]:
        """Get gate information"""
        try:
            for gate_info in self.gate_info.values():
                if gate_info.name == name:
                    return gate_info
            return None
            
        except Exception as e:
            logger.error(f"Error getting gate info: {e}")
            return None
    
    async def list_gates(self, gate_type: Optional[GateType] = None) -> List[str]:
        """List available gates"""
        try:
            gates = []
            for gate_info in self.gate_info.values():
                if gate_type is None or gate_info.gate_type == gate_type:
                    gates.append(gate_info.name)
            return gates
            
        except Exception as e:
            logger.error(f"Error listing gates: {e}")
            return []
    
    def _update_metrics(self):
        """Update gate metrics"""
        try:
            self.metrics.total_gates = len(self.gate_info)
            self.metrics.single_qubit_gates = len([g for g in self.gate_info.values() if g.gate_type == GateType.SINGLE_QUBIT])
            self.metrics.two_qubit_gates = len([g for g in self.gate_info.values() if g.gate_type == GateType.TWO_QUBIT])
            self.metrics.parameterized_gates = len([g for g in self.gate_info.values() if g.gate_type == GateType.PARAMETERIZED])
            self.metrics.custom_gates = len([g for g in self.gate_info.values() if g.gate_type == GateType.CUSTOM])
            
            # Calculate average matrix size
            matrix_sizes = [g.matrix.size for g in self.gate_info.values() if g.matrix is not None]
            if matrix_sizes:
                self.metrics.average_matrix_size = sum(matrix_sizes) / len(matrix_sizes)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_gates_status(self) -> Dict[str, Any]:
        """Get gates status"""
        return {
            'total_gates': self.metrics.total_gates,
            'single_qubit_gates': self.metrics.single_qubit_gates,
            'two_qubit_gates': self.metrics.two_qubit_gates,
            'parameterized_gates': self.metrics.parameterized_gates,
            'custom_gates': self.metrics.custom_gates,
            'average_matrix_size': self.metrics.average_matrix_size,
            'config': self.config,
            'quantum_available': QUANTUM_AVAILABLE
        }

# Global instance
quantum_gates = QuantumGates()




