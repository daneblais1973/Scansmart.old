"""
Quantum State Value Object
Advanced quantum state representation with superposition, entanglement, and coherence
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.linalg import norm
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass(frozen=True)
class QuantumState:
    """
    Immutable value object representing quantum states
    Supports superposition, entanglement, and quantum operations
    """
    
    # Core quantum properties
    amplitudes: np.ndarray  # Complex amplitudes for superposition
    basis_states: List[str]  # Basis state labels
    entanglement_qubits: List[int] = None  # Entangled qubit indices
    coherence_time: Optional[float] = None  # Coherence time in nanoseconds
    
    # Quantum metrics
    fidelity: Optional[float] = None  # State fidelity
    purity: Optional[float] = None    # State purity
    von_neumann_entropy: Optional[float] = None  # Quantum entropy
    
    # Metadata
    creation_time: Optional[float] = None  # Creation timestamp
    quantum_gates_applied: List[str] = None  # Applied quantum gates
    
    def __post_init__(self):
        """Validate quantum state invariants"""
        if not isinstance(self.amplitudes, np.ndarray):
            object.__setattr__(self, 'amplitudes', np.array(self.amplitudes))
        
        if not isinstance(self.basis_states, list):
            object.__setattr__(self, 'basis_states', list(self.basis_states))
        
        # Validate amplitudes
        if len(self.amplitudes) != len(self.basis_states):
            raise ValueError("Number of amplitudes must match number of basis states")
        
        if not np.allclose(np.sum(np.abs(self.amplitudes) ** 2), 1.0, atol=1e-6):
            raise ValueError("Quantum state must be normalized (amplitudes must sum to 1)")
        
        # Validate entanglement qubits
        if self.entanglement_qubits is None:
            object.__setattr__(self, 'entanglement_qubits', [])
        
        # Calculate quantum metrics
        self._calculate_quantum_metrics()
    
    def _calculate_quantum_metrics(self):
        """Calculate quantum state metrics"""
        # Calculate purity
        density_matrix = np.outer(self.amplitudes, np.conj(self.amplitudes))
        purity = np.trace(density_matrix @ density_matrix).real
        object.__setattr__(self, 'purity', purity)
        
        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvals(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical noise
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        object.__setattr__(self, 'von_neumann_entropy', entropy)
    
    def __str__(self) -> str:
        """String representation of quantum state"""
        state_str = "|ψ⟩ = "
        terms = []
        for i, (amp, basis) in enumerate(zip(self.amplitudes, self.basis_states)):
            if abs(amp) > 1e-6:  # Only show significant terms
                terms.append(f"{amp:.3f}|{basis}⟩")
        return state_str + " + ".join(terms)
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"QuantumState(amplitudes={self.amplitudes}, basis_states={self.basis_states})"
    
    @classmethod
    def from_computational_basis(cls, state_index: int, num_qubits: int) -> 'QuantumState':
        """Create quantum state from computational basis"""
        amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        amplitudes[state_index] = 1.0
        
        basis_states = [format(i, f'0{num_qubits}b') for i in range(2 ** num_qubits)]
        
        return cls(
            amplitudes=amplitudes,
            basis_states=basis_states,
            entanglement_qubits=[],
            coherence_time=None,
            fidelity=1.0,
            purity=1.0,
            von_neumann_entropy=0.0,
            creation_time=None,
            quantum_gates_applied=[]
        )
    
    @classmethod
    def from_superposition(cls, coefficients: List[complex], basis_labels: List[str]) -> 'QuantumState':
        """Create quantum state from superposition coefficients"""
        amplitudes = np.array(coefficients, dtype=complex)
        
        # Normalize amplitudes
        norm_factor = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm_factor > 1e-10:
            amplitudes = amplitudes / norm_factor
        
        return cls(
            amplitudes=amplitudes,
            basis_states=basis_labels,
            entanglement_qubits=[],
            coherence_time=None,
            fidelity=None,
            purity=None,
            von_neumann_entropy=None,
            creation_time=None,
            quantum_gates_applied=[]
        )
    
    @classmethod
    def bell_state(cls, qubit_pair: Tuple[int, int]) -> 'QuantumState':
        """Create Bell state (maximally entangled state)"""
        amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        basis_states = ['00', '01', '10', '11']
        
        return cls(
            amplitudes=amplitudes,
            basis_states=basis_states,
            entanglement_qubits=list(qubit_pair),
            coherence_time=None,
            fidelity=1.0,
            purity=0.5,  # Bell states are mixed
            von_neumann_entropy=1.0,  # Maximum entanglement
            creation_time=None,
            quantum_gates_applied=['H', 'CNOT']
        )
    
    @classmethod
    def ghz_state(cls, num_qubits: int) -> 'QuantumState':
        """Create GHZ state (Greenberger-Horne-Zeilinger state)"""
        amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        amplitudes[0] = 1/np.sqrt(2)
        amplitudes[-1] = 1/np.sqrt(2)
        
        basis_states = [format(i, f'0{num_qubits}b') for i in range(2 ** num_qubits)]
        
        return cls(
            amplitudes=amplitudes,
            basis_states=basis_states,
            entanglement_qubits=list(range(num_qubits)),
            coherence_time=None,
            fidelity=1.0,
            purity=0.5,
            von_neumann_entropy=1.0,
            creation_time=None,
            quantum_gates_applied=['H'] + ['CNOT'] * (num_qubits - 1)
        )
    
    def measure(self, basis: str = "computational") -> Tuple[str, 'QuantumState']:
        """Measure quantum state and return outcome and collapsed state"""
        if basis != "computational":
            raise ValueError("Only computational basis measurement supported")
        
        # Calculate measurement probabilities
        probabilities = np.abs(self.amplitudes) ** 2
        
        # Sample measurement outcome
        outcome_index = np.random.choice(len(probabilities), p=probabilities)
        outcome = self.basis_states[outcome_index]
        
        # Create collapsed state
        collapsed_amplitudes = np.zeros_like(self.amplitudes)
        collapsed_amplitudes[outcome_index] = 1.0
        
        collapsed_state = QuantumState(
            amplitudes=collapsed_amplitudes,
            basis_states=self.basis_states,
            entanglement_qubits=[],
            coherence_time=0.0,  # Decoherence after measurement
            fidelity=1.0,
            purity=1.0,
            von_neumann_entropy=0.0,
            creation_time=None,
            quantum_gates_applied=self.quantum_gates_applied + ['MEASURE']
        )
        
        return outcome, collapsed_state
    
    def apply_gate(self, gate_name: str, target_qubits: List[int]) -> 'QuantumState':
        """Apply quantum gate to state"""
        # This is a simplified implementation
        # In practice, this would involve matrix operations
        
        new_gates = (self.quantum_gates_applied or []) + [gate_name]
        
        return QuantumState(
            amplitudes=self.amplitudes,  # Simplified - would apply actual gate
            basis_states=self.basis_states,
            entanglement_qubits=self.entanglement_qubits,
            coherence_time=self.coherence_time,
            fidelity=self.fidelity,
            purity=self.purity,
            von_neumann_entropy=self.von_neumann_entropy,
            creation_time=self.creation_time,
            quantum_gates_applied=new_gates
        )
    
    def entangle_with(self, other_state: 'QuantumState') -> 'QuantumState':
        """Create entangled state with another quantum state"""
        # Tensor product of states
        combined_amplitudes = np.kron(self.amplitudes, other_state.amplitudes)
        
        # Combine basis states
        combined_basis = []
        for basis1 in self.basis_states:
            for basis2 in other_state.basis_states:
                combined_basis.append(basis1 + basis2)
        
        # Combine entanglement qubits
        combined_entanglement = (self.entanglement_qubits or []) + (other_state.entanglement_qubits or [])
        
        return QuantumState(
            amplitudes=combined_amplitudes,
            basis_states=combined_basis,
            entanglement_qubits=combined_entanglement,
            coherence_time=min(
                self.coherence_time or float('inf'),
                other_state.coherence_time or float('inf')
            ),
            fidelity=None,  # Would need to calculate
            purity=None,    # Would need to calculate
            von_neumann_entropy=None,  # Would need to calculate
            creation_time=None,
            quantum_gates_applied=(self.quantum_gates_applied or []) + (other_state.quantum_gates_applied or [])
        )
    
    def calculate_fidelity_with(self, target_state: 'QuantumState') -> float:
        """Calculate fidelity with target state"""
        if len(self.amplitudes) != len(target_state.amplitudes):
            return 0.0
        
        # Fidelity = |⟨ψ|φ⟩|²
        overlap = np.dot(np.conj(self.amplitudes), target_state.amplitudes)
        fidelity = np.abs(overlap) ** 2
        
        return float(fidelity)
    
    def is_entangled(self) -> bool:
        """Check if state is entangled"""
        return len(self.entanglement_qubits or []) > 0
    
    def is_pure(self) -> bool:
        """Check if state is pure"""
        return self.purity is not None and self.purity > 0.99
    
    def is_mixed(self) -> bool:
        """Check if state is mixed"""
        return not self.is_pure()
    
    def get_entanglement_entropy(self) -> float:
        """Get entanglement entropy"""
        if self.von_neumann_entropy is not None:
            return self.von_neumann_entropy
        
        # Calculate if not already computed
        density_matrix = np.outer(self.amplitudes, np.conj(self.amplitudes))
        eigenvalues = np.linalg.eigvals(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        return float(entropy)
    
    def to_density_matrix(self) -> np.ndarray:
        """Convert to density matrix representation"""
        return np.outer(self.amplitudes, np.conj(self.amplitudes))
    
    def to_json(self) -> str:
        """Convert to JSON representation"""
        state_dict = {
            'amplitudes': self.amplitudes.tolist(),
            'basis_states': self.basis_states,
            'entanglement_qubits': self.entanglement_qubits or [],
            'coherence_time': self.coherence_time,
            'fidelity': self.fidelity,
            'purity': self.purity,
            'von_neumann_entropy': self.von_neumann_entropy,
            'creation_time': self.creation_time,
            'quantum_gates_applied': self.quantum_gates_applied or []
        }
        return json.dumps(state_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QuantumState':
        """Create from JSON representation"""
        state_dict = json.loads(json_str)
        
        return cls(
            amplitudes=np.array(state_dict['amplitudes'], dtype=complex),
            basis_states=state_dict['basis_states'],
            entanglement_qubits=state_dict.get('entanglement_qubits', []),
            coherence_time=state_dict.get('coherence_time'),
            fidelity=state_dict.get('fidelity'),
            purity=state_dict.get('purity'),
            von_neumann_entropy=state_dict.get('von_neumann_entropy'),
            creation_time=state_dict.get('creation_time'),
            quantum_gates_applied=state_dict.get('quantum_gates_applied', [])
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, QuantumState):
            return False
        
        return (np.allclose(self.amplitudes, other.amplitudes) and
                self.basis_states == other.basis_states and
                (self.entanglement_qubits or []) == (other.entanglement_qubits or []))
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((tuple(self.amplitudes), tuple(self.basis_states)))

