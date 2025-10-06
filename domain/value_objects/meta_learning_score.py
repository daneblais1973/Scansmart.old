"""
Meta-Learning Score Value Object
Advanced meta-learning metrics with few-shot learning, transfer learning, and continual learning
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import entropy, beta
import json
from enum import Enum


class MetaLearningType(Enum):
    """Types of meta-learning approaches"""
    MAML = "maml"                    # Model-Agnostic Meta-Learning
    PROTO_NET = "proto_net"          # Prototypical Networks
    META_LSTM = "meta_lstm"          # Meta-LSTM
    META_GRADIENT = "meta_gradient"  # Meta-Gradient
    FEW_SHOT = "few_shot"            # Few-shot Learning
    TRANSFER = "transfer"            # Transfer Learning
    CONTINUAL = "continual"          # Continual Learning
    CONTRASTIVE_LEARNING = "contrastive_learning"  # Contrastive Learning


class AdaptationSpeed(Enum):
    """Meta-learning adaptation speed levels"""
    VERY_SLOW = "very_slow"      # > 100 iterations
    SLOW = "slow"                # 50 - 100 iterations
    MODERATE = "moderate"        # 20 - 50 iterations
    FAST = "fast"                # 5 - 20 iterations
    VERY_FAST = "very_fast"      # 1 - 5 iterations
    INSTANT = "instant"          # 1 iteration


@dataclass(frozen=True)
class MetaLearningScore:
    """
    Immutable value object representing meta-learning performance
    Incorporates adaptation speed, generalization, and knowledge transfer
    """
    
    # Core meta-learning metrics
    value: Decimal
    adaptation_speed: Decimal
    generalization_score: Decimal
    transfer_effectiveness: Decimal
    
    # Meta-learning properties
    meta_learning_type: MetaLearningType
    adaptation_speed_level: AdaptationSpeed
    num_tasks: int
    num_shots: int
    
    # Performance metrics
    base_accuracy: Decimal
    adapted_accuracy: Decimal
    improvement: Decimal
    
    # Advanced metrics
    task_diversity: Optional[Decimal] = None
    domain_shift: Optional[Decimal] = None
    catastrophic_forgetting: Optional[Decimal] = None
    
    # Temporal properties
    learning_curve: Optional[List[Decimal]] = None
    adaptation_time: Optional[Decimal] = None
    
    # Knowledge transfer
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    transfer_distance: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate meta-learning score invariants"""
        # Validate core values
        for value_name, value in [
            ('value', self.value),
            ('adaptation_speed', self.adaptation_speed),
            ('generalization_score', self.generalization_score),
            ('transfer_effectiveness', self.transfer_effectiveness)
        ]:
            if not (0 <= value <= 1):
                raise ValueError(f"{value_name} must be between 0 and 1")
        
        # Validate task parameters
        if self.num_tasks < 1:
            raise ValueError("Number of tasks must be at least 1")
        
        if self.num_shots < 1:
            raise ValueError("Number of shots must be at least 1")
        
        # Validate accuracy values
        for acc_name, acc_value in [
            ('base_accuracy', self.base_accuracy),
            ('adapted_accuracy', self.adapted_accuracy)
        ]:
            if not (0 <= acc_value <= 1):
                raise ValueError(f"{acc_name} must be between 0 and 1")
        
        # Validate improvement calculation
        expected_improvement = self.adapted_accuracy - self.base_accuracy
        if not abs(self.improvement - expected_improvement) < Decimal('0.01'):
            raise ValueError("Improvement should equal adapted_accuracy - base_accuracy")
        
        # Round to 4 decimal places
        rounded_value = self.value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_speed = self.adaptation_speed.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_gen = self.generalization_score.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_transfer = self.transfer_effectiveness.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        
        object.__setattr__(self, 'value', rounded_value)
        object.__setattr__(self, 'adaptation_speed', rounded_speed)
        object.__setattr__(self, 'generalization_score', rounded_gen)
        object.__setattr__(self, 'transfer_effectiveness', rounded_transfer)
    
    def __str__(self) -> str:
        """String representation"""
        return f"MetaLearningScore(value={self.value:.4f}, speed={self.adaptation_speed:.4f}, gen={self.generalization_score:.4f})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"MetaLearningScore(value={self.value}, speed={self.adaptation_speed}, gen={self.generalization_score})"
    
    @classmethod
    def from_adaptation_curve(cls, base_accuracy: float, adapted_accuracy: float, 
                            adaptation_iterations: int, meta_learning_type: MetaLearningType,
                            num_tasks: int, num_shots: int) -> 'MetaLearningScore':
        """Create meta-learning score from adaptation curve"""
        
        # Calculate improvement
        improvement = adapted_accuracy - base_accuracy
        
        # Calculate adaptation speed (inverse of iterations)
        max_iterations = 100  # Normalize to max iterations
        adaptation_speed = max(0, 1 - (adaptation_iterations / max_iterations))
        
        # Calculate generalization score (based on improvement)
        generalization_score = max(0, min(1, improvement + 0.5))
        
        # Calculate transfer effectiveness (simplified)
        transfer_effectiveness = min(1, adapted_accuracy)
        
        # Calculate overall meta-learning score
        meta_score = (adaptation_speed + generalization_score + transfer_effectiveness) / 3
        
        # Determine adaptation speed level
        speed_level = cls._determine_adaptation_speed(adaptation_iterations)
        
        return cls(
            value=Decimal(str(meta_score)),
            adaptation_speed=Decimal(str(adaptation_speed)),
            generalization_score=Decimal(str(generalization_score)),
            transfer_effectiveness=Decimal(str(transfer_effectiveness)),
            meta_learning_type=meta_learning_type,
            adaptation_speed_level=speed_level,
            num_tasks=num_tasks,
            num_shots=num_shots,
            base_accuracy=Decimal(str(base_accuracy)),
            adapted_accuracy=Decimal(str(adapted_accuracy)),
            improvement=Decimal(str(improvement))
        )
    
    @classmethod
    def from_few_shot_learning(cls, support_accuracy: float, query_accuracy: float,
                             num_shots: int, num_classes: int) -> 'MetaLearningScore':
        """Create meta-learning score from few-shot learning results"""
        
        # Calculate adaptation speed based on number of shots
        adaptation_speed = max(0, 1 - (num_shots / 100))  # Fewer shots = faster adaptation
        
        # Calculate generalization score
        generalization_score = query_accuracy
        
        # Calculate transfer effectiveness
        transfer_effectiveness = support_accuracy
        
        # Calculate overall score
        meta_score = (adaptation_speed + generalization_score + transfer_effectiveness) / 3
        
        # Determine adaptation speed level
        speed_level = cls._determine_adaptation_speed(num_shots)
        
        return cls(
            value=Decimal(str(meta_score)),
            adaptation_speed=Decimal(str(adaptation_speed)),
            generalization_score=Decimal(str(generalization_score)),
            transfer_effectiveness=Decimal(str(transfer_effectiveness)),
            meta_learning_type=MetaLearningType.FEW_SHOT,
            adaptation_speed_level=speed_level,
            num_tasks=1,
            num_shots=num_shots,
            base_accuracy=Decimal(str(support_accuracy)),
            adapted_accuracy=Decimal(str(query_accuracy)),
            improvement=Decimal(str(query_accuracy - support_accuracy))
        )
    
    @classmethod
    def from_transfer_learning(cls, source_accuracy: float, target_accuracy: float,
                             source_domain: str, target_domain: str,
                             adaptation_time: float) -> 'MetaLearningScore':
        """Create meta-learning score from transfer learning results"""
        
        # Calculate adaptation speed based on time
        max_time = 3600  # 1 hour in seconds
        adaptation_speed = max(0, 1 - (adaptation_time / max_time))
        
        # Calculate generalization score
        generalization_score = target_accuracy
        
        # Calculate transfer effectiveness
        transfer_effectiveness = min(1, target_accuracy / source_accuracy) if source_accuracy > 0 else 0
        
        # Calculate overall score
        meta_score = (adaptation_speed + generalization_score + transfer_effectiveness) / 3
        
        # Determine adaptation speed level
        speed_level = cls._determine_adaptation_speed(int(adaptation_time))
        
        return cls(
            value=Decimal(str(meta_score)),
            adaptation_speed=Decimal(str(adaptation_speed)),
            generalization_score=Decimal(str(generalization_score)),
            transfer_effectiveness=Decimal(str(transfer_effectiveness)),
            meta_learning_type=MetaLearningType.TRANSFER,
            adaptation_speed_level=speed_level,
            num_tasks=2,
            num_shots=1,
            base_accuracy=Decimal(str(source_accuracy)),
            adapted_accuracy=Decimal(str(target_accuracy)),
            improvement=Decimal(str(target_accuracy - source_accuracy)),
            source_domain=source_domain,
            target_domain=target_domain,
            adaptation_time=Decimal(str(adaptation_time))
        )
    
    @classmethod
    def from_continual_learning(cls, task_accuracies: List[float], forgetting_rates: List[float],
                               num_tasks: int) -> 'MetaLearningScore':
        """Create meta-learning score from continual learning results"""
        
        # Calculate average accuracy
        avg_accuracy = np.mean(task_accuracies)
        
        # Calculate adaptation speed (inverse of forgetting)
        avg_forgetting = np.mean(forgetting_rates)
        adaptation_speed = max(0, 1 - avg_forgetting)
        
        # Calculate generalization score
        generalization_score = avg_accuracy
        
        # Calculate transfer effectiveness (how well knowledge transfers)
        transfer_effectiveness = 1 - avg_forgetting
        
        # Calculate overall score
        meta_score = (adaptation_speed + generalization_score + transfer_effectiveness) / 3
        
        # Determine adaptation speed level
        speed_level = cls._determine_adaptation_speed(int(avg_forgetting * 100))
        
        return cls(
            value=Decimal(str(meta_score)),
            adaptation_speed=Decimal(str(adaptation_speed)),
            generalization_score=Decimal(str(generalization_score)),
            transfer_effectiveness=Decimal(str(transfer_effectiveness)),
            meta_learning_type=MetaLearningType.CONTINUAL,
            adaptation_speed_level=speed_level,
            num_tasks=num_tasks,
            num_shots=1,
            base_accuracy=Decimal(str(task_accuracies[0])),
            adapted_accuracy=Decimal(str(avg_accuracy)),
            improvement=Decimal(str(avg_accuracy - task_accuracies[0])),
            catastrophic_forgetting=Decimal(str(avg_forgetting))
        )
    
    @classmethod
    def _determine_adaptation_speed(cls, iterations_or_time: int) -> AdaptationSpeed:
        """Determine adaptation speed level"""
        if iterations_or_time <= 1:
            return AdaptationSpeed.INSTANT
        elif iterations_or_time <= 5:
            return AdaptationSpeed.VERY_FAST
        elif iterations_or_time <= 20:
            return AdaptationSpeed.FAST
        elif iterations_or_time <= 50:
            return AdaptationSpeed.MODERATE
        elif iterations_or_time <= 100:
            return AdaptationSpeed.SLOW
        else:
            return AdaptationSpeed.VERY_SLOW
    
    def is_fast_adaptation(self) -> bool:
        """Check if adaptation is fast"""
        return self.adaptation_speed_level in [AdaptationSpeed.INSTANT, AdaptationSpeed.VERY_FAST, AdaptationSpeed.FAST]
    
    def is_slow_adaptation(self) -> bool:
        """Check if adaptation is slow"""
        return self.adaptation_speed_level in [AdaptationSpeed.SLOW, AdaptationSpeed.VERY_SLOW]
    
    def has_high_generalization(self) -> bool:
        """Check if generalization is high"""
        return self.generalization_score > Decimal('0.8')
    
    def has_high_transfer(self) -> bool:
        """Check if transfer effectiveness is high"""
        return self.transfer_effectiveness > Decimal('0.8')
    
    def get_adaptation_efficiency(self) -> Decimal:
        """Get adaptation efficiency (improvement per iteration)"""
        if self.adaptation_time is not None and self.adaptation_time > 0:
            return self.improvement / self.adaptation_time
        return Decimal('0')
    
    def get_knowledge_retention(self) -> Decimal:
        """Get knowledge retention score"""
        if self.catastrophic_forgetting is not None:
            return Decimal('1.0') - self.catastrophic_forgetting
        return Decimal('1.0')
    
    def get_domain_adaptability(self) -> Decimal:
        """Get domain adaptability score"""
        if self.domain_shift is not None:
            return Decimal('1.0') - self.domain_shift
        return Decimal('1.0')
    
    def get_meta_learning_quality(self) -> str:
        """Get meta-learning quality assessment"""
        if self.value > Decimal('0.9'):
            return "excellent"
        elif self.value > Decimal('0.8'):
            return "very_good"
        elif self.value > Decimal('0.7'):
            return "good"
        elif self.value > Decimal('0.6'):
            return "fair"
        else:
            return "poor"
    
    def combine_with(self, other: 'MetaLearningScore', weight: Decimal = Decimal('0.5')) -> 'MetaLearningScore':
        """Combine with another meta-learning score"""
        if not isinstance(other, MetaLearningScore):
            raise TypeError("Can only combine with MetaLearningScore")
        
        # Weighted combination
        combined_value = weight * self.value + (1 - weight) * other.value
        combined_speed = weight * self.adaptation_speed + (1 - weight) * other.adaptation_speed
        combined_gen = weight * self.generalization_score + (1 - weight) * other.generalization_score
        combined_transfer = weight * self.transfer_effectiveness + (1 - weight) * other.transfer_effectiveness
        
        # Combine accuracies
        combined_base = weight * self.base_accuracy + (1 - weight) * other.base_accuracy
        combined_adapted = weight * self.adapted_accuracy + (1 - weight) * other.adapted_accuracy
        combined_improvement = combined_adapted - combined_base
        
        # Determine combined adaptation speed level
        combined_speed_level = cls._determine_adaptation_speed(
            int((self.num_tasks + other.num_tasks) / 2)
        )
        
        return MetaLearningScore(
            value=combined_value,
            adaptation_speed=combined_speed,
            generalization_score=combined_gen,
            transfer_effectiveness=combined_transfer,
            meta_learning_type=self.meta_learning_type,  # Keep first type
            adaptation_speed_level=combined_speed_level,
            num_tasks=self.num_tasks + other.num_tasks,
            num_shots=max(self.num_shots, other.num_shots),
            base_accuracy=combined_base,
            adapted_accuracy=combined_adapted,
            improvement=combined_improvement
        )
    
    def to_json(self) -> str:
        """Convert to JSON representation"""
        meta_dict = {
            'value': float(self.value),
            'adaptation_speed': float(self.adaptation_speed),
            'generalization_score': float(self.generalization_score),
            'transfer_effectiveness': float(self.transfer_effectiveness),
            'meta_learning_type': self.meta_learning_type.value,
            'adaptation_speed_level': self.adaptation_speed_level.value,
            'num_tasks': self.num_tasks,
            'num_shots': self.num_shots,
            'base_accuracy': float(self.base_accuracy),
            'adapted_accuracy': float(self.adapted_accuracy),
            'improvement': float(self.improvement),
            'task_diversity': float(self.task_diversity) if self.task_diversity else None,
            'domain_shift': float(self.domain_shift) if self.domain_shift else None,
            'catastrophic_forgetting': float(self.catastrophic_forgetting) if self.catastrophic_forgetting else None,
            'learning_curve': [float(x) for x in self.learning_curve] if self.learning_curve else None,
            'adaptation_time': float(self.adaptation_time) if self.adaptation_time else None,
            'source_domain': self.source_domain,
            'target_domain': self.target_domain,
            'transfer_distance': float(self.transfer_distance) if self.transfer_distance else None
        }
        return json.dumps(meta_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MetaLearningScore':
        """Create from JSON representation"""
        meta_dict = json.loads(json_str)
        
        return cls(
            value=Decimal(str(meta_dict['value'])),
            adaptation_speed=Decimal(str(meta_dict['adaptation_speed'])),
            generalization_score=Decimal(str(meta_dict['generalization_score'])),
            transfer_effectiveness=Decimal(str(meta_dict['transfer_effectiveness'])),
            meta_learning_type=MetaLearningType(meta_dict['meta_learning_type']),
            adaptation_speed_level=AdaptationSpeed(meta_dict['adaptation_speed_level']),
            num_tasks=meta_dict['num_tasks'],
            num_shots=meta_dict['num_shots'],
            base_accuracy=Decimal(str(meta_dict['base_accuracy'])),
            adapted_accuracy=Decimal(str(meta_dict['adapted_accuracy'])),
            improvement=Decimal(str(meta_dict['improvement'])),
            task_diversity=Decimal(str(meta_dict['task_diversity'])) if meta_dict.get('task_diversity') else None,
            domain_shift=Decimal(str(meta_dict['domain_shift'])) if meta_dict.get('domain_shift') else None,
            catastrophic_forgetting=Decimal(str(meta_dict['catastrophic_forgetting'])) if meta_dict.get('catastrophic_forgetting') else None,
            learning_curve=[Decimal(str(x)) for x in meta_dict['learning_curve']] if meta_dict.get('learning_curve') else None,
            adaptation_time=Decimal(str(meta_dict['adaptation_time'])) if meta_dict.get('adaptation_time') else None,
            source_domain=meta_dict.get('source_domain'),
            target_domain=meta_dict.get('target_domain'),
            transfer_distance=Decimal(str(meta_dict['transfer_distance'])) if meta_dict.get('transfer_distance') else None
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, MetaLearningScore):
            return False
        
        return (self.value == other.value and
                self.adaptation_speed == other.adaptation_speed and
                self.generalization_score == other.generalization_score and
                self.transfer_effectiveness == other.transfer_effectiveness and
                self.meta_learning_type == other.meta_learning_type)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((self.value, self.adaptation_speed, self.generalization_score, self.transfer_effectiveness, self.meta_learning_type))
