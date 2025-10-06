"""
Meta-Learning Model Entity
==========================
Enterprise-grade meta-learning model entity
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

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class MetaLearningType(Enum):
    """Meta-learning type categories"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    PROTONET = "protonet"  # Prototypical Networks
    FEW_SHOT = "few_shot"  # Few-Shot Learning
    META_GRADIENT = "meta_gradient"  # Meta-Gradient
    REPTILE = "reptile"  # Reptile
    META_SGD = "meta_sgd"  # Meta-SGD
    META_LSTM = "meta_lstm"  # Meta-LSTM
    CUSTOM = "custom"
    
    # Advanced Meta-Learning Types
    META_REINFORCEMENT = "meta_reinforcement"  # Meta-Reinforcement Learning
    META_IMITATION = "meta_imitation"  # Meta-Imitation Learning
    META_CURRICULUM = "meta_curriculum"  # Meta-Curriculum Learning
    META_MULTI_TASK = "meta_multi_task"  # Meta-Multi-Task Learning
    META_TRANSFER = "meta_transfer"  # Meta-Transfer Learning
    META_ADAPTATION = "meta_adaptation"  # Meta-Adaptation Learning
    
    # Quantum-Enhanced Meta-Learning
    QUANTUM_MAML = "quantum_maml"  # Quantum MAML
    QUANTUM_PROTONET = "quantum_protonet"  # Quantum Prototypical Networks
    QUANTUM_FEW_SHOT = "quantum_few_shot"  # Quantum Few-Shot Learning
    QUANTUM_META_GRADIENT = "quantum_meta_gradient"  # Quantum Meta-Gradient
    
    # Financial-Specific Meta-Learning
    FINANCIAL_MAML = "financial_maml"  # Financial MAML
    TRADING_META_LEARNING = "trading_meta_learning"  # Trading Meta-Learning
    PORTFOLIO_META_LEARNING = "portfolio_meta_learning"  # Portfolio Meta-Learning
    RISK_META_LEARNING = "risk_meta_learning"  # Risk Meta-Learning

class LearningMode(Enum):
    """Learning mode categories"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    CONTINUAL = "continual"

class ModelStatus(Enum):
    """Model status categories"""
    INITIALIZED = "initialized"
    TRAINING = "training"
    TRAINED = "trained"
    FINE_TUNING = "fine_tuning"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ERROR = "error"

@dataclass
class MetaLearningModel:
    """Meta-learning model entity"""
    model_id: str
    name: str
    meta_learning_type: MetaLearningType
    learning_mode: LearningMode
    status: ModelStatus
    
    # Model architecture
    base_model: Optional[Any] = None
    meta_parameters: Dict[str, Any] = field(default_factory=dict)
    task_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    meta_accuracy: float = 0.0
    few_shot_accuracy: float = 0.0
    adaptation_speed: float = 0.0
    generalization_score: float = 0.0
    task_diversity: float = 0.0
    
    # Training data
    support_sets: List[Dict[str, Any]] = field(default_factory=list)
    query_sets: List[Dict[str, Any]] = field(default_factory=list)
    meta_tasks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model capabilities
    max_tasks: int = 100
    max_episodes: int = 1000
    adaptation_steps: int = 5
    learning_rate: float = 0.001
    meta_learning_rate: float = 0.01
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate model invariants"""
        try:
            # Validate learning rate
            if self.learning_rate <= 0 or self.learning_rate > 1:
                raise ValueError("Learning rate must be between 0 and 1")
            
            if self.meta_learning_rate <= 0 or self.meta_learning_rate > 1:
                raise ValueError("Meta-learning rate must be between 0 and 1")
            
            # Validate adaptation steps
            if self.adaptation_steps < 1:
                raise ValueError("Adaptation steps must be at least 1")
            
        except Exception as e:
            logger.error(f"Meta-learning model validation error: {e}")
            raise
    
    def add_support_set(self, support_set: Dict[str, Any]) -> bool:
        """Add support set for meta-learning"""
        try:
            if not self._validate_support_set(support_set):
                raise ValueError("Invalid support set")
            
            self.support_sets.append(support_set)
            self.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error adding support set: {e}")
            return False
    
    def add_query_set(self, query_set: Dict[str, Any]) -> bool:
        """Add query set for meta-learning"""
        try:
            if not self._validate_query_set(query_set):
                raise ValueError("Invalid query set")
            
            self.query_sets.append(query_set)
            self.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error adding query set: {e}")
            return False
    
    def add_meta_task(self, task: Dict[str, Any]) -> bool:
        """Add meta-learning task"""
        try:
            if not self._validate_meta_task(task):
                raise ValueError("Invalid meta task")
            
            self.meta_tasks.append(task)
            self.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error adding meta task: {e}")
            return False
    
    def _validate_support_set(self, support_set: Dict[str, Any]) -> bool:
        """Validate support set"""
        try:
            required_fields = ['data', 'labels', 'task_id']
            return all(field in support_set for field in required_fields)
            
        except Exception as e:
            logger.error(f"Error validating support set: {e}")
            return False
    
    def _validate_query_set(self, query_set: Dict[str, Any]) -> bool:
        """Validate query set"""
        try:
            required_fields = ['data', 'labels', 'task_id']
            return all(field in query_set for field in required_fields)
            
        except Exception as e:
            logger.error(f"Error validating query set: {e}")
            return False
    
    def _validate_meta_task(self, task: Dict[str, Any]) -> bool:
        """Validate meta task"""
        try:
            required_fields = ['task_id', 'task_type', 'difficulty']
            return all(field in task for field in required_fields)
            
        except Exception as e:
            logger.error(f"Error validating meta task: {e}")
            return False
    
    def update_meta_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update meta-parameters"""
        try:
            self.meta_parameters.update(parameters)
            self.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error updating meta-parameters: {e}")
            return False
    
    def update_task_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update task-specific parameters"""
        try:
            self.task_parameters.update(parameters)
            self.updated_at = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error updating task parameters: {e}")
            return False
    
    def calculate_meta_learning_metrics(self) -> Dict[str, float]:
        """Calculate meta-learning specific metrics"""
        try:
            metrics = {}
            
            # Calculate adaptation speed
            metrics['adaptation_speed'] = self._calculate_adaptation_speed()
            
            # Calculate generalization score
            metrics['generalization_score'] = self._calculate_generalization()
            
            # Calculate task diversity
            metrics['task_diversity'] = self._calculate_task_diversity()
            
            # Calculate meta-learning efficiency
            metrics['meta_efficiency'] = self._calculate_meta_efficiency()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating meta-learning metrics: {e}")
            return {}
    
    def _calculate_adaptation_speed(self) -> float:
        """Calculate adaptation speed"""
        try:
            if not self.meta_tasks:
                return 0.0
            
            # Simulate adaptation speed calculation
            # In practice, this would measure how quickly the model adapts to new tasks
            total_tasks = len(self.meta_tasks)
            adaptation_steps = self.adaptation_steps
            
            # Speed is inversely proportional to adaptation steps
            speed = 1.0 / (adaptation_steps + 1)
            
            # Normalize by number of tasks
            return min(1.0, speed * total_tasks / 10.0)
            
        except Exception as e:
            logger.error(f"Error calculating adaptation speed: {e}")
            return 0.0
    
    def _calculate_generalization(self) -> float:
        """Calculate generalization score"""
        try:
            if not self.meta_tasks:
                return 0.0
            
            # Simulate generalization calculation
            # In practice, this would measure how well the model generalizes across tasks
            task_difficulties = [task.get('difficulty', 0.5) for task in self.meta_tasks]
            avg_difficulty = sum(task_difficulties) / len(task_difficulties)
            
            # Generalization is higher for more difficult tasks
            generalization = min(1.0, avg_difficulty * 1.2)
            
            return generalization
            
        except Exception as e:
            logger.error(f"Error calculating generalization: {e}")
            return 0.0
    
    def _calculate_task_diversity(self) -> float:
        """Calculate task diversity"""
        try:
            if not self.meta_tasks:
                return 0.0
            
            # Simulate task diversity calculation
            # In practice, this would measure the diversity of tasks the model can handle
            task_types = set(task.get('task_type', 'unknown') for task in self.meta_tasks)
            unique_types = len(task_types)
            
            # Diversity is proportional to number of unique task types
            diversity = min(1.0, unique_types / 5.0)
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating task diversity: {e}")
            return 0.0
    
    def _calculate_meta_efficiency(self) -> float:
        """Calculate meta-learning efficiency"""
        try:
            if not self.meta_tasks:
                return 0.0
            
            # Simulate meta-learning efficiency
            # In practice, this would measure how efficiently the model learns from few examples
            support_sets_count = len(self.support_sets)
            query_sets_count = len(self.query_sets)
            
            if support_sets_count == 0 or query_sets_count == 0:
                return 0.0
            
            # Efficiency is higher when there are more support sets relative to query sets
            efficiency = min(1.0, support_sets_count / max(1, query_sets_count))
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating meta efficiency: {e}")
            return 0.0
    
    def adapt_to_task(self, task_id: str, adaptation_steps: Optional[int] = None) -> bool:
        """Adapt model to specific task"""
        try:
            if adaptation_steps is None:
                adaptation_steps = self.adaptation_steps
            
            # Simulate task adaptation
            # In practice, this would perform actual meta-learning adaptation
            
            # Update status
            self.status = ModelStatus.FINE_TUNING
            
            # Simulate adaptation process
            for step in range(adaptation_steps):
                # In practice, this would perform gradient updates
                pass
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.updated_at = datetime.now()
            
            logger.info(f"Model adapted to task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adapting to task: {e}")
            self.status = ModelStatus.ERROR
            return False
    
    def evaluate_few_shot_performance(self, num_shots: int) -> Dict[str, float]:
        """Evaluate few-shot learning performance"""
        try:
            if not self.support_sets or not self.query_sets:
                return {}
            
            # Simulate few-shot evaluation
            # In practice, this would perform actual evaluation
            
            # Calculate accuracy based on number of shots
            base_accuracy = 0.5
            shot_bonus = min(0.4, num_shots * 0.1)
            accuracy = base_accuracy + shot_bonus
            
            # Calculate other metrics
            precision = accuracy * 0.9
            recall = accuracy * 0.85
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'num_shots': num_shots
            }
            
        except Exception as e:
            logger.error(f"Error evaluating few-shot performance: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        try:
            return {
                'model_id': self.model_id,
                'name': self.name,
                'meta_learning_type': self.meta_learning_type.value,
                'learning_mode': self.learning_mode.value,
                'status': self.status.value,
                'meta_accuracy': self.meta_accuracy,
                'few_shot_accuracy': self.few_shot_accuracy,
                'adaptation_speed': self.adaptation_speed,
                'generalization_score': self.generalization_score,
                'task_diversity': self.task_diversity,
                'num_support_sets': len(self.support_sets),
                'num_query_sets': len(self.query_sets),
                'num_meta_tasks': len(self.meta_tasks),
                'max_tasks': self.max_tasks,
                'max_episodes': self.max_episodes,
                'adaptation_steps': self.adaptation_steps,
                'learning_rate': self.learning_rate,
                'meta_learning_rate': self.meta_learning_rate,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        try:
            return {
                'model_id': self.model_id,
                'name': self.name,
                'meta_learning_type': self.meta_learning_type.value,
                'learning_mode': self.learning_mode.value,
                'status': self.status.value,
                'meta_parameters': self.meta_parameters,
                'task_parameters': self.task_parameters,
                'meta_accuracy': self.meta_accuracy,
                'few_shot_accuracy': self.few_shot_accuracy,
                'adaptation_speed': self.adaptation_speed,
                'generalization_score': self.generalization_score,
                'task_diversity': self.task_diversity,
                'support_sets': self.support_sets,
                'query_sets': self.query_sets,
                'meta_tasks': self.meta_tasks,
                'max_tasks': self.max_tasks,
                'max_episodes': self.max_episodes,
                'adaptation_steps': self.adaptation_steps,
                'learning_rate': self.learning_rate,
                'meta_learning_rate': self.meta_learning_rate,
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat(),
                'metadata': self.metadata
            }
            
        except Exception as e:
            logger.error(f"Error converting model to dict: {e}")
            return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaLearningModel':
        """Create model from dictionary"""
        try:
            return cls(
                model_id=data.get('model_id', str(uuid.uuid4())),
                name=data.get('name', ''),
                meta_learning_type=MetaLearningType(data.get('meta_learning_type', 'maml')),
                learning_mode=LearningMode(data.get('learning_mode', 'supervised')),
                status=ModelStatus(data.get('status', 'initialized')),
                meta_parameters=data.get('meta_parameters', {}),
                task_parameters=data.get('task_parameters', {}),
                meta_accuracy=data.get('meta_accuracy', 0.0),
                few_shot_accuracy=data.get('few_shot_accuracy', 0.0),
                adaptation_speed=data.get('adaptation_speed', 0.0),
                generalization_score=data.get('generalization_score', 0.0),
                task_diversity=data.get('task_diversity', 0.0),
                support_sets=data.get('support_sets', []),
                query_sets=data.get('query_sets', []),
                meta_tasks=data.get('meta_tasks', []),
                max_tasks=data.get('max_tasks', 100),
                max_episodes=data.get('max_episodes', 1000),
                adaptation_steps=data.get('adaptation_steps', 5),
                learning_rate=data.get('learning_rate', 0.001),
                meta_learning_rate=data.get('meta_learning_rate', 0.01),
                created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
                metadata=data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error creating model from dict: {e}")
            raise
