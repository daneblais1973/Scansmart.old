"""
Model Registry
==============
Enterprise-grade AI model registry for managing and tracking AI models
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    from sklearn.base import BaseEstimator
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model type categories"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    LINEAR = "linear"
    ENSEMBLE = "ensemble"
    QUANTUM = "quantum"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """Model status levels"""
    REGISTERED = "registered"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ERROR = "error"

class ModelVersion(Enum):
    """Model version types"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    EXPERIMENTAL = "experimental"

@dataclass
class ModelMetadata:
    """Model metadata container"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    model_size: Optional[int] = None
    memory_usage: Optional[float] = None
    inference_time: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelArtifact:
    """Model artifact container"""
    artifact_id: str
    model_id: str
    artifact_type: str
    file_path: str
    file_size: int
    checksum: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelRegistryMetrics:
    """Model registry metrics"""
    total_models: int
    active_models: int
    deployed_models: int
    training_models: int
    error_models: int
    average_accuracy: float
    average_inference_time: float
    total_model_size: int
    registry_health: float

class ModelRegistry:
    """Enterprise-grade AI model registry"""
    
    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self.artifacts: Dict[str, List[ModelArtifact]] = {}
        self.model_versions: Dict[str, List[str]] = {}
        self.model_dependencies: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.metrics = ModelRegistryMetrics(
            total_models=0, active_models=0, deployed_models=0,
            training_models=0, error_models=0, average_accuracy=0.0,
            average_inference_time=0.0, total_model_size=0, registry_health=0.0
        )
        
        # Model validation rules
        self.validation_rules = {
            'required_fields': ['name', 'model_type', 'description', 'author'],
            'max_name_length': 100,
            'max_description_length': 1000,
            'allowed_model_types': [mt.value for mt in ModelType],
            'allowed_statuses': [ms.value for ms in ModelStatus]
        }
        
        logger.info("Model Registry initialized")
    
    async def register_model(self, name: str, model_type: ModelType, 
                           description: str, author: str, 
                           model_object: Optional[Any] = None,
                           hyperparameters: Optional[Dict[str, Any]] = None,
                           tags: Optional[List[str]] = None,
                           dependencies: Optional[List[str]] = None) -> str:
        """Register a new model in the registry"""
        try:
            model_id = str(uuid.uuid4())
            
            # Validate input
            await self._validate_model_input(name, model_type, description, author)
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version="1.0.0",
                model_type=model_type,
                status=ModelStatus.REGISTERED,
                description=description,
                author=author,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=tags or [],
                dependencies=dependencies or [],
                hyperparameters=hyperparameters or {}
            )
            
            # Store model
            self.models[model_id] = metadata
            self.artifacts[model_id] = []
            self.model_versions[model_id] = ["1.0.0"]
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Model registered: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    async def update_model(self, model_id: str, **updates) -> bool:
        """Update model metadata"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Update allowed fields
            allowed_fields = ['name', 'description', 'tags', 'hyperparameters', 'performance_metrics']
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(model, field, value)
            
            model.updated_at = datetime.now()
            
            logger.info(f"Model updated: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    async def get_models_by_type(self, model_type: ModelType) -> List[ModelMetadata]:
        """Get models by type"""
        return [model for model in self.models.values() if model.model_type == model_type]
    
    async def get_models_by_status(self, status: ModelStatus) -> List[ModelMetadata]:
        """Get models by status"""
        return [model for model in self.models.values() if model.status == status]
    
    async def get_models_by_author(self, author: str) -> List[ModelMetadata]:
        """Get models by author"""
        return [model for model in self.models.values() if model.author == author]
    
    async def search_models(self, query: str) -> List[ModelMetadata]:
        """Search models by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for model in self.models.values():
            if (query_lower in model.name.lower() or 
                query_lower in model.description.lower() or
                any(query_lower in tag.lower() for tag in model.tags)):
                results.append(model)
        
        return results
    
    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            self.models[model_id].status = status
            self.models[model_id].updated_at = datetime.now()
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Model status updated: {model_id} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            return False
    
    async def add_model_artifact(self, model_id: str, artifact_type: str, 
                                file_path: str, file_size: int, 
                                checksum: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add artifact to model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            artifact_id = str(uuid.uuid4())
            
            artifact = ModelArtifact(
                artifact_id=artifact_id,
                model_id=model_id,
                artifact_type=artifact_type,
                file_path=file_path,
                file_size=file_size,
                checksum=checksum,
                created_at=datetime.now(),
                metadata=metadata or {}
            )
            
            self.artifacts[model_id].append(artifact)
            
            logger.info(f"Artifact added to model {model_id}: {artifact_type}")
            return artifact_id
            
        except Exception as e:
            logger.error(f"Error adding model artifact: {e}")
            raise
    
    async def get_model_artifacts(self, model_id: str) -> List[ModelArtifact]:
        """Get artifacts for model"""
        return self.artifacts.get(model_id, [])
    
    async def update_model_performance(self, model_id: str, 
                                     accuracy: Optional[float] = None,
                                     precision: Optional[float] = None,
                                     recall: Optional[float] = None,
                                     f1_score: Optional[float] = None,
                                     inference_time: Optional[float] = None,
                                     custom_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update model performance metrics"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            if accuracy is not None:
                model.accuracy = accuracy
            if precision is not None:
                model.precision = precision
            if recall is not None:
                model.recall = recall
            if f1_score is not None:
                model.f1_score = f1_score
            if inference_time is not None:
                model.inference_time = inference_time
            if custom_metrics is not None:
                model.custom_metrics.update(custom_metrics)
            
            model.updated_at = datetime.now()
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Model performance updated: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return False
    
    async def get_model_dependencies(self, model_id: str) -> List[str]:
        """Get model dependencies"""
        return self.model_dependencies.get(model_id, [])
    
    async def add_model_dependency(self, model_id: str, dependency_id: str) -> bool:
        """Add model dependency"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            if dependency_id not in self.models:
                raise ValueError(f"Dependency model {dependency_id} not found")
            
            if model_id not in self.model_dependencies:
                self.model_dependencies[model_id] = []
            
            if dependency_id not in self.model_dependencies[model_id]:
                self.model_dependencies[model_id].append(dependency_id)
            
            logger.info(f"Dependency added: {model_id} -> {dependency_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model dependency: {e}")
            return False
    
    async def get_registry_metrics(self) -> ModelRegistryMetrics:
        """Get registry metrics"""
        return self.metrics
    
    async def get_registry_health(self) -> float:
        """Get registry health score"""
        try:
            if self.metrics.total_models == 0:
                return 1.0
            
            # Calculate health based on various factors
            error_rate = self.metrics.error_models / self.metrics.total_models
            deployment_rate = self.metrics.deployed_models / self.metrics.total_models
            accuracy_score = self.metrics.average_accuracy
            
            # Weighted health score
            health = (1.0 - error_rate) * 0.4 + deployment_rate * 0.3 + accuracy_score * 0.3
            
            return max(0.0, min(1.0, health))
            
        except Exception as e:
            logger.error(f"Error calculating registry health: {e}")
            return 0.0
    
    async def _validate_model_input(self, name: str, model_type: ModelType, 
                                  description: str, author: str) -> None:
        """Validate model input"""
        try:
            # Validate name
            if not name or len(name) > self.validation_rules['max_name_length']:
                raise ValueError(f"Invalid name: {name}")
            
            # Validate description
            if not description or len(description) > self.validation_rules['max_description_length']:
                raise ValueError(f"Invalid description: {description}")
            
            # Validate author
            if not author:
                raise ValueError("Author is required")
            
            # Validate model type
            if model_type.value not in self.validation_rules['allowed_model_types']:
                raise ValueError(f"Invalid model type: {model_type}")
            
        except Exception as e:
            logger.error(f"Model input validation failed: {e}")
            raise
    
    def _update_metrics(self):
        """Update registry metrics"""
        try:
            self.metrics.total_models = len(self.models)
            self.metrics.active_models = len([m for m in self.models.values() if m.status != ModelStatus.DEPRECATED])
            self.metrics.deployed_models = len([m for m in self.models.values() if m.status == ModelStatus.DEPLOYED])
            self.metrics.training_models = len([m for m in self.models.values() if m.status == ModelStatus.TRAINING])
            self.metrics.error_models = len([m for m in self.models.values() if m.status == ModelStatus.ERROR])
            
            # Calculate average accuracy
            accuracies = [m.accuracy for m in self.models.values() if m.accuracy is not None]
            self.metrics.average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            
            # Calculate average inference time
            inference_times = [m.inference_time for m in self.models.values() if m.inference_time is not None]
            self.metrics.average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
            
            # Calculate total model size
            self.metrics.total_model_size = sum(m.model_size or 0 for m in self.models.values())
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

# Global instance
model_registry = ModelRegistry()




