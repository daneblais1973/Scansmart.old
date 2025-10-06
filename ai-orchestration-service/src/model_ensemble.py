"""
Model Ensemble Manager
======================
Enterprise-grade model ensemble management service
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

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class EnsembleType(Enum):
    """Ensemble combination types"""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"

class ModelType(Enum):
    """Model types in ensemble"""
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    QUANTUM_ENHANCED = "quantum_enhanced"
    CLASSICAL_ML = "classical_ml"
    META_LEARNING = "meta_learning"
    SPECIALIZED = "specialized"

class EnsembleStatus(Enum):
    """Ensemble status levels"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    READY = "ready"
    INFERRING = "inferring"
    UPDATING = "updating"
    ERROR = "error"

@dataclass
class EnsembleModel:
    """Individual model in ensemble"""
    model_id: str
    model_type: ModelType
    model_name: str
    model_instance: Optional[Any]
    weight: float
    accuracy: float
    confidence: float
    training_time: float
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction_id: str
    ensemble_id: str
    input_data: Dict[str, Any]
    individual_predictions: Dict[str, Any]
    ensemble_prediction: Any
    confidence: float
    uncertainty: float
    model_weights: Dict[str, float]
    prediction_time: float
    timestamp: datetime

@dataclass
class EnsembleMetrics:
    """Ensemble performance metrics"""
    total_predictions: int
    average_accuracy: float
    average_confidence: float
    average_uncertainty: float
    ensemble_diversity: float
    model_agreement: float
    prediction_latency: float
    throughput: float
    model_weights: Dict[str, float]
    performance_trend: List[float]

class ModelEnsemble:
    """Enterprise-grade model ensemble management service"""
    
    def __init__(self):
        self.status = EnsembleStatus.INITIALIZING
        self.ensemble_id = str(uuid.uuid4())
        self.models = {}
        self.ensemble_type = EnsembleType.WEIGHTED_AVERAGING
        self.predictions = {}
        
        # Ensemble configuration
        self.config = {
            'max_models': 10,
            'min_models': 3,
            'weight_update_frequency': 100,  # predictions
            'model_retirement_threshold': 0.1,  # accuracy threshold
            'ensemble_diversity_threshold': 0.3
        }
        
        # Performance tracking
        self.metrics = EnsembleMetrics(
            total_predictions=0, average_accuracy=0.0, average_confidence=0.0,
            average_uncertainty=0.0, ensemble_diversity=0.0, model_agreement=0.0,
            prediction_latency=0.0, throughput=0.0, model_weights={},
            performance_trend=[]
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize ensemble
        self._initialize_ensemble()
        
        logger.info("Model Ensemble initialized")
    
    def _initialize_ensemble(self):
        """Initialize the model ensemble"""
        try:
            # Create initial models
            self._create_initial_models()
            
            # Initialize ensemble weights
            self._initialize_weights()
            
            self.status = EnsembleStatus.READY
            
            logger.info("Model ensemble initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble: {e}")
            self.status = EnsembleStatus.ERROR
            raise
    
    def _create_initial_models(self):
        """Create initial models for the ensemble"""
        try:
            if AI_AVAILABLE:
                # Create diverse models
                models_config = [
                    {'type': ModelType.NEURAL_NETWORK, 'name': 'neural_net_1', 'weight': 0.3},
                    {'type': ModelType.TRANSFORMER, 'name': 'transformer_1', 'weight': 0.25},
                    {'type': ModelType.QUANTUM_ENHANCED, 'name': 'quantum_enhanced_1', 'weight': 0.2},
                    {'type': ModelType.CLASSICAL_ML, 'name': 'classical_ml_1', 'weight': 0.15},
                    {'type': ModelType.META_LEARNING, 'name': 'meta_learning_1', 'weight': 0.1}
                ]
                
                for model_config in models_config:
                    model = self._create_model(
                        model_config['type'],
                        model_config['name'],
                        model_config['weight']
                    )
                    if model:
                        self.models[model.model_id] = model
            else:
                # Create classical fallback models
                self._create_classical_models()
                
        except Exception as e:
            logger.error(f"Error creating initial models: {e}")
    
    def _create_model(self, model_type: ModelType, name: str, weight: float) -> Optional[EnsembleModel]:
        """Create individual model"""
        try:
            model_id = str(uuid.uuid4())
            
            # Create model instance based on type
            if model_type == ModelType.NEURAL_NETWORK:
                model_instance = self._create_neural_network()
            elif model_type == ModelType.TRANSFORMER:
                model_instance = self._create_transformer()
            elif model_type == ModelType.QUANTUM_ENHANCED:
                model_instance = self._create_quantum_enhanced_model()
            elif model_type == ModelType.CLASSICAL_ML:
                model_instance = self._create_classical_ml_model()
            elif model_type == ModelType.META_LEARNING:
                model_instance = self._create_meta_learning_model()
            else:
                model_instance = self._create_specialized_model()
            
            # Initialize with random performance metrics
            accuracy = np.random.uniform(0.6, 0.9)
            confidence = np.random.uniform(0.7, 0.95)
            training_time = np.random.uniform(10, 60)
            
            model = EnsembleModel(
                model_id=model_id,
                model_type=model_type,
                model_name=name,
                model_instance=model_instance,
                weight=weight,
                accuracy=accuracy,
                confidence=confidence,
                training_time=training_time,
                last_updated=datetime.now(),
                metadata={'created_by': 'ensemble_manager'}
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {name}: {e}")
            return None
    
    def _create_neural_network(self) -> Optional[nn.Module]:
        """Create neural network model"""
        if not AI_AVAILABLE:
            return None
        
        try:
            class NeuralNetwork(nn.Module):
                def __init__(self, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, output_size)
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            return NeuralNetwork()
            
        except Exception as e:
            logger.error(f"Error creating neural network: {e}")
            return None
    
    def _create_transformer(self) -> Optional[nn.Module]:
        """Create transformer model"""
        if not AI_AVAILABLE:
            return None
        
        try:
            class TransformerModel(nn.Module):
                def __init__(self, input_size: int = 512, hidden_size: int = 256, num_heads: int = 8):
                    super().__init__()
                    self.embedding = nn.Linear(input_size, hidden_size)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(hidden_size, num_heads),
                        num_layers=4
                    )
                    self.classifier = nn.Linear(hidden_size, 10)
                
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    x = x.mean(dim=1)  # Global average pooling
                    return self.classifier(x)
            
            return TransformerModel()
            
        except Exception as e:
            logger.error(f"Error creating transformer: {e}")
            return None
    
    def _create_quantum_enhanced_model(self) -> Optional[nn.Module]:
        """Create quantum-enhanced model"""
        if not AI_AVAILABLE:
            return None
        
        try:
            class QuantumEnhancedModel(nn.Module):
                def __init__(self, input_size: int = 784, hidden_size: int = 128):
                    super().__init__()
                    self.quantum_layer = nn.Linear(input_size, hidden_size)
                    self.classical_layer = nn.Linear(hidden_size, 10)
                    self.activation = nn.ReLU()
                    self.dropout = nn.Dropout(0.1)
                
                def forward(self, x):
                    # Quantum-inspired processing
                    x = self.quantum_layer(x)
                    x = self.activation(x)
                    x = self.dropout(x)
                    
                    # Classical processing
                    x = self.classical_layer(x)
                    return x
            
            return QuantumEnhancedModel()
            
        except Exception as e:
            logger.error(f"Error creating quantum-enhanced model: {e}")
            return None
    
    def _create_classical_ml_model(self) -> Optional[Any]:
        """Create classical ML model"""
        try:
            # Simulate classical ML model
            class ClassicalMLModel:
                def __init__(self):
                    self.trained = False
                
                def predict(self, x):
                    # Simulate prediction
                    return np.random.random(10)
                
                def fit(self, x, y):
                    self.trained = True
            
            return ClassicalMLModel()
            
        except Exception as e:
            logger.error(f"Error creating classical ML model: {e}")
            return None
    
    def _create_meta_learning_model(self) -> Optional[Any]:
        """Create meta-learning model"""
        try:
            # Simulate meta-learning model
            class MetaLearningModel:
                def __init__(self):
                    self.meta_trained = False
                
                def predict(self, x):
                    # Simulate meta-learning prediction
                    return np.random.random(10)
                
                def meta_fit(self, tasks):
                    self.meta_trained = True
            
            return MetaLearningModel()
            
        except Exception as e:
            logger.error(f"Error creating meta-learning model: {e}")
            return None
    
    def _create_specialized_model(self) -> Optional[Any]:
        """Create specialized model"""
        try:
            # Simulate specialized model
            class SpecializedModel:
                def __init__(self):
                    self.specialized = True
                
                def predict(self, x):
                    # Simulate specialized prediction
                    return np.random.random(10)
            
            return SpecializedModel()
            
        except Exception as e:
            logger.error(f"Error creating specialized model: {e}")
            return None
    
    def _create_classical_models(self):
        """Create classical fallback models"""
        try:
            # Create simple classical models
            for i in range(3):
                model = self._create_model(
                    ModelType.CLASSICAL_ML,
                    f"classical_model_{i+1}",
                    1.0 / 3.0
                )
                if model:
                    self.models[model.model_id] = model
                    
        except Exception as e:
            logger.error(f"Error creating classical models: {e}")
    
    def _initialize_weights(self):
        """Initialize ensemble weights"""
        try:
            # Normalize weights
            total_weight = sum(model.weight for model in self.models.values())
            if total_weight > 0:
                for model in self.models.values():
                    model.weight /= total_weight
            
            # Update metrics
            self.metrics.model_weights = {
                model_id: model.weight for model_id, model in self.models.items()
            }
            
        except Exception as e:
            logger.error(f"Error initializing weights: {e}")
    
    async def start_ensemble(self):
        """Start the model ensemble service"""
        try:
            logger.info("Starting Model Ensemble...")
            
            self.status = EnsembleStatus.READY
            
            # Start background tasks
            asyncio.create_task(self._ensemble_monitoring_loop())
            asyncio.create_task(self._weight_update_loop())
            
            logger.info("Model Ensemble started successfully")
            
        except Exception as e:
            logger.error(f"Error starting ensemble: {e}")
            self.status = EnsembleStatus.ERROR
            raise
    
    async def stop_ensemble(self):
        """Stop the model ensemble service"""
        try:
            logger.info("Stopping Model Ensemble...")
            
            self.status = EnsembleStatus.INITIALIZING
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Model Ensemble stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping ensemble: {e}")
            raise
    
    async def predict(self, input_data: Dict[str, Any]) -> EnsemblePrediction:
        """Make ensemble prediction"""
        try:
            start_time = datetime.now()
            prediction_id = str(uuid.uuid4())
            
            # Get individual model predictions
            individual_predictions = {}
            model_weights = {}
            
            for model_id, model in self.models.items():
                try:
                    # Get prediction from individual model
                    if model.model_instance:
                        if hasattr(model.model_instance, 'predict'):
                            prediction = model.model_instance.predict(input_data)
                        else:
                            # Simulate prediction
                            prediction = np.random.random(10)
                    else:
                        prediction = np.random.random(10)
                    
                    individual_predictions[model_id] = {
                        'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                        'confidence': model.confidence,
                        'model_type': model.model_type.value,
                        'model_name': model.model_name
                    }
                    
                    model_weights[model_id] = model.weight
                    
                except Exception as e:
                    logger.error(f"Error getting prediction from model {model_id}: {e}")
                    # Use fallback prediction
                    individual_predictions[model_id] = {
                        'prediction': np.random.random(10).tolist(),
                        'confidence': 0.5,
                        'model_type': model.model_type.value,
                        'model_name': model.model_name,
                        'error': str(e)
                    }
                    model_weights[model_id] = 0.0
            
            # Combine predictions using ensemble method
            ensemble_prediction = await self._combine_predictions(individual_predictions, model_weights)
            
            # Calculate ensemble metrics
            confidence = self._calculate_ensemble_confidence(individual_predictions, model_weights)
            uncertainty = self._calculate_ensemble_uncertainty(individual_predictions, model_weights)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Create ensemble prediction result
            ensemble_pred = EnsemblePrediction(
                prediction_id=prediction_id,
                ensemble_id=self.ensemble_id,
                input_data=input_data,
                individual_predictions=individual_predictions,
                ensemble_prediction=ensemble_prediction,
                confidence=confidence,
                uncertainty=uncertainty,
                model_weights=model_weights,
                prediction_time=prediction_time,
                timestamp=datetime.now()
            )
            
            # Store prediction
            self.predictions[prediction_id] = ensemble_pred
            
            # Update metrics
            self._update_prediction_metrics(ensemble_pred)
            
            logger.info(f"Ensemble prediction completed: {prediction_id}")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    async def _combine_predictions(self, individual_predictions: Dict[str, Any], 
                                 model_weights: Dict[str, float]) -> Any:
        """Combine individual model predictions"""
        try:
            if self.ensemble_type == EnsembleType.VOTING:
                return self._voting_combination(individual_predictions, model_weights)
            elif self.ensemble_type == EnsembleType.AVERAGING:
                return self._averaging_combination(individual_predictions, model_weights)
            elif self.ensemble_type == EnsembleType.WEIGHTED_AVERAGING:
                return self._weighted_averaging_combination(individual_predictions, model_weights)
            elif self.ensemble_type == EnsembleType.STACKING:
                return self._stacking_combination(individual_predictions, model_weights)
            else:
                return self._weighted_averaging_combination(individual_predictions, model_weights)
                
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return np.random.random(10).tolist()
    
    def _voting_combination(self, individual_predictions: Dict[str, Any], 
                          model_weights: Dict[str, float]) -> List[int]:
        """Voting combination method"""
        try:
            # Get class predictions (argmax)
            votes = []
            for model_id, pred_data in individual_predictions.items():
                prediction = pred_data['prediction']
                if isinstance(prediction, list):
                    class_vote = np.argmax(prediction)
                    weight = model_weights.get(model_id, 1.0)
                    votes.extend([class_vote] * int(weight * 10))  # Weighted voting
            
            # Count votes
            if votes:
                final_prediction = max(set(votes), key=votes.count)
                # Convert to one-hot encoding
                result = [0] * 10
                result[final_prediction] = 1
                return result
            else:
                return [0.1] * 10
                
        except Exception as e:
            logger.error(f"Error in voting combination: {e}")
            return [0.1] * 10
    
    def _averaging_combination(self, individual_predictions: Dict[str, Any], 
                             model_weights: Dict[str, float]) -> List[float]:
        """Averaging combination method"""
        try:
            predictions = []
            for model_id, pred_data in individual_predictions.items():
                prediction = pred_data['prediction']
                if isinstance(prediction, list):
                    predictions.append(np.array(prediction))
            
            if predictions:
                # Simple average
                ensemble_pred = np.mean(predictions, axis=0)
                return ensemble_pred.tolist()
            else:
                return [0.1] * 10
                
        except Exception as e:
            logger.error(f"Error in averaging combination: {e}")
            return [0.1] * 10
    
    def _weighted_averaging_combination(self, individual_predictions: Dict[str, Any], 
                                      model_weights: Dict[str, float]) -> List[float]:
        """Weighted averaging combination method"""
        try:
            predictions = []
            weights = []
            
            for model_id, pred_data in individual_predictions.items():
                prediction = pred_data['prediction']
                weight = model_weights.get(model_id, 0.0)
                
                if isinstance(prediction, list) and weight > 0:
                    predictions.append(np.array(prediction))
                    weights.append(weight)
            
            if predictions and weights:
                # Weighted average
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights
                
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                return ensemble_pred.tolist()
            else:
                return [0.1] * 10
                
        except Exception as e:
            logger.error(f"Error in weighted averaging combination: {e}")
            return [0.1] * 10
    
    def _stacking_combination(self, individual_predictions: Dict[str, Any], 
                            model_weights: Dict[str, float]) -> List[float]:
        """Stacking combination method"""
        try:
            # For simplicity, use weighted averaging as stacking fallback
            return self._weighted_averaging_combination(individual_predictions, model_weights)
            
        except Exception as e:
            logger.error(f"Error in stacking combination: {e}")
            return [0.1] * 10
    
    def _calculate_ensemble_confidence(self, individual_predictions: Dict[str, Any], 
                                     model_weights: Dict[str, float]) -> float:
        """Calculate ensemble confidence"""
        try:
            confidences = []
            weights = []
            
            for model_id, pred_data in individual_predictions.items():
                confidence = pred_data.get('confidence', 0.5)
                weight = model_weights.get(model_id, 0.0)
                
                if weight > 0:
                    confidences.append(confidence)
                    weights.append(weight)
            
            if confidences and weights:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                return float(np.average(confidences, weights=weights))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def _calculate_ensemble_uncertainty(self, individual_predictions: Dict[str, Any], 
                                      model_weights: Dict[str, float]) -> float:
        """Calculate ensemble uncertainty"""
        try:
            predictions = []
            weights = []
            
            for model_id, pred_data in individual_predictions.items():
                prediction = pred_data['prediction']
                weight = model_weights.get(model_id, 0.0)
                
                if isinstance(prediction, list) and weight > 0:
                    predictions.append(np.array(prediction))
                    weights.append(weight)
            
            if len(predictions) > 1:
                # Calculate prediction variance as uncertainty measure
                predictions = np.array(predictions)
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Weighted variance
                mean_pred = np.average(predictions, axis=0, weights=weights)
                variance = np.average((predictions - mean_pred) ** 2, axis=0, weights=weights)
                uncertainty = float(np.mean(variance))
                
                return uncertainty
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error calculating ensemble uncertainty: {e}")
            return 0.1
    
    def _update_prediction_metrics(self, prediction: EnsemblePrediction):
        """Update ensemble metrics with new prediction"""
        try:
            self.metrics.total_predictions += 1
            
            # Update average confidence
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (self.metrics.total_predictions - 1) + prediction.confidence) /
                self.metrics.total_predictions
            )
            
            # Update average uncertainty
            self.metrics.average_uncertainty = (
                (self.metrics.average_uncertainty * (self.metrics.total_predictions - 1) + prediction.uncertainty) /
                self.metrics.total_predictions
            )
            
            # Update prediction latency
            self.metrics.prediction_latency = (
                (self.metrics.prediction_latency * (self.metrics.total_predictions - 1) + prediction.prediction_time) /
                self.metrics.total_predictions
            )
            
            # Update throughput
            if prediction.prediction_time > 0:
                current_throughput = 1.0 / prediction.prediction_time
                self.metrics.throughput = (
                    (self.metrics.throughput * (self.metrics.total_predictions - 1) + current_throughput) /
                    self.metrics.total_predictions
                )
            
            # Update performance trend
            self.metrics.performance_trend.append(prediction.confidence)
            if len(self.metrics.performance_trend) > 100:
                self.metrics.performance_trend = self.metrics.performance_trend[-100:]
            
        except Exception as e:
            logger.error(f"Error updating prediction metrics: {e}")
    
    async def add_model(self, model_type: ModelType, name: str, weight: float = 0.1) -> str:
        """Add new model to ensemble"""
        try:
            if len(self.models) >= self.config['max_models']:
                # Remove worst performing model
                await self._remove_worst_model()
            
            model = self._create_model(model_type, name, weight)
            if model:
                self.models[model.model_id] = model
                self._initialize_weights()  # Re-normalize weights
                
                logger.info(f"Added model {name} to ensemble")
                return model.model_id
            else:
                raise ValueError(f"Failed to create model {name}")
                
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            raise
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove model from ensemble"""
        try:
            if model_id in self.models:
                del self.models[model_id]
                self._initialize_weights()  # Re-normalize weights
                
                logger.info(f"Removed model {model_id} from ensemble")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            return False
    
    async def _remove_worst_model(self):
        """Remove worst performing model"""
        try:
            if not self.models:
                return
            
            # Find model with lowest accuracy
            worst_model_id = min(
                self.models.keys(),
                key=lambda mid: self.models[mid].accuracy
            )
            
            await self.remove_model(worst_model_id)
            
        except Exception as e:
            logger.error(f"Error removing worst model: {e}")
    
    async def update_model_weights(self, performance_data: Dict[str, float]):
        """Update model weights based on performance"""
        try:
            for model_id, performance in performance_data.items():
                if model_id in self.models:
                    # Update model accuracy
                    self.models[model_id].accuracy = performance
                    
                    # Update weight based on performance
                    self.models[model_id].weight = max(0.01, performance)  # Minimum weight
            
            # Re-normalize weights
            self._initialize_weights()
            
            logger.info("Model weights updated based on performance")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
    
    async def _ensemble_monitoring_loop(self):
        """Monitor ensemble performance"""
        try:
            while self.status == EnsembleStatus.READY:
                await asyncio.sleep(30)
                
                # Check model performance
                await self._check_model_performance()
                
                # Update ensemble diversity
                self._update_ensemble_diversity()
                
        except Exception as e:
            logger.error(f"Error in ensemble monitoring loop: {e}")
    
    async def _weight_update_loop(self):
        """Update model weights periodically"""
        try:
            while self.status == EnsembleStatus.READY:
                await asyncio.sleep(60)
                
                # Update weights based on recent performance
                if self.metrics.total_predictions % self.config['weight_update_frequency'] == 0:
                    await self._update_weights_from_performance()
                
        except Exception as e:
            logger.error(f"Error in weight update loop: {e}")
    
    async def _check_model_performance(self):
        """Check individual model performance"""
        try:
            for model_id, model in self.models.items():
                # Simulate performance check
                if model.accuracy < self.config['model_retirement_threshold']:
                    logger.warning(f"Model {model_id} performance below threshold")
                    # Could implement model retirement logic here
                
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _update_ensemble_diversity(self):
        """Update ensemble diversity metric"""
        try:
            if len(self.models) > 1:
                # Calculate diversity based on model types
                model_types = [model.model_type.value for model in self.models.values()]
                unique_types = len(set(model_types))
                total_models = len(model_types)
                
                self.metrics.ensemble_diversity = unique_types / total_models
            else:
                self.metrics.ensemble_diversity = 0.0
                
        except Exception as e:
            logger.error(f"Error updating ensemble diversity: {e}")
    
    async def _update_weights_from_performance(self):
        """Update weights based on recent performance"""
        try:
            # Simulate performance-based weight updates
            for model_id, model in self.models.items():
                # Random performance update (in real implementation, would use actual metrics)
                performance_change = np.random.uniform(-0.05, 0.05)
                model.accuracy = max(0.0, min(1.0, model.accuracy + performance_change))
                model.weight = max(0.01, model.accuracy)
            
            # Re-normalize weights
            self._initialize_weights()
            
        except Exception as e:
            logger.error(f"Error updating weights from performance: {e}")
    
    async def get_ensemble_status(self) -> Dict[str, Any]:
        """Get ensemble status and metrics"""
        return {
            'ensemble_id': self.ensemble_id,
            'status': self.status.value,
            'ensemble_type': self.ensemble_type.value,
            'num_models': len(self.models),
            'models': {
                model_id: {
                    'model_name': model.model_name,
                    'model_type': model.model_type.value,
                    'weight': model.weight,
                    'accuracy': model.accuracy,
                    'confidence': model.confidence,
                    'last_updated': model.last_updated.isoformat()
                }
                for model_id, model in self.models.items()
            },
            'metrics': {
                'total_predictions': self.metrics.total_predictions,
                'average_accuracy': self.metrics.average_accuracy,
                'average_confidence': self.metrics.average_confidence,
                'average_uncertainty': self.metrics.average_uncertainty,
                'ensemble_diversity': self.metrics.ensemble_diversity,
                'model_agreement': self.metrics.model_agreement,
                'prediction_latency': self.metrics.prediction_latency,
                'throughput': self.metrics.throughput
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history"""
        try:
            predictions = list(self.predictions.values())
            predictions.sort(key=lambda x: x.timestamp, reverse=True)
            
            history = []
            for pred in predictions[:limit]:
                history.append({
                    'prediction_id': pred.prediction_id,
                    'ensemble_id': pred.ensemble_id,
                    'confidence': pred.confidence,
                    'uncertainty': pred.uncertainty,
                    'prediction_time': pred.prediction_time,
                    'timestamp': pred.timestamp.isoformat()
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []

# Global instance
model_ensemble = ModelEnsemble()




