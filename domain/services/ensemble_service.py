"""
Ensemble Service
================
Enterprise-grade ensemble learning domain service
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
    import numpy as np
    from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Ensemble method categories"""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ADABOOST = "adaboost"
    EXTRA_TREES = "extra_trees"

class EnsembleType(Enum):
    """Ensemble type categories"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

@dataclass
class EnsembleModel:
    """Ensemble model container"""
    model_id: str
    name: str
    ensemble_method: EnsembleMethod
    ensemble_type: EnsembleType
    base_models: List[Dict[str, Any]]
    weights: List[float]
    meta_model: Optional[Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsemblePrediction:
    """Ensemble prediction container"""
    prediction_id: str
    model_id: str
    input_data: Any
    predictions: List[Any]
    ensemble_prediction: Any
    confidence_scores: List[float]
    uncertainty: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleMetrics:
    """Ensemble service metrics"""
    total_models: int
    total_predictions: int
    average_accuracy: float
    average_confidence: float
    ensemble_diversity: float
    prediction_consensus: float

class EnsembleService:
    """Enterprise-grade ensemble learning domain service"""
    
    def __init__(self):
        self.ensemble_models: Dict[str, EnsembleModel] = {}
        self.ensemble_predictions: Dict[str, EnsemblePrediction] = {}
        self.base_model_registry: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics = EnsembleMetrics(
            total_models=0, total_predictions=0, average_accuracy=0.0,
            average_confidence=0.0, ensemble_diversity=0.0, prediction_consensus=0.0
        )
        
        # Ensemble configuration
        self.config = {
            'max_base_models': 20,
            'min_base_models': 3,
            'default_weights': 'uniform',
            'enable_weight_optimization': True,
            'enable_model_selection': True,
            'diversity_threshold': 0.5
        }
        
        # Initialize base model registry
        self._initialize_base_models()
        
        logger.info("Ensemble Service initialized")
    
    def _initialize_base_models(self):
        """Initialize base model registry"""
        try:
            if not AI_AVAILABLE:
                return
            
            # Classification models
            self.base_model_registry['logistic_regression'] = LogisticRegression
            self.base_model_registry['random_forest'] = RandomForestClassifier
            self.base_model_registry['gradient_boosting'] = GradientBoostingClassifier
            self.base_model_registry['svm'] = SVC
            self.base_model_registry['naive_bayes'] = GaussianNB
            self.base_model_registry['mlp'] = MLPClassifier
            
            # Regression models
            self.base_model_registry['linear_regression'] = LinearRegression
            self.base_model_registry['random_forest_reg'] = RandomForestRegressor
            self.base_model_registry['gradient_boosting_reg'] = GradientBoostingRegressor
            self.base_model_registry['svr'] = SVR
            self.base_model_registry['mlp_reg'] = MLPRegressor
            
            logger.info("Base model registry initialized")
            
        except Exception as e:
            logger.error(f"Error initializing base models: {e}")
    
    async def create_ensemble_model(self, name: str, ensemble_method: EnsembleMethod,
                                  ensemble_type: EnsembleType,
                                  base_models: List[Dict[str, Any]],
                                  weights: Optional[List[float]] = None) -> str:
        """Create ensemble model"""
        try:
            model_id = str(uuid.uuid4())
            
            # Validate base models
            if len(base_models) < self.config['min_base_models']:
                raise ValueError(f"Minimum {self.config['min_base_models']} base models required")
            
            if len(base_models) > self.config['max_base_models']:
                raise ValueError(f"Maximum {self.config['max_base_models']} base models allowed")
            
            # Set default weights if not provided
            if weights is None:
                weights = [1.0 / len(base_models)] * len(base_models)
            
            # Validate weights
            if len(weights) != len(base_models):
                raise ValueError("Number of weights must match number of base models")
            
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            
            # Create ensemble model
            ensemble_model = EnsembleModel(
                model_id=model_id,
                name=name,
                ensemble_method=ensemble_method,
                ensemble_type=ensemble_type,
                base_models=base_models,
                weights=weights,
                meta_model=None,
                performance_metrics={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store ensemble model
            self.ensemble_models[model_id] = ensemble_model
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Ensemble model created: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            raise
    
    async def train_ensemble_model(self, model_id: str, X_train: np.ndarray, 
                                 y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                                 y_val: Optional[np.ndarray] = None) -> bool:
        """Train ensemble model"""
        try:
            if model_id not in self.ensemble_models:
                raise ValueError(f"Ensemble model {model_id} not found")
            
            ensemble_model = self.ensemble_models[model_id]
            
            # Train based on ensemble method
            if ensemble_model.ensemble_method == EnsembleMethod.VOTING:
                success = await self._train_voting_ensemble(ensemble_model, X_train, y_train)
            elif ensemble_model.ensemble_method == EnsembleMethod.BAGGING:
                success = await self._train_bagging_ensemble(ensemble_model, X_train, y_train)
            elif ensemble_model.ensemble_method == EnsembleMethod.BOOSTING:
                success = await self._train_boosting_ensemble(ensemble_model, X_train, y_train)
            elif ensemble_model.ensemble_method == EnsembleMethod.STACKING:
                success = await self._train_stacking_ensemble(ensemble_model, X_train, y_train, X_val, y_val)
            else:
                success = await self._train_default_ensemble(ensemble_model, X_train, y_train)
            
            if success:
                ensemble_model.updated_at = datetime.now()
                logger.info(f"Ensemble model trained: {model_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return False
    
    async def _train_voting_ensemble(self, ensemble_model: EnsembleModel, 
                                   X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Train voting ensemble"""
        try:
            if not AI_AVAILABLE:
                return False
            
            # Create base models
            base_models = []
            for model_config in ensemble_model.base_models:
                model_name = model_config['name']
                model_params = model_config.get('parameters', {})
                
                if model_name in self.base_model_registry:
                    model_class = self.base_model_registry[model_name]
                    model = model_class(**model_params)
                    base_models.append((model_name, model))
            
            # Create voting ensemble
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                voting_ensemble = VotingClassifier(
                    estimators=base_models,
                    voting='soft' if ensemble_model.weights else 'hard'
                )
            else:
                voting_ensemble = VotingRegressor(
                    estimators=base_models,
                    weights=ensemble_model.weights if ensemble_model.weights else None
                )
            
            # Train ensemble
            voting_ensemble.fit(X_train, y_train)
            
            # Store meta model
            ensemble_model.meta_model = voting_ensemble
            
            return True
            
        except Exception as e:
            logger.error(f"Error training voting ensemble: {e}")
            return False
    
    async def _train_bagging_ensemble(self, ensemble_model: EnsembleModel, 
                                    X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Train bagging ensemble"""
        try:
            if not AI_AVAILABLE:
                return False
            
            # Create base model
            base_model_config = ensemble_model.base_models[0]
            model_name = base_model_config['name']
            model_params = base_model_config.get('parameters', {})
            
            if model_name in self.base_model_registry:
                base_model_class = self.base_model_registry[model_name]
                base_model = base_model_class(**model_params)
                
                # Create bagging ensemble
                if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                    bagging_ensemble = BaggingClassifier(
                        base_estimator=base_model,
                        n_estimators=len(ensemble_model.base_models),
                        random_state=42
                    )
                else:
                    bagging_ensemble = BaggingRegressor(
                        base_estimator=base_model,
                        n_estimators=len(ensemble_model.base_models),
                        random_state=42
                    )
                
                # Train ensemble
                bagging_ensemble.fit(X_train, y_train)
                
                # Store meta model
                ensemble_model.meta_model = bagging_ensemble
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training bagging ensemble: {e}")
            return False
    
    async def _train_boosting_ensemble(self, ensemble_model: EnsembleModel, 
                                     X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Train boosting ensemble"""
        try:
            if not AI_AVAILABLE:
                return False
            
            # Create boosting ensemble
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                boosting_ensemble = GradientBoostingClassifier(
                    n_estimators=len(ensemble_model.base_models),
                    random_state=42
                )
            else:
                boosting_ensemble = GradientBoostingRegressor(
                    n_estimators=len(ensemble_model.base_models),
                    random_state=42
                )
            
            # Train ensemble
            boosting_ensemble.fit(X_train, y_train)
            
            # Store meta model
            ensemble_model.meta_model = boosting_ensemble
            
            return True
            
        except Exception as e:
            logger.error(f"Error training boosting ensemble: {e}")
            return False
    
    async def _train_stacking_ensemble(self, ensemble_model: EnsembleModel, 
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: Optional[np.ndarray] = None,
                                     y_val: Optional[np.ndarray] = None) -> bool:
        """Train stacking ensemble"""
        try:
            if not AI_AVAILABLE:
                return False
            
            # Create base models
            base_models = []
            for model_config in ensemble_model.base_models:
                model_name = model_config['name']
                model_params = model_config.get('parameters', {})
                
                if model_name in self.base_model_registry:
                    model_class = self.base_model_registry[model_name]
                    model = model_class(**model_params)
                    base_models.append((model_name, model))
            
            # Create meta model
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                meta_model = LogisticRegression()
            else:
                meta_model = LinearRegression()
            
            # Create stacking ensemble
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                stacking_ensemble = VotingClassifier(
                    estimators=base_models,
                    voting='soft'
                )
            else:
                stacking_ensemble = VotingRegressor(
                    estimators=base_models
                )
            
            # Train ensemble
            stacking_ensemble.fit(X_train, y_train)
            
            # Store meta model
            ensemble_model.meta_model = stacking_ensemble
            
            return True
            
        except Exception as e:
            logger.error(f"Error training stacking ensemble: {e}")
            return False
    
    async def _train_default_ensemble(self, ensemble_model: EnsembleModel, 
                                    X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Train default ensemble"""
        try:
            if not AI_AVAILABLE:
                return False
            
            # Create base models
            base_models = []
            for model_config in ensemble_model.base_models:
                model_name = model_config['name']
                model_params = model_config.get('parameters', {})
                
                if model_name in self.base_model_registry:
                    model_class = self.base_model_registry[model_name]
                    model = model_class(**model_params)
                    base_models.append((model_name, model))
            
            # Create voting ensemble as default
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                ensemble = VotingClassifier(
                    estimators=base_models,
                    voting='soft'
                )
            else:
                ensemble = VotingRegressor(
                    estimators=base_models,
                    weights=ensemble_model.weights
                )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Store meta model
            ensemble_model.meta_model = ensemble
            
            return True
            
        except Exception as e:
            logger.error(f"Error training default ensemble: {e}")
            return False
    
    async def predict_with_ensemble(self, model_id: str, X: np.ndarray) -> str:
        """Make prediction with ensemble model"""
        try:
            if model_id not in self.ensemble_models:
                raise ValueError(f"Ensemble model {model_id} not found")
            
            ensemble_model = self.ensemble_models[model_id]
            
            if ensemble_model.meta_model is None:
                raise ValueError("Ensemble model not trained")
            
            # Make prediction
            prediction = ensemble_model.meta_model.predict(X)
            
            # Get individual model predictions if available
            individual_predictions = []
            if hasattr(ensemble_model.meta_model, 'estimators_'):
                for estimator in ensemble_model.meta_model.estimators_:
                    individual_predictions.append(estimator.predict(X))
            
            # Calculate confidence scores
            confidence_scores = []
            if hasattr(ensemble_model.meta_model, 'predict_proba'):
                proba = ensemble_model.meta_model.predict_proba(X)
                confidence_scores = np.max(proba, axis=1).tolist()
            else:
                confidence_scores = [0.5] * len(prediction)
            
            # Calculate uncertainty
            uncertainty = 1.0 - np.mean(confidence_scores) if confidence_scores else 0.5
            
            # Create prediction record
            prediction_id = str(uuid.uuid4())
            ensemble_prediction = EnsemblePrediction(
                prediction_id=prediction_id,
                model_id=model_id,
                input_data=X.tolist(),
                predictions=individual_predictions,
                ensemble_prediction=prediction.tolist(),
                confidence_scores=confidence_scores,
                uncertainty=uncertainty,
                created_at=datetime.now()
            )
            
            # Store prediction
            self.ensemble_predictions[prediction_id] = ensemble_prediction
            
            # Update metrics
            self._update_metrics()
            
            logger.info(f"Ensemble prediction made: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    async def get_ensemble_model(self, model_id: str) -> Optional[EnsembleModel]:
        """Get ensemble model by ID"""
        return self.ensemble_models.get(model_id)
    
    async def get_ensemble_prediction(self, prediction_id: str) -> Optional[EnsemblePrediction]:
        """Get ensemble prediction by ID"""
        return self.ensemble_predictions.get(prediction_id)
    
    async def evaluate_ensemble_model(self, model_id: str, X_test: np.ndarray, 
                                    y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model performance"""
        try:
            if model_id not in self.ensemble_models:
                raise ValueError(f"Ensemble model {model_id} not found")
            
            ensemble_model = self.ensemble_models[model_id]
            
            if ensemble_model.meta_model is None:
                raise ValueError("Ensemble model not trained")
            
            # Make predictions
            y_pred = ensemble_model.meta_model.predict(X_test)
            
            # Calculate performance metrics
            if ensemble_model.ensemble_type == EnsembleType.CLASSIFICATION:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }
            
            # Update ensemble model performance
            ensemble_model.performance_metrics = metrics
            ensemble_model.updated_at = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            return {}
    
    def _update_metrics(self):
        """Update ensemble service metrics"""
        try:
            self.metrics.total_models = len(self.ensemble_models)
            self.metrics.total_predictions = len(self.ensemble_predictions)
            
            # Calculate average accuracy
            if self.ensemble_models:
                accuracies = [model.performance_metrics.get('accuracy', 0.0) 
                            for model in self.ensemble_models.values() 
                            if 'accuracy' in model.performance_metrics]
                if accuracies:
                    self.metrics.average_accuracy = sum(accuracies) / len(accuracies)
            
            # Calculate average confidence
            if self.ensemble_predictions:
                confidences = [pred.confidence_scores for pred in self.ensemble_predictions.values()]
                all_confidences = [conf for conf_list in confidences for conf in conf_list]
                if all_confidences:
                    self.metrics.average_confidence = sum(all_confidences) / len(all_confidences)
            
            # Calculate ensemble diversity
            if self.ensemble_models:
                diversity_scores = []
                for model in self.ensemble_models.values():
                    if len(model.base_models) > 1:
                        # Diversity is higher with more different base models
                        diversity = min(1.0, len(model.base_models) / 10.0)
                        diversity_scores.append(diversity)
                
                if diversity_scores:
                    self.metrics.ensemble_diversity = sum(diversity_scores) / len(diversity_scores)
            
            # Calculate prediction consensus
            if self.ensemble_predictions:
                consensus_scores = []
                for pred in self.ensemble_predictions.values():
                    if pred.predictions and len(pred.predictions) > 1:
                        # Consensus is higher when individual predictions are similar
                        individual_preds = np.array(pred.predictions)
                        consensus = 1.0 - np.std(individual_preds) / (np.mean(individual_preds) + 1e-8)
                        consensus_scores.append(max(0.0, consensus))
                
                if consensus_scores:
                    self.metrics.prediction_consensus = sum(consensus_scores) / len(consensus_scores)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_ensemble_service_status(self) -> Dict[str, Any]:
        """Get ensemble service status"""
        return {
            'total_models': self.metrics.total_models,
            'total_predictions': self.metrics.total_predictions,
            'average_accuracy': self.metrics.average_accuracy,
            'average_confidence': self.metrics.average_confidence,
            'ensemble_diversity': self.metrics.ensemble_diversity,
            'prediction_consensus': self.metrics.prediction_consensus,
            'config': self.config,
            'ai_available': AI_AVAILABLE
        }

# Global instance
ensemble_service = EnsembleService()




