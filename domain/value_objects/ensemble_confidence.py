"""
Ensemble Confidence Value Object
Advanced ensemble confidence with model agreement, diversity, and uncertainty quantification
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import entropy, beta
import json
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels with statistical significance"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                 # 0.3 - 0.5
    MODERATE = "moderate"       # 0.5 - 0.7
    HIGH = "high"               # 0.7 - 0.85
    VERY_HIGH = "very_high"     # 0.85 - 0.95
    EXTREME = "extreme"         # > 0.95


class ModelAgreement(Enum):
    """Model agreement levels"""
    DISAGREEMENT = "disagreement"    # < 0.3
    WEAK_AGREEMENT = "weak_agreement"  # 0.3 - 0.5
    MODERATE_AGREEMENT = "moderate_agreement"  # 0.5 - 0.7
    STRONG_AGREEMENT = "strong_agreement"  # 0.7 - 0.9
    STRONG = "strong"                # 0.7 - 0.9 (alias for STRONG_AGREEMENT)
    CONSENSUS = "consensus"         # > 0.9


@dataclass(frozen=True)
class EnsembleConfidence:
    """
    Immutable value object representing ensemble confidence
    Incorporates model agreement, diversity, and uncertainty quantification
    """
    
    # Core confidence values
    value: Decimal
    model_agreement: Decimal
    model_diversity: Decimal
    prediction_variance: Decimal
    
    # Statistical properties
    confidence_level: ConfidenceLevel
    agreement_level: ModelAgreement
    sample_size: int
    
    # Model ensemble properties
    num_models: int
    model_weights: Optional[List[Decimal]] = None
    model_predictions: Optional[List[Decimal]] = None
    
    # Advanced metrics
    entropy: Optional[Decimal] = None
    mutual_information: Optional[Decimal] = None
    information_gain: Optional[Decimal] = None
    
    # Calibration metrics
    calibration_error: Optional[Decimal] = None
    reliability_score: Optional[Decimal] = None
    
    # Temporal properties
    confidence_trend: Optional[List[Decimal]] = None
    stability: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate ensemble confidence invariants"""
        # Validate core values
        for value_name, value in [
            ('value', self.value),
            ('model_agreement', self.model_agreement),
            ('model_diversity', self.model_diversity),
            ('prediction_variance', self.prediction_variance)
        ]:
            if not (0 <= value <= 1):
                raise ValueError(f"{value_name} must be between 0 and 1")
        
        # Validate sample size
        if self.sample_size < 1:
            raise ValueError("Sample size must be at least 1")
        
        # Validate number of models
        if self.num_models < 1:
            raise ValueError("Number of models must be at least 1")
        
        # Validate model weights if provided
        if self.model_weights is not None:
            if len(self.model_weights) != self.num_models:
                raise ValueError("Number of model weights must match number of models")
            
            weight_sum = sum(self.model_weights)
            if not abs(weight_sum - Decimal('1.0')) < Decimal('0.01'):
                raise ValueError("Model weights must sum to 1.0")
            
            for weight in self.model_weights:
                if not (0 <= weight <= 1):
                    raise ValueError("Model weights must be between 0 and 1")
        
        # Validate model predictions if provided
        if self.model_predictions is not None:
            if len(self.model_predictions) != self.num_models:
                raise ValueError("Number of model predictions must match number of models")
            
            for prediction in self.model_predictions:
                if not (0 <= prediction <= 1):
                    raise ValueError("Model predictions must be between 0 and 1")
        
        # Round to 4 decimal places
        rounded_value = self.value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_agreement = self.model_agreement.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_diversity = self.model_diversity.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_variance = self.prediction_variance.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        
        object.__setattr__(self, 'value', rounded_value)
        object.__setattr__(self, 'model_agreement', rounded_agreement)
        object.__setattr__(self, 'model_diversity', rounded_diversity)
        object.__setattr__(self, 'prediction_variance', rounded_variance)
    
    def __str__(self) -> str:
        """String representation"""
        return f"EnsembleConfidence(value={self.value:.4f}, agreement={self.model_agreement:.4f}, diversity={self.model_diversity:.4f})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"EnsembleConfidence(value={self.value}, agreement={self.model_agreement}, diversity={self.model_diversity})"
    
    @classmethod
    def from_predictions(cls, predictions: List[float], weights: Optional[List[float]] = None) -> 'EnsembleConfidence':
        """Create ensemble confidence from model predictions"""
        predictions = np.array(predictions)
        num_models = len(predictions)
        
        # Calculate model agreement (inverse of variance)
        mean_pred = np.mean(predictions)
        variance = np.var(predictions)
        agreement = 1.0 - variance  # Higher variance = lower agreement
        
        # Calculate model diversity (entropy of predictions)
        # Normalize predictions to probability distribution
        pred_probs = predictions / np.sum(predictions) if np.sum(predictions) > 0 else np.ones_like(predictions) / len(predictions)
        diversity = entropy(pred_probs) / np.log(num_models)  # Normalized entropy
        
        # Calculate prediction variance
        pred_variance = variance
        
        # Calculate ensemble confidence
        ensemble_value = mean_pred * agreement * (1 - diversity)  # Weighted by agreement and diversity
        
        # Determine confidence level
        confidence_level = cls._determine_confidence_level(ensemble_value)
        
        # Determine agreement level
        agreement_level = cls._determine_agreement_level(agreement)
        
        # Convert weights if provided
        model_weights = None
        if weights is not None:
            model_weights = [Decimal(str(w)) for w in weights]
        
        # Convert predictions
        model_predictions = [Decimal(str(p)) for p in predictions]
        
        return cls(
            value=Decimal(str(ensemble_value)),
            model_agreement=Decimal(str(agreement)),
            model_diversity=Decimal(str(diversity)),
            prediction_variance=Decimal(str(pred_variance)),
            confidence_level=confidence_level,
            agreement_level=agreement_level,
            sample_size=num_models,
            num_models=num_models,
            model_weights=model_weights,
            model_predictions=model_predictions
        )
    
    @classmethod
    def from_ensemble_weights(cls, predictions: List[float], weights: List[float], sample_size: int) -> 'EnsembleConfidence':
        """Create ensemble confidence with model weights"""
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Weighted ensemble prediction
        weighted_pred = np.average(predictions, weights=weights)
        
        # Calculate weighted variance
        weighted_variance = np.average((predictions - weighted_pred) ** 2, weights=weights)
        
        # Calculate model agreement (inverse of weighted variance)
        agreement = 1.0 - weighted_variance
        
        # Calculate model diversity (entropy of weights)
        weight_entropy = entropy(weights) / np.log(len(weights))
        diversity = weight_entropy
        
        # Calculate ensemble confidence
        ensemble_value = weighted_pred * agreement * (1 - diversity)
        
        # Determine levels
        confidence_level = cls._determine_confidence_level(ensemble_value)
        agreement_level = cls._determine_agreement_level(agreement)
        
        return cls(
            value=Decimal(str(ensemble_value)),
            model_agreement=Decimal(str(agreement)),
            model_diversity=Decimal(str(diversity)),
            prediction_variance=Decimal(str(weighted_variance)),
            confidence_level=confidence_level,
            agreement_level=agreement_level,
            sample_size=sample_size,
            num_models=len(predictions),
            model_weights=[Decimal(str(w)) for w in weights],
            model_predictions=[Decimal(str(p)) for p in predictions]
        )
    
    @classmethod
    def from_bootstrap(cls, bootstrap_samples: List[List[float]]) -> 'EnsembleConfidence':
        """Create ensemble confidence from bootstrap samples"""
        bootstrap_samples = np.array(bootstrap_samples)
        num_models = bootstrap_samples.shape[0]
        num_samples = bootstrap_samples.shape[1]
        
        # Calculate mean predictions across bootstrap samples
        mean_predictions = np.mean(bootstrap_samples, axis=1)
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(mean_predictions)
        
        # Calculate model agreement (inverse of variance across models)
        model_variance = np.var(mean_predictions)
        agreement = 1.0 - model_variance
        
        # Calculate model diversity (entropy of mean predictions)
        pred_probs = mean_predictions / np.sum(mean_predictions) if np.sum(mean_predictions) > 0 else np.ones_like(mean_predictions) / len(mean_predictions)
        diversity = entropy(pred_probs) / np.log(num_models)
        
        # Calculate prediction variance (average across bootstrap samples)
        pred_variance = np.mean(np.var(bootstrap_samples, axis=1))
        
        # Calculate ensemble confidence
        ensemble_value = ensemble_pred * agreement * (1 - diversity)
        
        # Determine levels
        confidence_level = cls._determine_confidence_level(ensemble_value)
        agreement_level = cls._determine_agreement_level(agreement)
        
        return cls(
            value=Decimal(str(ensemble_value)),
            model_agreement=Decimal(str(agreement)),
            model_diversity=Decimal(str(diversity)),
            prediction_variance=Decimal(str(pred_variance)),
            confidence_level=confidence_level,
            agreement_level=agreement_level,
            sample_size=num_samples,
            num_models=num_models
        )
    
    @classmethod
    def _determine_confidence_level(cls, confidence_value: float) -> ConfidenceLevel:
        """Determine confidence level from value"""
        if confidence_value < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence_value < 0.5:
            return ConfidenceLevel.LOW
        elif confidence_value < 0.7:
            return ConfidenceLevel.MODERATE
        elif confidence_value < 0.85:
            return ConfidenceLevel.HIGH
        elif confidence_value < 0.95:
            return ConfidenceLevel.VERY_HIGH
        else:
            return ConfidenceLevel.EXTREME
    
    @classmethod
    def _determine_agreement_level(cls, agreement_value: float) -> ModelAgreement:
        """Determine agreement level from value"""
        if agreement_value < 0.3:
            return ModelAgreement.DISAGREEMENT
        elif agreement_value < 0.5:
            return ModelAgreement.WEAK_AGREEMENT
        elif agreement_value < 0.7:
            return ModelAgreement.MODERATE_AGREEMENT
        elif agreement_value < 0.9:
            return ModelAgreement.STRONG_AGREEMENT
        else:
            return ModelAgreement.CONSENSUS
    
    def is_high_confidence(self) -> bool:
        """Check if confidence is high"""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH, ConfidenceLevel.EXTREME]
    
    def is_low_confidence(self) -> bool:
        """Check if confidence is low"""
        return self.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]
    
    def has_strong_agreement(self) -> bool:
        """Check if models have strong agreement"""
        return self.agreement_level in [ModelAgreement.STRONG_AGREEMENT, ModelAgreement.CONSENSUS]
    
    def has_weak_agreement(self) -> bool:
        """Check if models have weak agreement"""
        return self.agreement_level in [ModelAgreement.DISAGREEMENT, ModelAgreement.WEAK_AGREEMENT]
    
    def get_agreement_ratio(self) -> Decimal:
        """Get ratio of agreement to diversity"""
        if self.model_diversity == 0:
            return Decimal('inf')
        return self.model_agreement / self.model_diversity
    
    def get_confidence_stability(self) -> Optional[Decimal]:
        """Get confidence stability score"""
        if self.stability is not None:
            return self.stability
        
        if self.confidence_trend is None or len(self.confidence_trend) < 2:
            return None
        
        # Calculate stability as inverse of variance
        trend_array = np.array([float(x) for x in self.confidence_trend])
        stability = 1.0 - np.var(trend_array)
        return Decimal(str(max(0, stability)))
    
    def get_model_contribution(self, model_index: int) -> Optional[Decimal]:
        """Get contribution of specific model to ensemble"""
        if self.model_weights is None or model_index >= len(self.model_weights):
            return None
        
        return self.model_weights[model_index]
    
    def get_prediction_range(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get range of model predictions"""
        if self.model_predictions is None:
            return None
        
        predictions = [float(p) for p in self.model_predictions]
        return (Decimal(str(min(predictions))), Decimal(str(max(predictions))))
    
    def combine_with(self, other: 'EnsembleConfidence', weight: Decimal = Decimal('0.5')) -> 'EnsembleConfidence':
        """Combine with another ensemble confidence"""
        if not isinstance(other, EnsembleConfidence):
            raise TypeError("Can only combine with EnsembleConfidence")
        
        # Weighted combination
        combined_value = weight * self.value + (1 - weight) * other.value
        combined_agreement = weight * self.model_agreement + (1 - weight) * other.model_agreement
        combined_diversity = weight * self.model_diversity + (1 - weight) * other.model_diversity
        combined_variance = weight * self.prediction_variance + (1 - weight) * other.prediction_variance
        
        # Determine combined levels
        combined_confidence_level = self._determine_confidence_level(float(combined_value))
        combined_agreement_level = self._determine_agreement_level(float(combined_agreement))
        
        # Combine model information
        combined_num_models = self.num_models + other.num_models
        combined_sample_size = self.sample_size + other.sample_size
        
        # Combine model weights and predictions if available
        combined_weights = None
        combined_predictions = None
        
        if self.model_weights is not None and other.model_weights is not None:
            combined_weights = self.model_weights + other.model_weights
        
        if self.model_predictions is not None and other.model_predictions is not None:
            combined_predictions = self.model_predictions + other.model_predictions
        
        return EnsembleConfidence(
            value=combined_value,
            model_agreement=combined_agreement,
            model_diversity=combined_diversity,
            prediction_variance=combined_variance,
            confidence_level=combined_confidence_level,
            agreement_level=combined_agreement_level,
            sample_size=combined_sample_size,
            num_models=combined_num_models,
            model_weights=combined_weights,
            model_predictions=combined_predictions
        )
    
    def to_json(self) -> str:
        """Convert to JSON representation"""
        confidence_dict = {
            'value': float(self.value),
            'model_agreement': float(self.model_agreement),
            'model_diversity': float(self.model_diversity),
            'prediction_variance': float(self.prediction_variance),
            'confidence_level': self.confidence_level.value,
            'agreement_level': self.agreement_level.value,
            'sample_size': self.sample_size,
            'num_models': self.num_models,
            'model_weights': [float(w) for w in self.model_weights] if self.model_weights else None,
            'model_predictions': [float(p) for p in self.model_predictions] if self.model_predictions else None,
            'entropy': float(self.entropy) if self.entropy else None,
            'mutual_information': float(self.mutual_information) if self.mutual_information else None,
            'information_gain': float(self.information_gain) if self.information_gain else None,
            'calibration_error': float(self.calibration_error) if self.calibration_error else None,
            'reliability_score': float(self.reliability_score) if self.reliability_score else None,
            'confidence_trend': [float(x) for x in self.confidence_trend] if self.confidence_trend else None,
            'stability': float(self.stability) if self.stability else None
        }
        return json.dumps(confidence_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EnsembleConfidence':
        """Create from JSON representation"""
        confidence_dict = json.loads(json_str)
        
        return cls(
            value=Decimal(str(confidence_dict['value'])),
            model_agreement=Decimal(str(confidence_dict['model_agreement'])),
            model_diversity=Decimal(str(confidence_dict['model_diversity'])),
            prediction_variance=Decimal(str(confidence_dict['prediction_variance'])),
            confidence_level=ConfidenceLevel(confidence_dict['confidence_level']),
            agreement_level=ModelAgreement(confidence_dict['agreement_level']),
            sample_size=confidence_dict['sample_size'],
            num_models=confidence_dict['num_models'],
            model_weights=[Decimal(str(w)) for w in confidence_dict['model_weights']] if confidence_dict.get('model_weights') else None,
            model_predictions=[Decimal(str(p)) for p in confidence_dict['model_predictions']] if confidence_dict.get('model_predictions') else None,
            entropy=Decimal(str(confidence_dict['entropy'])) if confidence_dict.get('entropy') else None,
            mutual_information=Decimal(str(confidence_dict['mutual_information'])) if confidence_dict.get('mutual_information') else None,
            information_gain=Decimal(str(confidence_dict['information_gain'])) if confidence_dict.get('information_gain') else None,
            calibration_error=Decimal(str(confidence_dict['calibration_error'])) if confidence_dict.get('calibration_error') else None,
            reliability_score=Decimal(str(confidence_dict['reliability_score'])) if confidence_dict.get('reliability_score') else None,
            confidence_trend=[Decimal(str(x)) for x in confidence_dict['confidence_trend']] if confidence_dict.get('confidence_trend') else None,
            stability=Decimal(str(confidence_dict['stability'])) if confidence_dict.get('stability') else None
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, EnsembleConfidence):
            return False
        
        return (self.value == other.value and
                self.model_agreement == other.model_agreement and
                self.model_diversity == other.model_diversity and
                self.confidence_level == other.confidence_level and
                self.agreement_level == other.agreement_level)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((self.value, self.model_agreement, self.model_diversity, self.confidence_level, self.agreement_level))
