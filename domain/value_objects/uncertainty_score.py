"""
Uncertainty Score Value Object
Advanced uncertainty quantification with epistemic and aleatoric uncertainty
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import entropy, beta
import json
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty in AI models"""
    EPISTEMIC = "epistemic"      # Model uncertainty (reducible)
    ALEATORIC = "aleatoric"     # Data uncertainty (irreducible)
    TOTAL = "total"             # Combined uncertainty
    PREDICTIVE = "predictive"   # Predictive uncertainty
    MODEL = "model"            # Model parameter uncertainty
    DATA = "data"              # Data noise uncertainty


class UncertaintyLevel(Enum):
    """Uncertainty levels with confidence bounds"""
    NEGLIGIBLE = "negligible"    # < 0.05
    LOW = "low"                  # 0.05 - 0.20
    MODERATE = "moderate"        # 0.20 - 0.40
    MEDIUM = "medium"            # 0.20 - 0.40 (alias for MODERATE)
    HIGH = "high"                # 0.40 - 0.60
    VERY_HIGH = "very_high"      # 0.60 - 0.80
    EXTREME = "extreme"          # > 0.80


@dataclass(frozen=True)
class UncertaintyScore:
    """
    Immutable value object representing uncertainty scores
    Supports multiple uncertainty types and confidence intervals
    """
    
    # Core uncertainty values
    epistemic_uncertainty: Decimal
    aleatoric_uncertainty: Decimal
    total_uncertainty: Decimal
    
    # Confidence intervals
    confidence_interval_95: Tuple[Decimal, Decimal]
    confidence_interval_99: Tuple[Decimal, Decimal]
    
    # Uncertainty metadata
    uncertainty_type: UncertaintyType
    level: UncertaintyLevel
    sample_size: Optional[int] = None
    model_complexity: Optional[Decimal] = None
    
    # Advanced metrics
    entropy: Optional[Decimal] = None
    mutual_information: Optional[Decimal] = None
    information_gain: Optional[Decimal] = None
    
    # Calibration metrics
    calibration_error: Optional[Decimal] = None
    reliability_diagram: Optional[List[Tuple[Decimal, Decimal]]] = None
    
    # Temporal properties
    uncertainty_trend: Optional[List[Decimal]] = None
    volatility: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate uncertainty score invariants"""
        # Validate uncertainty values
        for uncertainty in [self.epistemic_uncertainty, self.aleatoric_uncertainty, self.total_uncertainty]:
            if not (0 <= uncertainty <= 1):
                raise ValueError("Uncertainty values must be between 0 and 1")
        
        # Validate confidence intervals
        ci_95_low, ci_95_high = self.confidence_interval_95
        ci_99_low, ci_99_high = self.confidence_interval_99
        
        if not (0 <= ci_95_low <= ci_95_high <= 1):
            raise ValueError("95% confidence interval must be within [0, 1]")
        
        if not (0 <= ci_99_low <= ci_99_high <= 1):
            raise ValueError("99% confidence interval must be within [0, 1]")
        
        # Validate total uncertainty calculation
        expected_total = self.epistemic_uncertainty + self.aleatoric_uncertainty
        if not abs(self.total_uncertainty - expected_total) < Decimal('0.01'):
            raise ValueError("Total uncertainty should equal epistemic + aleatoric")
        
        # Round to 4 decimal places
        rounded_epistemic = self.epistemic_uncertainty.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_aleatoric = self.aleatoric_uncertainty.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        rounded_total = self.total_uncertainty.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        
        object.__setattr__(self, 'epistemic_uncertainty', rounded_epistemic)
        object.__setattr__(self, 'aleatoric_uncertainty', rounded_aleatoric)
        object.__setattr__(self, 'total_uncertainty', rounded_total)
    
    def __str__(self) -> str:
        """String representation"""
        return f"Uncertainty(epistemic={self.epistemic_uncertainty:.4f}, aleatoric={self.aleatoric_uncertainty:.4f}, total={self.total_uncertainty:.4f})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"UncertaintyScore(epistemic={self.epistemic_uncertainty}, aleatoric={self.aleatoric_uncertainty}, total={self.total_uncertainty})"
    
    @classmethod
    def from_predictions(cls, predictions: List[float], uncertainty_type: UncertaintyType = UncertaintyType.TOTAL) -> 'UncertaintyScore':
        """Create uncertainty score from prediction distribution"""
        predictions = np.array(predictions)
        
        # Calculate epistemic uncertainty (model uncertainty)
        epistemic = np.var(predictions)
        
        # Calculate aleatoric uncertainty (data noise)
        # This is a simplified calculation - in practice would use more sophisticated methods
        aleatoric = np.mean(predictions) * (1 - np.mean(predictions))  # Simplified for binary case
        
        # Calculate total uncertainty
        total = epistemic + aleatoric
        
        # Calculate confidence intervals
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        ci_95_low = max(0, mean_pred - 1.96 * std_pred)
        ci_95_high = min(1, mean_pred + 1.96 * std_pred)
        
        ci_99_low = max(0, mean_pred - 2.58 * std_pred)
        ci_99_high = min(1, mean_pred + 2.58 * std_pred)
        
        # Determine uncertainty level
        level = cls._determine_level(total)
        
        return cls(
            epistemic_uncertainty=Decimal(str(epistemic)),
            aleatoric_uncertainty=Decimal(str(aleatoric)),
            total_uncertainty=Decimal(str(total)),
            confidence_interval_95=(Decimal(str(ci_95_low)), Decimal(str(ci_95_high))),
            confidence_interval_99=(Decimal(str(ci_99_low)), Decimal(str(ci_99_high))),
            uncertainty_type=uncertainty_type,
            level=level,
            sample_size=len(predictions)
        )
    
    @classmethod
    def from_ensemble(cls, ensemble_predictions: List[List[float]], uncertainty_type: UncertaintyType = UncertaintyType.TOTAL) -> 'UncertaintyScore':
        """Create uncertainty score from ensemble predictions"""
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate epistemic uncertainty (disagreement between models)
        model_means = np.mean(ensemble_predictions, axis=1)
        epistemic = np.var(model_means)
        
        # Calculate aleatoric uncertainty (average within-model variance)
        aleatoric = np.mean(np.var(ensemble_predictions, axis=1))
        
        # Calculate total uncertainty
        total = epistemic + aleatoric
        
        # Calculate confidence intervals
        all_predictions = ensemble_predictions.flatten()
        mean_pred = np.mean(all_predictions)
        std_pred = np.std(all_predictions)
        
        ci_95_low = max(0, mean_pred - 1.96 * std_pred)
        ci_95_high = min(1, mean_pred + 1.96 * std_pred)
        
        ci_99_low = max(0, mean_pred - 2.58 * std_pred)
        ci_99_high = min(1, mean_pred + 2.58 * std_pred)
        
        # Determine uncertainty level
        level = cls._determine_level(total)
        
        return cls(
            epistemic_uncertainty=Decimal(str(epistemic)),
            aleatoric_uncertainty=Decimal(str(aleatoric)),
            total_uncertainty=Decimal(str(total)),
            confidence_interval_95=(Decimal(str(ci_95_low)), Decimal(str(ci_95_high))),
            confidence_interval_99=(Decimal(str(ci_99_low)), Decimal(str(ci_99_high))),
            uncertainty_type=uncertainty_type,
            level=level,
            sample_size=ensemble_predictions.size
        )
    
    @classmethod
    def from_bayesian(cls, posterior_samples: List[float], uncertainty_type: UncertaintyType = UncertaintyType.TOTAL) -> 'UncertaintyScore':
        """Create uncertainty score from Bayesian posterior samples"""
        posterior_samples = np.array(posterior_samples)
        
        # Calculate epistemic uncertainty (posterior variance)
        epistemic = np.var(posterior_samples)
        
        # Calculate aleatoric uncertainty (likelihood variance)
        # This is a simplified calculation
        aleatoric = np.mean(posterior_samples) * (1 - np.mean(posterior_samples))
        
        # Calculate total uncertainty
        total = epistemic + aleatoric
        
        # Calculate confidence intervals from posterior
        ci_95_low = np.percentile(posterior_samples, 2.5)
        ci_95_high = np.percentile(posterior_samples, 97.5)
        
        ci_99_low = np.percentile(posterior_samples, 0.5)
        ci_99_high = np.percentile(posterior_samples, 99.5)
        
        # Determine uncertainty level
        level = cls._determine_level(total)
        
        return cls(
            epistemic_uncertainty=Decimal(str(epistemic)),
            aleatoric_uncertainty=Decimal(str(aleatoric)),
            total_uncertainty=Decimal(str(total)),
            confidence_interval_95=(Decimal(str(ci_95_low)), Decimal(str(ci_95_high))),
            confidence_interval_99=(Decimal(str(ci_99_low)), Decimal(str(ci_99_high))),
            uncertainty_type=uncertainty_type,
            level=level,
            sample_size=len(posterior_samples)
        )
    
    @classmethod
    def _determine_level(cls, total_uncertainty: float) -> UncertaintyLevel:
        """Determine uncertainty level from total uncertainty"""
        if total_uncertainty < 0.05:
            return UncertaintyLevel.NEGLIGIBLE
        elif total_uncertainty < 0.20:
            return UncertaintyLevel.LOW
        elif total_uncertainty < 0.40:
            return UncertaintyLevel.MODERATE
        elif total_uncertainty < 0.60:
            return UncertaintyLevel.HIGH
        elif total_uncertainty < 0.80:
            return UncertaintyLevel.VERY_HIGH
        else:
            return UncertaintyLevel.EXTREME
    
    def is_high_uncertainty(self) -> bool:
        """Check if uncertainty is high"""
        return self.level in [UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH, UncertaintyLevel.EXTREME]
    
    def is_low_uncertainty(self) -> bool:
        """Check if uncertainty is low"""
        return self.level in [UncertaintyLevel.NEGLIGIBLE, UncertaintyLevel.LOW]
    
    def is_epistemic_dominant(self) -> bool:
        """Check if epistemic uncertainty dominates"""
        return self.epistemic_uncertainty > self.aleatoric_uncertainty
    
    def is_aleatoric_dominant(self) -> bool:
        """Check if aleatoric uncertainty dominates"""
        return self.aleatoric_uncertainty > self.epistemic_uncertainty
    
    def get_uncertainty_ratio(self) -> Decimal:
        """Get ratio of epistemic to aleatoric uncertainty"""
        if self.aleatoric_uncertainty == 0:
            return Decimal('inf')
        return self.epistemic_uncertainty / self.aleatoric_uncertainty
    
    def get_confidence_width(self, confidence_level: float = 0.95) -> Decimal:
        """Get confidence interval width"""
        if confidence_level == 0.95:
            ci_low, ci_high = self.confidence_interval_95
        elif confidence_level == 0.99:
            ci_low, ci_high = self.confidence_interval_99
        else:
            raise ValueError("Confidence level must be 0.95 or 0.99")
        
        return ci_high - ci_low
    
    def is_well_calibrated(self, threshold: Decimal = Decimal('0.1')) -> bool:
        """Check if uncertainty is well calibrated"""
        if self.calibration_error is None:
            return False
        return self.calibration_error < threshold
    
    def get_uncertainty_trend(self) -> Optional[str]:
        """Get uncertainty trend direction"""
        if self.uncertainty_trend is None or len(self.uncertainty_trend) < 2:
            return None
        
        recent_trend = self.uncertainty_trend[-5:]  # Last 5 points
        if len(recent_trend) < 2:
            return None
        
        # Simple trend detection
        first_half = np.mean(recent_trend[:len(recent_trend)//2])
        second_half = np.mean(recent_trend[len(recent_trend)//2:])
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def combine_with(self, other: 'UncertaintyScore', weight: Decimal = Decimal('0.5')) -> 'UncertaintyScore':
        """Combine with another uncertainty score"""
        if not isinstance(other, UncertaintyScore):
            raise TypeError("Can only combine with UncertaintyScore")
        
        # Weighted combination
        combined_epistemic = weight * self.epistemic_uncertainty + (1 - weight) * other.epistemic_uncertainty
        combined_aleatoric = weight * self.aleatoric_uncertainty + (1 - weight) * other.aleatoric_uncertainty
        combined_total = combined_epistemic + combined_aleatoric
        
        # Combine confidence intervals
        ci_95_low = min(self.confidence_interval_95[0], other.confidence_interval_95[0])
        ci_95_high = max(self.confidence_interval_95[1], other.confidence_interval_95[1])
        
        ci_99_low = min(self.confidence_interval_99[0], other.confidence_interval_99[0])
        ci_99_high = max(self.confidence_interval_99[1], other.confidence_interval_99[1])
        
        # Determine combined level
        combined_level = self._determine_level(float(combined_total))
        
        return UncertaintyScore(
            epistemic_uncertainty=combined_epistemic,
            aleatoric_uncertainty=combined_aleatoric,
            total_uncertainty=combined_total,
            confidence_interval_95=(ci_95_low, ci_95_high),
            confidence_interval_99=(ci_99_low, ci_99_high),
            uncertainty_type=self.uncertainty_type,
            level=combined_level,
            sample_size=(self.sample_size or 0) + (other.sample_size or 0)
        )
    
    def to_json(self) -> str:
        """Convert to JSON representation"""
        uncertainty_dict = {
            'epistemic_uncertainty': float(self.epistemic_uncertainty),
            'aleatoric_uncertainty': float(self.aleatoric_uncertainty),
            'total_uncertainty': float(self.total_uncertainty),
            'confidence_interval_95': [float(self.confidence_interval_95[0]), float(self.confidence_interval_95[1])],
            'confidence_interval_99': [float(self.confidence_interval_99[0]), float(self.confidence_interval_99[1])],
            'uncertainty_type': self.uncertainty_type.value,
            'level': self.level.value,
            'sample_size': self.sample_size,
            'model_complexity': float(self.model_complexity) if self.model_complexity else None,
            'entropy': float(self.entropy) if self.entropy else None,
            'mutual_information': float(self.mutual_information) if self.mutual_information else None,
            'information_gain': float(self.information_gain) if self.information_gain else None,
            'calibration_error': float(self.calibration_error) if self.calibration_error else None,
            'uncertainty_trend': [float(x) for x in self.uncertainty_trend] if self.uncertainty_trend else None,
            'volatility': float(self.volatility) if self.volatility else None
        }
        return json.dumps(uncertainty_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UncertaintyScore':
        """Create from JSON representation"""
        uncertainty_dict = json.loads(json_str)
        
        return cls(
            epistemic_uncertainty=Decimal(str(uncertainty_dict['epistemic_uncertainty'])),
            aleatoric_uncertainty=Decimal(str(uncertainty_dict['aleatoric_uncertainty'])),
            total_uncertainty=Decimal(str(uncertainty_dict['total_uncertainty'])),
            confidence_interval_95=(Decimal(str(uncertainty_dict['confidence_interval_95'][0])), 
                                 Decimal(str(uncertainty_dict['confidence_interval_95'][1]))),
            confidence_interval_99=(Decimal(str(uncertainty_dict['confidence_interval_99'][0])), 
                                 Decimal(str(uncertainty_dict['confidence_interval_99'][1]))),
            uncertainty_type=UncertaintyType(uncertainty_dict['uncertainty_type']),
            level=UncertaintyLevel(uncertainty_dict['level']),
            sample_size=uncertainty_dict.get('sample_size'),
            model_complexity=Decimal(str(uncertainty_dict['model_complexity'])) if uncertainty_dict.get('model_complexity') else None,
            entropy=Decimal(str(uncertainty_dict['entropy'])) if uncertainty_dict.get('entropy') else None,
            mutual_information=Decimal(str(uncertainty_dict['mutual_information'])) if uncertainty_dict.get('mutual_information') else None,
            information_gain=Decimal(str(uncertainty_dict['information_gain'])) if uncertainty_dict.get('information_gain') else None,
            calibration_error=Decimal(str(uncertainty_dict['calibration_error'])) if uncertainty_dict.get('calibration_error') else None,
            uncertainty_trend=[Decimal(str(x)) for x in uncertainty_dict['uncertainty_trend']] if uncertainty_dict.get('uncertainty_trend') else None,
            volatility=Decimal(str(uncertainty_dict['volatility'])) if uncertainty_dict.get('volatility') else None
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, UncertaintyScore):
            return False
        
        return (self.epistemic_uncertainty == other.epistemic_uncertainty and
                self.aleatoric_uncertainty == other.aleatoric_uncertainty and
                self.total_uncertainty == other.total_uncertainty and
                self.uncertainty_type == other.uncertainty_type and
                self.level == other.level)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((self.epistemic_uncertainty, self.aleatoric_uncertainty, self.total_uncertainty, self.uncertainty_type, self.level))
