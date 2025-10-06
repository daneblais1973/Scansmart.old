"""
Confidence Value Object
Immutable representation of confidence scores
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union


@dataclass(frozen=True)
class Confidence:
    """
    Immutable value object representing confidence scores
    """
    
    value: Decimal
    
    def __post_init__(self):
        """Validate confidence invariants"""
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))
        
        if self.value < 0 or self.value > 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        # Round to 4 decimal places
        rounded_value = self.value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        object.__setattr__(self, 'value', rounded_value)
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.value:.4f}"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"Confidence({self.value})"
    
    @classmethod
    def from_percentage(cls, percentage: Union[int, float, Decimal]) -> 'Confidence':
        """Create confidence from percentage (50% = 0.5)"""
        if isinstance(percentage, Decimal):
            return cls(percentage / 100)
        return cls(Decimal(str(percentage)) / 100)
    
    def to_percentage(self) -> Decimal:
        """Convert to percentage (0.5 = 50%)"""
        return self.value * 100
    
    def is_high(self) -> bool:
        """Check if confidence is high (> 0.8)"""
        return self.value > 0.8
    
    def is_medium(self) -> bool:
        """Check if confidence is medium (0.5 - 0.8)"""
        return 0.5 <= self.value <= 0.8
    
    def is_low(self) -> bool:
        """Check if confidence is low (< 0.5)"""
        return self.value < 0.5





