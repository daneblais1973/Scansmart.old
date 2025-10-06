"""
Percentage Value Object
Immutable representation of percentage values
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union


@dataclass(frozen=True)
class Percentage:
    """
    Immutable value object representing percentage values
    """
    
    value: Decimal
    
    def __post_init__(self):
        """Validate percentage invariants"""
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))
        
        if self.value < 0 or self.value > 100:
            raise ValueError("Percentage must be between 0 and 100")
        
        # Round to 2 decimal places
        rounded_value = self.value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        object.__setattr__(self, 'value', rounded_value)
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.value}%"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"Percentage({self.value})"
    
    @classmethod
    def from_decimal(cls, decimal_value: Decimal) -> 'Percentage':
        """Create percentage from decimal (0.1 = 10%)"""
        return cls(decimal_value * 100)
    
    @classmethod
    def from_fraction(cls, fraction: float) -> 'Percentage':
        """Create percentage from fraction (0.1 = 10%)"""
        return cls(Decimal(str(fraction * 100)))
    
    def to_decimal(self) -> Decimal:
        """Convert to decimal (10% = 0.1)"""
        return self.value / 100
    
    def to_fraction(self) -> float:
        """Convert to fraction (10% = 0.1)"""
        return float(self.to_decimal())
    
    def is_positive(self) -> bool:
        """Check if percentage is positive"""
        return self.value > 0
    
    def is_zero(self) -> bool:
        """Check if percentage is zero"""
        return self.value == 0





