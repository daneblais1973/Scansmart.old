"""
Money Value Object
Immutable representation of monetary values with currency
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Union


@dataclass(frozen=True)
class Money:
    """
    Immutable value object representing monetary amounts
    Enforces business rules around currency and precision
    """
    
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate money invariants"""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))
        
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")
        
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter code")
        
        # Round to 2 decimal places for currency precision
        rounded_amount = self.amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        object.__setattr__(self, 'amount', rounded_amount)
    
    def __add__(self, other: 'Money') -> 'Money':
        """Add two money amounts"""
        if not isinstance(other, Money):
            raise TypeError("Can only add Money to Money")
        
        if self.currency != other.currency:
            raise ValueError("Cannot add money with different currencies")
        
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """Subtract two money amounts"""
        if not isinstance(other, Money):
            raise TypeError("Can only subtract Money from Money")
        
        if self.currency != other.currency:
            raise ValueError("Cannot subtract money with different currencies")
        
        result = self.amount - other.amount
        if result < 0:
            raise ValueError("Money subtraction cannot result in negative amount")
        
        return Money(result, self.currency)
    
    def __mul__(self, multiplier: Union[Decimal, int, float]) -> 'Money':
        """Multiply money by a number"""
        if not isinstance(multiplier, (Decimal, int, float)):
            raise TypeError("Multiplier must be a number")
        
        return Money(self.amount * Decimal(str(multiplier)), self.currency)
    
    def __truediv__(self, divisor: Union[Decimal, int, float]) -> 'Money':
        """Divide money by a number"""
        if not isinstance(divisor, (Decimal, int, float)):
            raise TypeError("Divisor must be a number")
        
        divisor_decimal = Decimal(str(divisor))
        if divisor_decimal == 0:
            raise ValueError("Cannot divide by zero")
        
        return Money(self.amount / divisor_decimal, self.currency)
    
    def __lt__(self, other: 'Money') -> bool:
        """Less than comparison"""
        if not isinstance(other, Money):
            raise TypeError("Can only compare Money with Money")
        
        if self.currency != other.currency:
            raise ValueError("Cannot compare money with different currencies")
        
        return self.amount < other.amount
    
    def __le__(self, other: 'Money') -> bool:
        """Less than or equal comparison"""
        return self < other or self == other
    
    def __gt__(self, other: 'Money') -> bool:
        """Greater than comparison"""
        return not self <= other
    
    def __ge__(self, other: 'Money') -> bool:
        """Greater than or equal comparison"""
        return not self < other
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, Money):
            return False
        
        return self.amount == other.amount and self.currency == other.currency
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries"""
        return hash((self.amount, self.currency))
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.currency} {self.amount:,.2f}"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"Money({self.amount}, '{self.currency}')"
    
    @classmethod
    def zero(cls, currency: str = "USD") -> 'Money':
        """Create zero money amount"""
        return cls(Decimal('0'), currency)
    
    @classmethod
    def from_string(cls, amount_str: str, currency: str = "USD") -> 'Money':
        """Create money from string representation"""
        try:
            amount = Decimal(amount_str)
            return cls(amount, currency)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid amount string: {amount_str}") from e
    
    def to_cents(self) -> int:
        """Convert to cents (integer)"""
        return int(self.amount * 100)
    
    @classmethod
    def from_cents(cls, cents: int, currency: str = "USD") -> 'Money':
        """Create money from cents"""
        return cls(Decimal(cents) / 100, currency)
    
    def is_positive(self) -> bool:
        """Check if amount is positive"""
        return self.amount > 0
    
    def is_zero(self) -> bool:
        """Check if amount is zero"""
        return self.amount == 0
    
    def abs(self) -> 'Money':
        """Return absolute value"""
        return Money(abs(self.amount), self.currency)
    
    def round_to_nearest(self, increment: Union[Decimal, int, float]) -> 'Money':
        """Round to nearest increment"""
        increment_decimal = Decimal(str(increment))
        rounded = (self.amount / increment_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        return Money(rounded * increment_decimal, self.currency)





