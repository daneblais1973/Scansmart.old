"""
Stock Symbol Value Object
Advanced stock symbol representation with exchange and market data
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import re


class Exchange(Enum):
    """Major stock exchanges"""
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    AMEX = "amex"
    LSE = "lse"
    TSE = "tse"
    HKEX = "hkex"
    SSE = "sse"
    SZSE = "szse"
    BSE = "bse"
    NSE = "nse"
    TSX = "tsx"
    ASX = "asx"
    EURONEXT = "euronext"
    DEUTSCHE_BOERSE = "deutsche_boerse"
    SWISS_EXCHANGE = "swiss_exchange"
    OTHER = "other"


class SecurityType(Enum):
    """Security types"""
    COMMON_STOCK = "common_stock"
    PREFERRED_STOCK = "preferred_stock"
    ETF = "etf"
    REIT = "reit"
    ADR = "adr"
    GDR = "gdr"
    WARRANT = "warrant"
    OPTION = "option"
    FUTURE = "future"
    BOND = "bond"
    CRYPTOCURRENCY = "cryptocurrency"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    OTHER = "other"


@dataclass(frozen=True)
class StockSymbol:
    """
    Immutable value object representing stock symbols
    Enhanced with exchange, security type, and market data
    """
    
    symbol: str
    exchange: Optional[Exchange] = None
    security_type: Optional[SecurityType] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: bool = True
    
    def __post_init__(self):
        """Validate stock symbol invariants"""
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Stock symbol cannot be empty")
        
        # Normalize symbol
        normalized_symbol = self.symbol.strip().upper()
        object.__setattr__(self, 'symbol', normalized_symbol)
        
        # Validate symbol format
        if not re.match(r'^[A-Z0-9.-]+$', normalized_symbol):
            raise ValueError("Invalid stock symbol format")
        
        # Validate exchange
        if self.exchange and not isinstance(self.exchange, Exchange):
            if isinstance(self.exchange, str):
                try:
                    object.__setattr__(self, 'exchange', Exchange(self.exchange.lower()))
                except ValueError:
                    object.__setattr__(self, 'exchange', Exchange.OTHER)
            else:
                raise ValueError("Invalid exchange type")
        
        # Validate security type
        if self.security_type and not isinstance(self.security_type, SecurityType):
            if isinstance(self.security_type, str):
                try:
                    object.__setattr__(self, 'security_type', SecurityType(self.security_type.lower()))
                except ValueError:
                    object.__setattr__(self, 'security_type', SecurityType.OTHER)
            else:
                raise ValueError("Invalid security type")
    
    def __str__(self) -> str:
        """String representation"""
        if self.exchange:
            return f"{self.symbol}.{self.exchange.value.upper()}"
        return self.symbol
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"StockSymbol('{self.symbol}', '{self.exchange}', '{self.security_type}')"
    
    @classmethod
    def from_string(cls, symbol_string: str, **kwargs) -> 'StockSymbol':
        """Create from string like 'AAPL.NASDAQ' or 'AAPL'"""
        if '.' in symbol_string:
            symbol, exchange_str = symbol_string.split('.', 1)
            try:
                exchange = Exchange(exchange_str.lower())
            except ValueError:
                exchange = Exchange.OTHER
            return cls(symbol, exchange, **kwargs)
        return cls(symbol_string, **kwargs)
    
    def with_exchange(self, exchange: Exchange) -> 'StockSymbol':
        """Create new symbol with exchange"""
        return StockSymbol(
            symbol=self.symbol,
            exchange=exchange,
            security_type=self.security_type,
            company_name=self.company_name,
            sector=self.sector,
            industry=self.industry,
            country=self.country,
            currency=self.currency,
            market_cap=self.market_cap,
            is_active=self.is_active
        )
    
    def with_security_type(self, security_type: SecurityType) -> 'StockSymbol':
        """Create new symbol with security type"""
        return StockSymbol(
            symbol=self.symbol,
            exchange=self.exchange,
            security_type=security_type,
            company_name=self.company_name,
            sector=self.sector,
            industry=self.industry,
            country=self.country,
            currency=self.currency,
            market_cap=self.market_cap,
            is_active=self.is_active
        )
    
    def with_company_info(self, company_name: str, sector: str = None, 
                         industry: str = None, country: str = None) -> 'StockSymbol':
        """Create new symbol with company information"""
        return StockSymbol(
            symbol=self.symbol,
            exchange=self.exchange,
            security_type=self.security_type,
            company_name=company_name,
            sector=sector,
            industry=industry,
            country=country,
            currency=self.currency,
            market_cap=self.market_cap,
            is_active=self.is_active
        )
    
    def with_market_data(self, market_cap: float = None, currency: str = None) -> 'StockSymbol':
        """Create new symbol with market data"""
        return StockSymbol(
            symbol=self.symbol,
            exchange=self.exchange,
            security_type=self.security_type,
            company_name=self.company_name,
            sector=self.sector,
            industry=self.industry,
            country=self.country,
            currency=currency,
            market_cap=market_cap,
            is_active=self.is_active
        )
    
    def is_equity(self) -> bool:
        """Check if symbol represents equity"""
        return self.security_type in [SecurityType.COMMON_STOCK, SecurityType.PREFERRED_STOCK]
    
    def is_etf(self) -> bool:
        """Check if symbol represents ETF"""
        return self.security_type == SecurityType.ETF
    
    def is_reit(self) -> bool:
        """Check if symbol represents REIT"""
        return self.security_type == SecurityType.REIT
    
    def is_crypto(self) -> bool:
        """Check if symbol represents cryptocurrency"""
        return self.security_type == SecurityType.CRYPTOCURRENCY
    
    def is_major_exchange(self) -> bool:
        """Check if symbol is on major exchange"""
        return self.exchange in [Exchange.NYSE, Exchange.NASDAQ, Exchange.LSE, Exchange.TSE]
    
    def get_full_name(self) -> str:
        """Get full name with exchange"""
        if self.exchange:
            return f"{self.symbol} ({self.exchange.value.upper()})"
        return self.symbol
    
    def get_display_name(self) -> str:
        """Get display name with company name if available"""
        if self.company_name:
            return f"{self.symbol} - {self.company_name}"
        return self.symbol
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange.value if self.exchange else None,
            'security_type': self.security_type.value if self.security_type else None,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'country': self.country,
            'currency': self.currency,
            'market_cap': self.market_cap,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockSymbol':
        """Create from dictionary"""
        return cls(
            symbol=data['symbol'],
            exchange=Exchange(data['exchange']) if data.get('exchange') else None,
            security_type=SecurityType(data['security_type']) if data.get('security_type') else None,
            company_name=data.get('company_name'),
            sector=data.get('sector'),
            industry=data.get('industry'),
            country=data.get('country'),
            currency=data.get('currency'),
            market_cap=data.get('market_cap'),
            is_active=data.get('is_active', True)
        )




