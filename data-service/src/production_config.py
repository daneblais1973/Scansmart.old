"""
Production Configuration for Data Service
========================================
Configuration settings for production-ready data fetching
"""

import os
from typing import List, Dict, Any

# Stock Data Sources Configuration
STOCK_DATA_CONFIG = {
    "primary_source": os.getenv("STOCK_DATA_SOURCE", "yfinance"),
    "backup_sources": ["alpha_vantage", "quandl"],
    "cache_duration_minutes": int(os.getenv("STOCK_CACHE_DURATION", "5")),
    "rate_limit_per_minute": int(os.getenv("STOCK_RATE_LIMIT", "60")),
    "timeout_seconds": int(os.getenv("STOCK_TIMEOUT", "30"))
}

# News Sources Configuration
NEWS_SOURCES_CONFIG = {
    "rss_feeds": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://feeds.cnn.com/rss/money_latest.rss",
        "https://feeds.nasdaq.com/nasdaq/marketnews"
    ],
    "api_sources": [
        "newsapi",
        "alpha_vantage_news",
        "polygon_news"
    ],
    "cache_duration_minutes": int(os.getenv("NEWS_CACHE_DURATION", "15")),
    "max_items_per_source": int(os.getenv("NEWS_MAX_ITEMS", "5")),
    "timeout_seconds": int(os.getenv("NEWS_TIMEOUT", "30"))
}

# Opportunities Analysis Configuration
OPPORTUNITIES_CONFIG = {
    "ai_analysis_service": os.getenv("AI_ANALYSIS_SERVICE_URL", "http://localhost:8001"),
    "catalyst_service": os.getenv("CATALYST_SERVICE_URL", "http://localhost:8003"),
    "screening_service": os.getenv("SCREENING_SERVICE_URL", "http://localhost:8004"),
    "cache_duration_minutes": int(os.getenv("OPPORTUNITIES_CACHE_DURATION", "10")),
    "min_confidence_threshold": float(os.getenv("MIN_CONFIDENCE", "0.7")),
    "max_opportunities": int(os.getenv("MAX_OPPORTUNITIES", "100"))
}

# Cache Configuration
CACHE_CONFIG = {
    "type": os.getenv("CACHE_TYPE", "memory"),  # memory, redis, memcached
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "default_expiry_minutes": int(os.getenv("CACHE_DEFAULT_EXPIRY", "10")),
    "max_cache_size": int(os.getenv("CACHE_MAX_SIZE", "1000"))
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    "enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
    "requests_per_minute": int(os.getenv("RATE_LIMIT_RPM", "100")),
    "burst_limit": int(os.getenv("RATE_LIMIT_BURST", "20"))
}

# Error Handling Configuration
ERROR_CONFIG = {
    "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "3")),
    "retry_delay_seconds": int(os.getenv("RETRY_DELAY", "1")),
    "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
    "circuit_breaker_timeout": int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": os.getenv("LOG_FILE", "logs/data_service.log"),
    "max_file_size": int(os.getenv("LOG_MAX_SIZE", "10485760")),  # 10MB
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5"))
}

# Health Check Configuration
HEALTH_CONFIG = {
    "check_interval_seconds": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
    "timeout_seconds": int(os.getenv("HEALTH_TIMEOUT", "5")),
    "dependencies": [
        "stock_data_service",
        "news_service", 
        "opportunities_service",
        "cache_service"
    ]
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "stock_data": STOCK_DATA_CONFIG,
        "news_sources": NEWS_SOURCES_CONFIG,
        "opportunities": OPPORTUNITIES_CONFIG,
        "cache": CACHE_CONFIG,
        "rate_limit": RATE_LIMIT_CONFIG,
        "error_handling": ERROR_CONFIG,
        "logging": LOGGING_CONFIG,
        "health": HEALTH_CONFIG
    }

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Validate numeric values
        assert STOCK_DATA_CONFIG["cache_duration_minutes"] > 0
        assert NEWS_SOURCES_CONFIG["cache_duration_minutes"] > 0
        assert OPPORTUNITIES_CONFIG["cache_duration_minutes"] > 0
        
        # Validate URLs
        for feed_url in NEWS_SOURCES_CONFIG["rss_feeds"]:
            assert feed_url.startswith("http")
        
        return True
    except (AssertionError, ValueError) as e:
        print(f"Configuration validation failed: {e}")
        return False


