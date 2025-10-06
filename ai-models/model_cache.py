"""
Model Cache
===========
Enterprise-grade AI model caching service for optimized model management
"""

import asyncio
import logging
import json
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# AI/ML imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    import numpy as np
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"  # Size-based eviction

class CacheStatus(Enum):
    """Cache status levels"""
    HIT = "hit"
    MISS = "miss"
    EVICTED = "evicted"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class CacheEntry:
    """Cache entry container"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheMetrics:
    """Cache metrics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    miss_rate: float
    evictions: int
    total_size_bytes: int
    average_access_time: float
    memory_usage: float

class ModelCache:
    """Enterprise-grade AI model caching service"""
    
    def __init__(self, max_size_mb: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.access_order: List[str] = []
        self.access_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.metrics = CacheMetrics(
            total_requests=0, cache_hits=0, cache_misses=0,
            hit_rate=0.0, miss_rate=0.0, evictions=0,
            total_size_bytes=0, average_access_time=0.0, memory_usage=0.0
        )
        
        # Cache configuration
        self.config = {
            'enable_compression': True,
            'enable_serialization': True,
            'default_ttl': 3600,  # 1 hour
            'cleanup_interval': 300,  # 5 minutes
            'max_entries': 1000,
            'enable_metrics': True
        }
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info(f"Model Cache initialized: {max_size_mb}MB, strategy: {strategy.value}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            start_time = datetime.now()
            self.metrics.total_requests += 1
            
            if key not in self.cache:
                self.metrics.cache_misses += 1
                self._update_metrics()
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                await self._remove_entry(key)
                self.metrics.cache_misses += 1
                self._update_metrics()
                return None
            
            # Update access information
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # Update access order for LRU
            if self.strategy == CacheStrategy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            self.metrics.cache_hits += 1
            self._update_metrics()
            
            # Calculate access time
            access_time = (datetime.now() - start_time).total_seconds()
            self.metrics.average_access_time = (
                (self.metrics.average_access_time * (self.metrics.total_requests - 1) + access_time) /
                self.metrics.total_requests
            )
            
            logger.debug(f"Cache hit: {key}")
            return entry.value
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.metrics.cache_misses += 1
            self._update_metrics()
            return None
    
    async def put(self, key: str, value: Any, ttl: Optional[timedelta] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache"""
        try:
            # Calculate size
            size_bytes = await self._calculate_size(value)
            
            # Check if we need to evict
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Evict if necessary
            while self._get_total_size() + size_bytes > self.max_size_bytes:
                await self._evict_entry()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or timedelta(seconds=self.config['default_ttl']),
                metadata=metadata or {}
            )
            
            # Store in cache
            self.cache[key] = entry
            self.access_counts[key] = 1
            
            # Update access order
            if self.strategy == CacheStrategy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            self._update_metrics()
            
            logger.debug(f"Cached: {key} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error putting in cache: {e}")
            return False
    
    async def remove(self, key: str) -> bool:
        """Remove value from cache"""
        try:
            if key not in self.cache:
                return False
            
            await self._remove_entry(key)
            self._update_metrics()
            
            logger.debug(f"Removed from cache: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing from cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            
            self._update_metrics()
            
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cache entry information"""
        try:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            return {
                'key': key,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'ttl': entry.ttl.total_seconds() if entry.ttl else None,
                'metadata': entry.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                'total_entries': len(self.cache),
                'total_size_bytes': self._get_total_size(),
                'max_size_bytes': self.max_size_bytes,
                'utilization_percent': (self._get_total_size() / self.max_size_bytes) * 100,
                'strategy': self.strategy.value,
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses,
                    'hit_rate': self.metrics.hit_rate,
                    'miss_rate': self.metrics.miss_rate,
                    'evictions': self.metrics.evictions,
                    'average_access_time': self.metrics.average_access_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def warm_cache(self, model_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Warm cache with models"""
        try:
            results = {}
            
            for config in model_configs:
                key = config.get('key')
                value = config.get('value')
                ttl = config.get('ttl')
                
                if ttl:
                    ttl = timedelta(seconds=ttl)
                
                success = await self.put(key, value, ttl)
                results[key] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return {}
    
    async def _remove_entry(self, key: str):
        """Remove entry from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            
            if key in self.access_counts:
                del self.access_counts[key]
                
        except Exception as e:
            logger.error(f"Error removing entry: {e}")
    
    async def _evict_entry(self):
        """Evict entry based on strategy"""
        try:
            if not self.cache:
                return
            
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                if self.access_order:
                    key_to_remove = self.access_order[0]
                    await self._remove_entry(key_to_remove)
            
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                if self.access_counts:
                    key_to_remove = min(self.access_counts, key=self.access_counts.get)
                    await self._remove_entry(key_to_remove)
            
            elif self.strategy == CacheStrategy.FIFO:
                # Remove first in
                if self.cache:
                    key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                    await self._remove_entry(key_to_remove)
            
            elif self.strategy == CacheStrategy.SIZE_BASED:
                # Remove largest entry
                if self.cache:
                    key_to_remove = max(self.cache.keys(), key=lambda k: self.cache[k].size_bytes)
                    await self._remove_entry(key_to_remove)
            
            self.metrics.evictions += 1
            
        except Exception as e:
            logger.error(f"Error evicting entry: {e}")
    
    async def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            if self.config['enable_serialization']:
                # Serialize to get accurate size
                serialized = pickle.dumps(value)
                return len(serialized)
            else:
                # Estimate size
                if hasattr(value, '__sizeof__'):
                    return value.__sizeof__()
                else:
                    return 1024  # Default estimate
                    
        except Exception as e:
            logger.error(f"Error calculating size: {e}")
            return 1024  # Default estimate
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def _update_metrics(self):
        """Update cache metrics"""
        try:
            if self.metrics.total_requests > 0:
                self.metrics.hit_rate = self.metrics.cache_hits / self.metrics.total_requests
                self.metrics.miss_rate = self.metrics.cache_misses / self.metrics.total_requests
            
            self.metrics.total_size_bytes = self._get_total_size()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        try:
            while True:
                await asyncio.sleep(self.config['cleanup_interval'])
                
                # Remove expired entries
                expired_keys = []
                for key, entry in self.cache.items():
                    if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    await self._remove_entry(key)
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired entries")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

# Global instance
model_cache = ModelCache()




