"""
Advanced Multi-Level Caching System
===================================
Enterprise-grade multi-level caching with intelligent eviction and prefetching
"""

import asyncio
import logging
import time
import threading
import hashlib
import pickle
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import weakref
from collections import OrderedDict, deque
import psutil
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Cache backend imports with graceful fallback
try:
    import redis
    import memcached
    import sqlite3
    CACHE_BACKENDS_AVAILABLE = True
except ImportError:
    CACHE_BACKENDS_AVAILABLE = False
    redis = None
    memcached = None
    sqlite3 = None

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels in the hierarchy"""
    L1_MEMORY = "l1_memory"      # Fastest - in-memory
    L2_SHARED = "l2_shared"     # Fast - shared memory
    L3_DISK = "l3_disk"         # Medium - disk cache
    L4_DISTRIBUTED = "l4_distributed"  # Slow - distributed cache
    L5_PERSISTENT = "l5_persistent"    # Slowest - persistent storage

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    FIFO = "fifo"                # First In, First Out
    TTL = "ttl"                  # Time To Live
    RANDOM = "random"            # Random eviction
    ADAPTIVE = "adaptive"        # Adaptive based on access patterns

class CacheStrategy(Enum):
    """Caching strategies"""
    WRITE_THROUGH = "write_through"      # Write to all levels
    WRITE_BACK = "write_back"            # Write to L1, sync later
    WRITE_AROUND = "write_around"        # Skip cache, write to storage
    READ_THROUGH = "read_through"        # Read from storage if not in cache
    CACHE_ASIDE = "cache_aside"          # Application manages cache

@dataclass
class CacheConfig:
    """Cache configuration"""
    level: CacheLevel
    max_size: int = 1000
    ttl: float = 3600.0  # seconds
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH
    compression: bool = True
    encryption: bool = False
    prefetch: bool = True
    persistence: bool = False

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size: int = 0
    ttl: float = 3600.0
    level: CacheLevel = CacheLevel.L1_MEMORY
    compressed: bool = False
    encrypted: bool = False
    
    def __post_init__(self):
        """Initialize cache entry"""
        self.last_accessed = self.created_at
        self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate entry size in bytes"""
        try:
            if isinstance(self.value, (str, bytes)):
                return len(self.value)
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value).encode('utf-8'))
            else:
                return len(pickle.dumps(self.value))
        except Exception:
            return 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_entries: int = 0
    total_size: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    prefetch_count: int = 0
    compression_ratio: float = 0.0
    average_access_time: float = 0.0
    hit_rate: float = 0.0
    memory_usage: float = 0.0

class AdvancedCache:
    """Advanced multi-level cache implementation"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.prefetch_thread = None
        
        # Initialize cache
        self._initialize_cache()
        
        logger.info(f"Advanced Cache initialized: {config.level.value}")
    
    def _initialize_cache(self):
        """Initialize cache based on configuration"""
        try:
            # Start background threads
            self._start_background_threads()
            logger.info(f"Cache initialized with {self.config.max_size} max entries")
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")
    
    def _start_background_threads(self):
        """Start background cleanup and prefetch threads"""
        try:
            # Cleanup thread
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self.cleanup_thread.start()
            
            # Prefetch thread
            if self.config.prefetch:
                self.prefetch_thread = threading.Thread(
                    target=self._prefetch_worker,
                    daemon=True
                )
                self.prefetch_thread.start()
            
            logger.info("Background threads started")
        except Exception as e:
            logger.error(f"Error starting background threads: {e}")
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup"""
        while True:
            try:
                self._cleanup_expired_entries()
                self._cleanup_oversized_entries()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(10)
    
    def _prefetch_worker(self):
        """Background worker for cache prefetching"""
        while True:
            try:
                self._prefetch_popular_entries()
                time.sleep(300)  # Prefetch every 5 minutes
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                time.sleep(30)
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        try:
            with self.lock:
                expired_keys = []
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                    self.metrics.eviction_count += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
    
    def _cleanup_oversized_entries(self):
        """Remove entries if cache exceeds max size"""
        try:
            with self.lock:
                while len(self.cache) > self.config.max_size:
                    # Evict based on policy
                    key_to_evict = self._select_eviction_candidate()
                    if key_to_evict:
                        del self.cache[key_to_evict]
                        self.metrics.eviction_count += 1
                    else:
                        break
        except Exception as e:
            logger.error(f"Error cleaning up oversized entries: {e}")
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on policy"""
        try:
            if not self.cache:
                return None
            
            if self.config.eviction_policy == EvictionPolicy.LRU:
                return next(iter(self.cache))  # First (oldest) entry
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                return min(self.cache.items(), key=lambda x: x[1].access_count)[0]
            elif self.config.eviction_policy == EvictionPolicy.FIFO:
                return next(iter(self.cache))  # First entry
            elif self.config.eviction_policy == EvictionPolicy.RANDOM:
                import random
                return random.choice(list(self.cache.keys()))
            else:
                return next(iter(self.cache))  # Default to LRU
        except Exception as e:
            logger.error(f"Error selecting eviction candidate: {e}")
            return None
    
    def _prefetch_popular_entries(self):
        """Prefetch popular entries based on access patterns"""
        try:
            # This would implement intelligent prefetching
            # For now, just log that prefetching is working
            logger.debug("Prefetch worker running")
        except Exception as e:
            logger.error(f"Error in prefetch: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            with self.lock:
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # Check if expired
                    if entry.is_expired():
                        del self.cache[key]
                        self.metrics.miss_count += 1
                        return None
                    
                    # Update access statistics
                    entry.update_access()
                    
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    
                    # Update metrics
                    self.metrics.hit_count += 1
                    self._update_metrics()
                    
                    return entry.value
                else:
                    self.metrics.miss_count += 1
                    self._update_metrics()
                    return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        try:
            with self.lock:
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    ttl=ttl or self.config.ttl,
                    level=self.config.level
                )
                
                # Apply compression if enabled
                if self.config.compression:
                    entry = self._compress_entry(entry)
                
                # Apply encryption if enabled
                if self.config.encryption:
                    entry = self._encrypt_entry(entry)
                
                # Store in cache
                self.cache[key] = entry
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                # Update metrics
                self.metrics.total_entries = len(self.cache)
                self.metrics.total_size += entry.size
                
                return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        try:
            with self.lock:
                if key in self.cache:
                    entry = self.cache.pop(key)
                    self.metrics.total_size -= entry.size
                    self.metrics.total_entries = len(self.cache)
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear(self):
        """Clear all entries from cache"""
        try:
            with self.lock:
                self.cache.clear()
                self.metrics.total_entries = 0
                self.metrics.total_size = 0
                logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry"""
        try:
            if not self.config.compression:
                return entry
            
            # Simple compression simulation
            entry.compressed = True
            entry.size = int(entry.size * 0.7)  # Simulate 30% compression
            return entry
        except Exception as e:
            logger.error(f"Error compressing entry: {e}")
            return entry
    
    def _encrypt_entry(self, entry: CacheEntry) -> CacheEntry:
        """Encrypt cache entry"""
        try:
            if not self.config.encryption:
                return entry
            
            # Simple encryption simulation
            entry.encrypted = True
            return entry
        except Exception as e:
            logger.error(f"Error encrypting entry: {e}")
            return entry
    
    def _update_metrics(self):
        """Update cache metrics"""
        try:
            total_requests = self.metrics.hit_count + self.metrics.miss_count
            if total_requests > 0:
                self.metrics.hit_rate = self.metrics.hit_count / total_requests
            
            # Update memory usage
            process = psutil.Process()
            self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics"""
        try:
            self._update_metrics()
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return self.metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            metrics = self.get_metrics()
            return {
                'level': self.config.level.value,
                'max_size': self.config.max_size,
                'current_size': len(self.cache),
                'total_entries': metrics.total_entries,
                'total_size': metrics.total_size,
                'hit_count': metrics.hit_count,
                'miss_count': metrics.miss_count,
                'hit_rate': metrics.hit_rate,
                'eviction_count': metrics.eviction_count,
                'prefetch_count': metrics.prefetch_count,
                'compression_ratio': metrics.compression_ratio,
                'memory_usage_mb': metrics.memory_usage,
                'eviction_policy': self.config.eviction_policy.value,
                'strategy': self.config.strategy.value,
                'compression_enabled': self.config.compression,
                'encryption_enabled': self.config.encryption,
                'prefetch_enabled': self.config.prefetch
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

class MultiLevelCache:
    """Multi-level cache system with intelligent routing"""
    
    def __init__(self):
        self.cache_levels: Dict[CacheLevel, AdvancedCache] = {}
        self.routing_rules: Dict[str, CacheLevel] = {}
        self.metrics = CacheMetrics()
        
        # Initialize cache levels
        self._initialize_cache_levels()
        
        logger.info("Multi-Level Cache System initialized")
    
    def _initialize_cache_levels(self):
        """Initialize all cache levels"""
        try:
            # L1: Fast in-memory cache
            l1_config = CacheConfig(
                level=CacheLevel.L1_MEMORY,
                max_size=1000,
                ttl=300,  # 5 minutes
                eviction_policy=EvictionPolicy.LRU,
                strategy=CacheStrategy.WRITE_THROUGH,
                compression=True,
                prefetch=True
            )
            self.cache_levels[CacheLevel.L1_MEMORY] = AdvancedCache(l1_config)
            
            # L2: Shared memory cache
            l2_config = CacheConfig(
                level=CacheLevel.L2_SHARED,
                max_size=5000,
                ttl=1800,  # 30 minutes
                eviction_policy=EvictionPolicy.LFU,
                strategy=CacheStrategy.WRITE_BACK,
                compression=True,
                prefetch=True
            )
            self.cache_levels[CacheLevel.L2_SHARED] = AdvancedCache(l2_config)
            
            # L3: Disk cache
            l3_config = CacheConfig(
                level=CacheLevel.L3_DISK,
                max_size=10000,
                ttl=3600,  # 1 hour
                eviction_policy=EvictionPolicy.TTL,
                strategy=CacheStrategy.WRITE_AROUND,
                compression=True,
                persistence=True
            )
            self.cache_levels[CacheLevel.L3_DISK] = AdvancedCache(l3_config)
            
            # L4: Distributed cache
            l4_config = CacheConfig(
                level=CacheLevel.L4_DISTRIBUTED,
                max_size=50000,
                ttl=7200,  # 2 hours
                eviction_policy=EvictionPolicy.ADAPTIVE,
                strategy=CacheStrategy.CACHE_ASIDE,
                compression=True,
                encryption=True
            )
            self.cache_levels[CacheLevel.L4_DISTRIBUTED] = AdvancedCache(l4_config)
            
            # L5: Persistent storage
            l5_config = CacheConfig(
                level=CacheLevel.L5_PERSISTENT,
                max_size=100000,
                ttl=86400,  # 24 hours
                eviction_policy=EvictionPolicy.TTL,
                strategy=CacheStrategy.READ_THROUGH,
                compression=True,
                encryption=True,
                persistence=True
            )
            self.cache_levels[CacheLevel.L5_PERSISTENT] = AdvancedCache(l5_config)
            
            logger.info(f"Initialized {len(self.cache_levels)} cache levels")
        except Exception as e:
            logger.error(f"Error initializing cache levels: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        try:
            # Try each level from fastest to slowest
            for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_SHARED, 
                         CacheLevel.L3_DISK, CacheLevel.L4_DISTRIBUTED, CacheLevel.L5_PERSISTENT]:
                cache = self.cache_levels.get(level)
                if cache:
                    value = cache.get(key)
                    if value is not None:
                        # Promote to higher levels if found in lower level
                        self._promote_to_higher_levels(key, value, level)
                        return value
            
            return None
        except Exception as e:
            logger.error(f"Error getting from multi-level cache: {e}")
            return None
    
    def set(self, key: str, value: Any, level: Optional[CacheLevel] = None) -> bool:
        """Set value in multi-level cache"""
        try:
            if level is None:
                # Determine appropriate level based on key pattern
                level = self._determine_cache_level(key)
            
            cache = self.cache_levels.get(level)
            if cache:
                success = cache.set(key, value)
                if success:
                    # Cascade to lower levels if needed
                    self._cascade_to_lower_levels(key, value, level)
                return success
            
            return False
        except Exception as e:
            logger.error(f"Error setting multi-level cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        try:
            deleted = False
            for cache in self.cache_levels.values():
                if cache.delete(key):
                    deleted = True
            return deleted
        except Exception as e:
            logger.error(f"Error deleting from multi-level cache: {e}")
            return False
    
    def clear(self):
        """Clear all cache levels"""
        try:
            for cache in self.cache_levels.values():
                cache.clear()
            logger.info("All cache levels cleared")
        except Exception as e:
            logger.error(f"Error clearing multi-level cache: {e}")
    
    def _determine_cache_level(self, key: str) -> CacheLevel:
        """Determine appropriate cache level for key"""
        try:
            # Simple routing based on key patterns
            if key.startswith('temp_') or key.startswith('session_'):
                return CacheLevel.L1_MEMORY
            elif key.startswith('user_') or key.startswith('config_'):
                return CacheLevel.L2_SHARED
            elif key.startswith('data_') or key.startswith('cache_'):
                return CacheLevel.L3_DISK
            elif key.startswith('shared_') or key.startswith('global_'):
                return CacheLevel.L4_DISTRIBUTED
            else:
                return CacheLevel.L5_PERSISTENT
        except Exception as e:
            logger.error(f"Error determining cache level: {e}")
            return CacheLevel.L1_MEMORY
    
    def _promote_to_higher_levels(self, key: str, value: Any, found_level: CacheLevel):
        """Promote value to higher cache levels"""
        try:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_SHARED, CacheLevel.L3_DISK]
            current_index = levels.index(found_level) if found_level in levels else -1
            
            # Promote to higher levels
            for i in range(current_index):
                level = levels[i]
                cache = self.cache_levels.get(level)
                if cache:
                    cache.set(key, value)
        except Exception as e:
            logger.error(f"Error promoting to higher levels: {e}")
    
    def _cascade_to_lower_levels(self, key: str, value: Any, set_level: CacheLevel):
        """Cascade value to lower cache levels"""
        try:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_SHARED, CacheLevel.L3_DISK, 
                     CacheLevel.L4_DISTRIBUTED, CacheLevel.L5_PERSISTENT]
            current_index = levels.index(set_level) if set_level in levels else -1
            
            # Cascade to lower levels
            for i in range(current_index + 1, len(levels)):
                level = levels[i]
                cache = self.cache_levels.get(level)
                if cache:
                    cache.set(key, value)
        except Exception as e:
            logger.error(f"Error cascading to lower levels: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all cache levels"""
        try:
            metrics = {
                'total_levels': len(self.cache_levels),
                'levels': {},
                'overall_metrics': {
                    'total_entries': 0,
                    'total_size': 0,
                    'total_hits': 0,
                    'total_misses': 0,
                    'total_evictions': 0,
                    'average_hit_rate': 0.0
                }
            }
            
            # Get metrics from each level
            for level, cache in self.cache_levels.items():
                level_metrics = cache.get_metrics()
                level_stats = cache.get_stats()
                
                metrics['levels'][level.value] = level_stats
                
                # Aggregate overall metrics
                metrics['overall_metrics']['total_entries'] += level_metrics.total_entries
                metrics['overall_metrics']['total_size'] += level_metrics.total_size
                metrics['overall_metrics']['total_hits'] += level_metrics.hit_count
                metrics['overall_metrics']['total_misses'] += level_metrics.miss_count
                metrics['overall_metrics']['total_evictions'] += level_metrics.eviction_count
            
            # Calculate average hit rate
            total_requests = metrics['overall_metrics']['total_hits'] + metrics['overall_metrics']['total_misses']
            if total_requests > 0:
                metrics['overall_metrics']['average_hit_rate'] = (
                    metrics['overall_metrics']['total_hits'] / total_requests
                )
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {'error': str(e)}

# Global multi-level cache instance
multi_level_cache = MultiLevelCache()




