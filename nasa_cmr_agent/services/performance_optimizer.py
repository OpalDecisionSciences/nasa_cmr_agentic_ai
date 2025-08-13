"""
Performance Optimization Service

Provides caching, query optimization, and performance monitoring capabilities.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from functools import wraps

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from ..core.config import settings

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    hit_count: int = 0
    ttl: int  # seconds
    size_bytes: int = 0


class QueryOptimizer:
    """
    Optimizes queries for better performance.
    """
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.query_patterns: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
    
    def optimize_cmr_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize CMR API parameters for better performance.
        
        Args:
            params: Original parameters
            
        Returns:
            Optimized parameters
        """
        optimized = params.copy()
        
        # Optimize page size for better performance
        if 'page_size' not in optimized:
            optimized['page_size'] = 100  # Optimal for most queries
        
        # Add fields to reduce payload size if not specified
        if 'fields' not in optimized and 'include_facets' not in optimized:
            # Only request necessary fields
            optimized['fields'] = [
                'concept_id', 'title', 'summary', 'data_center',
                'platforms', 'instruments', 'temporal', 'spatial'
            ]
        
        # Optimize temporal queries
        if 'temporal' in optimized:
            temporal = optimized['temporal']
            # Round to day boundaries for better caching
            if isinstance(temporal, str) and ',' in temporal:
                start, end = temporal.split(',')
                # Keep as-is for now, but could optimize date ranges
        
        # Add request headers for compression
        if 'headers' not in optimized:
            optimized['headers'] = {}
        optimized['headers']['Accept-Encoding'] = 'gzip, deflate'
        
        return optimized
    
    def should_parallelize(self, query_context: Any) -> bool:
        """
        Determine if query should be parallelized.
        
        Args:
            query_context: Query context
            
        Returns:
            Whether to parallelize
        """
        # Check for multiple data types or complex spatial/temporal constraints
        if hasattr(query_context, 'data_types') and len(query_context.data_types) > 1:
            return True
        
        if hasattr(query_context, 'spatial_constraints'):
            spatial = query_context.spatial_constraints
            if spatial and (spatial.north - spatial.south) * (spatial.east - spatial.west) > 1000:
                # Large spatial area, benefit from parallelization
                return True
        
        return False
    
    def get_batch_strategy(self, items: List[Any], max_concurrent: int = 5) -> List[List[Any]]:
        """
        Create optimal batches for parallel processing.
        
        Args:
            items: Items to process
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Batched items
        """
        if len(items) <= max_concurrent:
            return [[item] for item in items]
        
        # Create evenly distributed batches
        batch_size = len(items) // max_concurrent
        if len(items) % max_concurrent > 0:
            batch_size += 1
        
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        return batches


class PerformanceCache:
    """
    High-performance caching system with multiple layers.
    """
    
    def __init__(self):
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self.max_memory_entries = 1000
        self.default_ttl = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize cache connections."""
        try:
            self.redis_client = await aioredis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                decode_responses=False  # Handle binary data
            )
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache only: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        # Filter out non-serializable objects and convert to serializable format
        serializable_params = {}
        for k, v in params.items():
            try:
                if hasattr(v, 'dict'):
                    # Pydantic models
                    serializable_params[k] = v.dict()
                elif hasattr(v, '__dict__'):
                    # Other objects with __dict__
                    serializable_params[k] = str(v)
                else:
                    # Primitive types
                    json.dumps(v)  # Test if serializable
                    serializable_params[k] = v
            except (TypeError, AttributeError):
                # Non-serializable, use string representation
                serializable_params[k] = str(v)
        
        # Sort params for consistent hashing
        sorted_params = json.dumps(serializable_params, sort_keys=True)
        hash_digest = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if datetime.now() < entry.timestamp + timedelta(seconds=entry.ttl):
                entry.hit_count += 1
                logger.debug(f"Memory cache hit: {key}")
                return entry.value
            else:
                # Expired
                del self.memory_cache[key]
        
        # Check Redis if available
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    logger.debug(f"Redis cache hit: {key}")
                    # Deserialize and add to memory cache
                    deserialized = json.loads(value)
                    self._add_to_memory_cache(key, deserialized)
                    return deserialized
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        ttl = ttl or self.default_ttl
        
        # Add to memory cache
        self._add_to_memory_cache(key, value, ttl)
        
        # Add to Redis if available
        if self.redis_client:
            try:
                serialized = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized)
                logger.debug(f"Cached to Redis: {key}")
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        return True
    
    def _add_to_memory_cache(self, key: str, value: Any, ttl: int = None):
        """Add entry to memory cache with LRU eviction."""
        ttl = ttl or self.default_ttl
        
        # Evict oldest entries if at capacity
        if len(self.memory_cache) >= self.max_memory_entries:
            # Find least recently used
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].hit_count
            )
            del self.memory_cache[lru_key]
        
        # Calculate size
        size_bytes = len(json.dumps(value, default=str).encode())
        
        self.memory_cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl,
            size_bytes=size_bytes
        )
    
    async def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., "cmr:*")
        """
        # Clear from memory cache
        keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.memory_cache[key]
        
        # Clear from Redis
        if self.redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis pattern delete error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(e.size_bytes for e in self.memory_cache.values())
        total_hits = sum(e.hit_count for e in self.memory_cache.values())
        
        return {
            "memory_entries": len(self.memory_cache),
            "total_size_bytes": total_size,
            "total_hits": total_hits,
            "average_hit_rate": total_hits / len(self.memory_cache) if self.memory_cache else 0,
            "redis_available": self.redis_client is not None
        }


def cached(ttl: int = 3600, key_prefix: str = "func"):
    """
    Decorator for caching async function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable):
        cache = PerformanceCache()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Initialize cache if needed
            if not cache.redis_client and not cache.memory_cache:
                await cache.initialize()
            
            # Generate cache key
            cache_key = cache._generate_key(
                f"{key_prefix}:{func.__name__}",
                {"args": args, "kwargs": kwargs}
            )
            
            # Check cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Monitor and track performance metrics.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, float] = {
            "query_interpretation": 1.0,  # seconds
            "cmr_search": 5.0,
            "analysis": 3.0,
            "response_synthesis": 1.0
        }
    
    async def record_metric(
        self, 
        operation: str, 
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Additional metadata
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        
        # Check threshold
        if operation in self.thresholds and duration > self.thresholds[operation]:
            logger.warning(
                f"Performance threshold exceeded for {operation}: "
                f"{duration:.2f}s > {self.thresholds[operation]}s"
            )
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary."""
        summary = {}
        
        for operation, durations in self.metrics.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p50": self._percentile(durations, 50),
                    "p95": self._percentile(durations, 95),
                    "p99": self._percentile(durations, 99)
                }
        
        return summary
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


def measure_time(operation: str):
    """
    Decorator to measure function execution time.
    
    Args:
        operation: Operation name for metrics
    """
    monitor = PerformanceMonitor()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                await monitor.record_metric(operation, duration)
        
        return wrapper
    return decorator


# Global instances
query_optimizer = QueryOptimizer()
performance_cache = PerformanceCache()
performance_monitor = PerformanceMonitor()