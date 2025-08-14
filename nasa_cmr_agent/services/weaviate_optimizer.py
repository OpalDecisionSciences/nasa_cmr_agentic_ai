"""
Weaviate Performance Optimization Service.

Implements connection pooling, query caching, and response optimization
for improved Weaviate performance and reduced response times.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import weaviate
from contextlib import asynccontextmanager
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class WeaviateConnectionPool:
    """Optimized Weaviate connection pool with caching."""
    
    def __init__(self, pool_size: int = 5, cache_ttl_seconds: int = 300):
        self.pool_size = pool_size
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Connection pool
        self._connections: List[weaviate.WeaviateClient] = []
        self._available_connections = asyncio.Queue(maxsize=pool_size)
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        
        # Query cache
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Performance metrics
        self._connection_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time_ms": 0.0,
            "pool_utilization": 0.0
        }
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        async with self._pool_lock:
            if self._initialized:
                return
                
            logger.info(f"Initializing Weaviate connection pool (size: {self.pool_size})")
            
            try:
                # Create initial connections
                for i in range(self.pool_size):
                    connection = await self._create_connection()
                    if connection:
                        self._connections.append(connection)
                        await self._available_connections.put(connection)
                        
                logger.info(f"Weaviate connection pool initialized with {len(self._connections)} connections")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize Weaviate connection pool: {e}")
                raise
    
    async def _create_connection(self) -> Optional[weaviate.WeaviateClient]:
        """Create a new optimized Weaviate connection."""
        try:
            client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051
            )
            
            # Test connection
            if client and hasattr(client, 'is_live'):
                try:
                    is_live = client.is_live()
                    if is_live:
                        return client
                except:
                    pass
            
            logger.warning("Created Weaviate connection but health check unclear")
            return client
            
        except Exception as e:
            logger.warning(f"Failed to create Weaviate connection: {e}")
            return None
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        connection = None
        try:
            # Try to get connection from pool with timeout
            try:
                connection = await asyncio.wait_for(
                    self._available_connections.get(), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Create new connection if pool is exhausted
                logger.info("Connection pool exhausted, creating new connection")
                connection = await self._create_connection()
                if not connection:
                    raise Exception("Could not create new Weaviate connection")
            
            # Update pool utilization metrics
            pool_size = self._available_connections.qsize()
            self._connection_metrics["pool_utilization"] = 1.0 - (pool_size / self.pool_size)
            
            yield connection
            
        finally:
            # Return connection to pool if it's still valid
            if connection:
                try:
                    # Quick health check
                    if hasattr(connection, 'is_live'):
                        connection.is_live()
                    
                    # Return to pool if there's space
                    if not self._available_connections.full():
                        await self._available_connections.put(connection)
                    else:
                        # Close excess connection
                        try:
                            connection.close()
                        except:
                            pass
                            
                except Exception as e:
                    logger.debug(f"Connection health check failed, not returning to pool: {e}")
                    try:
                        connection.close()
                    except:
                        pass
    
    def _cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        cache_data = {
            "operation": operation,
            "params": params,
            "timestamp_bucket": int(time.time() / 60)  # 1-minute buckets
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        async with self._cache_lock:
            if cache_key in self._query_cache:
                cached = self._query_cache[cache_key]
                cache_time = datetime.fromisoformat(cached["timestamp"])
                age_seconds = (datetime.now(timezone.utc) - cache_time).total_seconds()
                
                if age_seconds < self.cache_ttl_seconds:
                    self._connection_metrics["cache_hits"] += 1
                    logger.debug(f"Cache hit for key {cache_key[:8]}... (age: {age_seconds:.1f}s)")
                    return cached["result"]
                else:
                    # Remove expired cache entry
                    del self._query_cache[cache_key]
            
            self._connection_metrics["cache_misses"] += 1
            return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache query result."""
        async with self._cache_lock:
            self._query_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Cleanup old cache entries (keep cache size reasonable)
            if len(self._query_cache) > 1000:
                # Remove oldest 10% of entries
                sorted_entries = sorted(
                    self._query_cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                for key, _ in sorted_entries[:100]:
                    del self._query_cache[key]
    
    async def list_collections(self, use_cache: bool = True) -> Dict[str, Any]:
        """List collections with caching optimization."""
        operation = "list_collections"
        params = {}
        cache_key = self._cache_key(operation, params)
        
        # Check cache first
        if use_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Execute query
        start_time = time.time()
        self._connection_metrics["total_requests"] += 1
        
        try:
            async with self.get_connection() as client:
                collections = client.collections.list_all()
                
                result = {
                    "collections": collections if isinstance(collections, dict) else {},
                    "count": len(collections) if isinstance(collections, dict) else 0,
                    "operation": operation,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Cache result
                if use_cache:
                    await self._cache_result(cache_key, result)
                
                # Update performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self._update_response_time_metric(duration_ms)
                
                logger.debug(f"Weaviate list_collections completed in {duration_ms:.1f}ms")
                return result
                
        except Exception as e:
            logger.error(f"Weaviate list_collections failed: {e}")
            raise
    
    async def query_collections(self, class_name: str, query_params: Dict[str, Any], 
                              use_cache: bool = True) -> Dict[str, Any]:
        """Query collections with caching optimization."""
        operation = "query_collections"
        params = {"class_name": class_name, **query_params}
        cache_key = self._cache_key(operation, params)
        
        # Check cache first
        if use_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Execute query
        start_time = time.time()
        self._connection_metrics["total_requests"] += 1
        
        try:
            async with self.get_connection() as client:
                # This would be a more complex query implementation
                # For now, return a basic structure
                result = {
                    "class_name": class_name,
                    "query_params": query_params,
                    "results": [],  # Would contain actual query results
                    "count": 0,
                    "operation": operation,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Cache result
                if use_cache:
                    await self._cache_result(cache_key, result)
                
                # Update performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self._update_response_time_metric(duration_ms)
                
                logger.debug(f"Weaviate query completed in {duration_ms:.1f}ms")
                return result
                
        except Exception as e:
            logger.error(f"Weaviate query failed: {e}")
            raise
    
    def _update_response_time_metric(self, duration_ms: float):
        """Update average response time metric."""
        current_avg = self._connection_metrics["avg_response_time_ms"]
        total_requests = self._connection_metrics["total_requests"]
        
        # Calculate new average
        if total_requests == 1:
            self._connection_metrics["avg_response_time_ms"] = duration_ms
        else:
            # Weighted average with more weight on recent requests
            weight = min(0.1, 1.0 / total_requests)
            self._connection_metrics["avg_response_time_ms"] = (
                current_avg * (1 - weight) + duration_ms * weight
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_hit_rate = 0.0
        total_cache_requests = (
            self._connection_metrics["cache_hits"] + 
            self._connection_metrics["cache_misses"]
        )
        
        if total_cache_requests > 0:
            cache_hit_rate = self._connection_metrics["cache_hits"] / total_cache_requests
        
        return {
            **self._connection_metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._query_cache),
            "active_connections": len(self._connections),
            "available_connections": self._available_connections.qsize()
        }
    
    async def close(self):
        """Close all connections in the pool."""
        logger.info("Closing Weaviate connection pool")
        
        for connection in self._connections:
            try:
                if hasattr(connection, 'close'):
                    connection.close()
            except Exception as e:
                logger.debug(f"Error closing Weaviate connection: {e}")
        
        self._connections.clear()
        
        # Clear cache
        async with self._cache_lock:
            self._query_cache.clear()
        
        self._initialized = False
        logger.info("Weaviate connection pool closed")


# Global connection pool instance
_weaviate_pool: Optional[WeaviateConnectionPool] = None


async def get_weaviate_pool() -> WeaviateConnectionPool:
    """Get or create the global Weaviate connection pool."""
    global _weaviate_pool
    
    if _weaviate_pool is None:
        _weaviate_pool = WeaviateConnectionPool(
            pool_size=getattr(settings, 'weaviate_pool_size', 5),
            cache_ttl_seconds=getattr(settings, 'weaviate_cache_ttl', 300)
        )
        await _weaviate_pool.initialize()
    
    return _weaviate_pool