"""
Neo4j Performance Optimization Service.

Implements connection pooling, query caching, and response optimization
for improved Neo4j performance and reduced response times.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from neo4j import GraphDatabase, Result
from contextlib import asynccontextmanager
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class Neo4jConnectionPool:
    """Optimized Neo4j connection pool with caching."""
    
    def __init__(self, pool_size: int = 5, cache_ttl_seconds: int = 300):
        self.pool_size = pool_size
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Connection pool
        self._driver: Optional[GraphDatabase.driver] = None
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
            "active_sessions": 0
        }
    
    async def initialize(self):
        """Initialize the Neo4j driver with optimized settings."""
        if self._initialized:
            return
        
        async with self._pool_lock:
            if self._initialized:
                return
                
            logger.info(f"Initializing Neo4j connection pool (size: {self.pool_size})")
            
            try:
                # Create optimized driver
                self._driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                    # Optimization: Configure connection pool settings
                    max_connection_pool_size=self.pool_size,
                    max_connection_lifetime=3600,  # 1 hour
                    connection_acquisition_timeout=30,
                    # Enable encrypted connections if available
                    encrypted=getattr(settings, 'neo4j_encrypted', False)
                )
                
                # Test connection
                await asyncio.to_thread(self._driver.verify_connectivity)
                
                logger.info("Neo4j connection pool initialized successfully")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j connection pool: {e}")
                raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get a Neo4j session from the pool."""
        if not self._initialized:
            await self.initialize()
        
        session = None
        try:
            # Get session from driver
            session = self._driver.session(
                # Optimization: Use read/write modes appropriately
                default_access_mode="WRITE",
                # Connection pool will handle session management
                fetch_size=1000  # Optimize for bulk operations
            )
            
            # Update active session metrics
            self._connection_metrics["active_sessions"] += 1
            
            yield session
            
        finally:
            # Close session and update metrics
            if session:
                try:
                    await asyncio.to_thread(session.close)
                    self._connection_metrics["active_sessions"] -= 1
                except Exception as e:
                    logger.debug(f"Error closing Neo4j session: {e}")
    
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
    
    async def execute_read_query(self, cypher_query: str, parameters: Dict[str, Any] = None, 
                                use_cache: bool = True) -> Dict[str, Any]:
        """Execute optimized read query with caching."""
        operation = "read_query"
        params = {"query": cypher_query, "parameters": parameters or {}}
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
            async with self.get_session() as session:
                # Execute read query with optimization
                def _execute_read():
                    result = session.run(cypher_query, parameters or {})
                    records = [record.data() for record in result]
                    summary = result.consume()
                    return records, summary
                
                records, summary = await asyncio.to_thread(_execute_read)
                
                result = {
                    "records": records,
                    "record_count": len(records),
                    "query": cypher_query,
                    "parameters": parameters or {},
                    "operation": operation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "database_hits": getattr(summary.counters, 'nodes_created', 0) + 
                                   getattr(summary.counters, 'relationships_created', 0)
                }
                
                # Cache result for read queries
                if use_cache:
                    await self._cache_result(cache_key, result)
                
                # Update performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self._update_response_time_metric(duration_ms)
                
                logger.debug(f"Neo4j read query completed in {duration_ms:.1f}ms")
                return result
                
        except Exception as e:
            logger.error(f"Neo4j read query failed: {e}")
            raise
    
    async def execute_write_query(self, cypher_query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute optimized write query."""
        operation = "write_query"
        
        # Execute query (no caching for write operations)
        start_time = time.time()
        self._connection_metrics["total_requests"] += 1
        
        try:
            async with self.get_session() as session:
                # Execute write query in transaction
                def _execute_write():
                    with session.begin_transaction() as tx:
                        result = tx.run(cypher_query, parameters or {})
                        records = [record.data() for record in result]
                        summary = result.consume()
                        tx.commit()
                        return records, summary
                
                records, summary = await asyncio.to_thread(_execute_write)
                
                result = {
                    "records": records,
                    "record_count": len(records),
                    "query": cypher_query,
                    "parameters": parameters or {},
                    "operation": operation,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "nodes_created": getattr(summary.counters, 'nodes_created', 0),
                    "relationships_created": getattr(summary.counters, 'relationships_created', 0),
                    "properties_set": getattr(summary.counters, 'properties_set', 0)
                }
                
                # Update performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self._update_response_time_metric(duration_ms)
                
                logger.debug(f"Neo4j write query completed in {duration_ms:.1f}ms")
                return result
                
        except Exception as e:
            logger.error(f"Neo4j write query failed: {e}")
            raise
    
    async def execute_batch_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute batch operations for optimal performance."""
        start_time = time.time()
        self._connection_metrics["total_requests"] += len(operations)
        
        try:
            async with self.get_session() as session:
                def _execute_batch():
                    results = []
                    with session.begin_transaction() as tx:
                        for op in operations:
                            result = tx.run(op["query"], op.get("parameters", {}))
                            records = [record.data() for record in result]
                            summary = result.consume()
                            results.append({
                                "records": records,
                                "record_count": len(records),
                                "query": op["query"],
                                "summary": summary
                            })
                        tx.commit()
                        return results
                
                batch_results = await asyncio.to_thread(_execute_batch)
                
                total_records = sum(r["record_count"] for r in batch_results)
                
                result = {
                    "batch_results": batch_results,
                    "total_operations": len(operations),
                    "total_records": total_records,
                    "operation": "batch_operations",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Update performance metrics
                duration_ms = (time.time() - start_time) * 1000
                self._update_response_time_metric(duration_ms)
                
                logger.debug(f"Neo4j batch operations completed in {duration_ms:.1f}ms")
                return result
                
        except Exception as e:
            logger.error(f"Neo4j batch operations failed: {e}")
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
            "driver_initialized": self._initialized
        }
    
    async def close(self):
        """Close the Neo4j driver and cleanup."""
        logger.info("Closing Neo4j connection pool")
        
        if self._driver:
            try:
                await asyncio.to_thread(self._driver.close)
            except Exception as e:
                logger.debug(f"Error closing Neo4j driver: {e}")
        
        # Clear cache
        async with self._cache_lock:
            self._query_cache.clear()
        
        self._initialized = False
        logger.info("Neo4j connection pool closed")


# Global connection pool instance
_neo4j_pool: Optional[Neo4jConnectionPool] = None


async def get_neo4j_pool() -> Neo4jConnectionPool:
    """Get or create the global Neo4j connection pool."""
    global _neo4j_pool
    
    if _neo4j_pool is None:
        _neo4j_pool = Neo4jConnectionPool(
            pool_size=getattr(settings, 'neo4j_pool_size', 5),
            cache_ttl_seconds=getattr(settings, 'neo4j_cache_ttl', 300)
        )
        await _neo4j_pool.initialize()
    
    return _neo4j_pool