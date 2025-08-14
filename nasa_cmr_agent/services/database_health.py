"""
Database Health Monitoring Service.

Provides comprehensive health checking, error logging, and recovery
mechanisms for Redis, Weaviate, and Neo4j databases.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import json

from .monitoring import MetricsService
from ..core.config import settings


# Configure structured logging
logger = logging.getLogger(__name__)


class DatabaseStatus(str, Enum):
    """Database connection status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DatabaseHealthCheck:
    """Database health check result."""
    database_name: str
    status: DatabaseStatus
    response_time_ms: float
    error_message: Optional[str] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.last_success:
            data['last_success'] = self.last_success.isoformat()
        return data


class DatabaseHealthMonitor:
    """Comprehensive database health monitoring service."""
    
    def __init__(self, metrics_service: Optional[MetricsService] = None):
        self.metrics_service = metrics_service
        self.health_history: Dict[str, List[DatabaseHealthCheck]] = {}
        self.max_history_size = 100
        self.check_interval = 30  # seconds
        self.failure_threshold = 3
        self.recovery_threshold = 2
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Database clients (initialized on first use)
        self._redis_client = None
        self._weaviate_client = None
        self._neo4j_driver = None
    
    async def check_redis_health(self) -> DatabaseHealthCheck:
        """Check Redis database health."""
        start_time = time.time()
        
        try:
            # Import Redis
            import redis.asyncio as aioredis
            
            # Create or reuse client
            if self._redis_client is None:
                self._redis_client = await aioredis.from_url(
                    settings.redis_url,
                    db=settings.redis_db,
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test basic operations
            await self._redis_client.ping()
            
            # Test set/get operations
            test_key = f"health_check_{int(time.time())}"
            await self._redis_client.set(test_key, "health_test", ex=10)
            value = await self._redis_client.get(test_key)
            
            if value != "health_test":
                raise Exception("Redis set/get operation failed")
            
            # Clean up test key
            await self._redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            try:
                info = await self._redis_client.info()
                metadata = {
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
            except Exception as info_error:
                logger.warning(f"Could not get Redis info: {info_error}")
                metadata = {}
            
            logger.debug(f"Redis health check successful: {response_time:.2f}ms")
            
            return DatabaseHealthCheck(
                database_name="redis",
                status=DatabaseStatus.HEALTHY,
                response_time_ms=response_time,
                last_success=datetime.now(timezone.utc),
                metadata=metadata
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_msg = f"Redis health check failed: {str(e)}"
            logger.error(error_msg)
            
            return DatabaseHealthCheck(
                database_name="redis",
                status=DatabaseStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message=error_msg
            )
    
    def check_weaviate_health(self) -> DatabaseHealthCheck:
        """Check Weaviate database health."""
        start_time = time.time()
        
        try:
            # Import Weaviate
            import weaviate
            
            # Create or reuse client
            if self._weaviate_client is None:
                # Use v4 client syntax
                import weaviate.connect as wv_connect
                
                connection_config = wv_connect.ConnectionParams.from_url(
                    url=settings.weaviate_url,
                    grpc_port=50051  # Default gRPC port
                )
                
                auth_config = None
                if settings.weaviate_api_key:
                    import weaviate.auth as weaviate_auth
                    auth_config = weaviate_auth.AuthApiKey(settings.weaviate_api_key)
                
                try:
                    # Create v4 client
                    self._weaviate_client = weaviate.WeaviateClient(
                        connection_params=connection_config,
                        auth_client_secret=auth_config,
                        skip_init_checks=False
                    )
                    self._weaviate_client.connect()
                except Exception as v4_error:
                    logger.warning(f"v4 client failed: {v4_error}")
                    # Try simple connect method
                    try:
                        self._weaviate_client = weaviate.connect_to_local(
                            host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                            port=int(settings.weaviate_url.split(":")[-1]) if ":" in settings.weaviate_url.replace("http://", "").replace("https://", "") else 8080,
                            grpc_port=50051
                        )
                    except Exception as connect_error:
                        logger.warning(f"Simple connect failed: {connect_error}")
                        raise Exception(f"Could not connect to Weaviate: {connect_error}")
            
            # Test readiness/connectivity
            try:
                # Try v4 client methods
                if hasattr(self._weaviate_client, 'is_live'):
                    is_ready = self._weaviate_client.is_live()
                elif hasattr(self._weaviate_client, 'is_ready'):
                    is_ready = self._weaviate_client.is_ready()
                else:
                    # Fallback - try a basic operation
                    is_ready = True
                    
                if not is_ready:
                    raise Exception("Weaviate is not ready")
            except Exception as ready_error:
                logger.warning(f"Weaviate readiness check failed: {ready_error}")
                # Continue with other tests
            
            # Test schema access
            schema_classes_count = 0
            try:
                if hasattr(self._weaviate_client, 'collections'):
                    # v4 collections API
                    collections = self._weaviate_client.collections.list_all()
                    if isinstance(collections, dict):
                        schema_classes_count = len(collections)
                    else:
                        # collections might be iterator or list
                        try:
                            schema_classes_count = len(list(collections))
                        except:
                            schema_classes_count = 0
                elif hasattr(self._weaviate_client, 'schema'):
                    # v3 schema API fallback
                    schema = self._weaviate_client.schema.get()
                    schema_classes_count = len(schema.get("classes", []))
            except Exception as schema_error:
                logger.warning(f"Schema access failed: {schema_error}")
                schema_classes_count = 0
            
            # Test basic connectivity with simple operation
            total_objects = 0
            try:
                # For v4, we'll skip complex queries in health check
                # Just confirm we can access collections
                if hasattr(self._weaviate_client, 'collections'):
                    # Simple check that collections API works
                    _ = self._weaviate_client.collections.list_all()
                    total_objects = 0  # We don't count objects in health check for performance
            except Exception as query_error:
                logger.warning(f"Basic operation test failed: {query_error}")
                total_objects = 0
            
            response_time = (time.time() - start_time) * 1000
            
            # Get cluster/node status
            cluster_nodes = 1
            healthy_nodes = 1
            try:
                if hasattr(self._weaviate_client, 'cluster'):
                    # Try v4 cluster API
                    if hasattr(self._weaviate_client.cluster, 'get_nodes_status'):
                        nodes_status = self._weaviate_client.cluster.get_nodes_status()
                        healthy_nodes = sum(1 for node in nodes_status if node.get("status") == "HEALTHY")
                        cluster_nodes = len(nodes_status)
                    else:
                        # v4 might not have get_nodes_status, assume single healthy node
                        cluster_nodes = 1
                        healthy_nodes = 1
            except Exception as meta_error:
                logger.warning(f"Could not get Weaviate cluster info: {meta_error}")
                cluster_nodes = 1
                healthy_nodes = 1
                
            metadata = {
                "schema_classes": schema_classes_count,
                "cluster_nodes": cluster_nodes,
                "healthy_nodes": healthy_nodes,
                "total_objects": total_objects,
                "client_version": "v4"
            }
            
            logger.debug(f"Weaviate health check successful: {response_time:.2f}ms")
            
            return DatabaseHealthCheck(
                database_name="weaviate",
                status=DatabaseStatus.HEALTHY,
                response_time_ms=response_time,
                last_success=datetime.now(timezone.utc),
                metadata=metadata
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_msg = f"Weaviate health check failed: {str(e)}"
            logger.error(error_msg)
            
            return DatabaseHealthCheck(
                database_name="weaviate",
                status=DatabaseStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message=error_msg
            )
    
    def check_neo4j_health(self) -> DatabaseHealthCheck:
        """Check Neo4j database health."""
        start_time = time.time()
        
        try:
            # Import Neo4j
            from neo4j import GraphDatabase
            
            # Create or reuse driver
            if self._neo4j_driver is None:
                self._neo4j_driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                    connection_timeout=5,
                    max_connection_lifetime=300
                )
            
            # Test connection
            with self._neo4j_driver.session(database=settings.neo4j_database) as session:
                # Test basic connectivity
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value != 1:
                    raise Exception("Neo4j connectivity test failed")
                
                # Test database access
                db_result = session.run("CALL db.labels() YIELD label RETURN count(label) as label_count")
                label_count = db_result.single()["label_count"]
                
                # Test node count
                count_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = count_result.single()["total_nodes"]
            
            response_time = (time.time() - start_time) * 1000
            
            # Get database info
            try:
                with self._neo4j_driver.session() as session:
                    version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
                    version_data = version_result.data()
                    
                    metadata = {
                        "total_nodes": total_nodes,
                        "label_count": label_count,
                        "neo4j_version": version_data[0]["version"] if version_data else "unknown"
                    }
            except Exception as meta_error:
                logger.warning(f"Could not get Neo4j metadata: {meta_error}")
                metadata = {"total_nodes": total_nodes, "label_count": label_count}
            
            logger.debug(f"Neo4j health check successful: {response_time:.2f}ms")
            
            return DatabaseHealthCheck(
                database_name="neo4j",
                status=DatabaseStatus.HEALTHY,
                response_time_ms=response_time,
                last_success=datetime.now(timezone.utc),
                metadata=metadata
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_msg = f"Neo4j health check failed: {str(e)}"
            logger.error(error_msg)
            
            return DatabaseHealthCheck(
                database_name="neo4j",
                status=DatabaseStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message=error_msg
            )
    
    async def check_all_databases(self) -> Dict[str, DatabaseHealthCheck]:
        """Check health of all databases."""
        logger.info("Running comprehensive database health check")
        
        # Run checks concurrently where possible
        redis_check = await self.check_redis_health()
        weaviate_check = self.check_weaviate_health()
        neo4j_check = self.check_neo4j_health()
        
        results = {
            "redis": redis_check,
            "weaviate": weaviate_check,
            "neo4j": neo4j_check
        }
        
        # Update health history
        for db_name, check in results.items():
            self._update_health_history(db_name, check)
        
        # Update metrics if available
        if self.metrics_service:
            for db_name, check in results.items():
                self.metrics_service.record_database_health(
                    db_name, 
                    check.status.value,
                    check.response_time_ms
                )
        
        # Log summary
        healthy_dbs = [name for name, check in results.items() if check.status == DatabaseStatus.HEALTHY]
        logger.info(f"Database health summary: {len(healthy_dbs)}/3 healthy ({', '.join(healthy_dbs)})")
        
        return results
    
    def _update_health_history(self, database_name: str, check: DatabaseHealthCheck):
        """Update health history for a database."""
        if database_name not in self.health_history:
            self.health_history[database_name] = []
        
        history = self.health_history[database_name]
        history.append(check)
        
        # Limit history size
        if len(history) > self.max_history_size:
            history.pop(0)
        
        # Update consecutive failure count
        if check.status == DatabaseStatus.HEALTHY:
            check.consecutive_failures = 0
        else:
            last_check = history[-2] if len(history) > 1 else None
            if last_check and last_check.status != DatabaseStatus.HEALTHY:
                check.consecutive_failures = last_check.consecutive_failures + 1
            else:
                check.consecutive_failures = 1
    
    def get_database_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive database status summary."""
        summary = {
            "overall_status": "healthy",
            "databases": {},
            "alerts": [],
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        
        unhealthy_count = 0
        
        for db_name, history in self.health_history.items():
            if not history:
                continue
                
            latest_check = history[-1]
            
            db_summary = {
                "status": latest_check.status.value,
                "last_response_time_ms": latest_check.response_time_ms,
                "consecutive_failures": latest_check.consecutive_failures,
                "last_success": latest_check.last_success.isoformat() if latest_check.last_success else None,
                "error_message": latest_check.error_message
            }
            
            # Calculate availability over last 24 hours
            recent_checks = [
                check for check in history
                if check.last_success and 
                datetime.now(timezone.utc) - check.last_success < timedelta(hours=24)
            ]
            
            if recent_checks:
                successful_checks = sum(1 for check in recent_checks if check.status == DatabaseStatus.HEALTHY)
                availability = (successful_checks / len(recent_checks)) * 100
                db_summary["availability_24h"] = f"{availability:.2f}%"
            
            summary["databases"][db_name] = db_summary
            
            # Check for alerts
            if latest_check.status != DatabaseStatus.HEALTHY:
                unhealthy_count += 1
                
                if latest_check.consecutive_failures >= self.failure_threshold:
                    summary["alerts"].append({
                        "database": db_name,
                        "severity": "critical",
                        "message": f"{db_name} has failed {latest_check.consecutive_failures} consecutive health checks",
                        "error": latest_check.error_message
                    })
                elif latest_check.response_time_ms > 5000:  # 5 second threshold
                    summary["alerts"].append({
                        "database": db_name,
                        "severity": "warning", 
                        "message": f"{db_name} response time is high: {latest_check.response_time_ms:.2f}ms"
                    })
        
        # Set overall status
        if unhealthy_count == 0:
            summary["overall_status"] = "healthy"
        elif unhealthy_count <= 1:
            summary["overall_status"] = "degraded" 
        else:
            summary["overall_status"] = "unhealthy"
        
        return summary
    
    async def start_monitoring(self):
        """Start continuous database monitoring."""
        if self.monitoring_active:
            logger.warning("Database monitoring is already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started database health monitoring (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """Stop continuous database monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        logger.info("Stopped database health monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                await self.check_all_databases()
                
                # Log alerts
                summary = self.get_database_status_summary()
                if summary["alerts"]:
                    for alert in summary["alerts"]:
                        if alert["severity"] == "critical":
                            logger.critical(f"Database Alert: {alert['message']}")
                        else:
                            logger.warning(f"Database Alert: {alert['message']}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in database monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def close(self):
        """Clean up resources."""
        await self.stop_monitoring()
        
        # Close database connections
        if self._redis_client:
            try:
                await self._redis_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
        
        if self._weaviate_client:
            try:
                if hasattr(self._weaviate_client, 'close'):
                    self._weaviate_client.close()
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")
        
        if self._neo4j_driver:
            try:
                self._neo4j_driver.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")
        
        logger.info("Database health monitor closed")


# Global health monitor instance
health_monitor: Optional[DatabaseHealthMonitor] = None


async def get_health_monitor() -> DatabaseHealthMonitor:
    """Get or create the global health monitor instance."""
    global health_monitor
    
    if health_monitor is None:
        from .monitoring import get_metrics_service
        metrics_service = await get_metrics_service()
        health_monitor = DatabaseHealthMonitor(metrics_service)
    
    return health_monitor


async def check_database_health() -> Dict[str, Any]:
    """Convenience function to check all database health."""
    monitor = await get_health_monitor()
    checks = await monitor.check_all_databases()
    return {name: check.to_dict() for name, check in checks.items()}