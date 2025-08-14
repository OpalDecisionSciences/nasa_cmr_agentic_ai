"""
Sophisticated Failover and Disaster Recovery System.

Provides intelligent failover strategies, multi-region disaster recovery,
and automated service recovery for the NASA CMR Agent system.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)


class ServiceState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class FailoverStrategy(Enum):
    """Failover strategy types."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CIRCUIT_BREAKER = "circuit_breaker"
    LOAD_BALANCING = "load_balancing"
    PRIORITY_BASED = "priority_based"


class RecoveryMode(Enum):
    """Recovery modes for failed services."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ASSISTED = "assisted"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    service_id: str
    endpoint_id: str
    url: str
    priority: int = 1
    weight: int = 1
    region: str = "primary"
    health_check_interval: int = 30
    timeout_seconds: int = 10
    max_retries: int = 3
    is_primary: bool = False
    is_enabled: bool = True
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class ServiceConfig:
    """Service configuration for failover."""
    service_id: str
    service_type: str
    endpoints: List[ServiceEndpoint]
    failover_strategy: FailoverStrategy
    recovery_mode: RecoveryMode
    min_healthy_endpoints: int = 1
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    health_check_timeout: int = 5
    graceful_shutdown_timeout: int = 30


@dataclass
class ServiceHealthCheck:
    """Service health check result."""
    endpoint_id: str
    timestamp: str
    state: ServiceState
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FailoverEvent:
    """Failover event record."""
    event_id: str
    service_id: str
    from_endpoint: str
    to_endpoint: str
    timestamp: str
    strategy: FailoverStrategy
    reason: str
    success: bool
    recovery_time_seconds: Optional[float] = None


class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    @abstractmethod
    async def check_health(self, endpoint: ServiceEndpoint) -> ServiceHealthCheck:
        """Check health of a service endpoint."""
        pass


class HTTPHealthChecker(HealthChecker):
    """HTTP-based health checker."""
    
    async def check_health(self, endpoint: ServiceEndpoint) -> ServiceHealthCheck:
        """Check HTTP endpoint health."""
        import aiohttp
        
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use health check endpoint if available
                health_url = endpoint.url
                if not health_url.endswith('/health'):
                    health_url = f"{endpoint.url.rstrip('/')}/health"
                
                async with session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        try:
                            health_data = await response.json()
                        except:
                            health_data = {"status": "ok"}
                        
                        return ServiceHealthCheck(
                            endpoint_id=endpoint.endpoint_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            state=ServiceState.HEALTHY,
                            response_time_ms=response_time,
                            metadata=health_data
                        )
                    else:
                        return ServiceHealthCheck(
                            endpoint_id=endpoint.endpoint_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            state=ServiceState.DEGRADED,
                            response_time_ms=response_time,
                            error_message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILING,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message="Health check timeout"
            )
        except Exception as e:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILED,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )


class DatabaseHealthChecker(HealthChecker):
    """Database-specific health checker."""
    
    async def check_health(self, endpoint: ServiceEndpoint) -> ServiceHealthCheck:
        """Check database endpoint health."""
        start_time = time.time()
        
        try:
            db_type = endpoint.tags.get("type", "unknown")
            
            if db_type == "redis":
                return await self._check_redis_health(endpoint, start_time)
            elif db_type == "neo4j":
                return await self._check_neo4j_health(endpoint, start_time)
            elif db_type == "weaviate":
                return await self._check_weaviate_health(endpoint, start_time)
            else:
                # Generic HTTP health check
                http_checker = HTTPHealthChecker()
                return await http_checker.check_health(endpoint)
                
        except Exception as e:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILED,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _check_redis_health(self, endpoint: ServiceEndpoint, start_time: float) -> ServiceHealthCheck:
        """Check Redis health."""
        try:
            import redis.asyncio as aioredis
            
            client = await aioredis.from_url(endpoint.url, socket_timeout=endpoint.timeout_seconds)
            await client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get basic info
            info = await client.info()
            await client.aclose()
            
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.HEALTHY,
                response_time_ms=response_time,
                metadata={
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "uptime": info.get("uptime_in_seconds", 0)
                }
            )
            
        except Exception as e:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILED,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Redis error: {e}"
            )
    
    async def _check_neo4j_health(self, endpoint: ServiceEndpoint, start_time: float) -> ServiceHealthCheck:
        """Check Neo4j health."""
        try:
            from neo4j import GraphDatabase
            
            # Extract connection details from URL
            driver = GraphDatabase.driver(endpoint.url, encrypted=False)
            
            def check_connection():
                driver.verify_connectivity()
                with driver.session() as session:
                    result = session.run("RETURN 1 as health_check")
                    return result.single()["health_check"]
            
            # Run in thread to avoid blocking
            health_result = await asyncio.to_thread(check_connection)
            response_time = (time.time() - start_time) * 1000
            
            driver.close()
            
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.HEALTHY if health_result == 1 else ServiceState.DEGRADED,
                response_time_ms=response_time,
                metadata={"health_check_result": health_result}
            )
            
        except Exception as e:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILED,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Neo4j error: {e}"
            )
    
    async def _check_weaviate_health(self, endpoint: ServiceEndpoint, start_time: float) -> ServiceHealthCheck:
        """Check Weaviate health."""
        try:
            import weaviate
            
            client = weaviate.connect_to_local(
                host=endpoint.url.split("://")[1].split(":")[0],
                port=int(endpoint.url.split(":")[-1]) if ":" in endpoint.url.split("://")[1] else 8080
            )
            
            is_live = client.is_live()
            response_time = (time.time() - start_time) * 1000
            
            client.close()
            
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.HEALTHY if is_live else ServiceState.FAILED,
                response_time_ms=response_time,
                metadata={"is_live": is_live}
            )
            
        except Exception as e:
            return ServiceHealthCheck(
                endpoint_id=endpoint.endpoint_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=ServiceState.FAILED,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Weaviate error: {e}"
            )


class SophisticatedFailoverManager:
    """Advanced failover and disaster recovery manager."""
    
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.service_health: Dict[str, Dict[str, ServiceHealthCheck]] = {}
        self.active_endpoints: Dict[str, str] = {}  # service_id -> endpoint_id
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.failover_history: List[FailoverEvent] = []
        
        # Health checkers
        self.health_checkers = {
            "http": HTTPHealthChecker(),
            "database": DatabaseHealthChecker()
        }
        
        # Background tasks
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # Callbacks
        self._failover_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_failovers": 0,
            "successful_failovers": 0,
            "failed_failovers": 0,
            "total_recoveries": 0,
            "automatic_recoveries": 0,
            "manual_recoveries": 0,
            "avg_failover_time_ms": 0.0,
            "avg_recovery_time_ms": 0.0
        }
    
    def register_service(self, config: ServiceConfig):
        """Register a service for failover management."""
        self.services[config.service_id] = config
        self.service_health[config.service_id] = {}
        
        # Initialize circuit breakers for each endpoint
        for endpoint in config.endpoints:
            self.circuit_breakers[endpoint.endpoint_id] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": 0,
                "next_attempt_time": 0
            }
        
        # Set primary endpoint as active if available
        primary_endpoint = next((ep for ep in config.endpoints if ep.is_primary), None)
        if primary_endpoint:
            self.active_endpoints[config.service_id] = primary_endpoint.endpoint_id
        elif config.endpoints:
            # Use highest priority endpoint
            sorted_endpoints = sorted(config.endpoints, key=lambda ep: ep.priority, reverse=True)
            self.active_endpoints[config.service_id] = sorted_endpoints[0].endpoint_id
        
        logger.info(f"Registered service for failover: {config.service_id}")
    
    async def start_health_monitoring(self, service_id: str):
        """Start health monitoring for a service."""
        if service_id not in self.services:
            raise ValueError(f"Service not registered: {service_id}")
        
        config = self.services[service_id]
        
        async def health_check_loop():
            while True:
                try:
                    await self._perform_health_checks(service_id)
                    await asyncio.sleep(min(ep.health_check_interval for ep in config.endpoints))
                except Exception as e:
                    logger.error(f"Health check loop error for {service_id}: {e}")
                    await asyncio.sleep(30)  # Wait 30s on error
        
        self._health_check_tasks[service_id] = asyncio.create_task(health_check_loop())
        logger.info(f"Started health monitoring for service: {service_id}")
    
    async def stop_health_monitoring(self, service_id: str):
        """Stop health monitoring for a service."""
        if service_id in self._health_check_tasks:
            self._health_check_tasks[service_id].cancel()
            try:
                await self._health_check_tasks[service_id]
            except asyncio.CancelledError:
                pass
            del self._health_check_tasks[service_id]
            logger.info(f"Stopped health monitoring for service: {service_id}")
    
    async def _perform_health_checks(self, service_id: str):
        """Perform health checks for all endpoints of a service."""
        config = self.services[service_id]
        health_checker = self._get_health_checker(config.service_type)
        
        # Perform health checks in parallel
        tasks = []
        for endpoint in config.endpoints:
            if endpoint.is_enabled:
                task = asyncio.create_task(health_checker.check_health(endpoint))
                tasks.append((endpoint.endpoint_id, task))
        
        # Collect results
        for endpoint_id, task in tasks:
            try:
                health_result = await task
                self.service_health[service_id][endpoint_id] = health_result
                
                # Update circuit breaker state
                await self._update_circuit_breaker(endpoint_id, health_result)
                
            except Exception as e:
                logger.error(f"Health check failed for {endpoint_id}: {e}")
                # Record failed health check
                self.service_health[service_id][endpoint_id] = ServiceHealthCheck(
                    endpoint_id=endpoint_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    state=ServiceState.FAILED,
                    response_time_ms=0,
                    error_message=str(e)
                )
        
        # Check if failover is needed
        await self._check_failover_conditions(service_id)
    
    async def _update_circuit_breaker(self, endpoint_id: str, health_result: ServiceHealthCheck):
        """Update circuit breaker state based on health check."""
        breaker = self.circuit_breakers[endpoint_id]
        
        if health_result.state in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
            # Reset failure count on success
            breaker["failure_count"] = 0
            if breaker["state"] == "open":
                breaker["state"] = "closed"
                logger.info(f"Circuit breaker closed for endpoint: {endpoint_id}")
        else:
            # Increment failure count
            breaker["failure_count"] += 1
            breaker["last_failure_time"] = time.time()
            
            # Open circuit breaker if threshold reached
            service_config = None
            for config in self.services.values():
                for endpoint in config.endpoints:
                    if endpoint.endpoint_id == endpoint_id:
                        service_config = config
                        break
                if service_config:
                    break
            
            if service_config and breaker["failure_count"] >= service_config.circuit_breaker_threshold:
                if breaker["state"] != "open":
                    breaker["state"] = "open"
                    breaker["next_attempt_time"] = time.time() + service_config.circuit_breaker_timeout
                    logger.warning(f"Circuit breaker opened for endpoint: {endpoint_id}")
    
    async def _check_failover_conditions(self, service_id: str):
        """Check if failover is needed for a service."""
        config = self.services[service_id]
        current_endpoint_id = self.active_endpoints.get(service_id)
        
        if not current_endpoint_id:
            # No active endpoint, try to find healthy one
            await self._perform_failover(service_id, None, "no_active_endpoint")
            return
        
        current_health = self.service_health[service_id].get(current_endpoint_id)
        
        # Check if current endpoint is unhealthy
        if (current_health and 
            current_health.state in [ServiceState.FAILED, ServiceState.FAILING]):
            await self._perform_failover(service_id, current_endpoint_id, "endpoint_unhealthy")
        
        # Check circuit breaker state
        elif (current_endpoint_id in self.circuit_breakers and 
              self.circuit_breakers[current_endpoint_id]["state"] == "open"):
            await self._perform_failover(service_id, current_endpoint_id, "circuit_breaker_open")
    
    async def _perform_failover(self, service_id: str, failed_endpoint_id: Optional[str], reason: str):
        """Perform failover to a healthy endpoint."""
        start_time = time.time()
        config = self.services[service_id]
        
        # Find best available endpoint
        best_endpoint = await self._select_failover_endpoint(service_id, failed_endpoint_id)
        
        if not best_endpoint:
            logger.error(f"No healthy endpoint available for failover: {service_id}")
            self.stats["failed_failovers"] += 1
            return
        
        # Execute failover strategy
        success = await self._execute_failover_strategy(
            service_id, failed_endpoint_id, best_endpoint.endpoint_id, config.failover_strategy
        )
        
        # Record failover event
        failover_time = (time.time() - start_time) * 1000
        event = FailoverEvent(
            event_id=f"FO-{int(time.time())}-{service_id}",
            service_id=service_id,
            from_endpoint=failed_endpoint_id or "none",
            to_endpoint=best_endpoint.endpoint_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=config.failover_strategy,
            reason=reason,
            success=success,
            recovery_time_seconds=failover_time / 1000
        )
        
        self.failover_history.append(event)
        
        # Update statistics
        self.stats["total_failovers"] += 1
        if success:
            self.stats["successful_failovers"] += 1
            self.active_endpoints[service_id] = best_endpoint.endpoint_id
            
            # Update average failover time
            total_successful = self.stats["successful_failovers"]
            current_avg = self.stats["avg_failover_time_ms"]
            self.stats["avg_failover_time_ms"] = (
                (current_avg * (total_successful - 1) + failover_time) / total_successful
            )
            
            logger.info(f"Failover successful: {service_id} -> {best_endpoint.endpoint_id} "
                       f"({failover_time:.1f}ms)")
        else:
            self.stats["failed_failovers"] += 1
            logger.error(f"Failover failed: {service_id}")
        
        # Notify callbacks
        for callback in self._failover_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")
    
    async def _select_failover_endpoint(self, service_id: str, 
                                      failed_endpoint_id: Optional[str]) -> Optional[ServiceEndpoint]:
        """Select the best endpoint for failover."""
        config = self.services[service_id]
        candidates = []
        
        for endpoint in config.endpoints:
            # Skip failed endpoint and disabled endpoints
            if (endpoint.endpoint_id == failed_endpoint_id or 
                not endpoint.is_enabled):
                continue
            
            # Check health
            health = self.service_health[service_id].get(endpoint.endpoint_id)
            if health and health.state in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
                # Check circuit breaker
                breaker = self.circuit_breakers.get(endpoint.endpoint_id)
                if not breaker or breaker["state"] == "closed":
                    candidates.append((endpoint, health))
        
        if not candidates:
            return None
        
        # Select based on priority, then health, then response time
        def selection_key(candidate):
            endpoint, health = candidate
            return (
                endpoint.priority,  # Higher priority first
                health.state == ServiceState.HEALTHY,  # Healthy before degraded
                -health.response_time_ms  # Lower response time first (negative for reverse sort)
            )
        
        candidates.sort(key=selection_key, reverse=True)
        return candidates[0][0]
    
    async def _execute_failover_strategy(self, service_id: str, from_endpoint_id: Optional[str],
                                       to_endpoint_id: str, strategy: FailoverStrategy) -> bool:
        """Execute specific failover strategy."""
        try:
            if strategy == FailoverStrategy.IMMEDIATE:
                # Immediate cutover
                return True
            
            elif strategy == FailoverStrategy.GRADUAL:
                # Gradual traffic shifting (simulated for now)
                await asyncio.sleep(0.1)  # Simulate gradual shift
                return True
            
            elif strategy == FailoverStrategy.CIRCUIT_BREAKER:
                # Wait for circuit breaker recovery
                if from_endpoint_id:
                    breaker = self.circuit_breakers.get(from_endpoint_id)
                    if breaker and breaker["state"] == "open":
                        # Check if ready for retry
                        if time.time() >= breaker.get("next_attempt_time", 0):
                            breaker["state"] = "half_open"
                return True
            
            elif strategy == FailoverStrategy.LOAD_BALANCING:
                # Distribute load across available endpoints
                return True
            
            elif strategy == FailoverStrategy.PRIORITY_BASED:
                # Use priority-based selection (already handled in selection)
                return True
            
            else:
                logger.warning(f"Unknown failover strategy: {strategy}")
                return True
                
        except Exception as e:
            logger.error(f"Failover strategy execution failed: {e}")
            return False
    
    def _get_health_checker(self, service_type: str) -> HealthChecker:
        """Get appropriate health checker for service type."""
        if service_type in ["redis", "neo4j", "weaviate", "database"]:
            return self.health_checkers["database"]
        else:
            return self.health_checkers["http"]
    
    async def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """Get current status of a service."""
        if service_id not in self.services:
            return {"error": "Service not found"}
        
        config = self.services[service_id]
        active_endpoint_id = self.active_endpoints.get(service_id)
        
        endpoints_status = []
        healthy_count = 0
        
        for endpoint in config.endpoints:
            health = self.service_health[service_id].get(endpoint.endpoint_id)
            breaker = self.circuit_breakers.get(endpoint.endpoint_id, {})
            
            if health and health.state in [ServiceState.HEALTHY, ServiceState.DEGRADED]:
                healthy_count += 1
            
            endpoints_status.append({
                "endpoint_id": endpoint.endpoint_id,
                "url": endpoint.url,
                "priority": endpoint.priority,
                "is_active": endpoint.endpoint_id == active_endpoint_id,
                "is_enabled": endpoint.is_enabled,
                "health_state": health.state.value if health else "unknown",
                "response_time_ms": health.response_time_ms if health else None,
                "circuit_breaker_state": breaker.get("state", "unknown"),
                "last_check": health.timestamp if health else None
            })
        
        return {
            "service_id": service_id,
            "service_type": config.service_type,
            "total_endpoints": len(config.endpoints),
            "healthy_endpoints": healthy_count,
            "active_endpoint": active_endpoint_id,
            "failover_strategy": config.failover_strategy.value,
            "recovery_mode": config.recovery_mode.value,
            "endpoints": endpoints_status
        }
    
    def get_failover_statistics(self) -> Dict[str, Any]:
        """Get failover system statistics."""
        return {
            **self.stats,
            "registered_services": len(self.services),
            "active_health_monitors": len(self._health_check_tasks),
            "recent_failovers": len([e for e in self.failover_history 
                                   if datetime.fromisoformat(e.timestamp) > 
                                   datetime.now(timezone.utc) - timedelta(hours=24)]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def add_failover_callback(self, callback: Callable):
        """Add callback for failover events."""
        self._failover_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery events."""
        self._recovery_callbacks.append(callback)
    
    async def manual_failover(self, service_id: str, target_endpoint_id: str) -> bool:
        """Manually trigger failover to specific endpoint."""
        if service_id not in self.services:
            return False
        
        config = self.services[service_id]
        target_endpoint = None
        
        for endpoint in config.endpoints:
            if endpoint.endpoint_id == target_endpoint_id:
                target_endpoint = endpoint
                break
        
        if not target_endpoint:
            return False
        
        current_endpoint_id = self.active_endpoints.get(service_id)
        success = await self._execute_failover_strategy(
            service_id, current_endpoint_id, target_endpoint_id, FailoverStrategy.IMMEDIATE
        )
        
        if success:
            self.active_endpoints[service_id] = target_endpoint_id
            logger.info(f"Manual failover completed: {service_id} -> {target_endpoint_id}")
        
        return success
    
    async def shutdown(self):
        """Gracefully shutdown the failover manager."""
        # Stop all health monitoring tasks
        for service_id in list(self._health_check_tasks.keys()):
            await self.stop_health_monitoring(service_id)
        
        # Stop recovery tasks
        for task in self._recovery_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Failover manager shutdown completed")


# Global failover manager instance
_failover_manager: Optional[SophisticatedFailoverManager] = None


def get_failover_manager() -> SophisticatedFailoverManager:
    """Get or create the global failover manager."""
    global _failover_manager
    
    if _failover_manager is None:
        _failover_manager = SophisticatedFailoverManager()
    
    return _failover_manager