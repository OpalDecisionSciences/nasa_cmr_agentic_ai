import time
from typing import Dict, Any, Optional
import threading
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)


class MetricsService:
    """
    Performance monitoring and metrics collection service.
    
    Tracks system performance, query statistics, and error rates
    using Prometheus metrics that can be scraped by monitoring systems.
    """
    
    def __init__(self):
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Query metrics
        self.query_counter = Counter(
            'cmr_agent_queries_total',
            'Total number of queries processed',
            ['status', 'intent'],
            registry=self.registry
        )
        
        self.query_duration = Histogram(
            'cmr_agent_query_duration_seconds',
            'Query processing duration in seconds',
            ['intent'],
            registry=self.registry
        )
        
        self.recommendation_count = Histogram(
            'cmr_agent_recommendations_count',
            'Number of recommendations generated per query',
            ['intent'],
            registry=self.registry
        )
        
        # CMR API metrics
        self.cmr_api_requests = Counter(
            'cmr_agent_cmr_requests_total',
            'Total CMR API requests made',
            ['endpoint', 'status'],
            registry=self.registry
        )
        
        self.cmr_api_duration = Histogram(
            'cmr_agent_cmr_request_duration_seconds',
            'CMR API request duration in seconds',
            ['endpoint'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'cmr_agent_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['service'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'cmr_agent_circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['service'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_requests = Counter(
            'cmr_agent_llm_requests_total',
            'Total LLM requests made',
            ['provider', 'status'],
            registry=self.registry
        )
        
        self.llm_tokens = Histogram(
            'cmr_agent_llm_tokens_used',
            'Number of tokens used in LLM requests',
            ['provider', 'type'],
            registry=self.registry
        )
        
        # System metrics
        self.active_queries = Gauge(
            'cmr_agent_active_queries',
            'Number of currently active queries',
            registry=self.registry
        )
        
        # Error tracking
        self.error_counter = Counter(
            'cmr_agent_errors_total',
            'Total errors by type and component',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # In-memory metrics for dashboard
        self.query_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=500)
        self.performance_stats = defaultdict(list)
        
        # Server reference
        self._metrics_server = None
        self._server_thread = None
        
        # Initialize start time for uptime calculation
        self._start_time = time.time()
    
    def start_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        try:
            def run_server():
                start_http_server(port, registry=self.registry)
                logger.info(f"Metrics server started on port {port}")
            
            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop_server(self):
        """Stop metrics server."""
        # Prometheus server doesn't have a clean shutdown method
        # In production, this would be handled by the container orchestrator
        if self._server_thread:
            logger.info("Metrics server shutdown requested")
    
    # Query metrics
    def record_query_start(self, intent: str = "unknown"):
        """Record start of query processing."""
        self.active_queries.inc()
        
        # Add to history
        self.query_history.append({
            "timestamp": time.time(),
            "event": "start",
            "intent": intent
        })
    
    def record_query_completion(
        self, 
        success: bool, 
        processing_time_ms: int,
        intent: str = "unknown",
        recommendation_count: int = 0
    ):
        """Record completion of query processing."""
        self.active_queries.dec()
        
        # Update counters
        status = "success" if success else "failure"
        self.query_counter.labels(status=status, intent=intent).inc()
        
        # Update histograms
        processing_time_seconds = processing_time_ms / 1000.0
        self.query_duration.labels(intent=intent).observe(processing_time_seconds)
        self.recommendation_count.labels(intent=intent).observe(recommendation_count)
        
        # Add to history
        self.query_history.append({
            "timestamp": time.time(),
            "event": "completion",
            "success": success,
            "processing_time_ms": processing_time_ms,
            "intent": intent,
            "recommendation_count": recommendation_count
        })
        
        # Update performance stats
        self.performance_stats["processing_times"].append(processing_time_ms)
        if len(self.performance_stats["processing_times"]) > 100:
            self.performance_stats["processing_times"].pop(0)
    
    def record_query_error(self, error: str, component: str = "unknown"):
        """Record query processing error."""
        self.active_queries.dec()
        self.error_counter.labels(error_type="query_error", component=component).inc()
        
        self.error_history.append({
            "timestamp": time.time(),
            "error": error,
            "component": component
        })
    
    # CMR API metrics
    def record_cmr_request(
        self, 
        endpoint: str, 
        success: bool, 
        duration_seconds: float
    ):
        """Record CMR API request metrics."""
        status = "success" if success else "failure"
        self.cmr_api_requests.labels(endpoint=endpoint, status=status).inc()
        self.cmr_api_duration.labels(endpoint=endpoint).observe(duration_seconds)
    
    # Circuit breaker metrics
    def update_circuit_breaker_state(self, service: str, state: str):
        """Update circuit breaker state metrics."""
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        state_value = state_map.get(state.lower(), 0)
        self.circuit_breaker_state.labels(service=service).set(state_value)
    
    def record_circuit_breaker_failure(self, service: str):
        """Record circuit breaker failure."""
        self.circuit_breaker_failures.labels(service=service).inc()
    
    # LLM metrics
    def record_llm_request(
        self, 
        provider: str, 
        success: bool, 
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Record LLM request metrics."""
        status = "success" if success else "failure"
        self.llm_requests.labels(provider=provider, status=status).inc()
        
        if input_tokens > 0:
            self.llm_tokens.labels(provider=provider, type="input").observe(input_tokens)
        if output_tokens > 0:
            self.llm_tokens.labels(provider=provider, type="output").observe(output_tokens)
    
    # Dashboard support
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboard display."""
        current_time = time.time()
        
        # Recent queries (last 5 minutes)
        recent_queries = [
            q for q in self.query_history 
            if current_time - q["timestamp"] < 300
        ]
        
        successful_queries = [q for q in recent_queries if q.get("success", False)]
        
        # Performance statistics
        recent_times = [
            q["processing_time_ms"] for q in recent_queries 
            if "processing_time_ms" in q
        ]
        
        avg_processing_time = (
            sum(recent_times) / len(recent_times) 
            if recent_times else 0
        )
        
        # Error statistics
        recent_errors = [
            e for e in self.error_history
            if current_time - e["timestamp"] < 300
        ]
        
        return {
            "queries": {
                "total_recent": len(recent_queries),
                "successful": len(successful_queries),
                "success_rate": (
                    len(successful_queries) / len(recent_queries) 
                    if recent_queries else 1.0
                ),
                "average_processing_time_ms": avg_processing_time
            },
            "errors": {
                "recent_count": len(recent_errors),
                "error_rate": (
                    len(recent_errors) / len(recent_queries) 
                    if recent_queries else 0.0
                )
            },
            "system": {
                "active_queries": self.active_queries._value.get(),
                "uptime_minutes": (current_time - self._start_time) / 60.0 if hasattr(self, '_start_time') else 0
            }
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trend data for monitoring dashboards."""
        # This would return time-series data for charting
        return {
            "processing_times": list(self.performance_stats["processing_times"]),
            "query_volume": self._calculate_query_volume_trend(),
            "error_rates": self._calculate_error_rate_trend()
        }
    
    def _calculate_query_volume_trend(self) -> list:
        """Calculate query volume trend over time."""
        # Group queries by time buckets (e.g., 5-minute intervals)
        current_time = time.time()
        buckets = []
        
        for i in range(12):  # Last 12 intervals (1 hour if 5-min buckets)
            bucket_start = current_time - (i + 1) * 300  # 5 minutes
            bucket_end = current_time - i * 300
            
            count = len([
                q for q in self.query_history
                if bucket_start <= q["timestamp"] < bucket_end
            ])
            
            buckets.append({
                "timestamp": bucket_end,
                "count": count
            })
        
        return list(reversed(buckets))  # Return chronologically
    
    def _calculate_error_rate_trend(self) -> list:
        """Calculate error rate trend over time."""
        current_time = time.time()
        buckets = []
        
        for i in range(12):  # Last 12 intervals
            bucket_start = current_time - (i + 1) * 300
            bucket_end = current_time - i * 300
            
            queries_in_bucket = [
                q for q in self.query_history
                if bucket_start <= q["timestamp"] < bucket_end
            ]
            
            errors_in_bucket = [
                e for e in self.error_history
                if bucket_start <= e["timestamp"] < bucket_end
            ]
            
            error_rate = (
                len(errors_in_bucket) / len(queries_in_bucket)
                if queries_in_bucket else 0
            )
            
            buckets.append({
                "timestamp": bucket_end,
                "error_rate": error_rate
            })
        
        return list(reversed(buckets))

    def record_database_health(self, database_name: str, status: str, response_time_ms: float):
        """Record database health metrics."""
        # Create database health metrics if they don't exist
        if not hasattr(self, 'database_health_gauge'):
            self.database_health_gauge = Gauge(
                'cmr_agent_database_health_status',
                'Database health status (1=healthy, 0=unhealthy)',
                ['database'],
                registry=self.registry
            )
            
            self.database_response_time = Histogram(
                'cmr_agent_database_response_time_seconds',
                'Database response time in seconds',
                ['database'],
                registry=self.registry
            )
        
        # Record metrics
        health_value = 1 if status.lower() == 'healthy' else 0
        self.database_health_gauge.labels(database=database_name).set(health_value)
        self.database_response_time.labels(database=database_name).observe(response_time_ms / 1000.0)


# Global metrics service instance
_metrics_service: Optional[MetricsService] = None


async def get_metrics_service() -> Optional[MetricsService]:
    """Get or create the global metrics service instance."""
    global _metrics_service
    
    if _metrics_service is None:
        try:
            from ..core.config import settings
            if settings.enable_metrics:
                _metrics_service = MetricsService()
                _metrics_service.start_server(settings.prometheus_port)
        except Exception as e:
            logger.warning(f"Could not initialize metrics service: {e}")
            
    return _metrics_service