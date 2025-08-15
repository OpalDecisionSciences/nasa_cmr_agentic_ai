"""
Comprehensive Performance Benchmarking System for NASA CMR Agent.

Provides documented performance benchmarks, accuracy measurements, success rate tracking,
and LangGraph state monitoring for production performance analysis.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog

# Optional dependency for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = structlog.get_logger(__name__)


class BenchmarkCategory(Enum):
    """Categories of performance benchmarks."""
    QUERY_PROCESSING = "query_processing"
    DATABASE_PERFORMANCE = "database_performance"
    LANGGRAPH_STATE = "langgraph_state"
    ACCURACY_QUALITY = "accuracy_quality"
    SUCCESS_RATES = "success_rates"
    SYSTEM_RESOURCES = "system_resources"
    USER_SATISFACTION = "user_satisfaction"


class BenchmarkSeverity(Enum):
    """Benchmark threshold severity levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BenchmarkThreshold:
    """Performance benchmark threshold definition."""
    name: str
    category: BenchmarkCategory
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str
    higher_is_better: bool = True


@dataclass
class APIDiversificationMetrics:
    """API diversification performance and success metrics."""
    cmr_api_success_rate: float
    giovanni_api_success_rate: float
    modaps_api_success_rate: float
    atmospheric_api_success_rate: float
    earthdata_auth_success_rate: float
    overall_diversification_score: float
    api_response_times: Dict[str, float]
    api_availability: Dict[str, bool]
    fallback_activations: Dict[str, int]
    data_coverage_enhancement: float  # % improvement from diversification


@dataclass
class BenchmarkResult:
    """Individual benchmark measurement result."""
    benchmark_name: str
    category: BenchmarkCategory
    measured_value: float
    target_value: float
    threshold_status: BenchmarkSeverity
    unit: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class QueryAccuracyMetrics:
    """Comprehensive query accuracy measurement."""
    query_id: str
    relevance_score: float      # NDCG@10 score
    completeness_score: float  # Metadata completeness
    accessibility_score: float # Data availability 
    coverage_score: float      # Spatial/temporal alignment
    freshness_score: float     # Data recency
    overall_accuracy: float    # Weighted average
    user_rating: Optional[float] = None
    expert_validation: Optional[bool] = None


@dataclass
class GraphStateBenchmarks:
    """LangGraph state performance metrics."""
    workflow_id: str
    graph_traversal_time: Dict[str, float]    # Time per agent node
    state_transition_latency: float           # Inter-agent communication
    state_memory_usage: float                # Memory per workflow (MB)
    state_serialization_size: int           # State object size (bytes)
    concurrent_workflows: int               # Parallel executions
    retry_attempts: int                     # Supervisor retries
    workflow_completion_time: float        # Total end-to-end time
    error_recovery_time: Optional[float]   # Error recovery latency


class PerformanceBenchmarkSystem:
    """Comprehensive performance benchmarking and monitoring system."""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
        self.accuracy_results: List[QueryAccuracyMetrics] = []
        self.graph_benchmarks: List[GraphStateBenchmarks] = []
        
        # Rolling windows for trend analysis
        self.recent_results = deque(maxlen=1000)
        self.hourly_aggregates = deque(maxlen=168)  # 7 days
        
        # Performance tracking
        self.query_times = deque(maxlen=1000)
        self.success_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # System resource monitoring
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.start_time = time.time()
        
        self._initialize_benchmark_thresholds()
    
    def _initialize_benchmark_thresholds(self):
        """Initialize performance benchmark thresholds."""
        self.thresholds = {
            # Query Processing Benchmarks
            "end_to_end_query_time": BenchmarkThreshold(
                name="End-to-End Query Processing Time",
                category=BenchmarkCategory.QUERY_PROCESSING,
                target_value=5.0,
                warning_threshold=8.0,
                critical_threshold=15.0,
                unit="seconds",
                description="Complete query processing from input to response",
                higher_is_better=False
            ),
            
            "query_interpreter_time": BenchmarkThreshold(
                name="Query Interpreter Processing Time",
                category=BenchmarkCategory.QUERY_PROCESSING,
                target_value=0.5,
                warning_threshold=1.0,
                critical_threshold=2.0,
                unit="seconds",
                description="Query interpretation and intent classification",
                higher_is_better=False
            ),
            
            "cmr_api_response_time": BenchmarkThreshold(
                name="CMR API Response Time",
                category=BenchmarkCategory.QUERY_PROCESSING,
                target_value=2.0,
                warning_threshold=5.0,
                critical_threshold=10.0,
                unit="seconds",
                description="CMR API search and data retrieval",
                higher_is_better=False
            ),
            
            # Database Performance Benchmarks
            "vector_search_latency": BenchmarkThreshold(
                name="Vector Similarity Search Latency",
                category=BenchmarkCategory.DATABASE_PERFORMANCE,
                target_value=0.1,
                warning_threshold=0.3,
                critical_threshold=1.0,
                unit="seconds",
                description="Weaviate vector similarity search for top-20 results",
                higher_is_better=False
            ),
            
            "knowledge_graph_query_time": BenchmarkThreshold(
                name="Knowledge Graph Query Time",
                category=BenchmarkCategory.DATABASE_PERFORMANCE,
                target_value=0.3,
                warning_threshold=0.8,
                critical_threshold=2.0,
                unit="seconds",
                description="Neo4j relationship traversal and analysis",
                higher_is_better=False
            ),
            
            # LangGraph State Benchmarks
            "state_transition_latency": BenchmarkThreshold(
                name="State Transition Latency",
                category=BenchmarkCategory.LANGGRAPH_STATE,
                target_value=0.01,
                warning_threshold=0.05,
                critical_threshold=0.1,
                unit="seconds",
                description="Inter-agent state transitions in workflow",
                higher_is_better=False
            ),
            
            "state_memory_usage": BenchmarkThreshold(
                name="Workflow State Memory Usage",
                category=BenchmarkCategory.LANGGRAPH_STATE,
                target_value=25.0,
                warning_threshold=50.0,
                critical_threshold=100.0,
                unit="MB",
                description="Memory consumption per active workflow",
                higher_is_better=False
            ),
            
            # Accuracy and Quality Benchmarks
            "recommendation_relevance": BenchmarkThreshold(
                name="Recommendation Relevance (NDCG@10)",
                category=BenchmarkCategory.ACCURACY_QUALITY,
                target_value=0.85,
                warning_threshold=0.70,
                critical_threshold=0.50,
                unit="score",
                description="Normalized Discounted Cumulative Gain for top-10 recommendations",
                higher_is_better=True
            ),
            
            "query_intent_accuracy": BenchmarkThreshold(
                name="Query Intent Classification Accuracy",
                category=BenchmarkCategory.ACCURACY_QUALITY,
                target_value=0.90,
                warning_threshold=0.80,
                critical_threshold=0.65,
                unit="percentage",
                description="Accuracy of query intent classification",
                higher_is_better=True
            ),
            
            # Success Rate Benchmarks
            "query_success_rate": BenchmarkThreshold(
                name="Query Success Rate",
                category=BenchmarkCategory.SUCCESS_RATES,
                target_value=0.95,
                warning_threshold=0.90,
                critical_threshold=0.80,
                unit="percentage",
                description="Percentage of queries completed successfully",
                higher_is_better=True
            ),
            
            "supervisor_validation_rate": BenchmarkThreshold(
                name="Supervisor Validation Pass Rate",
                category=BenchmarkCategory.SUCCESS_RATES,
                target_value=0.85,
                warning_threshold=0.75,
                critical_threshold=0.60,
                unit="percentage",
                description="First-attempt supervisor validation success rate",
                higher_is_better=True
            ),
            
            "user_satisfaction_score": BenchmarkThreshold(
                name="User Satisfaction Score",
                category=BenchmarkCategory.USER_SATISFACTION,
                target_value=4.2,
                warning_threshold=3.5,
                critical_threshold=2.5,
                unit="rating",
                description="Average user satisfaction rating (1-5 scale)",
                higher_is_better=True
            ),
            
            # System Resource Benchmarks
            "concurrent_queries_supported": BenchmarkThreshold(
                name="Concurrent Queries Supported",
                category=BenchmarkCategory.SYSTEM_RESOURCES,
                target_value=50.0,
                warning_threshold=30.0,
                critical_threshold=10.0,
                unit="count",
                description="Maximum concurrent queries without performance degradation",
                higher_is_better=True
            ),
            
            "system_availability": BenchmarkThreshold(
                name="System Availability",
                category=BenchmarkCategory.SYSTEM_RESOURCES,
                target_value=0.995,
                warning_threshold=0.990,
                critical_threshold=0.980,
                unit="percentage",
                description="System uptime and availability",
                higher_is_better=True
            )
        }
    
    async def measure_query_performance(self, query_id: str, query_text: str, 
                                       start_time: float, end_time: float,
                                       agent_timings: Dict[str, float],
                                       success: bool, error: Optional[str] = None) -> BenchmarkResult:
        """Measure and record query performance benchmarks."""
        
        total_time = end_time - start_time
        self.query_times.append(total_time)
        
        if success:
            self.success_counts["total"] += 1
        else:
            self.error_counts["total"] += 1
            if error:
                self.error_counts[error] += 1
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name="end_to_end_query_time",
            category=BenchmarkCategory.QUERY_PROCESSING,
            measured_value=total_time,
            target_value=self.thresholds["end_to_end_query_time"].target_value,
            threshold_status=self._evaluate_threshold("end_to_end_query_time", total_time),
            unit="seconds",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "query_id": query_id,
                "query_text": query_text[:100],  # Truncate for storage
                "agent_timings": agent_timings,
                "success": success,
                "error": error,
                "memory_usage_mb": self._get_memory_usage()
            }
        )
        
        self.benchmark_results.append(result)
        self.recent_results.append(result)
        
        logger.info(f"Query performance measured: {total_time:.3f}s ({result.threshold_status.value})")
        return result
    
    async def measure_database_performance(self, operation: str, duration: float, 
                                         result_count: int, database: str) -> BenchmarkResult:
        """Measure database operation performance."""
        
        benchmark_name = f"{database.lower()}_query_time"
        if database.lower() == "weaviate":
            threshold_key = "vector_search_latency"
        elif database.lower() == "neo4j":
            threshold_key = "knowledge_graph_query_time"
        else:
            threshold_key = "vector_search_latency"  # Default
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            category=BenchmarkCategory.DATABASE_PERFORMANCE,
            measured_value=duration,
            target_value=self.thresholds[threshold_key].target_value,
            threshold_status=self._evaluate_threshold(threshold_key, duration),
            unit="seconds",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "operation": operation,
                "result_count": result_count,
                "database": database,
                "results_per_second": result_count / max(duration, 0.001)
            }
        )
        
        self.benchmark_results.append(result)
        return result
    
    async def measure_langgraph_state_performance(self, workflow_id: str,
                                                agent_timings: Dict[str, float],
                                                state_transitions: int,
                                                memory_usage: float,
                                                state_size: int) -> GraphStateBenchmarks:
        """Measure LangGraph state management performance."""
        
        total_workflow_time = sum(agent_timings.values())
        avg_transition_time = total_workflow_time / max(state_transitions, 1)
        
        graph_benchmark = GraphStateBenchmarks(
            workflow_id=workflow_id,
            graph_traversal_time=agent_timings,
            state_transition_latency=avg_transition_time,
            state_memory_usage=memory_usage,
            state_serialization_size=state_size,
            concurrent_workflows=len(self.graph_benchmarks),  # Current active count
            retry_attempts=0,  # Would be populated from actual workflow
            workflow_completion_time=total_workflow_time
        )
        
        self.graph_benchmarks.append(graph_benchmark)
        
        # Create benchmark results for key metrics
        transition_result = BenchmarkResult(
            benchmark_name="state_transition_latency",
            category=BenchmarkCategory.LANGGRAPH_STATE,
            measured_value=avg_transition_time,
            target_value=self.thresholds["state_transition_latency"].target_value,
            threshold_status=self._evaluate_threshold("state_transition_latency", avg_transition_time),
            unit="seconds",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "workflow_id": workflow_id,
                "agent_count": len(agent_timings),
                "state_transitions": state_transitions
            }
        )
        
        memory_result = BenchmarkResult(
            benchmark_name="state_memory_usage",
            category=BenchmarkCategory.LANGGRAPH_STATE,
            measured_value=memory_usage,
            target_value=self.thresholds["state_memory_usage"].target_value,
            threshold_status=self._evaluate_threshold("state_memory_usage", memory_usage),
            unit="MB",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "workflow_id": workflow_id,
                "state_size_bytes": state_size
            }
        )
        
        self.benchmark_results.extend([transition_result, memory_result])
        return graph_benchmark
    
    async def measure_query_accuracy(self, query_id: str, 
                                   recommendations: List[Dict[str, Any]],
                                   user_feedback: Optional[Dict[str, Any]] = None,
                                   expert_validation: Optional[bool] = None) -> QueryAccuracyMetrics:
        """Measure query result accuracy using NDCG and other quality metrics."""
        
        # Calculate NDCG@10 for relevance
        relevance_score = self._calculate_ndcg(recommendations, k=10)
        
        # Calculate other quality scores
        completeness_score = self._calculate_completeness_score(recommendations)
        accessibility_score = self._calculate_accessibility_score(recommendations)
        coverage_score = self._calculate_coverage_score(recommendations)
        freshness_score = self._calculate_freshness_score(recommendations)
        
        # Calculate overall accuracy (weighted average)
        weights = {
            "relevance": 0.35,
            "completeness": 0.25,
            "accessibility": 0.15,
            "coverage": 0.15,
            "freshness": 0.10
        }
        
        overall_accuracy = (
            relevance_score * weights["relevance"] +
            completeness_score * weights["completeness"] +
            accessibility_score * weights["accessibility"] +
            coverage_score * weights["coverage"] +
            freshness_score * weights["freshness"]
        )
        
        user_rating = user_feedback.get("rating") if user_feedback else None
        
        accuracy_metrics = QueryAccuracyMetrics(
            query_id=query_id,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accessibility_score=accessibility_score,
            coverage_score=coverage_score,
            freshness_score=freshness_score,
            overall_accuracy=overall_accuracy,
            user_rating=user_rating,
            expert_validation=expert_validation
        )
        
        self.accuracy_results.append(accuracy_metrics)
        
        # Create benchmark result
        accuracy_result = BenchmarkResult(
            benchmark_name="recommendation_relevance",
            category=BenchmarkCategory.ACCURACY_QUALITY,
            measured_value=relevance_score,
            target_value=self.thresholds["recommendation_relevance"].target_value,
            threshold_status=self._evaluate_threshold("recommendation_relevance", relevance_score),
            unit="score",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "query_id": query_id,
                "recommendation_count": len(recommendations),
                "overall_accuracy": overall_accuracy,
                "user_rating": user_rating
            }
        )
        
        self.benchmark_results.append(accuracy_result)
        return accuracy_metrics
    
    async def measure_api_diversification(self) -> APIDiversificationMetrics:
        """Measure API diversification performance and success metrics."""
        
        # Test each API service availability and performance
        api_tests = {
            "cmr": self._test_cmr_api,
            "giovanni": self._test_giovanni_api,
            "modaps": self._test_modaps_api,
            "atmospheric": self._test_atmospheric_api,
            "earthdata_auth": self._test_earthdata_auth
        }
        
        api_success_rates = {}
        api_response_times = {}
        api_availability = {}
        
        for api_name, test_func in api_tests.items():
            try:
                start_time = time.time()
                success = await test_func()
                response_time = time.time() - start_time
                
                api_success_rates[api_name] = 1.0 if success else 0.0
                api_response_times[api_name] = response_time
                api_availability[api_name] = success
                
            except Exception as e:
                logger.error(f"API diversification test failed for {api_name}: {e}")
                api_success_rates[api_name] = 0.0
                api_response_times[api_name] = 999.0  # High penalty for failure
                api_availability[api_name] = False
        
        # Calculate overall diversification score
        # Higher score when more APIs are available and performing well
        total_apis = len(api_tests)
        available_apis = sum(api_availability.values())
        avg_success_rate = sum(api_success_rates.values()) / total_apis
        avg_response_time = sum(api_response_times.values()) / total_apis
        
        # Diversification score factors:
        # 1. API availability (40%)
        # 2. Success rates (40%) 
        # 3. Performance (20%)
        availability_score = available_apis / total_apis
        performance_score = max(0, 1.0 - (avg_response_time - 1.0) / 10.0)  # Penalty for >1s response
        
        overall_diversification_score = (
            availability_score * 0.4 +
            avg_success_rate * 0.4 +
            performance_score * 0.2
        )
        
        # Calculate data coverage enhancement from diversification
        data_coverage_enhancement = self._calculate_coverage_enhancement(api_availability)
        
        # Track fallback activations from circuit breakers
        fallback_activations = self._get_fallback_statistics()
        
        diversification_metrics = APIDiversificationMetrics(
            cmr_api_success_rate=api_success_rates.get("cmr", 0.0),
            giovanni_api_success_rate=api_success_rates.get("giovanni", 0.0),
            modaps_api_success_rate=api_success_rates.get("modaps", 0.0),
            atmospheric_api_success_rate=api_success_rates.get("atmospheric", 0.0),
            earthdata_auth_success_rate=api_success_rates.get("earthdata_auth", 0.0),
            overall_diversification_score=overall_diversification_score,
            api_response_times=api_response_times,
            api_availability=api_availability,
            fallback_activations=fallback_activations,
            data_coverage_enhancement=data_coverage_enhancement
        )
        
        # Create benchmark results for API diversification
        diversification_result = BenchmarkResult(
            benchmark_name="api_diversification_score",
            category=BenchmarkCategory.SUCCESS_RATES,
            measured_value=overall_diversification_score,
            target_value=0.85,  # Target 85% diversification success
            threshold_status=self._evaluate_diversification_threshold(overall_diversification_score),
            unit="score",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "available_apis": available_apis,
                "total_apis": total_apis,
                "avg_response_time": avg_response_time,
                "data_coverage_enhancement": data_coverage_enhancement
            }
        )
        
        coverage_result = BenchmarkResult(
            benchmark_name="data_coverage_enhancement",
            category=BenchmarkCategory.SUCCESS_RATES,
            measured_value=data_coverage_enhancement,
            target_value=25.0,  # Target 25% improvement from diversification
            threshold_status=self._evaluate_threshold_custom(
                data_coverage_enhancement, 25.0, 15.0, 5.0, True
            ),
            unit="percentage",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=asdict(diversification_metrics)
        )
        
        self.benchmark_results.extend([diversification_result, coverage_result])
        return diversification_metrics
    
    def calculate_success_rates(self) -> Dict[str, BenchmarkResult]:
        """Calculate various success rate benchmarks."""
        total_queries = self.success_counts["total"] + self.error_counts["total"]
        
        if total_queries == 0:
            return {}
        
        # Query Success Rate
        success_rate = self.success_counts["total"] / total_queries
        success_result = BenchmarkResult(
            benchmark_name="query_success_rate",
            category=BenchmarkCategory.SUCCESS_RATES,
            measured_value=success_rate,
            target_value=self.thresholds["query_success_rate"].target_value,
            threshold_status=self._evaluate_threshold("query_success_rate", success_rate),
            unit="percentage",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "total_queries": total_queries,
                "successful_queries": self.success_counts["total"],
                "failed_queries": self.error_counts["total"]
            }
        )
        
        # User Satisfaction (from accuracy results with ratings)
        rated_queries = [a for a in self.accuracy_results if a.user_rating is not None]
        if rated_queries:
            avg_rating = statistics.mean([a.user_rating for a in rated_queries])
            satisfaction_result = BenchmarkResult(
                benchmark_name="user_satisfaction_score",
                category=BenchmarkCategory.USER_SATISFACTION,
                measured_value=avg_rating,
                target_value=self.thresholds["user_satisfaction_score"].target_value,
                threshold_status=self._evaluate_threshold("user_satisfaction_score", avg_rating),
                unit="rating",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "rated_queries_count": len(rated_queries),
                    "rating_distribution": self._get_rating_distribution(rated_queries)
                }
            )
        else:
            satisfaction_result = None
        
        results = {"query_success_rate": success_result}
        if satisfaction_result:
            results["user_satisfaction_score"] = satisfaction_result
            
        return results
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard data."""
        
        # Recent performance trends
        recent_query_times = list(self.query_times)[-100:] if self.query_times else []
        
        # Success rates
        success_rates = self.calculate_success_rates()
        
        # System resource usage
        memory_usage = self._get_memory_usage()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Database performance trends
        db_results = [r for r in self.recent_results 
                     if r.category == BenchmarkCategory.DATABASE_PERFORMANCE]
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "avg_query_time": statistics.mean(recent_query_times) if recent_query_times else 0,
                "p95_query_time": self._percentile(recent_query_times, 95) if recent_query_times else 0,
                "queries_processed": len(self.query_times),
                "success_rate": success_rates.get("query_success_rate", {}).get("measured_value", 0),
                "system_health": self._calculate_system_health()
            },
            "performance_trends": {
                "query_times": recent_query_times[-20:],  # Last 20 queries
                "database_performance": [r.to_dict() for r in db_results[-10:]],
                "memory_usage_trend": [memory_usage],  # Would be historical
                "error_distribution": dict(self.error_counts)
            },
            "accuracy_metrics": {
                "avg_relevance_score": self._get_avg_accuracy_metric("relevance_score"),
                "avg_completeness_score": self._get_avg_accuracy_metric("completeness_score"),
                "avg_overall_accuracy": self._get_avg_accuracy_metric("overall_accuracy")
            },
            "langgraph_performance": {
                "active_workflows": len(self.graph_benchmarks),
                "avg_workflow_time": self._get_avg_workflow_time(),
                "state_memory_usage": memory_usage
            },
            "benchmark_status": self._get_benchmark_status_summary()
        }
    
    def _evaluate_threshold(self, threshold_key: str, measured_value: float) -> BenchmarkSeverity:
        """Evaluate measured value against benchmark thresholds."""
        threshold = self.thresholds[threshold_key]
        
        if threshold.higher_is_better:
            if measured_value >= threshold.target_value:
                return BenchmarkSeverity.EXCELLENT
            elif measured_value >= threshold.warning_threshold:
                return BenchmarkSeverity.GOOD
            elif measured_value >= threshold.critical_threshold:
                return BenchmarkSeverity.WARNING
            else:
                return BenchmarkSeverity.CRITICAL
        else:
            if measured_value <= threshold.target_value:
                return BenchmarkSeverity.EXCELLENT
            elif measured_value <= threshold.warning_threshold:
                return BenchmarkSeverity.GOOD
            elif measured_value <= threshold.critical_threshold:
                return BenchmarkSeverity.WARNING
            else:
                return BenchmarkSeverity.CRITICAL
    
    def _calculate_ndcg(self, recommendations: List[Dict[str, Any]], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain for recommendation relevance."""
        if not recommendations:
            return 0.0
        
        # Use relevance scores from recommendations (0-1 scale)
        relevance_scores = []
        for rec in recommendations[:k]:
            relevance = rec.get("relevance_score", 0.0)
            # Convert to graded relevance (0-3 scale for NDCG)
            if relevance >= 0.9:
                grade = 3
            elif relevance >= 0.7:
                grade = 2
            elif relevance >= 0.5:
                grade = 1
            else:
                grade = 0
            relevance_scores.append(grade)
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / (1 + i)  # Simplified DCG formula
        
        # Calculate ideal DCG (IDCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / (1 + i)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_completeness_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate metadata completeness score for recommendations."""
        if not recommendations:
            return 0.0
        
        required_fields = ["title", "summary", "temporal_coverage", "spatial_coverage", 
                          "platforms", "instruments", "data_center"]
        
        completeness_scores = []
        for rec in recommendations:
            collection = rec.get("collection", {})
            present_fields = sum(1 for field in required_fields if collection.get(field))
            completeness_scores.append(present_fields / len(required_fields))
        
        return statistics.mean(completeness_scores)
    
    def _calculate_accessibility_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate data accessibility score."""
        if not recommendations:
            return 0.0
        
        accessibility_scores = []
        for rec in recommendations:
            collection = rec.get("collection", {})
            
            # Score based on online availability, formats, etc.
            score = 0.0
            if collection.get("online"):
                score += 0.4
            if collection.get("downloadable"):
                score += 0.3
            if collection.get("cloud_hosted"):
                score += 0.3
            
            accessibility_scores.append(min(score, 1.0))
        
        return statistics.mean(accessibility_scores)
    
    def _calculate_coverage_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate temporal and spatial coverage alignment score."""
        if not recommendations:
            return 0.0
        
        # This would integrate with the actual query requirements
        # For now, return coverage scores from recommendations
        coverage_scores = [rec.get("coverage_score", 0.5) for rec in recommendations]
        return statistics.mean(coverage_scores)
    
    def _calculate_freshness_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate data freshness/recency score."""
        if not recommendations:
            return 0.0
        
        current_year = datetime.now().year
        freshness_scores = []
        
        for rec in recommendations:
            collection = rec.get("collection", {})
            
            # Score based on data recency
            temporal_coverage = collection.get("temporal_coverage", {})
            end_date = temporal_coverage.get("end_date", "1900")
            
            try:
                if isinstance(end_date, str) and len(end_date) >= 4:
                    end_year = int(end_date[:4])
                    years_old = current_year - end_year
                    
                    # More recent data gets higher scores
                    if years_old <= 1:
                        score = 1.0
                    elif years_old <= 3:
                        score = 0.8
                    elif years_old <= 5:
                        score = 0.6
                    elif years_old <= 10:
                        score = 0.4
                    else:
                        score = 0.2
                else:
                    score = 0.5  # Unknown
            except (ValueError, TypeError):
                score = 0.5
            
            freshness_scores.append(score)
        
        return statistics.mean(freshness_scores)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE or not self.process:
            return 0.0
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health status."""
        recent_results = list(self.recent_results)[-20:]  # Convert deque to list for slicing
        critical_issues = sum(1 for r in recent_results
                            if r.threshold_status == BenchmarkSeverity.CRITICAL)
        warning_issues = sum(1 for r in recent_results
                           if r.threshold_status == BenchmarkSeverity.WARNING)
        
        if critical_issues > 2:
            return "critical"
        elif critical_issues > 0 or warning_issues > 5:
            return "warning"
        else:
            return "healthy"
    
    def _get_avg_accuracy_metric(self, metric_name: str) -> float:
        """Get average value for accuracy metric."""
        if not self.accuracy_results:
            return 0.0
        
        values = [getattr(result, metric_name) for result in self.accuracy_results[-50:]]
        return statistics.mean(values) if values else 0.0
    
    def _get_avg_workflow_time(self) -> float:
        """Get average workflow completion time."""
        if not self.graph_benchmarks:
            return 0.0
        
        times = [gb.workflow_completion_time for gb in self.graph_benchmarks[-20:]]
        return statistics.mean(times) if times else 0.0
    
    def _get_rating_distribution(self, rated_queries: List[QueryAccuracyMetrics]) -> Dict[int, int]:
        """Get distribution of user ratings."""
        distribution = defaultdict(int)
        for query in rated_queries:
            if query.user_rating:
                rating_bucket = int(query.user_rating)
                distribution[rating_bucket] += 1
        return dict(distribution)
    
    def _get_benchmark_status_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark statuses."""
        if not self.recent_results:
            return {}
        
        recent_results = list(self.recent_results)[-50:]  # Convert deque to list for slicing
        status_counts = defaultdict(int)
        for result in recent_results:
            status_counts[result.threshold_status.value] += 1
        
        return {
            "total_benchmarks": len(recent_results),
            "status_distribution": dict(status_counts),
            "health_percentage": (
                status_counts["excellent"] + status_counts["good"]
            ) / max(len(recent_results), 1) * 100
        }
    
    async def _test_cmr_api(self) -> bool:
        """Test CMR API availability and basic functionality."""
        try:
            from ..agents.cmr_api_agent import CMRAPIAgent
            agent = CMRAPIAgent()
            
            # Simple test query
            results = await agent.search_collections(
                parameters=["precipitation"], 
                limit=1
            )
            return len(results) >= 0  # Even empty results indicate API is working
        except Exception as e:
            logger.warning(f"CMR API test failed: {e}")
            return False
    
    async def _test_giovanni_api(self) -> bool:
        """Test GIOVANNI API availability."""
        try:
            from ..services.nasa_giovanni import get_giovanni_service
            service = await get_giovanni_service()
            
            # Test by getting available datasets
            datasets = await service.get_available_datasets()
            return isinstance(datasets, list)
        except Exception as e:
            logger.warning(f"GIOVANNI API test failed: {e}")
            return False
    
    async def _test_modaps_api(self) -> bool:
        """Test MODAPS/LAADS API availability."""
        try:
            from ..services.modaps_laads import get_modaps_service, MODISProduct
            service = await get_modaps_service()
            
            # Test by getting product info
            info = await service.get_product_info(MODISProduct.MOD04_L2)
            return isinstance(info, dict)
        except Exception as e:
            logger.warning(f"MODAPS API test failed: {e}")
            return False
    
    async def _test_atmospheric_api(self) -> bool:
        """Test Atmospheric Data APIs availability."""
        try:
            from ..services.atmospheric_apis import get_atmospheric_service
            service = await get_atmospheric_service()
            
            # Test by getting available instruments
            instruments = await service.get_available_instruments()
            return isinstance(instruments, list) and len(instruments) > 0
        except Exception as e:
            logger.warning(f"Atmospheric API test failed: {e}")
            return False
    
    async def _test_earthdata_auth(self) -> bool:
        """Test Earthdata authentication service."""
        try:
            from ..services.earthdata_auth import get_earthdata_auth_service
            service = await get_earthdata_auth_service()
            
            # Test service initialization and status
            status = service.get_auth_status()
            return isinstance(status, dict) and "authenticated" in status
        except Exception as e:
            logger.warning(f"Earthdata auth test failed: {e}")
            return False
    
    def _calculate_coverage_enhancement(self, api_availability: Dict[str, bool]) -> float:
        """Calculate data coverage enhancement from API diversification."""
        
        # Base coverage from CMR only
        base_coverage = 100.0 if api_availability.get("cmr", False) else 0.0
        
        # Enhancement from additional APIs
        enhancement_factors = {
            "giovanni": 15.0,    # Analysis and visualization capabilities
            "modaps": 20.0,      # MODIS atmospheric and land data
            "atmospheric": 25.0,  # Specialized atmospheric instruments
            "earthdata_auth": 5.0 # Secure access to protected data
        }
        
        total_enhancement = 0.0
        for api, factor in enhancement_factors.items():
            if api_availability.get(api, False):
                total_enhancement += factor
        
        # Calculate percentage enhancement over base
        if base_coverage > 0:
            return (total_enhancement / base_coverage) * 100
        else:
            return 0.0
    
    def _get_fallback_statistics(self) -> Dict[str, int]:
        """Get circuit breaker fallback activation statistics."""
        
        # This would integrate with actual circuit breaker statistics
        # For now, return mock data structure
        return {
            "cmr_fallbacks": 0,
            "giovanni_fallbacks": 0,
            "modaps_fallbacks": 0,
            "atmospheric_fallbacks": 0,
            "auth_fallbacks": 0
        }
    
    def _evaluate_diversification_threshold(self, score: float) -> BenchmarkSeverity:
        """Evaluate diversification score against thresholds."""
        if score >= 0.90:
            return BenchmarkSeverity.EXCELLENT
        elif score >= 0.75:
            return BenchmarkSeverity.GOOD
        elif score >= 0.50:
            return BenchmarkSeverity.WARNING
        else:
            return BenchmarkSeverity.CRITICAL
    
    def _evaluate_threshold_custom(self, value: float, target: float, 
                                 warning: float, critical: float,
                                 higher_is_better: bool) -> BenchmarkSeverity:
        """Evaluate value against custom thresholds."""
        if higher_is_better:
            if value >= target:
                return BenchmarkSeverity.EXCELLENT
            elif value >= warning:
                return BenchmarkSeverity.GOOD
            elif value >= critical:
                return BenchmarkSeverity.WARNING
            else:
                return BenchmarkSeverity.CRITICAL
        else:
            if value <= target:
                return BenchmarkSeverity.EXCELLENT
            elif value <= warning:
                return BenchmarkSeverity.GOOD
            elif value <= critical:
                return BenchmarkSeverity.WARNING
            else:
                return BenchmarkSeverity.CRITICAL


# Global performance benchmark system
performance_benchmarks = PerformanceBenchmarkSystem()


def get_performance_benchmarks() -> PerformanceBenchmarkSystem:
    """Get the global performance benchmark system."""
    return performance_benchmarks