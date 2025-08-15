"""
Performance Benchmarking System Tests.

Tests the comprehensive performance benchmarking system including accuracy measurements,
LangGraph state monitoring, success rate tracking, and visualization integration.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Test comprehensive performance benchmarking system."""
    
    async def test_benchmark_system_initialization(self):
        """Test benchmark system initialization and thresholds."""
        logger.info("🎯 Testing benchmark system initialization")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Test threshold initialization
        assert len(benchmark_system.thresholds) > 0, "Should have initialized benchmark thresholds"
        
        # Test specific benchmark thresholds
        assert "end_to_end_query_time" in benchmark_system.thresholds
        assert "vector_search_latency" in benchmark_system.thresholds
        assert "state_transition_latency" in benchmark_system.thresholds
        assert "recommendation_relevance" in benchmark_system.thresholds
        
        # Test threshold properties
        query_time_threshold = benchmark_system.thresholds["end_to_end_query_time"]
        assert query_time_threshold.target_value == 5.0
        assert query_time_threshold.warning_threshold == 8.0
        assert query_time_threshold.critical_threshold == 15.0
        assert not query_time_threshold.higher_is_better  # Lower times are better
        
        relevance_threshold = benchmark_system.thresholds["recommendation_relevance"]
        assert relevance_threshold.target_value == 0.85
        assert relevance_threshold.higher_is_better  # Higher relevance is better
        
        logger.info("✅ Benchmark system initialization tests passed")
    
    async def test_query_performance_measurement(self):
        """Test query performance measurement and benchmarking."""
        logger.info("🎯 Testing query performance measurement")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Simulate query performance measurement
        query_id = "test_query_001"
        query_text = "precipitation data over North America"
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(0.1)  # 100ms processing time
        
        end_time = time.time()
        agent_timings = {
            "query_interpreter": 0.02,
            "cmr_api_agent": 0.05,
            "analysis_agent": 0.02,
            "response_agent": 0.01
        }
        
        # Measure performance
        result = await benchmark_system.measure_query_performance(
            query_id=query_id,
            query_text=query_text,
            start_time=start_time,
            end_time=end_time,
            agent_timings=agent_timings,
            success=True
        )
        
        # Validate result
        assert result.benchmark_name == "end_to_end_query_time"
        assert result.measured_value > 0
        assert result.measured_value < 1.0  # Should be under 1 second for this test
        assert result.metadata["query_id"] == query_id
        assert result.metadata["success"] is True
        assert result.threshold_status.value in ["excellent", "good", "warning", "critical"]
        
        # Check that result was recorded
        assert len(benchmark_system.benchmark_results) == 1
        assert len(benchmark_system.query_times) == 1
        
        logger.info(f"Query performance: {result.measured_value:.3f}s ({result.threshold_status.value})")
        logger.info("✅ Query performance measurement tests passed")
    
    async def test_database_performance_measurement(self):
        """Test database performance benchmarking."""
        logger.info("🎯 Testing database performance measurement")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Test Weaviate vector search performance
        weaviate_result = await benchmark_system.measure_database_performance(
            operation="vector_similarity_search",
            duration=0.08,  # 80ms
            result_count=20,
            database="Weaviate"
        )
        
        assert weaviate_result.benchmark_name == "weaviate_query_time"
        assert weaviate_result.measured_value == 0.08
        assert weaviate_result.metadata["database"] == "Weaviate"
        assert weaviate_result.metadata["result_count"] == 20
        
        # Test Neo4j knowledge graph performance
        neo4j_result = await benchmark_system.measure_database_performance(
            operation="relationship_traversal",
            duration=0.25,  # 250ms
            result_count=15,
            database="Neo4j"
        )
        
        assert neo4j_result.benchmark_name == "neo4j_query_time"
        assert neo4j_result.measured_value == 0.25
        assert neo4j_result.metadata["database"] == "Neo4j"
        
        logger.info("✅ Database performance measurement tests passed")
    
    async def test_langgraph_state_performance(self):
        """Test LangGraph state performance monitoring."""
        logger.info("🎯 Testing LangGraph state performance")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Simulate LangGraph workflow performance
        workflow_id = "workflow_test_001"
        agent_timings = {
            "query_interpreter": 0.5,
            "cmr_api_agent": 1.2,
            "analysis_agent": 0.8,
            "response_agent": 0.3
        }
        state_transitions = 4  # Number of agent transitions
        memory_usage = 32.5  # MB
        state_size = 2048  # bytes
        
        graph_benchmark = await benchmark_system.measure_langgraph_state_performance(
            workflow_id=workflow_id,
            agent_timings=agent_timings,
            state_transitions=state_transitions,
            memory_usage=memory_usage,
            state_size=state_size
        )
        
        # Validate graph benchmark
        assert graph_benchmark.workflow_id == workflow_id
        assert graph_benchmark.graph_traversal_time == agent_timings
        assert graph_benchmark.state_memory_usage == memory_usage
        assert graph_benchmark.state_serialization_size == state_size
        assert graph_benchmark.workflow_completion_time == sum(agent_timings.values())
        
        # Check benchmark results were created
        state_results = [r for r in benchmark_system.benchmark_results 
                        if r.category.value == "langgraph_state"]
        assert len(state_results) >= 2  # Should have transition and memory results
        
        logger.info(f"Workflow completion time: {graph_benchmark.workflow_completion_time:.2f}s")
        logger.info("✅ LangGraph state performance tests passed")
    
    async def test_query_accuracy_measurement(self):
        """Test query accuracy measurement using NDCG and quality metrics."""
        logger.info("🎯 Testing query accuracy measurement")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Mock recommendation data
        recommendations = [
            {
                "collection": {
                    "concept_id": "C123456789-PODAAC",
                    "title": "GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V06",
                    "summary": "Global Precipitation Measurement mission data",
                    "temporal_coverage": {"end_date": "2024-01-01"},
                    "spatial_coverage": {"bounding_rectangle": {}},
                    "platforms": ["GPM"],
                    "instruments": ["GMI", "DPR"],
                    "data_center": "PODAAC",
                    "online": True,
                    "downloadable": True,
                    "cloud_hosted": True
                },
                "relevance_score": 0.92,
                "coverage_score": 0.88,
                "quality_score": 0.95
            },
            {
                "collection": {
                    "concept_id": "C987654321-GES_DISC",
                    "title": "TRMM 3B42 Daily Precipitation Data",
                    "summary": "Tropical Rainfall Measuring Mission precipitation",
                    "temporal_coverage": {"end_date": "2015-04-01"},
                    "spatial_coverage": {"bounding_rectangle": {}},
                    "platforms": ["TRMM"],
                    "instruments": ["PR", "TMI"],
                    "data_center": "GES_DISC",
                    "online": True,
                    "downloadable": True
                },
                "relevance_score": 0.78,
                "coverage_score": 0.65,
                "quality_score": 0.82
            }
        ]
        
        user_feedback = {"rating": 4.2, "helpful": True}
        
        # Measure query accuracy
        accuracy_metrics = await benchmark_system.measure_query_accuracy(
            query_id="test_query_accuracy_001",
            recommendations=recommendations,
            user_feedback=user_feedback,
            expert_validation=True
        )
        
        # Validate accuracy metrics
        assert accuracy_metrics.query_id == "test_query_accuracy_001"
        assert 0 <= accuracy_metrics.relevance_score <= 1
        assert 0 <= accuracy_metrics.completeness_score <= 1
        assert 0 <= accuracy_metrics.accessibility_score <= 1
        assert 0 <= accuracy_metrics.coverage_score <= 1
        assert 0 <= accuracy_metrics.freshness_score <= 1
        assert 0 <= accuracy_metrics.overall_accuracy <= 1
        assert accuracy_metrics.user_rating == 4.2
        assert accuracy_metrics.expert_validation is True
        
        # Check that accuracy benchmark result was created
        accuracy_results = [r for r in benchmark_system.benchmark_results 
                           if r.benchmark_name == "recommendation_relevance"]
        assert len(accuracy_results) == 1
        
        logger.info(f"Overall accuracy: {accuracy_metrics.overall_accuracy:.3f}")
        logger.info(f"NDCG relevance: {accuracy_metrics.relevance_score:.3f}")
        logger.info("✅ Query accuracy measurement tests passed")
    
    async def test_success_rate_calculation(self):
        """Test success rate benchmarking."""
        logger.info("🎯 Testing success rate calculation")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Simulate multiple query results
        for i in range(10):
            success = i < 8  # 80% success rate
            
            await benchmark_system.measure_query_performance(
                query_id=f"success_test_{i}",
                query_text=f"test query {i}",
                start_time=time.time() - 1.0,
                end_time=time.time(),
                agent_timings={"test_agent": 0.5},
                success=success,
                error="test error" if not success else None
            )
        
        # Calculate success rates
        success_rates = benchmark_system.calculate_success_rates()
        
        # Validate success rate calculation
        assert "query_success_rate" in success_rates
        success_result = success_rates["query_success_rate"]
        
        assert success_result.measured_value == 0.8  # 80% success rate
        assert success_result.metadata["total_queries"] == 10
        assert success_result.metadata["successful_queries"] == 8
        assert success_result.metadata["failed_queries"] == 2
        
        logger.info(f"Success rate: {success_result.measured_value*100:.1f}%")
        logger.info("✅ Success rate calculation tests passed")
    
    async def test_performance_dashboard_generation(self):
        """Test performance dashboard data generation."""
        logger.info("🎯 Testing performance dashboard generation")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Add some test data
        for i in range(5):
            await benchmark_system.measure_query_performance(
                query_id=f"dashboard_test_{i}",
                query_text="test query for dashboard",
                start_time=time.time() - 1.0,
                end_time=time.time(),
                agent_timings={"test_agent": 0.5},
                success=True
            )
        
        # Generate dashboard data
        dashboard_data = benchmark_system.get_performance_dashboard_data()
        
        # Validate dashboard structure
        assert "timestamp" in dashboard_data
        assert "summary" in dashboard_data
        assert "performance_trends" in dashboard_data
        assert "accuracy_metrics" in dashboard_data
        assert "langgraph_performance" in dashboard_data
        assert "benchmark_status" in dashboard_data
        
        # Validate summary data
        summary = dashboard_data["summary"]
        assert "avg_query_time" in summary
        assert "queries_processed" in summary
        assert "system_health" in summary
        
        assert summary["queries_processed"] == 5
        assert summary["system_health"] in ["healthy", "warning", "critical"]
        
        logger.info(f"Dashboard generated with {summary['queries_processed']} queries")
        logger.info("✅ Performance dashboard generation tests passed")
    
    async def test_benchmark_visualization_system(self):
        """Test benchmark visualization system."""
        logger.info("🎯 Testing benchmark visualization system")
        
        from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
        from nasa_cmr_agent.monitoring.benchmark_visualizations import BenchmarkVisualizationSystem
        
        benchmark_system = PerformanceBenchmarkSystem()
        
        # Add test data
        for i in range(3):
            await benchmark_system.measure_query_performance(
                query_id=f"viz_test_{i}",
                query_text="visualization test query",
                start_time=time.time() - 2.0,
                end_time=time.time() - 1.5,
                agent_timings={"test_agent": 0.5},
                success=True
            )
        
        # Create visualization system
        viz_system = BenchmarkVisualizationSystem(benchmark_system)
        
        # Test dashboard creation
        dashboard = viz_system.create_performance_dashboard()
        
        assert "type" in dashboard
        assert "timestamp" in dashboard
        assert "summary" in dashboard
        assert "charts" in dashboard
        
        # Test report generation
        report = viz_system.generate_benchmark_report("24h")
        
        assert "report_id" in report
        assert "summary" in report
        assert "performance_analysis" in report
        assert "benchmark_compliance" in report
        assert "recommendations" in report
        
        logger.info(f"Generated dashboard with {len(dashboard.get('charts', {}))} charts")
        logger.info("✅ Benchmark visualization system tests passed")


@pytest.mark.asyncio
async def test_integrated_benchmark_workflow():
    """Test complete integrated benchmark workflow."""
    logger.info("🎯 Testing integrated benchmark workflow")
    
    from nasa_cmr_agent.monitoring.performance_benchmarks import PerformanceBenchmarkSystem
    
    benchmark_system = PerformanceBenchmarkSystem()
    
    # Simulate complete query workflow with benchmarking
    query_id = "integrated_test_001"
    start_time = time.time()
    
    # Step 1: Query performance
    await asyncio.sleep(0.05)  # Simulate processing
    end_time = time.time()
    
    query_result = await benchmark_system.measure_query_performance(
        query_id=query_id,
        query_text="integrated test query for precipitation data",
        start_time=start_time,
        end_time=end_time,
        agent_timings={
            "query_interpreter": 0.01,
            "cmr_api_agent": 0.02,
            "analysis_agent": 0.015,
            "response_agent": 0.005
        },
        success=True
    )
    
    # Step 2: Database performance
    db_result = await benchmark_system.measure_database_performance(
        operation="similarity_search",
        duration=0.12,
        result_count=25,
        database="Weaviate"
    )
    
    # Step 3: LangGraph state performance
    graph_result = await benchmark_system.measure_langgraph_state_performance(
        workflow_id=f"workflow_{query_id}",
        agent_timings={
            "query_interpreter": 0.01,
            "cmr_api_agent": 0.02,
            "analysis_agent": 0.015,
            "response_agent": 0.005
        },
        state_transitions=4,
        memory_usage=28.5,
        state_size=1024
    )
    
    # Step 4: Query accuracy
    mock_recommendations = [{
        "collection": {
            "title": "Test Dataset",
            "summary": "Test precipitation dataset",
            "temporal_coverage": {"end_date": "2024-01-01"},
            "platforms": ["Test Platform"],
            "instruments": ["Test Instrument"],
            "data_center": "Test Center",
            "online": True,
            "downloadable": True
        },
        "relevance_score": 0.88,
        "coverage_score": 0.82
    }]
    
    accuracy_result = await benchmark_system.measure_query_accuracy(
        query_id=query_id,
        recommendations=mock_recommendations,
        user_feedback={"rating": 4.5, "helpful": True}
    )
    
    # Validate integrated results
    assert len(benchmark_system.benchmark_results) >= 4  # Multiple benchmark types
    assert len(benchmark_system.accuracy_results) == 1
    assert len(benchmark_system.graph_benchmarks) == 1
    assert len(benchmark_system.query_times) >= 1
    
    # Generate final dashboard
    dashboard_data = benchmark_system.get_performance_dashboard_data()
    assert dashboard_data["summary"]["queries_processed"] >= 1
    
    logger.info("✅ Integrated benchmark workflow tests passed")


if __name__ == "__main__":
    # Allow running this test directly
    async def main():
        test_class = TestPerformanceBenchmarks()
        
        try:
            logger.info("🧪 Running comprehensive performance benchmark tests...")
            
            await test_class.test_benchmark_system_initialization()
            await test_class.test_query_performance_measurement()
            await test_class.test_database_performance_measurement()
            await test_class.test_langgraph_state_performance()
            await test_class.test_query_accuracy_measurement()
            await test_class.test_success_rate_calculation()
            await test_class.test_performance_dashboard_generation()
            await test_class.test_benchmark_visualization_system()
            
            await test_integrated_benchmark_workflow()
            
            logger.info("🎉 All performance benchmark tests PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)