"""
Performance benchmarking and load testing for NASA CMR Agent.

Tests response times, throughput, memory usage, and scalability
under various load conditions with production-realistic scenarios.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime, timezone
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.models.schemas import QueryContext, QueryIntent, SystemResponse


@pytest.mark.performance
class TestResponseTimePerformance:
    """Test response time performance for different query types."""
    
    @pytest.fixture
    async def performance_graph(self):
        """Create optimized graph for performance testing."""
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            graph = CMRAgentGraph()
            await graph.initialize()
            
            # Configure fast mock responses
            graph.query_interpreter.interpret_query = AsyncMock(return_value=QueryContext(
                original_query="Performance test query",
                intent=QueryIntent.EXPLORATORY
            ))
            graph.query_interpreter.validate_query = AsyncMock(return_value=True)
            graph.cmr_api_agent.search_collections = AsyncMock(return_value=[])
            graph.cmr_api_agent.search_granules = AsyncMock(return_value=[])
            graph.cmr_api_agent.close = AsyncMock()
            graph.analysis_agent.analyze_results = AsyncMock(return_value=[])
            graph.analysis_agent.close = AsyncMock()
            
            graph.response_agent.synthesize_response = AsyncMock(return_value=SystemResponse(
                query_id="perf_test",
                original_query="Performance test query",
                intent=QueryIntent.EXPLORATORY,
                recommendations=[],
                summary="Performance test response",
                execution_plan=["test"],
                total_execution_time_ms=100,
                success=True
            ))
            
            graph.supervisor.validate_final_response = AsyncMock(return_value=type('obj', (object,), {
                'passed': True,
                'quality_score': type('obj', (object,), {'overall': 0.8})(),
                'requires_retry': False
            })())
            
            yield graph
            await graph.cleanup()
    
    @pytest.mark.asyncio
    async def test_simple_query_response_time(self, performance_graph):
        """Test response time for simple exploratory queries."""
        query = "Find precipitation data"
        target_time = 1.0  # 1 second target
        
        start_time = time.time()
        response = await performance_graph.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.success is True
        assert response_time < target_time, f"Response time {response_time:.2f}s exceeds target {target_time}s"
        print(f"Simple query response time: {response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_complex_query_response_time(self, performance_graph):
        """Test response time for complex analytical queries."""
        query = ("Compare precipitation datasets suitable for drought monitoring "
                "in Sub-Saharan Africa between 2015-2023, considering both satellite "
                "and ground-based observations, and identify gaps in temporal coverage")
        target_time = 5.0  # 5 second target for complex queries
        
        # Add complexity to mock responses
        performance_graph.query_interpreter.interpret_query = AsyncMock(return_value=QueryContext(
            original_query=query,
            intent=QueryIntent.COMPARATIVE,
            complexity_score=0.9
        ))
        
        start_time = time.time()
        response = await performance_graph.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.success is True
        assert response_time < target_time, f"Response time {response_time:.2f}s exceeds target {target_time}s"
        print(f"Complex query response time: {response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_query_response_time_distribution(self, performance_graph):
        """Test response time distribution across multiple queries."""
        queries = [
            "Find temperature data",
            "Search precipitation datasets for Africa",
            "Ocean salinity measurements",
            "Atmospheric CO2 data",
            "Land surface temperature imagery"
        ]
        
        response_times = []
        
        for query in queries:
            start_time = time.time()
            response = await performance_graph.process_query(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response.success is True
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Response time statistics:")
        print(f"  Mean: {mean_time:.3f}s")
        print(f"  Median: {median_time:.3f}s") 
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        
        # Performance assertions
        assert mean_time < 2.0, f"Average response time {mean_time:.2f}s too high"
        assert max_time < 3.0, f"Maximum response time {max_time:.2f}s too high"
        assert all(t > 0 for t in response_times), "All queries should have positive response times"


@pytest.mark.performance
class TestThroughputPerformance:
    """Test system throughput and concurrent request handling."""
    
    @pytest.fixture
    async def throughput_graph(self):
        """Create graph optimized for throughput testing."""
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            graph = CMRAgentGraph()
            await graph.initialize()
            
            # Configure lightweight mock responses for throughput
            async def fast_interpret(query):
                await asyncio.sleep(0.01)  # Minimal processing time
                return QueryContext(original_query=query, intent=QueryIntent.EXPLORATORY)
            
            async def fast_validate(context):
                await asyncio.sleep(0.005)
                return True
            
            async def fast_search_collections(context):
                await asyncio.sleep(0.02)
                return []
            
            async def fast_search_granules(context, collection_id):
                await asyncio.sleep(0.01)
                return []
            
            async def fast_analyze(context, results, sub_queries=None):
                await asyncio.sleep(0.015)
                return []
            
            async def fast_synthesize(context, cmr_results, analysis_results, errors):
                await asyncio.sleep(0.01)
                return SystemResponse(
                    query_id="throughput_test",
                    original_query=context.original_query if context else "test",
                    intent=QueryIntent.EXPLORATORY,
                    recommendations=[],
                    summary="Throughput test response",
                    execution_plan=["test"],
                    total_execution_time_ms=50,
                    success=True
                )
            
            async def fast_validate_response(query, response):
                await asyncio.sleep(0.005)
                return type('obj', (object,), {
                    'passed': True,
                    'quality_score': type('obj', (object,), {'overall': 0.8})(),
                    'requires_retry': False
                })()
            
            graph.query_interpreter.interpret_query = fast_interpret
            graph.query_interpreter.validate_query = fast_validate
            graph.cmr_api_agent.search_collections = fast_search_collections
            graph.cmr_api_agent.search_granules = fast_search_granules
            graph.cmr_api_agent.close = AsyncMock()
            graph.analysis_agent.analyze_results = fast_analyze
            graph.analysis_agent.close = AsyncMock()
            graph.response_agent.synthesize_response = fast_synthesize
            graph.supervisor.validate_final_response = fast_validate_response
            
            yield graph
            await graph.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_request_throughput(self, throughput_graph):
        """Test throughput with concurrent requests."""
        concurrent_requests = 10
        query = "Throughput test query"
        
        start_time = time.time()
        
        # Execute concurrent requests
        tasks = [throughput_graph.process_query(f"{query} {i}") 
                for i in range(concurrent_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all requests succeeded
        successful_responses = [r for r in responses if isinstance(r, SystemResponse) and r.success]
        assert len(successful_responses) == concurrent_requests, "All concurrent requests should succeed"
        
        # Calculate throughput
        throughput = concurrent_requests / total_time
        print(f"Concurrent throughput: {throughput:.2f} requests/second")
        print(f"Total time for {concurrent_requests} requests: {total_time:.3f}s")
        
        # Performance assertion
        min_throughput = 5.0  # Minimum 5 requests per second
        assert throughput >= min_throughput, f"Throughput {throughput:.2f} req/s below minimum {min_throughput}"
    
    @pytest.mark.asyncio
    async def test_sustained_load_throughput(self, throughput_graph):
        """Test sustained load over time."""
        duration_seconds = 10
        target_rps = 2  # 2 requests per second
        
        start_time = time.time()
        completed_requests = 0
        
        async def sustained_requester():
            nonlocal completed_requests
            while time.time() - start_time < duration_seconds:
                try:
                    response = await throughput_graph.process_query(f"Sustained test {completed_requests}")
                    if response.success:
                        completed_requests += 1
                    await asyncio.sleep(1.0 / target_rps)  # Rate limiting
                except Exception as e:
                    print(f"Request failed: {e}")
        
        # Run sustained load test
        await sustained_requester()
        
        actual_duration = time.time() - start_time
        actual_rps = completed_requests / actual_duration
        
        print(f"Sustained load results:")
        print(f"  Duration: {actual_duration:.2f}s")
        print(f"  Completed requests: {completed_requests}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        
        # Should maintain at least 80% of target throughput
        min_acceptable_rps = target_rps * 0.8
        assert actual_rps >= min_acceptable_rps, f"Sustained RPS {actual_rps:.2f} below minimum {min_acceptable_rps}"
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self, throughput_graph):
        """Test handling sudden burst of requests."""
        burst_size = 20
        queries = [f"Burst test query {i}" for i in range(burst_size)]
        
        # Send all requests at once
        start_time = time.time()
        tasks = [throughput_graph.process_query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_responses = [r for r in responses if isinstance(r, SystemResponse) and r.success]
        
        print(f"Burst load results:")
        print(f"  Burst size: {burst_size}")
        print(f"  Successful responses: {len(successful_responses)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Success rate: {len(successful_responses)/burst_size*100:.1f}%")
        
        # Should handle at least 80% of burst requests successfully
        min_success_rate = 0.8
        success_rate = len(successful_responses) / burst_size
        assert success_rate >= min_success_rate, f"Success rate {success_rate:.2f} below minimum {min_success_rate}"


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and leak detection."""
    
    @pytest.fixture
    async def memory_graph(self):
        """Create graph for memory testing."""
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            graph = CMRAgentGraph()
            await graph.initialize()
            
            # Configure mocks
            graph.query_interpreter.interpret_query = AsyncMock(return_value=QueryContext(
                original_query="Memory test",
                intent=QueryIntent.EXPLORATORY
            ))
            graph.query_interpreter.validate_query = AsyncMock(return_value=True)
            graph.cmr_api_agent.search_collections = AsyncMock(return_value=[])
            graph.cmr_api_agent.search_granules = AsyncMock(return_value=[])
            graph.cmr_api_agent.close = AsyncMock()
            graph.analysis_agent.analyze_results = AsyncMock(return_value=[])
            graph.analysis_agent.close = AsyncMock()
            graph.response_agent.synthesize_response = AsyncMock(return_value=SystemResponse(
                query_id="memory_test",
                original_query="Memory test",
                intent=QueryIntent.EXPLORATORY,
                recommendations=[],
                summary="Memory test response",
                execution_plan=["test"],
                total_execution_time_ms=100,
                success=True
            ))
            graph.supervisor.validate_final_response = AsyncMock(return_value=type('obj', (object,), {
                'passed': True,
                'quality_score': type('obj', (object,), {'overall': 0.8})(),
                'requires_retry': False
            })())
            
            yield graph
            await graph.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_usage_baseline(self, memory_graph):
        """Test baseline memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a single query
        await memory_graph.process_query("Memory baseline test")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage baseline:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB") 
        print(f"  Increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable for a single query
        max_acceptable_increase = 50  # MB
        assert memory_increase < max_acceptable_increase, f"Memory increase {memory_increase:.2f} MB too high"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, memory_graph):
        """Test for memory leaks over multiple requests."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple queries
        num_queries = 50
        memory_samples = []
        
        for i in range(num_queries):
            await memory_graph.process_query(f"Memory leak test {i}")
            
            if i % 10 == 0:  # Sample every 10 queries
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"Memory leak test results:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Total increase: {total_increase:.2f} MB")
        print(f"  Queries processed: {num_queries}")
        print(f"  Memory per query: {total_increase/num_queries:.3f} MB")
        
        # Check for linear memory growth (potential leak)
        if len(memory_samples) > 2:
            # Simple linear regression to detect trend
            x = list(range(len(memory_samples)))
            y = memory_samples
            
            n = len(x)
            slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
            
            print(f"  Memory growth slope: {slope:.3f} MB per sample")
            
            # Slope should be minimal for no leaks
            max_acceptable_slope = 2.0  # MB per sample
            assert abs(slope) < max_acceptable_slope, f"Potential memory leak detected: slope {slope:.3f} MB/sample"
        
        # Overall memory increase should be bounded
        max_total_increase = 100  # MB for 50 queries
        assert total_increase < max_total_increase, f"Total memory increase {total_increase:.2f} MB too high"
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_tracemalloc(self, memory_graph):
        """Test detailed memory usage with tracemalloc."""
        tracemalloc.start()
        
        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Process queries
        for i in range(10):
            await memory_graph.process_query(f"Tracemalloc test {i}")
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        total_size_diff = sum(stat.size_diff for stat in top_stats)
        
        print(f"Memory allocation analysis:")
        print(f"  Total size difference: {total_size_diff / 1024 / 1024:.2f} MB")
        print(f"  Top memory allocations:")
        
        for index, stat in enumerate(top_stats[:5]):
            print(f"    {index + 1}. {stat}")
        
        tracemalloc.stop()
        
        # Memory allocations should be reasonable
        max_acceptable_allocation = 20 * 1024 * 1024  # 20 MB
        assert total_size_diff < max_acceptable_allocation, f"Memory allocation {total_size_diff/1024/1024:.2f} MB too high"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityPerformance:
    """Test system scalability under high load."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_scalability(self):
        """Test performance with high concurrency levels."""
        concurrency_levels = [1, 5, 10, 20, 50]
        results = {}
        
        for concurrency in concurrency_levels:
            with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
                 patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
                 patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
                 patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
                 patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
                
                graph = CMRAgentGraph()
                await graph.initialize()
                
                # Configure fast mocks for scalability testing
                graph.query_interpreter.interpret_query = AsyncMock(return_value=QueryContext(
                    original_query="Scalability test", intent=QueryIntent.EXPLORATORY
                ))
                graph.query_interpreter.validate_query = AsyncMock(return_value=True)
                graph.cmr_api_agent.search_collections = AsyncMock(return_value=[])
                graph.cmr_api_agent.search_granules = AsyncMock(return_value=[])
                graph.cmr_api_agent.close = AsyncMock()
                graph.analysis_agent.analyze_results = AsyncMock(return_value=[])
                graph.analysis_agent.close = AsyncMock()
                graph.response_agent.synthesize_response = AsyncMock(return_value=SystemResponse(
                    query_id="scalability_test",
                    original_query="Scalability test",
                    intent=QueryIntent.EXPLORATORY,
                    recommendations=[],
                    summary="Scalability test response",
                    execution_plan=["test"],
                    total_execution_time_ms=100,
                    success=True
                ))
                graph.supervisor.validate_final_response = AsyncMock(return_value=type('obj', (object,), {
                    'passed': True,
                    'quality_score': type('obj', (object,), {'overall': 0.8})(),
                    'requires_retry': False
                })())
                
                # Run concurrent requests
                start_time = time.time()
                tasks = [graph.process_query(f"Scalability test {i}") for i in range(concurrency)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                successful_responses = [r for r in responses if isinstance(r, SystemResponse) and r.success]
                success_rate = len(successful_responses) / concurrency
                throughput = len(successful_responses) / total_time
                
                results[concurrency] = {
                    'total_time': total_time,
                    'success_rate': success_rate,
                    'throughput': throughput
                }
                
                await graph.cleanup()
        
        # Analyze scalability results
        print("Scalability test results:")
        print("Concurrency | Time(s) | Success Rate | Throughput(req/s)")
        print("------------|---------|--------------|------------------")
        
        for concurrency in concurrency_levels:
            result = results[concurrency]
            print(f"{concurrency:10d} | {result['total_time']:7.2f} | {result['success_rate']:11.1%} | {result['throughput']:16.2f}")
        
        # Verify scalability characteristics
        for concurrency in concurrency_levels:
            result = results[concurrency]
            
            # Success rate should remain high
            min_success_rate = 0.9
            assert result['success_rate'] >= min_success_rate, f"Success rate dropped to {result['success_rate']:.1%} at concurrency {concurrency}"
            
            # Throughput should scale reasonably (at least linear with some overhead)
            if concurrency > 1:
                expected_min_throughput = concurrency * 0.5  # 50% efficiency
                assert result['throughput'] >= expected_min_throughput, f"Throughput {result['throughput']:.2f} too low for concurrency {concurrency}"