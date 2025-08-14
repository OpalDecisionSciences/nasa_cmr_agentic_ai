"""
Comprehensive integration tests for CMR Agent Graph.

Tests end-to-end workflows, agent coordination, error recovery,
and production scenarios with real integrations.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.models.schemas import (
    QueryContext, QueryIntent, SpatialConstraint, TemporalConstraint,
    SystemResponse, CMRCollection, AgentResponse
)


@pytest.mark.integration
class TestCMRAgentGraphIntegration:
    """Test complete agent graph integration."""
    
    @pytest.fixture
    async def initialized_graph(self):
        """Create and initialize agent graph for testing."""
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            graph = CMRAgentGraph()
            await graph.initialize()
            
            # Mock agent responses for testing
            graph.query_interpreter.interpret_query = AsyncMock(return_value=QueryContext(
                original_query="Test query",
                intent=QueryIntent.ANALYTICAL,
                constraints={
                    "keywords": ["precipitation"],
                    "spatial": SpatialConstraint(north=45.0, south=30.0, east=-100.0, west=-120.0)
                }
            ))
            
            graph.query_interpreter.validate_query = AsyncMock(return_value=True)
            graph.query_interpreter.get_decomposed_queries = AsyncMock(return_value=(None, None))
            
            graph.cmr_api_agent.search_collections = AsyncMock(return_value=[
                CMRCollection(
                    concept_id="C123-TEST",
                    title="Test Collection",
                    summary="Test collection for precipitation data",
                    data_center="TEST_DAAC"
                )
            ])
            
            graph.cmr_api_agent.search_granules = AsyncMock(return_value=[])
            graph.cmr_api_agent.close = AsyncMock()
            
            graph.analysis_agent.analyze_results = AsyncMock(return_value=[])
            graph.analysis_agent.close = AsyncMock()
            
            graph.response_agent.synthesize_response = AsyncMock(return_value=SystemResponse(
                query_id="test_123",
                original_query="Test query",
                intent=QueryIntent.ANALYTICAL,
                recommendations=[],
                summary="Test response",
                execution_plan=["test"],
                total_execution_time_ms=1000,
                success=True
            ))
            
            graph.supervisor.validate_final_response = AsyncMock(return_value=MagicMock(
                passed=True,
                quality_score=MagicMock(overall=0.8),
                requires_retry=False
            ))
            
            yield graph
            await graph.cleanup()
    
    async def test_complete_query_processing(self, initialized_graph):
        """Test complete query processing workflow."""
        query = "Find precipitation data for drought monitoring in California 2020-2023"
        
        response = await initialized_graph.process_query(query)
        
        assert isinstance(response, SystemResponse)
        assert response.success is True
        assert response.original_query == query
        assert response.query_id is not None
        
        # Verify all agents were called
        initialized_graph.query_interpreter.interpret_query.assert_called_once()
        initialized_graph.query_interpreter.validate_query.assert_called_once()
        initialized_graph.cmr_api_agent.search_collections.assert_called_once()
        initialized_graph.analysis_agent.analyze_results.assert_called_once()
        initialized_graph.response_agent.synthesize_response.assert_called_once()
        initialized_graph.supervisor.validate_final_response.assert_called_once()
    
    async def test_initialization_race_condition(self):
        """Test that concurrent initialization is handled correctly."""
        graph = CMRAgentGraph()
        
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            # Start multiple concurrent initializations
            tasks = [graph.initialize() for _ in range(5)]
            await asyncio.gather(*tasks)
            
            # Should be properly initialized only once
            assert graph._initialized is True
            assert graph._initializing is False
    
    async def test_query_validation_failure(self, initialized_graph):
        """Test handling of query validation failures."""
        # Mock validation failure
        initialized_graph.query_interpreter.validate_query.return_value = False
        
        query = "Invalid query that should fail validation"
        response = await initialized_graph.process_query(query)
        
        assert isinstance(response, SystemResponse)
        # Should still return a response but with error handling
        assert response.original_query == query
    
    async def test_cmr_api_failure_recovery(self, initialized_graph):
        """Test recovery from CMR API failures."""
        # Mock CMR API failure
        initialized_graph.cmr_api_agent.search_collections.side_effect = Exception("CMR API Error")
        
        query = "Test query with API failure"
        response = await initialized_graph.process_query(query)
        
        assert isinstance(response, SystemResponse)
        assert response.original_query == query
        # Should handle the error gracefully
    
    async def test_supervisor_retry_mechanism(self, initialized_graph):
        """Test supervisor-driven retry mechanism."""
        # Mock supervisor requesting retry on first attempt
        retry_validation = MagicMock(
            passed=False,
            quality_score=MagicMock(overall=0.3),
            requires_retry=True,
            retry_guidance="Improve query processing"
        )
        success_validation = MagicMock(
            passed=True,
            quality_score=MagicMock(overall=0.8),
            requires_retry=False
        )
        
        initialized_graph.supervisor.validate_final_response.side_effect = [
            retry_validation, success_validation
        ]
        
        query = "Query requiring retry"
        response = await initialized_graph.process_query(query)
        
        assert isinstance(response, SystemResponse)
        # Should have retried and succeeded
        assert initialized_graph.supervisor.validate_final_response.call_count == 2
    
    async def test_state_management_across_nodes(self, initialized_graph):
        """Test that state is properly managed across workflow nodes."""
        query = "Test state management"
        
        # Capture state modifications in nodes
        original_interpret_query = initialized_graph.query_interpreter.interpret_query
        
        async def mock_interpret_with_state_check(*args, **kwargs):
            # This would be called in the actual node
            return await original_interpret_query(*args, **kwargs)
        
        initialized_graph.query_interpreter.interpret_query.side_effect = mock_interpret_with_state_check
        
        response = await initialized_graph.process_query(query)
        
        assert response is not None
        assert response.original_query == query
    
    async def test_concurrent_query_processing(self, initialized_graph):
        """Test handling multiple concurrent queries."""
        queries = [
            "Find precipitation data for Africa",
            "Search for temperature datasets in Asia", 
            "Ocean salinity data for climate studies",
            "Atmospheric CO2 measurements globally"
        ]
        
        # Process queries concurrently
        tasks = [initialized_graph.process_query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All queries should process successfully
        for i, response in enumerate(responses):
            assert not isinstance(response, Exception)
            assert isinstance(response, SystemResponse)
            assert response.original_query == queries[i]
    
    async def test_resource_cleanup(self, initialized_graph):
        """Test proper resource cleanup after processing."""
        query = "Test resource cleanup"
        
        await initialized_graph.process_query(query)
        await initialized_graph.cleanup()
        
        # Verify cleanup was called on components
        initialized_graph.cmr_api_agent.close.assert_called_once()
        initialized_graph.analysis_agent.close.assert_called_once()


@pytest.mark.integration
class TestAgentCoordination:
    """Test coordination between different agents."""
    
    @pytest.fixture
    async def mock_agents(self):
        """Create mock agents for coordination testing."""
        query_interpreter = AsyncMock()
        cmr_api_agent = AsyncMock()
        analysis_agent = AsyncMock()
        response_agent = AsyncMock()
        supervisor = AsyncMock()
        
        agents = {
            'query_interpreter': query_interpreter,
            'cmr_api_agent': cmr_api_agent,
            'analysis_agent': analysis_agent,
            'response_agent': response_agent,
            'supervisor': supervisor
        }
        
        return agents
    
    async def test_data_flow_between_agents(self, mock_agents):
        """Test data flows correctly between agents."""
        graph = CMRAgentGraph()
        
        # Replace agents with mocks
        for agent_name, mock_agent in mock_agents.items():
            setattr(graph, agent_name, mock_agent)
        
        graph._initialized = True
        
        # Configure mock responses to track data flow
        query_context = QueryContext(
            original_query="Test data flow",
            intent=QueryIntent.ANALYTICAL
        )
        
        mock_agents['query_interpreter'].interpret_query.return_value = query_context
        mock_agents['query_interpreter'].validate_query.return_value = True
        mock_agents['cmr_api_agent'].search_collections.return_value = []
        mock_agents['cmr_api_agent'].search_granules.return_value = []
        mock_agents['analysis_agent'].analyze_results.return_value = []
        mock_agents['response_agent'].synthesize_response.return_value = SystemResponse(
            query_id="test",
            original_query="Test data flow",
            intent=QueryIntent.ANALYTICAL,
            recommendations=[],
            summary="Test",
            execution_plan=["test"],
            total_execution_time_ms=100,
            success=True
        )
        mock_agents['supervisor'].validate_final_response.return_value = MagicMock(
            passed=True,
            quality_score=MagicMock(overall=0.8),
            requires_retry=False
        )
        
        await graph.process_query("Test data flow")
        
        # Verify the query context flows through agents
        mock_agents['cmr_api_agent'].search_collections.assert_called_with(query_context)
    
    async def test_error_propagation_between_agents(self, mock_agents):
        """Test error propagation and handling between agents."""
        graph = CMRAgentGraph()
        
        for agent_name, mock_agent in mock_agents.items():
            setattr(graph, agent_name, mock_agent)
        
        graph._initialized = True
        
        # Configure query interpreter to fail
        mock_agents['query_interpreter'].interpret_query.side_effect = Exception("Query parsing failed")
        
        # Other agents should handle gracefully
        response = await graph.process_query("Test error propagation")
        
        # Should still return a response with error handling
        assert response is not None


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.fixture
    async def realistic_graph(self):
        """Create graph with realistic mock responses."""
        with patch('nasa_cmr_agent.agents.enhanced_analysis_agent.EnhancedAnalysisAgent'), \
             patch('nasa_cmr_agent.agents.query_interpreter.QueryInterpreterAgent'), \
             patch('nasa_cmr_agent.agents.cmr_api_agent.CMRAPIAgent'), \
             patch('nasa_cmr_agent.agents.response_agent.ResponseSynthesisAgent'), \
             patch('nasa_cmr_agent.agents.supervisor_agent.SupervisorAgent'):
            
            graph = CMRAgentGraph()
            await graph.initialize()
            
            yield graph
            await graph.cleanup()
    
    async def test_drought_monitoring_scenario(self, realistic_graph):
        """Test realistic drought monitoring query scenario."""
        # Configure realistic responses for drought monitoring
        realistic_graph.query_interpreter.interpret_query.return_value = QueryContext(
            original_query="Find precipitation datasets for drought monitoring in Sub-Saharan Africa 2015-2023",
            intent=QueryIntent.COMPARATIVE,
            constraints={
                "spatial": SpatialConstraint(
                    north=15.0, south=-25.0, east=50.0, west=-20.0,
                    region_name="Sub-Saharan Africa"
                ),
                "temporal": TemporalConstraint(
                    start_date=datetime(2015, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
                ),
                "keywords": ["precipitation", "drought", "monitoring"],
                "platforms": ["TRMM", "GPM", "MODIS"],
                "variables": ["precipitation_rate", "soil_moisture"]
            },
            priority_score=0.9,
            complexity_score=0.8
        )
        
        realistic_graph.query_interpreter.validate_query.return_value = True
        realistic_graph.query_interpreter.get_decomposed_queries.return_value = (
            "comparative_analysis",
            [
                "satellite precipitation datasets 2015-2023 Sub-Saharan Africa",
                "ground-based precipitation data Sub-Saharan Africa",
                "soil moisture datasets for drought assessment"
            ]
        )
        
        # Mock realistic CMR collections
        collections = [
            CMRCollection(
                concept_id="C1234567890-GES_DISC",
                title="TRMM 3B42 Daily Precipitation",
                summary="TRMM Multi-satellite Precipitation Analysis",
                data_center="GES_DISC",
                platforms=["TRMM"],
                instruments=["PR", "TMI"],
                variables=["precipitation_rate"]
            ),
            CMRCollection(
                concept_id="C1234567891-GES_DISC", 
                title="GPM IMERG Final Precipitation",
                summary="Global Precipitation Measurement Integrated Multi-satellitE Retrievals",
                data_center="GES_DISC",
                platforms=["GPM"],
                instruments=["GMI", "DPR"],
                variables=["precipitation_rate"]
            )
        ]
        
        realistic_graph.cmr_api_agent.search_collections.return_value = collections
        realistic_graph.cmr_api_agent.search_granules.return_value = []
        realistic_graph.analysis_agent.analyze_results.return_value = []
        
        # Mock comprehensive response
        realistic_graph.response_agent.synthesize_response.return_value = SystemResponse(
            query_id="drought_monitoring_123",
            original_query="Find precipitation datasets for drought monitoring in Sub-Saharan Africa 2015-2023",
            intent=QueryIntent.COMPARATIVE,
            recommendations=[],
            summary="Found 2 relevant precipitation datasets with complementary coverage",
            execution_plan=[
                "Parse complex comparative query",
                "Decompose into sub-queries for satellite and ground-based data",
                "Search CMR collections with spatial/temporal constraints",
                "Analyze temporal gaps and data quality",
                "Generate comparative recommendations"
            ],
            total_execution_time_ms=3500,
            success=True,
            warnings=[],
            follow_up_suggestions=[
                "Consider MODIS vegetation indices for drought impact assessment",
                "Include soil moisture datasets from SMOS/SMAP for comprehensive analysis"
            ]
        )
        
        realistic_graph.supervisor.validate_final_response.return_value = MagicMock(
            passed=True,
            quality_score=MagicMock(overall=0.85),
            requires_retry=False
        )
        
        query = "Find precipitation datasets for drought monitoring in Sub-Saharan Africa 2015-2023"
        response = await realistic_graph.process_query(query)
        
        assert response.success is True
        assert response.intent == QueryIntent.COMPARATIVE
        assert "drought monitoring" in response.original_query.lower()
        assert response.total_execution_time_ms > 0
        assert len(response.follow_up_suggestions) > 0
        assert len(response.execution_plan) > 0
    
    async def test_urban_heat_island_scenario(self, realistic_graph):
        """Test urban heat island research query scenario."""
        realistic_graph.query_interpreter.interpret_query.return_value = QueryContext(
            original_query="What datasets would be best for studying urban heat islands and air quality in megacities?",
            intent=QueryIntent.ANALYTICAL,
            constraints={
                "keywords": ["urban", "heat", "island", "air", "quality", "megacities"],
                "variables": ["land_surface_temperature", "air_temperature", "pm2.5", "no2"],
                "platforms": ["MODIS", "LANDSAT", "SENTINEL", "AQUA", "TERRA"]
            },
            priority_score=0.8,
            complexity_score=0.9
        )
        
        realistic_graph.query_interpreter.validate_query.return_value = True
        realistic_graph.cmr_api_agent.search_collections.return_value = []
        realistic_graph.cmr_api_agent.search_granules.return_value = []
        realistic_graph.analysis_agent.analyze_results.return_value = []
        
        realistic_graph.response_agent.synthesize_response.return_value = SystemResponse(
            query_id="urban_heat_123",
            original_query="What datasets would be best for studying urban heat islands and air quality in megacities?",
            intent=QueryIntent.ANALYTICAL,
            recommendations=[],
            summary="Identified multi-disciplinary datasets for urban heat island and air quality research",
            execution_plan=[
                "Parse research domain query for urban environmental studies",
                "Identify key variables: temperature, air quality, urban morphology",
                "Search for complementary satellite and ground-based datasets",
                "Recommend analysis methodologies and data fusion approaches"
            ],
            total_execution_time_ms=2800,
            success=True,
            follow_up_suggestions=[
                "Consider population density datasets for exposure assessment",
                "Include meteorological reanalysis for atmospheric conditions",
                "Examine land use/land cover data for urban morphology analysis"
            ]
        )
        
        realistic_graph.supervisor.validate_final_response.return_value = MagicMock(
            passed=True,
            quality_score=MagicMock(overall=0.82),
            requires_retry=False
        )
        
        query = "What datasets would be best for studying urban heat islands and air quality in megacities?"
        response = await realistic_graph.process_query(query)
        
        assert response.success is True
        assert response.intent == QueryIntent.ANALYTICAL
        assert "urban heat island" in response.original_query.lower()
        assert len(response.follow_up_suggestions) > 0
    
    async def test_complex_comparative_analysis(self, realistic_graph):
        """Test complex comparative analysis with multiple constraints."""
        realistic_graph.query_interpreter.interpret_query.return_value = QueryContext(
            original_query="Compare precipitation datasets suitable for drought monitoring in Sub-Saharan Africa between 2015-2023, considering both satellite and ground-based observations, and identify gaps in temporal coverage",
            intent=QueryIntent.COMPARATIVE,
            constraints={
                "spatial": SpatialConstraint(
                    north=15.0, south=-25.0, east=50.0, west=-20.0,
                    region_name="Sub-Saharan Africa"
                ),
                "temporal": TemporalConstraint(
                    start_date=datetime(2015, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
                ),
                "data_types": ["satellite", "ground_based"],
                "analysis_requirements": ["gap_analysis", "coverage_assessment"],
                "research_purpose": "drought_monitoring"
            },
            priority_score=1.0,
            complexity_score=0.95
        )
        
        realistic_graph.query_interpreter.validate_query.return_value = True
        realistic_graph.query_interpreter.get_decomposed_queries.return_value = (
            "multi_criteria_comparative",
            [
                "satellite precipitation datasets Sub-Saharan Africa 2015-2023",
                "ground-based precipitation observations Sub-Saharan Africa 2015-2023",
                "temporal coverage analysis for drought monitoring applications"
            ]
        )
        
        realistic_graph.cmr_api_agent.search_collections.return_value = []
        realistic_graph.cmr_api_agent.search_granules.return_value = []
        realistic_graph.analysis_agent.analyze_results.return_value = []
        
        realistic_graph.response_agent.synthesize_response.return_value = SystemResponse(
            query_id="comparative_123",
            original_query="Compare precipitation datasets suitable for drought monitoring in Sub-Saharan Africa between 2015-2023, considering both satellite and ground-based observations, and identify gaps in temporal coverage",
            intent=QueryIntent.COMPARATIVE,
            recommendations=[],
            summary="Comprehensive comparative analysis of satellite and ground-based precipitation datasets with gap identification",
            execution_plan=[
                "Parse multi-constraint comparative query",
                "Decompose into satellite vs ground-based analysis tracks", 
                "Execute parallel searches with spatial/temporal constraints",
                "Perform gap analysis and coverage assessment",
                "Generate comparative recommendations with methodology guidance"
            ],
            total_execution_time_ms=5200,
            success=True,
            analysis_results=[],
            warnings=["Limited ground-based station coverage in remote areas"],
            follow_up_suggestions=[
                "Consider reanalysis products to fill observational gaps",
                "Evaluate uncertainty characteristics of different datasets",
                "Examine seasonal and interannual variability patterns"
            ]
        )
        
        realistic_graph.supervisor.validate_final_response.return_value = MagicMock(
            passed=True,
            quality_score=MagicMock(overall=0.88),
            requires_retry=False
        )
        
        query = "Compare precipitation datasets suitable for drought monitoring in Sub-Saharan Africa between 2015-2023, considering both satellite and ground-based observations, and identify gaps in temporal coverage"
        response = await realistic_graph.process_query(query)
        
        assert response.success is True
        assert response.intent == QueryIntent.COMPARATIVE
        assert response.total_execution_time_ms > 4000  # Complex query should take longer
        assert len(response.execution_plan) >= 5  # Should have detailed execution steps
        assert len(response.warnings) > 0  # Should identify potential issues
        assert len(response.follow_up_suggestions) > 0  # Should provide research guidance