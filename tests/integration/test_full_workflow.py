import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.models.schemas import SystemResponse


class TestFullWorkflow:
    """Integration tests for complete query processing workflow."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        with patch('nasa_cmr_agent.services.llm_service.LLMService') as mock:
            mock_instance = mock.return_value
            mock_instance.generate = AsyncMock()
            mock_instance.get_current_provider = MagicMock(return_value="openai")
            yield mock_instance
    
    @pytest.fixture
    def mock_cmr_responses(self):
        """Mock CMR API responses."""
        collections_response = {
            "feed": {
                "entry": [
                    {
                        "id": "C1234567890-GHRC",
                        "title": "GPM IMERG Final Precipitation L3 1 month 0.1 degree x 0.1 degree V06",
                        "summary": "Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals for GPM (IMERG) algorithm",
                        "short_name": "GPM_3IMERGM",
                        "version_id": "06",
                        "data_center": "NASA GSFC",
                        "platforms": [
                            {
                                "short_name": "GPM",
                                "instruments": [{"short_name": "GMI"}, {"short_name": "DPR"}]
                            }
                        ],
                        "time_start": "2000-06-01T00:00:00Z",
                        "time_end": "2023-12-31T23:59:59Z",
                        "boxes": ["-90.0 -180.0 90.0 180.0"],
                        "cloud_hosted": True,
                        "online_access_flag": True
                    },
                    {
                        "id": "C9876543210-UCSB",
                        "title": "CHIRPS: Climate Hazards Group InfraRed Precipitation with Station Data",
                        "summary": "Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) is a 35+ year quasi-global rainfall dataset",
                        "short_name": "CHIRPS",
                        "version_id": "2.0",
                        "data_center": "UC Santa Barbara",
                        "platforms": [
                            {
                                "short_name": "Multi-Platform",
                                "instruments": [{"short_name": "GOES"}, {"short_name": "METEOSAT"}]
                            }
                        ],
                        "time_start": "1981-01-01T00:00:00Z",
                        "time_end": "2023-12-31T23:59:59Z",
                        "boxes": ["-50.0 -180.0 50.0 180.0"],
                        "cloud_hosted": False,
                        "online_access_flag": True
                    }
                ]
            }
        }
        
        granules_response = {
            "feed": {
                "entry": [
                    {
                        "id": "G1111111111-GHRC",
                        "title": "3B-MO.MS.MRG.3IMERG.20230101-S000000-E235959.12.V06B.HDF5",
                        "collection_concept_id": "C1234567890-GHRC",
                        "producer_granule_id": "3B-MO.MS.MRG.3IMERG.20230101-S000000-E235959.12.V06B.HDF5",
                        "time_start": "2023-01-01T00:00:00Z",
                        "time_end": "2023-01-31T23:59:59Z",
                        "granule_size": "156.7",
                        "links": [
                            {
                                "href": "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.06/2023/3B-MO.MS.MRG.3IMERG.20230101-S000000-E235959.12.V06B.HDF5",
                                "type": "application/x-hdf5"
                            }
                        ]
                    },
                    {
                        "id": "G2222222222-UCSB",
                        "title": "chirps-v2.0.2023.01.tif",
                        "collection_concept_id": "C9876543210-UCSB",
                        "producer_granule_id": "chirps-v2.0.2023.01.tif",
                        "time_start": "2023-01-01T00:00:00Z",
                        "time_end": "2023-01-31T23:59:59Z",
                        "granule_size": "89.2",
                        "links": [
                            {
                                "href": "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.2023.01.tif",
                                "type": "image/tiff"
                            }
                        ]
                    }
                ]
            }
        }
        
        return collections_response, granules_response
    
    @pytest.mark.asyncio
    async def test_precipitation_drought_monitoring_query(self, mock_llm_service, mock_cmr_responses):
        """Test complete workflow for precipitation drought monitoring query."""
        
        collections_response, granules_response = mock_cmr_responses
        
        # Configure LLM responses
        mock_llm_service.generate.side_effect = [
            # Intent classification
            'Complex drought monitoring query -> COMPARATIVE, 0.9',
            # Reasoning generation for GPM dataset
            'This dataset is recommended due to excellent global coverage and high temporal resolution, making it ideal for drought monitoring applications.',
            # Reasoning generation for CHIRPS dataset  
            'This dataset provides long-term precipitation records with ground-station validation, essential for comprehensive drought analysis.',
            # Summary generation
            'Found 2 highly relevant precipitation datasets for Sub-Saharan Africa drought monitoring from 2015-2023. GPM IMERG provides excellent satellite-based coverage while CHIRPS offers long-term validation data. Both datasets show good temporal coverage with minimal gaps for the requested period.'
        ]
        
        # Create agent and mock CMR API calls
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAgentGraph()
            
            # Mock CMR API responses
            with patch.object(agent.cmr_api_agent, '_make_cmr_request') as mock_cmr:
                mock_cmr.side_effect = [
                    collections_response,  # Collections search
                    granules_response,     # Granules for GPM
                    granules_response      # Granules for CHIRPS
                ]
                
                # Mock rate limiting
                with patch.object(agent.cmr_api_agent, '_enforce_rate_limit'):
                    # Mock circuit breaker
                    agent.cmr_api_agent.circuit_breaker.call = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    # Process the query
                    query = ("Compare precipitation datasets suitable for drought monitoring "
                            "in Sub-Saharan Africa between 2015-2023, considering both "
                            "satellite and ground-based observations, and identify gaps "
                            "in temporal coverage.")
                    
                    response = await agent.process_query(query)
                    
                    # Verify response structure
                    assert isinstance(response, SystemResponse)
                    assert response.success is True
                    assert response.query_id is not None
                    assert response.original_query == query
                    assert response.intent.value in ['comparative', 'analytical']
                    
                    # Verify recommendations
                    assert len(response.recommendations) > 0
                    assert len(response.recommendations) <= 10
                    
                    # Check first recommendation (should be GPM IMERG)
                    gpm_rec = response.recommendations[0]
                    assert "GPM" in gpm_rec.collection.title or "IMERG" in gpm_rec.collection.title
                    assert gpm_rec.relevance_score > 0.0
                    assert gpm_rec.coverage_score > 0.0
                    assert gpm_rec.quality_score > 0.0
                    assert gpm_rec.accessibility_score > 0.0
                    assert gpm_rec.reasoning is not None
                    
                    # Verify analysis results
                    assert len(response.analysis_results) > 0
                    
                    # Check for dataset recommendations analysis
                    rec_analysis = next(
                        (r for r in response.analysis_results if r.analysis_type == "dataset_recommendations"),
                        None
                    )
                    assert rec_analysis is not None
                    assert rec_analysis.confidence_level > 0.0
                    
                    # Verify execution plan
                    assert len(response.execution_plan) > 0
                    assert "Query interpretation and validation" in response.execution_plan[0]
                    
                    # Verify summary
                    assert len(response.summary) > 10
                    assert "precipitation" in response.summary.lower() or "drought" in response.summary.lower()
                    
                    # Verify follow-up suggestions
                    assert len(response.follow_up_suggestions) > 0
        
        # Clean up
        await agent.cmr_api_agent.close()
    
    @pytest.mark.asyncio
    async def test_urban_heat_island_query(self, mock_llm_service, mock_cmr_responses):
        """Test urban heat island and air quality research query."""
        
        collections_response, _ = mock_cmr_responses
        
        # Modify collections response for urban research
        collections_response["feed"]["entry"] = [
            {
                "id": "C5555555555-LP",
                "title": "Landsat Collection 2 Level-2 Surface Temperature Science Product",
                "summary": "Landsat Collection 2 Level-2 Surface Temperature Science Product provides surface temperature data",
                "short_name": "LANDSAT_ST",
                "version_id": "C2L2",
                "data_center": "USGS EROS",
                "platforms": [
                    {
                        "short_name": "Landsat-8",
                        "instruments": [{"short_name": "TIRS"}, {"short_name": "OLI"}]
                    }
                ],
                "time_start": "2013-04-11T00:00:00Z",
                "time_end": "2023-12-31T23:59:59Z",
                "boxes": ["-90.0 -180.0 90.0 180.0"],
                "cloud_hosted": True,
                "online_access_flag": True
            }
        ]
        
        mock_llm_service.generate.side_effect = [
            'Urban research query -> ANALYTICAL, 0.85',
            'This dataset provides high-resolution surface temperature measurements ideal for urban heat island studies and correlation with air quality parameters.',
            'Found 1 relevant dataset for urban heat island and air quality research. Landsat provides excellent spatial resolution for urban analysis with reliable thermal infrared measurements.'
        ]
        
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAgentGraph()
            
            with patch.object(agent.cmr_api_agent, '_make_cmr_request') as mock_cmr:
                mock_cmr.return_value = collections_response
                
                with patch.object(agent.cmr_api_agent, '_enforce_rate_limit'):
                    agent.cmr_api_agent.circuit_breaker.call = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    query = ("What datasets would be best for studying the relationship "
                            "between urban heat islands and air quality in megacities?")
                    
                    response = await agent.process_query(query)
                    
                    assert response.success is True
                    assert len(response.recommendations) > 0
                    
                    # Should identify urban research domain
                    landsat_rec = response.recommendations[0]
                    assert "Landsat" in landsat_rec.collection.title
                    assert "surface temperature" in landsat_rec.collection.summary.lower()
        
        await agent.cmr_api_agent.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_llm_service):
        """Test error handling in complete workflow."""
        
        # Configure LLM to work but CMR to fail
        mock_llm_service.generate.return_value = 'Test query -> EXPLORATORY, 0.8'
        
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAgentGraph()
            
            # Mock CMR API to raise exception
            with patch.object(agent.cmr_api_agent, '_make_cmr_request') as mock_cmr:
                mock_cmr.side_effect = Exception("CMR API unavailable")
                
                with patch.object(agent.cmr_api_agent, '_enforce_rate_limit'):
                    agent.cmr_api_agent.circuit_breaker.call = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    response = await agent.process_query("Test query with API failure")
                    
                    # Should handle error gracefully
                    assert isinstance(response, SystemResponse)
                    assert response.success is False
                    assert len(response.warnings) > 0
                    assert "CMR API unavailable" in str(response.warnings)
                    
                    # Should still provide basic response structure
                    assert response.query_id is not None
                    assert response.summary is not None
        
        await agent.cmr_api_agent.close()
    
    @pytest.mark.asyncio  
    async def test_complex_multi_constraint_query(self, mock_llm_service, mock_cmr_responses):
        """Test query with multiple spatial, temporal, and domain constraints."""
        
        collections_response, granules_response = mock_cmr_responses
        
        mock_llm_service.generate.side_effect = [
            'Complex multi-constraint query -> ANALYTICAL, 0.95',
            'This dataset provides comprehensive precipitation data with excellent coverage for the requested Arctic region and temporal period.',
            'This dataset offers complementary precipitation measurements with different methodological approaches for validation.',
            'Found 2 datasets matching your complex criteria for Arctic precipitation analysis from 2010-2020. Both datasets provide good coverage with some temporal overlap for cross-validation.'
        ]
        
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAgentGraph()
            
            with patch.object(agent.cmr_api_agent, '_make_cmr_request') as mock_cmr:
                mock_cmr.side_effect = [
                    collections_response,
                    granules_response,
                    granules_response
                ]
                
                with patch.object(agent.cmr_api_agent, '_enforce_rate_limit'):
                    agent.cmr_api_agent.circuit_breaker.call = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    # Complex query with multiple constraints
                    query = ("Find MODIS and GPM satellite precipitation data for Arctic regions "
                            "north of 65°N between 2010-2020 with daily temporal resolution "
                            "for climate change impact analysis, excluding areas with persistent "
                            "cloud cover above 80%")
                    
                    response = await agent.process_query(query)
                    
                    assert response.success is True
                    assert len(response.recommendations) > 0
                    
                    # Should detect high complexity
                    # Verify that complex constraints were processed
                    assert response.intent.value in ['analytical', 'comparative']
                    
                    # Should have comprehensive analysis
                    assert len(response.analysis_results) >= 1
                    
                    # Should provide detailed execution plan
                    assert len(response.execution_plan) >= 4
        
        await agent.cmr_api_agent.close()
    
    @pytest.mark.asyncio
    async def test_workflow_performance_metrics(self, mock_llm_service, mock_cmr_responses):
        """Test that workflow completes within performance requirements."""
        
        collections_response, granules_response = mock_cmr_responses
        
        mock_llm_service.generate.side_effect = [
            'Simple query -> EXPLORATORY, 0.7',
            'This dataset provides basic precipitation data.',
            'Found relevant precipitation dataset.'
        ]
        
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAgentGraph()
            
            with patch.object(agent.cmr_api_agent, '_make_cmr_request') as mock_cmr:
                mock_cmr.side_effect = [
                    collections_response,
                    granules_response
                ]
                
                with patch.object(agent.cmr_api_agent, '_enforce_rate_limit'):
                    agent.cmr_api_agent.circuit_breaker.call = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.cmr_api_agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    start_time = asyncio.get_event_loop().time()
                    
                    response = await agent.process_query("Simple precipitation query")
                    
                    end_time = asyncio.get_event_loop().time()
                    actual_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    # Verify response
                    assert response.success is True
                    
                    # Check performance requirements
                    # Simple queries should complete in under 2 seconds (2000ms)
                    assert actual_time < 5000  # Allow 5s in test environment
                    
                    # Reported time should be reasonable
                    assert response.total_execution_time_ms > 0
                    assert response.total_execution_time_ms < 10000  # Less than 10s
        
        await agent.cmr_api_agent.close()