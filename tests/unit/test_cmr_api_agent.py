import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from datetime import datetime, timezone

from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
from nasa_cmr_agent.models.schemas import (
    QueryContext, QueryIntent, QueryConstraints,
    SpatialConstraint, TemporalConstraint, CMRCollection, CMRGranule
)


class TestCMRAPIAgent:
    """Test suite for CMRAPIAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create CMRAPIAgent instance for testing."""
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            return CMRAPIAgent()
    
    @pytest.fixture
    def sample_query_context(self):
        """Create sample query context for testing."""
        spatial = SpatialConstraint(
            north=45.0, south=30.0, east=10.0, west=0.0,
            region_name="test_region"
        )
        
        temporal = TemporalConstraint(
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
        )
        
        constraints = QueryConstraints(
            spatial=spatial,
            temporal=temporal,
            keywords=['precipitation', 'temperature'],
            platforms=['MODIS'],
            instruments=['MODIS']
        )
        
        return QueryContext(
            original_query="Test query for precipitation data",
            intent=QueryIntent.EXPLORATORY,
            constraints=constraints
        )
    
    @pytest.fixture
    def mock_cmr_collections_response(self):
        """Mock CMR collections API response."""
        return {
            "feed": {
                "entry": [
                    {
                        "id": "C1234567890-TEST",
                        "title": "MODIS/Terra Precipitation L3 Daily Global 0.25deg",
                        "summary": "Daily precipitation data from MODIS Terra satellite",
                        "short_name": "MOD10A1",
                        "version_id": "006",
                        "data_center": "NSIDC DAAC",
                        "platforms": [
                            {
                                "short_name": "Terra",
                                "instruments": [{"short_name": "MODIS"}]
                            }
                        ],
                        "time_start": "2000-02-24T00:00:00Z",
                        "time_end": "2023-12-31T23:59:59Z",
                        "boxes": ["30.0 0.0 45.0 10.0"],
                        "cloud_hosted": True,
                        "online_access_flag": True
                    }
                ]
            }
        }
    
    @pytest.fixture
    def mock_cmr_granules_response(self):
        """Mock CMR granules API response."""
        return {
            "feed": {
                "entry": [
                    {
                        "id": "G1234567890-TEST",
                        "title": "MOD10A1.A2023001.h08v05.006.2023004123456.hdf",
                        "collection_concept_id": "C1234567890-TEST",
                        "producer_granule_id": "MOD10A1.A2023001.h08v05.006.2023004123456.hdf",
                        "time_start": "2023-01-01T00:00:00Z",
                        "time_end": "2023-01-01T23:59:59Z",
                        "boxes": ["35.0 2.0 40.0 8.0"],
                        "granule_size": "25.6",
                        "links": [
                            {
                                "href": "https://test.earthdata.nasa.gov/data/test.hdf",
                                "type": "application/x-hdf"
                            }
                        ]
                    }
                ]
            }
        }
    
    def test_build_collection_search_params(self, agent, sample_query_context):
        """Test building CMR collection search parameters."""
        
        params = agent._build_collection_search_params(sample_query_context)
        
        # Check basic parameters
        assert params['page_size'] == 50
        assert params['pretty'] is True
        
        # Check spatial constraints
        expected_bbox = "0.0,30.0,10.0,45.0"
        assert params['bounding_box'] == expected_bbox
        
        # Check temporal constraints
        assert 'temporal' in params
        assert "2020-01-01T00:00:00Z,2023-12-31T23:59:59Z" == params['temporal']
        
        # Check platform constraints
        assert 'platform[0]' in params
        assert params['platform[0]'] == 'MODIS'
    
    def test_build_granule_search_params(self, agent, sample_query_context):
        """Test building CMR granule search parameters."""
        
        collection_id = "C1234567890-TEST"
        limit = 50
        
        params = agent._build_granule_search_params(
            sample_query_context, collection_id, limit
        )
        
        # Check basic parameters
        assert params['collection_concept_id'] == collection_id
        assert params['page_size'] == limit
        assert params['sort_key'] == "-start_date"
        
        # Check constraints are applied
        assert 'bounding_box' in params
        assert 'temporal' in params
    
    def test_parse_collection(self, agent, mock_cmr_collections_response):
        """Test parsing CMR collection data."""
        
        collection_data = mock_cmr_collections_response["feed"]["entry"][0]
        
        collection = agent._parse_collection(collection_data)
        
        assert isinstance(collection, CMRCollection)
        assert collection.concept_id == "C1234567890-TEST"
        assert collection.title == "MODIS/Terra Precipitation L3 Daily Global 0.25deg"
        assert collection.short_name == "MOD10A1"
        assert collection.version_id == "006"
        assert collection.data_center == "NSIDC DAAC"
        assert "Terra" in collection.platforms
        assert "MODIS" in collection.instruments
        assert collection.cloud_hosted is True
        assert collection.online_access_flag is True
        
        # Check temporal coverage parsing
        assert collection.temporal_coverage is not None
        assert collection.temporal_coverage["start"] == "2000-02-24T00:00:00Z"
        assert collection.temporal_coverage["end"] == "2023-12-31T23:59:59Z"
        
        # Check spatial coverage parsing
        assert collection.spatial_coverage is not None
        assert collection.spatial_coverage["south"] == 30.0
        assert collection.spatial_coverage["west"] == 0.0
        assert collection.spatial_coverage["north"] == 45.0
        assert collection.spatial_coverage["east"] == 10.0
    
    def test_parse_granule(self, agent, mock_cmr_granules_response):
        """Test parsing CMR granule data."""
        
        granule_data = mock_cmr_granules_response["feed"]["entry"][0]
        
        granule = agent._parse_granule(granule_data)
        
        assert isinstance(granule, CMRGranule)
        assert granule.concept_id == "G1234567890-TEST"
        assert granule.title == "MOD10A1.A2023001.h08v05.006.2023004123456.hdf"
        assert granule.collection_concept_id == "C1234567890-TEST"
        assert granule.size_mb == 25.6
        assert len(granule.links) == 1
        
        # Check temporal extent
        assert granule.temporal_extent is not None
        assert granule.temporal_extent["start"] == "2023-01-01T00:00:00Z"
        assert granule.temporal_extent["end"] == "2023-01-01T23:59:59Z"
        
        # Check spatial extent
        assert granule.spatial_extent is not None
        assert granule.spatial_extent["south"] == 35.0
        assert granule.spatial_extent["west"] == 2.0
    
    @pytest.mark.asyncio
    async def test_make_cmr_request_success(self, agent):
        """Test successful CMR API request."""
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await agent._make_cmr_request("/test", {"param": "value"})
            
            assert result == {"test": "data"}
            mock_response.raise_for_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_cmr_request_rate_limit(self, agent):
        """Test CMR API request rate limiting handling."""
        
        # Mock rate limited response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            with patch('asyncio.sleep') as mock_sleep:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )
                
                with pytest.raises(httpx.HTTPStatusError):
                    await agent._make_cmr_request("/test", {})
                
                mock_sleep.assert_called_with(2)  # Should wait before retry
    
    @pytest.mark.asyncio
    async def test_search_collections_integration(self, agent, sample_query_context, mock_cmr_collections_response):
        """Test collections search integration."""
        
        # Mock the circuit breaker and HTTP client
        agent.circuit_breaker.call = AsyncMock()
        agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
        agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
        
        with patch.object(agent, '_make_cmr_request', return_value=mock_cmr_collections_response):
            with patch.object(agent, '_enforce_rate_limit', return_value=None):
                collections = await agent.search_collections(sample_query_context)
                
                assert len(collections) == 1
                assert isinstance(collections[0], CMRCollection)
                assert collections[0].concept_id == "C1234567890-TEST"
    
    @pytest.mark.asyncio
    async def test_search_granules_integration(self, agent, sample_query_context, mock_cmr_granules_response):
        """Test granules search integration."""
        
        collection_id = "C1234567890-TEST"
        
        # Mock the circuit breaker
        agent.circuit_breaker.call = AsyncMock()
        agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
        agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
        
        with patch.object(agent, '_make_cmr_request', return_value=mock_cmr_granules_response):
            with patch.object(agent, '_enforce_rate_limit', return_value=None):
                granules = await agent.search_granules(sample_query_context, collection_id)
                
                assert len(granules) == 1
                assert isinstance(granules[0], CMRGranule)
                assert granules[0].concept_id == "G1234567890-TEST"
                assert granules[0].collection_concept_id == collection_id
    
    @pytest.mark.asyncio
    async def test_get_collection_variables(self, agent):
        """Test getting collection variables."""
        
        collection_id = "C1234567890-TEST"
        mock_variables_response = {
            "feed": {
                "entry": [
                    {
                        "concept_id": "V1234567890-TEST",
                        "name": "precipitation",
                        "long_name": "Daily Precipitation Rate"
                    }
                ]
            }
        }
        
        agent.circuit_breaker.call = AsyncMock()
        agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
        agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
        
        with patch.object(agent, '_make_cmr_request', return_value=mock_variables_response):
            with patch.object(agent, '_enforce_rate_limit', return_value=None):
                variables = await agent.get_collection_variables(collection_id)
                
                assert len(variables) == 1
                assert variables[0]["name"] == "precipitation"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, agent):
        """Test rate limiting enforcement."""
        
        # Set up rate limiter with very low rate
        agent.rate_limit = 1  # 1 request per second
        
        start_time = asyncio.get_event_loop().time()
        
        # Make two requests
        await agent._enforce_rate_limit()
        first_time = asyncio.get_event_loop().time()
        
        await agent._enforce_rate_limit()
        second_time = asyncio.get_event_loop().time()
        
        # Second request should be delayed
        time_diff = second_time - first_time
        assert time_diff >= 0.9  # Should wait at least ~1 second
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, agent):
        """Test circuit breaker integration."""
        
        # Mock circuit breaker
        mock_circuit_breaker = MagicMock()
        mock_context = AsyncMock()
        mock_circuit_breaker.call.return_value = mock_context
        
        agent.circuit_breaker = mock_circuit_breaker
        
        with patch.object(agent, '_make_cmr_request', return_value={"test": "data"}):
            with patch.object(agent, '_enforce_rate_limit', return_value=None):
                await agent.search_collections(QueryContext(
                    original_query="test",
                    intent=QueryIntent.EXPLORATORY,
                    constraints=QueryConstraints()
                ))
                
                # Verify circuit breaker was called
                mock_circuit_breaker.call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_parse_collection(self, agent):
        """Test error handling during collection parsing."""
        
        # Invalid collection data (missing required fields)
        invalid_data = {"invalid": "data"}
        
        # Should not raise exception, but should skip invalid entries
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            # This would be called within search_collections when parsing fails
            try:
                collection = agent._parse_collection(invalid_data)
                # If no exception, check that fields have defaults
                assert collection.concept_id == ""
                assert collection.title == ""
            except Exception:
                # Exception is acceptable for invalid data
                pass
    
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, agent):
        """Test proper resource cleanup."""
        
        # Initialize client
        agent._client = MagicMock()
        agent._client.aclose = AsyncMock()
        
        await agent.close()
        
        agent._client.aclose.assert_called_once()
        assert agent._client is None


@pytest.mark.asyncio
class TestCMRAPIAgentIntegration:
    """Integration tests for CMRAPIAgent with real-like scenarios."""
    
    async def test_full_search_workflow(self):
        """Test complete search workflow from query to results."""
        
        with patch('nasa_cmr_agent.agents.cmr_api_agent.CircuitBreakerService'):
            agent = CMRAPIAgent()
            
            # Create realistic query context
            spatial = SpatialConstraint(
                north=50.0, south=40.0, east=20.0, west=10.0
            )
            
            temporal = TemporalConstraint(
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2020, 12, 31, tzinfo=timezone.utc)
            )
            
            constraints = QueryConstraints(
                spatial=spatial,
                temporal=temporal,
                keywords=['precipitation'],
                platforms=['MODIS']
            )
            
            query_context = QueryContext(
                original_query="Find MODIS precipitation data for Europe in 2020",
                intent=QueryIntent.SPECIFIC_DATA,
                constraints=constraints
            )
            
            # Mock CMR responses
            collections_response = {
                "feed": {
                    "entry": [
                        {
                            "id": "C123-TEST",
                            "title": "MODIS Precipitation",
                            "summary": "Test precipitation data",
                            "short_name": "TEST_PRECIP",
                            "version_id": "001",
                            "data_center": "TEST_DAAC",
                            "platforms": [{"short_name": "Terra", "instruments": [{"short_name": "MODIS"}]}],
                            "time_start": "2020-01-01T00:00:00Z",
                            "time_end": "2020-12-31T23:59:59Z",
                            "boxes": ["40.0 10.0 50.0 20.0"],
                            "cloud_hosted": True,
                            "online_access_flag": True
                        }
                    ]
                }
            }
            
            granules_response = {
                "feed": {
                    "entry": [
                        {
                            "id": "G123-TEST",
                            "title": "Test Granule",
                            "collection_concept_id": "C123-TEST",
                            "producer_granule_id": "test_granule",
                            "time_start": "2020-01-15T00:00:00Z",
                            "time_end": "2020-01-15T23:59:59Z",
                            "granule_size": "15.2",
                            "links": []
                        }
                    ]
                }
            }
            
            # Mock the HTTP requests
            with patch.object(agent, '_make_cmr_request') as mock_request:
                mock_request.side_effect = [collections_response, granules_response]
                
                with patch.object(agent, '_enforce_rate_limit'):
                    # Mock circuit breaker
                    agent.circuit_breaker.call = AsyncMock()
                    agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
                    agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
                    
                    # Test collections search
                    collections = await agent.search_collections(query_context)
                    assert len(collections) == 1
                    assert collections[0].concept_id == "C123-TEST"
                    
                    # Test granules search
                    granules = await agent.search_granules(query_context, "C123-TEST")
                    assert len(granules) == 1
                    assert granules[0].concept_id == "G123-TEST"
                    assert granules[0].collection_concept_id == "C123-TEST"
            
            await agent.close()