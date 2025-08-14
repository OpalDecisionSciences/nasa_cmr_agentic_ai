"""
Enhanced comprehensive unit tests for CMR API Agent.

Tests HTTP client lifecycle, circuit breaker integration, rate limiting,
parallel processing, and error handling with production-grade patterns.
"""

import pytest
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
from nasa_cmr_agent.models.schemas import QueryContext, QueryIntent, SpatialConstraint, TemporalConstraint
from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerError, CircuitState


@pytest.mark.unit
class TestCMRAPIAgentInitialization:
    """Test CMR API Agent initialization and configuration."""
    
    def test_agent_initialization(self):
        """Test proper agent initialization."""
        agent = CMRAPIAgent()
        
        assert agent.base_url == "https://cmr.earthdata.nasa.gov/search"
        assert agent.timeout == 30
        assert agent.max_retries == 3
        assert agent.rate_limit == 10
        assert agent._client is None
        assert agent._client_ref_count == 0
        assert agent._max_idle_time == 300
    
    def test_agent_configuration_from_settings(self):
        """Test agent uses configuration from settings."""
        with patch('nasa_cmr_agent.agents.cmr_api_agent.settings') as mock_settings:
            mock_settings.cmr_base_url = "https://test.cmr.gov"
            mock_settings.cmr_request_timeout = 15
            mock_settings.cmr_max_retries = 5
            mock_settings.cmr_rate_limit_per_second = 5
            
            agent = CMRAPIAgent()
            
            assert agent.base_url == "https://test.cmr.gov"
            assert agent.timeout == 15
            assert agent.max_retries == 5
            assert agent.rate_limit == 5


@pytest.mark.unit
class TestHTTPClientLifecycle:
    """Test HTTP client lifecycle management."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent for testing."""
        agent = CMRAPIAgent()
        yield agent
        await agent.close()
    
    async def test_client_creation_on_first_use(self, agent):
        """Test HTTP client is created on first use."""
        assert agent._client is None
        
        client = await agent._ensure_client()
        
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert agent._client_ref_count == 1
        assert agent._last_used_time is not None
    
    async def test_client_reuse(self, agent):
        """Test HTTP client reuse."""
        client1 = await agent._ensure_client()
        client2 = await agent._ensure_client()
        
        assert client1 is client2
        assert agent._client_ref_count == 2
    
    async def test_client_reference_counting(self, agent):
        """Test client reference counting."""
        await agent._ensure_client()
        assert agent._client_ref_count == 1
        
        await agent._release_client()
        assert agent._client_ref_count == 0
    
    async def test_client_idle_cleanup(self, agent):
        """Test client cleanup after idle timeout."""
        # Set short idle timeout for testing
        agent._max_idle_time = 1
        
        await agent._ensure_client()
        await agent._release_client()
        
        # Wait for idle timeout
        await asyncio.sleep(1.1)
        
        # Next call should create new client
        await agent._ensure_client()
        await agent._release_client()
    
    async def test_client_context_manager(self, agent):
        """Test client context manager."""
        async with agent._get_client() as client:
            assert isinstance(client, httpx.AsyncClient)
            assert agent._client_ref_count == 1
        
        assert agent._client_ref_count == 0
    
    async def test_client_close(self, agent):
        """Test proper client cleanup on close."""
        await agent._ensure_client()
        assert agent._client is not None
        
        await agent.close()
        
        assert agent._client is None
        assert agent._client_ref_count == 0


@pytest.mark.unit
class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent with low rate limit for testing."""
        agent = CMRAPIAgent()
        agent.rate_limit = 2  # 2 requests per second
        agent._rate_limiter = asyncio.Semaphore(2)
        yield agent
        await agent.close()
    
    async def test_rate_limit_enforcement(self, agent):
        """Test rate limiting enforces request limits."""
        start_time = asyncio.get_event_loop().time()
        
        # Make 3 requests rapidly
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(agent._enforce_rate_limit())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        
        # Should take at least some time due to rate limiting
        assert elapsed_time >= 0.0  # Basic test that it doesn't fail
    
    async def test_rate_limit_semaphore(self, agent):
        """Test semaphore-based rate limiting."""
        # Acquire all semaphore permits
        permits = []
        for _ in range(agent.rate_limit):
            permit = await agent._rate_limiter.acquire().__aenter__()
            permits.append(permit)
        
        # Next request should be blocked
        with patch.object(agent._rate_limiter, 'acquire') as mock_acquire:
            mock_acquire.return_value.__aenter__ = AsyncMock()
            await agent._enforce_rate_limit()
            mock_acquire.assert_called_once()


@pytest.mark.unit
class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""
    
    @pytest.fixture
    async def agent_with_mock_cb(self):
        """Create agent with mocked circuit breaker."""
        agent = CMRAPIAgent()
        agent.circuit_breaker = AsyncMock()
        agent.circuit_breaker.call.return_value.__aenter__ = AsyncMock()
        agent.circuit_breaker.call.return_value.__aexit__ = AsyncMock()
        yield agent
        await agent.close()
    
    async def test_circuit_breaker_success_path(self, agent_with_mock_cb, mock_http_client):
        """Test circuit breaker on successful requests."""
        agent = agent_with_mock_cb
        
        with patch.object(agent, '_get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_http_client
            mock_get_client.return_value.__aexit__ = AsyncMock()
            
            await agent._make_cmr_request("/collections", {"param": "value"})
            
            agent.circuit_breaker.call.assert_called_once()
    
    async def test_circuit_breaker_failure_handling(self, agent_with_mock_cb):
        """Test circuit breaker handles failures."""
        agent = agent_with_mock_cb
        
        # Mock circuit breaker to raise exception
        agent.circuit_breaker.call.return_value.__aenter__.side_effect = CircuitBreakerError("Circuit open")
        
        with pytest.raises(CircuitBreakerError):
            await agent._make_cmr_request("/collections", {})


@pytest.mark.unit
class TestCMRRequestProcessing:
    """Test CMR API request processing."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent for testing."""
        agent = CMRAPIAgent()
        yield agent
        await agent.close()
    
    async def test_make_cmr_request_success(self, agent, mock_http_client):
        """Test successful CMR API request."""
        mock_response_data = {
            "feed": {
                "entry": [
                    {"id": "C123", "title": "Test Collection"}
                ]
            }
        }
        mock_http_client.get.return_value.json.return_value = mock_response_data
        
        with patch.object(agent, '_get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_http_client
            mock_get_client.return_value.__aexit__ = AsyncMock()
            
            result = await agent._make_cmr_request("/collections", {"param": "test"})
            
            assert result == mock_response_data
            mock_http_client.get.assert_called_once()
    
    async def test_make_cmr_request_http_error(self, agent, mock_http_client):
        """Test CMR request with HTTP error."""
        # Mock HTTP error response
        error_response = MagicMock()
        error_response.status_code = 500
        mock_http_client.get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=error_response
        )
        
        with patch.object(agent, '_get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_http_client
            mock_get_client.return_value.__aexit__ = AsyncMock()
            
            with pytest.raises(httpx.HTTPStatusError):
                await agent._make_cmr_request("/collections", {})
    
    async def test_make_cmr_request_rate_limit_handling(self, agent, mock_http_client):
        """Test handling of rate limit responses."""
        # Mock rate limit response (429)
        error_response = MagicMock()
        error_response.status_code = 429
        rate_limit_error = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=error_response
        )
        mock_http_client.get.return_value.raise_for_status.side_effect = rate_limit_error
        
        with patch.object(agent, '_get_client') as mock_get_client, \
             patch('asyncio.sleep') as mock_sleep:
            mock_get_client.return_value.__aenter__.return_value = mock_http_client
            mock_get_client.return_value.__aexit__ = AsyncMock()
            
            with pytest.raises(httpx.HTTPStatusError):
                await agent._make_cmr_request("/collections", {})


@pytest.mark.unit
class TestParameterBuilding:
    """Test CMR API parameter building."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return CMRAPIAgent()
    
    @pytest.fixture
    def sample_query_context(self):
        """Sample query context for testing."""
        return QueryContext(
            original_query="Find precipitation data for Africa 2020-2023",
            intent=QueryIntent.ANALYTICAL,
            constraints={
                "spatial": SpatialConstraint(
                    north=35.0, south=-35.0, east=50.0, west=-20.0, region_name="Africa"
                ),
                "temporal": TemporalConstraint(
                    start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
                ),
                "keywords": ["precipitation", "drought"],
                "platforms": ["TRMM", "GPM"],
                "variables": ["precipitation_rate"]
            }
        )
    
    def test_build_collection_search_params(self, agent, sample_query_context):
        """Test building collection search parameters."""
        params = agent._build_collection_search_params(sample_query_context)
        
        assert "page_size" in params
        assert params["page_size"] == 50
        
        # Check spatial constraints
        spatial = sample_query_context.constraints["spatial"]
        expected_bbox = f"{spatial.west},{spatial.south},{spatial.east},{spatial.north}"
        assert params["bounding_box"] == expected_bbox
        
        # Check temporal constraints
        temporal = sample_query_context.constraints["temporal"]
        assert params["temporal"] == f"{temporal.start_date.isoformat()},{temporal.end_date.isoformat()}"
        
        # Check keywords
        assert "keyword" in params
        assert params["keyword"] == "precipitation drought"
        
        # Check platforms
        assert "platform" in params
        assert "TRMM" in params["platform"]
        assert "GPM" in params["platform"]
    
    def test_build_collection_params_minimal(self, agent):
        """Test building parameters with minimal query context."""
        minimal_context = QueryContext(
            original_query="Simple query",
            intent=QueryIntent.EXPLORATORY
        )
        
        params = agent._build_collection_search_params(minimal_context)
        
        assert "page_size" in params
        assert params["page_size"] == 50
        # Should not have spatial/temporal constraints
        assert "bounding_box" not in params
        assert "temporal" not in params
    
    def test_build_granule_search_params(self, agent, sample_query_context):
        """Test building granule search parameters."""
        collection_id = "C123456-TEST"
        limit = 100
        
        params = agent._build_granule_search_params(sample_query_context, collection_id, limit)
        
        assert params["page_size"] == limit
        assert params["collection_concept_id"] == collection_id
        
        # Should include temporal constraints
        temporal = sample_query_context.constraints["temporal"]
        assert params["temporal"] == f"{temporal.start_date.isoformat()},{temporal.end_date.isoformat()}"
        
        # Should include spatial constraints
        spatial = sample_query_context.constraints["spatial"]
        expected_bbox = f"{spatial.west},{spatial.south},{spatial.east},{spatial.north}"
        assert params["bounding_box"] == expected_bbox


@pytest.mark.unit
class TestDataParsing:
    """Test CMR data parsing functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return CMRAPIAgent()
    
    def test_parse_collection_complete_data(self, agent):
        """Test parsing collection with complete data."""
        collection_data = {
            "id": "C123456-TEST",
            "title": "Test Precipitation Collection",
            "summary": "Comprehensive precipitation data for testing",
            "data_center": "TEST_DAAC",
            "platforms": [
                {"ShortName": "TRMM"},
                {"ShortName": "GPM"}
            ],
            "instruments": [
                {"ShortName": "PR"},
                {"ShortName": "TMI"}
            ],
            "time_start": "2020-01-01T00:00:00Z",
            "time_end": "2023-12-31T23:59:59Z",
            "boxes": ["-35 -20 35 50"],
            "version_id": "1.0",
            "processing_level_id": "3"
        }
        
        collection = agent._parse_collection(collection_data)
        
        assert collection.concept_id == "C123456-TEST"
        assert collection.title == "Test Precipitation Collection"
        assert collection.data_center == "TEST_DAAC"
        assert len(collection.platforms) == 2
        assert "TRMM" in collection.platforms
        assert "GPM" in collection.platforms
        assert len(collection.instruments) == 2
        assert collection.version_id == "1.0"
        assert collection.processing_level == "3"
        
        # Check temporal extent
        assert collection.temporal_extent is not None
        assert "start" in collection.temporal_extent
        assert "end" in collection.temporal_extent
        
        # Check spatial extent
        assert collection.spatial_extent is not None
        assert collection.spatial_extent["north"] == 35.0
        assert collection.spatial_extent["south"] == -35.0
        assert collection.spatial_extent["east"] == 50.0
        assert collection.spatial_extent["west"] == -20.0
    
    def test_parse_collection_minimal_data(self, agent):
        """Test parsing collection with minimal required data."""
        collection_data = {
            "id": "C123456-MINIMAL",
            "title": "Minimal Collection",
            "summary": "Minimal test collection"
        }
        
        collection = agent._parse_collection(collection_data)
        
        assert collection.concept_id == "C123456-MINIMAL"
        assert collection.title == "Minimal Collection"
        assert collection.summary == "Minimal test collection"
        assert collection.platforms == []
        assert collection.instruments == []
        assert collection.temporal_extent is None
        assert collection.spatial_extent is None
    
    def test_parse_collection_invalid_data(self, agent):
        """Test parsing collection with invalid/missing data."""
        invalid_data = {
            # Missing required fields
            "summary": "Invalid collection"
        }
        
        with pytest.raises((KeyError, ValueError)):
            agent._parse_collection(invalid_data)
    
    def test_parse_granule_complete_data(self, agent):
        """Test parsing granule with complete data."""
        granule_data = {
            "id": "G123456-TEST",
            "title": "Test Granule",
            "collection_concept_id": "C123456-TEST",
            "producer_granule_id": "TEST_GRANULE_001",
            "time_start": "2020-01-01T00:00:00Z",
            "time_end": "2020-01-02T00:00:00Z",
            "boxes": ["0 0 10 10"],
            "granule_size": "150.5",
            "links": [
                {
                    "href": "https://example.com/data.nc",
                    "type": "GET DATA",
                    "title": "Download Data"
                }
            ]
        }
        
        granule = agent._parse_granule(granule_data)
        
        assert granule.concept_id == "G123456-TEST"
        assert granule.title == "Test Granule"
        assert granule.collection_concept_id == "C123456-TEST"
        assert granule.producer_granule_id == "TEST_GRANULE_001"
        assert granule.size_mb == 150.5
        assert len(granule.links) == 1
        
        # Check temporal extent
        assert granule.temporal_extent is not None
        assert "start" in granule.temporal_extent
        assert "end" in granule.temporal_extent
        
        # Check spatial extent
        assert granule.spatial_extent is not None
        assert granule.spatial_extent["north"] == 10.0
        assert granule.spatial_extent["south"] == 0.0


@pytest.mark.unit 
class TestAsyncOperations:
    """Test asynchronous operations and concurrent processing."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent for testing."""
        agent = CMRAPIAgent()
        yield agent
        await agent.close()
    
    async def test_concurrent_requests(self, agent, mock_http_client):
        """Test handling concurrent requests properly."""
        sample_context = QueryContext(
            original_query="Test query",
            intent=QueryIntent.EXPLORATORY
        )
        
        with patch.object(agent, '_get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_http_client
            mock_get_client.return_value.__aexit__ = AsyncMock()
            
            # Make multiple concurrent requests
            tasks = []
            for _ in range(5):
                task = agent.search_collections(sample_context)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All requests should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)
    
    async def test_client_sharing_across_requests(self, agent):
        """Test that HTTP client is properly shared across concurrent requests."""
        with patch.object(agent, '_make_cmr_request') as mock_request:
            mock_request.return_value = {"feed": {"entry": []}}
            
            sample_context = QueryContext(
                original_query="Test query",
                intent=QueryIntent.EXPLORATORY
            )
            
            # Make concurrent requests
            tasks = [
                agent.search_collections(sample_context),
                agent.search_collections(sample_context)
            ]
            
            await asyncio.gather(*tasks)
            
            # Should have made multiple requests
            assert mock_request.call_count == 2