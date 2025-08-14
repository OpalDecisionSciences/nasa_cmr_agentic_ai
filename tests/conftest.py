"""
Pytest configuration and shared fixtures for NASA CMR Agent test suite.

This module provides comprehensive test fixtures and configuration for unit,
integration, and performance testing with production-grade patterns.
"""

import pytest
import asyncio
import aiofiles
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from redis import asyncio as aioredis
import json

# Test imports
from nasa_cmr_agent.core.config import settings
from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
from nasa_cmr_agent.agents.query_interpreter import QueryInterpreterAgent
from nasa_cmr_agent.agents.enhanced_analysis_agent import EnhancedAnalysisAgent
from nasa_cmr_agent.agents.response_agent import ResponseSynthesisAgent
from nasa_cmr_agent.agents.supervisor_agent import SupervisorAgent
from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerService
from nasa_cmr_agent.services.monitoring import MetricsService
from nasa_cmr_agent.tools.scratchpad import ScratchpadManager
from nasa_cmr_agent.models.schemas import (
    QueryContext, QueryIntent, SpatialConstraint, TemporalConstraint,
    CMRCollection, CMRGranule, SystemResponse
)


# Test Configuration
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration with safe defaults."""
    return {
        "cmr_base_url": "https://cmr.earthdata.nasa.gov/search",
        "request_timeout": 10,
        "max_retries": 2,
        "rate_limit": 5,
        "redis_url": "redis://localhost:6379",
        "enable_metrics": False,
        "environment": "test",
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="nasa_cmr_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.hget.return_value = None
    mock_redis.hset.return_value = True
    mock_redis.hgetall.return_value = {}
    mock_redis.expire.return_value = True
    mock_redis.aclose = AsyncMock()
    return mock_redis


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API testing."""
    mock_client = AsyncMock()
    
    # Mock successful CMR response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "feed": {
            "entry": [
                {
                    "id": "C123456-TEST",
                    "title": "Test Collection",
                    "summary": "Test collection for unit testing",
                    "data_center": "TEST_CENTER",
                    "platforms": [{"ShortName": "TEST_PLATFORM"}],
                    "time_start": "2020-01-01T00:00:00Z",
                    "time_end": "2023-12-31T23:59:59Z",
                    "boxes": ["-90 -180 90 180"]
                }
            ]
        }
    }
    mock_response.raise_for_status.return_value = None
    
    mock_client.get.return_value = mock_response
    mock_client.aclose = AsyncMock()
    mock_client.is_closed = False
    
    return mock_client


@pytest.fixture
def sample_query_context():
    """Sample query context for testing."""
    return QueryContext(
        original_query="Find precipitation data for drought monitoring in Africa 2020-2023",
        intent=QueryIntent.ANALYTICAL,
        constraints={
            "spatial": SpatialConstraint(
                north=35.0,
                south=-35.0,
                east=50.0,
                west=-20.0,
                region_name="Africa"
            ),
            "temporal": TemporalConstraint(
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
            ),
            "keywords": ["precipitation", "drought", "monitoring"],
            "platforms": ["TRMM", "GPM"],
            "variables": ["precipitation_rate"]
        },
        priority_score=0.8,
        complexity_score=0.7
    )


@pytest.fixture
def sample_cmr_collection():
    """Sample CMR collection for testing."""
    return CMRCollection(
        concept_id="C123456-TEST",
        title="Test Precipitation Collection",
        summary="Test collection for precipitation data",
        data_center="TEST_CENTER",
        platforms=["TRMM"],
        instruments=["PR"],
        variables=["precipitation_rate"],
        temporal_extent={
            "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "end": datetime(2023, 12, 31, tzinfo=timezone.utc)
        },
        spatial_extent={
            "north": 90.0,
            "south": -90.0,
            "east": 180.0,
            "west": -180.0
        },
        version_id="1.0",
        processing_level="L3"
    )


@pytest.fixture
def sample_cmr_granule():
    """Sample CMR granule for testing."""
    return CMRGranule(
        concept_id="G123456-TEST",
        title="Test Granule",
        collection_concept_id="C123456-TEST",
        producer_granule_id="TEST_GRANULE_001",
        temporal_extent={
            "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "end": datetime(2020, 1, 2, tzinfo=timezone.utc)
        },
        spatial_extent={
            "north": 10.0,
            "south": 0.0,
            "east": 10.0,
            "west": 0.0
        },
        size_mb=150.5,
        links=[
            {
                "href": "https://example.com/data/test_granule.nc",
                "type": "GET DATA",
                "title": "Download Data"
            }
        ]
    )


# Mock external services for testing
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services to avoid dependencies during testing."""
    patches = []
    
    try:
        # Try to patch external dependencies that might not be available
        weaviate_patch = patch('weaviate.Client')
        neo4j_patch = patch('neo4j.GraphDatabase.driver')
        redis_patch = patch('redis.asyncio.from_url')
        
        patches = [weaviate_patch, neo4j_patch, redis_patch]
        
        mock_weaviate = weaviate_patch.start()
        mock_neo4j = neo4j_patch.start()
        mock_redis = redis_patch.start()
        
        # Configure mocks
        mock_weaviate.return_value = MagicMock()
        mock_neo4j.return_value = MagicMock()
        mock_redis.return_value = AsyncMock()
        
        yield {
            'weaviate': mock_weaviate,
            'neo4j': mock_neo4j,
            'redis': mock_redis
        }
        
    except Exception as e:
        # If patching fails, just provide empty mocks
        yield {
            'weaviate': MagicMock(),
            'neo4j': MagicMock(),
            'redis': AsyncMock()
        }
    finally:
        # Clean up patches
        for patch_obj in patches:
            try:
                patch_obj.stop()
            except:
                pass


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")


class TestDataFactory:
    """Factory for creating test data objects."""
    
    @staticmethod
    def create_query_context(
        query: str = "test query",
        intent: QueryIntent = QueryIntent.EXPLORATORY,
        **kwargs
    ) -> QueryContext:
        """Create test query context."""
        return QueryContext(
            original_query=query,
            intent=intent,
            constraints=kwargs.get('constraints', {}),
            priority_score=kwargs.get('priority_score', 0.5),
            complexity_score=kwargs.get('complexity_score', 0.5)
        )
    
    @staticmethod
    def create_cmr_collection(**kwargs) -> CMRCollection:
        """Create test CMR collection."""
        defaults = {
            "concept_id": "C123456-TEST",
            "title": "Test Collection",
            "summary": "Test collection",
            "data_center": "TEST_CENTER",
            "platforms": ["TEST_PLATFORM"],
            "instruments": ["TEST_INSTRUMENT"],
            "variables": ["test_variable"],
            "temporal_extent": {
                "start": datetime(2020, 1, 1, tzinfo=timezone.utc),
                "end": datetime(2023, 12, 31, tzinfo=timezone.utc)
            },
            "spatial_extent": {
                "north": 90.0, "south": -90.0,
                "east": 180.0, "west": -180.0
            }
        }
        defaults.update(kwargs)
        return CMRCollection(**defaults)


@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory