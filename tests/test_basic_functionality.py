"""
Basic functionality tests to verify core system components work correctly.
These tests serve as smoke tests and foundation for the comprehensive test suite.
"""

import pytest
import asyncio
from datetime import datetime, timezone

from nasa_cmr_agent.models.schemas import (
    QueryIntent, SpatialConstraint, TemporalConstraint, QueryContext
)
from nasa_cmr_agent.core.config import settings


@pytest.mark.unit
class TestBasicFunctionality:
    """Basic functionality tests."""
    
    def test_configuration_loading(self):
        """Test that configuration loads correctly."""
        assert settings is not None
        assert settings.cmr_base_url is not None
        assert settings.cmr_base_url.startswith("http")
        
    def test_enum_values(self):
        """Test enum values are accessible."""
        assert QueryIntent.EXPLORATORY == "exploratory"
        assert QueryIntent.ANALYTICAL == "analytical"
        assert QueryIntent.COMPARATIVE == "comparative"
    
    def test_spatial_constraint_creation(self):
        """Test basic spatial constraint creation."""
        constraint = SpatialConstraint(
            north=45.0,
            south=30.0,
            east=-100.0,
            west=-120.0
        )
        assert constraint.north == 45.0
        assert constraint.south == 30.0
    
    def test_temporal_constraint_creation(self):
        """Test basic temporal constraint creation."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        constraint = TemporalConstraint(
            start_date=start,
            end_date=end
        )
        assert constraint.start_date == start
        assert constraint.end_date == end
    
    def test_query_context_creation(self):
        """Test basic query context creation."""
        context = QueryContext(
            original_query="Test query",
            intent=QueryIntent.EXPLORATORY
        )
        assert context.original_query == "Test query"
        assert context.intent == QueryIntent.EXPLORATORY
        assert context.priority_score == 1.0  # default value
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test that async functionality works."""
        async def simple_async_function():
            await asyncio.sleep(0.001)  # Very short sleep
            return "async_test_completed"
        
        result = await simple_async_function()
        assert result == "async_test_completed"
    
    def test_imports_work(self):
        """Test that all major imports work without errors."""
        try:
            from nasa_cmr_agent.core.config import settings
            from nasa_cmr_agent.models.schemas import QueryContext
            from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerService
            # If we get here without ImportError, imports work
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
class TestValidationLogic:
    """Test validation logic in models."""
    
    def test_spatial_validation_success(self):
        """Test successful spatial validation."""
        # Valid coordinates
        constraint = SpatialConstraint(
            north=45.0,
            south=30.0,
            east=-100.0,
            west=-120.0
        )
        # If no ValidationError is raised, test passes
        assert constraint.north > constraint.south
    
    def test_temporal_validation_success(self):
        """Test successful temporal validation."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        constraint = TemporalConstraint(
            start_date=start,
            end_date=end
        )
        # If no ValidationError is raised, test passes
        assert constraint.end_date > constraint.start_date


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])