"""
Comprehensive unit tests for data models and schemas.

Tests all Pydantic models, validation logic, and data transformations
with edge cases and error conditions.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from nasa_cmr_agent.models.schemas import (
    QueryIntent, DataType, SpatialConstraint, TemporalConstraint,
    QueryConstraints, QueryContext, CMRCollection, CMRGranule,
    SystemResponse, DatasetRecommendation, AnalysisResult
)


@pytest.mark.unit
class TestSpatialConstraint:
    """Test spatial constraint validation and functionality."""
    
    def test_valid_spatial_constraint(self):
        """Test creation of valid spatial constraint."""
        constraint = SpatialConstraint(
            north=45.0,
            south=30.0,
            east=-100.0,
            west=-120.0,
            region_name="Test Region"
        )
        
        assert constraint.north == 45.0
        assert constraint.south == 30.0
        assert constraint.east == -100.0
        assert constraint.west == -120.0
        assert constraint.region_name == "Test Region"
    
    def test_spatial_constraint_validation_north_south(self):
        """Test validation that north > south."""
        with pytest.raises(ValidationError) as exc_info:
            SpatialConstraint(north=30.0, south=45.0)
        
        assert "North latitude must be greater than south latitude" in str(exc_info.value)
    
    def test_spatial_constraint_equal_coordinates(self):
        """Test validation with equal north/south coordinates."""
        with pytest.raises(ValidationError) as exc_info:
            SpatialConstraint(north=45.0, south=45.0)
        
        assert "North latitude must be greater than south latitude" in str(exc_info.value)
    
    def test_spatial_constraint_boundary_values(self):
        """Test boundary latitude/longitude values."""
        # Valid boundary values
        constraint = SpatialConstraint(
            north=90.0,
            south=-90.0,
            east=180.0,
            west=-180.0
        )
        assert constraint.north == 90.0
        assert constraint.south == -90.0
    
    def test_spatial_constraint_invalid_latitude(self):
        """Test invalid latitude values."""
        with pytest.raises(ValidationError):
            SpatialConstraint(north=100.0)  # > 90
        
        with pytest.raises(ValidationError):
            SpatialConstraint(south=-100.0)  # < -90
    
    def test_spatial_constraint_invalid_longitude(self):
        """Test invalid longitude values."""
        with pytest.raises(ValidationError):
            SpatialConstraint(east=200.0)  # > 180
        
        with pytest.raises(ValidationError):
            SpatialConstraint(west=-200.0)  # < -180
    
    def test_spatial_constraint_partial_data(self):
        """Test constraint with only some coordinates."""
        constraint = SpatialConstraint(
            north=45.0,
            region_name="Test Region"
        )
        assert constraint.north == 45.0
        assert constraint.south is None
        assert constraint.region_name == "Test Region"


@pytest.mark.unit
class TestTemporalConstraint:
    """Test temporal constraint validation and functionality."""
    
    def test_valid_temporal_constraint(self):
        """Test creation of valid temporal constraint."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        constraint = TemporalConstraint(
            start_date=start,
            end_date=end,
            temporal_resolution="daily"
        )
        
        assert constraint.start_date == start
        assert constraint.end_date == end
        assert constraint.temporal_resolution == "daily"
    
    def test_temporal_constraint_validation_dates(self):
        """Test validation that end_date > start_date."""
        start = datetime(2023, 12, 31, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        with pytest.raises(ValidationError) as exc_info:
            TemporalConstraint(start_date=start, end_date=end)
        
        assert "End date must be after start date" in str(exc_info.value)
    
    def test_temporal_constraint_equal_dates(self):
        """Test validation with equal start/end dates."""
        date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        with pytest.raises(ValidationError) as exc_info:
            TemporalConstraint(start_date=date, end_date=date)
        
        assert "End date must be after start date" in str(exc_info.value)
    
    def test_temporal_constraint_partial_data(self):
        """Test constraint with only start or end date."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        
        constraint = TemporalConstraint(start_date=start)
        assert constraint.start_date == start
        assert constraint.end_date is None
    
    def test_temporal_constraint_resolution_only(self):
        """Test constraint with only temporal resolution."""
        constraint = TemporalConstraint(temporal_resolution="monthly")
        assert constraint.temporal_resolution == "monthly"
        assert constraint.start_date is None
        assert constraint.end_date is None


@pytest.mark.unit
class TestQueryContext:
    """Test query context model and validation."""
    
    def test_valid_query_context(self):
        """Test creation of valid query context."""
        spatial = SpatialConstraint(north=45.0, south=30.0)
        temporal = TemporalConstraint(
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
        )
        
        context = QueryContext(
            original_query="Find precipitation data for Africa",
            intent=QueryIntent.ANALYTICAL,
            constraints={
                "spatial": spatial,
                "temporal": temporal,
                "keywords": ["precipitation", "Africa"],
                "platforms": ["TRMM"]
            },
            priority_score=0.8,
            complexity_score=0.6
        )
        
        assert context.original_query == "Find precipitation data for Africa"
        assert context.intent == QueryIntent.ANALYTICAL
        assert context.priority_score == 0.8
        assert context.complexity_score == 0.6
        assert "spatial" in context.constraints
        assert "temporal" in context.constraints
    
    def test_query_context_minimal(self):
        """Test query context with minimal required fields."""
        context = QueryContext(
            original_query="Test query",
            intent=QueryIntent.EXPLORATORY
        )
        
        assert context.original_query == "Test query"
        assert context.intent == QueryIntent.EXPLORATORY
        assert context.constraints == {}
        assert context.priority_score == 1.0  # default
        assert context.complexity_score == 0.5  # default
    
    def test_query_context_score_validation(self):
        """Test validation of priority and complexity scores."""
        # Valid scores (0-1 range)
        context = QueryContext(
            original_query="Test",
            intent=QueryIntent.EXPLORATORY,
            priority_score=0.0,
            complexity_score=1.0
        )
        assert context.priority_score == 0.0
        assert context.complexity_score == 1.0
        
        # Invalid scores should still work but might be outside expected range
        context = QueryContext(
            original_query="Test",
            intent=QueryIntent.EXPLORATORY,
            priority_score=1.5
        )
        assert context.priority_score == 1.5


@pytest.mark.unit
class TestCMRCollection:
    """Test CMR collection model."""
    
    def test_valid_cmr_collection(self):
        """Test creation of valid CMR collection."""
        collection = CMRCollection(
            concept_id="C123456-TEST",
            title="Test Collection",
            summary="Test collection summary",
            data_center="TEST_CENTER",
            platforms=["TRMM", "GPM"],
            instruments=["PR", "TMI"],
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
        
        assert collection.concept_id == "C123456-TEST"
        assert collection.title == "Test Collection"
        assert len(collection.platforms) == 2
        assert "precipitation_rate" in collection.variables
        assert collection.processing_level == "L3"
    
    def test_cmr_collection_minimal(self):
        """Test CMR collection with minimal required fields."""
        collection = CMRCollection(
            concept_id="C123456-TEST",
            title="Test Collection",
            summary="Test summary",
            data_center="TEST_CENTER"
        )
        
        assert collection.concept_id == "C123456-TEST"
        assert collection.platforms == []  # default empty list
        assert collection.instruments == []
        assert collection.variables == []


@pytest.mark.unit
class TestCMRGranule:
    """Test CMR granule model."""
    
    def test_valid_cmr_granule(self):
        """Test creation of valid CMR granule."""
        granule = CMRGranule(
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
                    "href": "https://example.com/data.nc",
                    "type": "GET DATA",
                    "title": "Download"
                }
            ]
        )
        
        assert granule.concept_id == "G123456-TEST"
        assert granule.collection_concept_id == "C123456-TEST"
        assert granule.size_mb == 150.5
        assert len(granule.links) == 1
    
    def test_cmr_granule_minimal(self):
        """Test CMR granule with minimal required fields."""
        granule = CMRGranule(
            concept_id="G123456-TEST",
            title="Test Granule",
            collection_concept_id="C123456-TEST"
        )
        
        assert granule.concept_id == "G123456-TEST"
        assert granule.links == []  # default empty list
        assert granule.size_mb is None


@pytest.mark.unit
class TestDatasetRecommendation:
    """Test dataset recommendation model."""
    
    def test_valid_recommendation(self):
        """Test creation of valid recommendation."""
        collection = CMRCollection(
            concept_id="C123456-TEST",
            title="Test Collection",
            summary="Test",
            data_center="TEST_CENTER"
        )
        
        recommendation = DatasetRecommendation(
            collection=collection,
            relevance_score=0.9,
            coverage_score=0.8,
            quality_score=0.85,
            accessibility_score=0.95,
            reasoning="High relevance for drought monitoring",
            granule_count=500,
            temporal_gaps=[],
            complementary_datasets=["C789-OTHER"]
        )
        
        assert recommendation.relevance_score == 0.9
        assert recommendation.coverage_score == 0.8
        assert recommendation.quality_score == 0.85
        assert recommendation.accessibility_score == 0.95
        assert recommendation.granule_count == 500
        assert len(recommendation.complementary_datasets) == 1
    
    def test_recommendation_score_bounds(self):
        """Test recommendation scores are within expected bounds."""
        collection = CMRCollection(
            concept_id="C123456-TEST",
            title="Test Collection",
            summary="Test",
            data_center="TEST_CENTER"
        )
        
        # Test with boundary values
        recommendation = DatasetRecommendation(
            collection=collection,
            relevance_score=1.0,
            coverage_score=0.0,
            quality_score=0.5,
            accessibility_score=1.0,
            reasoning="Test reasoning"
        )
        
        assert recommendation.relevance_score == 1.0
        assert recommendation.coverage_score == 0.0


@pytest.mark.unit
class TestSystemResponse:
    """Test system response model."""
    
    def test_valid_system_response(self):
        """Test creation of valid system response."""
        collection = CMRCollection(
            concept_id="C123456-TEST",
            title="Test Collection", 
            summary="Test",
            data_center="TEST_CENTER"
        )
        
        recommendation = DatasetRecommendation(
            collection=collection,
            relevance_score=0.9,
            coverage_score=0.8,
            quality_score=0.85,
            accessibility_score=0.95,
            reasoning="Test reasoning"
        )
        
        response = SystemResponse(
            query_id="test_query_123",
            original_query="Find precipitation data",
            intent=QueryIntent.ANALYTICAL,
            recommendations=[recommendation],
            summary="Found 1 relevant dataset for precipitation analysis",
            execution_plan=["Parse query", "Search collections", "Generate recommendations"],
            total_execution_time_ms=1500,
            success=True
        )
        
        assert response.query_id == "test_query_123"
        assert response.success is True
        assert len(response.recommendations) == 1
        assert response.total_execution_time_ms == 1500
        assert len(response.execution_plan) == 3
    
    def test_system_response_with_errors(self):
        """Test system response with error conditions."""
        response = SystemResponse(
            query_id="error_query_123",
            original_query="Invalid query",
            intent=QueryIntent.EXPLORATORY,
            recommendations=[],
            summary="Query processing failed",
            execution_plan=["Parse query", "Error occurred"],
            total_execution_time_ms=500,
            success=False,
            warnings=["Invalid spatial constraints"],
            errors=["CMR API timeout"]
        )
        
        assert response.success is False
        assert len(response.warnings) == 1
        assert len(response.errors) == 1
        assert "Invalid spatial constraints" in response.warnings
        assert "CMR API timeout" in response.errors


@pytest.mark.unit
class TestEnumValidation:
    """Test enum types and validation."""
    
    def test_query_intent_enum(self):
        """Test QueryIntent enum values."""
        assert QueryIntent.EXPLORATORY == "exploratory"
        assert QueryIntent.SPECIFIC_DATA == "specific_data"
        assert QueryIntent.ANALYTICAL == "analytical"
        assert QueryIntent.COMPARATIVE == "comparative"
        assert QueryIntent.TEMPORAL_ANALYSIS == "temporal_analysis"
        assert QueryIntent.SPATIAL_ANALYSIS == "spatial_analysis"
    
    def test_data_type_enum(self):
        """Test DataType enum values."""
        assert DataType.SATELLITE == "satellite"
        assert DataType.GROUND_BASED == "ground_based"
        assert DataType.MODEL == "model"
        assert DataType.HYBRID == "hybrid"
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values."""
        with pytest.raises(ValidationError):
            QueryContext(
                original_query="Test",
                intent="invalid_intent"  # This should fail validation
            )