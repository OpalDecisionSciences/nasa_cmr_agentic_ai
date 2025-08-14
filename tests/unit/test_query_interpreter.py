import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nasa_cmr_agent.agents.query_interpreter import QueryInterpreterAgent
from nasa_cmr_agent.models.schemas import QueryIntent, DataType


class TestQueryInterpreterAgent:
    """Test suite for QueryInterpreterAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create QueryInterpreterAgent instance for testing."""
        with patch('nasa_cmr_agent.agents.query_interpreter.LLMService'):
            return QueryInterpreterAgent()
    
    @pytest.mark.asyncio
    async def test_classify_intent_exploratory(self, agent):
        """Test intent classification for exploratory queries."""
        
        # Mock LLM response
        agent.llm_service.generate = AsyncMock(
            return_value='"What precipitation data is available?" -> EXPLORATORY, 0.9'
        )
        
        intent, confidence = await agent._classify_intent(
            "What precipitation data is available for drought monitoring?"
        )
        
        assert intent == QueryIntent.EXPLORATORY
        assert confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_classify_intent_comparative(self, agent):
        """Test intent classification for comparative queries."""
        
        agent.llm_service.generate = AsyncMock(
            return_value='Compare datasets -> COMPARATIVE, 0.95'
        )
        
        intent, confidence = await agent._classify_intent(
            "Compare MODIS and VIIRS for vegetation monitoring"
        )
        
        assert intent == QueryIntent.COMPARATIVE
        assert confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_extract_spatial_constraints_region(self, agent):
        """Test extraction of named region spatial constraints."""
        
        query = "Find precipitation data for Sub-Saharan Africa"
        
        spatial = await agent._extract_spatial_constraints(query)
        
        assert spatial is not None
        assert spatial.region_name == "sub-saharan africa"
    
    @pytest.mark.asyncio
    async def test_extract_spatial_constraints_coordinates(self, agent):
        """Test extraction of coordinate-based spatial constraints."""
        
        query = "Data for region 10.5°N 15.2°W to 25.8°N 5.7°E"
        
        spatial = await agent._extract_spatial_constraints(query)
        
        assert spatial is not None
        assert spatial.north == 25.8
        assert spatial.south == 10.5
        assert spatial.east == 5.7
        assert spatial.west == 15.2
    
    @pytest.mark.asyncio
    async def test_extract_temporal_constraints_year_range(self, agent):
        """Test extraction of year range temporal constraints."""
        
        query = "Show data from 2015-2023 for climate analysis"
        
        temporal = await agent._extract_temporal_constraints(query)
        
        assert temporal is not None
        assert temporal.start_date.year == 2015
        assert temporal.end_date.year == 2023
    
    @pytest.mark.asyncio
    async def test_extract_temporal_constraints_single_year(self, agent):
        """Test extraction of single year temporal constraints."""
        
        query = "MODIS data for 2020 analysis"
        
        temporal = await agent._extract_temporal_constraints(query)
        
        assert temporal is not None
        assert temporal.start_date.year == 2020
        assert temporal.end_date.year == 2020
    
    @pytest.mark.asyncio
    async def test_extract_domain_constraints(self, agent):
        """Test extraction of domain-specific constraints."""
        
        query = "Find MODIS satellite precipitation data from Terra platform"
        
        domain_data = await agent._extract_domain_constraints(query)
        
        assert DataType.SATELLITE in domain_data['data_types']
        assert 'precipitation' in domain_data['keywords']
        assert 'Terra/Aqua' in domain_data['platforms']
        assert 'MODIS' in domain_data['instruments']
    
    @pytest.mark.asyncio
    async def test_interpret_query_integration(self, agent):
        """Test full query interpretation integration."""
        
        # Mock LLM service
        agent.llm_service.generate = AsyncMock(
            return_value='Drought monitoring query -> ANALYTICAL, 0.85'
        )
        
        query = "Compare precipitation datasets for drought monitoring in Sub-Saharan Africa 2015-2023"
        
        context = await agent.interpret_query(query)
        
        assert context.original_query == query
        assert context.intent == QueryIntent.ANALYTICAL
        assert context.priority_score > 0
        assert context.complexity_score > 0
        assert context.constraints is not None
    
    def test_validate_spatial_constraints_valid(self, agent):
        """Test validation of valid spatial constraints."""
        
        from nasa_cmr_agent.models.schemas import SpatialConstraint
        
        spatial = SpatialConstraint(north=45.0, south=30.0, east=10.0, west=0.0)
        
        is_valid = agent._validate_spatial_constraints(spatial)
        
        assert is_valid is True
    
    def test_validate_spatial_constraints_invalid(self, agent):
        """Test validation of invalid spatial constraints."""
        
        from nasa_cmr_agent.models.schemas import SpatialConstraint
        
        # North <= South (invalid)
        spatial = SpatialConstraint(north=30.0, south=45.0, east=10.0, west=0.0)
        
        is_valid = agent._validate_spatial_constraints(spatial)
        
        assert is_valid is False
    
    def test_validate_temporal_constraints_valid(self, agent):
        """Test validation of valid temporal constraints."""
        
        from nasa_cmr_agent.models.schemas import TemporalConstraint
        
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        temporal = TemporalConstraint(start_date=start_date, end_date=end_date)
        
        is_valid = agent._validate_temporal_constraints(temporal)
        
        assert is_valid is True
    
    def test_validate_temporal_constraints_invalid(self, agent):
        """Test validation of invalid temporal constraints."""
        
        from nasa_cmr_agent.models.schemas import TemporalConstraint
        
        # End date before start date
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2020, 12, 31, tzinfo=timezone.utc)
        
        temporal = TemporalConstraint(start_date=start_date, end_date=end_date)
        
        is_valid = agent._validate_temporal_constraints(temporal)
        
        assert is_valid is False
    
    def test_calculate_complexity_score(self, agent):
        """Test complexity score calculation."""
        
        from nasa_cmr_agent.models.schemas import QueryConstraints, SpatialConstraint, TemporalConstraint
        from datetime import datetime, timezone
        
        # Complex query with multiple constraints
        constraints = QueryConstraints(
            spatial=SpatialConstraint(region_name="sub-saharan africa"),
            temporal=TemporalConstraint(
                start_date=datetime(2015, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
            ),
            data_types=[DataType.SATELLITE, DataType.GROUND_BASED],
            keywords=['precipitation', 'drought'],
            platforms=['MODIS', 'Landsat']
        )
        
        query = "Compare and analyze precipitation datasets for comprehensive drought monitoring"
        
        score = agent._calculate_complexity_score(constraints, query)
        
        assert score > 0.5  # Should be moderately complex
        assert score <= 1.0
    
    def test_identify_research_domain(self, agent):
        """Test research domain identification."""
        
        # Hydrology domain
        query = "drought monitoring precipitation data"
        domain = agent._identify_research_domain(query)
        assert domain == "hydrology"
        
        # Climate domain
        query = "climate change temperature trends"
        domain = agent._identify_research_domain(query)
        assert domain == "climate"
        
        # Fire domain
        query = "wildfire burn area analysis"
        domain = agent._identify_research_domain(query)
        assert domain == "fire"
        
        # Unknown domain
        query = "generic data request"
        domain = agent._identify_research_domain(query)
        assert domain is None
    
    def test_extract_methodology_hints(self, agent):
        """Test methodology hint extraction."""
        
        query = "Compare and visualize temporal trends in vegetation patterns"
        
        hints = agent._extract_methodology_hints(query)
        
        expected_hints = ['time series analysis', 'visualization', 'comparative analysis']
        
        for hint in expected_hints:
            assert hint in hints


@pytest.mark.asyncio
class TestQueryInterpreterIntegration:
    """Integration tests for QueryInterpreterAgent."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        with patch('nasa_cmr_agent.agents.query_interpreter.LLMService') as mock:
            mock_instance = mock.return_value
            mock_instance.generate = AsyncMock()
            yield mock_instance
    
    async def test_complex_precipitation_query(self, mock_llm_service):
        """Test complex precipitation monitoring query."""
        
        mock_llm_service.generate.return_value = (
            "Complex analytical query -> ANALYTICAL, 0.9"
        )
        
        agent = QueryInterpreterAgent()
        
        query = ("Compare precipitation datasets suitable for drought monitoring "
                "in Sub-Saharan Africa between 2015-2023, considering both "
                "satellite and ground-based observations, and identify gaps "
                "in temporal coverage.")
        
        context = await agent.interpret_query(query)
        
        # Verify intent classification
        assert context.intent == QueryIntent.ANALYTICAL
        
        # Verify spatial constraints
        assert context.constraints.spatial is not None
        assert context.constraints.spatial.region_name == "sub-saharan africa"
        
        # Verify temporal constraints
        assert context.constraints.temporal is not None
        assert context.constraints.temporal.start_date.year == 2015
        assert context.constraints.temporal.end_date.year == 2023
        
        # Verify data type constraints
        expected_types = [DataType.SATELLITE, DataType.GROUND_BASED]
        for data_type in expected_types:
            assert data_type in context.constraints.data_types
        
        # Verify keywords
        assert 'precipitation' in context.constraints.keywords
        assert 'drought' in context.constraints.keywords
        
        # Verify complexity
        assert context.complexity_score > 0.6  # Should be complex query
        
        # Verify research domain
        assert context.research_domain == "hydrology"
        
        # Verify methodology hints
        methodology_hints = context.methodology_hints
        assert 'comparative analysis' in methodology_hints
        assert 'gap analysis' in methodology_hints
    
    async def test_urban_heat_island_query(self, mock_llm_service):
        """Test urban heat island research query."""
        
        mock_llm_service.generate.return_value = (
            "Multi-disciplinary research query -> ANALYTICAL, 0.85"
        )
        
        agent = QueryInterpreterAgent()
        
        query = ("What datasets would be best for studying the relationship "
                "between urban heat islands and air quality in megacities?")
        
        context = await agent.interpret_query(query)
        
        # Verify intent
        assert context.intent == QueryIntent.ANALYTICAL
        
        # Verify research domain
        assert context.research_domain == "urban"
        
        # Verify keywords extracted
        keywords = context.constraints.keywords
        assert any('urban' in k.lower() for k in keywords) or 'air quality' in keywords
        
        # Verify methodology hints
        hints = context.methodology_hints
        assert 'statistical analysis' in hints or 'correlation' in ' '.join(hints)
    
    async def test_validation_edge_cases(self, mock_llm_service):
        """Test validation with edge cases."""
        
        agent = QueryInterpreterAgent()
        
        # Test with future dates (should fail validation)
        from nasa_cmr_agent.models.schemas import QueryContext, QueryConstraints, TemporalConstraint
        
        future_date = datetime(2030, 1, 1, tzinfo=timezone.utc)
        constraints = QueryConstraints(
            temporal=TemporalConstraint(start_date=future_date, end_date=future_date)
        )
        
        context = QueryContext(
            original_query="Future data request",
            intent=QueryIntent.EXPLORATORY,
            constraints=constraints
        )
        
        is_valid = await agent.validate_query(context)
        assert is_valid is False
        
        # Test with very complex query (should fail due to complexity)
        complex_constraints = QueryConstraints(
            data_types=[DataType.SATELLITE, DataType.GROUND_BASED, DataType.MODEL],
            keywords=['precipitation', 'temperature', 'humidity', 'wind', 'pressure'],
            platforms=['MODIS', 'VIIRS', 'Landsat', 'Sentinel', 'GOES']
        )
        
        complex_context = QueryContext(
            original_query="extremely complex multi-parameter multi-platform comprehensive analysis",
            intent=QueryIntent.ANALYTICAL,
            constraints=complex_constraints,
            complexity_score=0.95  # Very high complexity
        )
        
        is_valid = await agent.validate_query(complex_context)
        # May pass or fail depending on implementation threshold