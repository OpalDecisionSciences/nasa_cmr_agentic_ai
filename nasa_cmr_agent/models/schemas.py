from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class QueryIntent(str, Enum):
    """Types of user query intents."""
    EXPLORATORY = "exploratory"
    SPECIFIC_DATA = "specific_data"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    SPATIAL_ANALYSIS = "spatial_analysis"


class DataType(str, Enum):
    """Types of NASA data."""
    SATELLITE = "satellite"
    GROUND_BASED = "ground_based"
    MODEL = "model"
    HYBRID = "hybrid"


class SpatialConstraint(BaseModel):
    """Spatial boundary constraints for data queries."""
    north: Optional[float] = Field(None, ge=-90, le=90, description="Northern latitude boundary")
    south: Optional[float] = Field(None, ge=-90, le=90, description="Southern latitude boundary")
    east: Optional[float] = Field(None, ge=-180, le=180, description="Eastern longitude boundary")
    west: Optional[float] = Field(None, ge=-180, le=180, description="Western longitude boundary")
    region_name: Optional[str] = Field(None, description="Named geographical region")
    
    @validator('north')
    def validate_north_south(cls, v, values):
        if v is not None and 'south' in values and values['south'] is not None:
            if v <= values['south']:
                raise ValueError('North latitude must be greater than south latitude')
        return v


class TemporalConstraint(BaseModel):
    """Temporal boundary constraints for data queries."""
    start_date: Optional[datetime] = Field(None, description="Start date/time for data search")
    end_date: Optional[datetime] = Field(None, description="End date/time for data search")
    temporal_resolution: Optional[str] = Field(None, description="Required temporal resolution")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v is not None and 'start_date' in values and values['start_date'] is not None:
            if v <= values['start_date']:
                raise ValueError('End date must be after start date')
        return v


class QueryConstraints(BaseModel):
    """Complete set of constraints for a data query."""
    spatial: Optional[SpatialConstraint] = None
    temporal: Optional[TemporalConstraint] = None
    data_types: List[DataType] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=list)
    resolution_requirements: Optional[Dict[str, Any]] = None


class QueryContext(BaseModel):
    """Context and metadata for processing user queries."""
    original_query: str
    intent: QueryIntent
    constraints: QueryConstraints
    priority_score: float = Field(default=1.0, ge=0.0, le=1.0)
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    research_domain: Optional[str] = None
    methodology_hints: List[str] = Field(default_factory=list)


class CMRCollection(BaseModel):
    """NASA CMR Collection metadata."""
    concept_id: str
    short_name: str
    version_id: str
    title: str
    summary: str
    data_center: str
    platforms: List[str] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    temporal_coverage: Optional[Dict[str, Any]] = None
    spatial_coverage: Optional[Dict[str, Any]] = None
    variables: List[str] = Field(default_factory=list)
    processing_level: Optional[str] = None
    cloud_hosted: bool = False
    online_access_flag: bool = False


class CMRGranule(BaseModel):
    """NASA CMR Granule metadata."""
    concept_id: str
    title: str
    collection_concept_id: str
    producer_granule_id: str
    temporal_extent: Optional[Dict[str, Any]] = None
    spatial_extent: Optional[Dict[str, Any]] = None
    size_mb: Optional[float] = None
    cloud_cover: Optional[float] = None
    day_night_flag: Optional[str] = None
    links: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetRecommendation(BaseModel):
    """Recommendation for a specific dataset."""
    collection: CMRCollection
    relevance_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    accessibility_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    granule_count: Optional[int] = None
    temporal_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    complementary_datasets: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Results from data analysis operations."""
    analysis_type: str
    results: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    methodology: str
    confidence_level: float = Field(ge=0.0, le=1.0)


class AgentResponse(BaseModel):
    """Response from a specific agent in the system."""
    agent_name: str
    status: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemResponse(BaseModel):
    """Final system response to user query."""
    query_id: str
    original_query: str
    intent: QueryIntent
    recommendations: List[DatasetRecommendation]
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    summary: str
    execution_plan: List[str]
    total_execution_time_ms: int
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    success: bool = True
    warnings: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata including quality scores")