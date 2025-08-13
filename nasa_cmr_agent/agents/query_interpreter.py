import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..models.schemas import (
    QueryContext, QueryIntent, QueryConstraints, 
    SpatialConstraint, TemporalConstraint, DataType
)
from ..core.config import settings
from ..services.llm_service import LLMService
from ..tools.scratchpad import scratchpad_manager, NoteType
from .enhanced_query_decomposer import EnhancedQueryDecomposer


class QueryInterpreterAgent:
    """
    Intelligent query interpretation and validation agent.
    
    Responsibilities:
    - Parse natural language queries into structured QueryContext
    - Identify user intent and extract constraints
    - Validate query feasibility and suggest alternatives
    - Maintain context awareness across conversation turns
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.scratchpad = None  # Will be initialized async
        self.query_decomposer = EnhancedQueryDecomposer()
        self._setup_prompts()
        
        # Domain knowledge for validation
        self.valid_regions = {
            "sub-saharan africa", "sahel", "amazon", "arctic", "antarctica",
            "mediterranean", "great plains", "sahara", "himalayas", "andes",
            "pacific northwest", "gulf of mexico", "caribbean", "north america",
            "south america", "europe", "asia", "africa", "australia", "oceania"
        }
        
        self.common_variables = {
            "precipitation", "temperature", "humidity", "wind", "pressure",
            "sea surface temperature", "chlorophyll", "aerosols", "ozone",
            "carbon dioxide", "methane", "vegetation indices", "snow cover",
            "ice extent", "soil moisture", "evapotranspiration", "albedo",
            "fire activity", "air quality", "drought indices", "flood risk"
        }
    
    def _setup_prompts(self):
        """Setup LLM prompts for query interpretation."""
        self.intent_classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert NASA Earth science data analyst. Classify user queries into one of these intents:

EXPLORATORY: General discovery, "what data exists for X?"
SPECIFIC_DATA: Requesting particular datasets, "get MODIS data for..."
ANALYTICAL: Research questions requiring analysis, "compare trends..."
COMPARATIVE: Comparing datasets/regions/time periods OR discovering and comparing categories of datasets
TEMPORAL_ANALYSIS: Focus on time-series or temporal patterns
SPATIAL_ANALYSIS: Focus on geographic patterns or regions

IMPORTANT: COMPARATIVE includes both:
1. Direct comparison: "Compare MODIS vs VIIRS"
2. Discovery-then-compare: "Compare precipitation datasets suitable for drought monitoring" (find all suitable datasets, then compare them)

Keywords indicating COMPARATIVE intent:
- "compare", "versus", "vs", "between", "contrast", "difference"
- "suitable for" + comparison context
- Multiple data types mentioned: "satellite and ground-based"
- Evaluation contexts: "best", "most suitable", "optimal"

Consider the scientific domain and methodology hints in the query.
Respond with just the intent classification and confidence (0-1).

Examples:
"What precipitation data is available?" -> EXPLORATORY, 0.9
"Compare MODIS and VIIRS for drought monitoring" -> COMPARATIVE, 0.95
"Compare precipitation datasets suitable for drought monitoring" -> COMPARATIVE, 0.9
"Show temperature trends in Arctic 2000-2020" -> TEMPORAL_ANALYSIS, 0.9"""),
            ("human", "{query}")
        ])
        
        self.constraint_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract spatial, temporal, and domain constraints from the query.

Spatial constraints:
- Named regions (Sub-Saharan Africa, Pacific Northwest, etc.)
- Coordinates if provided
- Geographic features (coastal, mountainous, urban, etc.)

Temporal constraints:
- Date ranges (2015-2023, last decade, etc.)
- Temporal resolution needs (daily, monthly, annual)
- Specific seasons or periods

Domain constraints:
- Data types: satellite, ground-based, model, hybrid
- Variables: precipitation, temperature, vegetation, etc.
- Platforms/instruments: MODIS, Landsat, GOES, etc.
- Processing levels or quality requirements
- Research purpose or application context

Return as structured JSON with null for missing constraints."""),
            ("human", "{query}")
        ])
    
    async def interpret_query(self, user_query: str) -> QueryContext:
        """
        Interpret natural language query into structured QueryContext.
        
        Args:
            user_query: User's natural language query
            
        Returns:
            QueryContext with extracted intent and constraints
        """
        # Initialize scratchpad if needed
        if not self.scratchpad:
            self.scratchpad = await scratchpad_manager.get_scratchpad("query_interpreter")
        
        # Log query interpretation start
        await self.scratchpad.add_note(
            f"Starting query interpretation: '{user_query[:100]}...'",
            NoteType.OBSERVATION,
            {"query_length": len(user_query), "timestamp": datetime.now().isoformat()}
        )
        
        # Classify intent
        intent, confidence = await self._classify_intent(user_query)
        
        await self.scratchpad.add_note(
            f"Intent classified as {intent} with confidence {confidence}",
            NoteType.DECISION,
            {"intent": intent, "confidence": confidence}
        )
        
        # Extract constraints in parallel
        constraint_tasks = [
            self._extract_spatial_constraints(user_query),
            self._extract_temporal_constraints(user_query),
            self._extract_domain_constraints(user_query)
        ]
        
        spatial_constraints, temporal_constraints, domain_data = await asyncio.gather(
            *constraint_tasks
        )
        
        # Build complete constraints
        constraints = QueryConstraints(
            spatial=spatial_constraints,
            temporal=temporal_constraints,
            **domain_data
        )
        
        # Calculate complexity and priority scores
        complexity_score = self._calculate_complexity_score(constraints, user_query)
        priority_score = confidence
        
        # Extract research domain and methodology hints
        research_domain = self._identify_research_domain(user_query)
        methodology_hints = self._extract_methodology_hints(user_query)
        
        return QueryContext(
            original_query=user_query,
            intent=intent,
            constraints=constraints,
            priority_score=priority_score,
            complexity_score=complexity_score,
            research_domain=research_domain,
            methodology_hints=methodology_hints
        )
    
    async def validate_query(self, query_context: QueryContext) -> bool:
        """
        Validate query feasibility and constraints.
        
        Args:
            query_context: Parsed query context to validate
            
        Returns:
            True if query is valid and feasible
        """
        constraints = query_context.constraints
        
        # Validate spatial constraints
        if constraints.spatial:
            if not self._validate_spatial_constraints(constraints.spatial):
                return False
        
        # Validate temporal constraints
        if constraints.temporal:
            if not self._validate_temporal_constraints(constraints.temporal):
                return False
        
        # Validate data availability for constraints
        if not await self._check_data_availability(constraints):
            return False
        
        # Allow high complexity for comparative and analytical queries
        # Only reject if extremely high complexity (above 1.2)
        if query_context.complexity_score > 1.2:
            return False
        
        return True
    
    async def get_decomposed_queries(self, user_query: str, query_context: QueryContext):
        """
        Get decomposed sub-queries for complex queries.
        
        Args:
            user_query: Original user query
            query_context: Already interpreted query context
            
        Returns:
            Tuple of (strategy, sub_queries)
        """
        await self.scratchpad.add_note(
            f"Decomposing complex query for parallel execution",
            NoteType.DECISION,
            {"original_query": user_query}
        )
        
        strategy, sub_queries = await self.query_decomposer.decompose_query(user_query, query_context)
        
        await self.scratchpad.add_note(
            f"Query decomposed into {len(sub_queries)} sub-queries using {strategy.value} strategy",
            NoteType.SUCCESS,
            {
                "strategy": strategy.value,
                "sub_query_count": len(sub_queries),
                "sub_query_ids": [sq.id for sq in sub_queries]
            }
        )
        
        return strategy, sub_queries
    
    async def _classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Hybrid intent classification using both rule-based and LLM approaches."""
        # Get rule-based classification first
        rule_based_intent, rule_confidence = self._classify_intent_rule_based(query)
        
        try:
            # Get LLM classification
            response = await self.llm_service.generate(
                self.intent_classification_prompt.format(query=query),
                max_tokens=50
            )
            
            # Parse LLM response
            llm_intent, llm_confidence = self._parse_llm_intent_response(response)
            
            # Hybrid decision logic
            if rule_confidence > 0.8:
                # High confidence rule-based, use it
                return rule_based_intent, rule_confidence
            elif llm_confidence > 0.8 and llm_intent == rule_based_intent:
                # Both agree with high confidence
                return llm_intent, min(llm_confidence + 0.1, 1.0)
            elif llm_confidence > rule_confidence:
                # LLM more confident, use LLM result
                return llm_intent, llm_confidence
            else:
                # Use rule-based as fallback
                return rule_based_intent, rule_confidence
            
        except Exception:
            # LLM failed, use rule-based
            return rule_based_intent, rule_confidence
    
    def _classify_intent_rule_based(self, query: str) -> tuple[QueryIntent, float]:
        """Rule-based intent classification for reliability and nuanced comparative understanding."""
        query_lower = query.lower()
        
        # Comparative indicators (highest priority for nuanced understanding)
        comparative_keywords = [
            "compare", "versus", "vs", "between", "contrast", "difference"
        ]
        
        # Discovery-then-compare indicators (sophisticated comparative queries)
        discovery_compare_indicators = [
            "suitable for", "best for", "optimal for", "most appropriate",
            "datasets for", "data for", "options for"
        ]
        
        # Multi-data-type indicators
        multi_type_indicators = [
            "satellite and ground", "ground and satellite", "multiple datasets",
            "different data types", "various sources", "both satellite and ground"
        ]
        
        # Check for comparative intent
        has_comparative = any(kw in query_lower for kw in comparative_keywords)
        has_discovery_compare = any(indicator in query_lower for indicator in discovery_compare_indicators)
        has_multi_type = any(indicator in query_lower for indicator in multi_type_indicators)
        
        # Sophisticated comparative query detection
        if has_comparative and (has_discovery_compare or has_multi_type):
            # This is a nuanced "discover then compare" query
            return QueryIntent.COMPARATIVE, 0.95
        elif has_comparative:
            # Direct comparison query
            return QueryIntent.COMPARATIVE, 0.9
        elif has_multi_type and has_discovery_compare:
            # Implied comparison through multiple data types
            return QueryIntent.COMPARATIVE, 0.8
        
        # Temporal analysis indicators
        temporal_keywords = [
            "trend", "time series", "temporal", "over time", "seasonal",
            "annual", "monthly", "daily", "historical", "time period",
            "gaps", "coverage", "temporal coverage"
        ]
        if any(kw in query_lower for kw in temporal_keywords):
            return QueryIntent.TEMPORAL_ANALYSIS, 0.8
        
        # Spatial analysis indicators  
        spatial_keywords = [
            "spatial", "geographic", "regional", "area", "extent",
            "distribution", "pattern", "mapping"
        ]
        if any(kw in query_lower for kw in spatial_keywords):
            return QueryIntent.SPATIAL_ANALYSIS, 0.8
        
        # Specific data indicators
        specific_keywords = [
            "get", "download", "access", "retrieve", "obtain"
        ]
        instrument_keywords = [
            "modis", "landsat", "viirs", "sentinel", "goes", "avhrr"
        ]
        if (any(kw in query_lower for kw in specific_keywords) or 
            any(kw in query_lower for kw in instrument_keywords)):
            return QueryIntent.SPECIFIC_DATA, 0.7
        
        # Analytical indicators
        analytical_keywords = [
            "analyze", "analysis", "study", "research", "investigate",
            "correlation", "relationship", "impact", "effect"
        ]
        if any(kw in query_lower for kw in analytical_keywords):
            return QueryIntent.ANALYTICAL, 0.7
        
        # Default to exploratory with low confidence
        return QueryIntent.EXPLORATORY, 0.5
    
    def _parse_llm_intent_response(self, response: str) -> tuple[QueryIntent, float]:
        """Parse LLM intent classification response."""
        lines = response.strip().split('\n')
        for line in lines:
            if '->' in line:
                parts = line.split('->')
                if len(parts) >= 2:
                    intent_str = parts[1].split(',')[0].strip()
                    try:
                        confidence = float(parts[1].split(',')[1].strip())
                    except (IndexError, ValueError):
                        confidence = 0.5
                    
                    # Map to enum
                    intent_mapping = {
                        'EXPLORATORY': QueryIntent.EXPLORATORY,
                        'SPECIFIC_DATA': QueryIntent.SPECIFIC_DATA,
                        'ANALYTICAL': QueryIntent.ANALYTICAL,
                        'COMPARATIVE': QueryIntent.COMPARATIVE,
                        'TEMPORAL_ANALYSIS': QueryIntent.TEMPORAL_ANALYSIS,
                        'SPATIAL_ANALYSIS': QueryIntent.SPATIAL_ANALYSIS
                    }
                    
                    return intent_mapping.get(intent_str, QueryIntent.EXPLORATORY), confidence
        
        return QueryIntent.EXPLORATORY, 0.5
    
    async def _extract_spatial_constraints(self, query: str) -> Optional[SpatialConstraint]:
        """Extract spatial constraints from query."""
        # Look for named regions
        query_lower = query.lower()
        
        for region in self.valid_regions:
            if region in query_lower:
                return SpatialConstraint(region_name=region)
        
        # Look for coordinate patterns
        coord_pattern = r'(-?\d+(?:\.\d+)?)\s*[°,]\s*(-?\d+(?:\.\d+)?)'
        coords = re.findall(coord_pattern, query)
        
        if coords and len(coords) >= 2:
            # Assume first two coordinate pairs are bounding box
            try:
                lat1, lon1 = float(coords[0][0]), float(coords[0][1])
                lat2, lon2 = float(coords[1][0]), float(coords[1][1])
                
                return SpatialConstraint(
                    north=max(lat1, lat2),
                    south=min(lat1, lat2),
                    east=max(lon1, lon2),
                    west=min(lon1, lon2)
                )
            except ValueError:
                pass
        
        return None
    
    async def _extract_temporal_constraints(self, query: str) -> Optional[TemporalConstraint]:
        """Extract temporal constraints from query."""
        # Year range patterns
        year_range_pattern = r'(\d{4})-(\d{4})'
        year_matches = re.findall(year_range_pattern, query)
        
        if year_matches:
            start_year, end_year = year_matches[0]
            try:
                start_date = datetime(int(start_year), 1, 1, tzinfo=timezone.utc)
                end_date = datetime(int(end_year), 12, 31, tzinfo=timezone.utc)
                return TemporalConstraint(start_date=start_date, end_date=end_date)
            except ValueError:
                pass
        
        # Single year pattern
        single_year_pattern = r'\b(\d{4})\b'
        year_matches = re.findall(single_year_pattern, query)
        
        if year_matches:
            year = year_matches[-1]  # Use last mentioned year
            try:
                start_date = datetime(int(year), 1, 1, tzinfo=timezone.utc)
                end_date = datetime(int(year), 12, 31, tzinfo=timezone.utc)
                return TemporalConstraint(start_date=start_date, end_date=end_date)
            except ValueError:
                pass
        
        # Relative time patterns
        query_lower = query.lower()
        if 'last decade' in query_lower or 'past 10 years' in query_lower:
            current_year = datetime.now().year
            start_date = datetime(current_year - 10, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(current_year, 12, 31, tzinfo=timezone.utc)
            return TemporalConstraint(start_date=start_date, end_date=end_date)
        
        return None
    
    async def _extract_domain_constraints(self, query: str) -> Dict[str, Any]:
        """Extract domain-specific constraints."""
        query_lower = query.lower()
        
        # Data types
        data_types = []
        if 'satellite' in query_lower:
            data_types.append(DataType.SATELLITE)
        if 'ground' in query_lower or 'station' in query_lower:
            data_types.append(DataType.GROUND_BASED)
        if 'model' in query_lower:
            data_types.append(DataType.MODEL)
        
        # Variables/keywords
        keywords = []
        for variable in self.common_variables:
            if variable in query_lower:
                keywords.append(variable)
        
        # Platforms/instruments
        platforms = []
        instruments = []
        
        platform_patterns = {
            'modis': ['Terra', 'Aqua'],
            'landsat': 'Landsat',
            'sentinel': 'Sentinel',
            'goes': 'GOES',
            'viirs': 'VIIRS',
            'avhrr': 'AVHRR'
        }
        
        for pattern, platform in platform_patterns.items():
            if pattern in query_lower:
                if isinstance(platform, list):
                    platforms.extend(platform)
                else:
                    platforms.append(platform)
                instruments.append(pattern.upper())
        
        return {
            'data_types': data_types,
            'keywords': keywords,
            'platforms': platforms,
            'instruments': instruments,
            'variables': keywords  # Use same as keywords for now
        }
    
    def _calculate_complexity_score(self, constraints: QueryConstraints, query: str) -> float:
        """Calculate query complexity score."""
        score = 0.0
        
        # Base complexity from constraint count
        constraint_count = sum([
            1 if constraints.spatial else 0,
            1 if constraints.temporal else 0,
            len(constraints.data_types),
            len(constraints.keywords),
            len(constraints.platforms),
            len(constraints.instruments),
            len(constraints.variables)
        ])
        
        score += min(constraint_count * 0.1, 0.5)
        
        # Query length and complexity indicators
        query_lower = query.lower()
        
        complexity_indicators = [
            'compare', 'analyze', 'trend', 'correlation', 'relationship',
            'gap', 'coverage', 'quality', 'validation', 'fusion'
        ]
        
        for indicator in complexity_indicators:
            if indicator in query_lower:
                score += 0.1
        
        # Multi-region or multi-temporal complexity
        if constraints.spatial and constraints.temporal:
            score += 0.2
        
        return min(score, 1.0)
    
    def _identify_research_domain(self, query: str) -> Optional[str]:
        """Identify research domain from query."""
        query_lower = query.lower()
        
        domains = {
            'climate': ['climate', 'warming', 'temperature trend', 'precipitation pattern'],
            'hydrology': ['drought', 'flood', 'water', 'precipitation', 'streamflow'],
            'agriculture': ['crop', 'vegetation', 'farming', 'agricultural', 'ndvi'],
            'atmospheric': ['air quality', 'aerosol', 'ozone', 'atmospheric'],
            'oceanography': ['ocean', 'sea surface', 'marine', 'coastal'],
            'cryosphere': ['ice', 'snow', 'glacier', 'arctic', 'antarctic'],
            'fire': ['fire', 'wildfire', 'burn', 'smoke'],
            'urban': ['urban', 'city', 'metropolitan', 'heat island'],
            'ecology': ['ecosystem', 'biodiversity', 'habitat', 'species']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return None
    
    def _extract_methodology_hints(self, query: str) -> List[str]:
        """Extract methodology hints from query."""
        query_lower = query.lower()
        hints = []
        
        methodology_patterns = {
            'time series analysis': ['trend', 'temporal', 'time series', 'seasonal'],
            'spatial analysis': ['spatial', 'geographic', 'regional', 'pattern'],
            'statistical analysis': ['correlation', 'regression', 'statistical'],
            'machine learning': ['classification', 'clustering', 'ml', 'ai'],
            'visualization': ['map', 'plot', 'chart', 'visualize', 'show'],
            'gap analysis': ['gap', 'coverage', 'availability', 'missing'],
            'validation': ['validate', 'verify', 'accuracy', 'quality'],
            'comparative analysis': ['compare', 'versus', 'difference', 'contrast']
        }
        
        for methodology, keywords in methodology_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                hints.append(methodology)
        
        return hints
    
    def _validate_spatial_constraints(self, spatial: SpatialConstraint) -> bool:
        """Validate spatial constraints."""
        if spatial.north is not None and spatial.south is not None:
            if spatial.north <= spatial.south:
                return False
        
        if spatial.east is not None and spatial.west is not None:
            # Handle longitude wrap-around
            if spatial.east < spatial.west and abs(spatial.east - spatial.west) < 180:
                return False
        
        return True
    
    def _validate_temporal_constraints(self, temporal: TemporalConstraint) -> bool:
        """Validate temporal constraints."""
        if temporal.start_date and temporal.end_date:
            if temporal.end_date <= temporal.start_date:
                return False
        
        # Check if dates are reasonable (not too far in future)
        current_date = datetime.now(timezone.utc)
        if temporal.start_date and temporal.start_date > current_date:
            return False
        
        return True
    
    async def _check_data_availability(self, constraints: QueryConstraints) -> bool:
        """Basic data availability check."""
        # This is a simplified check - in practice would query CMR
        
        # Check if temporal range is reasonable for satellite data
        if constraints.temporal:
            start_year = constraints.temporal.start_date.year if constraints.temporal.start_date else 2000
            if start_year < 1972:  # Before Landsat era
                if DataType.SATELLITE in constraints.data_types:
                    return False
        
        return True