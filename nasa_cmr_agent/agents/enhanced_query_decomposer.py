"""
Enhanced Query Decomposer for Multi-Data-Type Comparative Queries

Specifically designed to handle complex queries requiring comparison across data types.
"""

import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.schemas import QueryContext, DataType, QueryIntent


class QueryDecompositionStrategy(Enum):
    SIMPLE = "simple"
    COMPARATIVE = "comparative"
    MULTI_DATA_TYPE = "multi_data_type"
    TEMPORAL_ANALYSIS = "temporal_analysis"


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""
    id: str
    description: str
    data_types: List[DataType]
    focus_area: str
    spatial_constraints: Any
    temporal_constraints: Any
    keywords: List[str]
    priority: int = 1  # Higher number = higher priority


class EnhancedQueryDecomposer:
    """
    Advanced query decomposer for complex multi-data-type comparative queries.
    
    Handles queries like:
    "Compare precipitation datasets suitable for drought monitoring in Sub-Saharan Africa
    between 2015-2023, considering both satellite and ground-based observations, and identify
    gaps in temporal coverage."
    """
    
    def __init__(self):
        pass  # Self-contained to avoid circular imports
    
    async def decompose_query(self, query: str, query_context: QueryContext) -> Tuple[QueryDecompositionStrategy, List[SubQuery]]:
        """
        Decompose complex query into optimized sub-queries for parallel execution.
        
        Args:
            query: Original complex query
            query_context: Pre-interpreted query context from QueryInterpreterAgent
            
        Returns:
            Tuple of (strategy, sub_queries)
        """
        # Determine decomposition strategy
        strategy = self._determine_strategy(query, query_context)
        
        # Generate sub-queries based on strategy
        sub_queries = await self._generate_sub_queries(strategy, query, query_context)
        
        return strategy, sub_queries
    
    def _determine_strategy(self, query: str, context: QueryContext) -> QueryDecompositionStrategy:
        """Determine the best decomposition strategy."""
        query_lower = query.lower()
        
        # Check for comparative keywords
        comparative_keywords = ["compare", "versus", "vs", "between", "different", "contrast"]
        has_comparative = any(kw in query_lower for kw in comparative_keywords)
        
        # Check for multiple data types
        data_type_count = len(context.constraints.data_types) if context.constraints and context.constraints.data_types else 0
        
        # Check for temporal analysis keywords
        temporal_analysis_keywords = ["gaps", "coverage", "temporal", "missing", "discontinu"]
        has_temporal_analysis = any(kw in query_lower for kw in temporal_analysis_keywords)
        
        # Check for multi-data-type indicators
        multi_type_indicators = [
            "satellite and ground", "ground and satellite", "both satellite and ground",
            "multiple data types", "different data types", "various sources"
        ]
        has_multi_type_explicit = any(indicator in query_lower for indicator in multi_type_indicators)
        
        # Enhanced strategy selection
        if has_comparative and (has_multi_type_explicit or data_type_count > 1) and has_temporal_analysis:
            return QueryDecompositionStrategy.MULTI_DATA_TYPE
        elif has_comparative and (has_multi_type_explicit or data_type_count > 1):
            return QueryDecompositionStrategy.MULTI_DATA_TYPE
        elif has_comparative:
            return QueryDecompositionStrategy.COMPARATIVE
        elif has_temporal_analysis:
            return QueryDecompositionStrategy.TEMPORAL_ANALYSIS
        else:
            return QueryDecompositionStrategy.SIMPLE
    
    async def _generate_sub_queries(
        self, 
        strategy: QueryDecompositionStrategy, 
        original_query: str, 
        context: QueryContext
    ) -> List[SubQuery]:
        """Generate sub-queries based on strategy."""
        
        if strategy == QueryDecompositionStrategy.MULTI_DATA_TYPE:
            return await self._generate_multi_data_type_queries(original_query, context)
        elif strategy == QueryDecompositionStrategy.COMPARATIVE:
            return await self._generate_comparative_queries(original_query, context)
        elif strategy == QueryDecompositionStrategy.TEMPORAL_ANALYSIS:
            return await self._generate_temporal_analysis_queries(original_query, context)
        else:
            return await self._generate_simple_queries(original_query, context)
    
    async def _generate_multi_data_type_queries(
        self, 
        query: str, 
        context: QueryContext
    ) -> List[SubQuery]:
        """
        Generate sub-queries for multi-data-type comparative analysis.
        
        Example: "Compare precipitation datasets for drought monitoring..."
        Generates:
        1. Satellite precipitation datasets for drought monitoring
        2. Ground-based precipitation datasets for drought monitoring  
        3. Comparative temporal coverage analysis
        4. Data fusion opportunities analysis
        """
        sub_queries = []
        
        # Extract core elements
        keywords = context.constraints.keywords if context.constraints else []
        spatial = context.constraints.spatial if context.constraints else None
        temporal = context.constraints.temporal if context.constraints else None
        data_types = context.constraints.data_types if context.constraints else []
        
        # Generate individual data type queries
        for i, data_type in enumerate(data_types):
            data_type_name = data_type.value.replace('_', ' ')
            
            sub_queries.append(SubQuery(
                id=f"dt_{i}_{data_type.value}",
                description=f"Find {data_type_name} datasets for {' '.join(keywords)}",
                data_types=[data_type],
                focus_area=f"{data_type_name}_collection",
                spatial_constraints=spatial,
                temporal_constraints=temporal,
                keywords=keywords,
                priority=2
            ))
        
        # Add comparative analysis sub-query
        sub_queries.append(SubQuery(
            id="comparative_analysis",
            description=f"Compare temporal coverage between {' and '.join([dt.value.replace('_', ' ') for dt in data_types])}",
            data_types=data_types,
            focus_area="temporal_comparison",
            spatial_constraints=spatial,
            temporal_constraints=temporal,
            keywords=keywords + ["temporal coverage", "gaps"],
            priority=3
        ))
        
        # Add data fusion opportunities analysis
        sub_queries.append(SubQuery(
            id="fusion_analysis", 
            description="Identify data fusion opportunities and complementary datasets",
            data_types=data_types,
            focus_area="data_fusion",
            spatial_constraints=spatial,
            temporal_constraints=temporal,
            keywords=keywords + ["complementary", "fusion"],
            priority=1
        ))
        
        return sub_queries
    
    async def _generate_comparative_queries(
        self, 
        query: str, 
        context: QueryContext
    ) -> List[SubQuery]:
        """Generate sub-queries for comparative analysis."""
        # Extract comparison subjects from query
        # This would need more sophisticated NLP parsing
        return [SubQuery(
            id="main_query",
            description=query,
            data_types=context.constraints.data_types if context.constraints else [],
            focus_area="comparative",
            spatial_constraints=context.constraints.spatial if context.constraints else None,
            temporal_constraints=context.constraints.temporal if context.constraints else None,
            keywords=context.constraints.keywords if context.constraints else [],
            priority=1
        )]
    
    async def _generate_temporal_analysis_queries(
        self, 
        query: str, 
        context: QueryContext
    ) -> List[SubQuery]:
        """Generate sub-queries for temporal analysis."""
        return [SubQuery(
            id="temporal_main",
            description=query,
            data_types=context.constraints.data_types if context.constraints else [],
            focus_area="temporal_analysis",
            spatial_constraints=context.constraints.spatial if context.constraints else None,
            temporal_constraints=context.constraints.temporal if context.constraints else None,
            keywords=context.constraints.keywords if context.constraints else [],
            priority=1
        )]
    
    async def _generate_simple_queries(
        self, 
        query: str, 
        context: QueryContext
    ) -> List[SubQuery]:
        """Generate sub-queries for simple queries."""
        return [SubQuery(
            id="simple_main",
            description=query,
            data_types=context.constraints.data_types if context.constraints else [],
            focus_area="general",
            spatial_constraints=context.constraints.spatial if context.constraints else None,
            temporal_constraints=context.constraints.temporal if context.constraints else None,
            keywords=context.constraints.keywords if context.constraints else [],
            priority=1
        )]