import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import structlog
from collections import defaultdict

from ..models.schemas import (
    QueryContext, CMRCollection, CMRGranule, DatasetRecommendation, 
    AnalysisResult, QueryIntent
)
from ..services.llm_service import LLMService
from ..services.comparative_analysis import ComparativeAnalysisEngine
from ..agents.enhanced_query_decomposer import SubQuery

logger = structlog.get_logger(__name__)


class DataAnalysisAgent:
    """
    Advanced data analysis and recommendation generation agent.
    
    Capabilities:
    - Dataset relevance scoring and ranking
    - Temporal and spatial coverage analysis
    - Data quality assessment
    - Gap analysis and completeness evaluation
    - Cross-dataset relationship identification
    - Statistical summaries and recommendations
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.comparative_engine = ComparativeAnalysisEngine()
    
    async def analyze_results(
        self, 
        query_context: QueryContext, 
        cmr_results: Dict[str, Any],
        sub_queries: Optional[List[SubQuery]] = None
    ) -> List[AnalysisResult]:
        """
        Comprehensive analysis of CMR search results.
        
        Args:
            query_context: Original query context and constraints
            cmr_results: Results from CMR API searches
            sub_queries: Optional decomposed sub-queries for comparative analysis
            
        Returns:
            List of analysis results with recommendations and insights
        """
        collections = cmr_results.get("collections", [])
        granules = cmr_results.get("granules", [])
        
        analysis_results = []
        
        try:
            # Generate dataset recommendations
            recommendations = await self._generate_recommendations(query_context, collections, granules)
            
            if recommendations:
                analysis_results.append(AnalysisResult(
                    analysis_type="dataset_recommendations",
                    results={"recommendations": [rec.dict() for rec in recommendations]},
                    methodology="Multi-criteria scoring with relevance, coverage, quality, and accessibility factors",
                    confidence_level=0.85,
                    statistics={"total_datasets": len(collections), "recommended": len(recommendations)}
                ))
            
            # Coverage analysis
            if query_context.intent in [QueryIntent.COMPARATIVE, QueryIntent.TEMPORAL_ANALYSIS, QueryIntent.SPATIAL_ANALYSIS]:
                coverage_analysis = await self._analyze_coverage(query_context, collections, granules)
                if coverage_analysis:
                    analysis_results.append(coverage_analysis)
            
            # Gap analysis for temporal queries
            if query_context.constraints.temporal:
                gap_analysis = await self._analyze_temporal_gaps(query_context, collections, granules)
                if gap_analysis:
                    analysis_results.append(gap_analysis)
            
            # Cross-dataset relationship analysis
            if len(collections) > 1:
                relationship_analysis = await self._analyze_dataset_relationships(collections)
                if relationship_analysis:
                    analysis_results.append(relationship_analysis)
            
            # Comparative analysis if sub-queries provided
            if sub_queries and len(sub_queries) > 1:
                comparative_analysis = await self._perform_comparative_analysis(
                    sub_queries, recommendations, query_context
                )
                if comparative_analysis:
                    analysis_results.append(comparative_analysis)
            
            logger.info("Analysis completed", analysis_count=len(analysis_results))
            
        except Exception as e:
            logger.error("Analysis failed", error=str(e))
            # Return basic analysis even if advanced features fail
            analysis_results.append(AnalysisResult(
                analysis_type="basic_summary",
                results={
                    "collections_found": len(collections),
                    "granules_found": len(granules),
                    "error": str(e)
                },
                methodology="Basic counting and summarization",
                confidence_level=1.0
            ))
        
        return analysis_results
    
    async def _generate_recommendations(
        self, 
        query_context: QueryContext, 
        collections: List[CMRCollection],
        granules: List[CMRGranule]
    ) -> List[DatasetRecommendation]:
        """Generate scored dataset recommendations."""
        recommendations = []
        
        # Group granules by collection
        granules_by_collection = defaultdict(list)
        for granule in granules:
            granules_by_collection[granule.collection_concept_id].append(granule)
        
        for collection in collections:
            try:
                # Calculate multiple scoring dimensions
                relevance_score = self._calculate_relevance_score(query_context, collection)
                coverage_score = self._calculate_coverage_score(query_context, collection)
                quality_score = self._calculate_quality_score(collection)
                accessibility_score = self._calculate_accessibility_score(collection)
                
                # Get granules for this collection
                collection_granules = granules_by_collection.get(collection.concept_id, [])
                
                # Analyze temporal and spatial gaps
                temporal_gaps = self._identify_temporal_gaps(query_context, collection_granules)
                spatial_gaps = self._identify_spatial_gaps(query_context, collection, collection_granules)
                
                # Generate reasoning
                reasoning = await self._generate_reasoning(
                    query_context, collection, relevance_score, coverage_score, 
                    quality_score, accessibility_score
                )
                
                # Find complementary datasets
                complementary_datasets = self._find_complementary_datasets(
                    collection, collections, query_context
                )
                
                recommendation = DatasetRecommendation(
                    collection=collection,
                    relevance_score=relevance_score,
                    coverage_score=coverage_score,
                    quality_score=quality_score,
                    accessibility_score=accessibility_score,
                    reasoning=reasoning,
                    granule_count=len(collection_granules),
                    temporal_gaps=temporal_gaps,
                    spatial_gaps=spatial_gaps,
                    complementary_datasets=complementary_datasets
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning("Failed to generate recommendation", 
                              collection_id=collection.concept_id, error=str(e))
                continue
        
        # Sort by overall score (weighted combination)
        recommendations.sort(key=lambda r: (
            r.relevance_score * 0.4 + 
            r.coverage_score * 0.3 + 
            r.quality_score * 0.2 + 
            r.accessibility_score * 0.1
        ), reverse=True)
        
        return recommendations[:10]  # Return top 10
    
    def _calculate_relevance_score(self, query_context: QueryContext, collection: CMRCollection) -> float:
        """Calculate relevance score based on query matching."""
        score = 0.0
        
        # Keyword matching in title and summary
        query_keywords = set(query_context.original_query.lower().split())
        title_keywords = set(collection.title.lower().split()) if collection.title else set()
        summary_keywords = set(collection.summary.lower().split()) if collection.summary else set()
        
        title_overlap = len(query_keywords.intersection(title_keywords)) / max(len(query_keywords), 1)
        summary_overlap = len(query_keywords.intersection(summary_keywords)) / max(len(query_keywords), 1)
        
        score += title_overlap * 0.4 + summary_overlap * 0.2
        
        # Platform/instrument matching
        if query_context.constraints.platforms:
            platform_matches = len(set(query_context.constraints.platforms).intersection(
                set(collection.platforms)
            )) / len(query_context.constraints.platforms)
            score += platform_matches * 0.2
        
        # Variable/keyword matching
        if query_context.constraints.keywords:
            keyword_matches = 0
            for keyword in query_context.constraints.keywords:
                if any(keyword.lower() in text.lower() 
                      for text in [collection.title, collection.summary] if text):
                    keyword_matches += 1
            score += (keyword_matches / len(query_context.constraints.keywords)) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_coverage_score(self, query_context: QueryContext, collection: CMRCollection) -> float:
        """Calculate temporal and spatial coverage score."""
        score = 0.0
        
        # Temporal coverage
        if (query_context.constraints.temporal and 
            query_context.constraints.temporal.start_date and
            query_context.constraints.temporal.end_date and
            collection.temporal_coverage):
            
            query_start = query_context.constraints.temporal.start_date
            query_end = query_context.constraints.temporal.end_date
            
            # Parse collection temporal coverage
            try:
                if collection.temporal_coverage.get("start"):
                    coll_start = datetime.fromisoformat(
                        collection.temporal_coverage["start"].replace('Z', '+00:00')
                    )
                else:
                    coll_start = query_start
                    
                if collection.temporal_coverage.get("end"):
                    coll_end = datetime.fromisoformat(
                        collection.temporal_coverage["end"].replace('Z', '+00:00')
                    )
                else:
                    coll_end = datetime.now()
                
                # Calculate overlap
                overlap_start = max(query_start, coll_start)
                overlap_end = min(query_end, coll_end)
                
                if overlap_end > overlap_start:
                    overlap_days = (overlap_end - overlap_start).days
                    query_days = (query_end - query_start).days
                    temporal_coverage = min(overlap_days / max(query_days, 1), 1.0)
                    score += temporal_coverage * 0.5
                
            except (ValueError, TypeError):
                score += 0.25  # Partial score if can't parse dates
        else:
            score += 0.5  # Full temporal score if no constraints
        
        # Spatial coverage
        if (query_context.constraints.spatial and 
            collection.spatial_coverage and
            all(k in collection.spatial_coverage for k in ['north', 'south', 'east', 'west'])):
            
            # Calculate bounding box overlap (simplified)
            try:
                query_spatial = query_context.constraints.spatial
                coll_spatial = collection.spatial_coverage
                
                if all([query_spatial.north, query_spatial.south, 
                       query_spatial.east, query_spatial.west]):
                    
                    # Check for overlap
                    lat_overlap = (min(query_spatial.north, coll_spatial['north']) > 
                                  max(query_spatial.south, coll_spatial['south']))
                    lon_overlap = (min(query_spatial.east, coll_spatial['east']) > 
                                  max(query_spatial.west, coll_spatial['west']))
                    
                    if lat_overlap and lon_overlap:
                        score += 0.5
                    else:
                        score += 0.1  # Partial score for proximity
                else:
                    score += 0.25  # Partial score if incomplete constraints
            except (KeyError, TypeError):
                pass
        else:
            score += 0.5  # Full spatial score if no constraints
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, collection: CMRCollection) -> float:
        """Calculate data quality score based on metadata completeness and flags."""
        score = 0.0
        
        # Metadata completeness
        metadata_fields = [
            collection.title, collection.summary, collection.data_center,
            collection.platforms, collection.instruments, 
            collection.temporal_coverage, collection.spatial_coverage
        ]
        
        completeness = sum(1 for field in metadata_fields if field) / len(metadata_fields)
        score += completeness * 0.4
        
        # Online access availability
        if collection.online_access_flag:
            score += 0.3
        
        # Cloud hosting (indicates modern, accessible data)
        if collection.cloud_hosted:
            score += 0.2
        
        # Processing level (higher levels often mean more processed/ready to use)
        if collection.processing_level:
            try:
                level_num = int(collection.processing_level.replace('L', ''))
                score += min(level_num / 4.0, 0.1)  # L4 gets full score
            except (ValueError, AttributeError):
                pass
        
        return min(score, 1.0)
    
    def _calculate_accessibility_score(self, collection: CMRCollection) -> float:
        """Calculate data accessibility score."""
        score = 0.0
        
        # Online access flag
        if collection.online_access_flag:
            score += 0.4
        
        # Cloud hosting
        if collection.cloud_hosted:
            score += 0.4
        
        # Data center reliability (NASA centers get higher scores)
        nasa_centers = ['GSFC', 'JPL', 'LARC', 'MSFC', 'NSIDC', 'ORNL', 'SEDAC']
        if any(center in collection.data_center.upper() for center in nasa_centers):
            score += 0.2
        
        return min(score, 1.0)
    
    def _identify_temporal_gaps(
        self, 
        query_context: QueryContext, 
        granules: List[CMRGranule]
    ) -> List[Dict[str, Any]]:
        """Identify temporal gaps in data coverage."""
        gaps = []
        
        if not query_context.constraints.temporal or not granules:
            return gaps
        
        # Sort granules by start time
        sorted_granules = []
        for granule in granules:
            if (granule.temporal_extent and 
                granule.temporal_extent.get("start")):
                try:
                    start_time = datetime.fromisoformat(
                        granule.temporal_extent["start"].replace('Z', '+00:00')
                    )
                    sorted_granules.append((start_time, granule))
                except (ValueError, TypeError):
                    continue
        
        sorted_granules.sort(key=lambda x: x[0])
        
        # Find gaps larger than expected interval
        expected_interval = timedelta(days=1)  # Assume daily data
        
        for i in range(len(sorted_granules) - 1):
            current_end = sorted_granules[i][0]
            next_start = sorted_granules[i + 1][0]
            
            gap_duration = next_start - current_end
            if gap_duration > expected_interval * 2:  # Gap is more than 2x expected
                gaps.append({
                    "start": current_end.isoformat(),
                    "end": next_start.isoformat(),
                    "duration_days": gap_duration.days,
                    "type": "temporal_gap"
                })
        
        return gaps[:5]  # Return up to 5 largest gaps
    
    def _identify_spatial_gaps(
        self, 
        query_context: QueryContext, 
        collection: CMRCollection,
        granules: List[CMRGranule]
    ) -> List[Dict[str, Any]]:
        """Identify spatial gaps in data coverage."""
        # Simplified spatial gap analysis
        gaps = []
        
        if (not query_context.constraints.spatial or 
            not collection.spatial_coverage):
            return gaps
        
        # Basic check: does collection cover requested region?
        query_spatial = query_context.constraints.spatial
        coll_spatial = collection.spatial_coverage
        
        if (all([query_spatial.north, query_spatial.south, 
                query_spatial.east, query_spatial.west]) and
            all(k in coll_spatial for k in ['north', 'south', 'east', 'west'])):
            
            # Check for gaps in coverage
            if query_spatial.north > coll_spatial['north']:
                gaps.append({
                    "type": "northern_boundary_gap",
                    "requested": query_spatial.north,
                    "available": coll_spatial['north']
                })
            
            if query_spatial.south < coll_spatial['south']:
                gaps.append({
                    "type": "southern_boundary_gap", 
                    "requested": query_spatial.south,
                    "available": coll_spatial['south']
                })
        
        return gaps
    
    async def _generate_reasoning(
        self,
        query_context: QueryContext,
        collection: CMRCollection,
        relevance_score: float,
        coverage_score: float,
        quality_score: float,
        accessibility_score: float
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        try:
            reasoning_prompt = f"""
            Explain why this NASA dataset is recommended for the query: "{query_context.original_query}"
            
            Dataset: {collection.title}
            Summary: {collection.summary[:200]}...
            
            Scores:
            - Relevance: {relevance_score:.2f}
            - Coverage: {coverage_score:.2f}
            - Quality: {quality_score:.2f}
            - Accessibility: {accessibility_score:.2f}
            
            Provide a concise 2-3 sentence explanation focusing on the strongest aspects.
            """
            
            reasoning = await self.llm_service.generate(reasoning_prompt, max_tokens=100)
            return reasoning.strip()
            
        except Exception:
            # Fallback to rule-based reasoning
            reasons = []
            
            if relevance_score > 0.7:
                reasons.append("high relevance to query terms")
            if coverage_score > 0.7:
                reasons.append("excellent temporal/spatial coverage")
            if quality_score > 0.7:
                reasons.append("high-quality metadata and processing")
            if accessibility_score > 0.7:
                reasons.append("easily accessible data")
            
            if reasons:
                return f"This dataset is recommended due to {', '.join(reasons)}."
            else:
                return "This dataset provides relevant data for your query."
    
    def _find_complementary_datasets(
        self,
        target_collection: CMRCollection,
        all_collections: List[CMRCollection],
        query_context: QueryContext
    ) -> List[str]:
        """Find datasets that complement the target dataset."""
        complementary = []
        
        for collection in all_collections:
            if collection.concept_id == target_collection.concept_id:
                continue
            
            # Look for complementary platforms/instruments
            if (set(target_collection.platforms) != set(collection.platforms) and
                any(platform in collection.platforms 
                    for platform in ['MODIS', 'VIIRS', 'Landsat', 'Sentinel'])):
                complementary.append(collection.short_name or collection.title)
            
            # Different temporal resolution or coverage
            if (collection.temporal_coverage and target_collection.temporal_coverage):
                # This would need more sophisticated comparison
                pass
        
        return complementary[:3]  # Return up to 3 complementary datasets
    
    async def _analyze_coverage(
        self, 
        query_context: QueryContext,
        collections: List[CMRCollection], 
        granules: List[CMRGranule]
    ) -> Optional[AnalysisResult]:
        """Analyze temporal and spatial coverage statistics."""
        coverage_stats = {}
        
        # Temporal coverage statistics
        if query_context.constraints.temporal:
            temporal_stats = self._calculate_temporal_coverage_stats(collections, granules)
            coverage_stats["temporal"] = temporal_stats
        
        # Spatial coverage statistics  
        if query_context.constraints.spatial:
            spatial_stats = self._calculate_spatial_coverage_stats(collections)
            coverage_stats["spatial"] = spatial_stats
        
        if coverage_stats:
            return AnalysisResult(
                analysis_type="coverage_analysis",
                results=coverage_stats,
                methodology="Statistical analysis of temporal and spatial coverage",
                confidence_level=0.9,
                statistics={"collections_analyzed": len(collections)}
            )
        
        return None
    
    def _calculate_temporal_coverage_stats(
        self, 
        collections: List[CMRCollection],
        granules: List[CMRGranule]
    ) -> Dict[str, Any]:
        """Calculate temporal coverage statistics."""
        stats = {}
        
        # Collection-level temporal stats
        collection_dates = []
        for collection in collections:
            if collection.temporal_coverage:
                if collection.temporal_coverage.get("start"):
                    try:
                        start_date = datetime.fromisoformat(
                            collection.temporal_coverage["start"].replace('Z', '+00:00')
                        )
                        collection_dates.append(start_date)
                    except (ValueError, TypeError):
                        pass
        
        if collection_dates:
            stats["earliest_data"] = min(collection_dates).isoformat()
            stats["collection_count_with_temporal"] = len(collection_dates)
            
            # Calculate coverage span
            latest_date = max(collection_dates)
            earliest_date = min(collection_dates)
            total_span = (latest_date - earliest_date).days
            stats["total_temporal_span_days"] = total_span
        
        # Granule-level temporal stats
        if granules:
            granule_dates = []
            for granule in granules:
                if (granule.temporal_extent and 
                    granule.temporal_extent.get("start")):
                    try:
                        start_date = datetime.fromisoformat(
                            granule.temporal_extent["start"].replace('Z', '+00:00')
                        )
                        granule_dates.append(start_date)
                    except (ValueError, TypeError):
                        pass
            
            if granule_dates:
                stats["granule_date_range"] = {
                    "start": min(granule_dates).isoformat(),
                    "end": max(granule_dates).isoformat()
                }
                stats["granule_count"] = len(granule_dates)
        
        return stats
    
    def _calculate_spatial_coverage_stats(self, collections: List[CMRCollection]) -> Dict[str, Any]:
        """Calculate spatial coverage statistics."""
        stats = {}
        
        spatial_collections = [c for c in collections if c.spatial_coverage]
        stats["collections_with_spatial"] = len(spatial_collections)
        
        if spatial_collections:
            # Calculate bounding box statistics
            north_values = []
            south_values = []
            east_values = []
            west_values = []
            
            for collection in spatial_collections:
                spatial = collection.spatial_coverage
                if all(k in spatial for k in ['north', 'south', 'east', 'west']):
                    north_values.append(spatial['north'])
                    south_values.append(spatial['south'])
                    east_values.append(spatial['east'])
                    west_values.append(spatial['west'])
            
            if north_values:
                stats["bounding_box"] = {
                    "north": max(north_values),
                    "south": min(south_values),
                    "east": max(east_values),
                    "west": min(west_values)
                }
                
                # Calculate average coverage area (approximate)
                total_area = 0
                for i in range(len(north_values)):
                    lat_span = north_values[i] - south_values[i]
                    lon_span = east_values[i] - west_values[i]
                    total_area += lat_span * lon_span
                
                stats["average_coverage_area_deg2"] = total_area / len(north_values)
        
        return stats
    
    async def _analyze_temporal_gaps(
        self,
        query_context: QueryContext,
        collections: List[CMRCollection],
        granules: List[CMRGranule]
    ) -> Optional[AnalysisResult]:
        """Analyze temporal gaps across datasets."""
        if not query_context.constraints.temporal:
            return None
        
        gap_analysis = {}
        
        # Analyze gaps for each collection
        granules_by_collection = defaultdict(list)
        for granule in granules:
            granules_by_collection[granule.collection_concept_id].append(granule)
        
        collection_gaps = {}
        for collection in collections:
            collection_granules = granules_by_collection.get(collection.concept_id, [])
            gaps = self._identify_temporal_gaps(query_context, collection_granules)
            if gaps:
                collection_gaps[collection.short_name or collection.title] = gaps
        
        gap_analysis["collection_gaps"] = collection_gaps
        
        # Overall gap statistics
        all_gaps = []
        for gaps in collection_gaps.values():
            all_gaps.extend(gaps)
        
        if all_gaps:
            gap_durations = [gap["duration_days"] for gap in all_gaps]
            gap_analysis["statistics"] = {
                "total_gaps": len(all_gaps),
                "average_gap_days": sum(gap_durations) / len(gap_durations),
                "longest_gap_days": max(gap_durations)
            }
        
        if gap_analysis:
            return AnalysisResult(
                analysis_type="temporal_gap_analysis",
                results=gap_analysis,
                methodology="Temporal gap identification based on expected data frequency",
                confidence_level=0.8
            )
        
        return None
    
    async def _analyze_dataset_relationships(
        self, 
        collections: List[CMRCollection]
    ) -> Optional[AnalysisResult]:
        """Analyze relationships between datasets."""
        relationships = {}
        
        # Group by platform
        platform_groups = defaultdict(list)
        for collection in collections:
            for platform in collection.platforms:
                platform_groups[platform].append(collection.short_name or collection.title)
        
        relationships["platform_groups"] = dict(platform_groups)
        
        # Group by data center
        datacenter_groups = defaultdict(list)
        for collection in collections:
            datacenter_groups[collection.data_center].append(
                collection.short_name or collection.title
            )
        
        relationships["datacenter_groups"] = dict(datacenter_groups)
        
        # Find potential fusion opportunities
        fusion_pairs = []
        for i, coll1 in enumerate(collections):
            for j, coll2 in enumerate(collections[i+1:], i+1):
                if (set(coll1.platforms).intersection(set(coll2.platforms)) or
                    coll1.data_center == coll2.data_center):
                    fusion_pairs.append([
                        coll1.short_name or coll1.title,
                        coll2.short_name or coll2.title
                    ])
        
        relationships["potential_fusion_pairs"] = fusion_pairs[:5]  # Top 5
        
        if relationships:
            return AnalysisResult(
                analysis_type="dataset_relationship_analysis",
                results=relationships,
                methodology="Platform and datacenter grouping analysis",
                confidence_level=0.7
            )
        
        return None
    
    async def _perform_comparative_analysis(
        self,
        sub_queries: List[SubQuery],
        recommendations: List[DatasetRecommendation],
        query_context: QueryContext
    ) -> Optional[AnalysisResult]:
        """Perform comparative analysis using the comparative analysis engine."""
        try:
            # Group recommendations by sub-query ID
            recommendations_by_subquery = {}
            for rec in recommendations:
                # For now, group by data type since we don't have sub-query ID in recommendations
                for sub_query in sub_queries:
                    data_types = query_context.constraints.data_types if query_context.constraints else []
                    if any(dt in sub_query.data_types for dt in data_types):
                        recommendations_by_subquery[sub_query.id] = recommendations_by_subquery.get(sub_query.id, [])
                        recommendations_by_subquery[sub_query.id].append(rec)
            
            # Perform comparative analysis
            comparative_results = await self.comparative_engine.perform_comparative_analysis(
                sub_queries, recommendations_by_subquery
            )
            
            return AnalysisResult(
                analysis_type="comparative_analysis",
                results=comparative_results,
                methodology="Multi-data-type comparative analysis with gap identification and fusion opportunities",
                confidence_level=0.9,
                statistics={
                    "sub_queries_analyzed": len(sub_queries),
                    "data_types_compared": len(set().union(*[sq.data_types for sq in sub_queries]))
                }
            )
            
        except Exception as e:
            logger.error("Comparative analysis failed", error=str(e))
            return None