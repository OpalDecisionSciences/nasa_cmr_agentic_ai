"""
Comparative Analysis Engine

Provides sophisticated comparison capabilities for multi-data-type queries.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np

from ..models.schemas import DatasetRecommendation, CMRCollection, CMRGranule, DataType
from ..agents.enhanced_query_decomposer import SubQuery


@dataclass
class ComparisonResult:
    """Results of comparative analysis."""
    data_type: DataType
    dataset_count: int
    temporal_coverage: Dict[str, Any]
    spatial_coverage: Dict[str, Any]
    quality_metrics: Dict[str, float]
    gaps_identified: List[Dict[str, Any]]
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class DataFusionOpportunity:
    """Represents a data fusion opportunity."""
    primary_dataset: str
    complementary_datasets: List[str]
    fusion_type: str  # temporal, spatial, spectral, etc.
    benefits: List[str]
    requirements: List[str]
    confidence_score: float


class ComparativeAnalysisEngine:
    """
    Advanced comparative analysis engine for multi-data-type queries.
    
    Provides:
    - Data type comparison (satellite vs ground-based)
    - Temporal coverage analysis and gap identification
    - Spatial coverage comparison
    - Data fusion opportunity identification
    - Drought monitoring suitability assessment
    """
    
    def __init__(self):
        self.drought_indicators = {
            "precipitation": {"weight": 0.4, "critical": True},
            "soil_moisture": {"weight": 0.3, "critical": True},
            "temperature": {"weight": 0.2, "critical": False},
            "vegetation_indices": {"weight": 0.1, "critical": False}
        }
    
    async def perform_comparative_analysis(
        self,
        sub_queries: List[SubQuery],
        recommendations_by_subquery: Dict[str, List[DatasetRecommendation]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive comparative analysis across data types.
        
        Args:
            sub_queries: List of decomposed sub-queries
            recommendations_by_subquery: Recommendations for each sub-query
            
        Returns:
            Comprehensive comparison results
        """
        analysis_results = {}
        
        # Group recommendations by data type
        recommendations_by_type = self._group_by_data_type(
            sub_queries, recommendations_by_subquery
        )
        
        # Perform data type comparisons
        if len(recommendations_by_type) > 1:
            analysis_results["data_type_comparison"] = await self._compare_data_types(
                recommendations_by_type
            )
        
        # Perform temporal coverage analysis
        analysis_results["temporal_analysis"] = await self._analyze_temporal_coverage(
            recommendations_by_type
        )
        
        # Identify data fusion opportunities
        analysis_results["fusion_opportunities"] = await self._identify_fusion_opportunities(
            recommendations_by_type
        )
        
        # Generate drought monitoring assessment
        analysis_results["drought_suitability"] = await self._assess_drought_monitoring_suitability(
            recommendations_by_type
        )
        
        # Create overall comparison table
        analysis_results["comparison_table"] = self._create_comparison_table(
            recommendations_by_type
        )
        
        return analysis_results
    
    def _group_by_data_type(
        self,
        sub_queries: List[SubQuery],
        recommendations_by_subquery: Dict[str, List[DatasetRecommendation]]
    ) -> Dict[DataType, List[DatasetRecommendation]]:
        """Group recommendations by data type."""
        grouped = defaultdict(list)
        
        for sub_query in sub_queries:
            if sub_query.id in recommendations_by_subquery:
                recommendations = recommendations_by_subquery[sub_query.id]
                for data_type in sub_query.data_types:
                    grouped[data_type].extend(recommendations)
        
        return dict(grouped)
    
    async def _compare_data_types(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> List[ComparisonResult]:
        """Compare different data types."""
        comparison_results = []
        
        for data_type, recommendations in recommendations_by_type.items():
            if not recommendations:
                continue
                
            # Calculate temporal coverage
            temporal_coverage = self._calculate_temporal_coverage(recommendations)
            
            # Calculate spatial coverage  
            spatial_coverage = self._calculate_spatial_coverage(recommendations)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(recommendations)
            
            # Identify gaps
            gaps = self._identify_comprehensive_gaps(recommendations)
            
            # Determine strengths and weaknesses
            strengths, weaknesses = self._assess_strengths_weaknesses(
                data_type, recommendations, temporal_coverage, quality_metrics
            )
            
            comparison_results.append(ComparisonResult(
                data_type=data_type,
                dataset_count=len(recommendations),
                temporal_coverage=temporal_coverage,
                spatial_coverage=spatial_coverage,
                quality_metrics=quality_metrics,
                gaps_identified=gaps,
                strengths=strengths,
                weaknesses=weaknesses
            ))
        
        return comparison_results
    
    def _calculate_temporal_coverage(
        self,
        recommendations: List[DatasetRecommendation]
    ) -> Dict[str, Any]:
        """Calculate comprehensive temporal coverage metrics."""
        if not recommendations:
            return {}
        
        # Extract all temporal ranges
        temporal_ranges = []
        for rec in recommendations:
            if rec.collection.temporal_extent:
                start = rec.collection.temporal_extent.get("start")
                end = rec.collection.temporal_extent.get("end") 
                if start and end:
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                        temporal_ranges.append((start_dt, end_dt))
                    except:
                        continue
        
        if not temporal_ranges:
            return {"error": "No temporal data available"}
        
        # Calculate coverage statistics
        earliest_start = min(tr[0] for tr in temporal_ranges)
        latest_end = max(tr[1] for tr in temporal_ranges)
        total_span = latest_end - earliest_start
        
        # Calculate coverage density
        coverage_days = set()
        for start, end in temporal_ranges:
            current = start
            while current <= end:
                coverage_days.add(current.date())
                current += timedelta(days=1)
        
        total_possible_days = total_span.days + 1
        coverage_percentage = len(coverage_days) / total_possible_days * 100
        
        return {
            "earliest_start": earliest_start.isoformat(),
            "latest_end": latest_end.isoformat(),
            "total_span_days": total_span.days,
            "coverage_percentage": coverage_percentage,
            "datasets_with_temporal_data": len(temporal_ranges),
            "average_dataset_duration": sum((tr[1] - tr[0]).days for tr in temporal_ranges) / len(temporal_ranges)
        }
    
    def _calculate_spatial_coverage(
        self,
        recommendations: List[DatasetRecommendation]
    ) -> Dict[str, Any]:
        """Calculate spatial coverage metrics."""
        if not recommendations:
            return {}
        
        spatial_extents = []
        for rec in recommendations:
            if rec.collection.spatial_extent:
                spatial_extents.append(rec.collection.spatial_extent)
        
        if not spatial_extents:
            return {"error": "No spatial data available"}
        
        # Calculate bounding box union
        all_norths = [ext.get("north", 90) for ext in spatial_extents if ext.get("north")]
        all_souths = [ext.get("south", -90) for ext in spatial_extents if ext.get("south")]
        all_easts = [ext.get("east", 180) for ext in spatial_extents if ext.get("east")]
        all_wests = [ext.get("west", -180) for ext in spatial_extents if ext.get("west")]
        
        return {
            "bounding_box": {
                "north": max(all_norths) if all_norths else None,
                "south": min(all_souths) if all_souths else None,
                "east": max(all_easts) if all_easts else None,
                "west": min(all_wests) if all_wests else None
            },
            "datasets_with_spatial_data": len(spatial_extents),
            "global_coverage": any(
                abs(ext.get("north", 0) - ext.get("south", 0)) > 170 
                for ext in spatial_extents
            )
        }
    
    def _calculate_quality_metrics(
        self,
        recommendations: List[DatasetRecommendation]
    ) -> Dict[str, float]:
        """Calculate aggregated quality metrics."""
        if not recommendations:
            return {}
        
        avg_relevance = sum(rec.relevance_score for rec in recommendations) / len(recommendations)
        avg_coverage = sum(rec.coverage_score for rec in recommendations) / len(recommendations)
        avg_quality = sum(rec.quality_score for rec in recommendations) / len(recommendations)
        avg_accessibility = sum(rec.accessibility_score for rec in recommendations) / len(recommendations)
        
        return {
            "average_relevance": avg_relevance,
            "average_coverage": avg_coverage,
            "average_quality": avg_quality,
            "average_accessibility": avg_accessibility,
            "overall_score": (avg_relevance + avg_coverage + avg_quality + avg_accessibility) / 4
        }
    
    def _identify_comprehensive_gaps(
        self,
        recommendations: List[DatasetRecommendation]
    ) -> List[Dict[str, Any]]:
        """Identify comprehensive gaps across all datasets."""
        gaps = []
        
        # Aggregate temporal gaps from all recommendations
        all_temporal_gaps = []
        for rec in recommendations:
            if rec.temporal_gaps:
                all_temporal_gaps.extend(rec.temporal_gaps)
        
        # Sort and merge overlapping gaps
        if all_temporal_gaps:
            sorted_gaps = sorted(all_temporal_gaps, key=lambda g: g.get("start", ""))
            merged_gaps = self._merge_temporal_gaps(sorted_gaps)
            gaps.extend(merged_gaps)
        
        return gaps
    
    def _merge_temporal_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping temporal gaps."""
        if not gaps:
            return []
        
        merged = [gaps[0]]
        
        for current in gaps[1:]:
            last_merged = merged[-1]
            
            # Try to parse dates for comparison
            try:
                last_end = datetime.fromisoformat(last_merged["end"].replace('Z', '+00:00'))
                current_start = datetime.fromisoformat(current["start"].replace('Z', '+00:00'))
                
                # If gaps overlap or are adjacent, merge them
                if current_start <= last_end + timedelta(days=1):
                    current_end = datetime.fromisoformat(current["end"].replace('Z', '+00:00'))
                    if current_end > last_end:
                        last_merged["end"] = current["end"]
                        last_merged["duration_days"] = (current_end - datetime.fromisoformat(
                            last_merged["start"].replace('Z', '+00:00')
                        )).days
                else:
                    merged.append(current)
            except:
                # If date parsing fails, just add as separate gap
                merged.append(current)
        
        return merged
    
    def _assess_strengths_weaknesses(
        self,
        data_type: DataType,
        recommendations: List[DatasetRecommendation],
        temporal_coverage: Dict[str, Any],
        quality_metrics: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Assess strengths and weaknesses of data type."""
        strengths = []
        weaknesses = []
        
        # Data type specific assessments
        if data_type == DataType.SATELLITE:
            strengths.append("Global spatial coverage")
            strengths.append("Consistent temporal sampling")
            if quality_metrics.get("average_accessibility", 0) > 0.7:
                strengths.append("High data accessibility")
            
            if temporal_coverage.get("coverage_percentage", 0) < 80:
                weaknesses.append("Incomplete temporal coverage")
            weaknesses.append("Potential cloud contamination")
            
        elif data_type == DataType.GROUND_BASED:
            strengths.append("High measurement accuracy")
            strengths.append("Direct observations")
            if quality_metrics.get("average_quality", 0) > 0.8:
                strengths.append("Excellent data quality")
            
            weaknesses.append("Limited spatial coverage")
            weaknesses.append("Sparse station distribution")
            if len(recommendations) < 5:
                weaknesses.append("Limited dataset availability")
        
        # General quality-based assessments
        if quality_metrics.get("average_relevance", 0) > 0.8:
            strengths.append("High relevance to query")
        if quality_metrics.get("average_coverage", 0) > 0.8:
            strengths.append("Good temporal/spatial coverage")
            
        if quality_metrics.get("average_accessibility", 0) < 0.5:
            weaknesses.append("Limited data accessibility")
        
        return strengths, weaknesses
    
    async def _analyze_temporal_coverage(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> Dict[str, Any]:
        """Analyze temporal coverage across data types."""
        coverage_analysis = {}
        
        for data_type, recommendations in recommendations_by_type.items():
            temporal_coverage = self._calculate_temporal_coverage(recommendations)
            coverage_analysis[data_type.value] = temporal_coverage
        
        # Compare coverage between data types
        if len(recommendations_by_type) > 1:
            coverage_analysis["comparison"] = self._compare_temporal_coverage(
                recommendations_by_type
            )
        
        return coverage_analysis
    
    def _compare_temporal_coverage(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> Dict[str, Any]:
        """Compare temporal coverage between data types."""
        comparison = {}
        
        # Calculate coverage stats for each type
        type_stats = {}
        for data_type, recommendations in recommendations_by_type.items():
            temporal_coverage = self._calculate_temporal_coverage(recommendations)
            type_stats[data_type.value] = temporal_coverage
        
        # Find best and worst coverage
        coverage_scores = {
            dt: stats.get("coverage_percentage", 0) 
            for dt, stats in type_stats.items()
        }
        
        if coverage_scores:
            best_coverage = max(coverage_scores.items(), key=lambda x: x[1])
            worst_coverage = min(coverage_scores.items(), key=lambda x: x[1])
            
            comparison.update({
                "best_coverage": {
                    "data_type": best_coverage[0],
                    "percentage": best_coverage[1]
                },
                "worst_coverage": {
                    "data_type": worst_coverage[0], 
                    "percentage": worst_coverage[1]
                },
                "coverage_gap": best_coverage[1] - worst_coverage[1]
            })
        
        return comparison
    
    async def _identify_fusion_opportunities(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> List[DataFusionOpportunity]:
        """Identify data fusion opportunities."""
        opportunities = []
        
        # For precipitation + drought monitoring specifically
        if DataType.SATELLITE in recommendations_by_type and DataType.GROUND_BASED in recommendations_by_type:
            satellite_recs = recommendations_by_type[DataType.SATELLITE]
            ground_recs = recommendations_by_type[DataType.GROUND_BASED]
            
            # Find best satellite dataset
            best_satellite = max(satellite_recs, key=lambda r: r.relevance_score) if satellite_recs else None
            
            if best_satellite:
                opportunities.append(DataFusionOpportunity(
                    primary_dataset=best_satellite.collection.title,
                    complementary_datasets=[r.collection.title for r in ground_recs[:3]],
                    fusion_type="calibration_validation",
                    benefits=[
                        "Improved accuracy through ground truth validation",
                        "Bias correction for satellite measurements",
                        "Enhanced drought monitoring capabilities"
                    ],
                    requirements=[
                        "Temporal overlap between datasets",
                        "Spatial co-location of measurements",
                        "Quality control procedures"
                    ],
                    confidence_score=0.85
                ))
        
        return opportunities
    
    async def _assess_drought_monitoring_suitability(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> Dict[str, Any]:
        """Assess suitability for drought monitoring."""
        suitability = {}
        
        for data_type, recommendations in recommendations_by_type.items():
            drought_scores = []
            
            for rec in recommendations:
                # Score based on drought-relevant indicators
                score = 0.0
                
                # Check if dataset contains drought-relevant variables
                title_lower = rec.collection.title.lower()
                summary_lower = rec.collection.summary.lower() if rec.collection.summary else ""
                
                for indicator, config in self.drought_indicators.items():
                    if indicator.replace('_', ' ') in title_lower or indicator.replace('_', ' ') in summary_lower:
                        score += config["weight"]
                        if config["critical"]:
                            score += 0.1  # Bonus for critical indicators
                
                # Factor in temporal coverage
                temporal_coverage = rec.temporal_gaps if rec.temporal_gaps else []
                coverage_penalty = len(temporal_coverage) * 0.05
                score = max(0, score - coverage_penalty)
                
                drought_scores.append({
                    "dataset": rec.collection.title,
                    "drought_suitability_score": score,
                    "relevance_score": rec.relevance_score,
                    "combined_score": (score + rec.relevance_score) / 2
                })
            
            # Sort by combined score
            drought_scores.sort(key=lambda x: x["combined_score"], reverse=True)
            
            suitability[data_type.value] = {
                "ranked_datasets": drought_scores,
                "average_suitability": sum(d["drought_suitability_score"] for d in drought_scores) / len(drought_scores) if drought_scores else 0,
                "top_recommendation": drought_scores[0] if drought_scores else None
            }
        
        return suitability
    
    def _create_comparison_table(
        self,
        recommendations_by_type: Dict[DataType, List[DatasetRecommendation]]
    ) -> pd.DataFrame:
        """Create a comparison table for easy visualization."""
        rows = []
        
        for data_type, recommendations in recommendations_by_type.items():
            if not recommendations:
                continue
                
            temporal_coverage = self._calculate_temporal_coverage(recommendations)
            quality_metrics = self._calculate_quality_metrics(recommendations)
            
            rows.append({
                "Data Type": data_type.value.replace('_', ' ').title(),
                "Dataset Count": len(recommendations),
                "Avg Relevance": f"{quality_metrics.get('average_relevance', 0):.2f}",
                "Avg Coverage": f"{quality_metrics.get('average_coverage', 0):.2f}",
                "Avg Quality": f"{quality_metrics.get('average_quality', 0):.2f}",
                "Temporal Coverage %": f"{temporal_coverage.get('coverage_percentage', 0):.1f}%",
                "Temporal Span (days)": temporal_coverage.get('total_span_days', 0),
                "Top Dataset": recommendations[0].collection.title if recommendations else "None"
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()