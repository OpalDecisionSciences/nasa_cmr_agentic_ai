import asyncio
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import structlog

from ..models.schemas import (
    QueryContext, SystemResponse, AnalysisResult, DatasetRecommendation,
    AgentResponse
)
from ..services.llm_service import LLMService

logger = structlog.get_logger(__name__)


class ResponseSynthesisAgent:
    """
    Response synthesis and formatting agent.
    
    Responsibilities:
    - Synthesize results from all agents into coherent response
    - Generate natural language summaries
    - Format recommendations and analysis results
    - Create follow-up suggestions
    - Handle error cases gracefully
    """
    
    def __init__(self):
        self.llm_service = LLMService()
    
    async def synthesize_response(
        self,
        query_context: Optional[QueryContext],
        cmr_results: Dict[str, Any],
        analysis_results: List[AnalysisResult],
        errors: List[str],
        agent_responses: Optional[List[AgentResponse]] = None
    ) -> SystemResponse:
        """
        Synthesize comprehensive system response.
        
        Args:
            query_context: Original query context
            cmr_results: Results from CMR API searches
            analysis_results: Analysis results from analysis agent
            errors: List of errors encountered
            agent_responses: Responses from individual agents
            
        Returns:
            Complete SystemResponse with recommendations and summary
        """
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        # Handle case where query processing failed
        if not query_context:
            return self._create_error_response(
                query_id=query_id,
                original_query="Unknown query",
                errors=errors or ["Query processing failed"],
                agent_responses=agent_responses or []
            )
        
        try:
            # Extract recommendations from analysis results
            recommendations = self._extract_recommendations(analysis_results)
            
            # Generate summary (enhanced for comparative analysis)
            summary = await self._generate_summary(
                query_context, cmr_results, analysis_results, errors
            )
            
            # Create execution plan
            execution_plan = self._create_execution_plan(
                query_context, len(cmr_results.get("collections", [])), analysis_results
            )
            
            # Generate follow-up suggestions
            follow_up_suggestions = await self._generate_follow_up_suggestions(
                query_context, recommendations, analysis_results
            )
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Determine success status
            success = len(errors) == 0 and len(recommendations) > 0
            
            response = SystemResponse(
                query_id=query_id,
                original_query=query_context.original_query,
                intent=query_context.intent,
                recommendations=recommendations,
                analysis_results=analysis_results,
                summary=summary,
                execution_plan=execution_plan,
                total_execution_time_ms=execution_time,
                agent_responses=agent_responses or [],
                success=success,
                warnings=errors,
                follow_up_suggestions=follow_up_suggestions
            )
            
            logger.info("Response synthesis completed", 
                       query_id=query_id, success=success, 
                       recommendation_count=len(recommendations))
            
            return response
            
        except Exception as e:
            logger.error("Response synthesis failed", error=str(e))
            
            return self._create_error_response(
                query_id=query_id,
                original_query=query_context.original_query,
                errors=errors + [f"Response synthesis failed: {str(e)}"],
                agent_responses=agent_responses or []
            )
    
    def _extract_recommendations(self, analysis_results: List[AnalysisResult]) -> List[DatasetRecommendation]:
        """Extract dataset recommendations from analysis results."""
        recommendations = []
        
        for result in analysis_results:
            if result.analysis_type == "dataset_recommendations":
                rec_data = result.results.get("recommendations", [])
                for rec_dict in rec_data:
                    try:
                        recommendation = DatasetRecommendation(**rec_dict)
                        recommendations.append(recommendation)
                    except Exception as e:
                        logger.warning("Failed to parse recommendation", error=str(e))
                        continue
        
        return recommendations
    
    async def _generate_summary(
        self,
        query_context: QueryContext,
        cmr_results: Dict[str, Any],
        analysis_results: List[AnalysisResult],
        errors: List[str]
    ) -> str:
        """Generate natural language summary of results."""
        try:
            # Prepare summary context
            collections_count = len(cmr_results.get("collections", []))
            granules_count = len(cmr_results.get("granules", []))
            
            recommendations_count = 0
            for result in analysis_results:
                if result.analysis_type == "dataset_recommendations":
                    recommendations_count = len(result.results.get("recommendations", []))
                    break
            
            # Create summary prompt
            summary_prompt = f"""
            Summarize the results for this NASA data discovery query: "{query_context.original_query}"
            
            Results found:
            - {collections_count} collections
            - {granules_count} granules  
            - {recommendations_count} recommendations generated
            
            Query intent: {query_context.intent.value}
            Research domain: {query_context.research_domain or "General"}
            
            Analysis performed: {', '.join([r.analysis_type for r in analysis_results])}
            
            Errors encountered: {len(errors)}
            
            Write a concise 2-3 sentence summary focusing on what was found and key insights.
            Be specific about data availability and quality for the user's needs.
            If comparative analysis was performed, highlight key comparisons between data types.
            """
            
            summary = await self.llm_service.generate(summary_prompt, max_tokens=150)
            return summary.strip()
            
        except Exception as e:
            logger.warning("Failed to generate AI summary, using fallback", error=str(e))
            
            # Fallback to rule-based summary
            return self._create_fallback_summary(query_context, cmr_results, analysis_results, errors)
    
    def _create_fallback_summary(
        self,
        query_context: QueryContext,
        cmr_results: Dict[str, Any],
        analysis_results: List[AnalysisResult],
        errors: List[str]
    ) -> str:
        """Create rule-based summary when LLM fails."""
        collections_count = len(cmr_results.get("collections", []))
        granules_count = len(cmr_results.get("granules", []))
        
        if collections_count == 0:
            return f"No datasets found matching your query '{query_context.original_query}'. Consider broadening search criteria or checking for alternative keywords."
        
        summary_parts = [
            f"Found {collections_count} relevant dataset{'s' if collections_count != 1 else ''}"
        ]
        
        if granules_count > 0:
            summary_parts.append(f"with {granules_count} data granules available")
        
        # Add analysis insights
        for result in analysis_results:
            if result.analysis_type == "dataset_recommendations":
                rec_count = len(result.results.get("recommendations", []))
                if rec_count > 0:
                    summary_parts.append(f"Top {rec_count} datasets recommended based on relevance and coverage")
            
            elif result.analysis_type == "coverage_analysis":
                summary_parts.append("Coverage analysis completed")
            
            elif result.analysis_type == "temporal_gap_analysis":
                gaps = result.results.get("statistics", {}).get("total_gaps", 0)
                if gaps > 0:
                    summary_parts.append(f"Identified {gaps} temporal gaps in coverage")
        
        if errors:
            summary_parts.append(f"Note: {len(errors)} warning{'s' if len(errors) != 1 else ''} encountered during processing")
        
        return ". ".join(summary_parts) + "."
    
    def _create_execution_plan(
        self,
        query_context: QueryContext,
        collections_found: int,
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """Create execution plan summary."""
        plan = [
            "Query interpretation and validation",
            f"CMR API search (found {collections_found} collections)"
        ]
        
        if collections_found > 0:
            plan.append("Granule search and metadata retrieval")
        
        # Add analysis steps
        analysis_types = [r.analysis_type for r in analysis_results]
        
        if "dataset_recommendations" in analysis_types:
            plan.append("Dataset ranking and recommendation generation")
        
        if "coverage_analysis" in analysis_types:
            plan.append("Temporal and spatial coverage analysis")
        
        if "temporal_gap_analysis" in analysis_types:
            plan.append("Temporal gap identification")
        
        if "dataset_relationship_analysis" in analysis_types:
            plan.append("Cross-dataset relationship analysis")
        
        if "comparative_analysis" in analysis_types:
            plan.append("Multi-data-type comparative analysis")
        
        plan.append("Response synthesis and formatting")
        
        return plan
    
    async def _generate_follow_up_suggestions(
        self,
        query_context: QueryContext,
        recommendations: List[DatasetRecommendation],
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """Generate intelligent follow-up suggestions."""
        suggestions = []
        
        try:
            # Intent-specific suggestions
            if query_context.intent.value in ["comparative", "analytical"]:
                if len(recommendations) > 1:
                    suggestions.append("Consider combining multiple datasets for comprehensive analysis")
                
                # Look for complementary data
                complementary_found = False
                for rec in recommendations:
                    if rec.complementary_datasets:
                        complementary_found = True
                        break
                
                if complementary_found:
                    suggestions.append("Explore complementary datasets for multi-sensor analysis")
                
                # Check for fusion opportunities from comparative analysis
                for result in analysis_results:
                    if result.analysis_type == "comparative_analysis":
                        fusion_opps = result.results.get("fusion_opportunities", [])
                        if fusion_opps:
                            suggestions.append("Data fusion opportunities identified for enhanced analysis")
            
            # Temporal analysis suggestions
            if query_context.intent.value == "temporal_analysis":
                # Check for gaps
                for result in analysis_results:
                    if (result.analysis_type == "temporal_gap_analysis" and
                        result.results.get("statistics", {}).get("total_gaps", 0) > 0):
                        suggestions.append("Address temporal gaps by combining datasets or interpolating data")
                        break
            
            # Coverage-based suggestions
            if query_context.constraints.spatial:
                suggestions.append("Verify spatial coverage meets your study area requirements")
            
            if query_context.constraints.temporal:
                suggestions.append("Check temporal resolution compatibility with your analysis needs")
            
            # Data access suggestions
            cloud_datasets = sum(1 for rec in recommendations if rec.collection.cloud_hosted)
            if cloud_datasets > 0:
                suggestions.append(f"Consider cloud-hosted datasets ({cloud_datasets} available) for faster access")
            
            # Quality suggestions
            high_quality = sum(1 for rec in recommendations if rec.quality_score > 0.8)
            if high_quality > 0:
                suggestions.append("Prioritize high-quality datasets for most reliable results")
            
            # Research domain suggestions
            if query_context.research_domain:
                domain_suggestions = {
                    "climate": "Consider long-term climate data records (CDRs) for trend analysis",
                    "hydrology": "Combine satellite and in-situ observations for validation",
                    "agriculture": "Use vegetation indices and meteorological data together",
                    "fire": "Include pre and post-fire data for impact assessment"
                }
                
                if query_context.research_domain in domain_suggestions:
                    suggestions.append(domain_suggestions[query_context.research_domain])
            
            # Limit suggestions
            return suggestions[:5]
            
        except Exception as e:
            logger.warning("Failed to generate follow-up suggestions", error=str(e))
            
            # Basic fallback suggestions
            return [
                "Review dataset documentation before download",
                "Consider data processing requirements for your analysis",
                "Verify coordinate systems and projections match your needs"
            ]
    
    def _create_error_response(
        self,
        query_id: str,
        original_query: str,
        errors: List[str],
        agent_responses: List[AgentResponse]
    ) -> SystemResponse:
        """Create error response when processing fails."""
        return SystemResponse(
            query_id=query_id,
            original_query=original_query,
            intent="exploratory",  # Default intent
            recommendations=[],
            analysis_results=[],
            summary=f"Unable to process query due to errors: {'; '.join(errors[:3])}",
            execution_plan=["error_handling"],
            total_execution_time_ms=0,
            agent_responses=agent_responses,
            success=False,
            warnings=errors,
            follow_up_suggestions=[
                "Try rephrasing your query with more specific terms",
                "Check your spatial and temporal constraints",
                "Contact support if the issue persists"
            ]
        )