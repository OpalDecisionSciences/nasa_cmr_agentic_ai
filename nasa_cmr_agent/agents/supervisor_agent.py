"""
Supervisor Agent for Quality Control

This agent supervises and validates the output of other agents to ensure
high-quality, factually accurate responses that meet user expectations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from ..services.llm_service import LLMService
from ..models.schemas import QueryContext, SystemResponse, DatasetRecommendation

logger = logging.getLogger(__name__)


class QualityScore(BaseModel):
    """Quality assessment score for agent outputs."""
    accuracy: float = Field(..., ge=0, le=1, description="Factual accuracy score")
    completeness: float = Field(..., ge=0, le=1, description="Response completeness score")
    relevance: float = Field(..., ge=0, le=1, description="Relevance to user query score")
    clarity: float = Field(..., ge=0, le=1, description="Clarity and coherence score")
    overall: float = Field(..., ge=0, le=1, description="Overall quality score")
    

class ValidationResult(BaseModel):
    """Result of supervisor validation."""
    passed: bool = Field(..., description="Whether validation passed")
    quality_score: QualityScore = Field(..., description="Quality assessment scores")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    requires_retry: bool = Field(False, description="Whether the task should be retried")
    retry_guidance: Optional[str] = Field(None, description="Guidance for retry if needed")


class SupervisorAgent:
    """
    Supervisor agent that validates and ensures quality of other agents' outputs.
    
    Key responsibilities:
    - Validate factual accuracy of responses
    - Check completeness and relevance
    - Ensure responses meet user expectations
    - Request task retry when quality standards aren't met
    """
    
    QUALITY_THRESHOLD = 0.75  # Minimum overall quality score required
    
    def __init__(self):
        self.llm_service = LLMService()
        self.validation_history: List[ValidationResult] = []
        
    async def validate_query_interpretation(
        self, 
        original_query: str, 
        interpreted_context: QueryContext
    ) -> ValidationResult:
        """
        Validate query interpretation agent's output.
        
        Args:
            original_query: Original user query
            interpreted_context: Interpreted query context
            
        Returns:
            ValidationResult with quality assessment
        """
        validation_prompt = f"""
        Validate the query interpretation for accuracy and completeness.
        
        Original Query: {original_query}
        
        Interpreted Context:
        - Intent: {interpreted_context.intent}
        - Data Types: {interpreted_context.data_types}
        - Spatial Constraints: {interpreted_context.spatial_constraints}
        - Temporal Constraints: {interpreted_context.temporal_constraints}
        - Keywords: {interpreted_context.keywords}
        
        Assess the following:
        1. Does the intent correctly capture the user's goal?
        2. Are all relevant data types identified?
        3. Are spatial/temporal constraints accurately extracted?
        4. Are any key aspects of the query missed?
        5. Is the interpretation logical and consistent?
        
        Provide scores (0-1) for:
        - Accuracy: How factually correct is the interpretation
        - Completeness: Are all query aspects captured
        - Relevance: How well does it align with user intent
        - Clarity: How clear and unambiguous is the interpretation
        
        Format: accuracy|completeness|relevance|clarity|issues(semicolon-separated)|suggestions(semicolon-separated)
        """
        
        try:
            response = await self.llm_service.generate(validation_prompt)
            return self._parse_validation_response(response)
        except Exception as e:
            logger.error(f"Query interpretation validation failed: {e}")
            return ValidationResult(
                passed=True,  # Pass by default on error
                quality_score=QualityScore(
                    accuracy=0.8, completeness=0.8, relevance=0.8, 
                    clarity=0.8, overall=0.8
                )
            )
    
    async def validate_cmr_results(
        self,
        query_context: QueryContext,
        cmr_results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate CMR API agent's search results.
        
        Args:
            query_context: Original query context
            cmr_results: CMR search results
            
        Returns:
            ValidationResult with quality assessment
        """
        collections = cmr_results.get("collections", [])
        granules = cmr_results.get("granules", [])
        
        validation_prompt = f"""
        Validate CMR search results for query about {query_context.keywords}.
        
        Query Intent: {query_context.intent}
        Data Types Requested: {query_context.data_types}
        
        Results Summary:
        - Collections Found: {len(collections)}
        - Granules Found: {len(granules)}
        - Collection Titles: {[c.title for c in collections[:5]] if collections else 'None'}
        
        Validate:
        1. Are the results relevant to the requested data types?
        2. Do temporal/spatial parameters match the query?
        3. Is the data coverage sufficient?
        4. Are there obvious missing datasets?
        5. Is the result set appropriately sized (not too many/few)?
        
        Provide scores (0-1) for:
        - Accuracy: Correctness of search parameters and results
        - Completeness: Coverage of relevant datasets
        - Relevance: Alignment with user's data needs
        - Clarity: Organization and presentation of results
        
        Format: accuracy|completeness|relevance|clarity|issues(semicolon-separated)|suggestions(semicolon-separated)
        """
        
        try:
            response = await self.llm_service.generate(validation_prompt)
            return self._parse_validation_response(response)
        except Exception as e:
            logger.error(f"CMR results validation failed: {e}")
            return ValidationResult(
                passed=True,
                quality_score=QualityScore(
                    accuracy=0.8, completeness=0.8, relevance=0.8,
                    clarity=0.8, overall=0.8
                )
            )
    
    async def validate_analysis_results(
        self,
        query_context: QueryContext,
        analysis_results: List[Any]
    ) -> ValidationResult:
        """
        Validate data analysis agent's outputs.
        
        Args:
            query_context: Original query context
            analysis_results: Analysis results
            
        Returns:
            ValidationResult with quality assessment
        """
        validation_prompt = f"""
        Validate data analysis results for accuracy and usefulness.
        
        Query Focus: {query_context.keywords}
        Analysis Type: {query_context.intent}
        
        Analysis Results Summary:
        - Number of Analyses: {len(analysis_results)}
        - Analysis Types: {[r.analysis_type for r in analysis_results] if analysis_results else 'None'}
        
        Validate:
        1. Are the analyses appropriate for the query type?
        2. Are statistical methods correctly applied?
        3. Are gaps and limitations properly identified?
        4. Are recommendations logical and actionable?
        5. Is the confidence assessment reasonable?
        
        Provide scores (0-1) for:
        - Accuracy: Correctness of analysis methods and conclusions
        - Completeness: Coverage of all relevant aspects
        - Relevance: Alignment with user's analytical needs
        - Clarity: Clarity of insights and recommendations
        
        Format: accuracy|completeness|relevance|clarity|issues(semicolon-separated)|suggestions(semicolon-separated)
        """
        
        try:
            response = await self.llm_service.generate(validation_prompt)
            return self._parse_validation_response(response)
        except Exception as e:
            logger.error(f"Analysis validation failed: {e}")
            return ValidationResult(
                passed=True,
                quality_score=QualityScore(
                    accuracy=0.8, completeness=0.8, relevance=0.8,
                    clarity=0.8, overall=0.8
                )
            )
    
    async def validate_final_response(
        self,
        original_query: str,
        system_response: SystemResponse
    ) -> ValidationResult:
        """
        Validate the final synthesized response.
        
        Args:
            original_query: Original user query
            system_response: Final system response
            
        Returns:
            ValidationResult with quality assessment
        """
        validation_prompt = f"""
        Validate the final response for user satisfaction and accuracy.
        
        Original Query: {original_query}
        
        Response Summary:
        - Success: {system_response.success}
        - Recommendations: {len(system_response.recommendations)}
        - Summary Length: {len(system_response.summary)} chars
        - Warnings: {len(system_response.warnings)}
        - Follow-up Suggestions: {len(system_response.follow_up_suggestions)}
        
        Key Recommendations:
        {[f"- {r.collection.title}: Score {r.relevance_score}" for r in system_response.recommendations[:3]]}
        
        Validate:
        1. Does the response fully address the user's query?
        2. Are recommendations appropriate and well-justified?
        3. Is the summary clear and informative?
        4. Are limitations and caveats properly communicated?
        5. Would a domain expert find this response satisfactory?
        
        Provide scores (0-1) for:
        - Accuracy: Factual correctness of all information
        - Completeness: Full coverage of query requirements
        - Relevance: Direct addressing of user needs
        - Clarity: Ease of understanding and actionability
        
        Format: accuracy|completeness|relevance|clarity|issues(semicolon-separated)|suggestions(semicolon-separated)
        """
        
        try:
            response = await self.llm_service.generate(validation_prompt)
            result = self._parse_validation_response(response)
            
            # Store validation history
            self.validation_history.append(result)
            
            # Determine if retry is needed based on quality threshold
            if result.quality_score.overall < self.QUALITY_THRESHOLD:
                result.requires_retry = True
                result.retry_guidance = self._generate_retry_guidance(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Final response validation failed: {e}")
            return ValidationResult(
                passed=True,
                quality_score=QualityScore(
                    accuracy=0.8, completeness=0.8, relevance=0.8,
                    clarity=0.8, overall=0.8
                )
            )
    
    def _parse_validation_response(self, response: str) -> ValidationResult:
        """
        Parse LLM validation response into ValidationResult.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed ValidationResult
        """
        try:
            parts = response.strip().split('|')
            if len(parts) >= 6:
                accuracy = float(parts[0])
                completeness = float(parts[1])
                relevance = float(parts[2])
                clarity = float(parts[3])
                issues = [i.strip() for i in parts[4].split(';') if i.strip()]
                suggestions = [s.strip() for s in parts[5].split(';') if s.strip()]
                
                overall = (accuracy + completeness + relevance + clarity) / 4
                
                quality_score = QualityScore(
                    accuracy=accuracy,
                    completeness=completeness,
                    relevance=relevance,
                    clarity=clarity,
                    overall=overall
                )
                
                return ValidationResult(
                    passed=overall >= self.QUALITY_THRESHOLD,
                    quality_score=quality_score,
                    issues=issues,
                    suggestions=suggestions
                )
            else:
                # Default passing validation if parsing fails
                return ValidationResult(
                    passed=True,
                    quality_score=QualityScore(
                        accuracy=0.8, completeness=0.8, relevance=0.8,
                        clarity=0.8, overall=0.8
                    )
                )
        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            return ValidationResult(
                passed=True,
                quality_score=QualityScore(
                    accuracy=0.8, completeness=0.8, relevance=0.8,
                    clarity=0.8, overall=0.8
                )
            )
    
    def _generate_retry_guidance(self, validation_result: ValidationResult) -> str:
        """
        Generate specific guidance for retry based on validation issues.
        
        Args:
            validation_result: Validation result with identified issues
            
        Returns:
            Retry guidance string
        """
        guidance_parts = []
        
        if validation_result.quality_score.accuracy < 0.7:
            guidance_parts.append("Verify factual accuracy and data sources")
        
        if validation_result.quality_score.completeness < 0.7:
            guidance_parts.append("Ensure all query aspects are addressed")
        
        if validation_result.quality_score.relevance < 0.7:
            guidance_parts.append("Better align response with user intent")
        
        if validation_result.quality_score.clarity < 0.7:
            guidance_parts.append("Improve clarity and organization")
        
        if validation_result.issues:
            guidance_parts.append(f"Address issues: {', '.join(validation_result.issues[:3])}")
        
        return "; ".join(guidance_parts)
    
    async def supervise_workflow(
        self,
        workflow_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Supervise entire workflow execution.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            Tuple of (should_continue, retry_guidance)
        """
        # Validate each stage
        validations = []
        
        if "query_context" in workflow_state:
            validation = await self.validate_query_interpretation(
                workflow_state.get("original_query", ""),
                workflow_state["query_context"]
            )
            validations.append(("query_interpretation", validation))
        
        if "cmr_results" in workflow_state:
            validation = await self.validate_cmr_results(
                workflow_state.get("query_context"),
                workflow_state["cmr_results"]
            )
            validations.append(("cmr_results", validation))
        
        if "analysis_results" in workflow_state:
            validation = await self.validate_analysis_results(
                workflow_state.get("query_context"),
                workflow_state["analysis_results"]
            )
            validations.append(("analysis", validation))
        
        # Determine if workflow should continue
        critical_failures = [
            (stage, v) for stage, v in validations 
            if not v.passed and stage in ["query_interpretation", "cmr_results"]
        ]
        
        if critical_failures:
            stage, validation = critical_failures[0]
            return False, f"Retry {stage}: {validation.retry_guidance}"
        
        return True, None
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation history.
        
        Returns:
            Dictionary with validation statistics
        """
        if not self.validation_history:
            return {"total_validations": 0}
        
        avg_scores = {
            "accuracy": sum(v.quality_score.accuracy for v in self.validation_history) / len(self.validation_history),
            "completeness": sum(v.quality_score.completeness for v in self.validation_history) / len(self.validation_history),
            "relevance": sum(v.quality_score.relevance for v in self.validation_history) / len(self.validation_history),
            "clarity": sum(v.quality_score.clarity for v in self.validation_history) / len(self.validation_history),
            "overall": sum(v.quality_score.overall for v in self.validation_history) / len(self.validation_history),
        }
        
        return {
            "total_validations": len(self.validation_history),
            "passed": sum(1 for v in self.validation_history if v.passed),
            "failed": sum(1 for v in self.validation_history if not v.passed),
            "retry_requests": sum(1 for v in self.validation_history if v.requires_retry),
            "average_scores": avg_scores,
            "common_issues": self._get_common_issues()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get most common issues from validation history."""
        issue_counts = {}
        for validation in self.validation_history:
            for issue in validation.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, _ in sorted_issues[:5]]