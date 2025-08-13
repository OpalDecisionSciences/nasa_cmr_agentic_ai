"""
Supervisor Learning Integration

Enables agents to learn from supervisor feedback while maintaining human-centric optimization.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..agents.supervisor_agent import ValidationResult
from .adaptive_learning import adaptive_learner

logger = logging.getLogger(__name__)


class SupervisorLearningIntegration:
    """
    Integrates supervisor feedback into the adaptive learning system
    with appropriate safeguards.
    """
    
    # Weight configuration for different feedback sources
    FEEDBACK_WEIGHTS = {
        "human": 0.7,      # Primary weight to human feedback
        "supervisor": 0.3   # Secondary weight to supervisor
    }
    
    # Minimum confidence for supervisor feedback to be used
    MIN_SUPERVISOR_CONFIDENCE = 0.8
    
    # Maximum divergence allowed between supervisor and human feedback
    MAX_DIVERGENCE_THRESHOLD = 0.3
    
    def __init__(self):
        self.divergence_history = []
        self.supervisor_accuracy_tracking = []
        
    async def process_supervisor_feedback(
        self,
        query: str,
        context: Any,
        validation_result: ValidationResult,
        agent_name: str
    ):
        """
        Process supervisor feedback for learning, with safeguards.
        
        Args:
            query: Original query
            context: Query context
            validation_result: Supervisor validation result
            agent_name: Name of the agent being evaluated
        """
        # Only learn from high-confidence supervisor feedback
        if validation_result.quality_score.overall < self.MIN_SUPERVISOR_CONFIDENCE:
            logger.debug(
                f"Supervisor confidence too low for learning: {validation_result.quality_score.overall}"
            )
            return
        
        # Check for recent human feedback on similar queries
        recent_human_feedback = self._get_recent_human_feedback(query, context)
        
        if recent_human_feedback:
            # Calculate divergence between supervisor and human assessment
            divergence = self._calculate_divergence(
                validation_result,
                recent_human_feedback
            )
            
            self.divergence_history.append({
                "timestamp": datetime.now(),
                "divergence": divergence,
                "agent": agent_name
            })
            
            # If divergence is too high, skip learning or reduce weight
            if divergence > self.MAX_DIVERGENCE_THRESHOLD:
                logger.warning(
                    f"High divergence between supervisor and human feedback: {divergence}"
                )
                # Could trigger supervisor recalibration here
                return
        
        # Apply weighted learning from supervisor feedback
        self._apply_weighted_learning(
            query,
            context,
            validation_result,
            agent_name,
            weight=self.FEEDBACK_WEIGHTS["supervisor"]
        )
    
    def _apply_weighted_learning(
        self,
        query: str,
        context: Any,
        validation_result: ValidationResult,
        agent_name: str,
        weight: float
    ):
        """
        Apply learning with appropriate weight.
        
        Args:
            query: Query text
            context: Query context
            validation_result: Validation result
            agent_name: Agent name
            weight: Learning weight (0-1)
        """
        # Extract learning signals from validation
        success = validation_result.passed
        quality_score = validation_result.quality_score.overall
        
        # Weighted success indicator
        weighted_success = success if weight == 1.0 else (quality_score * weight)
        
        # Update pattern learning with weighted success
        adaptive_learner.learn_from_query(
            query=query,
            context=context,
            response={"validation": validation_result.model_dump()},
            execution_time=0,  # Not relevant for validation
            success=weighted_success > 0.5
        )
        
        # Track specific issues for targeted improvement
        if validation_result.issues:
            self._track_improvement_areas(agent_name, validation_result.issues)
        
        # Store successful patterns from high-quality validations
        if quality_score > 0.9:
            self._store_excellence_pattern(query, context, validation_result)
    
    def _calculate_divergence(
        self,
        supervisor_result: ValidationResult,
        human_feedback: Dict[str, Any]
    ) -> float:
        """
        Calculate divergence between supervisor and human assessment.
        
        Args:
            supervisor_result: Supervisor validation
            human_feedback: Human feedback data
            
        Returns:
            Divergence score (0-1, lower is better)
        """
        # Normalize human rating to 0-1 scale
        human_score = (human_feedback.get("rating", 3) - 1) / 4
        supervisor_score = supervisor_result.quality_score.overall
        
        # Calculate absolute difference
        divergence = abs(human_score - supervisor_score)
        
        # Factor in specific disagreements
        if human_feedback.get("helpful") != supervisor_result.passed:
            divergence += 0.2
        
        return min(divergence, 1.0)
    
    def _get_recent_human_feedback(
        self,
        query: str,
        context: Any,
        window_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get recent human feedback for similar queries.
        
        Args:
            query: Current query
            context: Query context
            window_hours: Time window to consider
            
        Returns:
            Recent human feedback if found
        """
        # This would query the adaptive learner's feedback history
        # For now, return None (would be implemented with actual storage)
        return None
    
    def _track_improvement_areas(
        self,
        agent_name: str,
        issues: list
    ):
        """
        Track areas needing improvement for specific agents.
        
        Args:
            agent_name: Name of agent
            issues: List of issues identified
        """
        # This would update agent-specific improvement tracking
        logger.info(f"Agent {agent_name} improvement areas: {issues}")
    
    def _store_excellence_pattern(
        self,
        query: str,
        context: Any,
        validation_result: ValidationResult
    ):
        """
        Store patterns from excellent validations for replication.
        
        Args:
            query: Query that received excellent validation
            context: Query context
            validation_result: Excellent validation result
        """
        # Store as a reference pattern for future queries
        logger.info(
            f"Storing excellence pattern with score {validation_result.quality_score.overall}"
        )
    
    def recalibrate_supervisor(self) -> Dict[str, Any]:
        """
        Analyze divergence patterns and suggest supervisor recalibration.
        
        Returns:
            Recalibration recommendations
        """
        if not self.divergence_history:
            return {"status": "No divergence data available"}
        
        recent_divergences = [
            d["divergence"] for d in self.divergence_history[-100:]
        ]
        
        avg_divergence = sum(recent_divergences) / len(recent_divergences)
        
        recommendations = {
            "average_divergence": avg_divergence,
            "needs_recalibration": avg_divergence > 0.2,
            "suggested_adjustments": []
        }
        
        if avg_divergence > 0.3:
            recommendations["suggested_adjustments"].append(
                "Supervisor is significantly misaligned with human feedback"
            )
        elif avg_divergence > 0.2:
            recommendations["suggested_adjustments"].append(
                "Minor supervisor calibration recommended"
            )
        
        return recommendations
    
    def get_learning_effectiveness(self) -> Dict[str, Any]:
        """
        Measure the effectiveness of supervisor-based learning.
        
        Returns:
            Effectiveness metrics
        """
        return {
            "total_supervisor_feedback": len(self.divergence_history),
            "average_divergence": sum(d["divergence"] for d in self.divergence_history) / len(self.divergence_history) if self.divergence_history else 0,
            "high_divergence_count": sum(1 for d in self.divergence_history if d["divergence"] > self.MAX_DIVERGENCE_THRESHOLD),
            "supervisor_weight": self.FEEDBACK_WEIGHTS["supervisor"],
            "human_weight": self.FEEDBACK_WEIGHTS["human"]
        }


# Global instance
supervisor_learning = SupervisorLearningIntegration()