"""
Adaptive Learning Service

Enables the system to learn from interactions and improve over time.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryPattern(BaseModel):
    """Represents a query pattern learned from user interactions."""
    pattern_id: str
    keywords: List[str]
    intent: str
    data_types: List[str]
    success_rate: float
    avg_response_time: float
    usage_count: int
    last_used: datetime
    successful_parameters: Dict[str, Any] = Field(default_factory=dict)


class UserFeedback(BaseModel):
    """User feedback on system responses."""
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class AdaptiveLearningService:
    """
    Service that learns from user interactions to improve system performance.
    
    Features:
    - Pattern recognition from successful queries
    - Parameter optimization based on outcomes
    - User preference learning
    - Query suggestion generation
    """
    
    def __init__(self, storage_dir: str = "data/learning"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.user_feedback: List[UserFeedback] = []
        self.parameter_success_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.query_embeddings: Dict[str, np.ndarray] = {}
        
        self._load_learning_data()
    
    def learn_from_query(
        self,
        query: str,
        context: Any,
        response: Any,
        execution_time: float,
        success: bool
    ):
        """
        Learn from a query execution.
        
        Args:
            query: Original query
            context: Query context
            response: System response
            execution_time: Time taken to execute
            success: Whether query was successful
        """
        # Extract pattern features
        pattern_features = self._extract_pattern_features(query, context)
        pattern_id = self._generate_pattern_id(pattern_features)
        
        # Update or create pattern
        if pattern_id in self.query_patterns:
            pattern = self.query_patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # Update success rate
            pattern.success_rate = (
                (pattern.success_rate * (pattern.usage_count - 1) + (1 if success else 0))
                / pattern.usage_count
            )
            
            # Update average response time
            pattern.avg_response_time = (
                (pattern.avg_response_time * (pattern.usage_count - 1) + execution_time)
                / pattern.usage_count
            )
        else:
            # Create new pattern
            pattern = QueryPattern(
                pattern_id=pattern_id,
                keywords=pattern_features['keywords'],
                intent=pattern_features['intent'],
                data_types=pattern_features['data_types'],
                success_rate=1.0 if success else 0.0,
                avg_response_time=execution_time,
                usage_count=1,
                last_used=datetime.now()
            )
            self.query_patterns[pattern_id] = pattern
        
        # Learn successful parameters
        if success and hasattr(context, '__dict__'):
            self._update_successful_parameters(pattern, context.__dict__)
        
        # Save learning data
        self._save_learning_data()
    
    def suggest_query_improvements(self, query: str, context: Any) -> List[str]:
        """
        Suggest improvements to a query based on learned patterns.
        
        Args:
            query: User query
            context: Query context
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Find similar successful patterns
        similar_patterns = self._find_similar_patterns(query, context)
        
        for pattern in similar_patterns[:3]:
            if pattern.success_rate > 0.8:
                # Suggest successful parameters
                if pattern.successful_parameters:
                    param_suggestions = []
                    for param, value in pattern.successful_parameters.items():
                        if hasattr(context, param) and getattr(context, param) != value:
                            param_suggestions.append(f"Consider using {param}={value}")
                    
                    if param_suggestions:
                        suggestions.append(
                            f"Based on similar successful queries: {', '.join(param_suggestions[:2])}"
                        )
                
                # Suggest additional keywords
                current_keywords = set(query.lower().split())
                pattern_keywords = set(pattern.keywords)
                missing_keywords = pattern_keywords - current_keywords
                
                if missing_keywords:
                    suggestions.append(
                        f"Try including keywords: {', '.join(list(missing_keywords)[:3])}"
                    )
        
        return suggestions
    
    def get_optimal_parameters(
        self,
        intent: str,
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        Get optimal parameters for a given query type.
        
        Args:
            intent: Query intent
            data_types: Data types requested
            
        Returns:
            Optimal parameters
        """
        optimal_params = {}
        
        # Find patterns matching intent and data types
        matching_patterns = [
            p for p in self.query_patterns.values()
            if p.intent == intent and any(dt in p.data_types for dt in data_types)
        ]
        
        if not matching_patterns:
            return optimal_params
        
        # Sort by success rate and recency
        matching_patterns.sort(
            key=lambda p: (p.success_rate, -p.last_used.timestamp()),
            reverse=True
        )
        
        # Aggregate successful parameters
        param_scores = defaultdict(lambda: defaultdict(float))
        
        for pattern in matching_patterns[:10]:
            weight = pattern.success_rate * pattern.usage_count
            for param, value in pattern.successful_parameters.items():
                param_scores[param][str(value)] += weight
        
        # Select best value for each parameter
        for param, values in param_scores.items():
            best_value = max(values.items(), key=lambda x: x[1])
            try:
                # Try to convert back to original type
                optimal_params[param] = json.loads(best_value[0])
            except:
                optimal_params[param] = best_value[0]
        
        return optimal_params
    
    def record_user_feedback(self, feedback: UserFeedback):
        """
        Record user feedback for learning.
        
        Args:
            feedback: User feedback
        """
        self.user_feedback.append(feedback)
        
        # Update pattern success rates based on feedback
        if feedback.rating >= 4:
            # Positive feedback - reinforce patterns
            self._reinforce_recent_patterns(positive=True)
        elif feedback.rating <= 2:
            # Negative feedback - reduce pattern confidence
            self._reinforce_recent_patterns(positive=False)
        
        self._save_learning_data()
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on learned patterns.
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of query suggestions
        """
        suggestions = []
        query_lower = partial_query.lower()
        
        # Find patterns with matching keywords
        for pattern in self.query_patterns.values():
            if any(keyword.startswith(query_lower) for keyword in pattern.keywords):
                # Generate suggestion based on pattern
                suggestion = self._generate_suggestion_from_pattern(pattern)
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        # Sort by success rate and usage
        suggestions.sort(
            key=lambda s: self._score_suggestion(s),
            reverse=True
        )
        
        return suggestions[:5]
    
    def identify_common_failures(self) -> List[Dict[str, Any]]:
        """
        Identify common failure patterns.
        
        Returns:
            List of failure patterns with details
        """
        failures = []
        
        for pattern in self.query_patterns.values():
            if pattern.success_rate < 0.5 and pattern.usage_count >= 3:
                failures.append({
                    "pattern": pattern.keywords,
                    "intent": pattern.intent,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "avg_response_time": pattern.avg_response_time
                })
        
        # Sort by usage count (most common failures first)
        failures.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return failures[:10]
    
    def _extract_pattern_features(self, query: str, context: Any) -> Dict[str, Any]:
        """Extract features from query and context."""
        keywords = query.lower().split()
        
        features = {
            "keywords": keywords[:10],  # Limit keywords
            "intent": getattr(context, 'intent', 'unknown'),
            "data_types": getattr(context, 'data_types', [])
        }
        
        return features
    
    def _generate_pattern_id(self, features: Dict[str, Any]) -> str:
        """Generate unique pattern ID."""
        key_parts = [
            features['intent'],
            '_'.join(sorted(features['data_types']))[:50],
            '_'.join(sorted(features['keywords'][:5]))
        ]
        return '-'.join(key_parts)
    
    def _find_similar_patterns(
        self,
        query: str,
        context: Any,
        threshold: float = 0.5
    ) -> List[QueryPattern]:
        """Find patterns similar to current query."""
        query_features = self._extract_pattern_features(query, context)
        similar_patterns = []
        
        for pattern in self.query_patterns.values():
            similarity = self._calculate_similarity(query_features, pattern)
            if similarity > threshold:
                similar_patterns.append(pattern)
        
        # Sort by similarity and success rate
        similar_patterns.sort(
            key=lambda p: (
                self._calculate_similarity(query_features, p),
                p.success_rate
            ),
            reverse=True
        )
        
        return similar_patterns
    
    def _calculate_similarity(
        self,
        features: Dict[str, Any],
        pattern: QueryPattern
    ) -> float:
        """Calculate similarity between features and pattern."""
        # Keyword similarity
        feature_keywords = set(features['keywords'])
        pattern_keywords = set(pattern.keywords)
        
        if not feature_keywords or not pattern_keywords:
            keyword_sim = 0
        else:
            keyword_sim = len(feature_keywords & pattern_keywords) / len(
                feature_keywords | pattern_keywords
            )
        
        # Intent similarity
        intent_sim = 1.0 if features['intent'] == pattern.intent else 0.0
        
        # Data type similarity
        feature_types = set(features['data_types'])
        pattern_types = set(pattern.data_types)
        
        if not feature_types or not pattern_types:
            type_sim = 0
        else:
            type_sim = len(feature_types & pattern_types) / len(
                feature_types | pattern_types
            )
        
        # Weighted average
        return 0.4 * keyword_sim + 0.3 * intent_sim + 0.3 * type_sim
    
    def _update_successful_parameters(self, pattern: QueryPattern, params: Dict[str, Any]):
        """Update successful parameters for a pattern."""
        for key, value in params.items():
            if key not in ['messages', 'agent_responses', 'errors']:
                # Store serializable version
                try:
                    serializable_value = json.loads(json.dumps(value, default=str))
                    pattern.successful_parameters[key] = serializable_value
                except:
                    pass
    
    def _reinforce_recent_patterns(self, positive: bool, window_hours: int = 1):
        """Reinforce recent patterns based on feedback."""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        adjustment = 0.05 if positive else -0.05
        
        for pattern in self.query_patterns.values():
            if pattern.last_used > cutoff:
                # Adjust success rate with bounds
                pattern.success_rate = max(0, min(1, pattern.success_rate + adjustment))
    
    def _generate_suggestion_from_pattern(self, pattern: QueryPattern) -> str:
        """Generate query suggestion from pattern."""
        # Combine high-value keywords
        keywords = pattern.keywords[:5]
        
        # Add intent-specific terms
        intent_terms = {
            "exploratory": ["find", "show"],
            "specific": ["get", "retrieve"],
            "comparative": ["compare", "versus"],
            "analytical": ["analyze", "study"]
        }
        
        if pattern.intent in intent_terms:
            keywords = intent_terms[pattern.intent][:1] + keywords
        
        return " ".join(keywords)
    
    def _score_suggestion(self, suggestion: str) -> float:
        """Score a suggestion based on patterns."""
        score = 0.0
        
        for pattern in self.query_patterns.values():
            if any(kw in suggestion for kw in pattern.keywords):
                score += pattern.success_rate * pattern.usage_count
        
        return score
    
    def _load_learning_data(self):
        """Load learning data from storage."""
        patterns_file = self.storage_dir / "patterns.json"
        feedback_file = self.storage_dir / "feedback.json"
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = QueryPattern(**pattern_data)
                        self.query_patterns[pattern.pattern_id] = pattern
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.user_feedback = [UserFeedback(**fb) for fb in data]
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")
    
    def _save_learning_data(self):
        """Save learning data to storage."""
        patterns_file = self.storage_dir / "patterns.json"
        feedback_file = self.storage_dir / "feedback.json"
        
        try:
            # Save patterns
            patterns_data = [p.model_dump() for p in self.query_patterns.values()]
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, default=str, indent=2)
            
            # Save feedback (keep only recent)
            recent_feedback = self.user_feedback[-1000:]  # Keep last 1000
            feedback_data = [fb.model_dump() for fb in recent_feedback]
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning data."""
        if not self.query_patterns:
            return {"status": "No learning data available"}
        
        avg_success_rate = sum(
            p.success_rate for p in self.query_patterns.values()
        ) / len(self.query_patterns)
        
        most_successful = max(
            self.query_patterns.values(),
            key=lambda p: p.success_rate * p.usage_count
        )
        
        most_used = max(
            self.query_patterns.values(),
            key=lambda p: p.usage_count
        )
        
        return {
            "total_patterns": len(self.query_patterns),
            "average_success_rate": avg_success_rate,
            "total_feedback": len(self.user_feedback),
            "most_successful_pattern": {
                "keywords": most_successful.keywords,
                "success_rate": most_successful.success_rate,
                "usage_count": most_successful.usage_count
            },
            "most_used_pattern": {
                "keywords": most_used.keywords,
                "success_rate": most_used.success_rate,
                "usage_count": most_used.usage_count
            },
            "common_failures": self.identify_common_failures()[:3]
        }


# Global instance
adaptive_learner = AdaptiveLearningService()