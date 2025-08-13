import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import WeaviateHybridSearchRetriever

from ..models.schemas import QueryContext, CMRCollection
from ..services.vector_database import VectorDatabaseService
from ..services.llm_service import LLMService

logger = structlog.get_logger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service for NASA CMR data discovery.
    
    Provides:
    - Dynamic context retrieval based on query similarity
    - Augmented query processing with relevant documentation
    - Enhanced dataset recommendations with contextual information
    - Multi-hop reasoning over dataset relationships
    """
    
    def __init__(self):
        self.vector_db = VectorDatabaseService()
        self.llm_service = LLMService()
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup RAG-specific prompts for enhanced responses."""
        
        self.context_enhanced_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a NASA Earth science data expert with access to comprehensive dataset documentation.

Use the provided context about similar datasets and documentation to enhance your analysis of the user's query.

Context Information:
{context}

Instructions:
1. Use the context to identify relevant datasets and their characteristics
2. Explain relationships between datasets mentioned in the context
3. Highlight unique capabilities or limitations of specific datasets
4. Suggest complementary datasets based on the contextual information
5. Provide domain-specific insights based on the documentation

Be specific and reference the contextual information in your response."""),
            ("human", "User Query: {query}\n\nPlease provide an enhanced analysis using the contextual information.")
        ])
        
        self.documentation_synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize information from multiple NASA dataset documentation sources to answer the user's question.

Documentation Sources:
{documentation}

Your task:
1. Extract key information relevant to the user's query
2. Identify common themes and patterns across datasets  
3. Highlight unique features or capabilities
4. Explain technical details in accessible language
5. Provide actionable recommendations

Focus on practical applications and scientific value."""),
            ("human", "Question: {question}")
        ])
        
        self.relationship_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze relationships between NASA Earth science datasets based on the provided information.

Dataset Information:
{dataset_info}

Analyze:
1. Complementary measurement capabilities
2. Temporal and spatial overlap opportunities  
3. Validation and cross-calibration potential
4. Data fusion possibilities
5. Research application synergies

Provide specific, actionable insights about how these datasets can work together."""),
            ("human", "Analyze the relationships between these datasets for: {research_objective}")
        ])
    
    async def enhance_query_with_context(
        self, 
        query_context: QueryContext,
        max_context_items: int = 5
    ) -> Tuple[QueryContext, List[Dict[str, Any]]]:
        """
        Enhance query processing with relevant contextual information.
        
        Args:
            query_context: Original parsed query context
            max_context_items: Maximum number of context items to retrieve
            
        Returns:
            Tuple of enhanced query context and retrieved context items
        """
        try:
            # Retrieve relevant context based on query
            context_items = await self.vector_db.semantic_search(
                query=query_context.original_query,
                limit=max_context_items
            )
            
            if not context_items:
                logger.info("No contextual information found for query enhancement")
                return query_context, []
            
            # Extract additional insights from context
            enhanced_constraints = await self._extract_constraints_from_context(
                query_context, context_items
            )
            
            # Update query context with enhanced information
            enhanced_context = QueryContext(
                original_query=query_context.original_query,
                intent=query_context.intent,
                constraints=enhanced_constraints,
                priority_score=query_context.priority_score,
                complexity_score=query_context.complexity_score,
                research_domain=query_context.research_domain,
                methodology_hints=query_context.methodology_hints
            )
            
            logger.info(f"Enhanced query context with {len(context_items)} contextual items")
            return enhanced_context, context_items
            
        except Exception as e:
            logger.error(f"Failed to enhance query with context: {e}")
            return query_context, []
    
    async def generate_contextual_recommendations(
        self,
        query_context: QueryContext,
        initial_collections: List[CMRCollection],
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate enhanced recommendations using contextual information.
        
        Args:
            query_context: Enhanced query context
            initial_collections: Initial dataset collections found
            context_items: Retrieved contextual information
            
        Returns:
            Dictionary containing enhanced recommendations and insights
        """
        try:
            # Create context string from retrieved items
            context_text = self._format_context_for_llm(context_items)
            
            # Generate enhanced analysis using LLM
            enhanced_analysis = await self.llm_service.generate(
                self.context_enhanced_analysis_prompt.format(
                    context=context_text,
                    query=query_context.original_query
                ),
                max_tokens=500
            )
            
            # Find similar datasets for each initial collection
            similarity_insights = []
            for collection in initial_collections[:3]:  # Limit to top 3
                similar = await self.vector_db.find_similar_datasets(
                    collection.concept_id,
                    limit=3
                )
                if similar:
                    similarity_insights.append({
                        "dataset": collection.short_name or collection.title,
                        "similar_datasets": similar
                    })
            
            # Generate relationship analysis
            relationship_analysis = ""
            if len(initial_collections) > 1:
                dataset_info = self._format_datasets_for_analysis(initial_collections)
                relationship_analysis = await self.llm_service.generate(
                    self.relationship_analysis_prompt.format(
                        dataset_info=dataset_info,
                        research_objective=query_context.original_query
                    ),
                    max_tokens=400
                )
            
            return {
                "enhanced_analysis": enhanced_analysis,
                "similarity_insights": similarity_insights,
                "relationship_analysis": relationship_analysis,
                "context_items_used": len(context_items)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate contextual recommendations: {e}")
            return {
                "enhanced_analysis": "Enhanced analysis unavailable due to processing error",
                "similarity_insights": [],
                "relationship_analysis": "",
                "context_items_used": 0
            }
    
    async def retrieve_documentation_context(
        self,
        dataset_ids: List[str],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation for specific datasets.
        
        Args:
            dataset_ids: List of dataset concept IDs
            query: User query for context-relevant retrieval
            
        Returns:
            List of documentation context items
        """
        documentation_context = []
        
        try:
            for dataset_id in dataset_ids:
                dataset_info = await self.vector_db.get_dataset_by_id(dataset_id)
                if dataset_info:
                    # Find additional context for this specific dataset
                    similar_docs = await self.vector_db.find_similar_datasets(
                        dataset_id, 
                        limit=2
                    )
                    
                    documentation_context.append({
                        "dataset_id": dataset_id,
                        "primary_info": dataset_info,
                        "related_datasets": similar_docs,
                        "relevance_context": query
                    })
            
            return documentation_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve documentation context: {e}")
            return []
    
    async def synthesize_multi_dataset_insights(
        self,
        datasets: List[CMRCollection],
        query_context: QueryContext
    ) -> str:
        """
        Synthesize insights across multiple datasets using RAG.
        
        Args:
            datasets: List of dataset collections to analyze
            query_context: User's query context
            
        Returns:
            Synthesized insights text
        """
        try:
            # Retrieve documentation for each dataset
            dataset_ids = [d.concept_id for d in datasets]
            docs_context = await self.retrieve_documentation_context(
                dataset_ids,
                query_context.original_query
            )
            
            # Format documentation for synthesis
            documentation_text = self._format_documentation_for_synthesis(docs_context)
            
            # Generate synthesized insights
            synthesized_insights = await self.llm_service.generate(
                self.documentation_synthesis_prompt.format(
                    documentation=documentation_text,
                    question=query_context.original_query
                ),
                max_tokens=600
            )
            
            return synthesized_insights
            
        except Exception as e:
            logger.error(f"Failed to synthesize multi-dataset insights: {e}")
            return "Multi-dataset synthesis unavailable due to processing error."
    
    async def generate_methodology_suggestions(
        self,
        query_context: QueryContext,
        recommended_datasets: List[CMRCollection]
    ) -> List[str]:
        """
        Generate methodology suggestions based on contextual information.
        
        Args:
            query_context: User's query context
            recommended_datasets: List of recommended datasets
            
        Returns:
            List of methodology suggestions
        """
        try:
            # Search for methodology-related context
            methodology_query = f"methodology analysis approach {query_context.research_domain} {' '.join(query_context.methodology_hints)}"
            
            method_context = await self.vector_db.semantic_search(
                query=methodology_query,
                limit=3
            )
            
            # Generate suggestions using context
            suggestions = []
            
            # Add domain-specific suggestions
            if query_context.research_domain:
                domain_methods = self._get_domain_methodologies(query_context.research_domain)
                suggestions.extend(domain_methods)
            
            # Add dataset-specific suggestions
            for dataset in recommended_datasets[:3]:
                dataset_methods = self._get_dataset_methodologies(dataset)
                suggestions.extend(dataset_methods)
            
            # Add context-enhanced suggestions
            if method_context:
                context_methods = self._extract_methodologies_from_context(method_context)
                suggestions.extend(context_methods)
            
            # Remove duplicates and limit
            unique_suggestions = list(set(suggestions))
            return unique_suggestions[:8]
            
        except Exception as e:
            logger.error(f"Failed to generate methodology suggestions: {e}")
            return ["Consider standard statistical analysis approaches for your research domain"]
    
    async def _extract_constraints_from_context(
        self,
        query_context: QueryContext,
        context_items: List[Dict[str, Any]]
    ):
        """Extract additional constraints from contextual information."""
        enhanced_constraints = query_context.constraints
        
        # Extract additional platforms and instruments from context
        context_platforms = set()
        context_instruments = set()
        context_variables = set()
        
        for item in context_items:
            if item.get("platforms"):
                context_platforms.update(item["platforms"])
            if item.get("instruments"):
                context_instruments.update(item["instruments"])
            if item.get("variables"):
                context_variables.update(item["variables"])
        
        # Add to existing constraints if not already present
        all_platforms = list(set(enhanced_constraints.platforms + list(context_platforms)))
        all_instruments = list(set(enhanced_constraints.instruments + list(context_instruments)))
        all_variables = list(set(enhanced_constraints.variables + list(context_variables)))
        
        # Create enhanced constraints object
        from ..models.schemas import QueryConstraints
        enhanced_constraints = QueryConstraints(
            spatial=enhanced_constraints.spatial,
            temporal=enhanced_constraints.temporal,
            data_types=enhanced_constraints.data_types,
            keywords=enhanced_constraints.keywords,
            platforms=all_platforms[:10],  # Limit to avoid noise
            instruments=all_instruments[:10],
            variables=all_variables[:15],
            resolution_requirements=enhanced_constraints.resolution_requirements
        )
        
        return enhanced_constraints
    
    def _format_context_for_llm(self, context_items: List[Dict[str, Any]]) -> str:
        """Format context items for LLM consumption."""
        formatted_context = []
        
        for i, item in enumerate(context_items, 1):
            context_piece = f"""
Dataset {i}: {item.get('title', 'Unknown')}
- Concept ID: {item.get('concept_id', 'N/A')}
- Summary: {item.get('summary', 'No summary available')[:200]}...
- Platforms: {', '.join(item.get('platforms', []))}
- Instruments: {', '.join(item.get('instruments', []))}
- Data Center: {item.get('data_center', 'N/A')}
- Similarity Score: {item.get('similarity_score', 0):.3f}
            """.strip()
            formatted_context.append(context_piece)
        
        return "\n\n".join(formatted_context)
    
    def _format_datasets_for_analysis(self, datasets: List[CMRCollection]) -> str:
        """Format datasets for relationship analysis."""
        formatted_datasets = []
        
        for dataset in datasets:
            dataset_info = f"""
Dataset: {dataset.title}
- Short Name: {dataset.short_name}
- Platforms: {', '.join(dataset.platforms) if dataset.platforms else 'N/A'}
- Instruments: {', '.join(dataset.instruments) if dataset.instruments else 'N/A'}
- Data Center: {dataset.data_center}
- Summary: {dataset.summary[:150] if dataset.summary else 'No summary'}...
- Temporal Coverage: {dataset.temporal_coverage}
- Cloud Hosted: {dataset.cloud_hosted}
            """.strip()
            formatted_datasets.append(dataset_info)
        
        return "\n\n".join(formatted_datasets)
    
    def _format_documentation_for_synthesis(self, docs_context: List[Dict[str, Any]]) -> str:
        """Format documentation context for synthesis."""
        formatted_docs = []
        
        for doc_ctx in docs_context:
            primary = doc_ctx.get("primary_info", {})
            related = doc_ctx.get("related_datasets", [])
            
            doc_text = f"""
Primary Dataset: {primary.get('title', 'Unknown')}
- Summary: {primary.get('summary', 'No summary available')}
- Platforms: {', '.join(primary.get('platforms', []))}
- Variables: {', '.join(primary.get('variables', []))}

Related Datasets:
{chr(10).join([f"- {r.get('title', 'Unknown')} (Similarity: {r.get('similarity_score', 0):.2f})" for r in related])}
            """.strip()
            formatted_docs.append(doc_text)
        
        return "\n\n".join(formatted_docs)
    
    def _get_domain_methodologies(self, domain: str) -> List[str]:
        """Get methodology suggestions for specific research domains."""
        domain_methods = {
            "hydrology": [
                "Time series analysis for temporal trends",
                "Statistical correlation with ground station data",
                "Hydrological modeling validation"
            ],
            "climate": [
                "Long-term trend analysis with climate data records",
                "Anomaly detection using standardized indices",
                "Multi-decadal climatology comparison"
            ],
            "agriculture": [
                "Vegetation index time series analysis",
                "Crop phenology monitoring",
                "Yield correlation studies"
            ],
            "fire": [
                "Burn area progression analysis",
                "Fire weather correlation studies",
                "Pre/post fire impact assessment"
            ]
        }
        
        return domain_methods.get(domain, [])
    
    def _get_dataset_methodologies(self, dataset: CMRCollection) -> List[str]:
        """Get methodology suggestions for specific datasets."""
        suggestions = []
        
        # Platform-specific methodologies
        if "MODIS" in str(dataset.platforms):
            suggestions.append("Multi-temporal MODIS composite analysis")
        
        if "Landsat" in str(dataset.platforms):
            suggestions.append("Landsat time series change detection")
        
        if "GPM" in str(dataset.title) or "precipitation" in str(dataset.summary).lower():
            suggestions.append("Precipitation validation with gauge networks")
        
        # Processing level specific
        if dataset.processing_level:
            if "L3" in dataset.processing_level:
                suggestions.append("Gridded product spatial analysis")
            elif "L2" in dataset.processing_level:
                suggestions.append("Swath-based temporal compositing")
        
        return suggestions
    
    def _extract_methodologies_from_context(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract methodology suggestions from contextual information."""
        methodologies = []
        
        for item in context:
            # Look for methodology hints in summaries
            summary = item.get("summary", "").lower()
            
            if "time series" in summary:
                methodologies.append("Time series analysis approach")
            
            if "validation" in summary:
                methodologies.append("Cross-validation with reference data")
            
            if "composite" in summary or "mosaic" in summary:
                methodologies.append("Temporal compositing techniques")
            
            if "anomaly" in summary:
                methodologies.append("Anomaly detection methods")
        
        return methodologies
    
    async def close(self):
        """Clean up RAG service resources."""
        await self.vector_db.close()