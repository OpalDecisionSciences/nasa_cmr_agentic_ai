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
from ..services.vector_database import VectorDatabaseService
from ..services.rag_service import RAGService
from ..services.knowledge_graph import KnowledgeGraphService

logger = structlog.get_logger(__name__)


class EnhancedAnalysisAgent:
    """
    Enhanced data analysis and recommendation generation agent with advanced capabilities.
    
    Integrates:
    - Vector database semantic search
    - RAG-enhanced contextual analysis
    - Knowledge graph relationship discovery
    - Multi-modal recommendation scoring
    - Advanced analytics with external knowledge
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_db = VectorDatabaseService()
        self.rag_service = RAGService()
        self.knowledge_graph = KnowledgeGraphService()
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize advanced services."""
        try:
            # Initialize in background
            asyncio.create_task(self._async_initialize())
        except Exception as e:
            logger.warning(f"Failed to initialize advanced services: {e}")
    
    async def _async_initialize(self):
        """Asynchronously initialize services."""
        try:
            # Initialize schemas
            await self.vector_db.initialize_schema()
            await self.knowledge_graph.initialize_schema()
            logger.info("Advanced analysis services initialized")
        except Exception as e:
            logger.warning(f"Advanced services initialization warning: {e}")
    
    async def analyze_results_enhanced(
        self, 
        query_context: QueryContext, 
        cmr_results: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """
        Enhanced analysis using vector search, RAG, and knowledge graph.
        
        Args:
            query_context: Original query context and constraints
            cmr_results: Results from CMR API searches
            
        Returns:
            List of enhanced analysis results with advanced insights
        """
        collections = cmr_results.get("collections", [])
        granules = cmr_results.get("granules", [])
        
        analysis_results = []
        
        try:
            # Index collections in vector database for future searches
            if collections:
                try:
                    await self._index_collections_async(collections)
                except Exception as e:
                    logger.warning(f"Vector database indexing failed: {e}")
            
            # Enhance query with vector search context
            context_items = []
            enhanced_context = query_context
            try:
                enhanced_context, context_items = await self.rag_service.enhance_query_with_context(
                    query_context, max_context_items=8
                )
            except Exception as e:
                logger.warning(f"RAG enhancement failed: {e}")
            
            # Generate enhanced recommendations with RAG
            enhanced_recommendations = []
            try:
                enhanced_recommendations = await self._generate_enhanced_recommendations(
                    enhanced_context, collections, granules, context_items
                )
            except Exception as e:
                logger.warning(f"Enhanced recommendations failed: {e}")
                # Fall back to basic recommendations
                enhanced_recommendations = collections[:10] if collections else []
            
            if enhanced_recommendations:
                analysis_results.append(AnalysisResult(
                    analysis_type="enhanced_dataset_recommendations",
                    results={"recommendations": [rec.model_dump() if hasattr(rec, 'model_dump') else rec for rec in enhanced_recommendations]},
                    methodology="Multi-criteria scoring enhanced with vector similarity, RAG context, and knowledge graph relationships",
                    confidence_level=0.92,
                    statistics={
                        "total_datasets": len(collections), 
                        "recommended": len(enhanced_recommendations),
                        "context_items_used": len(context_items)
                    }
                ))
            
            # Perform knowledge graph relationship analysis
            if collections:
                try:
                    kg_relationships = await self._analyze_knowledge_graph_relationships(collections)
                    if kg_relationships:
                        analysis_results.append(kg_relationships)
                except Exception as e:
                    logger.warning(f"Knowledge graph analysis failed: {e}")
            
            # Vector similarity analysis
            if context_items:
                try:
                    similarity_analysis = await self._analyze_vector_similarities(
                        query_context, collections, context_items
                    )
                    if similarity_analysis:
                        analysis_results.append(similarity_analysis)
                except Exception as e:
                    logger.warning(f"Vector similarity analysis failed: {e}")
            
            # RAG-enhanced gap analysis
            if query_context.constraints.temporal:
                try:
                    rag_gap_analysis = await self._rag_enhanced_gap_analysis(
                        query_context, collections, granules
                    )
                    if rag_gap_analysis:
                        analysis_results.append(rag_gap_analysis)
                except Exception as e:
                    logger.warning(f"RAG gap analysis failed: {e}")
            
            # Data fusion opportunity analysis
            try:
                fusion_analysis = await self._analyze_fusion_opportunities(
                    query_context, collections[:5]  # Top 5 collections
                )
                if fusion_analysis:
                    analysis_results.append(fusion_analysis)
            except Exception as e:
                logger.warning(f"Data fusion analysis failed: {e}")
            
            # Research pathway analysis
            if query_context.research_domain:
                try:
                    pathway_analysis = await self._analyze_research_pathways(
                        query_context.research_domain, collections
                    )
                    if pathway_analysis:
                        analysis_results.append(pathway_analysis)
                except Exception as e:
                    logger.warning(f"Research pathway analysis failed: {e}")
            
            logger.info(f"Enhanced analysis completed with {len(analysis_results)} result types")
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            # Fallback to basic analysis
            basic_analysis = await self._basic_analysis_fallback(query_context, cmr_results)
            analysis_results.extend(basic_analysis)
        
        return analysis_results
    
    async def _index_collections_async(self, collections: List[CMRCollection]):
        """Index collections in vector database asynchronously."""
        try:
            # Index in batches for better performance
            batch_size = 50
            for i in range(0, len(collections), batch_size):
                batch = collections[i:i+batch_size]
                indexed_count = await self.vector_db.index_datasets_batch(batch)
                logger.debug(f"Indexed {indexed_count} datasets in vector database")
        except Exception as e:
            logger.warning(f"Failed to index collections: {e}")
    
    async def _generate_enhanced_recommendations(
        self, 
        query_context: QueryContext, 
        collections: List[CMRCollection],
        granules: List[CMRGranule],
        context_items: List[Dict[str, Any]]
    ) -> List[DatasetRecommendation]:
        """Generate enhanced recommendations using all available context."""
        
        recommendations = []
        
        # Group granules by collection
        granules_by_collection = defaultdict(list)
        for granule in granules:
            granules_by_collection[granule.collection_concept_id].append(granule)
        
        for collection in collections:
            try:
                # Basic scoring
                relevance_score = self._calculate_relevance_score(query_context, collection)
                coverage_score = self._calculate_coverage_score(query_context, collection)
                quality_score = self._calculate_quality_score(collection)
                accessibility_score = self._calculate_accessibility_score(collection)
                
                # Enhanced scoring with vector similarity
                vector_score = await self._calculate_vector_similarity_score(
                    collection, query_context.original_query
                )
                
                # Knowledge graph relationship scoring
                kg_score = await self._calculate_knowledge_graph_score(
                    collection, collections
                )
                
                # Context-based scoring
                context_score = self._calculate_context_score(collection, context_items)
                
                # Combined enhanced scoring
                enhanced_relevance = (relevance_score * 0.6 + 
                                    vector_score * 0.2 + 
                                    context_score * 0.2)
                
                enhanced_quality = (quality_score * 0.7 + kg_score * 0.3)
                
                # Get granules for this collection
                collection_granules = granules_by_collection.get(collection.concept_id, [])
                
                # Enhanced gap analysis
                temporal_gaps = await self._enhanced_temporal_gap_analysis(
                    query_context, collection_granules, context_items
                )
                
                # Knowledge graph relationships
                kg_relationships = await self._get_collection_relationships(collection.concept_id)
                
                # Generate enhanced reasoning with RAG
                reasoning = await self._generate_enhanced_reasoning(
                    query_context, collection, enhanced_relevance, coverage_score,
                    enhanced_quality, accessibility_score, kg_relationships
                )
                
                # Find complementary datasets using knowledge graph
                complementary_datasets = await self._find_kg_complementary_datasets(
                    collection, collections
                )
                
                recommendation = DatasetRecommendation(
                    collection=collection,
                    relevance_score=enhanced_relevance,
                    coverage_score=coverage_score,
                    quality_score=enhanced_quality,
                    accessibility_score=accessibility_score,
                    reasoning=reasoning,
                    granule_count=len(collection_granules),
                    temporal_gaps=temporal_gaps,
                    spatial_gaps=[],  # Enhanced spatial analysis could be added here
                    complementary_datasets=complementary_datasets
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(f"Failed to generate enhanced recommendation for {collection.concept_id}: {e}")
                continue
        
        # Enhanced ranking with multiple factors
        recommendations.sort(key=lambda r: (
            r.relevance_score * 0.35 + 
            r.coverage_score * 0.25 + 
            r.quality_score * 0.25 + 
            r.accessibility_score * 0.15
        ), reverse=True)
        
        return recommendations[:10]
    
    async def _calculate_vector_similarity_score(
        self, 
        collection: CMRCollection, 
        query: str
    ) -> float:
        """Calculate vector similarity score for a collection."""
        try:
            # Search for similar datasets
            similar_datasets = await self.vector_db.semantic_search(
                query=query,
                limit=20
            )
            
            # Find this collection in results
            for dataset in similar_datasets:
                if dataset.get("concept_id") == collection.concept_id:
                    return dataset.get("similarity_score", 0.0)
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Vector similarity calculation failed: {e}")
            return 0.0
    
    async def _calculate_knowledge_graph_score(
        self, 
        collection: CMRCollection, 
        all_collections: List[CMRCollection]
    ) -> float:
        """Calculate knowledge graph connectivity score."""
        try:
            # Ingest collection if not already in graph
            await self.knowledge_graph.ingest_dataset(collection)
            
            # Get relationships
            relationships = await self.knowledge_graph.find_dataset_relationships(
                collection.concept_id
            )
            
            # Calculate connectivity score
            total_relationships = sum(len(rel_list) for rel_list in relationships.values())
            
            # Normalize based on total available collections
            max_possible = len(all_collections) - 1
            if max_possible > 0:
                connectivity_score = min(total_relationships / max_possible, 1.0)
            else:
                connectivity_score = 0.0
            
            return connectivity_score * 0.8  # Scale down to avoid over-weighting
            
        except Exception as e:
            logger.debug(f"Knowledge graph score calculation failed: {e}")
            return 0.0
    
    def _calculate_context_score(
        self, 
        collection: CMRCollection, 
        context_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate score based on contextual similarity."""
        if not context_items:
            return 0.0
        
        score = 0.0
        
        # Check if collection appears in context
        for item in context_items:
            if item.get("concept_id") == collection.concept_id:
                score += item.get("similarity_score", 0.0) * 0.3
            
            # Check platform/instrument matches
            collection_platforms = set(collection.platforms)
            context_platforms = set(item.get("platforms", []))
            platform_overlap = len(collection_platforms.intersection(context_platforms))
            if platform_overlap > 0:
                score += platform_overlap * 0.1
            
            # Check for similar data centers
            if collection.data_center == item.get("data_center"):
                score += 0.1
        
        return min(score, 1.0)
    
    async def _analyze_knowledge_graph_relationships(
        self, 
        collections: List[CMRCollection]
    ) -> Optional[AnalysisResult]:
        """Analyze relationships using knowledge graph."""
        try:
            # Ingest collections into knowledge graph
            for collection in collections[:10]:  # Limit for performance
                await self.knowledge_graph.ingest_dataset(collection)
            
            # Analyze relationships for top collections
            relationship_analysis = {}
            
            for collection in collections[:5]:
                relationships = await self.knowledge_graph.find_dataset_relationships(
                    collection.concept_id
                )
                if relationships:
                    relationship_analysis[collection.concept_id] = {
                        "title": collection.title,
                        "relationships": relationships
                    }
            
            if relationship_analysis:
                return AnalysisResult(
                    analysis_type="knowledge_graph_relationships",
                    results=relationship_analysis,
                    methodology="Neo4j graph database relationship discovery and analysis",
                    confidence_level=0.88,
                    statistics={
                        "collections_analyzed": len(relationship_analysis),
                        "total_relationships": sum(
                            sum(len(rel_list) for rel_list in data["relationships"].values())
                            for data in relationship_analysis.values()
                        )
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Knowledge graph relationship analysis failed: {e}")
            return None
    
    async def _analyze_vector_similarities(
        self, 
        query_context: QueryContext,
        collections: List[CMRCollection],
        context_items: List[Dict[str, Any]]
    ) -> Optional[AnalysisResult]:
        """Analyze vector similarities and semantic relationships."""
        try:
            # Perform hybrid search to get additional insights
            hybrid_results = await self.vector_db.hybrid_search(
                query_context.original_query,
                query_context,
                limit=15
            )
            
            # Analyze similarity patterns
            similarity_patterns = {
                "high_similarity": [r for r in hybrid_results if r.get("combined_score", 0) > 0.8],
                "medium_similarity": [r for r in hybrid_results if 0.5 <= r.get("combined_score", 0) <= 0.8],
                "thematic_clusters": self._identify_thematic_clusters(hybrid_results),
                "platform_clusters": self._identify_platform_clusters(hybrid_results)
            }
            
            return AnalysisResult(
                analysis_type="vector_similarity_analysis",
                results=similarity_patterns,
                methodology="Weaviate vector database semantic similarity analysis with hybrid search",
                confidence_level=0.85,
                statistics={
                    "total_similar_datasets": len(hybrid_results),
                    "high_similarity_count": len(similarity_patterns["high_similarity"]),
                    "thematic_clusters_found": len(similarity_patterns["thematic_clusters"])
                }
            )
            
        except Exception as e:
            logger.error(f"Vector similarity analysis failed: {e}")
            return None
    
    async def _rag_enhanced_gap_analysis(
        self,
        query_context: QueryContext,
        collections: List[CMRCollection],
        granules: List[CMRGranule]
    ) -> Optional[AnalysisResult]:
        """Perform RAG-enhanced gap analysis with contextual insights."""
        try:
            # Get contextual information about gaps
            gap_context = await self.vector_db.semantic_search(
                f"temporal gaps data coverage {query_context.research_domain}",
                limit=5
            )
            
            # Generate insights using RAG
            gap_insights = await self.rag_service.synthesize_multi_dataset_insights(
                collections, query_context
            )
            
            # Standard gap analysis
            basic_gaps = {}
            granules_by_collection = defaultdict(list)
            for granule in granules:
                granules_by_collection[granule.collection_concept_id].append(granule)
            
            for collection in collections:
                collection_granules = granules_by_collection.get(collection.concept_id, [])
                gaps = self._identify_temporal_gaps(query_context, collection_granules)
                if gaps:
                    basic_gaps[collection.short_name or collection.title] = gaps
            
            return AnalysisResult(
                analysis_type="rag_enhanced_gap_analysis",
                results={
                    "traditional_gaps": basic_gaps,
                    "contextual_insights": gap_insights,
                    "context_datasets_used": len(gap_context)
                },
                methodology="RAG-enhanced temporal gap analysis with contextual dataset insights",
                confidence_level=0.90
            )
            
        except Exception as e:
            logger.error(f"RAG-enhanced gap analysis failed: {e}")
            return None
    
    async def _analyze_fusion_opportunities(
        self,
        query_context: QueryContext,
        collections: List[CMRCollection]
    ) -> Optional[AnalysisResult]:
        """Analyze data fusion opportunities using knowledge graph."""
        try:
            # Get fusion opportunities from knowledge graph
            primary_ids = [c.concept_id for c in collections]
            
            fusion_opportunities = await self.knowledge_graph.find_fusion_opportunities(
                query_context, primary_ids
            )
            
            if fusion_opportunities:
                # Enhance with methodology suggestions
                methodology_suggestions = await self.rag_service.generate_methodology_suggestions(
                    query_context, collections
                )
                
                return AnalysisResult(
                    analysis_type="data_fusion_opportunities",
                    results={
                        "fusion_opportunities": fusion_opportunities,
                        "methodology_suggestions": methodology_suggestions
                    },
                    methodology="Knowledge graph-based fusion opportunity identification with RAG-enhanced methodology suggestions",
                    confidence_level=0.87,
                    statistics={
                        "fusion_pairs_found": len(fusion_opportunities),
                        "methodology_suggestions": len(methodology_suggestions)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Fusion opportunity analysis failed: {e}")
            return None
    
    async def _analyze_research_pathways(
        self,
        research_domain: str,
        collections: List[CMRCollection]
    ) -> Optional[AnalysisResult]:
        """Analyze research pathways using knowledge graph."""
        try:
            # Map research domain to phenomena
            domain_phenomena_map = {
                "hydrology": "precipitation",
                "climate": "temperature",
                "agriculture": "vegetation", 
                "fire": "fire",
                "urban": "temperature"  # For urban heat island
            }
            
            phenomenon = domain_phenomena_map.get(research_domain, research_domain)
            
            # Get research pathway analysis
            pathway_analysis = await self.knowledge_graph.analyze_research_pathways(
                phenomenon, max_depth=3
            )
            
            if pathway_analysis:
                return AnalysisResult(
                    analysis_type="research_pathway_analysis",
                    results=pathway_analysis,
                    methodology="Knowledge graph traversal for research pathway discovery",
                    confidence_level=0.83,
                    statistics={
                        "datasets_in_pathway": pathway_analysis.get("total_datasets", 0),
                        "platforms_involved": len(pathway_analysis.get("platforms", [])),
                        "instruments_involved": len(pathway_analysis.get("instruments", []))
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Research pathway analysis failed: {e}")
            return None
    
    def _identify_thematic_clusters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify thematic clusters in search results."""
        clusters = defaultdict(list)
        
        for result in results:
            # Simple clustering based on keywords in title/summary
            title_summary = f"{result.get('title', '')} {result.get('summary', '')}".lower()
            
            themes = {
                "precipitation": ["precipitation", "rain", "precip"],
                "temperature": ["temperature", "thermal", "temp"],
                "vegetation": ["vegetation", "ndvi", "green"],
                "ocean": ["ocean", "sea", "marine"],
                "atmosphere": ["atmospheric", "air", "weather"]
            }
            
            for theme, keywords in themes.items():
                if any(keyword in title_summary for keyword in keywords):
                    clusters[theme].append(result)
                    break
        
        return [{"theme": theme, "datasets": datasets} for theme, datasets in clusters.items()]
    
    def _identify_platform_clusters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify platform-based clusters in search results."""
        platform_clusters = defaultdict(list)
        
        for result in results:
            platforms = result.get("platforms", [])
            for platform in platforms:
                platform_clusters[platform].append(result)
        
        return [{"platform": platform, "datasets": datasets} 
                for platform, datasets in platform_clusters.items() 
                if len(datasets) > 1]
    
    async def _enhanced_temporal_gap_analysis(
        self,
        query_context: QueryContext,
        granules: List[CMRGranule],
        context_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhanced temporal gap analysis with contextual information."""
        # Basic gap analysis
        basic_gaps = self._identify_temporal_gaps(query_context, granules)
        
        # Add context about typical data frequencies
        enhanced_gaps = []
        for gap in basic_gaps:
            enhanced_gap = gap.copy()
            
            # Add expected frequency information based on context
            gap_duration = gap.get("duration_days", 0)
            if gap_duration > 30:
                enhanced_gap["severity"] = "high"
                enhanced_gap["impact"] = "Significant gap may affect trend analysis"
            elif gap_duration > 7:
                enhanced_gap["severity"] = "medium"
                enhanced_gap["impact"] = "Moderate gap, consider interpolation"
            else:
                enhanced_gap["severity"] = "low"
                enhanced_gap["impact"] = "Minor gap, minimal impact on analysis"
            
            enhanced_gaps.append(enhanced_gap)
        
        return enhanced_gaps
    
    async def _get_collection_relationships(self, concept_id: str) -> List[str]:
        """Get relationships for a collection from knowledge graph."""
        try:
            relationships = await self.knowledge_graph.find_dataset_relationships(concept_id)
            
            related_datasets = []
            for rel_type, datasets in relationships.items():
                for dataset in datasets[:2]:  # Limit to 2 per type
                    related_datasets.append(dataset.get("short_name", dataset.get("title", "")))
            
            return related_datasets[:5]  # Limit total
            
        except Exception:
            return []
    
    async def _generate_enhanced_reasoning(
        self,
        query_context: QueryContext,
        collection: CMRCollection,
        relevance_score: float,
        coverage_score: float,
        quality_score: float,
        accessibility_score: float,
        relationships: List[str]
    ) -> str:
        """Generate enhanced reasoning with RAG context."""
        try:
            # Get contextual recommendations
            contextual_recs = await self.rag_service.generate_contextual_recommendations(
                query_context, [collection], []
            )
            
            enhanced_analysis = contextual_recs.get("enhanced_analysis", "")
            
            if enhanced_analysis and len(enhanced_analysis) > 50:
                return enhanced_analysis[:200] + "..."
            
            # Fallback to basic reasoning
            return self._generate_basic_reasoning(
                collection, relevance_score, coverage_score, quality_score, accessibility_score
            )
            
        except Exception:
            return self._generate_basic_reasoning(
                collection, relevance_score, coverage_score, quality_score, accessibility_score
            )
    
    def _generate_basic_reasoning(
        self,
        collection: CMRCollection,
        relevance_score: float,
        coverage_score: float,
        quality_score: float,
        accessibility_score: float
    ) -> str:
        """Generate basic reasoning for recommendations."""
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
    
    async def _find_kg_complementary_datasets(
        self,
        target_collection: CMRCollection,
        all_collections: List[CMRCollection]
    ) -> List[str]:
        """Find complementary datasets using knowledge graph."""
        try:
            relationships = await self.knowledge_graph.find_dataset_relationships(
                target_collection.concept_id
            )
            
            complementary = []
            
            # Get from same platform relationships
            for dataset in relationships.get("same_platform", []):
                complementary.append(dataset.get("short_name", dataset.get("title", "")))
            
            # Get from same instrument relationships  
            for dataset in relationships.get("same_instrument", []):
                complementary.append(dataset.get("short_name", dataset.get("title", "")))
            
            return list(set(complementary))[:3]  # Unique, limit to 3
            
        except Exception:
            return []
    
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
    
    async def _basic_analysis_fallback(
        self,
        query_context: QueryContext,
        cmr_results: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Fallback to basic analysis if advanced features fail."""
        from .analysis_agent import DataAnalysisAgent
        
        basic_agent = DataAnalysisAgent()
        return await basic_agent.analyze_results(query_context, cmr_results)
    
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
                score += 0.25
        else:
            score += 0.5
        
        # Spatial coverage (simplified)
        if (query_context.constraints.spatial and 
            collection.spatial_coverage):
            score += 0.5
        else:
            score += 0.5
        
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
        
        # Cloud hosting
        if collection.cloud_hosted:
            score += 0.2
        
        # Processing level
        if collection.processing_level:
            try:
                level_num = int(collection.processing_level.replace('L', ''))
                score += min(level_num / 4.0, 0.1)
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
        
        # Data center reliability
        nasa_centers = ['GSFC', 'JPL', 'LARC', 'MSFC', 'NSIDC', 'ORNL', 'SEDAC']
        if any(center in collection.data_center.upper() for center in nasa_centers):
            score += 0.2
        
        return min(score, 1.0)
    
    async def close(self):
        """Clean up resources."""
        try:
            await self.vector_db.close()
            await self.rag_service.close()
            await self.knowledge_graph.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")