import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
import structlog
import hashlib
import json
import uuid

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

from ..models.schemas import CMRCollection, QueryContext
from ..core.config import settings

logger = structlog.get_logger(__name__)


class VectorDatabaseService:
    """
    Advanced vector database service using Weaviate for semantic search.
    
    Provides:
    - Semantic search over NASA dataset documentation
    - Dynamic context retrieval based on query similarity
    - Embedding generation and management
    - Hybrid search combining vector similarity and metadata filtering
    """
    
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.collection_name = "NASADataset"
        self._initialize_client()
        self._initialize_embedding_model()
    
    def _initialize_client(self):
        """Initialize Weaviate client."""
        try:
            # For local Weaviate instance
            weaviate_url = getattr(settings, 'weaviate_url', 'http://localhost:8080')
            weaviate_api_key = getattr(settings, 'weaviate_api_key', None)
            
            if weaviate_api_key:
                auth_config = Auth.api_key(weaviate_api_key)
                self.client = weaviate.connect_to_custom(
                    http_host=weaviate_url.replace('http://', '').replace('https://', ''),
                    http_port=8080,
                    http_secure=False,
                    auth_credentials=auth_config
                )
            else:
                # Local development setup with gRPC disabled for Docker
                self.client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080,
                    grpc_port=50051,
                    skip_init_checks=True
                )
            
            logger.info("Weaviate client initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Weaviate client: {e}")
            self.client = None
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        try:
            model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def initialize_schema(self):
        """Initialize Weaviate schema for NASA datasets."""
        if not self.client:
            logger.warning("Weaviate client not available")
            return False
        
        try:
            # Check if collection already exists
            try:
                # Use proper collection checking method
                if self.client.collections.exists(self.collection_name):
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return True
            except Exception as e:
                logger.warning(f"Could not check collection existence: {e}. Attempting to create new collection.")
            
            # Create collection with robust schema handling for different Weaviate versions
            try:
                # Try modern configuration first
                collection = self.client.collections.create(
                    name=self.collection_name,
                    properties=[
                        wvc.config.Property(
                            name="concept_id",
                            data_type=wvc.config.DataType.TEXT,
                            description="NASA CMR concept ID"
                        ),
                        wvc.config.Property(
                            name="title",
                            data_type=wvc.config.DataType.TEXT,
                            description="Dataset title"
                        ),
                        wvc.config.Property(
                            name="summary",
                            data_type=wvc.config.DataType.TEXT,
                            description="Dataset summary/description"
                        ),
                        wvc.config.Property(
                            name="short_name",
                            data_type=wvc.config.DataType.TEXT,
                            description="Dataset short name"
                        ),
                        wvc.config.Property(
                            name="data_center",
                            data_type=wvc.config.DataType.TEXT,
                            description="Data center/provider"
                        ),
                        wvc.config.Property(
                            name="platforms",
                            data_type=wvc.config.DataType.TEXT_ARRAY,
                            description="Satellite platforms"
                        ),
                        wvc.config.Property(
                            name="instruments",
                            data_type=wvc.config.DataType.TEXT_ARRAY,
                            description="Instruments used"
                        ),
                        wvc.config.Property(
                            name="variables",
                            data_type=wvc.config.DataType.TEXT_ARRAY,
                            description="Measured variables"
                        ),
                        wvc.config.Property(
                            name="temporal_start",
                            data_type=wvc.config.DataType.DATE,
                            description="Temporal coverage start"
                        ),
                        wvc.config.Property(
                            name="temporal_end", 
                            data_type=wvc.config.DataType.DATE,
                            description="Temporal coverage end"
                        ),
                        wvc.config.Property(
                            name="spatial_coverage",
                            data_type=wvc.config.DataType.TEXT,
                            description="Spatial coverage description"
                        ),
                        wvc.config.Property(
                            name="processing_level",
                            data_type=wvc.config.DataType.TEXT,
                            description="Data processing level"
                        ),
                        wvc.config.Property(
                            name="keywords",
                            data_type=wvc.config.DataType.TEXT_ARRAY,
                            description="Science keywords"
                        ),
                        wvc.config.Property(
                            name="cloud_hosted",
                            data_type=wvc.config.DataType.BOOL,
                            description="Whether dataset is cloud hosted"
                        ),
                        wvc.config.Property(
                            name="online_access",
                            data_type=wvc.config.DataType.BOOL,
                            description="Online access availability"
                        ),
                        wvc.config.Property(
                            name="combined_text",
                            data_type=wvc.config.DataType.TEXT,
                            description="Combined searchable text"
                        ),
                        wvc.config.Property(
                            name="created_at",
                            data_type=wvc.config.DataType.DATE,
                            description="Record creation timestamp"
                        )
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
            except Exception as schema_error:
                logger.warning(f"Modern schema creation failed: {schema_error}")
                
                # Fallback: Try simplified schema without boolean fields
                try:
                    collection = self.client.collections.create(
                        name=self.collection_name,
                        properties=[
                            wvc.config.Property(
                                name="concept_id",
                                data_type=wvc.config.DataType.TEXT,
                                description="NASA CMR concept ID"
                            ),
                            wvc.config.Property(
                                name="title",
                                data_type=wvc.config.DataType.TEXT,
                                description="Dataset title"
                            ),
                            wvc.config.Property(
                                name="summary",
                                data_type=wvc.config.DataType.TEXT,
                                description="Dataset summary/description"
                            ),
                            wvc.config.Property(
                                name="platforms",
                                data_type=wvc.config.DataType.TEXT_ARRAY,
                                description="Satellite platforms"
                            ),
                            wvc.config.Property(
                                name="instruments",
                                data_type=wvc.config.DataType.TEXT_ARRAY,
                                description="Instruments used"
                            ),
                            wvc.config.Property(
                                name="variables",
                                data_type=wvc.config.DataType.TEXT_ARRAY,
                                description="Measured variables"
                            ),
                            wvc.config.Property(
                                name="combined_text",
                                data_type=wvc.config.DataType.TEXT,
                                description="Combined searchable text"
                            )
                        ]
                    )
                    logger.info(f"Created simplified Weaviate collection '{self.collection_name}'")
                except Exception as fallback_error:
                    logger.error(f"Fallback schema creation also failed: {fallback_error}")
                    return False
            
            logger.info(f"Created Weaviate collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate schema: {e}")
            return False
    
    async def index_dataset(self, collection: CMRCollection) -> bool:
        """Index a single dataset in the vector database."""
        if not self.client or not self.embedding_model:
            return False
        
        try:
            # Create combined text for semantic search
            combined_text = self._create_combined_text(collection)
            
            # Parse temporal coverage
            temporal_start = None
            temporal_end = None
            if collection.temporal_coverage:
                try:
                    if collection.temporal_coverage.get("start"):
                        temporal_start = datetime.fromisoformat(
                            collection.temporal_coverage["start"].replace('Z', '+00:00')
                        )
                    if collection.temporal_coverage.get("end"):
                        temporal_end = datetime.fromisoformat(
                            collection.temporal_coverage["end"].replace('Z', '+00:00')
                        )
                except (ValueError, TypeError):
                    pass
            
            # Prepare spatial coverage text
            spatial_text = ""
            if collection.spatial_coverage:
                if isinstance(collection.spatial_coverage, dict):
                    spatial_text = json.dumps(collection.spatial_coverage)
                else:
                    spatial_text = str(collection.spatial_coverage)
            
            # Extract keywords from summary and title
            keywords = self._extract_keywords(collection)
            
            # Prepare document for indexing
            document = {
                "concept_id": collection.concept_id,
                "title": collection.title or "",
                "summary": collection.summary or "",
                "short_name": collection.short_name or "",
                "data_center": collection.data_center or "",
                "platforms": collection.platforms or [],
                "instruments": collection.instruments or [],
                "variables": collection.variables or [],
                "temporal_start": temporal_start,
                "temporal_end": temporal_end,
                "spatial_coverage": spatial_text,
                "processing_level": collection.processing_level or "",
                "keywords": keywords,
                "cloud_hosted": collection.cloud_hosted,
                "online_access": collection.online_access_flag,
                "combined_text": combined_text,
                "created_at": datetime.now(timezone.utc)
            }
            
            # Get collection from client
            nasa_collection = self.client.collections.get(self.collection_name)
            
            # Generate proper UUID from concept_id
            # Create UUID5 from concept_id for consistent UUIDs
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
            document_uuid = uuid.uuid5(namespace, collection.concept_id)
            
            try:
                # Generate high-quality vector embedding for semantic search
                vector = None
                if self.embedding_model and combined_text:
                    try:
                        # Optimize text for NASA Earth science domain and generate embeddings
                        optimized_text = self._optimize_text_for_embedding(combined_text, collection)
                        vector = self.embedding_model.encode(optimized_text, convert_to_tensor=False)
                        if hasattr(vector, 'tolist'):
                            vector = vector.tolist()
                        elif isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                        logger.debug(f"Generated {len(vector)}-dimensional vector for {collection.concept_id}")
                    except Exception as vec_error:
                        logger.warning(f"Vector generation failed for {collection.concept_id}: {vec_error}")
                
                # Check if document already exists using concept_id as string UUID
                try:
                    existing = nasa_collection.query.fetch_object_by_id(str(document_uuid))
                except:
                    existing = None
                
                if existing:
                    # Update existing document with vector
                    if vector:
                        nasa_collection.data.update(
                            uuid=str(document_uuid),
                            properties=document,
                            vector=vector
                        )
                    else:
                        nasa_collection.data.update(
                            uuid=str(document_uuid),
                            properties=document
                        )
                    logger.debug(f"Updated dataset {collection.concept_id} in vector database")
                else:
                    # Insert new document with vector
                    if vector:
                        nasa_collection.data.insert(
                            uuid=str(document_uuid),
                            properties=document,
                            vector=vector
                        )
                    else:
                        nasa_collection.data.insert(
                            uuid=str(document_uuid),
                            properties=document
                        )
                    logger.debug(f"Indexed dataset {collection.concept_id} in vector database")
            
            except Exception as uuid_error:
                logger.warning(f"UUID operation failed for {collection.concept_id}: {uuid_error}")
                # Try without explicit UUID (let Weaviate generate)
                nasa_collection.data.insert(properties=document)
                logger.debug(f"Indexed dataset {collection.concept_id} with auto-generated UUID")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to index dataset {collection.concept_id}: {e}")
            return False
    
    async def index_datasets_batch(self, collections: List[CMRCollection]) -> int:
        """Index multiple datasets in batch for better performance."""
        if not self.client:
            return 0
        
        successful_indexes = 0
        
        try:
            nasa_collection = self.client.collections.get(self.collection_name)
            
            # Prepare batch data
            batch_data = []
            for collection in collections:
                try:
                    combined_text = self._create_combined_text(collection)
                    keywords = self._extract_keywords(collection)
                    
                    # Parse temporal coverage
                    temporal_start = None
                    temporal_end = None
                    if collection.temporal_coverage:
                        try:
                            if collection.temporal_coverage.get("start"):
                                temporal_start = datetime.fromisoformat(
                                    collection.temporal_coverage["start"].replace('Z', '+00:00')
                                )
                            if collection.temporal_coverage.get("end"):
                                temporal_end = datetime.fromisoformat(
                                    collection.temporal_coverage["end"].replace('Z', '+00:00')
                                )
                        except (ValueError, TypeError):
                            pass
                    
                    spatial_text = ""
                    if collection.spatial_coverage:
                        if isinstance(collection.spatial_coverage, dict):
                            spatial_text = json.dumps(collection.spatial_coverage)
                        else:
                            spatial_text = str(collection.spatial_coverage)
                    
                    document = {
                        "concept_id": collection.concept_id,
                        "title": collection.title or "",
                        "summary": collection.summary or "",
                        "short_name": collection.short_name or "",
                        "data_center": collection.data_center or "",
                        "platforms": collection.platforms or [],
                        "instruments": collection.instruments or [],
                        "variables": collection.variables or [],
                        "temporal_start": temporal_start,
                        "temporal_end": temporal_end,
                        "spatial_coverage": spatial_text,
                        "processing_level": collection.processing_level or "",
                        "keywords": keywords,
                        "cloud_hosted": collection.cloud_hosted,
                        "online_access": collection.online_access_flag,
                        "combined_text": combined_text,
                        "created_at": datetime.now(timezone.utc)
                    }
                    
                    # Generate proper UUID for batch operation
                    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                    document_uuid = uuid.uuid5(namespace, collection.concept_id)
                    
                    batch_data.append({
                        "uuid": str(document_uuid),
                        "properties": document
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare dataset {collection.concept_id} for batch: {e}")
                    continue
            
            # Execute batch insert with vectors
            if batch_data:
                with nasa_collection.batch.dynamic() as batch:
                    for item in batch_data:
                        # Generate optimized vector for each item
                        combined_text = item["properties"].get("combined_text", "")
                        vector = None
                        
                        if self.embedding_model and combined_text:
                            try:
                                # Find the original collection for this item
                                concept_id = item["properties"].get("concept_id")
                                original_collection = next((c for c in collections if c.concept_id == concept_id), None)
                                
                                if original_collection:
                                    optimized_text = self._optimize_text_for_embedding(combined_text, original_collection)
                                    vector = self.embedding_model.encode(optimized_text, convert_to_tensor=False)
                                else:
                                    vector = self.embedding_model.encode(combined_text, convert_to_tensor=False)
                                if hasattr(vector, 'tolist'):
                                    vector = vector.tolist()
                                elif isinstance(vector, np.ndarray):
                                    vector = vector.tolist()
                            except Exception as vec_error:
                                logger.warning(f"Batch vector generation failed: {vec_error}")
                        
                        if vector:
                            batch.add_object(
                                uuid=item["uuid"],
                                properties=item["properties"],
                                vector=vector
                            )
                        else:
                            batch.add_object(
                                uuid=item["uuid"],
                                properties=item["properties"]
                            )
                
                successful_indexes = len(batch_data)
                logger.info(f"Successfully batch indexed {successful_indexes} datasets")
            
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
        
        return successful_indexes
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search over indexed datasets."""
        if not self.client:
            logger.warning("Vector database not available")
            return []
        
        try:
            nasa_collection = self.client.collections.get(self.collection_name)
            
            # Build query
            query_builder = nasa_collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True, distance=True)
            )
            
            # Apply filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query_builder = query_builder.where(where_filter)
            
            # Execute search
            results = query_builder.objects
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "concept_id": result.properties.get("concept_id"),
                    "title": result.properties.get("title"),
                    "summary": result.properties.get("summary"),
                    "short_name": result.properties.get("short_name"),
                    "data_center": result.properties.get("data_center"),
                    "platforms": result.properties.get("platforms", []),
                    "instruments": result.properties.get("instruments", []),
                    "variables": result.properties.get("variables", []),
                    "keywords": result.properties.get("keywords", []),
                    "cloud_hosted": result.properties.get("cloud_hosted"),
                    "online_access": result.properties.get("online_access"),
                    "similarity_score": result.metadata.score if result.metadata else 0.0,
                    "distance": result.metadata.distance if result.metadata else 1.0
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Semantic search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        query_context: QueryContext,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic similarity and metadata filtering."""
        # Build filters from query context
        filters = self._build_context_filters(query_context)
        
        # Perform semantic search with filters
        semantic_results = await self.semantic_search(query, limit * 2, filters)
        
        # Re-rank results based on query context
        ranked_results = self._rerank_by_context(semantic_results, query_context)
        
        return ranked_results[:limit]
    
    async def find_similar_datasets(
        self, 
        concept_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find datasets similar to a given dataset."""
        if not self.client:
            return []
        
        try:
            nasa_collection = self.client.collections.get(self.collection_name)
            
            # Get the reference dataset using proper UUID format
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
            document_uuid = str(uuid.uuid5(namespace, concept_id))
            reference = nasa_collection.query.fetch_object_by_id(document_uuid)
            if not reference:
                return []
            
            # Use the combined text of reference for similarity search
            reference_text = reference.properties.get("combined_text", "")
            if not reference_text:
                return []
            
            # Find similar datasets
            results = nasa_collection.query.near_text(
                query=reference_text,
                limit=limit + 1,  # +1 to exclude self
                return_metadata=MetadataQuery(score=True)
            ).objects
            
            # Filter out the reference dataset itself and format results
            similar_datasets = []
            for result in results:
                if result.properties.get("concept_id") != concept_id:
                    similar_datasets.append({
                        "concept_id": result.properties.get("concept_id"),
                        "title": result.properties.get("title"),
                        "short_name": result.properties.get("short_name"),
                        "platforms": result.properties.get("platforms", []),
                        "instruments": result.properties.get("instruments", []),
                        "similarity_score": result.metadata.score if result.metadata else 0.0
                    })
            
            return similar_datasets[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar datasets for {concept_id}: {e}")
            return []
    
    async def get_dataset_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific dataset by concept ID."""
        if not self.client:
            return None
        
        try:
            nasa_collection = self.client.collections.get(self.collection_name)
            result = nasa_collection.query.fetch_object_by_id(concept_id)
            
            if result:
                return {
                    "concept_id": result.properties.get("concept_id"),
                    "title": result.properties.get("title"),
                    "summary": result.properties.get("summary"),
                    "short_name": result.properties.get("short_name"),
                    "data_center": result.properties.get("data_center"),
                    "platforms": result.properties.get("platforms", []),
                    "instruments": result.properties.get("instruments", []),
                    "variables": result.properties.get("variables", []),
                    "keywords": result.properties.get("keywords", [])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve dataset {concept_id}: {e}")
            return None
    
    def _create_combined_text(self, collection: CMRCollection) -> str:
        """Create combined searchable text from collection metadata."""
        parts = []
        
        if collection.title:
            parts.append(collection.title)
        
        if collection.summary:
            parts.append(collection.summary)
        
        if collection.short_name:
            parts.append(collection.short_name)
        
        if collection.platforms:
            parts.extend(collection.platforms)
        
        if collection.instruments:
            parts.extend(collection.instruments)
        
        if collection.variables:
            parts.extend(collection.variables)
        
        if collection.data_center:
            parts.append(collection.data_center)
        
        return " ".join(parts)
    
    def _extract_keywords(self, collection: CMRCollection) -> List[str]:
        """Extract keywords from dataset metadata."""
        keywords = set()
        
        # Extract from title
        if collection.title:
            title_words = collection.title.lower().split()
            keywords.update([w for w in title_words if len(w) > 3])
        
        # Extract from summary  
        if collection.summary:
            summary_words = collection.summary.lower().split()
            keywords.update([w for w in summary_words[:50] if len(w) > 4])  # Limit to avoid noise
        
        # Add platforms and instruments
        if collection.platforms:
            keywords.update([p.lower() for p in collection.platforms])
        
        if collection.instruments:
            keywords.update([i.lower() for i in collection.instruments])
        
        return list(keywords)[:20]  # Limit to 20 keywords
    
    def _build_where_filter(self, filters: Dict[str, Any]):
        """Build Weaviate where filter from dictionary."""
        # This would need to be implemented based on specific filter requirements
        # For now, return None (no filters applied)
        return None
    
    def _build_context_filters(self, query_context: QueryContext) -> Dict[str, Any]:
        """Build search filters from query context."""
        filters = {}
        
        constraints = query_context.constraints
        
        # Platform filters
        if constraints.platforms:
            filters["platforms"] = constraints.platforms
        
        # Instrument filters
        if constraints.instruments:
            filters["instruments"] = constraints.instruments
        
        # Temporal filters
        if constraints.temporal:
            if constraints.temporal.start_date:
                filters["temporal_start"] = constraints.temporal.start_date
            if constraints.temporal.end_date:
                filters["temporal_end"] = constraints.temporal.end_date
        
        return filters
    
    def _rerank_by_context(
        self, 
        results: List[Dict[str, Any]], 
        query_context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Re-rank search results based on query context."""
        # Apply context-specific scoring
        for result in results:
            context_score = 0.0
            
            # Platform matching bonus
            if query_context.constraints.platforms:
                result_platforms = result.get("platforms", [])
                matching_platforms = set(query_context.constraints.platforms).intersection(set(result_platforms))
                context_score += len(matching_platforms) * 0.1
            
            # Instrument matching bonus
            if query_context.constraints.instruments:
                result_instruments = result.get("instruments", [])
                matching_instruments = set(query_context.constraints.instruments).intersection(set(result_instruments))
                context_score += len(matching_instruments) * 0.1
            
            # Cloud hosting preference for accessibility
            if result.get("cloud_hosted"):
                context_score += 0.05
            
            if result.get("online_access"):
                context_score += 0.05
            
            # Combine with similarity score
            result["context_score"] = context_score
            result["combined_score"] = result.get("similarity_score", 0.0) + context_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        return results
    
    def _optimize_text_for_embedding(self, combined_text: str, collection: CMRCollection) -> str:
        """Optimize text for NASA Earth science domain semantic embedding."""
        # Enhance text with domain-specific context for better embeddings
        optimization_parts = [combined_text]
        
        # Add structured metadata context
        if collection.platforms:
            optimization_parts.append(f"Satellite platforms: {', '.join(collection.platforms)}")
        
        if collection.instruments:
            optimization_parts.append(f"Scientific instruments: {', '.join(collection.instruments)}")
        
        if collection.variables:
            optimization_parts.append(f"Measured variables: {', '.join(collection.variables)}")
        
        if collection.processing_level:
            optimization_parts.append(f"Processing level: {collection.processing_level}")
        
        # Add temporal context if available
        if collection.temporal_coverage:
            temporal_info = []
            if collection.temporal_coverage.get("start"):
                temporal_info.append(f"from {collection.temporal_coverage['start'][:4]}")
            if collection.temporal_coverage.get("end"):
                temporal_info.append(f"to {collection.temporal_coverage['end'][:4]}")
            if temporal_info:
                optimization_parts.append(f"Temporal coverage: {' '.join(temporal_info)}")
        
        # Add NASA Earth science domain keywords for better semantic understanding
        domain_context = "NASA Earth observation satellite remote sensing geoscience"
        optimization_parts.append(domain_context)
        
        optimized_text = " | ".join(optimization_parts)
        
        # Limit text length for optimal embedding performance
        if len(optimized_text) > 2000:
            optimized_text = optimized_text[:2000] + "..."
        
        return optimized_text
    
    async def close(self):
        """Close Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client connection closed")