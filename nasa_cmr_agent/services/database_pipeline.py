import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from datetime import datetime
import json

from ..models.schemas import CMRCollection, CMRGranule, QueryContext
from ..services.vector_database import VectorDatabaseService
from ..services.knowledge_graph import KnowledgeGraphService
from ..agents.cmr_api_agent import CMRAPIAgent

logger = structlog.get_logger(__name__)


class DatabasePipelineService:
    """
    Automated pipeline to populate databases from CMR API data.
    
    Features:
    - Dynamic schema initialization and validation
    - Real-time CMR data ingestion into both Neo4j and Weaviate
    - Relationship extraction and graph building
    - Variable and metadata enrichment
    - Incremental updates and data synchronization
    """
    
    def __init__(self):
        self.vector_db = VectorDatabaseService()
        self.knowledge_graph = KnowledgeGraphService()
        self.cmr_agent = CMRAPIAgent()
        self._initialized = False
        self._ingested_collections = set()
        
    async def initialize(self) -> bool:
        """Initialize all database schemas and connections."""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing database pipeline")
            
            # Initialize database schemas
            vector_success = await self.vector_db.initialize_schema()
            kg_success = await self.knowledge_graph.initialize_schema()
            
            if not vector_success:
                logger.warning("Weaviate schema initialization failed, but continuing")
            if not kg_success:
                logger.warning("Neo4j schema initialization failed, but continuing")
            
            self._initialized = True
            logger.info("Database pipeline initialized successfully", 
                       vector_db=vector_success, knowledge_graph=kg_success)
            return True
            
        except Exception as e:
            logger.error(f"Database pipeline initialization failed: {e}")
            return False
    
    async def ingest_query_results(
        self, 
        query_context: QueryContext,
        collections: List[CMRCollection], 
        granules: List[CMRGranule] = None
    ) -> Dict[str, Any]:
        """
        Ingest CMR query results into both databases with relationship building.
        
        Args:
            query_context: Original query context
            collections: Collections from CMR search
            granules: Optional granules data
            
        Returns:
            Ingestion statistics and status
        """
        if not self._initialized:
            await self.initialize()
        
        stats = {
            "collections_processed": 0,
            "collections_indexed": 0,
            "relationships_created": 0,
            "variables_enriched": 0,
            "kg_collections": 0,
            "cross_relationships": 0,
            "errors": []
        }
        
        try:
            logger.info(f"Starting ingestion of {len(collections)} collections")
            
            # Phase 1: Enrich collections with variables and metadata
            enriched_collections = await self._enrich_collections_with_variables(collections)
            
            # Phase 2: Batch ingest into vector database
            vector_stats = await self._ingest_into_vector_db(enriched_collections)
            stats.update(vector_stats)
            
            # Phase 3: Ingest into knowledge graph with relationship building
            kg_stats = await self._ingest_into_knowledge_graph(enriched_collections, granules)
            stats["kg_collections"] = kg_stats.get("kg_collections", 0)
            stats["relationships_created"] += kg_stats.get("relationships_created", 0)
            if kg_stats.get("kg_errors"):
                stats["errors"].extend(kg_stats["kg_errors"])
            
            # Phase 4: Build cross-collection relationships
            relationship_stats = await self._build_cross_collection_relationships(enriched_collections)
            stats["cross_relationships"] = relationship_stats.get("cross_relationships", 0)
            if relationship_stats.get("relationship_errors"):
                stats["errors"].extend(relationship_stats["relationship_errors"])
            
            # Update collections processed
            stats["collections_processed"] = len(enriched_collections)
            stats["variables_enriched"] = sum(len(c.variables) for c in enriched_collections if c.variables)
            
            logger.info("Database ingestion completed", stats=stats)
            return stats
            
        except Exception as e:
            logger.error(f"Database ingestion failed: {e}")
            stats["errors"].append(str(e))
            return stats
    
    async def _enrich_collections_with_variables(
        self, 
        collections: List[CMRCollection]
    ) -> List[CMRCollection]:
        """Enrich collections with variables from CMR variables endpoint."""
        enriched = []
        
        for collection in collections:
            try:
                # Skip if already processed recently
                if collection.concept_id in self._ingested_collections:
                    enriched.append(collection)
                    continue
                
                # Extract variables from collection metadata (more reliable than API)
                variables = self._extract_variables_from_metadata(collection)
                
                # Optional: Try CMR variables API as enhancement (not critical for core functionality)
                try:
                    if len(variables) < 3:  # Only try API if we need more variables
                        variables_data = await self.cmr_agent.get_collection_variables(collection.concept_id)
                        
                        # Extract additional variable names
                        for var_data in variables_data:
                            if isinstance(var_data, dict):
                                var_name = var_data.get("name") or var_data.get("variable_name") or var_data.get("long_name")
                                if var_name and var_name not in variables:
                                    variables.append(var_name)
                except Exception as var_error:
                    logger.debug(f"Variables API enhancement failed for {collection.concept_id}: {var_error}")
                    # Continue with metadata-extracted variables
                
                # Update collection with variables
                collection.variables = variables
                enriched.append(collection)
                
                # Mark as processed
                self._ingested_collections.add(collection.concept_id)
                
                logger.debug(f"Enriched collection {collection.concept_id} with {len(variables)} variables")
                
            except Exception as e:
                logger.warning(f"Failed to enrich collection {collection.concept_id}: {e}")
                enriched.append(collection)  # Include without enrichment
        
        return enriched
    
    async def _ingest_into_vector_db(self, collections: List[CMRCollection]) -> Dict[str, Any]:
        """Ingest collections into Weaviate vector database."""
        stats = {"collections_indexed": 0, "vector_errors": []}
        
        try:
            # Batch ingest for better performance
            indexed_count = await self.vector_db.index_datasets_batch(collections)
            stats["collections_indexed"] = indexed_count
            
            logger.info(f"Vector database: indexed {indexed_count} collections")
            
        except Exception as e:
            error_msg = f"Vector database ingestion failed: {e}"
            logger.warning(error_msg)
            stats["vector_errors"].append(error_msg)
        
        return stats
    
    async def _ingest_into_knowledge_graph(
        self, 
        collections: List[CMRCollection],
        granules: List[CMRGranule] = None
    ) -> Dict[str, Any]:
        """Ingest collections into Neo4j knowledge graph."""
        stats = {"kg_collections": 0, "relationships_created": 0, "kg_errors": []}
        
        try:
            # Ingest collections and build relationships
            for collection in collections:
                try:
                    success = await self.knowledge_graph.ingest_dataset(collection)
                    if success:
                        stats["kg_collections"] += 1
                    
                    # Build granule relationships if available
                    if granules:
                        collection_granules = [g for g in granules if g.collection_concept_id == collection.concept_id]
                        if collection_granules:
                            await self._create_granule_relationships(collection, collection_granules)
                            stats["relationships_created"] += len(collection_granules)
                    
                except Exception as e:
                    error_msg = f"Failed to ingest collection {collection.concept_id}: {e}"
                    logger.warning(error_msg)
                    stats["kg_errors"].append(error_msg)
            
            logger.info(f"Knowledge graph: ingested {stats['kg_collections']} collections")
            
        except Exception as e:
            error_msg = f"Knowledge graph ingestion failed: {e}"
            logger.warning(error_msg)
            stats["kg_errors"].append(error_msg)
        
        return stats
    
    async def _create_granule_relationships(
        self, 
        collection: CMRCollection, 
        granules: List[CMRGranule]
    ):
        """Create granule-to-collection relationships in Neo4j."""
        if not self.knowledge_graph.driver:
            return
        
        try:
            with self.knowledge_graph.driver.session() as session:
                for granule in granules:
                    # Create granule node and relationship to collection
                    session.run("""
                        MATCH (d:Dataset {concept_id: $collection_id})
                        MERGE (g:Granule {concept_id: $granule_id})
                        SET g.title = $title,
                            g.temporal_start = $temporal_start,
                            g.temporal_end = $temporal_end,
                            g.size_mb = $size_mb
                        MERGE (g)-[:BELONGS_TO]->(d)
                    """, {
                        "collection_id": collection.concept_id,
                        "granule_id": granule.concept_id,
                        "title": granule.title or "",
                        "temporal_start": self._parse_datetime(granule.temporal_extent.get("start") if granule.temporal_extent else None),
                        "temporal_end": self._parse_datetime(granule.temporal_extent.get("end") if granule.temporal_extent else None),
                        "size_mb": granule.size_mb
                    })
        
        except Exception as e:
            logger.warning(f"Failed to create granule relationships for {collection.concept_id}: {e}")
    
    async def _build_cross_collection_relationships(
        self, 
        collections: List[CMRCollection]
    ) -> Dict[str, Any]:
        """Build advanced cross-collection relationships."""
        stats = {"cross_relationships": 0, "relationship_errors": []}
        
        if not self.knowledge_graph.driver:
            return stats
        
        try:
            with self.knowledge_graph.driver.session() as session:
                # Create temporal overlap relationships
                temporal_relationships = await self._create_temporal_overlap_relationships(session, collections)
                stats["cross_relationships"] += temporal_relationships
                
                # Create spatial overlap relationships
                spatial_relationships = await self._create_spatial_overlap_relationships(session, collections)
                stats["cross_relationships"] += spatial_relationships
                
                # Create data fusion opportunity relationships
                fusion_relationships = await self._create_data_fusion_relationships(session, collections)
                stats["cross_relationships"] += fusion_relationships
                
                logger.info(f"Created {stats['cross_relationships']} cross-collection relationships")
        
        except Exception as e:
            error_msg = f"Cross-relationship creation failed: {e}"
            logger.warning(error_msg)
            stats["relationship_errors"].append(error_msg)
        
        return stats
    
    async def _create_temporal_overlap_relationships(
        self, 
        session, 
        collections: List[CMRCollection]
    ) -> int:
        """Create temporal overlap relationships between collections."""
        relationship_count = 0
        
        try:
            # Create TEMPORAL_OVERLAP relationships
            result = session.run("""
                MATCH (d1:Dataset), (d2:Dataset)
                WHERE d1 <> d2 
                AND d1.temporal_start IS NOT NULL 
                AND d1.temporal_end IS NOT NULL
                AND d2.temporal_start IS NOT NULL 
                AND d2.temporal_end IS NOT NULL
                AND (d1.temporal_start <= d2.temporal_end AND d1.temporal_end >= d2.temporal_start)
                MERGE (d1)-[r:TEMPORAL_OVERLAP]->(d2)
                SET r.overlap_start = CASE 
                    WHEN d1.temporal_start > d2.temporal_start THEN d1.temporal_start 
                    ELSE d2.temporal_start END,
                    r.overlap_end = CASE 
                    WHEN d1.temporal_end < d2.temporal_end THEN d1.temporal_end 
                    ELSE d2.temporal_end END
                RETURN COUNT(r) as relationship_count
            """)
            
            record = result.single()
            if record:
                relationship_count = record["relationship_count"]
        
        except Exception as e:
            logger.warning(f"Failed to create temporal overlap relationships: {e}")
        
        return relationship_count
    
    async def _create_spatial_overlap_relationships(
        self, 
        session, 
        collections: List[CMRCollection]
    ) -> int:
        """Create spatial overlap relationships between collections."""
        relationship_count = 0
        
        try:
            # Create SPATIAL_OVERLAP relationships
            # This is a simplified overlap check - could be enhanced with proper spatial functions
            result = session.run("""
                MATCH (d1:Dataset), (d2:Dataset)
                WHERE d1 <> d2 
                AND d1.spatial_coverage IS NOT NULL 
                AND d2.spatial_coverage IS NOT NULL
                AND d1.spatial_coverage <> '' 
                AND d2.spatial_coverage <> ''
                MERGE (d1)-[r:SPATIAL_OVERLAP]->(d2)
                SET r.created_at = datetime()
                RETURN COUNT(r) as relationship_count
            """)
            
            record = result.single()
            if record:
                relationship_count = record["relationship_count"]
        
        except Exception as e:
            logger.warning(f"Failed to create spatial overlap relationships: {e}")
        
        return relationship_count
    
    async def _create_data_fusion_relationships(
        self, 
        session, 
        collections: List[CMRCollection]
    ) -> int:
        """Create data fusion opportunity relationships."""
        relationship_count = 0
        
        try:
            # Create FUSION_COMPATIBLE relationships for datasets that:
            # 1. Have temporal overlap
            # 2. Share similar phenomena (via variables)
            # 3. Use different instruments (complementary measurements)
            result = session.run("""
                MATCH (d1:Dataset)-[:TEMPORAL_OVERLAP]-(d2:Dataset)
                MATCH (d1)-[:MEASURED_BY]->(i1:Instrument)
                MATCH (d2)-[:MEASURED_BY]->(i2:Instrument)
                MATCH (d1)-[:MEASURES]->(v1:Variable)
                MATCH (d2)-[:MEASURES]->(v2:Variable)
                WHERE i1 <> i2
                AND (v1.name CONTAINS 'temperature' OR v1.name CONTAINS 'precipitation' OR v1.name CONTAINS 'humidity')
                AND (v2.name CONTAINS 'temperature' OR v2.name CONTAINS 'precipitation' OR v2.name CONTAINS 'humidity')
                MERGE (d1)-[r:FUSION_COMPATIBLE]->(d2)
                SET r.fusion_type = 'complementary_instruments',
                    r.created_at = datetime()
                RETURN COUNT(DISTINCT r) as relationship_count
            """)
            
            record = result.single()
            if record:
                relationship_count = record["relationship_count"]
        
        except Exception as e:
            logger.warning(f"Failed to create fusion relationships: {e}")
        
        return relationship_count
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse CMR datetime string."""
        if not date_str:
            return None
        
        try:
            # Remove 'Z' and add timezone info
            clean_date = date_str.replace('Z', '+00:00')
            return datetime.fromisoformat(clean_date)
        except (ValueError, TypeError):
            return None
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about data ingestion."""
        stats = {
            "total_collections_ingested": len(self._ingested_collections),
            "vector_db_status": "connected" if self.vector_db.client else "disconnected",
            "knowledge_graph_status": "connected" if self.knowledge_graph.driver else "disconnected",
            "pipeline_initialized": self._initialized
        }
        
        # Get database-specific stats
        if self.knowledge_graph.driver:
            try:
                with self.knowledge_graph.driver.session() as session:
                    result = session.run("""
                        MATCH (d:Dataset) 
                        OPTIONAL MATCH (d)-[r]->() 
                        RETURN COUNT(DISTINCT d) as datasets, COUNT(r) as relationships
                    """)
                    record = result.single()
                    if record:
                        stats["neo4j_datasets"] = record["datasets"]
                        stats["neo4j_relationships"] = record["relationships"]
            except Exception as e:
                logger.warning(f"Failed to get Neo4j stats: {e}")
        
        return stats
    
    def _extract_variables_from_metadata(self, collection: CMRCollection) -> List[str]:
        """Extract likely variable names from collection title and summary."""
        variables = []
        
        # Common Earth science variables
        variable_patterns = {
            "temperature": ["temperature", "thermal", "temp", "sst", "lst"],
            "precipitation": ["precipitation", "rainfall", "rain", "precip"],
            "humidity": ["humidity", "moisture", "water vapor"],
            "wind": ["wind", "velocity", "speed"],
            "pressure": ["pressure", "atmospheric pressure"],
            "radiation": ["radiation", "radiance", "irradiance", "solar"],
            "vegetation": ["vegetation", "ndvi", "evi", "biomass", "leaf area"],
            "cloud": ["cloud", "cloud cover", "cloudiness"],
            "aerosol": ["aerosol", "dust", "particulate"],
            "ozone": ["ozone", "o3"],
            "carbon": ["carbon", "co2", "carbon dioxide"],
            "albedo": ["albedo", "reflectance"],
            "elevation": ["elevation", "dem", "topography", "altitude"]
        }
        
        # Check collection title and summary
        text_content = f"{collection.title or ''} {collection.summary or ''}".lower()
        
        for variable_name, patterns in variable_patterns.items():
            if any(pattern in text_content for pattern in patterns):
                variables.append(variable_name)
        
        # Add instrument-specific variables if available
        for instrument in collection.instruments:
            instrument_lower = instrument.lower()
            if "modis" in instrument_lower and "temperature" not in variables:
                variables.extend(["temperature", "vegetation", "cloud"])
            elif "viirs" in instrument_lower and "temperature" not in variables:
                variables.extend(["temperature", "vegetation"])
            elif "gpm" in instrument_lower or "trmm" in instrument_lower:
                variables.append("precipitation")
        
        return variables[:5]  # Limit to 5 most relevant variables
    
    async def close(self):
        """Close all database connections."""
        await self.vector_db.close()
        await self.knowledge_graph.close()
        await self.cmr_agent.close()