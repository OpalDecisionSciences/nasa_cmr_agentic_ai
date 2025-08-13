import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from datetime import datetime, timezone
import json

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..models.schemas import CMRCollection, CMRGranule, QueryContext
from ..core.config import settings

logger = structlog.get_logger(__name__)


class KnowledgeGraphService:
    """
    Neo4j-based knowledge graph service for NASA Earth science datasets.
    
    Constructs and queries a knowledge graph containing:
    - Dataset relationships and hierarchies
    - Instrument and platform connections
    - Temporal and spatial overlaps
    - Scientific phenomenon associations
    - Data fusion opportunities
    """
    
    def __init__(self):
        self.driver = None
        self.database = getattr(settings, 'neo4j_database', 'neo4j')
        self._initialize_driver()
    
    def _initialize_driver(self):
        """Initialize Neo4j driver connection."""
        try:
            neo4j_uri = getattr(settings, 'neo4j_uri', 'bolt://localhost:7687')
            neo4j_user = getattr(settings, 'neo4j_user', 'neo4j')
            neo4j_password = getattr(settings, 'neo4j_password', 'password')
            
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password),
                database=self.database
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Neo4j driver initialized successfully")
            
        except (ServiceUnavailable, AuthError) as e:
            logger.warning(f"Failed to initialize Neo4j driver: {e}")
            self.driver = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Neo4j: {e}")
            self.driver = None
    
    async def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return False
        
        try:
            with self.driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT dataset_concept_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.concept_id IS UNIQUE",
                    "CREATE CONSTRAINT platform_name IF NOT EXISTS FOR (p:Platform) REQUIRE p.name IS UNIQUE",
                    "CREATE CONSTRAINT instrument_name IF NOT EXISTS FOR (i:Instrument) REQUIRE i.name IS UNIQUE",
                    "CREATE CONSTRAINT data_center_name IF NOT EXISTS FOR (dc:DataCenter) REQUIRE dc.name IS UNIQUE",
                    "CREATE CONSTRAINT phenomenon_name IF NOT EXISTS FOR (ph:Phenomenon) REQUIRE ph.name IS UNIQUE",
                    "CREATE CONSTRAINT variable_name IF NOT EXISTS FOR (v:Variable) REQUIRE v.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation note: {e}")
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX dataset_title IF NOT EXISTS FOR (d:Dataset) ON (d.title)",
                    "CREATE INDEX dataset_temporal_start IF NOT EXISTS FOR (d:Dataset) ON (d.temporal_start)",
                    "CREATE INDEX dataset_temporal_end IF NOT EXISTS FOR (d:Dataset) ON (d.temporal_end)",
                    "CREATE INDEX platform_type IF NOT EXISTS FOR (p:Platform) ON (p.type)",
                    "CREATE INDEX instrument_type IF NOT EXISTS FOR (i:Instrument) ON (i.type)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation note: {e}")
            
            logger.info("Neo4j schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
            return False
    
    async def ingest_dataset(self, collection: CMRCollection) -> bool:
        """Ingest a dataset and its relationships into the knowledge graph."""
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                # Create or update dataset node
                dataset_query = """
                MERGE (d:Dataset {concept_id: $concept_id})
                SET d.title = $title,
                    d.short_name = $short_name,
                    d.summary = $summary,
                    d.version_id = $version_id,
                    d.processing_level = $processing_level,
                    d.temporal_start = $temporal_start,
                    d.temporal_end = $temporal_end,
                    d.spatial_coverage = $spatial_coverage,
                    d.cloud_hosted = $cloud_hosted,
                    d.online_access = $online_access,
                    d.updated_at = datetime()
                RETURN d
                """
                
                # Prepare temporal dates
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
                
                # Execute dataset creation
                session.run(dataset_query, {
                    "concept_id": collection.concept_id,
                    "title": collection.title or "",
                    "short_name": collection.short_name or "",
                    "summary": collection.summary or "",
                    "version_id": collection.version_id or "",
                    "processing_level": collection.processing_level or "",
                    "temporal_start": temporal_start,
                    "temporal_end": temporal_end,
                    "spatial_coverage": json.dumps(collection.spatial_coverage) if collection.spatial_coverage else "",
                    "cloud_hosted": collection.cloud_hosted,
                    "online_access": collection.online_access_flag
                })
                
                # Create data center relationship
                if collection.data_center:
                    session.run("""
                        MATCH (d:Dataset {concept_id: $concept_id})
                        MERGE (dc:DataCenter {name: $data_center_name})
                        MERGE (d)-[:PROVIDED_BY]->(dc)
                    """, {
                        "concept_id": collection.concept_id,
                        "data_center_name": collection.data_center
                    })
                
                # Create platform and instrument relationships
                for platform_name in collection.platforms:
                    # Create platform
                    session.run("""
                        MATCH (d:Dataset {concept_id: $concept_id})
                        MERGE (p:Platform {name: $platform_name})
                        MERGE (d)-[:COLLECTED_BY]->(p)
                    """, {
                        "concept_id": collection.concept_id,
                        "platform_name": platform_name
                    })
                
                for instrument_name in collection.instruments:
                    # Create instrument and link to dataset
                    session.run("""
                        MATCH (d:Dataset {concept_id: $concept_id})
                        MERGE (i:Instrument {name: $instrument_name})
                        MERGE (d)-[:MEASURED_BY]->(i)
                    """, {
                        "concept_id": collection.concept_id,
                        "instrument_name": instrument_name
                    })
                    
                    # Link instrument to platforms if both exist
                    for platform_name in collection.platforms:
                        session.run("""
                            MATCH (p:Platform {name: $platform_name})
                            MATCH (i:Instrument {name: $instrument_name})
                            MERGE (i)-[:MOUNTED_ON]->(p)
                        """, {
                            "platform_name": platform_name,
                            "instrument_name": instrument_name
                        })
                
                # Create variable relationships
                for variable_name in collection.variables:
                    session.run("""
                        MATCH (d:Dataset {concept_id: $concept_id})
                        MERGE (v:Variable {name: $variable_name})
                        MERGE (d)-[:MEASURES]->(v)
                    """, {
                        "concept_id": collection.concept_id,
                        "variable_name": variable_name
                    })
                
                # Create phenomenon relationships based on content analysis
                phenomena = self._extract_phenomena(collection)
                for phenomenon in phenomena:
                    session.run("""
                        MATCH (d:Dataset {concept_id: $concept_id})
                        MERGE (ph:Phenomenon {name: $phenomenon_name})
                        MERGE (d)-[:STUDIES]->(ph)
                    """, {
                        "concept_id": collection.concept_id,
                        "phenomenon_name": phenomenon
                    })
            
            logger.debug(f"Ingested dataset {collection.concept_id} into knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest dataset {collection.concept_id}: {e}")
            return False
    
    async def find_dataset_relationships(
        self, 
        concept_id: str, 
        relationship_types: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find various types of relationships for a given dataset.
        
        Args:
            concept_id: Dataset concept ID
            relationship_types: Types of relationships to find
            
        Returns:
            Dictionary with relationship categories and related datasets
        """
        if not self.driver:
            return {}
        
        relationships = {
            "same_platform": [],
            "same_instrument": [],
            "same_phenomenon": [],
            "temporal_overlap": [],
            "complementary_variables": [],
            "same_data_center": []
        }
        
        try:
            with self.driver.session() as session:
                # Find datasets with same platforms
                same_platform_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})-[:COLLECTED_BY]->(p:Platform)
                    MATCH (d2:Dataset)-[:COLLECTED_BY]->(p)
                    WHERE d1 <> d2
                    RETURN DISTINCT d2.concept_id as concept_id, 
                           d2.title as title, 
                           d2.short_name as short_name,
                           p.name as platform_name
                    LIMIT 10
                """, {"concept_id": concept_id})
                
                relationships["same_platform"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "platform": record["platform_name"],
                        "relationship_type": "same_platform"
                    }
                    for record in same_platform_result
                ]
                
                # Find datasets with same instruments
                same_instrument_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})-[:MEASURED_BY]->(i:Instrument)
                    MATCH (d2:Dataset)-[:MEASURED_BY]->(i)
                    WHERE d1 <> d2
                    RETURN DISTINCT d2.concept_id as concept_id,
                           d2.title as title,
                           d2.short_name as short_name,
                           i.name as instrument_name
                    LIMIT 10
                """, {"concept_id": concept_id})
                
                relationships["same_instrument"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "instrument": record["instrument_name"],
                        "relationship_type": "same_instrument"
                    }
                    for record in same_instrument_result
                ]
                
                # Find datasets studying same phenomena
                same_phenomenon_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})-[:STUDIES]->(ph:Phenomenon)
                    MATCH (d2:Dataset)-[:STUDIES]->(ph)
                    WHERE d1 <> d2
                    RETURN DISTINCT d2.concept_id as concept_id,
                           d2.title as title,
                           d2.short_name as short_name,
                           ph.name as phenomenon_name
                    LIMIT 10
                """, {"concept_id": concept_id})
                
                relationships["same_phenomenon"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "phenomenon": record["phenomenon_name"],
                        "relationship_type": "same_phenomenon"
                    }
                    for record in same_phenomenon_result
                ]
                
                # Find temporally overlapping datasets
                temporal_overlap_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})
                    MATCH (d2:Dataset)
                    WHERE d1 <> d2 
                    AND d1.temporal_start IS NOT NULL 
                    AND d1.temporal_end IS NOT NULL
                    AND d2.temporal_start IS NOT NULL 
                    AND d2.temporal_end IS NOT NULL
                    AND (d1.temporal_start <= d2.temporal_end AND d1.temporal_end >= d2.temporal_start)
                    RETURN DISTINCT d2.concept_id as concept_id,
                           d2.title as title,
                           d2.short_name as short_name,
                           d2.temporal_start as start_date,
                           d2.temporal_end as end_date
                    LIMIT 10
                """, {"concept_id": concept_id})
                
                relationships["temporal_overlap"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "temporal_start": record["start_date"],
                        "temporal_end": record["end_date"],
                        "relationship_type": "temporal_overlap"
                    }
                    for record in temporal_overlap_result
                ]
                
                # Find datasets with complementary variables
                complementary_vars_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})-[:MEASURES]->(v1:Variable)
                    MATCH (d2:Dataset)-[:MEASURES]->(v2:Variable)
                    WHERE d1 <> d2 
                    AND v1 <> v2
                    AND (v1.name CONTAINS 'temperature' OR v1.name CONTAINS 'precipitation' OR v1.name CONTAINS 'humidity')
                    AND (v2.name CONTAINS 'temperature' OR v2.name CONTAINS 'precipitation' OR v2.name CONTAINS 'humidity')
                    RETURN DISTINCT d2.concept_id as concept_id,
                           d2.title as title,
                           d2.short_name as short_name,
                           COLLECT(DISTINCT v2.name) as variables
                    LIMIT 8
                """, {"concept_id": concept_id})
                
                relationships["complementary_variables"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "variables": record["variables"],
                        "relationship_type": "complementary_variables"
                    }
                    for record in complementary_vars_result
                ]
                
                # Find datasets from same data center
                same_center_result = session.run("""
                    MATCH (d1:Dataset {concept_id: $concept_id})-[:PROVIDED_BY]->(dc:DataCenter)
                    MATCH (d2:Dataset)-[:PROVIDED_BY]->(dc)
                    WHERE d1 <> d2
                    RETURN DISTINCT d2.concept_id as concept_id,
                           d2.title as title,
                           d2.short_name as short_name,
                           dc.name as data_center
                    LIMIT 8
                """, {"concept_id": concept_id})
                
                relationships["same_data_center"] = [
                    {
                        "concept_id": record["concept_id"],
                        "title": record["title"],
                        "short_name": record["short_name"],
                        "data_center": record["data_center"],
                        "relationship_type": "same_data_center"
                    }
                    for record in same_center_result
                ]
            
            logger.info(f"Found relationships for dataset {concept_id}")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to find relationships for {concept_id}: {e}")
            return relationships
    
    async def find_fusion_opportunities(
        self, 
        query_context: QueryContext,
        primary_datasets: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find data fusion opportunities based on query context and primary datasets.
        
        Args:
            query_context: User's query context
            primary_datasets: List of primary dataset concept IDs
            
        Returns:
            List of fusion opportunity recommendations
        """
        if not self.driver or not primary_datasets:
            return []
        
        fusion_opportunities = []
        
        try:
            with self.driver.session() as session:
                # Find datasets that complement the primary datasets
                for primary_id in primary_datasets[:3]:  # Limit to top 3
                    fusion_query = """
                    MATCH (d1:Dataset {concept_id: $primary_id})
                    
                    // Find datasets with complementary instruments
                    MATCH (d1)-[:MEASURED_BY]->(i1:Instrument)
                    MATCH (d2:Dataset)-[:MEASURED_BY]->(i2:Instrument)
                    WHERE d1 <> d2 AND i1 <> i2
                    
                    // Check temporal overlap
                    WITH d1, d2, i1, i2
                    WHERE d1.temporal_start IS NOT NULL 
                    AND d1.temporal_end IS NOT NULL
                    AND d2.temporal_start IS NOT NULL 
                    AND d2.temporal_end IS NOT NULL
                    AND (d1.temporal_start <= d2.temporal_end AND d1.temporal_end >= d2.temporal_start)
                    
                    // Get common phenomena they study
                    MATCH (d1)-[:STUDIES]->(ph:Phenomenon)<-[:STUDIES]-(d2)
                    
                    RETURN DISTINCT d2.concept_id as fusion_dataset_id,
                           d2.title as fusion_dataset_title,
                           d2.short_name as fusion_dataset_short_name,
                           i1.name as primary_instrument,
                           i2.name as fusion_instrument,
                           COLLECT(DISTINCT ph.name) as common_phenomena,
                           d1.temporal_start as primary_start,
                           d1.temporal_end as primary_end,
                           d2.temporal_start as fusion_start,
                           d2.temporal_end as fusion_end
                    ORDER BY SIZE(common_phenomena) DESC
                    LIMIT 5
                    """
                    
                    result = session.run(fusion_query, {"primary_id": primary_id})
                    
                    for record in result:
                        # Calculate temporal overlap percentage
                        overlap_score = self._calculate_temporal_overlap(
                            record["primary_start"], record["primary_end"],
                            record["fusion_start"], record["fusion_end"]
                        )
                        
                        fusion_opportunity = {
                            "primary_dataset_id": primary_id,
                            "fusion_dataset_id": record["fusion_dataset_id"],
                            "fusion_dataset_title": record["fusion_dataset_title"],
                            "fusion_dataset_short_name": record["fusion_dataset_short_name"],
                            "primary_instrument": record["primary_instrument"],
                            "fusion_instrument": record["fusion_instrument"],
                            "common_phenomena": record["common_phenomena"],
                            "temporal_overlap_score": overlap_score,
                            "fusion_type": "complementary_instruments",
                            "fusion_rationale": self._generate_fusion_rationale(
                                record["primary_instrument"],
                                record["fusion_instrument"],
                                record["common_phenomena"]
                            )
                        }
                        
                        fusion_opportunities.append(fusion_opportunity)
            
            # Sort by relevance (temporal overlap + common phenomena count)
            fusion_opportunities.sort(
                key=lambda x: x["temporal_overlap_score"] + len(x["common_phenomena"]) * 0.1,
                reverse=True
            )
            
            return fusion_opportunities[:10]
            
        except Exception as e:
            logger.error(f"Failed to find fusion opportunities: {e}")
            return []
    
    async def analyze_research_pathways(
        self, 
        phenomenon: str, 
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze research pathways for a given phenomenon using graph traversal.
        
        Args:
            phenomenon: Scientific phenomenon to analyze
            max_depth: Maximum depth for graph traversal
            
        Returns:
            Analysis of research pathways and connections
        """
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                # Multi-hop analysis to find research pathways
                pathway_query = """
                MATCH path = (ph:Phenomenon {name: $phenomenon})<-[:STUDIES]-(d:Dataset)
                OPTIONAL MATCH (d)-[:MEASURED_BY]->(i:Instrument)-[:MOUNTED_ON]->(p:Platform)
                OPTIONAL MATCH (d)-[:PROVIDED_BY]->(dc:DataCenter)
                OPTIONAL MATCH (d)-[:MEASURES]->(v:Variable)
                
                WITH ph, d, i, p, dc, COLLECT(DISTINCT v.name) as variables
                
                RETURN d.concept_id as dataset_id,
                       d.title as dataset_title,
                       d.short_name as short_name,
                       i.name as instrument,
                       p.name as platform,
                       dc.name as data_center,
                       variables,
                       d.temporal_start as temporal_start,
                       d.temporal_end as temporal_end
                ORDER BY d.temporal_start
                """
                
                result = session.run(pathway_query, {"phenomenon": phenomenon})
                
                datasets = []
                platforms = set()
                instruments = set()
                data_centers = set()
                all_variables = set()
                
                for record in result:
                    datasets.append({
                        "concept_id": record["dataset_id"],
                        "title": record["dataset_title"],
                        "short_name": record["short_name"],
                        "instrument": record["instrument"],
                        "platform": record["platform"],
                        "data_center": record["data_center"],
                        "variables": record["variables"] or [],
                        "temporal_start": record["temporal_start"],
                        "temporal_end": record["temporal_end"]
                    })
                    
                    if record["platform"]:
                        platforms.add(record["platform"])
                    if record["instrument"]:
                        instruments.add(record["instrument"])
                    if record["data_center"]:
                        data_centers.add(record["data_center"])
                    if record["variables"]:
                        all_variables.update(record["variables"])
                
                # Find connections between datasets
                connections = []
                if len(datasets) > 1:
                    connection_query = """
                    MATCH (d1:Dataset)-[:STUDIES]->(ph:Phenomenon {name: $phenomenon})<-[:STUDIES]-(d2:Dataset)
                    WHERE d1 <> d2
                    OPTIONAL MATCH (d1)-[:COLLECTED_BY]->(p:Platform)<-[:COLLECTED_BY]-(d2)
                    OPTIONAL MATCH (d1)-[:MEASURED_BY]->(i:Instrument)<-[:MEASURED_BY]-(d2)
                    
                    RETURN d1.concept_id as dataset1,
                           d2.concept_id as dataset2,
                           CASE WHEN p IS NOT NULL THEN 'shared_platform' ELSE NULL END as platform_connection,
                           CASE WHEN i IS NOT NULL THEN 'shared_instrument' ELSE NULL END as instrument_connection
                    """
                    
                    conn_result = session.run(connection_query, {"phenomenon": phenomenon})
                    connections = list(conn_result)
                
                return {
                    "phenomenon": phenomenon,
                    "total_datasets": len(datasets),
                    "datasets": datasets,
                    "platforms": list(platforms),
                    "instruments": list(instruments),
                    "data_centers": list(data_centers),
                    "variables": list(all_variables),
                    "connections": connections,
                    "temporal_span": {
                        "earliest": min([d["temporal_start"] for d in datasets if d["temporal_start"]], default=None),
                        "latest": max([d["temporal_end"] for d in datasets if d["temporal_end"]], default=None)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze research pathways for {phenomenon}: {e}")
            return {}
    
    async def get_platform_ecosystem(self, platform_name: str) -> Dict[str, Any]:
        """
        Get comprehensive ecosystem view for a specific platform.
        
        Args:
            platform_name: Name of the platform
            
        Returns:
            Comprehensive platform ecosystem information
        """
        if not self.driver:
            return {}
        
        try:
            with self.driver.session() as session:
                ecosystem_query = """
                MATCH (p:Platform {name: $platform_name})
                
                // Get all instruments on this platform
                OPTIONAL MATCH (i:Instrument)-[:MOUNTED_ON]->(p)
                
                // Get all datasets collected by this platform
                OPTIONAL MATCH (d:Dataset)-[:COLLECTED_BY]->(p)
                
                // Get all phenomena studied by datasets from this platform
                OPTIONAL MATCH (d)-[:STUDIES]->(ph:Phenomenon)
                
                // Get all variables measured by datasets from this platform
                OPTIONAL MATCH (d)-[:MEASURES]->(v:Variable)
                
                RETURN p.name as platform,
                       COLLECT(DISTINCT i.name) as instruments,
                       COLLECT(DISTINCT {
                           concept_id: d.concept_id,
                           title: d.title,
                           short_name: d.short_name
                       }) as datasets,
                       COLLECT(DISTINCT ph.name) as phenomena,
                       COLLECT(DISTINCT v.name) as variables
                """
                
                result = session.run(ecosystem_query, {"platform_name": platform_name})
                record = result.single()
                
                if record:
                    return {
                        "platform": record["platform"],
                        "instruments": [i for i in record["instruments"] if i],
                        "datasets": [d for d in record["datasets"] if d["concept_id"]],
                        "phenomena": [p for p in record["phenomena"] if p],
                        "variables": [v for v in record["variables"] if v],
                        "ecosystem_size": {
                            "instruments": len([i for i in record["instruments"] if i]),
                            "datasets": len([d for d in record["datasets"] if d["concept_id"]]),
                            "phenomena": len([p for p in record["phenomena"] if p]),
                            "variables": len([v for v in record["variables"] if v])
                        }
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get platform ecosystem for {platform_name}: {e}")
            return {}
    
    def _extract_phenomena(self, collection: CMRCollection) -> List[str]:
        """Extract scientific phenomena from dataset metadata."""
        phenomena = set()
        
        # Extract from title and summary
        text_content = f"{collection.title} {collection.summary}".lower()
        
        # Define phenomenon keywords
        phenomenon_keywords = {
            "precipitation": ["precipitation", "rainfall", "rain", "precip"],
            "temperature": ["temperature", "thermal", "temp", "sst"],
            "vegetation": ["vegetation", "ndvi", "evi", "leaf area", "biomass"],
            "drought": ["drought", "aridity", "dry", "water stress"],
            "floods": ["flood", "inundation", "water level"],
            "fire": ["fire", "burn", "wildfire", "combustion"],
            "aerosols": ["aerosol", "dust", "particulate", "pollution"],
            "ozone": ["ozone", "o3", "stratospheric"],
            "carbon": ["carbon", "co2", "carbon dioxide", "ghg"],
            "ice": ["ice", "snow", "glacier", "frozen"],
            "ocean": ["ocean", "sea surface", "marine", "sst"],
            "land surface": ["land surface", "soil", "surface"],
            "atmosphere": ["atmospheric", "meteorological", "weather"],
            "climate": ["climate", "climatology", "long-term"]
        }
        
        for phenomenon, keywords in phenomenon_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                phenomena.add(phenomenon)
        
        # Extract from variables if available
        for variable in collection.variables:
            variable_lower = variable.lower()
            for phenomenon, keywords in phenomenon_keywords.items():
                if any(keyword in variable_lower for keyword in keywords):
                    phenomena.add(phenomenon)
                    break
        
        return list(phenomena)
    
    def _calculate_temporal_overlap(self, start1, end1, start2, end2) -> float:
        """Calculate temporal overlap percentage between two time ranges."""
        if not all([start1, end1, start2, end2]):
            return 0.0
        
        try:
            # Convert to datetime if string
            if isinstance(start1, str):
                start1 = datetime.fromisoformat(start1.replace('Z', '+00:00'))
            if isinstance(end1, str):
                end1 = datetime.fromisoformat(end1.replace('Z', '+00:00'))
            if isinstance(start2, str):
                start2 = datetime.fromisoformat(start2.replace('Z', '+00:00'))
            if isinstance(end2, str):
                end2 = datetime.fromisoformat(end2.replace('Z', '+00:00'))
            
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_end <= overlap_start:
                return 0.0
            
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            total_duration = min((end1 - start1).total_seconds(), (end2 - start2).total_seconds())
            
            if total_duration <= 0:
                return 0.0
            
            return min(overlap_duration / total_duration, 1.0)
            
        except Exception:
            return 0.0
    
    def _generate_fusion_rationale(
        self, 
        instrument1: str, 
        instrument2: str, 
        phenomena: List[str]
    ) -> str:
        """Generate rationale for data fusion opportunity."""
        if not phenomena:
            return f"Complementary measurements from {instrument1} and {instrument2}"
        
        phenomenon_text = ", ".join(phenomena[:3])
        return f"Both instruments study {phenomenon_text}, enabling cross-validation and enhanced analysis through {instrument1} and {instrument2} synergy"
    
    async def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed")