import asyncio
from typing import Dict, Any, List
import structlog
from datetime import datetime

from ..core.config import settings
from .database_pipeline import DatabasePipelineService
from .vector_database import VectorDatabaseService
from .knowledge_graph import KnowledgeGraphService
from ..agents.cmr_api_agent import CMRAPIAgent
from ..models.schemas import QueryContext, QueryConstraints, QueryIntent

logger = structlog.get_logger(__name__)


class StartupValidator:
    """
    Comprehensive startup validation and database initialization service.
    
    Features:
    - Database connection validation
    - Schema initialization verification  
    - Sample data ingestion testing
    - Health checks and diagnostics
    """
    
    def __init__(self):
        self.pipeline = DatabasePipelineService()
        self.vector_db = VectorDatabaseService()
        self.knowledge_graph = KnowledgeGraphService()
        self.cmr_agent = CMRAPIAgent()
    
    async def validate_startup(self) -> Dict[str, Any]:
        """
        Comprehensive startup validation.
        
        Returns:
            Validation results and status
        """
        validation_results = {
            "overall_status": "unknown",
            "database_connections": {},
            "schema_initialization": {},
            "sample_data_test": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        logger.info("Starting comprehensive startup validation")
        
        try:
            # Phase 1: Database connection validation
            connection_results = await self._validate_database_connections()
            validation_results["database_connections"] = connection_results
            
            # Phase 2: Schema initialization
            schema_results = await self._validate_schema_initialization()
            validation_results["schema_initialization"] = schema_results
            
            # Phase 3: Sample data ingestion test - always attempt even with partial connections
            sample_results = await self._test_sample_data_ingestion()
            validation_results["sample_data_test"] = sample_results
            
            # Phase 4: Overall assessment
            overall_status = self._assess_overall_status(validation_results)
            validation_results["overall_status"] = overall_status
            
            # Phase 5: Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            validation_results["recommendations"] = recommendations
            
            logger.info("Startup validation completed", status=overall_status)
            return validation_results
            
        except Exception as e:
            logger.error(f"Startup validation failed: {e}")
            validation_results["overall_status"] = "failed"
            validation_results["errors"].append(str(e))
            return validation_results
    
    async def _validate_database_connections(self) -> Dict[str, Any]:
        """Validate all database connections."""
        results = {
            "weaviate": {"connected": False, "error": None},
            "neo4j": {"connected": False, "error": None},
            "cmr_api": {"connected": False, "error": None},
            "all_connected": False
        }
        
        # Test Weaviate connection
        try:
            if self.vector_db.client:
                # Try a simple operation
                collections = self.vector_db.client.collections.list_all()
                results["weaviate"]["connected"] = True
                logger.info("Weaviate connection validated")
            else:
                results["weaviate"]["error"] = "Client not initialized"
        except Exception as e:
            results["weaviate"]["error"] = str(e)
            logger.warning(f"Weaviate connection failed: {e}")
        
        # Test Neo4j connection
        try:
            if self.knowledge_graph.driver:
                with self.knowledge_graph.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    record = result.single()
                    if record and record["test"] == 1:
                        results["neo4j"]["connected"] = True
                        logger.info("Neo4j connection validated")
            else:
                results["neo4j"]["error"] = "Driver not initialized"
        except Exception as e:
            results["neo4j"]["error"] = str(e)
            logger.warning(f"Neo4j connection failed: {e}")
        
        # Test CMR API connection
        try:
            # Simple test query
            test_context = QueryContext(
                original_query="test connection",
                intent=QueryIntent.EXPLORATORY,
                constraints=QueryConstraints()
            )
            
            # This will test the CMR API connection
            collections = await self.cmr_agent.search_collections(test_context)
            results["cmr_api"]["connected"] = True
            logger.info("CMR API connection validated")
        except Exception as e:
            results["cmr_api"]["error"] = str(e)
            logger.warning(f"CMR API connection failed: {e}")
        
        # Check if all are connected
        results["all_connected"] = all(
            db["connected"] for db in [results["weaviate"], results["neo4j"], results["cmr_api"]]
        )
        
        return results
    
    async def _validate_schema_initialization(self) -> Dict[str, Any]:
        """Validate database schema initialization."""
        results = {
            "pipeline_initialized": False,
            "weaviate_schema": False,
            "neo4j_schema": False,
            "errors": []
        }
        
        try:
            # Test pipeline initialization
            pipeline_success = await self.pipeline.initialize()
            results["pipeline_initialized"] = pipeline_success
            
            # Test Weaviate schema
            weaviate_success = await self.vector_db.initialize_schema()
            results["weaviate_schema"] = weaviate_success
            
            # Test Neo4j schema
            neo4j_success = await self.knowledge_graph.initialize_schema()
            results["neo4j_schema"] = neo4j_success
            
            logger.info("Schema initialization validated", results=results)
            
        except Exception as e:
            error_msg = f"Schema validation failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    async def _test_sample_data_ingestion(self) -> Dict[str, Any]:
        """Test sample data ingestion to validate the pipeline."""
        results = {
            "test_performed": False,
            "ingestion_successful": False,
            "collections_processed": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        try:
            logger.info("Starting sample data ingestion test")
            
            # Create a test query for a common dataset - use broader search to ensure results
            test_context = QueryContext(
                original_query="precipitation data",
                intent=QueryIntent.SPECIFIC_DATA,
                constraints=QueryConstraints(
                    keywords=["precipitation", "rainfall"],
                    # Remove platform constraint to get more results
                )
            )
            
            # Get sample collections from CMR
            collections = await self.cmr_agent.search_collections(test_context)
            
            if collections:
                # Use more collections for testing to ensure relationships can be created
                test_collections = collections[:10]  # Increased from 3 to 10
                
                # Test ingestion pipeline
                ingestion_stats = await self.pipeline.ingest_query_results(
                    test_context, test_collections
                )
                
                results["test_performed"] = True
                results["ingestion_successful"] = len(ingestion_stats.get("errors", [])) == 0
                results["collections_processed"] = ingestion_stats.get("collections_processed", 0)
                results["relationships_created"] = ingestion_stats.get("relationships_created", 0)
                
                logger.info("Sample data ingestion test completed", stats=ingestion_stats)
            else:
                results["errors"].append("No collections retrieved for testing")
                
        except Exception as e:
            error_msg = f"Sample data ingestion test failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _assess_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Assess overall startup status."""
        db_connections = validation_results.get("database_connections", {})
        schema_init = validation_results.get("schema_initialization", {})
        sample_test = validation_results.get("sample_data_test", {})
        
        # Check critical components
        weaviate_ok = db_connections.get("weaviate", {}).get("connected", False)
        neo4j_ok = db_connections.get("neo4j", {}).get("connected", False)
        cmr_api_ok = db_connections.get("cmr_api", {}).get("connected", False)
        pipeline_ok = schema_init.get("pipeline_initialized", False)
        
        # Determine status
        if weaviate_ok and neo4j_ok and cmr_api_ok and pipeline_ok:
            if sample_test.get("ingestion_successful", False):
                return "excellent"
            else:
                return "good"
        elif cmr_api_ok and (weaviate_ok or neo4j_ok):
            return "degraded"
        elif cmr_api_ok:
            return "minimal"
        else:
            return "failed"
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        db_connections = validation_results.get("database_connections", {})
        schema_init = validation_results.get("schema_initialization", {})
        
        # Database connection recommendations
        if not db_connections.get("weaviate", {}).get("connected", False):
            recommendations.append(
                "Start Weaviate server: docker run -p 8080:8080 semitechnologies/weaviate:latest"
            )
        
        if not db_connections.get("neo4j", {}).get("connected", False):
            recommendations.append(
                "Start Neo4j server: docker run -p 7474:7474 -p 7687:7687 neo4j:latest"
            )
        
        if not db_connections.get("cmr_api", {}).get("connected", False):
            recommendations.append(
                "Check internet connection and CMR API availability"
            )
        
        # Schema recommendations
        if not schema_init.get("pipeline_initialized", False):
            recommendations.append(
                "Run manual schema initialization before starting the application"
            )
        
        # Overall recommendations
        status = validation_results.get("overall_status")
        if status in ["failed", "minimal"]:
            recommendations.append(
                "Consider running the application in fallback mode without advanced features"
            )
        elif status == "degraded":
            recommendations.append(
                "Some advanced features may be limited. Check database connections."
            )
        
        return recommendations
    
    async def fix_common_issues(self) -> Dict[str, Any]:
        """Attempt to fix common startup issues automatically."""
        fixes = {
            "attempted_fixes": [],
            "successful_fixes": [],
            "failed_fixes": [],
            "manual_intervention_needed": []
        }
        
        logger.info("Attempting to fix common startup issues")
        
        # Fix 1: Reset circuit breaker and reinitialize schemas
        try:
            fixes["attempted_fixes"].append("circuit_breaker_reset")
            
            # Reset circuit breaker to allow requests
            await self.cmr_agent.circuit_breaker.reset()
            fixes["successful_fixes"].append("circuit_breaker_reset")
            
            # Reinitialize schemas
            fixes["attempted_fixes"].append("schema_reinitialization")
            
            vector_success = await self.vector_db.initialize_schema()
            kg_success = await self.knowledge_graph.initialize_schema()
            
            if vector_success and kg_success:
                fixes["successful_fixes"].append("schema_reinitialization")
            else:
                fixes["failed_fixes"].append("schema_reinitialization")
                
        except Exception as e:
            fixes["failed_fixes"].append(f"schema_reinitialization: {e}")
        
        # Fix 2: Bootstrap sample data to create relationships
        try:
            fixes["attempted_fixes"].append("bootstrap_sample_data")
            
            # Bootstrap with a broader query to ensure we get datasets with variables
            bootstrap_context = QueryContext(
                original_query="satellite Earth observation data",
                intent=QueryIntent.EXPLORATORY,
                constraints=QueryConstraints(
                    keywords=["temperature", "precipitation", "vegetation"]
                )
            )
            
            # Get more sample collections for bootstrapping
            bootstrap_collections = await self.cmr_agent.search_collections(bootstrap_context)
            
            if bootstrap_collections:
                # Limit to 15 collections for bootstrap
                bootstrap_stats = await self.pipeline.ingest_query_results(
                    bootstrap_context, bootstrap_collections[:15]
                )
                
                if bootstrap_stats.get("relationships_created", 0) > 0:
                    fixes["successful_fixes"].append(f"bootstrap_sample_data: {bootstrap_stats['relationships_created']} relationships created")
                else:
                    fixes["failed_fixes"].append("bootstrap_sample_data: no relationships created")
            else:
                fixes["failed_fixes"].append("bootstrap_sample_data: no collections found")
            
        except Exception as e:
            fixes["failed_fixes"].append(f"bootstrap_sample_data: {e}")
        
        return fixes
    
    async def close(self):
        """Close all connections."""
        await self.pipeline.close()
        await self.cmr_agent.close()


# Convenience function for quick validation
async def run_startup_validation() -> Dict[str, Any]:
    """Run startup validation and return results."""
    validator = StartupValidator()
    try:
        results = await validator.validate_startup()
        return results
    finally:
        await validator.close()


if __name__ == "__main__":
    # Run validation if script is executed directly
    async def main():
        results = await run_startup_validation()
        print(f"Startup validation completed with status: {results['overall_status']}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"- {rec}")
    
    asyncio.run(main())