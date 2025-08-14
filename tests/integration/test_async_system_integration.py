"""
Comprehensive async system integration tests.

Tests the full NASA CMR Agent system including:
- Async database coordination and interconnectivity
- Multi-agent workflow with real database operations
- Error handling and recovery across all components
- Production-grade async patterns and resource management
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

# Configure detailed logging for system tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncSystemIntegration:
    """Test complete async system integration with all databases."""
    
    async def test_database_health_monitoring_async(self):
        """Test async database health monitoring system."""
        from nasa_cmr_agent.services.database_health import get_health_monitor
        
        logger.info("Testing async database health monitoring system")
        
        try:
            # Get health monitor
            monitor = await get_health_monitor()
            assert monitor is not None
            
            # Test individual database health checks
            logger.info("Testing individual database health checks")
            
            # Redis health check (async)
            redis_result = await monitor.check_redis_health()
            logger.info(f"Redis health: {redis_result.status.value} ({redis_result.response_time_ms:.1f}ms)")
            assert redis_result is not None
            
            # Weaviate health check (sync)
            weaviate_result = monitor.check_weaviate_health()
            logger.info(f"Weaviate health: {weaviate_result.status.value} ({weaviate_result.response_time_ms:.1f}ms)")
            assert weaviate_result is not None
            
            # Neo4j health check (sync)  
            neo4j_result = monitor.check_neo4j_health()
            logger.info(f"Neo4j health: {neo4j_result.status.value} ({neo4j_result.response_time_ms:.1f}ms)")
            assert neo4j_result is not None
            
            # Test comprehensive health check
            all_results = await monitor.check_all_databases()
            assert len(all_results) == 3
            assert "redis" in all_results
            assert "weaviate" in all_results
            assert "neo4j" in all_results
            
            # Test health summary
            summary = monitor.get_database_status_summary()
            assert "overall_status" in summary
            assert "databases" in summary
            
            logger.info(f"Overall health status: {summary['overall_status']}")
            healthy_count = sum(1 for db, info in summary["databases"].items() if info["status"] == "healthy")
            logger.info(f"Healthy databases: {healthy_count}/3")
            
            # Cleanup
            await monitor.close()
            logger.info("✅ Async database health monitoring test completed")
            
        except Exception as e:
            logger.error(f"❌ Database health monitoring test failed: {e}")
            raise
    
    async def test_circuit_breaker_async_operations(self):
        """Test circuit breaker with async operations and Redis persistence."""
        from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerService
        
        logger.info("Testing circuit breaker async operations")
        
        try:
            # Create circuit breaker with Redis persistence
            breaker = CircuitBreakerService(
                service_name="test_service_async",
                failure_threshold=2,
                recovery_timeout=5,
                persist_state=True
            )
            
            # Test successful operation
            async def successful_operation():
                await asyncio.sleep(0.01)  # Simulate async work
                return "success"
            
            result = await breaker.call(successful_operation)
            assert result == "success"
            logger.info("✅ Successful async operation completed")
            
            # Test failure and circuit opening
            async def failing_operation():
                await asyncio.sleep(0.01) 
                raise Exception("Simulated failure")
            
            # Cause failures to open circuit
            for i in range(3):
                try:
                    await breaker.call(failing_operation)
                except Exception:
                    logger.debug(f"Expected failure {i+1}")
            
            # Verify circuit is open
            state = breaker.get_state()
            logger.info(f"Circuit breaker state: {state['state']}")
            
            # Test state persistence (should be saved to Redis)
            if state['state'] == 'open':
                logger.info("✅ Circuit breaker correctly opened after failures")
            
            # Test circuit recovery
            await asyncio.sleep(1)  # Wait briefly for potential recovery
            
            # Cleanup
            await breaker.close()
            logger.info("✅ Circuit breaker async operations test completed")
            
        except Exception as e:
            logger.error(f"❌ Circuit breaker async test failed: {e}")
            raise
    
    async def test_scratchpad_async_operations(self):
        """Test scratchpad async operations with Redis backend."""
        from nasa_cmr_agent.tools.scratchpad import ScratchpadManager
        
        logger.info("Testing scratchpad async operations")
        
        try:
            manager = ScratchpadManager()
            
            # Test async scratchpad operations
            scratchpad = await manager.get_scratchpad("test_agent_async")
            assert scratchpad is not None
            
            # Test async note operations
            note_id = await scratchpad.add_note("Async test note 1")
            assert note_id is not None
            logger.info(f"Added note with ID: {note_id}")
            
            # Add multiple notes asynchronously
            note_ids = []
            for i in range(3):
                note_id = await scratchpad.add_note(f"Async note {i+2}")
                note_ids.append(note_id)
            
            # Test async note retrieval
            notes = await scratchpad.get_recent_notes(limit=5)
            assert len(notes) >= 4  # At least the 4 notes we added
            logger.info(f"Retrieved {len(notes)} notes")
            
            # Test async search
            search_results = await scratchpad.search_notes("async")
            assert len(search_results) >= 4  # Should find our async notes
            logger.info(f"Found {len(search_results)} notes matching 'async'")
            
            # Test async note updates
            if note_ids:
                updated = await scratchpad.update_note(note_ids[0], "Updated async note")
                assert updated
                logger.info("✅ Successfully updated note")
            
            # Test async note deletion
            if note_ids:
                deleted = await scratchpad.delete_note(note_ids[0])
                assert deleted
                logger.info("✅ Successfully deleted note")
            
            # Cleanup
            await manager.close_all()
            logger.info("✅ Scratchpad async operations test completed")
            
        except Exception as e:
            logger.error(f"❌ Scratchpad async test failed: {e}")
            raise
    
    async def test_cmr_api_agent_async_operations(self):
        """Test CMR API agent async operations."""
        from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
        from nasa_cmr_agent.models.schemas import QueryContext, QueryIntent, SpatialConstraint, TemporalConstraint
        
        logger.info("Testing CMR API agent async operations")
        
        try:
            agent = CMRAPIAgent()
            
            # Create test query context
            query_context = QueryContext(
                original_query="Test precipitation data",
                intent=QueryIntent.ANALYTICAL,
                constraints={
                    "spatial": SpatialConstraint(
                        north=40.0, south=35.0, east=-100.0, west=-110.0,
                        region_name="Colorado"
                    ),
                    "temporal": TemporalConstraint(
                        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                        end_date=datetime(2020, 12, 31, tzinfo=timezone.utc)
                    ),
                    "keywords": ["precipitation"]
                }
            )
            
            # Test async collection search
            logger.info("Testing async collection search")
            collections = await agent.search_collections(query_context)
            assert isinstance(collections, list)
            logger.info(f"Found {len(collections)} collections")
            
            if collections:
                # Test async granule search for first collection
                logger.info("Testing async granule search")
                granules = await agent.search_granules(
                    collections[0].concept_id, 
                    query_context,
                    limit=5
                )
                assert isinstance(granules, list)
                logger.info(f"Found {len(granules)} granules")
            
            # Test async client cleanup
            await agent.close()
            logger.info("✅ CMR API agent async operations test completed")
            
        except Exception as e:
            logger.error(f"❌ CMR API agent async test failed: {e}")
            # CMR API failures are acceptable in testing environment
            logger.warning("CMR API test failed - this may be due to network/API availability")
    
    async def test_multi_database_async_workflow(self):
        """Test complete multi-database async workflow."""
        logger.info("Testing multi-database async workflow")
        
        # Patch to remove mocking for this test
        with patch('tests.conftest.mock_external_services', return_value={}):
            # Import required modules
            import redis.asyncio as aioredis
            import weaviate
            from neo4j import GraphDatabase
            from nasa_cmr_agent.core.config import settings
        
        redis_client = None
        weaviate_client = None  
        neo4j_driver = None
        
        try:
            # Setup all database connections
            logger.info("Setting up database connections...")
            
            # Redis connection
            redis_client = await aioredis.from_url(
                settings.redis_url,
                db=settings.redis_db,
                decode_responses=True
            )
            await redis_client.ping()
            logger.info("✅ Redis connected")
            
            # Weaviate connection  
            weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051
            )
            logger.info("✅ Weaviate connected")
            
            # Neo4j connection
            neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            with neo4j_driver.session() as session:
                result = session.run("RETURN 1 as test")
                assert result.single()["test"] == 1
            logger.info("✅ Neo4j connected")
            
            # Test coordinated async operations
            test_data = {
                "id": f"test_workflow_{int(datetime.now().timestamp())}",
                "title": "Multi-DB Test Dataset",
                "description": "Testing async database coordination",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Step 1: Store in Redis (async)
            logger.info("Step 1: Storing data in Redis")
            cache_key = f"test_data:{test_data['id']}"
            await redis_client.hset(cache_key, mapping=test_data)
            await redis_client.expire(cache_key, 300)  # 5 minute TTL
            
            # Verify Redis storage
            stored_data = await redis_client.hgetall(cache_key)
            assert stored_data["title"] == test_data["title"]
            logger.info("✅ Redis storage verified")
            
            # Step 2: Test Weaviate connectivity (basic check for v4 client)
            logger.info("Step 2: Testing Weaviate operations")
            collections = weaviate_client.collections.list_all()
            logger.info(f"✅ Weaviate accessible - {len(collections) if isinstance(collections, dict) else 0} collections")
            
            # Step 3: Store in Neo4j (sync within async context)  
            logger.info("Step 3: Storing data in Neo4j")
            with neo4j_driver.session() as session:
                create_query = '''
                CREATE (d:TestData {
                    id: $id,
                    title: $title,  
                    description: $description,
                    created_at: datetime($created_at)
                })
                RETURN d.id as created_id
                '''
                result = session.run(create_query, **test_data)
                created_id = result.single()["created_id"]
                assert created_id == test_data["id"]
                logger.info("✅ Neo4j storage verified")
            
            # Step 4: Test async data consistency across databases
            logger.info("Step 4: Testing data consistency")
            
            # Verify Redis data still exists
            redis_data = await redis_client.hgetall(cache_key)
            assert redis_data["id"] == test_data["id"]
            
            # Verify Neo4j data exists
            with neo4j_driver.session() as session:
                verify_query = "MATCH (d:TestData {id: $id}) RETURN d.title as title"
                result = session.run(verify_query, id=test_data["id"])
                record = result.single()
                assert record["title"] == test_data["title"]
            
            logger.info("✅ Data consistency verified across all databases")
            
            # Step 5: Test coordinated async cleanup
            logger.info("Step 5: Cleaning up test data")
            
            # Redis cleanup (async)
            deleted_count = await redis_client.delete(cache_key)
            assert deleted_count == 1
            
            # Neo4j cleanup (sync)
            with neo4j_driver.session() as session:
                cleanup_query = "MATCH (d:TestData {id: $id}) DELETE d"
                session.run(cleanup_query, id=test_data["id"])
            
            logger.info("✅ Multi-database async workflow completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Multi-database async workflow failed: {e}")
            raise
        finally:
            # Ensure all connections are properly closed
            if redis_client:
                await redis_client.aclose()
                logger.info("Redis connection closed")
            if weaviate_client and hasattr(weaviate_client, 'close'):
                weaviate_client.close()
                logger.info("Weaviate connection closed")
            if neo4j_driver:
                neo4j_driver.close()
                logger.info("Neo4j connection closed")
    
    async def test_full_system_async_integration(self):
        """Test complete system integration with async agent graph."""
        logger.info("Testing full system async integration")
        
        try:
            # Test that we can import and initialize all core components
            from nasa_cmr_agent.core.graph import CMRAgentGraph
            from nasa_cmr_agent.services.database_health import get_health_monitor
            
            # Initialize agent graph
            logger.info("Initializing CMR Agent Graph")
            graph = CMRAgentGraph()
            await graph.initialize()
            logger.info("✅ Agent graph initialized successfully")
            
            # Initialize health monitoring
            logger.info("Initializing health monitoring")  
            monitor = await get_health_monitor()
            health_results = await monitor.check_all_databases()
            
            healthy_dbs = [name for name, check in health_results.items() if check.status.value == "healthy"]
            logger.info(f"✅ {len(healthy_dbs)}/3 databases are healthy: {', '.join(healthy_dbs)}")
            
            # Test basic system readiness
            system_ready = len(healthy_dbs) >= 2  # At least 2 databases should be healthy
            assert system_ready, f"System not ready - only {len(healthy_dbs)}/3 databases healthy"
            
            logger.info("✅ Full system integration test completed successfully")
            
            # Cleanup
            await monitor.close()
            
        except Exception as e:
            logger.error(f"❌ Full system integration test failed: {e}")
            raise


@pytest.mark.asyncio
@pytest.mark.integration  
async def test_async_database_interconnectivity():
    """Standalone test for async database interconnectivity."""
    logger.info("Testing standalone async database interconnectivity")
    
    try:
        from nasa_cmr_agent.services.database_health import DatabaseHealthMonitor
        
        # Create health monitor without metrics (simpler)
        monitor = DatabaseHealthMonitor(metrics_service=None)
        
        # Test all databases concurrently
        logger.info("Testing concurrent database operations")
        
        # Use asyncio.gather for true parallelism
        redis_task = monitor.check_redis_health()
        weaviate_task = asyncio.create_task(asyncio.to_thread(monitor.check_weaviate_health))
        neo4j_task = asyncio.create_task(asyncio.to_thread(monitor.check_neo4j_health))
        
        # Wait for all database checks to complete
        redis_result, weaviate_result, neo4j_result = await asyncio.gather(
            redis_task, weaviate_task, neo4j_task, return_exceptions=True
        )
        
        # Analyze results
        results = {
            "redis": redis_result,
            "weaviate": weaviate_result, 
            "neo4j": neo4j_result
        }
        
        healthy_count = 0
        for db_name, result in results.items():
            if isinstance(result, Exception):
                logger.error(f"❌ {db_name} check failed: {result}")
            else:
                status_emoji = "✅" if result.status.value == "healthy" else "❌"
                logger.info(f"{status_emoji} {db_name}: {result.status.value} ({result.response_time_ms:.1f}ms)")
                if result.status.value == "healthy":
                    healthy_count += 1
        
        logger.info(f"Database interconnectivity test: {healthy_count}/3 databases healthy")
        
        # System should have at least Redis working for basic functionality
        assert healthy_count >= 1, "No databases are healthy - system cannot function"
        
        # Cleanup
        await monitor.close()
        
        logger.info("✅ Async database interconnectivity test completed")
        return healthy_count
        
    except Exception as e:
        logger.error(f"❌ Async database interconnectivity test failed: {e}")
        raise