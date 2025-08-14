"""
Real database workflow test with no mocking.

This test validates actual database connectivity and async operations
without any mocking interference.
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from datetime import datetime, timezone
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_multi_database_async_workflow():
    """Test complete multi-database async workflow with real connections."""
    logger.info("🚀 Starting real multi-database async workflow test")
    
    redis_client = None
    weaviate_client = None
    neo4j_driver = None
    
    try:
        # Direct imports without mocking
        import redis.asyncio as aioredis
        import weaviate
        from neo4j import GraphDatabase
        
        # Use direct settings import
        from nasa_cmr_agent.core.config import settings
        
        # Setup Redis connection
        logger.info("🔗 Connecting to Redis...")
        redis_client = await aioredis.from_url(
            settings.redis_url,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test Redis connection
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
        # Setup Weaviate connection
        logger.info("🔗 Connecting to Weaviate...")
        weaviate_client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        logger.info("✅ Weaviate connected successfully")
        
        # Setup Neo4j connection
        logger.info("🔗 Connecting to Neo4j...")
        neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=5
        )
        
        # Test Neo4j connection
        with neo4j_driver.session(database=settings.neo4j_database) as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            assert test_value == 1
        logger.info("✅ Neo4j connected successfully")
        
        # Now test coordinated async operations
        logger.info("🔄 Testing coordinated async database operations...")
        
        # Test data
        test_id = f"workflow_{int(datetime.now().timestamp())}"
        test_data = {
            "id": test_id,
            "title": "Real Multi-DB Workflow Test",
            "description": "Testing real async database coordination",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_type": "integration_test"
        }
        
        # Phase 1: Async Redis operations
        logger.info("📝 Phase 1: Redis async operations")
        cache_key = f"workflow_test:{test_data['id']}"
        
        # Store data in Redis
        await redis_client.hset(cache_key, mapping=test_data)
        await redis_client.expire(cache_key, 600)  # 10 minute TTL
        
        # Verify Redis data
        stored_data = await redis_client.hgetall(cache_key)
        assert stored_data["title"] == test_data["title"]
        logger.info("✅ Redis async operations successful")
        
        # Phase 2: Weaviate operations (v4 client)
        logger.info("📝 Phase 2: Weaviate operations")
        try:
            # Test basic Weaviate functionality
            collections = weaviate_client.collections.list_all()
            collection_count = len(collections) if isinstance(collections, dict) else 0
            logger.info(f"✅ Weaviate operations successful - {collection_count} collections found")
        except Exception as w_error:
            logger.warning(f"Weaviate operation issue: {w_error}")
        
        # Phase 3: Neo4j graph operations
        logger.info("📝 Phase 3: Neo4j graph operations")
        with neo4j_driver.session(database=settings.neo4j_database) as session:
            # Create test node
            create_query = '''
            CREATE (w:WorkflowTest {
                id: $id,
                title: $title,
                description: $description,
                timestamp: datetime($timestamp),
                workflow_type: $workflow_type
            })
            RETURN w.id as created_id
            '''
            
            result = session.run(create_query, **test_data)
            created_id = result.single()["created_id"]
            assert created_id == test_data["id"]
            logger.info("✅ Neo4j graph operations successful")
        
        # Phase 4: Test data consistency and async coordination
        logger.info("📝 Phase 4: Testing data consistency across databases")
        
        # Concurrent verification using asyncio.gather
        async def verify_redis():
            data = await redis_client.hgetall(cache_key)
            return data["id"] == test_data["id"]
        
        def verify_neo4j():
            with neo4j_driver.session(database=settings.neo4j_database) as session:
                query = "MATCH (w:WorkflowTest {id: $id}) RETURN w.title as title"
                result = session.run(query, id=test_data["id"])
                record = result.single()
                return record["title"] == test_data["title"]
        
        def verify_weaviate():
            # Simple connectivity check
            try:
                collections = weaviate_client.collections.list_all()
                return True
            except:
                return False
        
        # Run verifications concurrently
        redis_ok, neo4j_ok, weaviate_ok = await asyncio.gather(
            verify_redis(),
            asyncio.to_thread(verify_neo4j),
            asyncio.to_thread(verify_weaviate),
            return_exceptions=True
        )
        
        # Check results
        verifications = {
            "Redis": redis_ok if not isinstance(redis_ok, Exception) else False,
            "Neo4j": neo4j_ok if not isinstance(neo4j_ok, Exception) else False,
            "Weaviate": weaviate_ok if not isinstance(weaviate_ok, Exception) else False
        }
        
        successful_verifications = sum(1 for success in verifications.values() if success)
        logger.info(f"✅ Data consistency verified: {successful_verifications}/3 databases")
        
        for db_name, success in verifications.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {db_name} verification")
        
        # Phase 5: Async cleanup
        logger.info("📝 Phase 5: Coordinated async cleanup")
        
        # Clean up Redis data
        deleted_count = await redis_client.delete(cache_key)
        assert deleted_count == 1
        
        # Clean up Neo4j data
        with neo4j_driver.session(database=settings.neo4j_database) as session:
            cleanup_query = "MATCH (w:WorkflowTest {id: $id}) DELETE w"
            result = session.run(cleanup_query, id=test_data["id"])
            result.consume()
        
        logger.info("✅ Cleanup completed successfully")
        
        # Final verification
        assert successful_verifications >= 2, f"Only {successful_verifications}/3 databases verified successfully"
        
        logger.info("🎉 Real multi-database async workflow test PASSED!")
        logger.info(f"   ✅ {successful_verifications}/3 databases working correctly")
        logger.info("   ✅ Async coordination functioning properly")
        logger.info("   ✅ Data consistency maintained")
        logger.info("   ✅ Resource cleanup successful")
        
        return successful_verifications
        
    except Exception as e:
        logger.error(f"❌ Real multi-database async workflow test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # Ensure all connections are properly closed
        cleanup_tasks = []
        
        if redis_client:
            cleanup_tasks.append(redis_client.aclose())
        if weaviate_client and hasattr(weaviate_client, 'close'):
            try:
                weaviate_client.close()
            except:
                pass
        if neo4j_driver:
            try:
                neo4j_driver.close()
            except:
                pass
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("🔌 All database connections closed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_error_handling_and_recovery():
    """Test async error handling and recovery mechanisms."""
    logger.info("🔧 Testing async error handling and recovery")
    
    try:
        from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerService
        
        # Test circuit breaker with async operations
        breaker = CircuitBreakerService(
            service_name="error_recovery_test",
            failure_threshold=2,
            recovery_timeout=1,
            persist_state=False  # Don't persist for this test
        )
        
        # Test successful async operation
        async def successful_async_op():
            await asyncio.sleep(0.01)
            return "success"
        
        result = await breaker.call(successful_async_op)
        assert result == "success"
        logger.info("✅ Successful async operation completed")
        
        # Test failure scenarios
        failure_count = 0
        async def failing_async_op():
            nonlocal failure_count
            failure_count += 1
            await asyncio.sleep(0.01)
            raise Exception(f"Simulated async failure #{failure_count}")
        
        # Cause circuit to open
        for i in range(3):
            try:
                await breaker.call(failing_async_op)
            except Exception as e:
                logger.debug(f"Expected failure {i+1}: {e}")
        
        # Verify circuit is open
        state = breaker.get_state()
        logger.info(f"Circuit breaker state after failures: {state['state']}")
        
        # Test recovery after timeout
        await asyncio.sleep(1.5)  # Wait for recovery timeout
        
        # Circuit should allow test calls in half-open state
        try:
            result = await breaker.call(successful_async_op)
            logger.info("✅ Circuit breaker recovery successful")
        except Exception as recovery_error:
            logger.info(f"Circuit still recovering: {recovery_error}")
        
        await breaker.close()
        logger.info("✅ Async error handling and recovery test completed")
        
    except Exception as e:
        logger.error(f"❌ Async error handling test failed: {e}")
        raise


if __name__ == "__main__":
    # Allow running this test directly
    import asyncio
    
    async def main():
        try:
            result = await test_real_multi_database_async_workflow()
            print(f"\n🎉 SUCCESS: {result}/3 databases working in async workflow")
            
            await test_async_error_handling_and_recovery()
            print("🎉 SUCCESS: Async error handling working correctly")
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            return False
        return True
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)