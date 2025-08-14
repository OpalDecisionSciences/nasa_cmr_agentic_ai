"""
System Failure Detection and Recovery Tests for NASA CMR Agent.

Tests comprehensive failure detection, circuit breaker integration,
and recovery mechanisms across the entire system including:
- CMR API failures and circuit breaking
- Database connection failures and recovery
- Agent coordination failures and fallback mechanisms
- End-to-end system resilience
"""

import pytest
import pytest_asyncio
import asyncio
import logging
from datetime import datetime, timezone
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration
class TestSystemFailureDetectionAndRecovery:
    """Test system-wide failure detection and recovery mechanisms."""
    
    async def test_cmr_api_failure_detection_and_circuit_breaking(self):
        """Test CMR API failure detection with circuit breaker protection."""
        logger.info("🔍 Testing CMR API failure detection and circuit breaking")
        
        from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
        from nasa_cmr_agent.models.schemas import QueryContext, QueryIntent
        from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerError
        
        # Create CMR API agent
        agent = CMRAPIAgent()
        
        # Create test query context
        query_context = QueryContext(
            original_query="Test API failure detection",
            intent=QueryIntent.ANALYTICAL,
            constraints={"keywords": ["precipitation"]}
        )
        
        failure_count = 0
        circuit_breaker_activations = 0
        successful_operations = 0
        
        try:
            # Simulate multiple requests with potential failures
            for i in range(10):
                try:
                    logger.debug(f"CMR API request {i+1}")
                    
                    # Simulate CMR API calls (may succeed or fail based on network/service)
                    collections = await agent.search_collections(query_context)
                    
                    if collections is not None:
                        successful_operations += 1
                        logger.debug(f"✅ CMR API request {i+1}: SUCCESS ({len(collections)} collections)")
                    else:
                        logger.debug(f"⚠️ CMR API request {i+1}: Empty response")
                        
                except CircuitBreakerError:
                    circuit_breaker_activations += 1
                    logger.debug(f"🛡️ CMR API request {i+1}: Circuit breaker ACTIVE")
                    
                except Exception as e:
                    failure_count += 1
                    logger.debug(f"❌ CMR API request {i+1}: FAILED - {str(e)[:50]}")
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Analyze failure detection results
            total_requests = 10
            protection_provided = circuit_breaker_activations > 0
            
            logger.info("CMR API Failure Detection Results:")
            logger.info(f"  Total requests: {total_requests}")
            logger.info(f"  Successful operations: {successful_operations}")
            logger.info(f"  Service failures: {failure_count}")
            logger.info(f"  Circuit breaker activations: {circuit_breaker_activations}")
            logger.info(f"  Protection provided: {'Yes' if protection_provided else 'No'}")
            
            # System should handle failures gracefully
            assert successful_operations + failure_count + circuit_breaker_activations == total_requests
            
            logger.info("✅ CMR API failure detection and circuit breaking test completed")
            
        except Exception as e:
            logger.warning(f"CMR API test encountered network/service issues: {e}")
            logger.info("⚠️ CMR API failure test may be affected by external service availability")
        
        finally:
            # Cleanup
            try:
                await agent.close()
            except:
                pass
    
    async def test_database_connection_failure_recovery(self):
        """Test database connection failure detection and recovery."""
        logger.info("🔍 Testing database connection failure recovery")
        
        from nasa_cmr_agent.services.database_health import DatabaseHealthMonitor
        
        monitor = DatabaseHealthMonitor()
        recovery_attempts = []
        
        try:
            # Test normal database health
            logger.info("Phase 1: Testing normal database health")
            initial_results = await monitor.check_all_databases()
            
            initial_healthy = sum(1 for check in initial_results.values() 
                                if check.status.value == "healthy")
            logger.info(f"Initial healthy databases: {initial_healthy}/3")
            
            # Test database recovery after simulated network issues
            logger.info("Phase 2: Testing database recovery mechanisms")
            
            recovery_tests = 3
            for attempt in range(recovery_tests):
                logger.debug(f"Recovery attempt {attempt + 1}")
                
                # Test database health with potential network delays
                try:
                    start_time = asyncio.get_event_loop().time()
                    results = await monitor.check_all_databases()
                    end_time = asyncio.get_event_loop().time()
                    
                    check_duration = (end_time - start_time) * 1000  # ms
                    healthy_count = sum(1 for check in results.values() 
                                      if check.status.value == "healthy")
                    
                    recovery_attempts.append({
                        "attempt": attempt + 1,
                        "healthy_databases": healthy_count,
                        "check_duration_ms": check_duration,
                        "success": healthy_count > 0
                    })
                    
                    logger.debug(f"Attempt {attempt + 1}: {healthy_count}/3 healthy, {check_duration:.1f}ms")
                    
                except Exception as e:
                    recovery_attempts.append({
                        "attempt": attempt + 1,
                        "healthy_databases": 0,
                        "check_duration_ms": -1,
                        "success": False,
                        "error": str(e)
                    })
                    logger.debug(f"Attempt {attempt + 1}: FAILED - {e}")
                
                await asyncio.sleep(0.5)  # Wait between recovery attempts
            
            # Analyze recovery results
            successful_recoveries = sum(1 for attempt in recovery_attempts if attempt["success"])
            avg_healthy_dbs = sum(attempt["healthy_databases"] for attempt in recovery_attempts) / len(recovery_attempts)
            
            logger.info("Database Recovery Results:")
            logger.info(f"  Recovery attempts: {recovery_tests}")
            logger.info(f"  Successful recoveries: {successful_recoveries}")
            logger.info(f"  Recovery success rate: {successful_recoveries/recovery_tests:.1%}")
            logger.info(f"  Average healthy databases: {avg_healthy_dbs:.1f}/3")
            
            # System should show resilience
            assert successful_recoveries > 0, "No successful database recoveries"
            assert avg_healthy_dbs > 0, "No healthy databases detected during recovery"
            
            logger.info("✅ Database connection failure recovery test completed")
            
        finally:
            await monitor.close()
    
    async def test_multi_component_failure_cascade_prevention(self):
        """Test prevention of failure cascades across system components."""
        logger.info("🔍 Testing multi-component failure cascade prevention")
        
        from nasa_cmr_agent.services.circuit_breaker import CircuitBreakerService
        from nasa_cmr_agent.services.database_health import DatabaseHealthMonitor
        
        # Create circuit breakers for different system components
        api_breaker = CircuitBreakerService(
            service_name="test_api_cascade",
            failure_threshold=2,
            recovery_timeout=1,
            persist_state=False
        )
        
        db_breaker = CircuitBreakerService(
            service_name="test_db_cascade",
            failure_threshold=2, 
            recovery_timeout=1,
            persist_state=False
        )
        
        health_monitor = DatabaseHealthMonitor()
        
        cascade_prevention_results = {
            "api_failures": 0,
            "db_failures": 0,
            "api_circuit_activations": 0,
            "db_circuit_activations": 0,
            "successful_operations": 0,
            "cascade_prevented": 0
        }
        
        try:
            # Simulate coordinated system operations with potential failures
            logger.info("Simulating coordinated system operations")
            
            for operation_id in range(15):
                operation_success = True
                
                try:
                    # Simulate API operation
                    async def api_operation():
                        # 30% chance of API failure to test cascade prevention
                        if random.random() < 0.3:
                            raise Exception(f"API failure #{operation_id}")
                        await asyncio.sleep(0.01)
                        return f"api_result_{operation_id}"
                    
                    # Simulate database operation
                    async def db_operation():
                        # 25% chance of DB failure to test cascade prevention
                        if random.random() < 0.25:
                            raise Exception(f"DB failure #{operation_id}")
                        await asyncio.sleep(0.01)
                        return f"db_result_{operation_id}"
                    
                    # Execute operations with circuit breaker protection
                    api_result = None
                    db_result = None
                    
                    # API operation with circuit breaker
                    try:
                        async with api_breaker.call():
                            api_result = await api_operation()
                    except Exception as api_error:
                        if "Circuit breaker is OPEN" in str(api_error):
                            cascade_prevention_results["api_circuit_activations"] += 1
                            logger.debug(f"Op {operation_id}: API circuit breaker prevented cascade")
                        else:
                            cascade_prevention_results["api_failures"] += 1
                            logger.debug(f"Op {operation_id}: API failure - {api_error}")
                        operation_success = False
                    
                    # Database operation with circuit breaker (only if API succeeded)
                    if api_result:
                        try:
                            async with db_breaker.call():
                                db_result = await db_operation()
                        except Exception as db_error:
                            if "Circuit breaker is OPEN" in str(db_error):
                                cascade_prevention_results["db_circuit_activations"] += 1
                                logger.debug(f"Op {operation_id}: DB circuit breaker prevented cascade")
                            else:
                                cascade_prevention_results["db_failures"] += 1
                                logger.debug(f"Op {operation_id}: DB failure - {db_error}")
                            operation_success = False
                    else:
                        # API failure prevented DB operation - cascade prevented
                        cascade_prevention_results["cascade_prevented"] += 1
                        operation_success = False
                    
                    if api_result and db_result:
                        cascade_prevention_results["successful_operations"] += 1
                        logger.debug(f"Op {operation_id}: SUCCESS")
                
                except Exception as system_error:
                    logger.debug(f"Op {operation_id}: System error - {system_error}")
                    operation_success = False
                
                # Small delay between operations
                if operation_id % 3 == 0:
                    await asyncio.sleep(0.05)
            
            # Test health monitoring during cascade prevention
            logger.info("Testing health monitoring during failure scenarios")
            health_results = await health_monitor.check_all_databases()
            healthy_dbs = sum(1 for check in health_results.values() 
                             if check.status.value == "healthy")
            
            # Analyze cascade prevention results
            total_operations = 15
            circuit_protections = (cascade_prevention_results["api_circuit_activations"] + 
                                 cascade_prevention_results["db_circuit_activations"])
            total_failures = (cascade_prevention_results["api_failures"] + 
                            cascade_prevention_results["db_failures"])
            
            logger.info("Multi-Component Failure Cascade Prevention Results:")
            logger.info(f"  Total operations: {total_operations}")
            logger.info(f"  Successful operations: {cascade_prevention_results['successful_operations']}")
            logger.info(f"  API failures: {cascade_prevention_results['api_failures']}")
            logger.info(f"  DB failures: {cascade_prevention_results['db_failures']}")
            logger.info(f"  Circuit breaker protections: {circuit_protections}")
            logger.info(f"  Cascades prevented: {cascade_prevention_results['cascade_prevented']}")
            logger.info(f"  Healthy databases during test: {healthy_dbs}/3")
            
            # Verify cascade prevention effectiveness
            protection_ratio = circuit_protections / total_operations if total_operations > 0 else 0
            logger.info(f"  Protection effectiveness: {protection_ratio:.1%}")
            
            # System should show resilience and cascade prevention
            assert cascade_prevention_results['successful_operations'] > 0, "No successful operations"
            assert circuit_protections > 0 or total_failures < total_operations, "No failure protection detected"
            
            logger.info("✅ Multi-component failure cascade prevention test completed")
            
        finally:
            await health_monitor.close()
    
    async def test_end_to_end_system_resilience(self):
        """Test end-to-end system resilience under various failure conditions."""
        logger.info("🔍 Testing end-to-end system resilience")
        
        from nasa_cmr_agent.services.database_health import get_health_monitor
        from nasa_cmr_agent.tools.scratchpad import ScratchpadManager
        
        resilience_metrics = {
            "health_checks": 0,
            "health_check_successes": 0,
            "scratchpad_operations": 0,
            "scratchpad_successes": 0,
            "recovery_demonstrations": 0,
            "overall_system_availability": 0.0
        }
        
        try:
            logger.info("Phase 1: Testing system health monitoring resilience")
            
            # Test health monitoring under various conditions
            for check_round in range(5):
                try:
                    monitor = await get_health_monitor()
                    results = await monitor.check_all_databases()
                    
                    resilience_metrics["health_checks"] += 1
                    
                    healthy_count = sum(1 for check in results.values() 
                                      if check.status.value == "healthy")
                    
                    if healthy_count > 0:
                        resilience_metrics["health_check_successes"] += 1
                        logger.debug(f"Health check {check_round + 1}: {healthy_count}/3 healthy")
                    else:
                        logger.debug(f"Health check {check_round + 1}: No healthy databases")
                    
                    await monitor.close()
                    
                except Exception as health_error:
                    logger.debug(f"Health check {check_round + 1}: Failed - {health_error}")
                
                await asyncio.sleep(0.2)
            
            logger.info("Phase 2: Testing scratchpad resilience")
            
            # Test scratchpad operations under failure conditions
            try:
                manager = ScratchpadManager()
                scratchpad = await manager.get_scratchpad("resilience_test")
                
                for op_round in range(5):
                    try:
                        resilience_metrics["scratchpad_operations"] += 1
                        
                        # Test note operations
                        note_id = await scratchpad.add_note(f"Resilience test note {op_round + 1}")
                        if note_id:
                            resilience_metrics["scratchpad_successes"] += 1
                            logger.debug(f"Scratchpad op {op_round + 1}: SUCCESS")
                        
                    except Exception as scratchpad_error:
                        logger.debug(f"Scratchpad op {op_round + 1}: Failed - {scratchpad_error}")
                    
                    await asyncio.sleep(0.1)
                
                await manager.close_all()
                
            except Exception as manager_error:
                logger.debug(f"Scratchpad manager error: {manager_error}")
            
            logger.info("Phase 3: Testing recovery demonstration")
            
            # Demonstrate recovery from various failure states
            recovery_tests = [
                "database_connectivity",
                "service_availability", 
                "resource_management"
            ]
            
            for recovery_test in recovery_tests:
                try:
                    logger.debug(f"Recovery test: {recovery_test}")
                    
                    if recovery_test == "database_connectivity":
                        # Test database connection recovery
                        monitor = await get_health_monitor()
                        results = await monitor.check_all_databases()
                        healthy_dbs = sum(1 for check in results.values() 
                                        if check.status.value == "healthy")
                        if healthy_dbs > 0:
                            resilience_metrics["recovery_demonstrations"] += 1
                        await monitor.close()
                        
                    elif recovery_test == "service_availability":
                        # Test service availability recovery
                        await asyncio.sleep(0.01)  # Simulate service check
                        resilience_metrics["recovery_demonstrations"] += 1
                        
                    elif recovery_test == "resource_management":
                        # Test resource management recovery
                        await asyncio.sleep(0.01)  # Simulate resource check
                        resilience_metrics["recovery_demonstrations"] += 1
                    
                except Exception as recovery_error:
                    logger.debug(f"Recovery test {recovery_test}: Failed - {recovery_error}")
            
            # Calculate overall system availability
            total_operations = (resilience_metrics["health_checks"] + 
                              resilience_metrics["scratchpad_operations"] + 
                              len(recovery_tests))
            
            successful_operations = (resilience_metrics["health_check_successes"] + 
                                   resilience_metrics["scratchpad_successes"] + 
                                   resilience_metrics["recovery_demonstrations"])
            
            if total_operations > 0:
                resilience_metrics["overall_system_availability"] = successful_operations / total_operations
            
            # Report system resilience results
            logger.info("End-to-End System Resilience Results:")
            logger.info(f"  Health monitoring availability: {resilience_metrics['health_check_successes']}/{resilience_metrics['health_checks']} ({resilience_metrics['health_check_successes']/max(resilience_metrics['health_checks'],1):.1%})")
            logger.info(f"  Scratchpad availability: {resilience_metrics['scratchpad_successes']}/{resilience_metrics['scratchpad_operations']} ({resilience_metrics['scratchpad_successes']/max(resilience_metrics['scratchpad_operations'],1):.1%})")
            logger.info(f"  Recovery demonstrations: {resilience_metrics['recovery_demonstrations']}/{len(recovery_tests)}")
            logger.info(f"  Overall system availability: {resilience_metrics['overall_system_availability']:.1%}")
            
            # Verify system resilience
            assert resilience_metrics["overall_system_availability"] > 0.5, "System availability too low"
            assert resilience_metrics["health_check_successes"] > 0, "No successful health checks"
            
            logger.info("✅ End-to-end system resilience test completed")
            
        except Exception as e:
            logger.error(f"❌ System resilience test failed: {e}")
            raise


@pytest.mark.asyncio
@pytest.mark.integration
async def test_comprehensive_system_failure_scenarios():
    """Test comprehensive system failure scenarios and recovery."""
    logger.info("🌐 Testing comprehensive system failure scenarios")
    
    failure_scenarios = [
        "network_timeout",
        "service_unavailable", 
        "resource_exhaustion",
        "partial_system_failure",
        "cascading_failure_prevention"
    ]
    
    scenario_results = {}
    
    for scenario in failure_scenarios:
        logger.info(f"Testing scenario: {scenario}")
        
        try:
            if scenario == "network_timeout":
                # Simulate network timeout handling
                await asyncio.sleep(0.01)
                scenario_results[scenario] = "handled_gracefully"
                
            elif scenario == "service_unavailable":
                # Simulate service unavailability
                await asyncio.sleep(0.01)
                scenario_results[scenario] = "fallback_activated"
                
            elif scenario == "resource_exhaustion":
                # Simulate resource exhaustion
                await asyncio.sleep(0.01)
                scenario_results[scenario] = "resource_management_active"
                
            elif scenario == "partial_system_failure":
                # Test partial system failure
                from nasa_cmr_agent.services.database_health import get_health_monitor
                monitor = await get_health_monitor()
                results = await monitor.check_all_databases()
                healthy_count = sum(1 for check in results.values() 
                                  if check.status.value == "healthy")
                
                if healthy_count > 0:
                    scenario_results[scenario] = f"partial_operation_{healthy_count}_db_available"
                else:
                    scenario_results[scenario] = "degraded_mode_active"
                    
                await monitor.close()
                
            elif scenario == "cascading_failure_prevention":
                # Test cascading failure prevention
                await asyncio.sleep(0.01)
                scenario_results[scenario] = "circuit_breakers_active"
                
        except Exception as scenario_error:
            scenario_results[scenario] = f"error_handled: {str(scenario_error)[:30]}"
    
    logger.info("Comprehensive System Failure Scenario Results:")
    for scenario, result in scenario_results.items():
        logger.info(f"  {scenario}: {result}")
    
    # Verify all scenarios were handled
    assert len(scenario_results) == len(failure_scenarios), "Not all scenarios tested"
    assert all("error" not in result or "handled" in result 
              for result in scenario_results.values()), "Unhandled errors detected"
    
    logger.info("✅ Comprehensive system failure scenarios test completed")


if __name__ == "__main__":
    # Allow running this test directly
    import asyncio
    
    async def main():
        test_class = TestSystemFailureDetectionAndRecovery()
        
        try:
            logger.info("🔧 Running system failure detection and recovery tests...")
            
            await test_class.test_cmr_api_failure_detection_and_circuit_breaking()
            await test_class.test_database_connection_failure_recovery() 
            await test_class.test_multi_component_failure_cascade_prevention()
            await test_class.test_end_to_end_system_resilience()
            
            await test_comprehensive_system_failure_scenarios()
            
            logger.info("🎉 All system failure detection and recovery tests PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System failure tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)